import argparse
import os
import warnings

import accelerate
import torch

from diffsynth.core import UnifiedDataset, load_state_dict
from diffsynth.core.data.operators import LoadAudio, LoadVideo, ImageCropAndResize, ToAbsolutePath
from diffsynth.diffusion import *
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

from diffsynth.extensions.eawm import (
    LoadActionCSV,
    LoadImageFolder,
    EAWMCheckpointLogger,
    inject_eawm,
    prepare_eawm_kvaf_branch,
    eawm_flow_match_sft_loss,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanEAWMTrainingModule(DiffusionTrainingModule):
    KVAF_BLOCK_PREFIX = "pipe.dit.eawm_kvaf_branch.blocks."
    FUSION_PREFIX = "pipe.dit.eawm_kvaf_branch.bidirectional_cross_attentions."
    KVAF_HEAD_PREFIX = "pipe.dit.eawm_kvaf_branch.kvaf_head."

    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        kvaf_input_key="kvaf_path",
        action_input_key="action_path",
        context_interval=1,
        context_residual_scale=1.0,
        event_loss_weight=1.0,
        resume_dit_checkpoint=None,
    ):
        super().__init__()

        if not use_gradient_checkpointing:
            warnings.warn(
                "Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing."
            )
            use_gradient_checkpointing = True

        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            fp8_models=fp8_models,
            offload_models=offload_models,
            device=device,
        )
        tokenizer_config = (
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")
            if tokenizer_path is None else ModelConfig(tokenizer_path)
        )
        audio_processor_config = self.parse_path_or_model_id(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(
            redirect_common_files=False,
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            audio_processor_config=audio_processor_config,
        )
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        prepared_kvaf_branch = prepare_eawm_kvaf_branch(
            self.pipe,
            context_interval=context_interval,
        )
        if lora_base_model is not None and not task.endswith(":data_process"):
            prepared_kvaf_branch = self.add_lora_to_model(
                prepared_kvaf_branch,
                target_modules=self.parse_lora_target_modules(prepared_kvaf_branch, lora_target_modules),
                lora_rank=lora_rank,
                upcast_dtype=self.pipe.torch_dtype,
            )

        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint,
            preset_lora_path,
            preset_lora_model,
            task=task,
        )

        inject_eawm(
            self.pipe,
            kvaf_input_key=kvaf_input_key,
            context_interval=context_interval,
            prebuilt_kvaf_branch=prepared_kvaf_branch,
            context_residual_scale=context_residual_scale,
        )
        self.load_resume_dit_checkpoint(resume_dit_checkpoint)

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs not in (None, "") else []
        self.fp8_models = fp8_models
        self.task = task
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.kvaf_input_key = kvaf_input_key
        self.action_input_key = action_input_key
        self.context_residual_scale = context_residual_scale
        self.event_loss_weight = float(event_loss_weight)

        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: eawm_flow_match_sft_loss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: eawm_flow_match_sft_loss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.current_stage = "stage2"

    @staticmethod
    def _resolve_resume_dit_checkpoint_path(checkpoint_path: str) -> str:
        candidates = [checkpoint_path]
        if not checkpoint_path.endswith(".safetensors"):
            candidates.append(f"{checkpoint_path}.safetensors")
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(
            f"Resume checkpoint not found: {checkpoint_path}. "
            f"Tried {', '.join(candidates)}."
        )

    def load_resume_dit_checkpoint(self, checkpoint_path: str):
        if checkpoint_path in (None, ""):
            return

        checkpoint_path = self._resolve_resume_dit_checkpoint_path(str(checkpoint_path))
        state_dict = load_state_dict(checkpoint_path, torch_dtype=torch.float32, device="cpu")
        normalized_state_dict = {}
        for name, value in state_dict.items():
            if name.startswith("pipe.dit."):
                name = name[len("pipe.dit."):]
            name = name.replace("robot_sparse_context_branch.", "eawm_kvaf_branch.")
            name = name.replace("eawm_sparse_context_branch.", "eawm_kvaf_branch.")
            name = name.replace("uv_from_video", "kvaf_from_video")
            name = name.replace("video_from_uv", "video_from_kvaf")
            name = name.replace("uv_head", "kvaf_head")
            normalized_state_dict[name] = value

        model_state_dict = self.pipe.dit.state_dict()
        matched_keys = [name for name in normalized_state_dict if name in model_state_dict]
        unexpected_keys = [name for name in normalized_state_dict if name not in model_state_dict]
        load_result = self.pipe.dit.load_state_dict(normalized_state_dict, strict=False)
        if unexpected_keys:
            preview = ", ".join(unexpected_keys[:8])
            print(
                f"[ResumeCheckpoint] unexpected checkpoint keys skipped: {preview}"
                + (" ..." if len(unexpected_keys) > 8 else "")
            )
        print(
            f"[ResumeCheckpoint] loaded {len(matched_keys)}/{len(normalized_state_dict)} tensors "
            f"from {checkpoint_path} into pipe.dit."
        )
        if len(load_result.unexpected_keys) > 0:
            print(f"[ResumeCheckpoint] unexpected model load keys: {list(load_result.unexpected_keys)[:8]}")

    @staticmethod
    def _is_lora_param(name: str) -> bool:
        return ".lora_A." in name or ".lora_B." in name

    @classmethod
    def _is_kvaf_block_param(cls, name: str) -> bool:
        return name.startswith(cls.KVAF_BLOCK_PREFIX)

    @classmethod
    def _is_fusion_param(cls, name: str) -> bool:
        return name.startswith(cls.FUSION_PREFIX)

    @classmethod
    def _is_kvaf_head_param(cls, name: str) -> bool:
        return name.startswith(cls.KVAF_HEAD_PREFIX)

    @classmethod
    def _is_kvaf_block_lora_param(cls, name: str) -> bool:
        return cls._is_kvaf_block_param(name) and cls._is_lora_param(name)

    @classmethod
    def _is_main_lora_param(cls, name: str) -> bool:
        return cls._is_lora_param(name) and not name.startswith("pipe.dit.eawm_kvaf_branch.")

    def _group_named_parameters(self):
        groups = {
            "main_lora": [],
            "fusion": [],
            "kvaf_head": [],
            "kvaf_lora": [],
            "other_trainable": [],
        }
        for name, param in self.named_parameters():
            if self._is_main_lora_param(name):
                groups["main_lora"].append((name, param))
            elif self._is_fusion_param(name):
                groups["fusion"].append((name, param))
            elif self._is_kvaf_head_param(name):
                groups["kvaf_head"].append((name, param))
            elif self._is_kvaf_block_lora_param(name):
                groups["kvaf_lora"].append((name, param))
            elif param.requires_grad:
                groups["other_trainable"].append((name, param))
        return groups

    def zero_init_context_fusion_output_projections(self):
        branch = getattr(getattr(self.pipe, "dit", None), "eawm_kvaf_branch", None)
        if branch is None:
            return
        with torch.no_grad():
            for cross_attention in branch.bidirectional_cross_attentions:
                if hasattr(cross_attention, "zero_init_output_projections"):
                    cross_attention.zero_init_output_projections()

    def set_training_stage(self, stage: str):
        if stage not in {"stage1", "stage2"}:
            raise ValueError(f"Unsupported training stage: {stage}")

        previous_stage = self.current_stage
        for name, param in self.named_parameters():
            if self._is_main_lora_param(name) or self._is_kvaf_head_param(name):
                param.requires_grad = True
            elif self._is_fusion_param(name):
                param.requires_grad = (stage == "stage2")
            elif self._is_kvaf_block_lora_param(name):
                param.requires_grad = True
            elif self._is_kvaf_block_param(name):
                param.requires_grad = False

        if previous_stage == "stage1" and stage == "stage2":
            self.zero_init_context_fusion_output_projections()
            print("[TrainStage] zero-initialized bidirectional cross-attention output projections for stage2.")

        self.current_stage = stage

        grouped = self._group_named_parameters()

        def _numel(named_params):
            return sum(param.numel() for _, param in named_params)

        def _numel_trainable(named_params):
            return sum(param.numel() for _, param in named_params if param.requires_grad)

        print(
            f"[TrainStage] {stage}: "
            f"MainLoRA={_numel_trainable(grouped['main_lora'])}/{_numel(grouped['main_lora'])}, "
            f"Fusion={_numel_trainable(grouped['fusion'])}/{_numel(grouped['fusion'])}, "
            f"KVAFHead={_numel_trainable(grouped['kvaf_head'])}/{_numel(grouped['kvaf_head'])}, "
            f"KVAFLoRA={_numel_trainable(grouped['kvaf_lora'])}/{_numel(grouped['kvaf_lora'])} trainable params"
        )

    def build_optimizer_param_groups(self, args=None, base_learning_rate=1e-4, weight_decay=1e-2):
        lora_lr = getattr(args, "lora_learning_rate", None) if args is not None else None
        context_lr = getattr(args, "context_learning_rate", None) if args is not None else None
        context_block_lr = getattr(args, "context_block_learning_rate", None) if args is not None else None

        lora_lr = base_learning_rate if lora_lr is None else float(lora_lr)
        context_lr = base_learning_rate if context_lr is None else float(context_lr)
        context_block_lr = context_lr if context_block_lr is None else float(context_block_lr)

        grouped = self._group_named_parameters()
        param_groups = []

        if len(grouped["main_lora"]) > 0:
            param_groups.append(
                {
                    "params": [param for _, param in grouped["main_lora"]],
                    "lr": lora_lr,
                    "weight_decay": weight_decay,
                }
            )
        if len(grouped["fusion"]) > 0:
            param_groups.append(
                {
                    "params": [param for _, param in grouped["fusion"]],
                    "lr": context_lr,
                    "weight_decay": weight_decay,
                }
            )
        if len(grouped["kvaf_head"]) > 0:
            param_groups.append(
                {
                    "params": [param for _, param in grouped["kvaf_head"]],
                    "lr": context_lr,
                    "weight_decay": weight_decay,
                }
            )
        if len(grouped["kvaf_lora"]) > 0:
            param_groups.append(
                {
                    "params": [param for _, param in grouped["kvaf_lora"]],
                    "lr": context_block_lr,
                    "weight_decay": weight_decay,
                }
            )
        if len(grouped["other_trainable"]) > 0:
            param_groups.append(
                {
                    "params": [param for _, param in grouped["other_trainable"]],
                    "lr": base_learning_rate,
                    "weight_decay": weight_decay,
                }
            )

        def _numel(named_params):
            return sum(param.numel() for _, param in named_params)

        print(
            "[OptimizerGroups] "
            f"lora_lr={lora_lr}, context_lr={context_lr}, context_block_lr={context_block_lr}, "
            f"main_lora_params={_numel(grouped['main_lora'])}, "
            f"fusion_params={_numel(grouped['fusion'])}, "
            f"kvaf_head_params={_numel(grouped['kvaf_head'])}, "
            f"kvaf_lora_params={_numel(grouped['kvaf_lora'])}, "
            f"other_params={_numel(grouped['other_trainable'])}"
        )

        return param_groups

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input in {"reference_image", "vace_reference_image"}:
                inputs_shared[extra_input] = data[extra_input][0]
            elif extra_input in data:
                inputs_shared[extra_input] = data[extra_input]
        if inputs_shared.get("framewise_decoding", False):
            inputs_shared["num_frames"] = 4 * (len(data["video"]) - 1) + 1
        return inputs_shared

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        print(inputs_posi)
        inputs_nega = {}
        inputs_shared = {
            "input_video": data["video"],
            "input_image": data["video"][0],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "event_loss_weight": self.event_loss_weight,
            self.kvaf_input_key: data.get(self.kvaf_input_key),
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss



def wan_eawm_parser():
    parser = argparse.ArgumentParser(description="Train EA-WM on top of Wan2.2-TI2V-5B.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--local-rank", type=int, default=-1, help="local rank")
    parser.set_defaults(data_file_keys="video,action_path,kvaf_path")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0)
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0)
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true")
    parser.add_argument("--framewise_decoding", default=False, action="store_true")
    parser.add_argument("--kvaf_input_key", type=str, default="kvaf_path", help="Metadata column that points to the KVAF heatmap folder.")
    parser.add_argument("--action_input_key", type=str, default="action_path", help="Metadata column kept for dataset parsing only. Action injection is removed.")
    parser.add_argument("--context_interval", type=int, default=1, help="Bidirectional cross-attention injection interval. The KVAF branch still copies and runs through every DiT block.")
    parser.add_argument("--context_residual_scale", type=float, default=1.0, help="Legacy argument kept for compatibility. It is unused in bidirectional cross-attention mode.")
    parser.add_argument("--stage1_epochs", type=int, default=0, help="Epochs for stage-1 training. Stage 1 freezes bidirectional cross-attention and trains the main-branch LoRA, KVAF-branch LoRA, and KVAF prediction head. Stage 2 unfreezes bidirectional cross-attention with zero-output initialization.")
    parser.add_argument("--lora_learning_rate", type=float, default=None, help="Learning rate for LoRA parameters. Defaults to --learning_rate.")
    parser.add_argument("--context_learning_rate", type=float, default=None, help="Learning rate for bidirectional fusion and the KVAF prediction head. Defaults to --learning_rate.")
    parser.add_argument("--context_block_learning_rate", type=float, default=None, help="Learning rate for KVAF-branch LoRA parameters. Defaults to --context_learning_rate.")
    parser.add_argument("--event_loss_weight", type=float, default=0.1, help="Weight for direct MSE supervision on per-fusion event latent predictions.")
    parser.add_argument("--resume_dit_checkpoint", type=str, default=None, help="Optional combined training checkpoint exported by this script, e.g. epoch-5.safetensors. When provided, it is loaded into pipe.dit after EAWM branch/LoRA construction so training can continue from a previous run.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping max norm. Set <=0 to disable clipping.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["constant", "linear", "cosine"], help="LR scheduler after warmup.")
    parser.add_argument("--lr_warmup_steps", type=int, default=100, help="Warmup steps for LR scheduler.")
    return parser


if __name__ == "__main__":
    parser = wan_eawm_parser()
    args = parser.parse_args()

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)
        ],
    )

    video_frame_processor = ImageCropAndResize(
        args.height,
        args.width,
        args.max_pixels,
        16,
        16,
    )

    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4 if not args.framewise_decoding else 1,
            time_division_remainder=1 if not args.framewise_decoding else 0,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(
                args.num_frames,
                4,
                1,
                frame_processor=ImageCropAndResize(512, 512, None, 16, 16),
            ),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
            "wantodance_music_path": ToAbsolutePath(args.dataset_base_path),
            args.kvaf_input_key: ToAbsolutePath(args.dataset_base_path) >> LoadImageFolder(
                frame_processor=video_frame_processor,
                max_frames=args.num_frames,
            ),
            args.action_input_key: ToAbsolutePath(args.dataset_base_path) >> LoadActionCSV(action_dim=14),
        },
    )

    model = WanEAWMTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        kvaf_input_key=args.kvaf_input_key,
        action_input_key=args.action_input_key,
        context_interval=args.context_interval,
        context_residual_scale=args.context_residual_scale,
        event_loss_weight=args.event_loss_weight,
        resume_dit_checkpoint=args.resume_dit_checkpoint,
    )

    model_logger = EAWMCheckpointLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )

    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
