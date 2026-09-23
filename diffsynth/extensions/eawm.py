import copy
import inspect
import os
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from einops import rearrange

from diffsynth.core.data.operators import DataProcessingOperator
from diffsynth.diffusion.base_pipeline import PipelineUnit
from diffsynth.models.wan_video_dit import (
    CrossAttention,
    gradient_checkpoint_forward,
    sinusoidal_embedding_1d,
)


IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}


def _as_float_tensor(image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = image.detach().float().cpu()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[0] not in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        elif tensor.shape[0] >= 3:
            tensor = tensor[:3]
        else:
            raise ValueError(f"Unsupported tensor image shape: {tuple(tensor.shape)}")
        if tensor.max().item() > 1.0:
            tensor = tensor / 255.0
        return tensor.clamp(0.0, 1.0)

    if not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported KVAF frame type: {type(image)}")

    image = image.convert("RGB")
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = data.reshape(image.height, image.width, 3).permute(2, 0, 1).float() / 255.0
    return tensor.clamp(0.0, 1.0)


def _tensor_to_pil_rgb(image: Union[Image.Image, torch.Tensor]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    tensor = _as_float_tensor(image)
    tensor = (tensor * 255.0).round().to(torch.uint8)
    array = tensor.permute(1, 2, 0).contiguous().numpy()
    return Image.fromarray(array, mode="RGB")


def _video_to_float_frames(video: Union[List[Image.Image], Tuple[Image.Image, ...], torch.Tensor]) -> List[torch.Tensor]:
    if isinstance(video, torch.Tensor):
        tensor = video.detach().float().cpu()
        if tensor.ndim == 5:
            if tensor.shape[0] != 1:
                raise ValueError(f"Expected a batch size of 1 for video tensor, got shape {tuple(tensor.shape)}")
            tensor = tensor[0]
        if tensor.ndim == 4:
            if tensor.shape[0] in (1, 3, 4):
                tensor = tensor.permute(1, 0, 2, 3)
            elif tensor.shape[-1] in (1, 3, 4):
                tensor = tensor.permute(0, 3, 1, 2)
            elif tensor.shape[1] not in (1, 3, 4):
                raise ValueError(f"Unsupported video tensor shape: {tuple(tensor.shape)}")
        else:
            raise ValueError(f"Unsupported video tensor shape: {tuple(tensor.shape)}")

        if tensor.min().item() < 0.0:
            tensor = (tensor + 1.0) * 0.5
        elif tensor.max().item() > 1.0:
            tensor = tensor / 255.0
        return [_as_float_tensor(frame) for frame in tensor.clamp(0.0, 1.0)]

    if not isinstance(video, (list, tuple)):
        raise TypeError(f"Unsupported video type for event difference encoding: {type(video)}")
    return [_as_float_tensor(frame) for frame in video]


def _absolute_difference_video_tensor(
    video: Union[List[Image.Image], Tuple[Image.Image, ...], torch.Tensor],
    device,
    dtype,
) -> torch.Tensor:
    frames = _video_to_float_frames(video)
    if len(frames) == 0:
        raise RuntimeError("Cannot build event difference target from an empty video.")

    diff_frames = [torch.zeros_like(frames[0])]
    for prev_frame, cur_frame in zip(frames[:-1], frames[1:]):
        if prev_frame.shape != cur_frame.shape:
            raise RuntimeError(
                f"Video frame shape changed while building event difference target: "
                f"{tuple(prev_frame.shape)} vs {tuple(cur_frame.shape)}"
            )
        diff_frames.append((cur_frame - prev_frame).abs().clamp(0.0, 1.0))

    diff_video = torch.stack(diff_frames, dim=1).unsqueeze(0)
    diff_video = diff_video * 2.0 - 1.0
    return diff_video.to(device=device, dtype=dtype)


def _first_param_or_buffer(module: nn.Module):
    for tensor in module.parameters(recurse=True):
        return tensor
    for tensor in module.buffers(recurse=True):
        return tensor
    return None


def _get_module_device_dtype(module: nn.Module, default_device=None, default_dtype=None):
    ref = _first_param_or_buffer(module)
    if ref is None:
        return default_device, default_dtype
    return ref.device, ref.dtype


def _assert_module_has_no_lora_params(module: nn.Module, module_name: str):
    lora_param_names = [
        name for name, _ in module.named_parameters()
        if "lora_" in name or ".lora_" in name
    ]
    if len(lora_param_names) > 0:
        preview = ", ".join(lora_param_names[:8])
        raise RuntimeError(
            f"{module_name} unexpectedly contains LoRA parameters: {preview}. "
            "The KVAF branch must be copied before LoRA adapters are attached."
        )


class LoadImageFolder(DataProcessingOperator):
    def __init__(self, frame_processor=lambda x: x, max_frames: Optional[int] = None):
        self.frame_processor = frame_processor
        self.max_frames = max_frames

    def __call__(self, data: str) -> List[Image.Image]:
        folder = Path(data)
        if not folder.is_dir():
            raise FileNotFoundError(f"kvaf_path is not a directory: {data}")
        frame_paths = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower().lstrip(".") in IMAGE_EXTENSIONS
        )
        if len(frame_paths) == 0:
            raise RuntimeError(f"No heatmap frames found under {data}")
        if self.max_frames is not None:
            frame_paths = frame_paths[: self.max_frames]
        frames: List[Image.Image] = []
        for frame_path in frame_paths:
            frame = Image.open(frame_path).convert("RGB")
            frame = self.frame_processor(frame)
            frames.append(frame)
        return frames


class LoadActionCSV(DataProcessingOperator):
    def __init__(self, action_dim: int = 14):
        self.action_dim = action_dim

    def __call__(self, data: str) -> torch.Tensor:
        table = pd.read_csv(data)
        if len(table) == 0:
            raise RuntimeError(f"Action csv is empty: {data}")

        drop_cols = [
            c for c in table.columns
            if c in {"from_sampled_frame_idx", "to_sampled_frame_idx"}
            or c.endswith("frame_idx")
            or c.endswith("frame_id")
        ]
        if len(drop_cols) > 0:
            table = table.drop(columns=drop_cols)

        numeric_cols = [c for c in table.columns if pd.api.types.is_numeric_dtype(table[c])]
        table = table[numeric_cols]
        if table.shape[1] < self.action_dim:
            raise RuntimeError(
                f"Action csv has only {table.shape[1]} numeric columns after preprocessing, expected >= {self.action_dim}."
            )
        if table.shape[1] > self.action_dim:
            table = table.iloc[:, : self.action_dim]
        return torch.tensor(table.values, dtype=torch.float32)


class WanVideoUnit_EAWMKVAFEncoder(PipelineUnit):
    def __init__(self, kvaf_input_key: str = "kvaf_path"):
        super().__init__(
            input_params=(kvaf_input_key, "tiled", "tile_size", "tile_stride", "framewise_decoding"),
            output_params=("kvaf_context_latents",),
            onload_model_names=("vae",),
        )
        self.kvaf_input_key = kvaf_input_key

    def process(self, pipe, kvaf_path, tiled, tile_size, tile_stride, framewise_decoding):
        if kvaf_path is None:
            return {"kvaf_context_latents": None}
        if getattr(pipe, "vae", None) is None:
            raise RuntimeError("pipe.vae is required for KVAF encoding but was not found.")
        if not callable(getattr(pipe, "preprocess_video", None)):
            raise RuntimeError(
                "Current WanVideoPipeline does not provide preprocess_video(), cannot strictly follow official VAE preprocessing."
            )
        if not callable(getattr(pipe.vae, "encode", None)):
            raise RuntimeError(
                "Current pipe.vae does not provide encode(), cannot strictly follow official Wan VAE encoding."
            )
        if framewise_decoding:
            encode_framewise = getattr(pipe.vae, "encode_framewise", None)
            if not callable(encode_framewise):
                raise RuntimeError("framewise_decoding=True but pipe.vae.encode_framewise() is unavailable.")

        pipe.load_models_to_device(self.onload_model_names)
        rgb_frames = [_tensor_to_pil_rgb(frame) for frame in kvaf_path]
        kvaf_video = pipe.preprocess_video(rgb_frames)

        with torch.no_grad():
            if framewise_decoding:
                kvaf_context_latents = pipe.vae.encode_framewise(kvaf_video, device=pipe.device)
            else:
                kvaf_context_latents = pipe.vae.encode(
                    kvaf_video,
                    device=pipe.device,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                )

        kvaf_context_latents = kvaf_context_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"kvaf_context_latents": kvaf_context_latents}


class WanVideoUnit_EventDifferenceEncoder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "tiled", "tile_size", "tile_stride", "framewise_decoding"),
            output_params=("event_difference_latents",),
            onload_model_names=("vae",),
        )

    def process(self, pipe, input_video, tiled, tile_size, tile_stride, framewise_decoding):
        if input_video is None:
            return {"event_difference_latents": None}
        if getattr(pipe, "vae", None) is None:
            raise RuntimeError("pipe.vae is required for event difference encoding but was not found.")
        if not callable(getattr(pipe.vae, "encode", None)):
            raise RuntimeError("Current pipe.vae does not provide encode(), cannot build event difference latent targets.")
        if framewise_decoding:
            encode_framewise = getattr(pipe.vae, "encode_framewise", None)
            if not callable(encode_framewise):
                raise RuntimeError("framewise_decoding=True but pipe.vae.encode_framewise() is unavailable.")

        pipe.load_models_to_device(self.onload_model_names)
        event_video = _absolute_difference_video_tensor(
            input_video,
            device=pipe.device,
            dtype=pipe.torch_dtype,
        )

        with torch.no_grad():
            if framewise_decoding:
                event_difference_latents = pipe.vae.encode_framewise(event_video, device=pipe.device)
            else:
                event_difference_latents = pipe.vae.encode(
                    event_video,
                    device=pipe.device,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                )

        event_difference_latents = event_difference_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"event_difference_latents": event_difference_latents}


class EAWMCheckpointLogger:
    def __init__(self, output_path: str, remove_prefix_in_ckpt: Optional[str] = None, state_dict_converter=lambda x: x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0

    @staticmethod
    def split_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        lora_keys = [
            key for key in state_dict
            if ".lora_A." in key or ".lora_B." in key
        ]
        lora_state = {k: state_dict[k] for k in lora_keys}
        eawm_state = {k: v for k, v in state_dict.items() if k not in lora_state}
        return lora_state, eawm_state

    def _prepare_state_dict(self, accelerator, model) -> Dict[str, torch.Tensor]:
        while isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
            model = model.module
        state_dict = model.state_dict()
        state_dict = model.export_trainable_state_dict(
            state_dict,
            remove_prefix=self.remove_prefix_in_ckpt,
        )
        state_dict = self.state_dict_converter(state_dict)
        return state_dict

    def _save(self, accelerator, model, file_name: str):
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            return

        os.makedirs(self.output_path, exist_ok=True)
        state_dict = self._prepare_state_dict(accelerator, model)
        base_path = os.path.join(self.output_path, file_name)
        accelerator.save(state_dict, base_path, safe_serialization=True)

        stem = base_path[:-12] if base_path.endswith(".safetensors") else base_path
        lora_state, eawm_state = self.split_state_dict(state_dict)
        if len(lora_state) > 0:
            accelerator.save(lora_state, f"{stem}.lora.safetensors", safe_serialization=True)
        if len(eawm_state) > 0:
            accelerator.save(eawm_state, f"{stem}.eawm.safetensors", safe_serialization=True)

    def on_step_end(self, accelerator, model, save_steps=None, loss=None, **kwargs):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self._save(accelerator, model, f"step-{self.num_steps}.safetensors")

    def on_epoch_end(self, accelerator, model, epoch_id, **kwargs):
        self._save(accelerator, model, f"epoch-{epoch_id}.safetensors")

    def on_training_end(self, accelerator, model, save_steps=None, **kwargs):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self._save(accelerator, model, f"step-{self.num_steps}.safetensors")


class ZeroLinear(nn.Linear):
    def reset_parameters(self):
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class EAWMEventTokenMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        event_out_dim: int,
        eps: float = 1e-6,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        hidden_dim = max(128, dim // 8) if hidden_dim is None else int(hidden_dim)
        self.video_norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.kvaf_norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.event_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.event_mlp = nn.Sequential(
            nn.Linear(dim * 3, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, dim),
            nn.GELU(approximate="tanh"),
        )
        self.event_gate = nn.Linear(dim, 1)
        self.event_head = nn.Linear(dim, event_out_dim)
        self.zero_init_gate()
        self.zero_init_head()

    def zero_init_gate(self):
        nn.init.zeros_(self.event_gate.weight)
        if self.event_gate.bias is not None:
            nn.init.zeros_(self.event_gate.bias)

    def zero_init_head(self):
        nn.init.zeros_(self.event_head.weight)
        if self.event_head.bias is not None:
            nn.init.zeros_(self.event_head.bias)

    def forward(self, video_hidden: torch.Tensor, kvaf_hidden: torch.Tensor):
        event_query = self.event_query.to(device=video_hidden.device, dtype=video_hidden.dtype)
        event_query = event_query.expand(video_hidden.shape[0], video_hidden.shape[1], -1)
        event_input = torch.cat([
            self.video_norm(video_hidden),
            self.kvaf_norm(kvaf_hidden),
            event_query,
        ], dim=-1)
        event_token = self.event_mlp(event_input)
        event_gate = 2.0 * torch.sigmoid(self.event_gate(event_token))
        event_pred = self.event_head(event_token)
        return event_gate, event_pred


class EAWMBidirectionalCrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, event_out_dim: int, eps: float = 1e-6):
        super().__init__()
        self.video_norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.kvaf_norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.event_token = EAWMEventTokenMLP(dim, event_out_dim, eps=eps)
        self.video_from_kvaf = CrossAttention(dim, num_heads, eps=eps, has_image_input=False)
        self.kvaf_from_video = CrossAttention(dim, num_heads, eps=eps, has_image_input=False)
        self._zero_init_output_projection(self.video_from_kvaf)
        self._zero_init_output_projection(self.kvaf_from_video)

    @staticmethod
    def _zero_init_output_projection(cross_attention: CrossAttention):
        nn.init.zeros_(cross_attention.o.weight)
        if cross_attention.o.bias is not None:
            nn.init.zeros_(cross_attention.o.bias)

    def zero_init_output_projections(self):
        self._zero_init_output_projection(self.video_from_kvaf)
        self._zero_init_output_projection(self.kvaf_from_video)

    @staticmethod
    def _flatten_frames(hidden: torch.Tensor, num_frames: int) -> torch.Tensor:
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        batch_size, num_tokens, dim = hidden.shape
        if num_tokens % num_frames != 0:
            raise RuntimeError(
                f"Cannot reshape token sequence of length {num_tokens} into {num_frames} frames."
            )
        tokens_per_frame = num_tokens // num_frames
        return rearrange(
            hidden.contiguous(),
            "b (f n) c -> (b f) n c",
            f=num_frames,
            n=tokens_per_frame,
        )

    @staticmethod
    def _unflatten_frames(hidden: torch.Tensor, batch_size: int, num_frames: int) -> torch.Tensor:
        return rearrange(hidden.contiguous(), "(b f) n c -> b (f n) c", b=batch_size, f=num_frames)

    def forward(self, video_hidden: torch.Tensor, kvaf_hidden: torch.Tensor, num_frames: int):
        if video_hidden.shape != kvaf_hidden.shape:
            raise RuntimeError(
                f"Video/KVAF hidden shape mismatch in eawm bidirectional fusion: "
                f"{tuple(video_hidden.shape)} vs {tuple(kvaf_hidden.shape)}"
            )
        video_hidden_input = video_hidden
        kvaf_hidden_input = kvaf_hidden
        event_gate, event_pred = self.event_token(video_hidden_input, kvaf_hidden_input)

        batch_size = video_hidden_input.shape[0]
        video_queries = self._flatten_frames(self.video_norm(video_hidden_input), num_frames)
        kvaf_queries = self._flatten_frames(self.kvaf_norm(kvaf_hidden_input), num_frames)
        kvaf_context = self._flatten_frames(kvaf_hidden_input, num_frames)
        video_context = self._flatten_frames(video_hidden_input, num_frames)

        # Match the per-frame patch layout: each frame only attends to tokens from the same frame.
        video_hidden_delta = self.video_from_kvaf(video_queries, kvaf_context)
        kvaf_hidden_delta = self.kvaf_from_video(kvaf_queries, video_context)
        video_hidden_delta = self._unflatten_frames(video_hidden_delta, batch_size, num_frames)
        kvaf_hidden_delta = self._unflatten_frames(kvaf_hidden_delta, batch_size, num_frames)

        video_hidden = video_hidden_input + video_hidden_delta * event_gate
        kvaf_hidden = kvaf_hidden_input + kvaf_hidden_delta * event_gate
        return video_hidden, kvaf_hidden, event_pred


class EAWMKVAFBranch(nn.Module):
    def __init__(self, reference_dit: nn.Module, interval: int = 1):
        super().__init__()
        if interval <= 0:
            raise ValueError(f"interval must be positive, got {interval}")
        self.interval = interval
        self.block_indices = list(range(len(reference_dit.blocks)))
        self.injection_block_indices = list(range(0, len(reference_dit.blocks), interval))
        self.blocks = nn.ModuleList()
        self.bidirectional_cross_attentions = nn.ModuleList()

        dit_device, dit_dtype = _get_module_device_dtype(reference_dit, default_device=None, default_dtype=torch.float32)
        hidden_dim = int(reference_dit.dim)
        block_eps = float(reference_dit.blocks[0].norm1.eps)
        num_heads = int(reference_dit.blocks[0].num_heads)
        event_out_dim = int(reference_dit.head.head.out_features)
        for main_block_index in self.block_indices:
            copied_block = copy.deepcopy(reference_dit.blocks[main_block_index]).to(device=dit_device, dtype=dit_dtype)
            _assert_module_has_no_lora_params(copied_block, f"context block copy from main block {main_block_index}")
            self.blocks.append(copied_block)

        for main_block_index in self.injection_block_indices:
            bidirectional_cross_attention = EAWMBidirectionalCrossAttentionBlock(
                hidden_dim,
                num_heads=num_heads,
                event_out_dim=event_out_dim,
                eps=block_eps,
            ).to(device=dit_device, dtype=dit_dtype)
            _assert_module_has_no_lora_params(
                bidirectional_cross_attention,
                f"bidirectional cross attention around main block {main_block_index}",
            )
            self.bidirectional_cross_attentions.append(bidirectional_cross_attention)

        self.kvaf_head = copy.deepcopy(reference_dit.head).to(
            device=dit_device,
            dtype=dit_dtype,
        )
        _assert_module_has_no_lora_params(self.kvaf_head, "eawm kvaf prediction head")

        self.hidden_dim = hidden_dim


class EAWMLoader:
    def __init__(self, pipe, kvaf_input_key: str = "kvaf_path"):
        self.pipe = pipe
        self.kvaf_input_key = kvaf_input_key

    def patch_units(self):
        units = list(self.pipe.units)
        if len(units) < 2:
            raise RuntimeError("Unexpected Wan pipeline layout: units list is too short.")
        new_units = []
        inserted_kvaf = False
        inserted_event = False
        for unit in units:
            new_units.append(unit)
            if unit.__class__.__name__ == "WanVideoUnit_ShapeChecker":
                new_units.append(WanVideoUnit_EAWMKVAFEncoder(kvaf_input_key=self.kvaf_input_key))
                new_units.append(WanVideoUnit_EventDifferenceEncoder())
                inserted_kvaf = True
                inserted_event = True
        if not inserted_kvaf or not inserted_event:
            raise RuntimeError(
                "Unable to locate WanVideoUnit_ShapeChecker in pipe.units; refusing to inject EAWM encoders with a non-official layout."
            )
        self.pipe.units = new_units

    def patch_model_fn(self):
        self.pipe.model_fn = eawm_model_fn_wan_video

    def patch_unit_runner(self):
        base_unit_runner = self.pipe.unit_runner

        def unit_runner_with_eawm(pipeline_self, unit, pipe, inputs_shared, inputs_posi, inputs_nega):
            extra_inputs = getattr(pipeline_self, "_eawm_extra_inputs", None)
            if extra_inputs:
                inputs_shared = dict(inputs_shared)
                for key, value in extra_inputs.items():
                    if value is not None:
                        inputs_shared[key] = value

            return base_unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)

        self.pipe.unit_runner = types.MethodType(unit_runner_with_eawm, self.pipe)


def eawm_model_fn_wan_video(
    dit,
    motion_controller=None,
    vace=None,
    vap=None,
    animate_adapter=None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents=None,
    vace_context=None,
    vace_scale=1.0,
    audio_embeds: Optional[torch.Tensor] = None,
    motion_latents: Optional[torch.Tensor] = None,
    s2v_pose_latents: Optional[torch.Tensor] = None,
    vap_hidden_state=None,
    vap_clip_feature=None,
    context_vap=None,
    drop_motion_frames: bool = True,
    tea_cache=None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    pose_latents=None,
    face_pixel_values=None,
    longcat_latents=None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input=None,
    fuse_vae_embedding_in_latents: bool = False,
    wantodance_refimage_feature=None,
    wantodance_fps: float = 30.0,
    music_feature=None,
    skip_9th_layer: bool = False,
    kvaf_context_latents: Optional[torch.Tensor] = None,
    return_eawm_aux: bool = False,
    **kwargs,
):
    unsupported_features = []
    if sliding_window_size is not None or sliding_window_stride is not None:
        unsupported_features.append("sliding_window")
    if audio_embeds is not None or motion_latents is not None or s2v_pose_latents is not None:
        unsupported_features.append("wan2.2_s2v")
    if use_unified_sequence_parallel:
        unsupported_features.append("unified_sequence_parallel")
    if vace is not None or vace_context is not None:
        unsupported_features.append("vace")
    if vap is not None or vap_hidden_state is not None or vap_clip_feature is not None or context_vap is not None:
        unsupported_features.append("vap")
    if animate_adapter is not None or pose_latents is not None or face_pixel_values is not None:
        unsupported_features.append("animate_adapter")
    if longcat_latents is not None:
        unsupported_features.append("longcat")
    if tea_cache is not None:
        unsupported_features.append("tea_cache")
    if reference_latents is not None:
        unsupported_features.append("reference_latents")
    if getattr(dit, "wantodance_enable_global", False) or wantodance_refimage_feature is not None or music_feature is not None:
        unsupported_features.append("wantodance")
    if control_camera_latents_input is not None:
        unsupported_features.append("control_camera")
    if skip_9th_layer:
        unsupported_features.append("skip_9th_layer")

    if unsupported_features:
        raise RuntimeError(
            "The EAWM KVAF branch currently supports the standard Wan2.2 TI2V path. "
            f"Unsupported active features detected: {', '.join(unsupported_features)}"
        )

    if latents is None or timestep is None or context is None:
        raise RuntimeError("eawm_model_fn_wan_video requires latents, timestep, and context.")
    if getattr(dit, "eawm_kvaf_branch", None) is None:
        raise RuntimeError("eawm_kvaf_branch is missing from DiT before model_fn execution.")
    if not callable(getattr(dit, "patchify", None)):
        raise RuntimeError("Current Wan DiT has no patchify() method; strict official EAWM patch refuses to continue.")
    if getattr(dit, "patch_embedding", None) is None:
        raise RuntimeError("Current Wan DiT has no patch_embedding module; strict official EAWM patch refuses to continue.")
    if kvaf_context_latents is None:
        raise RuntimeError(
            "kvaf_context_latents is missing while EAWM sparse context injection is enabled. "
            "This strict variant requires every training/inference step to provide KVAF context."
        )

    raw_timestep = timestep
    use_kvaf_separate_timestep = bool(getattr(dit, "eawm_use_kvaf_separate_timestep", True))
    if getattr(dit, "seperated_timestep", False) and fuse_vae_embedding_in_latents:
        timestep = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep,
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))

        if use_kvaf_separate_timestep:
            # The main TI2V branch locks the first frame as an image condition,
            # so its first-frame timestep is zero. The KVAF branch denoises every
            # frame, including frame 0, so it must use the actual diffusion timestep.
            kvaf_t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, raw_timestep))
            kvaf_t_mod = dit.time_projection(kvaf_t).unflatten(1, (6, dit.dim))
        else:
            kvaf_t = t
            kvaf_t_mod = t_mod
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
        kvaf_t = t
        kvaf_t_mod = t_mod

    if motion_bucket_id is not None and motion_controller is not None:
        motion_t_mod = motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
        t_mod = t_mod + motion_t_mod
        kvaf_t_mod = kvaf_t_mod + motion_t_mod
    context = dit.text_embedding(context)

    x = latents
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    if y is not None and getattr(dit, "require_vae_embedding", False):
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and getattr(dit, "require_clip_embedding", False):
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    x = dit.patchify(x, control_camera_latents_input)
    f, h, w = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()

    branch_device, branch_dtype = _get_module_device_dtype(
        dit.eawm_kvaf_branch,
        default_device=latents.device,
        default_dtype=latents.dtype,
    )
    kvaf_context_latents = kvaf_context_latents.to(device=branch_device, dtype=branch_dtype)
    if kvaf_context_latents.shape[0] != context.shape[0]:
        kvaf_context_latents = torch.concat([kvaf_context_latents] * context.shape[0], dim=0)
    branch_hidden = dit.patchify(kvaf_context_latents, None)
    kvaf_f, kvaf_h, kvaf_w = branch_hidden.shape[2:]
    if (kvaf_f, kvaf_h, kvaf_w) != (f, h, w):
        raise RuntimeError(
            f"KVAF context grid {(kvaf_f, kvaf_h, kvaf_w)} does not match main DiT grid {(f, h, w)} after official patchify()."
        )
    branch_hidden = rearrange(branch_hidden, 'b c f h w -> b (f h w) c').contiguous()
    if branch_hidden.shape != x.shape:
        raise RuntimeError(
            f"KVAF context token shape {tuple(branch_hidden.shape)} does not match main token shape {tuple(x.shape)} after official patchify()."
        )

    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    branch = dit.eawm_kvaf_branch
    injection_map = dit.eawm_context_injection_map
    branch_call_count = 0
    fusion_call_count = 0
    event_pred_latents = [] if return_eawm_aux else None

    for block_id, block in enumerate(dit.blocks):
        context_block = branch.blocks[block_id]
        branch_hidden = gradient_checkpoint_forward(
            context_block,
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            branch_hidden,
            context,
            kvaf_t_mod,
            freqs,
        )
        branch_call_count += 1

        fusion_idx = injection_map.get(block_id)
        if fusion_idx is not None:
            bidirectional_cross_attention = branch.bidirectional_cross_attentions[fusion_idx]
            x, branch_hidden, event_pred_tokens = gradient_checkpoint_forward(
                bidirectional_cross_attention,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                x,
                branch_hidden,
                f,
            )
            if return_eawm_aux:
                event_pred_latents.append(dit.unpatchify(event_pred_tokens, (f, h, w)))
            fusion_call_count += 1

        x = gradient_checkpoint_forward(
            block,
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            x,
            context,
            t_mod,
            freqs,
        )

    if branch_call_count != len(branch.blocks):
        raise RuntimeError(
            f"EAWM KVAF branch was expected to run {len(branch.blocks)} times, but ran {branch_call_count} times."
        )
    if fusion_call_count != len(branch.bidirectional_cross_attentions):
        raise RuntimeError(
            "EAWM sparse context fusion was expected to run "
            f"{len(branch.bidirectional_cross_attentions)} times, but ran {fusion_call_count} times."
        )

    eawm_aux = None
    if return_eawm_aux:
        kvaf_pred_latents = branch.kvaf_head(branch_hidden, kvaf_t)
        kvaf_pred_latents = dit.unpatchify(kvaf_pred_latents, (f, h, w))
        eawm_aux = {
            "kvaf_pred_latents": kvaf_pred_latents,
            "event_pred_latents": event_pred_latents,
        }

    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    if return_eawm_aux:
        return x, eawm_aux
    return x


def prepare_eawm_kvaf_branch(pipe, context_interval: int = 1) -> EAWMKVAFBranch:
    if getattr(pipe, "dit", None) is None:
        raise RuntimeError("prepare_eawm_kvaf_branch requires pipe.dit to exist.")
    if getattr(pipe.dit, "_eawm_patched", False):
        raise RuntimeError("DiT is already patched; prepare_eawm_kvaf_branch must be called before final EAWM injection.")
    branch = EAWMKVAFBranch(pipe.dit, interval=context_interval)
    _assert_module_has_no_lora_params(branch, "prepared EAWM KVAF branch")
    return branch


def _patch_single_dit(dit: nn.Module, context_interval: int = 1, prebuilt_kvaf_branch: Optional[EAWMKVAFBranch] = None):
    if getattr(dit, "_eawm_patched", False):
        return

    if not callable(getattr(dit, "patchify", None)):
        raise RuntimeError("Current Wan DiT has no patchify() method; strict official EAWM patch refuses to continue.")
    if getattr(dit, "patch_embedding", None) is None:
        raise RuntimeError("Current Wan DiT has no patch_embedding module; strict official EAWM patch refuses to continue.")

    dit_device, dit_dtype = _get_module_device_dtype(dit, default_device=None, default_dtype=torch.float32)
    kvaf_branch = prebuilt_kvaf_branch if prebuilt_kvaf_branch is not None else EAWMKVAFBranch(dit, interval=context_interval)
    kvaf_branch = kvaf_branch.to(device=dit_device, dtype=dit_dtype)

    if kvaf_branch.interval != context_interval:
        raise RuntimeError(
            f"Prebuilt KVAF branch interval {kvaf_branch.interval} does not match requested interval {context_interval}."
        )

    dit.eawm_kvaf_branch = kvaf_branch
    dit.eawm_context_interval = context_interval
    dit.eawm_kvaf_block_indices = list(dit.eawm_kvaf_branch.block_indices)
    dit.eawm_fusion_block_indices = list(dit.eawm_kvaf_branch.injection_block_indices)
    dit.eawm_context_injection_map = {
        main_block_index: fusion_index
        for fusion_index, main_block_index in enumerate(dit.eawm_fusion_block_indices)
    }
    dit._eawm_patched = True


def inject_eawm(
    pipe,
    kvaf_input_key: str = "kvaf_path",
    context_interval: int = 1,
    prebuilt_kvaf_branch: Optional[EAWMKVAFBranch] = None,
    context_residual_scale: Optional[float] = 1.0,
    eawm_context_residual_scale: Optional[float] = None,
):
    def _bound_eawm_generate(pipeline_self, *args, **kwargs):
        return eawm_generate(pipeline_self, *args, **kwargs)

    if eawm_context_residual_scale is not None:
        context_residual_scale = eawm_context_residual_scale
    context_residual_scale = float(context_residual_scale)
    if getattr(pipe, "_eawm_injected", False):
        pipe.dit.eawm_context_residual_scale = context_residual_scale
        pipe.dit.eawm_kvaf_input_key = kvaf_input_key
        pipe.eawm_kvaf_input_key = kvaf_input_key
        pipe.eawm_generate = types.MethodType(_bound_eawm_generate, pipe)
        return pipe

    if getattr(pipe, "dit", None) is None:
        raise RuntimeError("This patch currently expects pipe.dit to exist (Wan2.2-TI2V-5B path).")
    if getattr(pipe, "vae", None) is None:
        raise RuntimeError("This patch requires pipe.vae to exist so KVAF heatmaps can be encoded with the official Wan VAE.")

    _patch_single_dit(pipe.dit, context_interval=context_interval, prebuilt_kvaf_branch=prebuilt_kvaf_branch)
    pipe.dit.eawm_context_residual_scale = context_residual_scale
    pipe.dit.eawm_kvaf_input_key = kvaf_input_key
    pipe.eawm_kvaf_input_key = kvaf_input_key

    loader = EAWMLoader(pipe, kvaf_input_key=kvaf_input_key)
    loader.patch_units()
    loader.patch_model_fn()
    loader.patch_unit_runner()

    pipe.vae.eval()
    for param in pipe.vae.parameters():
        param.requires_grad = False

    for param in pipe.dit.eawm_kvaf_branch.parameters():
        param.requires_grad = True

    pipe.eawm_generate = types.MethodType(_bound_eawm_generate, pipe)
    pipe._eawm_injected = True
    return pipe


def _bind_wan_call_args(pipe, call_kwargs: dict) -> dict:
    call_signature = inspect.signature(type(pipe).__call__)
    bound = call_signature.bind(pipe, **call_kwargs)
    bound.apply_defaults()
    call_args = dict(bound.arguments)
    call_args.pop("self", None)
    return call_args


def _prepare_eawm_generation_inputs(pipe, call_args: dict, kvaf_path):
    pipe.scheduler.set_timesteps(
        call_args["num_inference_steps"],
        denoising_strength=call_args["denoising_strength"],
        shift=call_args["sigma_shift"],
    )

    inputs_posi = {
        "prompt": call_args["prompt"],
        "vap_prompt": call_args["vap_prompt"],
        "tea_cache_l1_thresh": call_args["tea_cache_l1_thresh"],
        "tea_cache_model_id": call_args["tea_cache_model_id"],
        "num_inference_steps": call_args["num_inference_steps"],
    }
    inputs_nega = {
        "negative_prompt": call_args["negative_prompt"],
        "negative_vap_prompt": call_args["negative_vap_prompt"],
        "tea_cache_l1_thresh": call_args["tea_cache_l1_thresh"],
        "tea_cache_model_id": call_args["tea_cache_model_id"],
        "num_inference_steps": call_args["num_inference_steps"],
    }
    inputs_shared = {
        name: value
        for name, value in call_args.items()
        if name not in {
            "prompt",
            "negative_prompt",
            "vap_prompt",
            "negative_vap_prompt",
            "tea_cache_l1_thresh",
            "tea_cache_model_id",
            "num_inference_steps",
            "progress_bar_cmd",
            "output_type",
            "switch_DiT_boundary",
        }
    }
    inputs_shared[getattr(pipe, "eawm_kvaf_input_key", "kvaf_path")] = kvaf_path

    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)

    return inputs_shared, inputs_posi, inputs_nega


def _initialize_eawm_kvaf_latents_for_inference(
    pipe,
    inputs_shared: dict,
    *,
    sample_kvaf_from_noise: bool,
    kvaf_seed: Optional[int],
):
    kvaf_context_latents = inputs_shared.get("kvaf_context_latents")
    if not sample_kvaf_from_noise:
        if kvaf_context_latents is None:
            kvaf_input_key = getattr(pipe, "eawm_kvaf_input_key", "kvaf_path")
            raise RuntimeError(
                f"EAWM generation with sample_kvaf_from_noise=False requires `{kvaf_input_key}` or precomputed `kvaf_context_latents`."
            )
        inputs_shared["kvaf_context_latents"] = kvaf_context_latents.to(device=pipe.device, dtype=pipe.torch_dtype)
        return

    latent_reference = inputs_shared.get("latents")
    if latent_reference is None:
        latent_reference = inputs_shared.get("noise")
    if latent_reference is None:
        raise RuntimeError("Unable to initialize EAWM KVAF latents because neither `latents` nor `noise` is available.")

    if kvaf_seed is None and inputs_shared.get("seed") is not None:
        kvaf_seed = int(inputs_shared["seed"]) + 1
    rand_device = inputs_shared.get("rand_device", "cpu")
    inputs_shared["kvaf_context_latents"] = pipe.generate_noise(
        tuple(latent_reference.shape),
        seed=kvaf_seed,
        rand_device=rand_device,
        device=pipe.device,
        torch_dtype=pipe.torch_dtype,
    )


def _extract_eawm_kvaf_prediction(eawm_aux: Optional[dict]) -> torch.Tensor:
    kvaf_pred_latents = None if eawm_aux is None else eawm_aux.get("kvaf_pred_latents")
    if kvaf_pred_latents is None:
        raise RuntimeError("EAWM generation expected `kvaf_pred_latents` in eawm_aux, but none were returned.")
    return kvaf_pred_latents


def _decode_eawm_kvaf_latents(
    pipe,
    kvaf_latents: torch.Tensor,
    *,
    tiled: bool,
    tile_size,
    tile_stride,
    framewise_decoding: bool,
    output_type: str,
):
    kvaf_latents = kvaf_latents.to(device=pipe.device, dtype=pipe.torch_dtype)
    if framewise_decoding:
        kvaf_video = pipe.vae.decode_framewise(kvaf_latents, device=pipe.device)
    else:
        kvaf_video = pipe.vae.decode(
            kvaf_latents,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
    if output_type == "quantized":
        return pipe.vae_output_to_video(kvaf_video)
    if output_type == "floatpoint":
        return kvaf_video
    raise ValueError(f"Unsupported output_type: {output_type}")


@torch.no_grad()
def eawm_generate(
    pipe,
    *,
    kvaf_path=None,
    sample_kvaf_from_noise: bool = True,
    kvaf_seed: Optional[int] = None,
    return_kvaf_video: bool = True,
    return_kvaf_latents: bool = False,
    return_dict: bool = True,
    context_residual_scale: Optional[float] = None,
    eawm_context_residual_scale: Optional[float] = None,
    kvaf_cfg_scale: float = 1.0,
    use_kvaf_separate_timestep: Optional[bool] = None,
    **kwargs,
):
    if eawm_context_residual_scale is not None:
        context_residual_scale = eawm_context_residual_scale
    if context_residual_scale is not None and getattr(pipe, "dit", None) is not None:
        pipe.dit.eawm_context_residual_scale = float(context_residual_scale)

    if not sample_kvaf_from_noise and kvaf_path is None:
        raise RuntimeError("eawm_generate requires `kvaf_path` when sample_kvaf_from_noise=False.")

    call_args = _bind_wan_call_args(pipe, kwargs)
    progress_bar_cmd = call_args["progress_bar_cmd"]
    switch_DiT_boundary = call_args["switch_DiT_boundary"]
    cfg_scale = call_args["cfg_scale"]
    cfg_merge = call_args["cfg_merge"]
    tiled = call_args["tiled"]
    tile_size = call_args["tile_size"]
    tile_stride = call_args["tile_stride"]
    framewise_decoding = call_args["framewise_decoding"]
    output_type = call_args["output_type"]

    original_extra_inputs = getattr(pipe, "_eawm_extra_inputs", None)
    original_use_kvaf_separate_timestep = None
    if use_kvaf_separate_timestep is not None and getattr(pipe, "dit", None) is not None:
        original_use_kvaf_separate_timestep = getattr(pipe.dit, "eawm_use_kvaf_separate_timestep", None)
        pipe.dit.eawm_use_kvaf_separate_timestep = bool(use_kvaf_separate_timestep)
    try:
        pipe._eawm_extra_inputs = None
        inputs_shared, inputs_posi, inputs_nega = _prepare_eawm_generation_inputs(
            pipe,
            call_args,
            None if sample_kvaf_from_noise else kvaf_path,
        )
        _initialize_eawm_kvaf_latents_for_inference(
            pipe,
            inputs_shared,
            sample_kvaf_from_noise=sample_kvaf_from_noise,
            kvaf_seed=kvaf_seed,
        )

        pipe.load_models_to_device(pipe.in_iteration_models)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(pipe.scheduler.timesteps)):
            if timestep.item() < switch_DiT_boundary * 1000 and pipe.dit2 is not None and not models["dit"] is pipe.dit2:
                pipe.load_models_to_device(pipe.in_iteration_models_2)
                models["dit"] = pipe.dit2
                models["vace"] = pipe.vace2

            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred_posi, eawm_aux_posi = pipe.model_fn(
                **models,
                **inputs_shared,
                **inputs_posi,
                timestep=timestep,
                return_eawm_aux=True,
            )
            kvaf_noise_pred_posi = _extract_eawm_kvaf_prediction(eawm_aux_posi)

            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                    kvaf_noise_pred_posi, kvaf_noise_pred_nega = kvaf_noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega, eawm_aux_nega = pipe.model_fn(
                        **models,
                        **inputs_shared,
                        **inputs_nega,
                        timestep=timestep,
                        return_eawm_aux=True,
                    )
                    kvaf_noise_pred_nega = _extract_eawm_kvaf_prediction(eawm_aux_nega)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                kvaf_noise_pred = kvaf_noise_pred_nega + kvaf_cfg_scale * (kvaf_noise_pred_posi - kvaf_noise_pred_nega)
            else:
                noise_pred = noise_pred_posi
                kvaf_noise_pred = kvaf_noise_pred_posi

            inputs_shared["latents"] = pipe.scheduler.step(
                noise_pred,
                pipe.scheduler.timesteps[progress_id],
                inputs_shared["latents"],
            )
            if sample_kvaf_from_noise:
                inputs_shared["kvaf_context_latents"] = pipe.scheduler.step(
                    kvaf_noise_pred,
                    pipe.scheduler.timesteps[progress_id],
                    inputs_shared["kvaf_context_latents"],
                )
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]

        for unit in pipe.post_units:
            inputs_shared, _, _ = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)

        pipe.load_models_to_device(["vae"])
        if framewise_decoding:
            video = pipe.vae.decode_framewise(inputs_shared["latents"], device=pipe.device)
        else:
            video = pipe.vae.decode(
                inputs_shared["latents"],
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
        if output_type == "quantized":
            video = pipe.vae_output_to_video(video)
        elif output_type != "floatpoint":
            raise ValueError(f"Unsupported output_type: {output_type}")

        kvaf_video = None
        if return_kvaf_video:
            kvaf_video = _decode_eawm_kvaf_latents(
                pipe,
                inputs_shared["kvaf_context_latents"],
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
                framewise_decoding=framewise_decoding,
                output_type=output_type,
            )

        if not return_dict and not return_kvaf_video and not return_kvaf_latents:
            return video

        outputs = {"video": video}
        if return_kvaf_video:
            outputs["kvaf_video"] = kvaf_video
        if return_kvaf_latents:
            outputs["kvaf_latents"] = inputs_shared["kvaf_context_latents"]
        return outputs if return_dict or len(outputs) > 1 else video
    finally:
        if use_kvaf_separate_timestep is not None and getattr(pipe, "dit", None) is not None:
            if original_use_kvaf_separate_timestep is None:
                if hasattr(pipe.dit, "eawm_use_kvaf_separate_timestep"):
                    delattr(pipe.dit, "eawm_use_kvaf_separate_timestep")
            else:
                pipe.dit.eawm_use_kvaf_separate_timestep = original_use_kvaf_separate_timestep
        pipe._eawm_extra_inputs = original_extra_inputs
        pipe.load_models_to_device([])


def build_eawm_kvaf_flow_match_target(pipe, kvaf_context_latents: torch.Tensor, timestep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if kvaf_context_latents is None:
        raise RuntimeError("KVAF latent supervision requires kvaf_context_latents, but none were provided.")
    kvaf_context_latents = kvaf_context_latents.to(device=pipe.device, dtype=pipe.torch_dtype)
    kvaf_noise = torch.randn_like(kvaf_context_latents)
    kvaf_latents = pipe.scheduler.add_noise(kvaf_context_latents, kvaf_noise, timestep)
    kvaf_training_target = pipe.scheduler.training_target(kvaf_context_latents, kvaf_noise, timestep)
    return kvaf_latents, kvaf_training_target



def eawm_flow_match_sft_loss(pipe, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    inputs = dict(inputs)
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)

    kvaf_context_latents = inputs.get("kvaf_context_latents")
    inputs["kvaf_context_latents"], kvaf_training_target = build_eawm_kvaf_flow_match_target(
        pipe,
        kvaf_context_latents,
        timestep,
    )

    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred, eawm_aux = pipe.model_fn(
        **models,
        **inputs,
        timestep=timestep,
        return_eawm_aux=True,
    )

    if "first_frame_latents" in inputs:
        noise_pred = noise_pred[:, :, 1:]
        training_target = training_target[:, :, 1:]

    video_loss = F.mse_loss(noise_pred.float(), training_target.float())

    kvaf_pred_latents = eawm_aux.get("kvaf_pred_latents") if eawm_aux is not None else None
    if kvaf_pred_latents is None:
        raise RuntimeError("EAWM KVAF supervision is enabled but kvaf_pred_latents were not returned by the model.")
    if kvaf_pred_latents.shape != kvaf_training_target.shape:
        raise RuntimeError(
            f"KVAF latent prediction shape {tuple(kvaf_pred_latents.shape)} does not match target latent shape {tuple(kvaf_training_target.shape)}."
        )

    kvaf_loss = F.mse_loss(kvaf_pred_latents.float(), kvaf_training_target.float())
    event_target_latents = inputs.get("event_difference_latents")
    if event_target_latents is None:
        raise RuntimeError("EAWM event supervision requires event_difference_latents, but none were provided.")
    event_target_latents = event_target_latents.to(device=pipe.device, dtype=pipe.torch_dtype)

    event_pred_latents = eawm_aux.get("event_pred_latents") if eawm_aux is not None else None
    if not isinstance(event_pred_latents, list) or len(event_pred_latents) == 0:
        raise RuntimeError("EAWM event supervision is enabled but no event_pred_latents were returned by the model.")

    event_losses = []
    for event_idx, event_pred_latent in enumerate(event_pred_latents):
        if event_pred_latent.shape != event_target_latents.shape:
            raise RuntimeError(
                f"Event latent prediction {event_idx} shape {tuple(event_pred_latent.shape)} "
                f"does not match target latent shape {tuple(event_target_latents.shape)}."
            )
        event_losses.append(F.mse_loss(event_pred_latent.float(), event_target_latents.float()))
    event_loss = torch.stack(event_losses).mean()

    flow_loss = (video_loss + kvaf_loss) * pipe.scheduler.training_weight(timestep)
    event_loss_weight = float(inputs.get("event_loss_weight", 1.0))
    loss = flow_loss + event_loss_weight * event_loss
    pipe._last_eawm_loss_components = {
        "video_loss": video_loss.detach(),
        "kvaf_loss": kvaf_loss.detach(),
        "event_loss": event_loss.detach(),
        "flow_loss": flow_loss.detach(),
        "total_loss": loss.detach(),
    }
    return loss



def load_eawm_checkpoint(pipe, checkpoint_path: str, lora_alpha: float = 1.0):
    def _load_state_dict_file(path: str):
        if path.endswith(".pt"):
            return torch.load(path, map_location="cpu")
        from diffsynth.core import load_state_dict
        return load_state_dict(path, torch_dtype=torch.float32, device="cpu")

    def _is_lora_key(key: str) -> bool:
        return ".lora_A." in key or ".lora_B." in key

    def _normalize_dit_key(key: str) -> str:
        if key.startswith("pipe.dit."):
            key = key[len("pipe.dit."):]
        key = key.replace("robot_sparse_context_branch.", "eawm_kvaf_branch.")
        key = key.replace("eawm_sparse_context_branch.", "eawm_kvaf_branch.")
        key = key.replace("uv_from_video", "kvaf_from_video")
        key = key.replace("video_from_uv", "video_from_kvaf")
        key = key.replace("uv_head", "kvaf_head")
        return key

    def _extract_eawm_branch_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        branch_state = {}
        prefixes = (
            "eawm_kvaf_branch.",
            "pipe.dit.eawm_kvaf_branch.",
            "eawm_sparse_context_branch.",
            "pipe.dit.eawm_sparse_context_branch.",
            "robot_sparse_context_branch.",
            "pipe.dit.robot_sparse_context_branch.",
        )
        for key, value in state_dict.items():
            if _is_lora_key(key):
                continue
            key = _normalize_dit_key(key)
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    break
            else:
                continue
            if ".base_layer." in key:
                key = key.replace(".base_layer.", ".")
            branch_state[key] = value
        return branch_state

    def _report_eawm_branch_load_result(load_result):
        missing_keys = list(load_result.missing_keys)
        unexpected_keys = list(load_result.unexpected_keys)
        important_missing = [key for key in missing_keys if not key.startswith("blocks.")]
        if important_missing or unexpected_keys:
            print("[EAWMCheckpoint] important missing_keys", important_missing)
            print("[EAWMCheckpoint] unexpected_keys", unexpected_keys)
        else:
            expected_missing = [key for key in missing_keys if key.startswith("blocks.")]
            print(
                "[EAWMCheckpoint] KVAF branch loaded successfully. "
                f"Expected missing copied-block base params: {len(expected_missing)}"
            )

    checkpoint_path = str(checkpoint_path)
    if not checkpoint_path.endswith((".safetensors", ".pt")) and os.path.exists(f"{checkpoint_path}.safetensors"):
        checkpoint_path = f"{checkpoint_path}.safetensors"
    base_path = checkpoint_path[:-12] if checkpoint_path.endswith(".safetensors") else checkpoint_path

    lora_path = f"{base_path}.lora.safetensors"
    eawm_path = f"{base_path}.eawm.safetensors"
    legacy_context_path = f"{base_path}.context.safetensors"

    component_path = eawm_path if os.path.exists(eawm_path) else legacy_context_path
    combined_state = None
    if os.path.exists(component_path):
        state_dict = _load_state_dict_file(component_path)
        eawm_state = _extract_eawm_branch_state_dict(state_dict)
        print(f"[EAWMCheckpoint] loading KVAF-branch weights from {component_path}.")
        load_result = pipe.dit.eawm_kvaf_branch.load_state_dict(eawm_state, strict=False)
        _report_eawm_branch_load_result(load_result)
    elif os.path.exists(checkpoint_path):
        combined_state = _load_state_dict_file(checkpoint_path)
        eawm_state = _extract_eawm_branch_state_dict(combined_state)
        if len(eawm_state) > 0:
            print(f"[EAWMCheckpoint] loading KVAF-branch weights from {checkpoint_path}.")
            load_result = pipe.dit.eawm_kvaf_branch.load_state_dict(eawm_state, strict=False)
            _report_eawm_branch_load_result(load_result)

    if os.path.exists(lora_path):
        lora_state = _load_state_dict_file(lora_path)
        lora_state = {_normalize_dit_key(key): value for key, value in lora_state.items() if _is_lora_key(key)}
        pipe.load_lora(pipe.dit, state_dict=lora_state, alpha=lora_alpha)
    elif os.path.exists(checkpoint_path):
        if combined_state is None:
            combined_state = _load_state_dict_file(checkpoint_path)
        lora_state = {_normalize_dit_key(key): value for key, value in combined_state.items() if _is_lora_key(key)}
        if lora_state:
            pipe.load_lora(pipe.dit, state_dict=lora_state, alpha=lora_alpha)

    return pipe


__all__ = [
    "LoadImageFolder",
    "LoadActionCSV",
    "WanVideoUnit_EAWMKVAFEncoder",
    "WanVideoUnit_EventDifferenceEncoder",
    "EAWMKVAFBranch",
    "EAWMBidirectionalCrossAttentionBlock",
    "prepare_eawm_kvaf_branch",
    "inject_eawm",
    "load_eawm_checkpoint",
    "eawm_generate",
    "eawm_flow_match_sft_loss",
    "eawm_model_fn_wan_video",
    "EAWMCheckpointLogger",
]
