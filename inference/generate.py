import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

import torch
from PIL import Image
from tqdm import tqdm

from diffsynth.extensions.eawm import (
    inject_eawm,
    load_eawm_checkpoint,
    prepare_eawm_kvaf_branch,
)
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import VideoData, save_video


DEFAULT_MODEL_SPECS = [
    "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors",
    "Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth",
    "Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth",
]

DEFAULT_NEGATIVE_PROMPT = (
    "overexposed, blurry, low quality, static, distorted hands, distorted object, subtitle, watermark"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run EA-WM batch inference with Wan2.2-TI2V-5B."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint stem or .safetensors path.")
    parser.add_argument("--task_json", type=str, required=True, help="JSON file containing inference tasks.")
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/inference/default_run",
        help="Directory for generated videos.",
    )
    parser.add_argument(
        "--model_spec",
        action="append",
        default=None,
        help="Model spec in the form model_id:origin_pattern. Repeat to override the default Wan2.2 triplet.",
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--context_interval", type=int, default=5)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--kvaf_cfg_scale", type=float, default=5.0)
    parser.add_argument("--context_residual_scale", type=float, default=1.0)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--seed_offset", type=int, default=1)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=10)
    parser.add_argument("--kvaf_key", type=str, default="kvaf_path", help="Task-json key that points to the KVAF frame folder.")
    parser.add_argument("--rand_device", type=str, default="cpu")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument(
        "--use_reference_kvaf",
        action="store_true",
        help="Use the KVAF frames from each task entry instead of sampling KVAF latents from noise.",
    )
    parser.add_argument(
        "--framewise_decoding",
        action="store_true",
        help="Decode the video and KVAF branch frame-by-frame with the VAE helper.",
    )
    parser.add_argument(
        "--tiled",
        dest="tiled",
        action="store_true",
        help="Enable VAE tiled encode/decode.",
    )
    parser.add_argument(
        "--no_tiled",
        dest="tiled",
        action="store_false",
        help="Disable VAE tiled encode/decode.",
    )
    parser.set_defaults(tiled=True)
    return parser.parse_args()


def parse_model_configs(model_specs: List[str]) -> List[ModelConfig]:
    model_configs = []
    for spec in model_specs:
        if ":" not in spec:
            raise ValueError(f"Invalid --model_spec value: {spec}")
        model_id, origin_file_pattern = spec.split(":", 1)
        model_configs.append(ModelConfig(model_id=model_id, origin_file_pattern=origin_file_pattern))
    return model_configs


def load_tasks(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("items", [])
    if not isinstance(payload, list):
        raise TypeError("Task JSON must be a list or a dict with an `items` list.")
    return payload


def resolve_input_image(item: dict, height: int, width: int):
    image_path = item.get("input_image") or item.get("image_path")
    if image_path:
        return Image.open(image_path).convert("RGB")

    video_path = item.get("video_path") or item.get("gt_path")
    if video_path:
        return VideoData(video_path, height=height, width=width)[0]

    raise ValueError("Each task entry must provide either `input_image`/`image_path` or `video_path`/`gt_path`.")


def output_name_for_item(item: dict, index: int) -> str:
    gen_name = item.get("gen_name") or f"sample_{index:04d}.mp4"
    if Path(gen_name).suffix == "":
        gen_name = f"{gen_name}.mp4"
    return gen_name


def identity_progress_bar(values: Iterable):
    return values


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        torch_dtype = torch.bfloat16
    else:
        device = "cpu"
        torch_dtype = torch.float32

    model_specs = args.model_spec or DEFAULT_MODEL_SPECS
    pipe = WanVideoPipeline.from_pretrained(
        redirect_common_files=False,
        torch_dtype=torch_dtype,
        device=device,
        model_configs=parse_model_configs(model_specs),
    )

    prepared_kvaf_branch = prepare_eawm_kvaf_branch(pipe, context_interval=args.context_interval)
    inject_eawm(
        pipe,
        kvaf_input_key=args.kvaf_key,
        context_interval=args.context_interval,
        prebuilt_kvaf_branch=prepared_kvaf_branch,
        context_residual_scale=args.context_residual_scale,
    )
    load_eawm_checkpoint(pipe, args.checkpoint, lora_alpha=args.lora_alpha)

    tasks = load_tasks(args.task_json)
    output_root = Path(args.output_root)
    video_dir = output_root / "videos"
    kvaf_video_dir = output_root / "kvaf_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    kvaf_video_dir.mkdir(parents=True, exist_ok=True)

    progress_bar = tqdm if rank == 0 else identity_progress_bar
    print(f"[rank{rank}] checkpoint={args.checkpoint}")
    print(f"[rank{rank}] tasks={len(tasks)} world_size={world_size}")

    for task_index, item in enumerate(tasks):
        if task_index % world_size != rank:
            continue

        input_image = resolve_input_image(item, height=args.height, width=args.width)
        kvaf_path = item.get(args.kvaf_key) or item.get("kvaf_path")
        if args.use_reference_kvaf and kvaf_path is None:
            raise ValueError(f"Task {task_index} is missing `{args.kvaf_key}` but --use_reference_kvaf was enabled.")

        gen_name = output_name_for_item(item, task_index)
        video_path = video_dir / gen_name
        kvaf_video_path = kvaf_video_dir / f"{Path(gen_name).stem}_kvaf{Path(gen_name).suffix}"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        kvaf_video_path.parent.mkdir(parents=True, exist_ok=True)

        if video_path.exists() and kvaf_video_path.exists():
            print(f"[rank{rank}] skip existing outputs for {gen_name}")
            continue

        outputs = pipe.eawm_generate(
            prompt=item["prompt"],
            negative_prompt=args.negative_prompt,
            input_image=input_image,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            sample_kvaf_from_noise=not args.use_reference_kvaf,
            kvaf_path=kvaf_path,
            return_kvaf_video=True,
            return_dict=True,
            context_residual_scale=args.context_residual_scale,
            cfg_scale=args.cfg_scale,
            kvaf_cfg_scale=args.kvaf_cfg_scale,
            seed=args.seed_offset + task_index,
            tiled=args.tiled,
            framewise_decoding=args.framewise_decoding,
            rand_device=args.rand_device,
            progress_bar_cmd=progress_bar,
        )

        save_video(outputs["video"], str(video_path), fps=args.fps, quality=args.quality)
        save_video(outputs["kvaf_video"], str(kvaf_video_path), fps=args.fps, quality=args.quality)
        print(f"[rank{rank}] wrote {video_path}")
        print(f"[rank{rank}] wrote {kvaf_video_path}")

    print(f"[rank{rank}] inference complete")


if __name__ == "__main__":
    main()
