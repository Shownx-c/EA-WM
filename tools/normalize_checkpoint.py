#!/usr/bin/env python3
"""Normalize EA-WM checkpoint names for release and verify the result."""

from __future__ import annotations

import argparse
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


KEY_REPLACEMENTS = (
    ("robot_sparse_context_branch.", "eawm_kvaf_branch."),
    ("eawm_sparse_context_branch.", "eawm_kvaf_branch."),
    ("uv_from_video", "kvaf_from_video"),
    ("video_from_uv", "video_from_kvaf"),
    ("uv_head", "kvaf_head"),
)


def normalize_key(key: str) -> str:
    if key.startswith("pipe.dit."):
        key = key[len("pipe.dit."):]
    for source, target in KEY_REPLACEMENTS:
        key = key.replace(source, target)
    return key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize an EA-WM safetensors checkpoint.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_dict = load_file(str(args.input), device="cpu")
    normalized = {}
    for key, tensor in state_dict.items():
        normalized_key = normalize_key(key)
        if normalized_key in normalized:
            raise RuntimeError(f"Key collision after normalization: {normalized_key}")
        normalized[normalized_key] = tensor

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "pt",
        "architecture": "EA-WM",
        "base_model": "Wan-AI/Wan2.2-TI2V-5B",
        "stage": args.stage,
        "lora_rank": "32",
        "context_interval": "5",
    }
    save_file(normalized, str(args.output), metadata=metadata)

    with safe_open(str(args.output), framework="pt", device="cpu") as handle:
        output_keys = list(handle.keys())
        output_metadata = handle.metadata()
    if len(output_keys) != len(state_dict):
        raise RuntimeError(f"Tensor count changed: {len(state_dict)} -> {len(output_keys)}")
    forbidden = ("robot_sparse_context_branch", "eawm_sparse_context_branch", "uv_from_video", "video_from_uv", "uv_head")
    invalid_keys = [key for key in output_keys if any(token in key for token in forbidden)]
    if invalid_keys:
        raise RuntimeError(f"Unnormalized checkpoint keys remain: {invalid_keys[:5]}")

    print(f"wrote {len(output_keys)} tensors to {args.output}")
    print(f"metadata: {output_metadata}")


if __name__ == "__main__":
    main()
