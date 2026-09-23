#!/usr/bin/env bash
set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT}"
: "${TASK_JSON:?Set TASK_JSON}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$($PYTHON_BIN -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
REFERENCE_KVAF_ARGS=()
if [[ "${USE_REFERENCE_KVAF:-0}" == "1" ]]; then
  REFERENCE_KVAF_ARGS+=(--use_reference_kvaf)
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$REPO_ROOT/models}"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-huggingface}"
export TOKENIZERS_PARALLELISM=false

cd "$REPO_ROOT"

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  inference/generate.py \
  --checkpoint "$CHECKPOINT" \
  --task_json "$TASK_JSON" \
  --output_root "${OUTPUT_ROOT:-outputs/inference}" \
  --height 480 \
  --width 640 \
  --num_frames 81 \
  --num_inference_steps 50 \
  --context_interval 5 \
  --cfg_scale 5.0 \
  --kvaf_cfg_scale 5.0 \
  --lora_alpha 1.0 \
  --seed_offset 1 \
  --fps 15 \
  --quality 10 \
  --tiled \
  "${REFERENCE_KVAF_ARGS[@]}"
