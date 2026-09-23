#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_ROOT:?Set DATASET_ROOT}"
: "${METADATA_PATH:?Set METADATA_PATH}"
: "${STAGE1_CHECKPOINT:?Set STAGE1_CHECKPOINT}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$REPO_ROOT/models}"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-huggingface}"
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

cd "$REPO_ROOT"

"$PYTHON_BIN" -m torch.distributed.run \
  --nnodes="$NNODES" \
  --nproc_per_node="$NPROC_PER_NODE" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  training/train_eawm.py \
  --dataset_base_path "$DATASET_ROOT" \
  --dataset_metadata_path "$METADATA_PATH" \
  --data_file_keys "video,action_path,kvaf_path" \
  --height 480 \
  --width 640 \
  --num_frames 81 \
  --dataset_repeat 10 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-4 \
  --lora_learning_rate 8e-5 \
  --context_learning_rate 8e-5 \
  --context_block_learning_rate 8e-5 \
  --num_epochs 8 \
  --stage1_epochs 0 \
  --max_grad_norm 1.0 \
  --lr_scheduler_type cosine \
  --lr_warmup_steps 100 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$OUTPUT_DIR" \
  --resume_dit_checkpoint "$STAGE1_CHECKPOINT" \
  --lora_base_model dit \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --context_interval 5 \
  --extra_inputs "input_image,kvaf_path" \
  --event_loss_weight 0.1 \
  --use_gradient_checkpointing
