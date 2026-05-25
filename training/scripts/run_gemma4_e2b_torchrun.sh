#!/usr/bin/env sh
set -eu

# Run Gemma4 E2B IR LoRA training with one process per GPU.
# Override any value at launch time, for example:
#   A2UI_REPO_DIR=/home/k_anup/code/GenUI A2UI_GPU_IDS=0,1,2,3 training/scripts/run_gemma4_e2b_torchrun.sh

REPO_DIR="${A2UI_REPO_DIR:-/home/k_anup/code/GenUI}"
VENV_DIR="${A2UI_VENV:-/home/k_anup/gemma4_env}"
TRAIN_CONFIG="${A2UI_TRAIN_CONFIG:-training/configs/models/gemma4_e2b_ir_lora.yaml}"
GPU_IDS="${A2UI_GPU_IDS:-0,1,2,3}"
NUM_GPUS="${A2UI_NUM_GPUS:-}"

if [ -z "$NUM_GPUS" ]; then
  # Count comma-separated GPU ids without depending on Bash arrays or Python.
  NUM_GPUS="$(printf '%s' "$GPU_IDS" | awk -F',' '{ print NF }')"
fi

cd "$REPO_DIR"

if [ -f "$VENV_DIR/bin/activate" ]; then
  # shellcheck disable=SC1090
  . "$VENV_DIR/bin/activate"
else
  echo "ERROR: virtualenv not found at $VENV_DIR" >&2
  echo "Set A2UI_VENV=/path/to/venv or create the expected environment." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

echo "A2UI repo: $PWD"
echo "Venv: $VENV_DIR"
echo "Config: $TRAIN_CONFIG"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "nproc_per_node: $NUM_GPUS"

nvidia-smi || true

exec torchrun --standalone --nproc_per_node="$NUM_GPUS" \
  training/scripts/train_sft.py \
  --config "$TRAIN_CONFIG"
