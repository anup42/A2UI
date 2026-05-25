#!/usr/bin/env bash
#SBATCH --job-name=a2ui-gemma4-e2b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=training/logs/slurm-%j.out
#SBATCH --error=training/logs/slurm-%j.err

set -euo pipefail

# Override these at submit time if the training machine uses different paths:
#   sbatch --export=ALL,A2UI_REPO_DIR=/path/to/A2UI,A2UI_VENV=/path/to/venv training/scripts/slurm_train_gemma4_e2b.sbatch
REPO_DIR="${A2UI_REPO_DIR:-/home/k_anup/code/GenUI}"
VENV_DIR="${A2UI_VENV:-/home/k_anup/gemma4_env}"
TRAIN_CONFIG="${A2UI_TRAIN_CONFIG:-training/configs/models/gemma4_e2b_ir_lora.yaml}"

cd "$REPO_DIR"
mkdir -p training/logs training/outputs training/runs

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
else
  echo "ERROR: virtualenv not found at $VENV_DIR" >&2
  echo "Set A2UI_VENV=/path/to/venv or create the expected environment." >&2
  exit 2
fi

# Gemma4 SFT is intentionally single-GPU here. Multiple visible GPUs cause
# PyTorch Trainer to DataParallel-wrap the model and can trigger CUDA failures
# on replica devices even when model.device_map is pinned.
export A2UI_CUDA_VISIBLE_DEVICES="${A2UI_CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="$A2UI_CUDA_VISIBLE_DEVICES"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

python - <<'PY'
import os
print("A2UI repo:", os.getcwd())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("HF_HOME:", os.environ.get("HF_HOME"))
PY

nvidia-smi || true

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available")
print("visible cuda device count:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(f"visible cuda device {index}: {torch.cuda.get_device_name(index)}")
PY

exec srun --ntasks=1 --kill-on-bad-exit=1 \
  python training/scripts/train_sft.py --config "$TRAIN_CONFIG"
