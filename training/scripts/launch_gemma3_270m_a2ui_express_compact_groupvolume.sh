#!/usr/bin/env bash
set -euo pipefail

TASK=/group-volume/k.anup/working_dir/gemma3_270m_a2ui_express_compact_restart_20260831
REPO=/group-volume/k.anup/working_dir/fulltrain-mobile-20260809/repo
ENV_ROOT=/group-volume/k.anup/working_dir/genui-space-h100-20260807/env

export PYTHONPATH="$REPO/src"
export A2UI_TASK_ROOT="$TASK"
export HF_HOME="$TASK/cache/huggingface"
export TRANSFORMERS_CACHE="$TASK/cache/huggingface"
export TORCH_HOME="$TASK/cache/torch"
export TORCH_EXTENSIONS_DIR="$TASK/cache/torch_extensions"
export XDG_CACHE_HOME="$TASK/cache"
export TMPDIR="$TASK/tmp"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1

mkdir -p "$TASK/logs" "$TASK/tmp" "$TASK/cache/huggingface" "$TASK/cache/torch" "$TASK/cache/torch_extensions"
cd "$REPO"
echo "launcher_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "node=$(hostname)"
echo "task=$TASK"
echo "python=$ENV_ROOT/bin/python"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
exec "$ENV_ROOT/bin/python" "$REPO/scripts/train_sft.py" --config "$TASK/training_config.yaml"
