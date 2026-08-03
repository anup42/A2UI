#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-4B-Instruct}"
DATASET_NAME="${DATASET_NAME:-Anthropic/hh-rlhf}"

python3 src/train.py \
  --output_dir ./runs/qwen25_3b_hh_dpo_v2 \
  --model_name "$MODEL_NAME" \
  --dataset_name "$DATASET_NAME" \
  --train_samples 160000 \
  --eval_samples 200 \
  --epochs 3 \
  --max_length 1024 \
  --max_prompt_length 768 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --logging_steps 100 \
  --eval_steps 500
