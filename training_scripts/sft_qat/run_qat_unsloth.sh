#!/bin/bash
set -e

# QAT training launcher (Unsloth native quantizer). Single GPU only --
# Unsloth does not officially support multi-GPU/DDP.
#
# Usage:
#   bash run_qat_unsloth.sh [GPU_ID]
#   RESUME=auto bash run_qat_unsloth.sh [GPU_ID]        # resume from the latest checkpoint
#   RESUME=/path/to/checkpoint-1200 bash run_qat_unsloth.sh [GPU_ID]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="${1:-0}"
RESUME_FROM_CHECKPOINT="${RESUME:-}"
MODEL_PATH="/home/adarsh_ag/models/gemma-3-270m"
DATA_PATH="/home/adarsh_ag/GenUI-LM/training_scripts/sft/gen_data_v2/genui_processed_clean_merged.jsonl"
OUTPUT_DIR="${SCRIPT_DIR}/runs/sft_run_gemma3_270m_qat_unsloth"
TEST_DATA_PATH="/home/adarsh_ag/GenUI-LM/training_scripts/sft/test_data/golden100_url_mapping.jsonl"
HEURISTIC_UTIL_FOLDER="/home/adarsh_ag/GenUI-LM/dataset/data/runs/golden100_gt"

# Validate paths exist
if [ ! -e "$MODEL_PATH" ]; then
    echo "ERROR: Model path does not exist: $MODEL_PATH"
    exit 1
fi
if [ ! -e "$DATA_PATH" ]; then
    echo "ERROR: Data path does not exist: $DATA_PATH"
    exit 1
fi
if [ -n "$TEST_DATA_PATH" ] && [ ! -e "$TEST_DATA_PATH" ]; then
    echo "ERROR: Test data path does not exist: $TEST_DATA_PATH"
    exit 1
fi

# --test_samples caps how many rows of TEST_DATA_PATH get predictions generated
# every --eval_steps. This is SEPARATE from --eval_samples (validation split
# size carved from --data_path) -- without it (default -1), EVERY row in
# TEST_DATA_PATH gets a prediction every eval_steps, which at max_new_tokens
# 6144 is very slow.

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "QAT Training (train_qat_unsloth.py)"
echo "=========================================="
echo "GPU:        $GPU_ID"
echo "Model:      $MODEL_PATH"
echo "Data:       $DATA_PATH"
echo "Output:     $OUTPUT_DIR"
if [ -n "$TEST_DATA_PATH" ]; then
    echo "Test Data:  $TEST_DATA_PATH"
fi
echo "=========================================="

# Count devices with the CUDA runtime rather than NVML. This container presets
# CUDA_VISIBLE_DEVICES=0,1,2,3 and nvidia-smi lists 4 GPUs, but the runtime only
# initializes 3 -- NVML-based counting then walks off the end and trips
# "device >= 0 && device < num_gpus INTERNAL ASSERT FAILED".
export PYTORCH_NVML_BASED_CUDA_CHECK=0

# Pin to a single GPU. The explicit GPU_ID argument must win over the container's
# preset CUDA_VISIBLE_DEVICES, otherwise the argument is silently ignored.
# Only a real distributed launcher (LOCAL_RANK) owns device assignment instead.
if [ -n "$LOCAL_RANK" ]; then
    echo "Distributed launcher detected (LOCAL_RANK=$LOCAL_RANK); leaving GPU assignment alone."
else
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    echo "Set CUDA_VISIBLE_DEVICES=$GPU_ID"
fi

python "$SCRIPT_DIR/train_qat_unsloth.py" \
    --model_name "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --test_data_path "$TEST_DATA_PATH" \
    --heuristic_util_folder "$HEURISTIC_UTIL_FOLDER" \
    --epochs 4 \
    --lr 5e-5 \
    --batch_size 1 \
    --eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_length 6144 \
    --max_new_tokens 6144 \
    --logging_steps 50 \
    --eval_steps 200 \
    --eval_samples 100 \
    --test_samples 20 \
    --save_steps 600 \
    --lora_r 16 \
    --lora_alpha 32 \
    --qat_scheme int8-int4 \
    --save_predictions_interval 1 \
    --save_predictions_limit 0 \
    --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT" \
    --report_to tensorboard

echo ""
echo "=========================================="
echo "Training completed!"
echo "Output: $OUTPUT_DIR"
echo "TensorBoard: tensorboard --logdir $OUTPUT_DIR"
echo "=========================================="
