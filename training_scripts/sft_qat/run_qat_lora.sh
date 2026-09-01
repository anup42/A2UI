#!/bin/bash
set -e

# QAT + LoRA training launcher
# Usage:
#   bash run_qat_lora.sh [MODE] [NUM_GPUS] [MODEL_PATH] [DATA_PATH] [OUTPUT_DIR] [TEST_DATA_PATH]
#
# Examples:
#   # Single GPU, LoRA+QAT on a local model
#   bash run_qat_lora.sh single 1 ./models/gemma-2b data.jsonl ./output test.jsonl
#
#   # Multi-GPU with 4 GPUs
#   bash run_qat_lora.sh multigpu 4 ./models/gemma-2b data.jsonl ./output test.jsonl
#
#   # Multi-GPU with DeepSpeed ZeRO-2
#   bash run_qat_lora.sh deepspeed_zero2 4 ./models/gemma-2b data.jsonl ./output test.jsonl

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-single}"
NUM_GPUS="${2:-1}"
MODEL_PATH="/home/adarsh_ag/models/gemma-4-E2B-it"
DATA_PATH="/home/adarsh_ag/GenUI-LM/training_scripts/sft/gen_data_v2/genui_processed_clean_merged.jsonl"
OUTPUT_DIR="${SCRIPT_DIR}/runs/sft_run_gemma4_E2B_qat_torchao"
TEST_DATA_PATH="/home/adarsh_ag/GenUI-LM/training_scripts/sft/test_data/golden100_url_mapping.jsonl"

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

mkdir -p "$OUTPUT_DIR"

# Set up environment for accelerate/DeepSpeed
export NCCL_TIMEOUT=7200000  # 120 min (for long heuristic evals)
export NCCL_BLOCKING_WAIT=1

# Auto-detect CUDA_HOME for DeepSpeed compatibility
if [ -z "$CUDA_HOME" ]; then
    CUDA_HOME_DETECT=$(python3 -c "import torch; print(torch.utils.cpp_extension.CUDA_HOME or '')" 2>/dev/null || echo "")
    if [ -n "$CUDA_HOME_DETECT" ] && [ -d "$CUDA_HOME_DETECT" ]; then
        export CUDA_HOME="$CUDA_HOME_DETECT"
    fi
fi

# Common training arguments
COMMON_ARGS="
    --model_name $MODEL_PATH
    --data_path $DATA_PATH
    --output_dir $OUTPUT_DIR
    --epochs 4
    --lr 5e-5
    --batch_size 1
    --eval_batch_size 1
    --gradient_accumulation_steps 16
    --max_length 6144
    --max_new_tokens 8192
    --logging_steps 50
    --eval_steps 100
    --eval_samples 100
    --save_steps 500
    --lora_r 16
    --lora_alpha 32
    --lora_layers all
    --qat_scheme int4
    --qat_group_size 32
    --qat_warmup_steps 300
    --qat_diag_interval 50
    --report_to tensorboard
"

if [ -n "$TEST_DATA_PATH" ]; then
    COMMON_ARGS="$COMMON_ARGS --test_data_path $TEST_DATA_PATH"
fi

echo "=========================================="
echo "QAT + LoRA Training (train_qat_lora.py)"
echo "=========================================="
echo "Mode:       $MODE"
echo "Num GPUs:   $NUM_GPUS"
echo "Model:      $MODEL_PATH"
echo "Data:       $DATA_PATH"
echo "Output:     $OUTPUT_DIR"
if [ -n "$TEST_DATA_PATH" ]; then
    echo "Test Data:  $TEST_DATA_PATH"
fi
echo "=========================================="

case "$MODE" in
    single)
        echo "Running on single GPU..."
        accelerate launch --num_processes 1 --mixed_precision bf16 \
            "$SCRIPT_DIR/train_qat_lora.py" $COMMON_ARGS
        ;;
    multigpu)
        echo "Running on $NUM_GPUS GPUs with DDP..."
        accelerate launch --num_processes "$NUM_GPUS" --mixed_precision bf16 \
            "$SCRIPT_DIR/train_qat_lora.py" $COMMON_ARGS
        ;;
    deepspeed_zero2)
        echo "Running on $NUM_GPUS GPUs with DeepSpeed ZeRO-2..."
        if [ ! -f "$SCRIPT_DIR/ds_config_zero2.json" ]; then
            echo "ERROR: ds_config_zero2.json not found in $SCRIPT_DIR"
            exit 1
        fi
        accelerate launch --use_deepspeed --deepspeed_config_file "$SCRIPT_DIR/ds_config_zero2.json" \
            --num_processes "$NUM_GPUS" \
            "$SCRIPT_DIR/train_qat_lora.py" $COMMON_ARGS
        ;;
    deepspeed_zero3)
        echo "Running on $NUM_GPUS GPUs with DeepSpeed ZeRO-3..."
        if [ ! -f "$SCRIPT_DIR/ds_config_zero3.json" ]; then
            echo "ERROR: ds_config_zero3.json not found in $SCRIPT_DIR"
            exit 1
        fi
        accelerate launch --use_deepspeed --deepspeed_config_file "$SCRIPT_DIR/ds_config_zero3.json" \
            --num_processes "$NUM_GPUS" \
            "$SCRIPT_DIR/train_qat_lora.py" $COMMON_ARGS
        ;;
    *)
        echo "ERROR: Unknown mode '$MODE'. Use: single, multigpu, deepspeed_zero2, deepspeed_zero3"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Training completed!"
echo "Output: $OUTPUT_DIR"
echo "TensorBoard: tensorboard --logdir $OUTPUT_DIR"
echo "=========================================="
