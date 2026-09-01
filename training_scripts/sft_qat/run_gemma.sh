#!/bin/bash
# ============================================================
# SFT Training Run Scripts (using Accelerate)
# ============================================================
# Usage:
#   Single GPU:   bash run.sh
#   Multi-GPU:    bash run.sh multigpu [NUM_GPUS]
#   DeepSpeed:    bash run.sh deepspeed_zero2 [NUM_GPUS]
#                 bash run.sh deepspeed_zero3 [NUM_GPUS]
#   Eval:         bash run.sh eval [NUM_GPUS] [NUM_PREDICTIONS]
#                 (NUM_PREDICTIONS defaults to 100)
# ============================================================

set -e

# ---- NCCL Timeout Configuration ----
# Increase timeout for long evaluation operations (default is 30 min = 1800000 ms)
# Set to 60 minutes = 3600000 milliseconds
export NCCL_TIMEOUT=7200000
export NCCL_BLOCKING_WAIT=1
echo "NCCL timeout set to 120 minutes (7200000 ms)"

# ---- Auto-detect CUDA_HOME from PyTorch ----
if [ -z "${CUDA_HOME}" ]; then
    DETECTED_CUDA=$(python3 -c "import torch; import os; cuda_home = getattr(torch.utils.cpp_extension, 'CUDA_HOME', None); print(cuda_home if cuda_home else '')" 2>/dev/null || echo "")
    if [ -n "${DETECTED_CUDA}" ] && [ -d "${DETECTED_CUDA}" ]; then
        export CUDA_HOME="${DETECTED_CUDA}"
        echo "Auto-detected CUDA_HOME=${CUDA_HOME}"
    fi
fi

# ---- Configuration ----
# Get the directory where this script is located (GenUI-LM/training_scripts/sft)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the GenUI-LM root directory (parent of training_scripts)
GENUI_LM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Paths relative to GenUI-LM root
MODEL_NAME="/home/adarsh_ag/Storage_gpu/PlatformIntelligenceTeam/adarsh_ag/models/gemma-3-270m"
DATA_PATH="${SCRIPT_DIR}/gen_data_v3/genui_combined_20260706_195244_masked_clean.jsonl"
TEST_DATA_PATH="/home/adarsh_ag/GenUI-LM/training_scripts/sft/test_data/golden100_url_mapping.jsonl"
OUTPUT_DIR="${SCRIPT_DIR}/runs/sft_run_gemma3_270m_15_07"
NUM_GPUS=${2:-$(nvidia-smi -L | wc -l)}
NUM_PREDICTIONS=${3:-50}  # Number of predictions to save during eval (default: 50 = all test samples)

# ---- LoRA Configuration ----
LORA_R=16  # LoRA rank
LORA_ALPHA=$((LORA_R * 2))  # LoRA alpha automatically set to 2*rank

# ---- Common training args ----
COMMON_ARGS="\
  --output_dir ${OUTPUT_DIR} \
  --model_name ${MODEL_NAME} \
  --data_path ${DATA_PATH} \
  --test_data_path ${TEST_DATA_PATH} \
  --lora_layers all \
  --epochs 4 \
  --lr 5e-5 \
  --batch_size 1 \
  --eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_length 8192 \
  --max_new_tokens 8192 \
  --logging_steps 50 \
  --eval_steps 100 \
  --eval_samples 100 \
  --save_steps 500 \
  --use_lora \
  --lora_r ${LORA_R} \
  --lora_alpha ${LORA_ALPHA} \
  --report_to tensorboard \
"

MODE=${1:-single}

echo "============================================"
echo "SFT Training - Mode: ${MODE}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"

case ${MODE} in
  single)
    echo "Running single-GPU training..."
    accelerate launch \
      --num_processes 1 \
      --mixed_precision bf16 \
      train.py ${COMMON_ARGS}
    ;;

  multigpu)
    echo "Running multi-GPU training on ${NUM_GPUS} GPUs..."
    accelerate launch \
      --num_processes ${NUM_GPUS} \
      --mixed_precision bf16 \
      train.py ${COMMON_ARGS}
    ;;

  deepspeed_zero2)
    echo "Running DeepSpeed ZeRO-2 on ${NUM_GPUS} GPUs..."
    if [ -z "${CUDA_HOME}" ]; then
      echo "ERROR: CUDA_HOME not set and could not be auto-detected."
      echo "DeepSpeed requires CUDA_HOME. Set it with: export CUDA_HOME=/path/to/cuda"
      echo "Falling back to multi-GPU without DeepSpeed..."
      accelerate launch \
        --num_processes ${NUM_GPUS} \
        --mixed_precision bf16 \
        train.py ${COMMON_ARGS}
    else
      accelerate launch \
        --use_deepspeed \
        --deepspeed_config_file ds_config_zero2.json \
        --num_processes ${NUM_GPUS} \
        train.py ${COMMON_ARGS}
    fi
    ;;

  deepspeed_zero3)
    echo "Running DeepSpeed ZeRO-3 on ${NUM_GPUS} GPUs..."
    if [ -z "${CUDA_HOME}" ]; then
      echo "ERROR: CUDA_HOME not set and could not be auto-detected."
      echo "DeepSpeed requires CUDA_HOME. Set it with: export CUDA_HOME=/path/to/cuda"
      echo "Falling back to multi-GPU without DeepSpeed..."
      accelerate launch \
        --num_processes ${NUM_GPUS} \
        --mixed_precision bf16 \
        train.py ${COMMON_ARGS}
    else
      accelerate launch \
        --use_deepspeed \
        --deepspeed_config_file ds_config_zero3.json \
        --num_processes ${NUM_GPUS} \
        train.py ${COMMON_ARGS}
    fi
    ;;

  eval)
    echo "Running evaluation on test data (single GPU)..."
    python3 eval.py \
      --model_name ${MODEL_NAME} \
      --adapter_dir ${OUTPUT_DIR}/best_test_checkpoint \
      --data_path ${TEST_DATA_PATH} \
      --output_dir ${OUTPUT_DIR}/eval \
      --input_field response_text \
      --output_field genui_json \
      --max_length 4096 \
      --max_new_tokens 4096 \
      --batch_size 4 \
      --num_samples ${NUM_PREDICTIONS} \
      --log_to_tensorboard
    ;;

  multigpu_eval)
    echo "Running evaluation on test data (multi-GPU on ${NUM_GPUS} GPUs)..."
    accelerate launch \
      --num_processes ${NUM_GPUS} \
      eval.py \
      --model_name ${MODEL_NAME} \
      --adapter_dir ${OUTPUT_DIR}/best_test_checkpoint \
      --data_path ${TEST_DATA_PATH} \
      --output_dir ${OUTPUT_DIR}/eval \
      --input_field response_text \
      --output_field genui_json \
      --max_length 4096 \
      --max_new_tokens 4096 \
      --batch_size 4 \
      --num_samples ${NUM_PREDICTIONS} \
      --log_to_tensorboard
    ;;

  *)
    echo "Unknown mode: ${MODE}"
    echo ""
    echo "Usage: bash run.sh [MODE] [NUM_GPUS] [NUM_PREDICTIONS]"
    echo ""
    echo "Available modes:"
    echo "  single           - Single GPU training (default)"
    echo "  multigpu         - Multi-GPU with Accelerate DDP"
    echo "  deepspeed_zero2  - Multi-GPU with DeepSpeed ZeRO-2 (needs CUDA_HOME)"
    echo "  deepspeed_zero3  - Multi-GPU with DeepSpeed ZeRO-3 (needs CUDA_HOME)"
    echo "  eval             - Run evaluation on trained model (single GPU)"
    echo "  multigpu_eval    - Run evaluation on trained model (multi-GPU)"
    echo ""
    echo "Arguments:"
    echo "  NUM_GPUS         - Number of GPUs (default: auto-detect)"
    echo "  NUM_PREDICTIONS  - Number of predictions to save during eval (default: 100)"
    exit 1
    ;;
esac

echo "============================================"
echo "Done! Mode: ${MODE}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Monitor with:"
echo "  TensorBoard:  tensorboard --logdir ${OUTPUT_DIR}"
echo "============================================"