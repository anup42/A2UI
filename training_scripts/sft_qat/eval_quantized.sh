#!/bin/bash
# ============================================================
# Eval Quantized Model (TorchAO checkpoint)
# ============================================================
# Evaluates a TorchAO quantized checkpoint (.pt) on test data.
#
# Usage:
#   Single GPU:   bash eval_merged_quantized.sh
#   Multi-GPU:    bash eval_merged_quantized.sh multigpu [NUM_GPUS]
# ============================================================

set -e

# ---- NCCL Timeout Configuration ----
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
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Model paths ----
# Base model (for architecture + tokenizer)
MODEL_NAME="/home/h_sagri/Storage_gpu/PlatformIntelligenceTeam/h_sagri/models/gemma-3-270m-it"

# TorchAO quantized checkpoint (.pt file)
# Can be from convert_to_torchao.py (quantized.pt) or merge_and_quantize.py (merged_quantized.pt)
CHECKPOINT="${SCRIPT_DIR}/runs/sft_run_gemma3_270m_qat_unsloth/checkpoint-600/merged-int4/merged_quantized.pt"

# ---- Data paths ----
TEST_DATA_PATH="/home/h_sagri/GenUI-LM/training_scripts/sft/test_data/golden100_url_mapping.jsonl"

# ---- Output ----
OUTPUT_DIR="${SCRIPT_DIR}/runs/sft_run_gemma3_270m_qat_unsloth/checkpoint-600/merged-int4/eval_merged_quantized"

# ---- Generation config ----
MAX_LENGTH=6144
MAX_NEW_TOKENS=4096
BATCH_SIZE=2
NUM_SAMPLES=10  # -1 for all

# ---- Heuristic pipeline config ----
RUN_HEURISTIC=false
HEURISTIC_UTIL_FOLDER="/home/h_sagri/GenUI-LM/dataset/data/runs/golden100_v3"
HEURISTIC_CONFIG_PATH="/home/h_sagri/GenUI-LM/dataset/configs/run.yaml"
HEURISTIC_SCRIPT_DIR=""

# ---- Multi-GPU config ----
NUM_GPUS=${2:-$(nvidia-smi -L | wc -l)}

MODE=${1:-single}

echo "============================================"
echo "Eval Quantized - Mode: ${MODE}"
echo "Base model:  ${MODEL_NAME}"
echo "Checkpoint:  ${CHECKPOINT}"
echo "Test data:   ${TEST_DATA_PATH}"
echo "Output:      ${OUTPUT_DIR}"
echo "============================================"

# ---- Build common args ----
COMMON_ARGS="\
  --model_name ${MODEL_NAME} \
  --checkpoint ${CHECKPOINT} \
  --data_path ${TEST_DATA_PATH} \
  --output_dir ${OUTPUT_DIR} \
  --input_field response_text \
  --output_field genui_json \
  --max_length ${MAX_LENGTH} \
  --max_new_tokens ${MAX_NEW_TOKENS} \
  --batch_size ${BATCH_SIZE} \
  --num_samples ${NUM_SAMPLES} \
"

# Add heuristic args if enabled
if [ "${RUN_HEURISTIC}" = true ]; then
  COMMON_ARGS="${COMMON_ARGS} \
    --run_heuristic \
    --heuristic_util_folder ${HEURISTIC_UTIL_FOLDER} \
    --heuristic_config_path ${HEURISTIC_CONFIG_PATH} \
  "
  if [ -n "${HEURISTIC_SCRIPT_DIR}" ]; then
    COMMON_ARGS="${COMMON_ARGS} --heuristic_script_dir ${HEURISTIC_SCRIPT_DIR}"
  fi
fi

case ${MODE} in
  single)
    echo "Running single-GPU evaluation..."
    python3 eval_merged_quantized.py ${COMMON_ARGS}
    ;;

  multigpu)
    echo "Running multi-GPU evaluation on ${NUM_GPUS} GPUs..."
    accelerate launch \
      --num_processes ${NUM_GPUS} \
      --mixed_precision bf16 \
      eval_merged_quantized.py ${COMMON_ARGS}
    ;;

  *)
    echo "Unknown mode: ${MODE}"
    echo ""
    echo "Usage: bash eval_merged_quantized.sh [MODE] [NUM_GPUS]"
    echo ""
    echo "Available modes:"
    echo "  single    - Single GPU evaluation (default)"
    echo "  multigpu  - Multi-GPU evaluation with Accelerate"
    echo ""
    echo "Arguments:"
    echo "  NUM_GPUS  - Number of GPUs (default: auto-detect)"
    exit 1
    ;;
esac

echo "============================================"
echo "Done! Mode: ${MODE}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Results:"
echo "  Predictions: ${OUTPUT_DIR}/predictions.jsonl"
echo "  Summary:     ${OUTPUT_DIR}/eval_summary.json"
echo "============================================"
