#!/bin/bash
set -e

# QAT+LoRA via Unsloth with DDP multi-GPU and fast batched eval.
#
# Usage:
#   ./run_qat_unsloth_ddp.sh [NUM_GPUS] [GPU_IDS]
#
# Examples:
#   ./run_qat_unsloth_ddp.sh              # 2 GPUs, ids 0,1
#   ./run_qat_unsloth_ddp.sh 4            # 4 GPUs, ids 0,1,2,3
#   ./run_qat_unsloth_ddp.sh 2 1,2        # 2 GPUs, ids 1 and 2
#   ./run_qat_unsloth_ddp.sh 1            # single process (no DDP)
#   RESUME=auto ./run_qat_unsloth_ddp.sh 2         # resume from the latest checkpoint
#   RESUME=/path/to/checkpoint-1200 ./run_qat_unsloth_ddp.sh 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_GPUS="${1:-2}"
GPU_IDS="${2:-}"
RESUME_FROM_CHECKPOINT="${RESUME:-}"

MODEL_PATH="/home/adarsh_ag/models/gemma-3-270m"
DATA_PATH="/home/adarsh_ag/GenUI-LM/training_scripts/sft/gen_data_v2/genui_processed_clean_merged.jsonl"
OUTPUT_DIR="${SCRIPT_DIR}/runs/sft_run_gemma3_270m_qat_unsloth_ddp"
TEST_DATA_PATH="/home/adarsh_ag/GenUI-LM/training_scripts/sft/test_data/golden100_url_mapping.jsonl"
HEURISTIC_UTIL_FOLDER="/home/adarsh_ag/GenUI-LM/dataset/data/runs/golden100_gt"

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

# This container presets CUDA_VISIBLE_DEVICES=0,1,2,3 and nvidia-smi lists 4 GPUs,
# but the CUDA runtime only initializes 3. NVML-based device counting then walks
# off the end and trips:
#   "device >= 0 && device < num_gpus INTERNAL ASSERT FAILED ... device=3, num_gpus=3"
# Counting with the CUDA runtime instead keeps the two views consistent.
export PYTORCH_NVML_BASED_CUDA_CHECK=0

# Restrict the visible set only if GPU_IDS was given explicitly. torchrun assigns
# one rank per visible device and owns LOCAL_RANK from there -- do not pin
# per-rank devices by hand.
if [ -n "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    echo "Restricted to GPUs: $GPU_IDS"
fi

# Long heuristic evals run as a blocking subprocess on rank 0 while other ranks
# wait at the next collective; without a raised timeout NCCL aborts the job.
export NCCL_TIMEOUT=7200000
export NCCL_BLOCKING_WAIT=1
export TOKENIZERS_PARALLELISM=false

echo "=========================================="
echo "QAT+LoRA DDP Training (train_qat_unsloth_ddp.py)"
echo "=========================================="
echo "Processes:  $NUM_GPUS"
echo "Model:      $MODEL_PATH"
echo "Data:       $DATA_PATH"
echo "Output:     $OUTPUT_DIR"
echo "Test Data:  $TEST_DATA_PATH"
echo "=========================================="

torchrun --nproc_per_node="$NUM_GPUS" --master_port=29511 \
    "$SCRIPT_DIR/train_qat_unsloth_ddp.py" \
    --model_name "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --test_data_path "$TEST_DATA_PATH" \
    --heuristic_util_folder "$HEURISTIC_UTIL_FOLDER" \
    --epochs 4 \
    --lr 5e-5 \
    --batch_size 8 \
    --eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --max_length 6144 \
    --logging_steps 50 \
    --eval_steps 200 \
    --eval_samples 100 \
    --save_steps 600 \
    --lora_r 16 \
    --lora_alpha 32 \
    --qat_scheme int8-int4 \
    --fast_eval_sources both \
    --fast_eval_batch_size 16 \
    --fast_eval_max_new_tokens 0 \
    --fast_eval_val_samples 50 \
    --fast_eval_selftest \
    --save_predictions_interval 1 \
    --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT" \
    --report_to tensorboard

echo ""
echo "=========================================="
echo "Training completed!"
echo "Output: $OUTPUT_DIR"
echo "TensorBoard: tensorboard --logdir $OUTPUT_DIR"
echo "=========================================="
