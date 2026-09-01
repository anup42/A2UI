#!/bin/bash
# Convert QAT+LoRA checkpoint to real quantized model

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Force NVIDIA GPU and disable Unsloth's problematic AMD detection
export FORCE_CUDA=1

# GPU configuration (edit to set which GPUs to use)
GPU_IDS="${GPU_IDS:-0}"  # Can override with: GPU_IDS="0,1,2,3" ./run_convert_qat.sh

# Paths (edit these)
CHECKPOINT_PATH="${SCRIPT_DIR}/runs/sft_run_gemma4_E2B_qat_unsloth_test/checkpoint-10"
OUTPUT_DIR="${SCRIPT_DIR}/runs/sft_run_gemma4_E2B_qat_unsloth_test/model_int4_quantized"
MODEL_NAME="/home/h_sagri/Storage_gpu/PlatformIntelligenceTeam/h_sagri/mdc/models/gemma-4-E2B-it"
TEST_DATA_PATH="/home/h_sagri/GenUI-LM/dataset/data/runs/golden100_v3/golden50_url_mapping.jsonl"

# QAT+LoRA hyperparameters -- MUST match the values used in the original
# train_qat_unsloth.py run (check eval_config.json in the run directory
# if unsure), otherwise the rebuilt structure won't match the checkpoint.
LORA_R=16
LORA_ALPHA=32
QAT_SCHEME="int4"
# Also match eval samples from training for consistency
EVAL_SAMPLES=6
EVAL_SAMPLES=6

echo "=========================================="
echo "QAT to Quantized Model Converter"
echo "=========================================="
echo "Checkpoint:  $CHECKPOINT_PATH"
echo "Output:      $OUTPUT_DIR"
echo "Test data:   $TEST_DATA_PATH"
echo "=========================================="

# For int8 instead of int4, add: --quantization_bits 8
python "$SCRIPT_DIR/convert_qat_to_quantized.py" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_PATH" \
    --gpu_ids "$GPU_IDS" \
    --test_data_path "$TEST_DATA_PATH" \
    --eval_samples "$EVAL_SAMPLES" \
    --eval_batch_size 2 \
    --quantization_bits 4 \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --qat_scheme "$QAT_SCHEME" \
    --max_length 6144 \
    --max_new_tokens 2048

echo ""
echo "=========================================="
echo "✓ Done!"
echo "Quantized model: $OUTPUT_DIR/model_quantized"
echo "=========================================="
