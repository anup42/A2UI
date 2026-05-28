#!/usr/bin/env bash
set -euo pipefail

# Start Qwen3.6-35B-A3B through the custom CUDA 12.8 vLLM SIF.
# Run inside a Slurm GPU allocation, for example:
#   srun --gres=gpu:2 --cpus-per-task=16 --mem=160G --pty bash
#   module load apptainer
#   VLLM_SIF=/isilonhome/k_anup/containers/a2ui-vllm-cu128-source_qwen36.sif \
#   bash dataset/scripts/run_qwen36_vllm_cu128_apptainer.sh

QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3.6-35B-A3B}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${HOME}/models/Qwen--Qwen3.6-35B-A3B}"
VLLM_SIF="${VLLM_SIF:-${HOME}/containers/a2ui-vllm-cu128-source_qwen36.sif}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-qwen3}"
VLLM_CONTAINER_MODEL_PATH="${VLLM_CONTAINER_MODEL_PATH:-/models/qwen36}"

if [[ ! -f "${VLLM_SIF}" ]]; then
  echo "SIF not found: ${VLLM_SIF}" >&2
  exit 1
fi

if [[ ! -d "${QWEN_MODEL_PATH}" ]]; then
  echo "Model folder not found: ${QWEN_MODEL_PATH}" >&2
  exit 1
fi

if ! command -v apptainer >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
  module load apptainer || true
fi

if command -v apptainer >/dev/null 2>&1; then
  RUNTIME="apptainer"
elif command -v singularity >/dev/null 2>&1; then
  RUNTIME="singularity"
else
  echo "apptainer or singularity is required. Try: module load apptainer" >&2
  exit 1
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | paste -sd, -)"
  export CUDA_VISIBLE_DEVICES
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
  A2UI_VLLM_GPUS="${A2UI_VLLM_GPUS:-${#gpu_ids[@]}}"
else
  A2UI_VLLM_GPUS="${A2UI_VLLM_GPUS:-1}"
fi

echo "Runtime: ${RUNTIME}"
echo "SIF: ${VLLM_SIF}"
echo "Model path: ${QWEN_MODEL_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Tensor parallel GPUs: ${A2UI_VLLM_GPUS}"

exec "${RUNTIME}" exec --nv --cleanenv \
  --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
  --bind "${QWEN_MODEL_PATH}:${VLLM_CONTAINER_MODEL_PATH}:ro" \
  "${VLLM_SIF}" \
  vllm serve "${VLLM_CONTAINER_MODEL_PATH}" \
    --served-model-name "${QWEN_MODEL_ID}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size "${A2UI_VLLM_GPUS}" \
    --dtype "${VLLM_DTYPE}" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --reasoning-parser "${VLLM_REASONING_PARSER}" \
    --trust-remote-code
