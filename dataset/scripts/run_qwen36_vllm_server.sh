#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3.6-35B-A3B}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${HOME}/models/Qwen--Qwen3.6-35B-A3B}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_SWAP_SPACE="${VLLM_SWAP_SPACE:-8}"
VLLM_QUANTIZATION_MODE="${VLLM_QUANTIZATION_MODE:-none}"
VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-}"
VLLM_CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}"
VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-qwen3}"
VLLM_ARCHITECTURE_OVERRIDE="${VLLM_ARCHITECTURE_OVERRIDE:-auto}"

if [[ ! -d "${QWEN_MODEL_PATH}" ]]; then
  echo "Model folder not found: ${QWEN_MODEL_PATH}" >&2
  echo "Set QWEN_MODEL_PATH to the copied Hugging Face snapshot folder." >&2
  exit 1
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | paste -sd, -)"
    export CUDA_VISIBLE_DEVICES
  fi
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
  A2UI_VLLM_GPUS="${A2UI_VLLM_GPUS:-${#_gpu_ids[@]}}"
else
  A2UI_VLLM_GPUS="${A2UI_VLLM_GPUS:-1}"
fi

echo "python: $(python --version 2>&1)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "A2UI_VLLM_GPUS=${A2UI_VLLM_GPUS}"
echo "QWEN_MODEL_PATH=${QWEN_MODEL_PATH}"
echo "VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION}"
echo "VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN}"
echo "VLLM_QUANTIZATION_MODE=${VLLM_QUANTIZATION_MODE}"
echo "VLLM_KV_CACHE_DTYPE=${VLLM_KV_CACHE_DTYPE:-auto}"
echo "VLLM_CPU_OFFLOAD_GB=${VLLM_CPU_OFFLOAD_GB:-0}"
echo "VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-vLLM default}"

SERVER_ARGS=(
  --model-path "${QWEN_MODEL_PATH}"
  --served-model-name "${QWEN_MODEL_ID}"
  --gpus "${A2UI_VLLM_GPUS}"
  --host "${VLLM_HOST}"
  --port "${VLLM_PORT}"
  --dtype "${VLLM_DTYPE}"
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
  --max-model-len "${VLLM_MAX_MODEL_LEN}"
  --swap-space "${VLLM_SWAP_SPACE}"
  --trust-remote-code
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES:-}"
  --enable-reasoning
  --reasoning-parser "${VLLM_REASONING_PARSER}"
  --architecture-override "${VLLM_ARCHITECTURE_OVERRIDE}"
  --quantization-mode "${VLLM_QUANTIZATION_MODE}"
)

if [[ -n "${VLLM_KV_CACHE_DTYPE}" ]]; then
  SERVER_ARGS+=(--kv-cache-dtype "${VLLM_KV_CACHE_DTYPE}")
fi

if [[ -n "${VLLM_CPU_OFFLOAD_GB}" ]]; then
  SERVER_ARGS+=(--cpu-offload-gb "${VLLM_CPU_OFFLOAD_GB}")
fi

if [[ -n "${VLLM_MAX_NUM_SEQS}" ]]; then
  SERVER_ARGS+=(--max-num-seqs "${VLLM_MAX_NUM_SEQS}")
fi

exec python "${REPO_ROOT}/dataset/scripts/serve_qwen_vllm.py" \
  "${SERVER_ARGS[@]}"
