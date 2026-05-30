#!/usr/bin/env bash
set -euo pipefail

# Start Gemma4 31B on the CUDA 13.0/Ada speculative-decoding container.
# This wrapper keeps the existing Gemma4 runtime behavior but changes defaults
# for the CU130 SIF and auto-enables every visible GPU.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export VLLM_SIF="${VLLM_SIF:-${HOME}/containers/a2ui-vllm-cu130-source_gemma4_speculative_ada.sif}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
export VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
export GEMMA4_SPECULATIVE_MODE="${GEMMA4_SPECULATIVE_MODE:-draft}"
export GEMMA4_SPECULATIVE_TOKENS="${GEMMA4_SPECULATIVE_TOKENS:-4}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | paste -sd, -)"
  export CUDA_VISIBLE_DEVICES
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && -z "${A2UI_VLLM_GPUS:-}" ]]; then
  IFS=',' read -r -a _a2ui_gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
  export A2UI_VLLM_GPUS="${#_a2ui_gpu_ids[@]}"
fi

exec bash "${SCRIPT_DIR}/run_gemma4_vllm_singularity.sh"
