#!/usr/bin/env bash
set -euo pipefail

# Run this inside an interactive Slurm allocation, e.g. after:
#   srun --gres=gpu:2 --cpus-per-task=16 --mem=120G --time=08:00:00 --pty bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_ID="${RUN_ID:-dataset_qwen36_slurm_50}"
TARGET_COUNT="${TARGET_COUNT:-50}"
MODEL_NAME="${MODEL_NAME:-qwen36_35b_a3b_vllm_reasoning}"
RATE_LIMIT_QPS="${RATE_LIMIT_QPS:-0.2}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-900}"
QWEN_ENV_ACTIVATE="${QWEN_ENV_ACTIVATE:-${REPO_ROOT}/qwen36_vllm_env/activate_qwen36_vllm.sh}"

if [[ -f "${QWEN_ENV_ACTIVATE}" ]]; then
  # shellcheck source=/dev/null
  source "${QWEN_ENV_ACTIVATE}"
fi

export QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${HOME}/models/Qwen--Qwen3.6-35B-A3B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export A2UI_VLLM_GPUS="${A2UI_VLLM_GPUS:-2}"
export LOCAL_ALLOW_HTTP_ENDPOINT="${LOCAL_ALLOW_HTTP_ENDPOINT:-1}"
export LOCAL_STRICT_OFFLINE="${LOCAL_STRICT_OFFLINE:-0}"
export LOCAL_VLLM_ENABLE_THINKING="${LOCAL_VLLM_ENABLE_THINKING:-1}"
export VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-qwen3}"

RUN_DIR="${REPO_ROOT}/dataset/data/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"

vllm_ready() {
  curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1
}

VLLM_PID=""
if vllm_ready; then
  echo "vLLM already running on port ${VLLM_PORT}"
else
  echo "Starting vLLM on port ${VLLM_PORT}"
  bash dataset/scripts/run_qwen36_vllm_python.sh > "${RUN_DIR}/vllm_server.log" 2>&1 &
  VLLM_PID="$!"
  trap 'if [[ -n "${VLLM_PID}" ]]; then kill "${VLLM_PID}" 2>/dev/null || true; fi' EXIT

  deadline=$((SECONDS + VLLM_READY_TIMEOUT))
  until vllm_ready; do
    if (( SECONDS >= deadline )); then
      echo "vLLM did not become ready within ${VLLM_READY_TIMEOUT}s" >&2
      tail -80 "${RUN_DIR}/vllm_server.log" >&2 || true
      exit 1
    fi
    sleep 5
  done
fi

echo "Running Stage 1 for ${TARGET_COUNT} samples, run_id=${RUN_ID}"
STAGE=1 \
RUN_ID="${RUN_ID}" \
MODEL_NAME="${MODEL_NAME}" \
RATE_LIMIT_QPS="${RATE_LIMIT_QPS}" \
MAX_QUERIES_TOTAL="${TARGET_COUNT}" \
bash dataset/scripts/run_qwen36_dataset_stages.sh

echo "Stage 1 complete: ${RUN_DIR}/queries.jsonl"
