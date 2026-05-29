#!/usr/bin/env bash
set -euo pipefail

# Single-terminal helper: start Gemma4 31B vLLM in the background, wait until
# /v1/models is ready, then run Stage 1, Stage 2, and Stage 3 sequentially.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_ID="${RUN_ID:-dataset_gemma4_31b_vllm_50}"
RUN_DIR="${REPO_ROOT}/dataset/data/runs/${RUN_ID}"
TARGET_COUNT="${TARGET_COUNT:-50}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_START_TIMEOUT="${VLLM_START_TIMEOUT:-1200}"
START_VLLM="${START_VLLM:-1}"

mkdir -p "${RUN_DIR}/logs"

VLLM_PID=""
cleanup() {
  if [[ -n "${VLLM_PID}" ]]; then
    kill "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "${START_VLLM}" != "0" ]]; then
  echo "Starting Gemma4 vLLM server..."
  bash dataset/scripts/run_gemma4_vllm_singularity.sh > "${RUN_DIR}/logs/vllm_server.log" 2>&1 &
  VLLM_PID=$!
  echo "vLLM PID=${VLLM_PID}"
fi

deadline=$((SECONDS + VLLM_START_TIMEOUT))
until curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    echo "vLLM did not become ready within ${VLLM_START_TIMEOUT}s" >&2
    tail -120 "${RUN_DIR}/logs/vllm_server.log" >&2 || true
    exit 1
  fi
  echo "waiting for Gemma4 vLLM..."
  tail -n 20 "${RUN_DIR}/logs/vllm_server.log" || true
  sleep 10
done

echo "Gemma4 vLLM ready"

STAGE=1 \
RUN_ID="${RUN_ID}" \
MAX_QUERIES_TOTAL="${TARGET_COUNT}" \
bash dataset/scripts/run_gemma4_dataset_stages.sh \
  2>&1 | tee "${RUN_DIR}/logs/stage1.log"

STAGE=2 \
RUN_ID="${RUN_ID}" \
MAX_RESPONSES_TOTAL="${TARGET_COUNT}" \
bash dataset/scripts/run_gemma4_dataset_stages.sh \
  2>&1 | tee "${RUN_DIR}/logs/stage2.log"

STAGE=3 \
RUN_ID="${RUN_ID}" \
MAX_GENUI_TOTAL="${TARGET_COUNT}" \
bash dataset/scripts/run_gemma4_dataset_stages.sh \
  2>&1 | tee "${RUN_DIR}/logs/stage3.log"

echo "Complete: ${RUN_DIR}"
wc -l "${RUN_DIR}/queries.jsonl" "${RUN_DIR}/responses.jsonl" "${RUN_DIR}/genui.jsonl" || true
