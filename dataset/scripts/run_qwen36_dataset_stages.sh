#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_ID="${RUN_ID:-dataset_qwen36_vllm_reasoning_v0}"
MODEL_NAME="${MODEL_NAME:-qwen36_35b_a3b_vllm_reasoning}"
RATE_LIMIT_QPS="${RATE_LIMIT_QPS:-0.2}"
STAGE3_BATCH_SIZE="${STAGE3_BATCH_SIZE:-1}"
MAX_QUERIES_TOTAL="${MAX_QUERIES_TOTAL:-}"
MAX_RESPONSES_TOTAL="${MAX_RESPONSES_TOTAL:-}"
MAX_GENUI_TOTAL="${MAX_GENUI_TOTAL:-}"
STAGE="${STAGE:-all}"

export LOCAL_ALLOW_HTTP_ENDPOINT="${LOCAL_ALLOW_HTTP_ENDPOINT:-1}"
export LOCAL_STRICT_OFFLINE="${LOCAL_STRICT_OFFLINE:-0}"
export LOCAL_VLLM_ENABLE_THINKING="${LOCAL_VLLM_ENABLE_THINKING:-1}"
export VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-qwen3}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

run_stage() {
  local stage="$1"
  shift
  local max_args=()
  if [[ -n "${MAX_QUERIES_TOTAL}" ]]; then
    max_args+=(--max_queries_total "${MAX_QUERIES_TOTAL}")
  fi
  if [[ -n "${MAX_RESPONSES_TOTAL}" ]]; then
    max_args+=(--max_responses_total "${MAX_RESPONSES_TOTAL}")
  fi
  if [[ -n "${MAX_GENUI_TOTAL}" ]]; then
    max_args+=(--max_genui_total "${MAX_GENUI_TOTAL}")
  fi
  echo "Running Stage ${stage} with ${MODEL_NAME}, run_id=${RUN_ID}"
  python dataset/src/main.py \
    --stage "${stage}" \
    --model "${MODEL_NAME}" \
    --run_id "${RUN_ID}" \
    --rate_limit_qps "${RATE_LIMIT_QPS}" \
    "${max_args[@]}" \
    "$@"
}

case "${STAGE}" in
  1)
    run_stage 1
    ;;
  2)
    run_stage 2
    ;;
  3)
    run_stage 3 --genui_batch_size "${STAGE3_BATCH_SIZE}"
    ;;
  all)
    run_stage 1
    run_stage 2
    run_stage 3 --genui_batch_size "${STAGE3_BATCH_SIZE}"
    ;;
  *)
    echo "Unknown STAGE=${STAGE}; use 1, 2, 3, or all" >&2
    exit 1
    ;;
esac
