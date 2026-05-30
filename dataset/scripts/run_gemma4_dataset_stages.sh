#!/usr/bin/env bash
set -euo pipefail

# Run dataset Stage 1/2/3/4/5 against a running Gemma4 31B vLLM
# OpenAI-compatible endpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GEMMA4_ENABLE_REASONING="${GEMMA4_ENABLE_REASONING:-0}"
if [[ "${GEMMA4_ENABLE_REASONING,,}" =~ ^(1|true|yes|y|on)$ ]]; then
  MODEL_NAME="${MODEL_NAME:-gemma4_31b_vllm_reasoning}"
  export LOCAL_VLLM_ENABLE_THINKING="${LOCAL_VLLM_ENABLE_THINKING:-1}"
else
  MODEL_NAME="${MODEL_NAME:-gemma4_31b_vllm}"
  export LOCAL_VLLM_ENABLE_THINKING="${LOCAL_VLLM_ENABLE_THINKING:-0}"
fi

RUN_ID="${RUN_ID:-dataset_gemma4_31b_vllm_v0}"
MODEL_ROOT="${MODEL_ROOT:-${LOCAL_MODEL_ROOT:-${A2UI_MODEL_ROOT:-${GEMMA4_MODEL_ROOT:-}}}}"
RATE_LIMIT_QPS="${RATE_LIMIT_QPS:-0.2}"
STAGE3_BATCH_SIZE="${STAGE3_BATCH_SIZE:-1}"
RENDER_WORKERS="${RENDER_WORKERS:-1}"
MAX_QUERIES_TOTAL="${MAX_QUERIES_TOTAL:-}"
MAX_RESPONSES_TOTAL="${MAX_RESPONSES_TOTAL:-}"
MAX_GENUI_TOTAL="${MAX_GENUI_TOTAL:-}"
STAGE="${STAGE:-all}"

export LOCAL_ALLOW_HTTP_ENDPOINT="${LOCAL_ALLOW_HTTP_ENDPOINT:-1}"
export LOCAL_STRICT_OFFLINE="${LOCAL_STRICT_OFFLINE:-0}"
export LOCAL_VLLM_STRIP_THINKING="${LOCAL_VLLM_STRIP_THINKING:-1}"
export LOCAL_VLLM_TIMEOUT_SECONDS="${LOCAL_VLLM_TIMEOUT_SECONDS:-600}"
export LOCAL_VLLM_MAX_OUTPUT_TOKENS="${LOCAL_VLLM_MAX_OUTPUT_TOKENS:-8192}"
export LOCAL_STAGE3_PROMPT_MAX_TOKENS="${LOCAL_STAGE3_PROMPT_MAX_TOKENS:-8192}"
export A2UI_QUERY_MAX_TOKENS="${A2UI_QUERY_MAX_TOKENS:-8192}"
export A2UI_RESPONSE_MAX_TOKENS="${A2UI_RESPONSE_MAX_TOKENS:-8192}"
export A2UI_GENUI_MAX_TOKENS="${A2UI_GENUI_MAX_TOKENS:-8192}"
export A2UI_GENUI_PROMPT_MAX_TOKENS="${A2UI_GENUI_PROMPT_MAX_TOKENS:-8192}"
export GEMMA4_MODEL_ID="${GEMMA4_MODEL_ID:-google/gemma-4-31b-it}"
export MODEL_ROOT
export LOCAL_MODEL_ROOT="${LOCAL_MODEL_ROOT:-${MODEL_ROOT}}"
export A2UI_MODEL_ROOT="${A2UI_MODEL_ROOT:-${MODEL_ROOT}}"
export GEMMA4_MODEL_ROOT="${GEMMA4_MODEL_ROOT:-${MODEL_ROOT}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export A2UI_DISABLE_SSL_VERIFY="${A2UI_DISABLE_SSL_VERIFY:-1}"
if [[ "${A2UI_DISABLE_SSL_VERIFY}" = "1" ]]; then
  export CURL_CA_BUNDLE=""
  export REQUESTS_CA_BUNDLE=""
  export SSL_CERT_FILE=""
  export PYTHONHTTPSVERIFY=0
  export GIT_SSL_NO_VERIFY=1
  export HF_HUB_DISABLE_SSL_VERIFICATION=1
fi

ensure_vllm_served_model() {
  if [[ "${A2UI_SKIP_VLLM_MODEL_CHECK:-0}" = "1" ]]; then
    export LOCAL_VLLM_SERVED_MODEL="${LOCAL_VLLM_SERVED_MODEL:-${GEMMA4_MODEL_ID}}"
    return 0
  fi

  local expected="${LOCAL_VLLM_SERVED_MODEL:-${GEMMA4_MODEL_ID}}"
  local models_url="${LOCAL_VLLM_MODELS_URL:-http://127.0.0.1:8000/v1/models}"
  local served
  served="$(python - "${models_url}" "${expected}" <<'PY'
import json
import sys
import urllib.request

models_url = sys.argv[1]
expected = sys.argv[2]
try:
    with urllib.request.urlopen(models_url, timeout=5) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
except Exception as exc:
    print(f"Could not read local vLLM models from {models_url}: {exc}", file=sys.stderr)
    raise SystemExit(2)

ids = [
    item.get("id")
    for item in payload.get("data", [])
    if isinstance(item, dict) and isinstance(item.get("id"), str)
]
if expected in ids:
    print(expected)
    raise SystemExit(0)
for model_id in ids:
    if model_id.lower() == expected.lower():
        print(model_id)
        raise SystemExit(0)

print(
    "Local vLLM is running, but the requested model is not served.\n"
    f"  requested: {expected}\n"
    f"  served: {ids}\n"
    "Fix: restart vLLM with dataset/scripts/run_gemma4_vllm_python.sh, "
    "or export LOCAL_VLLM_SERVED_MODEL to one of the served ids.",
    file=sys.stderr,
)
raise SystemExit(3)
PY
)" || return $?

  export LOCAL_VLLM_SERVED_MODEL="${served}"
  echo "Using local vLLM served model: ${LOCAL_VLLM_SERVED_MODEL}"
}

case "${STAGE}" in
  1|2|3|all)
    ensure_vllm_served_model
    ;;
esac

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
  echo "Running Stage ${stage} with ${MODEL_NAME}, run_id=${RUN_ID}, reasoning=${GEMMA4_ENABLE_REASONING}"
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
  4)
    run_stage 4 --render_workers "${RENDER_WORKERS}"
    ;;
  5)
    run_stage 5
    ;;
  all)
    run_stage 1
    run_stage 2
    run_stage 3 --genui_batch_size "${STAGE3_BATCH_SIZE}"
    run_stage 4 --render_workers "${RENDER_WORKERS}"
    run_stage 5
    ;;
  *)
    echo "Unknown STAGE=${STAGE}; use 1, 2, 3, 4, 5, or all" >&2
    exit 1
    ;;
esac
