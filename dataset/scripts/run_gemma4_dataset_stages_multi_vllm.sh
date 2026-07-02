#!/usr/bin/env bash
set -euo pipefail

# Run dataset stages against multiple already-running vLLM servers.
#
# The dataset process remains single-writer for run artifacts, while
# dataset/src/llm/local_adapter.py distributes concurrent batch requests over
# LOCAL_VLLM_ENDPOINTS in round-robin order.
#
# Endpoint discovery order:
#   1. LOCAL_VLLM_ENDPOINTS
#   2. VLLM_ENDPOINTS_ENV_FILE, default /tmp/a2ui_gemma4_vllm_replicas.env
#   3. VLLM_PORTS, e.g. 8000,8001,8002,8003
#   4. Inferred ports from visible GPU count, starting at VLLM_BASE_PORT
#
# Example:
#   source /tmp/a2ui_gemma4_vllm_replicas.env
#   RUN_ID=dataset_gemma4_multi STAGE=cyclic MAX_GENERATION_TOTAL=50000 \
#     bash dataset/scripts/run_gemma4_dataset_stages_multi_vllm.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

BASE_STAGE_SCRIPT="${A2UI_DATASET_STAGE_SCRIPT:-${SCRIPT_DIR}/run_gemma4_dataset_stages.sh}"
if [[ ! -f "${BASE_STAGE_SCRIPT}" ]]; then
  echo "Dataset stage script not found: ${BASE_STAGE_SCRIPT}" >&2
  exit 1
fi

VLLM_ENDPOINTS_ENV_FILE="${VLLM_ENDPOINTS_ENV_FILE:-/tmp/a2ui_gemma4_vllm_replicas.env}"
VLLM_BASE_PORT="${VLLM_BASE_PORT:-8000}"
VLLM_HOST_FOR_CLIENT="${VLLM_HOST_FOR_CLIENT:-127.0.0.1}"
LOCAL_VLLM_REQUESTS_PER_SERVER="${LOCAL_VLLM_REQUESTS_PER_SERVER:-2}"
LOCAL_VLLM_WAIT_ALL_ENDPOINTS="${LOCAL_VLLM_WAIT_ALL_ENDPOINTS:-1}"
LOCAL_VLLM_WAIT_TIMEOUT_SECONDS="${LOCAL_VLLM_WAIT_TIMEOUT_SECONDS:-1800}"
LOCAL_VLLM_WAIT_INTERVAL_SECONDS="${LOCAL_VLLM_WAIT_INTERVAL_SECONDS:-10}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

if [[ -z "${LOCAL_VLLM_ENDPOINTS:-}" && -f "${VLLM_ENDPOINTS_ENV_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${VLLM_ENDPOINTS_ENV_FILE}"
fi

normalize_endpoint() {
  local endpoint="${1:-}"
  endpoint="${endpoint%% }"
  endpoint="${endpoint## }"
  endpoint="${endpoint%/}"
  if [[ -z "${endpoint}" ]]; then
    return 0
  fi
  case "${endpoint}" in
    */v1/chat/completions|*/chat/completions)
      printf '%s\n' "${endpoint}"
      ;;
    */v1)
      printf '%s/chat/completions\n' "${endpoint}"
      ;;
    http://*|https://*)
      printf '%s/v1/chat/completions\n' "${endpoint}"
      ;;
    *)
      printf 'http://%s/v1/chat/completions\n' "${endpoint}"
      ;;
  esac
}

detect_gpu_count() {
  if [[ -n "${REPLICA_GPU_IDS:-}" ]]; then
    local raw="${REPLICA_GPU_IDS//,/ }"
    raw="${raw//;/ }"
    # shellcheck disable=SC2206
    local items=( ${raw} )
    echo "${#items[@]}"
    return 0
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local raw="${CUDA_VISIBLE_DEVICES//,/ }"
    raw="${raw//;/ }"
    # shellcheck disable=SC2206
    local items=( ${raw} )
    echo "${#items[@]}"
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | awk 'NF {count++} END {print count > 0 ? count : 1}'
    return 0
  fi
  echo 1
}

split_list() {
  local raw="${1:-}"
  raw="${raw//,/ }"
  raw="${raw//;/ }"
  local item
  for item in ${raw}; do
    [[ -n "${item}" ]] && printf '%s\n' "${item}"
  done
}

declare -a ENDPOINTS=()
if [[ -n "${LOCAL_VLLM_ENDPOINTS:-}" ]]; then
  while IFS= read -r endpoint; do
    [[ -n "${endpoint}" ]] && ENDPOINTS+=("${endpoint}")
  done < <(split_list "${LOCAL_VLLM_ENDPOINTS}")
elif [[ -n "${VLLM_PORTS:-}" ]]; then
  while IFS= read -r port; do
    [[ -n "${port}" ]] && ENDPOINTS+=("http://${VLLM_HOST_FOR_CLIENT}:${port}/v1/chat/completions")
  done < <(split_list "${VLLM_PORTS}")
else
  gpu_count="$(detect_gpu_count)"
  for (( idx=0; idx<gpu_count; idx++ )); do
    ENDPOINTS+=("http://${VLLM_HOST_FOR_CLIENT}:$((VLLM_BASE_PORT + idx))/v1/chat/completions")
  done
fi

if (( ${#ENDPOINTS[@]} == 0 )); then
  echo "No vLLM endpoints resolved. Set LOCAL_VLLM_ENDPOINTS or VLLM_PORTS." >&2
  exit 1
fi

declare -a NORMALIZED_ENDPOINTS=()
for endpoint in "${ENDPOINTS[@]}"; do
  normalized="$(normalize_endpoint "${endpoint}")"
  [[ -n "${normalized}" ]] && NORMALIZED_ENDPOINTS+=("${normalized}")
done
ENDPOINTS=("${NORMALIZED_ENDPOINTS[@]}")

endpoint_count="${#ENDPOINTS[@]}"
parallelism=$(( endpoint_count * LOCAL_VLLM_REQUESTS_PER_SERVER ))
if (( parallelism < 1 )); then
  parallelism=1
fi

export LOCAL_VLLM_ENDPOINTS="$(IFS=,; echo "${ENDPOINTS[*]}")"
export LOCAL_VLLM_MODELS_URL="${LOCAL_VLLM_MODELS_URL:-${ENDPOINTS[0]%/chat/completions}/models}"
export LOCAL_ALLOW_HTTP_ENDPOINT="${LOCAL_ALLOW_HTTP_ENDPOINT:-1}"
export LOCAL_STRICT_OFFLINE="${LOCAL_STRICT_OFFLINE:-0}"
export LOCAL_VLLM_LOG_REQUESTS="${LOCAL_VLLM_LOG_REQUESTS:-1}"
export LOCAL_VLLM_BATCH_PARALLELISM="${LOCAL_VLLM_BATCH_PARALLELISM:-${parallelism}}"
export LOCAL_VLLM_PARALLEL_REQUESTS="${LOCAL_VLLM_PARALLEL_REQUESTS:-${LOCAL_VLLM_BATCH_PARALLELISM}}"
export STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-${LOCAL_VLLM_BATCH_PARALLELISM}}"
export STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-${LOCAL_VLLM_BATCH_PARALLELISM}}"
export STAGE3_BATCH_SIZE="${STAGE3_BATCH_SIZE:-${LOCAL_VLLM_BATCH_PARALLELISM}}"
export LOCAL_VLLM_RETRY_CONNECTION_ERRORS="${LOCAL_VLLM_RETRY_CONNECTION_ERRORS:-1}"
export RATE_LIMIT_QPS="${RATE_LIMIT_QPS:-0}"

if [[ "${LOCAL_VLLM_WAIT_ALL_ENDPOINTS}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  "${PYTHON_BIN}" - <<'PY'
import json
import os
import re
import sys
import time
import urllib.request
from urllib.parse import urlparse

endpoints = [
    item
    for item in re.split(r"[,;\s]+", os.environ.get("LOCAL_VLLM_ENDPOINTS", ""))
    if item
]
expected = (
    os.environ.get("LOCAL_VLLM_SERVED_MODEL")
    or os.environ.get("GEMMA4_MODEL_ID")
    or "google/gemma-4-31b-it"
)
timeout_s = float(os.environ.get("LOCAL_VLLM_WAIT_TIMEOUT_SECONDS", "1800") or "1800")
interval_s = float(os.environ.get("LOCAL_VLLM_WAIT_INTERVAL_SECONDS", "10") or "10")

def models_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/chat/completions"):
        return endpoint[: -len("/chat/completions")] + "/models"
    if endpoint.endswith("/v1"):
        return endpoint + "/models"
    parsed = urlparse(endpoint)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}/v1/models"
    return endpoint

model_urls = [models_url(endpoint) for endpoint in endpoints]
start = time.time()
while True:
    missing: list[str] = []
    wrong: list[str] = []
    for url in model_urls:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            ids = [
                item.get("id")
                for item in payload.get("data", [])
                if isinstance(item, dict) and isinstance(item.get("id"), str)
            ]
        except Exception as exc:
            missing.append(f"{url} ({exc})")
            continue
        if expected not in ids and expected.lower() not in {item.lower() for item in ids}:
            wrong.append(f"{url} served={ids}")
    if not missing and not wrong:
        break
    elapsed = time.time() - start
    if elapsed >= timeout_s:
        print("Timed out waiting for all vLLM endpoints.", file=sys.stderr)
        if missing:
            print("Unavailable endpoints:", *missing, sep="\n  ", file=sys.stderr)
        if wrong:
            print("Wrong model endpoints:", *wrong, sep="\n  ", file=sys.stderr)
        raise SystemExit(2)
    print(
        f"Waiting for {len(missing) + len(wrong)} vLLM endpoints "
        f"({elapsed:.0f}s/{timeout_s:.0f}s)...",
        file=sys.stderr,
        flush=True,
    )
    time.sleep(max(1.0, interval_s))
PY
fi

echo "Using ${endpoint_count} local vLLM endpoints"
printf '  %s\n' "${ENDPOINTS[@]}"
echo "Batch parallelism: ${LOCAL_VLLM_BATCH_PARALLELISM} (${LOCAL_VLLM_REQUESTS_PER_SERVER} request(s)/server)"
echo "Stage batch defaults: stage1=${STAGE1_BATCH_SIZE} stage2=${STAGE2_BATCH_SIZE} stage3=${STAGE3_BATCH_SIZE}"

exec bash "${BASE_STAGE_SCRIPT}" "$@"
