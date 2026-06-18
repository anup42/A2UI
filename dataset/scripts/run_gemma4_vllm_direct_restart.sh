#!/usr/bin/env bash
set -euo pipefail

# Simple direct vLLM serve wrapper for Gemma4 31B.
#
# Defaults:
#   - uses all GPUs visible to the current process
#   - tensor parallel size = number of visible GPUs
#   - reasoning parser/config enabled
#   - speculative decoding enabled with assistant model and 4 tokens
#   - infinite restart on crash
#
# Example:
#   ENV_DIR=$PWD/vllm_cu131_py312_env \
#   MODEL_ROOT=/path/to/models \
#   bash dataset/scripts/run_gemma4_vllm_direct_restart.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ENV_DIR="${ENV_DIR:-}"
if [[ -n "${ENV_DIR}" && -f "${ENV_DIR}/bin/activate" && -z "${VIRTUAL_ENV:-}" ]]; then
  # shellcheck source=/dev/null
  source "${ENV_DIR}/bin/activate"
  hash -r
fi

MODEL_ROOT="${MODEL_ROOT:-${LOCAL_MODEL_ROOT:-${A2UI_MODEL_ROOT:-${GEMMA4_MODEL_ROOT:-}}}}"
GEMMA4_MODEL_ID="${GEMMA4_MODEL_ID:-google/gemma-4-31b-it}"
GEMMA4_ASSISTANT_MODEL_ID="${GEMMA4_ASSISTANT_MODEL_ID:-google/gemma-4-31b-it-assistant}"
GEMMA4_MODEL_PATH="${GEMMA4_MODEL_PATH:-}"
GEMMA4_ASSISTANT_MODEL_PATH="${GEMMA4_ASSISTANT_MODEL_PATH:-}"

VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.98}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
GEMMA4_SPECULATIVE_TOKENS="${GEMMA4_SPECULATIVE_TOKENS:-4}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

# Dataset client sampling defaults. These are request-time settings consumed by
# dataset/src/llm/local_adapter.py, not vLLM server flags.
export A2UI_QUERY_TEMPERATURE="${A2UI_QUERY_TEMPERATURE:-1.0}"
export A2UI_RESPONSE_TEMPERATURES="${A2UI_RESPONSE_TEMPERATURES:-1.0}"
export A2UI_GENUI_TEMPERATURE="${A2UI_GENUI_TEMPERATURE:-0.7}"
export A2UI_GENUI_REPAIR_TEMPERATURE="${A2UI_GENUI_REPAIR_TEMPERATURE:-0.2}"
export A2UI_GENUI_FINAL_REGEN_TEMPERATURE="${A2UI_GENUI_FINAL_REGEN_TEMPERATURE:-0.7}"
export LOCAL_VLLM_TOP_P="${LOCAL_VLLM_TOP_P:-0.95}"
export LOCAL_VLLM_TOP_K="${LOCAL_VLLM_TOP_K:-64}"
export LOCAL_VLLM_REPETITION_PENALTY="${LOCAL_VLLM_REPETITION_PENALTY:-1.0}"

# Optional server-side default generation config for direct curl/manual calls
# that do not pass temperature/top_p/top_k in the request body.
VLLM_GENERATION_CONFIG="${VLLM_GENERATION_CONFIG:-a2ui}"
VLLM_DEFAULT_TEMPERATURE="${VLLM_DEFAULT_TEMPERATURE:-1.0}"
VLLM_DEFAULT_TOP_P="${VLLM_DEFAULT_TOP_P:-${LOCAL_VLLM_TOP_P}}"
VLLM_DEFAULT_TOP_K="${VLLM_DEFAULT_TOP_K:-${LOCAL_VLLM_TOP_K}}"
VLLM_DEFAULT_REPETITION_PENALTY="${VLLM_DEFAULT_REPETITION_PENALTY:-${LOCAL_VLLM_REPETITION_PENALTY}}"
VLLM_GENERATION_CONFIG_DIR="${VLLM_GENERATION_CONFIG_DIR:-/tmp/a2ui_gemma4_generation_config}"

VLLM_CLEAN_STALE_PROCESSES="${VLLM_CLEAN_STALE_PROCESSES:-1}"
# all: clean same-user vLLM processes on all GPUs plus the configured port.
# gpu: clean same-user vLLM processes only on CUDA_VISIBLE_DEVICES plus the configured port.
# port: clean only the configured port listener.
VLLM_CLEAN_STALE_SCOPE="${VLLM_CLEAN_STALE_SCOPE:-all}"
VLLM_CLEAN_STALE_FORCE_AFTER_SECONDS="${VLLM_CLEAN_STALE_FORCE_AFTER_SECONDS:-10}"
VLLM_RESTART_ON_CRASH="${VLLM_RESTART_ON_CRASH:-1}"
# 0 means infinite restarts.
VLLM_MAX_RESTARTS="${VLLM_MAX_RESTARTS:-0}"
VLLM_RESTART_BACKOFF_SECONDS="${VLLM_RESTART_BACKOFF_SECONDS:-15}"
VLLM_RUN_LOG="${VLLM_RUN_LOG:-/tmp/a2ui_gemma4_vllm_direct_restart.log}"

export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export FLASHINFER_DISABLE_VERSION__CHECK="${FLASHINFER_DISABLE_VERSION__CHECK:-1}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

vllm_supports_flag() {
  local flag="$1"
  vllm serve --help 2>&1 | grep -q -- "${flag}"
}

resolve_model() {
  local explicit_path="$1"
  local model_id="$2"
  if [[ -n "${explicit_path}" && -d "${explicit_path}" ]]; then
    printf '%s\n' "${explicit_path}"
    return 0
  fi
  if [[ -n "${MODEL_ROOT}" ]]; then
    python "${REPO_ROOT}/dataset/scripts/resolve_model_from_root.py" \
      --model "${model_id}" \
      --root "${MODEL_ROOT}" \
      --quiet 2>/dev/null || true
  fi
}

detect_gpu_layout() {
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr -d ' ' | paste -sd, -)"
      export CUDA_VISIBLE_DEVICES
    fi
  fi
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="0"
  fi
  if [[ -z "${A2UI_VLLM_GPUS:-}" ]]; then
    A2UI_VLLM_GPUS="$(python - <<'PY'
import os
items = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
print(max(1, len(items)))
PY
)"
    export A2UI_VLLM_GPUS
  fi
}

current_user() {
  printf '%s\n' "${USER:-$(id -un 2>/dev/null || true)}"
}

pid_owner() {
  ps -o user= -p "$1" 2>/dev/null | awk '{print $1}'
}

pid_alive() {
  kill -0 "$1" 2>/dev/null
}

user_pids_matching() {
  local pattern="$1"
  local user_name
  user_name="$(current_user)"
  ps -u "${user_name}" -o pid=,comm=,args= 2>/dev/null \
    | awk -v pat="${pattern}" -v self="$$" '$0 ~ pat && $1 != self {print $1}' \
    | sort -u
}

port_listener_pids() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -ti "TCP:${VLLM_PORT}" -sTCP:LISTEN 2>/dev/null | sort -u || true
  elif command -v fuser >/dev/null 2>&1; then
    fuser -n tcp "${VLLM_PORT}" 2>/dev/null | tr ' ' '\n' | awk 'NF' | sort -u || true
  fi
}

gpu_vllm_pids() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local pid cmdline
  local -a query_cmd=(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
  if [[ "${VLLM_CLEAN_STALE_SCOPE}" = "gpu" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local gpu
    for gpu in ${CUDA_VISIBLE_DEVICES//,/ }; do
      [[ -n "${gpu}" ]] || continue
      nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true
    done
  else
    "${query_cmd[@]}" 2>/dev/null
  fi \
    | tr -d ' ' \
    | awk 'NF && $0 ~ /^[0-9]+$/ {print $0}' \
    | sort -u \
    | while IFS= read -r pid; do
        [[ -n "${pid}" && "${pid}" != "$$" ]] || continue
        cmdline="$(ps -o comm=,args= -p "${pid}" 2>/dev/null || true)"
        if grep -Eq 'VLLM.*Worker_TP|EngineCore|vllm[[:space:]]+serve|vllm\.entrypoints\.openai|multiproc_executor' <<<"${cmdline}"; then
          printf '%s\n' "${pid}"
        fi
      done
}

terminate_pids() {
  local reason="$1"
  shift || true
  local -a selected=()
  local pid owner user_name
  user_name="$(current_user)"
  for pid in "$@"; do
    [[ -n "${pid:-}" && "${pid}" =~ ^[0-9]+$ ]] || continue
    [[ "${pid}" != "$$" ]] || continue
    pid_alive "${pid}" || continue
    owner="$(pid_owner "${pid}")"
    [[ -z "${owner}" || "${owner}" = "${user_name}" ]] || continue
    selected+=("${pid}")
  done
  if (( ${#selected[@]} == 0 )); then
    return 0
  fi
  mapfile -t selected < <(printf '%s\n' "${selected[@]}" | sort -nu)
  echo "${reason}: ${selected[*]}"
  kill -TERM "${selected[@]}" 2>/dev/null || true

  local waited=0
  while (( waited < VLLM_CLEAN_STALE_FORCE_AFTER_SECONDS )); do
    local -a still_alive=()
    for pid in "${selected[@]}"; do
      if pid_alive "${pid}"; then
        still_alive+=("${pid}")
      fi
    done
    if (( ${#still_alive[@]} == 0 )); then
      return 0
    fi
    sleep 1
    waited=$(( waited + 1 ))
  done

  local -a remaining=()
  for pid in "${selected[@]}"; do
    if pid_alive "${pid}"; then
      remaining+=("${pid}")
    fi
  done
  if (( ${#remaining[@]} > 0 )); then
    echo "Force killing remaining vLLM processes: ${remaining[*]}"
    kill -KILL "${remaining[@]}" 2>/dev/null || true
  fi
}

cleanup_stale_vllm_processes() {
  if ! is_truthy "${VLLM_CLEAN_STALE_PROCESSES}"; then
    return 0
  fi
  if [[ ! "${VLLM_CLEAN_STALE_SCOPE}" =~ ^(all|gpu|port)$ ]]; then
    echo "Unsupported VLLM_CLEAN_STALE_SCOPE=${VLLM_CLEAN_STALE_SCOPE}; use all, gpu, or port." >&2
    exit 1
  fi
  local -a pids=()
  local pid
  if [[ "${VLLM_CLEAN_STALE_SCOPE}" = "all" ]]; then
    while IFS= read -r pid; do
      [[ -n "${pid}" ]] && pids+=("${pid}")
    done < <(user_pids_matching 'VLLM.*Worker_TP|EngineCore|vllm[[:space:]]+serve|vllm\.entrypoints\.openai|multiproc_executor')
  fi
  if [[ "${VLLM_CLEAN_STALE_SCOPE}" != "port" ]]; then
    while IFS= read -r pid; do
      [[ -n "${pid}" ]] && pids+=("${pid}")
    done < <(gpu_vllm_pids)
  fi
  while IFS= read -r pid; do
    [[ -n "${pid}" ]] && pids+=("${pid}")
  done < <(port_listener_pids)

  if (( ${#pids[@]} > 0 )); then
    terminate_pids "Cleaning stale vLLM processes before start/restart" "${pids[@]}"
  fi
}

TARGET_MODEL_PATH="$(resolve_model "${GEMMA4_MODEL_PATH}" "${GEMMA4_MODEL_ID}")"
ASSISTANT_MODEL_PATH="$(resolve_model "${GEMMA4_ASSISTANT_MODEL_PATH}" "${GEMMA4_ASSISTANT_MODEL_ID}")"

if [[ -z "${TARGET_MODEL_PATH}" || ! -d "${TARGET_MODEL_PATH}" ]]; then
  echo "Gemma4 target model not found." >&2
  echo "Set GEMMA4_MODEL_PATH directly, or set MODEL_ROOT to the parent folder containing ${GEMMA4_MODEL_ID}." >&2
  exit 1
fi
if [[ -z "${ASSISTANT_MODEL_PATH}" || ! -d "${ASSISTANT_MODEL_PATH}" ]]; then
  echo "Gemma4 assistant model not found." >&2
  echo "Set GEMMA4_ASSISTANT_MODEL_PATH directly, or set MODEL_ROOT to the parent folder containing ${GEMMA4_ASSISTANT_MODEL_ID}." >&2
  exit 1
fi

detect_gpu_layout

if [[ "${VLLM_GENERATION_CONFIG}" = "a2ui" ]]; then
  mkdir -p "${VLLM_GENERATION_CONFIG_DIR}"
  python - "${VLLM_GENERATION_CONFIG_DIR}/generation_config.json" \
    "${VLLM_DEFAULT_TEMPERATURE}" \
    "${VLLM_DEFAULT_TOP_P}" \
    "${VLLM_DEFAULT_TOP_K}" \
    "${VLLM_DEFAULT_REPETITION_PENALTY}" <<'PY'
import json
import sys

path, temperature, top_p, top_k, repetition_penalty = sys.argv[1:6]
payload = {
    "do_sample": True,
    "temperature": float(temperature),
    "top_p": float(top_p),
    "top_k": int(top_k),
    "repetition_penalty": float(repetition_penalty),
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
PY
  VLLM_GENERATION_CONFIG="${VLLM_GENERATION_CONFIG_DIR}"
fi

cmd=(
  vllm serve "${TARGET_MODEL_PATH}"
  --served-model-name "${GEMMA4_MODEL_ID}"
  --host "${VLLM_HOST}"
  --port "${VLLM_PORT}"
  --tensor-parallel-size "${A2UI_VLLM_GPUS}"
  --dtype "${VLLM_DTYPE}"
  --max-model-len "${VLLM_MAX_MODEL_LEN}"
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
  --max-num-seqs "${VLLM_MAX_NUM_SEQS}"
  --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}"
  --trust-remote-code
  --reasoning-parser gemma4
  --reasoning-config '{"reasoning_start_str":"<|channel>thought","reasoning_end_str":"<channel|>"}'
  --default-chat-template-kwargs '{"enable_thinking":true}'
  --speculative-config "{\"model\":\"${ASSISTANT_MODEL_PATH}\",\"num_speculative_tokens\":${GEMMA4_SPECULATIVE_TOKENS}}"
)

if [[ -n "${VLLM_GENERATION_CONFIG}" && "${VLLM_GENERATION_CONFIG}" != "none" ]]; then
  if vllm_supports_flag "--generation-config"; then
    cmd+=(--generation-config "${VLLM_GENERATION_CONFIG}")
  else
    echo "Warning: this vLLM install does not expose --generation-config; server defaults will use model/vLLM defaults." >&2
  fi
fi

if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${VLLM_EXTRA_ARGS} )
  cmd+=("${extra[@]}")
fi

echo "Starting direct Gemma4 vLLM server"
echo "  model: ${TARGET_MODEL_PATH}"
echo "  assistant: ${ASSISTANT_MODEL_PATH}"
echo "  served model: ${GEMMA4_MODEL_ID}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  tensor_parallel_size=${A2UI_VLLM_GPUS}"
echo "  max_model_len=${VLLM_MAX_MODEL_LEN}"
echo "  gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION}"
echo "  max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS}"
echo "  speculative_tokens=${GEMMA4_SPECULATIVE_TOKENS}"
echo "  request_sampling: query_temp=${A2UI_QUERY_TEMPERATURE} response_temps=${A2UI_RESPONSE_TEMPERATURES} genui_temp=${A2UI_GENUI_TEMPERATURE} top_p=${LOCAL_VLLM_TOP_P} top_k=${LOCAL_VLLM_TOP_K} repetition_penalty=${LOCAL_VLLM_REPETITION_PENALTY}"
echo "  server_generation_config=${VLLM_GENERATION_CONFIG}"
echo "  log: ${VLLM_RUN_LOG}"
printf '  command:'
printf ' %q' "${cmd[@]}"
printf '\n'

attempt=0
while true; do
  cleanup_stale_vllm_processes
  attempt=$(( attempt + 1 ))
  echo "vLLM launch attempt ${attempt} at $(date -Is)"
  set +e
  "${cmd[@]}" 2>&1 | tee "${VLLM_RUN_LOG}"
  rc=${PIPESTATUS[0]}
  set -e
  echo "vLLM exited rc=${rc} at $(date -Is)"
  if ! is_truthy "${VLLM_RESTART_ON_CRASH}"; then
    exit "${rc}"
  fi
  if (( VLLM_MAX_RESTARTS > 0 && attempt >= VLLM_MAX_RESTARTS )); then
    echo "Reached VLLM_MAX_RESTARTS=${VLLM_MAX_RESTARTS}; exiting." >&2
    exit "${rc}"
  fi
  echo "Restarting in ${VLLM_RESTART_BACKOFF_SECONDS}s. Set VLLM_RESTART_ON_CRASH=0 to disable."
  sleep "${VLLM_RESTART_BACKOFF_SECONDS}"
done
