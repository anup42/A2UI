#!/usr/bin/env bash
set -euo pipefail

# Cron-safe external supervisor for Gemma4 vLLM.
#
# Intended cron entry:
#   * * * * * cd /path/to/A2UI && bash dataset/scripts/watch_gemma4_vllm_when_gpu_free.sh >> /tmp/a2ui_gemma4_vllm_watch_cron.log 2>&1
#
# Behavior:
#   - exits if the vLLM /v1/models endpoint is already reachable
#   - exits if run_gemma4_vllm_python.sh is already alive or starting
#   - exits if selected GPUs are busy
#   - starts run_gemma4_vllm_python.sh in the background when GPUs are free
#   - uses a lock so overlapping cron invocations cannot start duplicates

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

WATCH_VLLM_ENV_FILE="${WATCH_VLLM_ENV_FILE:-}"
if [[ -n "${WATCH_VLLM_ENV_FILE}" && -f "${WATCH_VLLM_ENV_FILE}" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${WATCH_VLLM_ENV_FILE}"
  set +a
fi

WATCH_VLLM_SCRIPT="${WATCH_VLLM_SCRIPT:-${SCRIPT_DIR}/run_gemma4_vllm_python.sh}"
WATCH_VLLM_LOCK_FILE="${WATCH_VLLM_LOCK_FILE:-/tmp/a2ui_gemma4_vllm_watch.lock}"
WATCH_VLLM_PID_FILE="${WATCH_VLLM_PID_FILE:-/tmp/a2ui_gemma4_vllm_python.pid}"
WATCH_VLLM_START_LOG="${WATCH_VLLM_START_LOG:-/tmp/a2ui_gemma4_vllm_supervisor.log}"
WATCH_VLLM_PORT="${WATCH_VLLM_PORT:-${VLLM_PORT:-8000}}"
WATCH_VLLM_MODELS_URL="${WATCH_VLLM_MODELS_URL:-${LOCAL_VLLM_MODELS_URL:-http://127.0.0.1:${WATCH_VLLM_PORT}/v1/models}}"
WATCH_VLLM_GPU_IDS="${WATCH_VLLM_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}"
WATCH_VLLM_MEMORY_USED_MB_MAX="${WATCH_VLLM_MEMORY_USED_MB_MAX:-1024}"
WATCH_VLLM_UTILIZATION_MAX="${WATCH_VLLM_UTILIZATION_MAX:-5}"
WATCH_VLLM_IGNORE_COMPUTE_PIDS="${WATCH_VLLM_IGNORE_COMPUTE_PIDS:-0}"
WATCH_VLLM_ALLOW_START_WITH_BUSY_PORT="${WATCH_VLLM_ALLOW_START_WITH_BUSY_PORT:-0}"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

acquire_lock() {
  mkdir -p "$(dirname "${WATCH_VLLM_LOCK_FILE}")"
  if command -v flock >/dev/null 2>&1; then
    exec 9>"${WATCH_VLLM_LOCK_FILE}"
    if ! flock -n 9; then
      log "watcher already running; exiting"
      exit 0
    fi
  else
    WATCH_VLLM_LOCK_DIR="${WATCH_VLLM_LOCK_FILE}.d"
    if ! mkdir "${WATCH_VLLM_LOCK_DIR}" 2>/dev/null; then
      log "watcher already running; exiting"
      exit 0
    fi
    trap 'rmdir "${WATCH_VLLM_LOCK_DIR}" 2>/dev/null || true' EXIT
  fi
}

model_endpoint_ready() {
  if command -v curl >/dev/null 2>&1; then
    curl -fsS --max-time 5 "${WATCH_VLLM_MODELS_URL}" >/dev/null 2>&1
    return $?
  fi
  python - "${WATCH_VLLM_MODELS_URL}" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1], timeout=5) as resp:
    resp.read(1)
PY
}

pid_alive() {
  [[ -n "${1:-}" && "${1}" =~ ^[0-9]+$ ]] && kill -0 "$1" 2>/dev/null
}

existing_pid_from_file() {
  [[ -f "${WATCH_VLLM_PID_FILE}" ]] || return 1
  local pid
  pid="$(tr -dc '0-9' < "${WATCH_VLLM_PID_FILE}" 2>/dev/null || true)"
  if pid_alive "${pid}"; then
    printf '%s\n' "${pid}"
    return 0
  fi
  rm -f "${WATCH_VLLM_PID_FILE}" 2>/dev/null || true
  return 1
}

existing_script_pids() {
  local current_user script_name
  current_user="${USER:-$(id -un 2>/dev/null || true)}"
  script_name="$(basename "${WATCH_VLLM_SCRIPT}")"
  ps -u "${current_user}" -o pid=,args= 2>/dev/null \
    | awk -v self="$$" -v script="${script_name}" '$1 != self && index($0, script) {print $1}' \
    | sort -u
}

port_listener_pids() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -ti "TCP:${WATCH_VLLM_PORT}" -sTCP:LISTEN 2>/dev/null | sort -u || true
  elif command -v fuser >/dev/null 2>&1; then
    fuser -n tcp "${WATCH_VLLM_PORT}" 2>/dev/null | tr ' ' '\n' | awk 'NF' | sort -u || true
  fi
}

resolve_gpu_ids() {
  if [[ -n "${WATCH_VLLM_GPU_IDS}" ]]; then
    printf '%s\n' "${WATCH_VLLM_GPU_IDS}" | tr ',;' ' ' | xargs
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null \
      | tr -d ' ' \
      | paste -sd' ' -
    return 0
  fi
  return 1
}

gpu_busy_reason() {
  local gpu="$1"
  local memory_used utilization pids
  memory_used="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || true)"
  utilization="$(nvidia-smi -i "${gpu}" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || true)"
  pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' ' \
    | awk 'NF && $0 ~ /^[0-9]+$/ {print $0}' \
    | sort -u \
    | paste -sd, - || true)"

  if [[ -n "${pids}" ]] && ! is_truthy "${WATCH_VLLM_IGNORE_COMPUTE_PIDS}"; then
    printf 'gpu=%s compute_pids=%s' "${gpu}" "${pids}"
    return 0
  fi
  if [[ -n "${memory_used}" && "${memory_used}" =~ ^[0-9]+$ && "${memory_used}" -gt "${WATCH_VLLM_MEMORY_USED_MB_MAX}" ]]; then
    printf 'gpu=%s memory_used_mb=%s > %s' "${gpu}" "${memory_used}" "${WATCH_VLLM_MEMORY_USED_MB_MAX}"
    return 0
  fi
  if [[ -n "${utilization}" && "${utilization}" =~ ^[0-9]+$ && "${utilization}" -gt "${WATCH_VLLM_UTILIZATION_MAX}" ]]; then
    printf 'gpu=%s utilization_pct=%s > %s' "${gpu}" "${utilization}" "${WATCH_VLLM_UTILIZATION_MAX}"
    return 0
  fi
  return 1
}

gpus_are_free() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi not found; cannot verify GPU availability"
    return 1
  fi

  local gpu_ids gpu reason busy=0
  gpu_ids="$(resolve_gpu_ids || true)"
  if [[ -z "${gpu_ids}" ]]; then
    log "no GPU ids resolved; set WATCH_VLLM_GPU_IDS or CUDA_VISIBLE_DEVICES"
    return 1
  fi

  for gpu in ${gpu_ids}; do
    if reason="$(gpu_busy_reason "${gpu}")"; then
      log "GPU busy: ${reason}"
      busy=1
    fi
  done
  [[ "${busy}" = "0" ]]
}

start_vllm_supervisor() {
  if [[ ! -f "${WATCH_VLLM_SCRIPT}" ]]; then
    log "vLLM script not found: ${WATCH_VLLM_SCRIPT}"
    return 1
  fi
  mkdir -p "$(dirname "${WATCH_VLLM_PID_FILE}")" "$(dirname "${WATCH_VLLM_START_LOG}")"
  log "starting vLLM supervisor: ${WATCH_VLLM_SCRIPT}"
  nohup bash "${WATCH_VLLM_SCRIPT}" >> "${WATCH_VLLM_START_LOG}" 2>&1 &
  local pid="$!"
  printf '%s\n' "${pid}" > "${WATCH_VLLM_PID_FILE}"
  log "started vLLM supervisor pid=${pid}; log=${WATCH_VLLM_START_LOG}"
}

main() {
  acquire_lock

  if model_endpoint_ready; then
    log "vLLM endpoint already reachable: ${WATCH_VLLM_MODELS_URL}"
    return 0
  fi

  local pid
  if pid="$(existing_pid_from_file)"; then
    log "vLLM supervisor already running from pid file: pid=${pid}"
    return 0
  fi

  local existing
  existing="$(existing_script_pids | paste -sd, - || true)"
  if [[ -n "${existing}" ]]; then
    log "vLLM supervisor script already running: pids=${existing}"
    return 0
  fi

  local port_pids
  port_pids="$(port_listener_pids | paste -sd, - || true)"
  if [[ -n "${port_pids}" ]] && ! is_truthy "${WATCH_VLLM_ALLOW_START_WITH_BUSY_PORT}"; then
    log "port ${WATCH_VLLM_PORT} already has listener pids=${port_pids}; not starting duplicate server"
    return 0
  fi

  if ! gpus_are_free; then
    log "selected GPUs are not free; not starting vLLM"
    return 0
  fi

  start_vllm_supervisor
}

main "$@"
