#!/usr/bin/env bash
set -euo pipefail

# Long-running Gemma4 vLLM watchdog for Jupyter/MLP environments without cron.
#
# This script keeps running, prints clear timestamped diagnostics every cycle,
# and invokes watch_gemma4_vllm_when_gpu_free.sh once per cycle. Use it under
# nohup/tmux/screen or as the foreground process of a scheduler job.
#
# Example:
#   cd /path/to/A2UI
#   WATCH_VLLM_ENV_FILE=/tmp/a2ui_gemma4_vllm_watch.env \
#   nohup bash dataset/scripts/run_gemma4_vllm_watch_loop.sh \
#     > /tmp/a2ui_gemma4_vllm_watch_loop.nohup 2>&1 &

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

WATCH_LOOP_INTERVAL_SECONDS="${WATCH_LOOP_INTERVAL_SECONDS:-60}"
WATCH_LOOP_LOG="${WATCH_LOOP_LOG:-/tmp/a2ui_gemma4_vllm_watch_loop.log}"
WATCH_LOOP_LOCK_FILE="${WATCH_LOOP_LOCK_FILE:-/tmp/a2ui_gemma4_vllm_watch_loop.lock}"
WATCH_LOOP_ONESHOT_SCRIPT="${WATCH_LOOP_ONESHOT_SCRIPT:-${SCRIPT_DIR}/watch_gemma4_vllm_when_gpu_free.sh}"
WATCH_VLLM_ENV_FILE="${WATCH_VLLM_ENV_FILE:-}"
WATCH_VLLM_PORT="${WATCH_VLLM_PORT:-${VLLM_PORT:-8000}}"
WATCH_VLLM_MODELS_URL="${WATCH_VLLM_MODELS_URL:-${LOCAL_VLLM_MODELS_URL:-http://127.0.0.1:${WATCH_VLLM_PORT}/v1/models}}"
WATCH_VLLM_GPU_IDS="${WATCH_VLLM_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}"
WATCH_VLLM_PID_FILE="${WATCH_VLLM_PID_FILE:-/tmp/a2ui_gemma4_vllm_python.pid}"
WATCH_VLLM_START_LOG="${WATCH_VLLM_START_LOG:-/tmp/a2ui_gemma4_vllm_supervisor.log}"
VLLM_RUN_LOG="${VLLM_RUN_LOG:-/tmp/a2ui_gemma4_vllm_server.log}"

mkdir -p "$(dirname "${WATCH_LOOP_LOG}")"
exec > >(tee -a "${WATCH_LOOP_LOG}") 2>&1

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

acquire_loop_lock() {
  mkdir -p "$(dirname "${WATCH_LOOP_LOCK_FILE}")"
  if command -v flock >/dev/null 2>&1; then
    exec 8>"${WATCH_LOOP_LOCK_FILE}"
    if ! flock -n 8; then
      log "another watch loop is already running; exiting"
      exit 0
    fi
  else
    local lock_dir="${WATCH_LOOP_LOCK_FILE}.d"
    if ! mkdir "${lock_dir}" 2>/dev/null; then
      log "another watch loop is already running; exiting"
      exit 0
    fi
    trap 'rmdir "'"${lock_dir}"'" 2>/dev/null || true' EXIT
  fi
}

load_env_file_for_loop() {
  if [[ -n "${WATCH_VLLM_ENV_FILE}" && -f "${WATCH_VLLM_ENV_FILE}" ]]; then
    log "loading env file: ${WATCH_VLLM_ENV_FILE}"
    set -a
    # shellcheck source=/dev/null
    source "${WATCH_VLLM_ENV_FILE}"
    set +a
    WATCH_VLLM_PORT="${WATCH_VLLM_PORT:-${VLLM_PORT:-8000}}"
    WATCH_VLLM_MODELS_URL="${WATCH_VLLM_MODELS_URL:-${LOCAL_VLLM_MODELS_URL:-http://127.0.0.1:${WATCH_VLLM_PORT}/v1/models}}"
    WATCH_VLLM_GPU_IDS="${WATCH_VLLM_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}"
    WATCH_VLLM_PID_FILE="${WATCH_VLLM_PID_FILE:-/tmp/a2ui_gemma4_vllm_python.pid}"
    WATCH_VLLM_START_LOG="${WATCH_VLLM_START_LOG:-/tmp/a2ui_gemma4_vllm_supervisor.log}"
    VLLM_RUN_LOG="${VLLM_RUN_LOG:-/tmp/a2ui_gemma4_vllm_server.log}"
  elif [[ -n "${WATCH_VLLM_ENV_FILE}" ]]; then
    log "env file not found: ${WATCH_VLLM_ENV_FILE}"
  fi
}

server_check() {
  log "checking vLLM endpoint: ${WATCH_VLLM_MODELS_URL}"
  if command -v curl >/dev/null 2>&1; then
    if curl -fsS --max-time 5 "${WATCH_VLLM_MODELS_URL}" >/tmp/a2ui_vllm_models_check.json 2>/tmp/a2ui_vllm_models_check.err; then
      log "endpoint reachable"
      head -c 500 /tmp/a2ui_vllm_models_check.json || true
      printf '\n'
    else
      log "endpoint not reachable"
      sed -n '1,5p' /tmp/a2ui_vllm_models_check.err 2>/dev/null || true
    fi
  else
    log "curl not found; skipping endpoint curl check"
  fi
}

pid_file_check() {
  log "pid file: ${WATCH_VLLM_PID_FILE}"
  if [[ -f "${WATCH_VLLM_PID_FILE}" ]]; then
    local pid
    pid="$(tr -dc '0-9' < "${WATCH_VLLM_PID_FILE}" 2>/dev/null || true)"
    log "pid file content: ${pid:-empty}"
    if [[ -n "${pid}" ]]; then
      ps -fp "${pid}" 2>/dev/null || log "pid ${pid} is not alive"
    fi
  else
    log "pid file missing"
  fi
}

process_check() {
  log "matching processes"
  ps -ef 2>/dev/null \
    | grep -E 'watch_gemma4_vllm_when_gpu_free|run_gemma4_vllm_watch_loop|run_gemma4_vllm_python|vllm serve|vllm\.entrypoints\.openai|EngineCore|Worker_TP|multiproc_executor' \
    | grep -v grep || log "no matching vLLM/watch processes"
}

port_check() {
  log "port listener check: ${WATCH_VLLM_PORT}"
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -i "TCP:${WATCH_VLLM_PORT}" -sTCP:LISTEN 2>/dev/null || log "no lsof listener on port ${WATCH_VLLM_PORT}"
  elif command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null | grep ":${WATCH_VLLM_PORT} " || log "no ss listener on port ${WATCH_VLLM_PORT}"
  elif command -v netstat >/dev/null 2>&1; then
    netstat -ltnp 2>/dev/null | grep ":${WATCH_VLLM_PORT} " || log "no netstat listener on port ${WATCH_VLLM_PORT}"
  else
    log "no lsof/ss/netstat available for port check"
  fi
}

gpu_check() {
  log "GPU selection: WATCH_VLLM_GPU_IDS=${WATCH_VLLM_GPU_IDS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null || true
    log "GPU compute apps"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv 2>/dev/null || log "no compute-app details available"
  else
    log "nvidia-smi not found"
  fi
}

log_tail_check() {
  log "expected supervisor log: ${WATCH_VLLM_START_LOG}"
  if [[ -f "${WATCH_VLLM_START_LOG}" ]]; then
    tail -40 "${WATCH_VLLM_START_LOG}" || true
  else
    log "supervisor log missing"
  fi

  log "expected vLLM run log: ${VLLM_RUN_LOG}"
  if [[ -f "${VLLM_RUN_LOG}" ]]; then
    tail -40 "${VLLM_RUN_LOG}" || true
  else
    log "vLLM run log missing"
  fi
}

run_oneshot_watcher() {
  if [[ ! -f "${WATCH_LOOP_ONESHOT_SCRIPT}" ]]; then
    log "one-shot watcher missing: ${WATCH_LOOP_ONESHOT_SCRIPT}"
    return 1
  fi
  log "running one-shot watcher: ${WATCH_LOOP_ONESHOT_SCRIPT}"
  set +e
  WATCH_VLLM_ENV_FILE="${WATCH_VLLM_ENV_FILE}" bash "${WATCH_LOOP_ONESHOT_SCRIPT}"
  local rc=$?
  set -e
  log "one-shot watcher rc=${rc}"
  return 0
}

cycle() {
  log "========== watch cycle start =========="
  log "host=$(hostname 2>/dev/null || true) user=${USER:-unknown} cwd=${REPO_ROOT}"
  log "loop_log=${WATCH_LOOP_LOG}"
  log "env_file=${WATCH_VLLM_ENV_FILE:-unset}"
  log "interval_seconds=${WATCH_LOOP_INTERVAL_SECONDS}"
  load_env_file_for_loop
  log "watch_models_url=${WATCH_VLLM_MODELS_URL}"
  log "watch_port=${WATCH_VLLM_PORT}"
  log "watch_pid_file=${WATCH_VLLM_PID_FILE}"
  server_check
  pid_file_check
  process_check
  port_check
  gpu_check
  run_oneshot_watcher
  log_tail_check
  log "========== watch cycle end =========="
}

main() {
  acquire_loop_lock
  log "starting Gemma4 vLLM watch loop"
  while true; do
    cycle
    log "sleeping ${WATCH_LOOP_INTERVAL_SECONDS}s"
    sleep "${WATCH_LOOP_INTERVAL_SECONDS}"
  done
}

main "$@"
