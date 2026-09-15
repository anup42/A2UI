#!/usr/bin/env bash
set -euo pipefail

# Start independent Gemma4 vLLM replicas, optionally with multiple GPUs each.
#
# This script intentionally reuses run_gemma4_vllm_direct_restart.sh so the
# single-server behavior stays identical: same model resolution, reasoning,
# speculative decoding, sampling defaults, stale-process cleanup, and restart
# policy. REPLICA_TP_SIZE selects GPUs per replica (default 1).
#
# Example:
#   MODEL_ROOT=/path/to/models \
#   VLLM_BASE_PORT=8000 \
#   bash dataset/scripts/run_gemma4_vllm_replicas.sh
#
# Then, in another shell:
#   source /tmp/a2ui_gemma4_vllm_replicas.env
#   RUN_ID=dataset_gemma4_multi STAGE=cyclic MAX_GENERATION_TOTAL=50000 \
#     bash dataset/scripts/run_gemma4_dataset_stages_multi_vllm.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SERVER_SCRIPT="${A2UI_VLLM_SERVER_SCRIPT:-${SCRIPT_DIR}/run_gemma4_vllm_direct_restart.sh}"
if [[ ! -f "${SERVER_SCRIPT}" ]]; then
  echo "Server script not found: ${SERVER_SCRIPT}" >&2
  exit 1
fi

VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_BASE_PORT="${VLLM_BASE_PORT:-8000}"
REPLICA_GPU_IDS="${REPLICA_GPU_IDS:-${A2UI_VLLM_REPLICA_GPUS:-}}"
REPLICA_LOG_DIR="${REPLICA_LOG_DIR:-/tmp/a2ui_gemma4_vllm_replicas}"
REPLICA_ENV_FILE="${REPLICA_ENV_FILE:-/tmp/a2ui_gemma4_vllm_replicas.env}"
REPLICA_START_DELAY_SECONDS="${REPLICA_START_DELAY_SECONDS:-5}"
REPLICA_TP_SIZE="${REPLICA_TP_SIZE:-1}"
REPLICA_STOP_GRACE_SECONDS="${REPLICA_STOP_GRACE_SECONDS:-10}"
if ! [[ "${REPLICA_STOP_GRACE_SECONDS}" =~ ^[1-9][0-9]*$ ]] || (( REPLICA_STOP_GRACE_SECONDS < 8 )); then
  echo 'REPLICA_STOP_GRACE_SECONDS must be at least 8 so child supervisors can stop their workers.' >&2
  exit 2
fi

detect_gpu_ids() {
  if [[ -n "${REPLICA_GPU_IDS}" ]]; then
    printf '%s\n' "${REPLICA_GPU_IDS}" | tr ',;' ' ' | xargs
    return 0
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    printf '%s\n' "${CUDA_VISIBLE_DEVICES}" | tr ',;' ' ' | xargs
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits \
      | tr -d ' ' \
      | paste -sd' ' -
    return 0
  fi
  printf '0\n'
}

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

mkdir -p "${REPLICA_LOG_DIR}"

read -r -a GPU_IDS <<<"$(detect_gpu_ids)"
if (( ${#GPU_IDS[@]} == 0 )); then
  echo "No GPUs detected. Set REPLICA_GPU_IDS=0,1 or CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi
if ! [[ "${REPLICA_TP_SIZE}" =~ ^[1-9][0-9]*$ ]] || (( ${#GPU_IDS[@]} % REPLICA_TP_SIZE != 0 )); then
  echo "REPLICA_TP_SIZE must divide the number of selected GPUs exactly." >&2
  exit 2
fi
declare -A SEEN_GPU_IDS=()
for gpu in "${GPU_IDS[@]}"; do
  if [[ -n "${SEEN_GPU_IDS[${gpu}]:-}" ]]; then
    echo "Duplicate GPU in replica assignment: ${gpu}" >&2
    exit 2
  fi
  SEEN_GPU_IDS["${gpu}"]=1
done
replica_count=$(( ${#GPU_IDS[@]} / REPLICA_TP_SIZE ))

declare -a CHILD_PIDS=()
declare -a ENDPOINTS=()
declare -a PORTS=()

stop_children() {
  local pid
  if (( ${#CHILD_PIDS[@]} > 0 )); then
    echo "Stopping vLLM replica supervisors: ${CHILD_PIDS[*]}"
    for pid in "${CHILD_PIDS[@]}"; do
      kill -TERM "${pid}" 2>/dev/null || true
    done
    # The Python supervisor waits five seconds before killing its worker group.
    # Do not kill that supervisor before it can finish the escalation.
    for (( elapsed=0; elapsed<REPLICA_STOP_GRACE_SECONDS; elapsed++ )); do
      alive=0
      for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then alive=1; fi
      done
      (( alive == 0 )) && break
      sleep 1
    done
    for pid in "${CHILD_PIDS[@]}"; do
      kill -KILL "${pid}" 2>/dev/null || true
    done
  fi
}
trap stop_children INT TERM EXIT

echo "Starting ${replica_count} Gemma4 vLLM replicas, TP=${REPLICA_TP_SIZE} each"
echo "  GPUs: ${GPU_IDS[*]}"
echo "  base_port: ${VLLM_BASE_PORT}"
echo "  logs: ${REPLICA_LOG_DIR}"

for (( idx=0; idx<replica_count; idx++ )); do
  group=("${GPU_IDS[@]:idx*REPLICA_TP_SIZE:REPLICA_TP_SIZE}")
  gpu="$(IFS=,; echo "${group[*]}")"
  port=$(( VLLM_BASE_PORT + idx ))
  endpoint="http://127.0.0.1:${port}/v1/chat/completions"
  ENDPOINTS+=("${endpoint}")
  PORTS+=("${port}")

  log_path="${REPLICA_LOG_DIR}/vllm_gpu${gpu}_port${port}.log"
  echo "Launching replica gpu=${gpu} port=${port} log=${log_path}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export A2UI_VLLM_GPUS="${REPLICA_TP_SIZE}"
    export VLLM_HOST="${VLLM_HOST}"
    export VLLM_PORT="${port}"
    export VLLM_RUN_LOG="${log_path}"
    export VLLM_CLEAN_STALE_SCOPE="${VLLM_CLEAN_STALE_SCOPE:-gpu}"
    exec bash "${SERVER_SCRIPT}"
  ) &
  CHILD_PIDS+=("$!")
  sleep "${REPLICA_START_DELAY_SECONDS}"
done

joined_endpoints="$(IFS=,; echo "${ENDPOINTS[*]}")"
joined_ports="$(IFS=,; echo "${PORTS[*]}")"
first_models_url="${ENDPOINTS[0]%/chat/completions}/models"
cat > "${REPLICA_ENV_FILE}" <<EOF
export LOCAL_VLLM_ENDPOINTS="${joined_endpoints}"
export LOCAL_VLLM_MODELS_URL="${first_models_url}"
export VLLM_PORTS="${joined_ports}"
EOF

echo "Replica endpoints:"
printf '  %s\n' "${ENDPOINTS[@]}"
echo "Environment file written: ${REPLICA_ENV_FILE}"
echo "Use all replicas with:"
echo "  source ${REPLICA_ENV_FILE}"
echo "  bash dataset/scripts/run_gemma4_dataset_stages_multi_vllm.sh"

if is_truthy "${REPLICA_WAIT_FOREVER:-1}"; then
  wait
fi
