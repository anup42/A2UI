#!/usr/bin/env bash
set -euo pipefail

# Start Qwen3.6 35B A3B through vLLM from a normal Python virtualenv.
# The local model can be supplied directly with QWEN_MODEL_PATH, or resolved
# from MODEL_ROOT/QWEN_MODEL_ROOT. A common layout is:
#   MODEL_ROOT=/path/to/models
#   /path/to/models/Qwen/Qwen3.6-35B-A3B/config.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ENV_DIR="${ENV_DIR:-${REPO_ROOT}/qwen_vllm_env}"
if [[ -f "${ENV_DIR}/bin/activate" && -z "${VIRTUAL_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "${ENV_DIR}/bin/activate"
fi

MODEL_ROOT="${MODEL_ROOT:-${LOCAL_MODEL_ROOT:-${A2UI_MODEL_ROOT:-}}}"
QWEN_MODEL_ROOT="${QWEN_MODEL_ROOT:-${MODEL_ROOT}}"
QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3.6-35B-A3B}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-}"

VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_SWAP_SPACE="${VLLM_SWAP_SPACE:-8}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
VLLM_QUANTIZATION_MODE="${VLLM_QUANTIZATION_MODE:-none}"
VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-}"
VLLM_CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-}"
VLLM_ARCHITECTURE_OVERRIDE="${VLLM_ARCHITECTURE_OVERRIDE:-auto}"
VLLM_GENERATION_CONFIG="${VLLM_GENERATION_CONFIG:-auto}"
VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-qwen3}"
QWEN36_ENABLE_REASONING="${QWEN36_ENABLE_REASONING:-1}"

A2UI_DISABLE_SSL_VERIFY="${A2UI_DISABLE_SSL_VERIFY:-1}"
if [[ "${A2UI_DISABLE_SSL_VERIFY}" = "1" ]]; then
  export CURL_CA_BUNDLE=""
  export REQUESTS_CA_BUNDLE=""
  export SSL_CERT_FILE=""
  export PYTHONHTTPSVERIFY=0
  export GIT_SSL_NO_VERIFY=1
  export HF_HUB_DISABLE_SSL_VERIFICATION=1
fi

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

is_dir() {
  [[ -n "${1:-}" && -d "$1" ]]
}

find_child_dir() {
  local root="$1"
  shift
  local name
  for name in "$@"; do
    if [[ -d "${root}/${name}" && -f "${root}/${name}/config.json" ]]; then
      printf '%s\n' "${root}/${name}"
      return 0
    fi
  done
  return 1
}

resolve_model_from_roots() {
  local model_id="$1"
  shift
  python "${REPO_ROOT}/dataset/scripts/resolve_model_from_root.py" \
    --model "${model_id}" \
    "$@" \
    --quiet 2>/dev/null || true
}

resolve_target_model() {
  if is_dir "${QWEN_MODEL_PATH}"; then
    printf '%s\n' "${QWEN_MODEL_PATH}"
    return 0
  fi
  local mapped
  mapped="$(resolve_model_from_roots "${QWEN_MODEL_ID}" --root "${QWEN_MODEL_ROOT}")"
  if is_dir "${mapped}"; then
    printf '%s\n' "${mapped}"
    return 0
  fi
  if is_dir "${QWEN_MODEL_ROOT}"; then
    if [[ -f "${QWEN_MODEL_ROOT}/config.json" ]]; then
      printf '%s\n' "${QWEN_MODEL_ROOT}"
      return 0
    fi
    find_child_dir "${QWEN_MODEL_ROOT}" \
      Qwen/Qwen3.6-35B-A3B \
      Qwen--Qwen3.6-35B-A3B \
      Qwen3.6-35B-A3B \
      qwen3.6-35b-a3b \
      Qwen36-35B-A3B \
      qwen36-35b-a3b && return 0
  fi
  return 1
}

detect_gpu_layout() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    if [[ -z "${A2UI_VLLM_GPUS:-}" ]]; then
      A2UI_VLLM_GPUS="$(python - <<'PY'
import os
items = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
print(max(1, len(items)))
PY
)"
    fi
    return 0
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    local ids
    ids="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | paste -sd, - || true)"
    if [[ -n "${ids}" ]]; then
      export CUDA_VISIBLE_DEVICES="${ids}"
      if [[ -z "${A2UI_VLLM_GPUS:-}" ]]; then
        A2UI_VLLM_GPUS="$(python - <<'PY'
import os
items = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
print(max(1, len(items)))
PY
)"
      fi
      return 0
    fi
  fi

  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  A2UI_VLLM_GPUS="${A2UI_VLLM_GPUS:-1}"
}

TARGET_MODEL_PATH="$(resolve_target_model || true)"
if [[ -z "${TARGET_MODEL_PATH}" ]]; then
  echo "Qwen3.6 target model not found." >&2
  echo "Set MODEL_ROOT/QWEN_MODEL_ROOT to the parent folder containing Qwen/Qwen3.6-35B-A3B, or set QWEN_MODEL_PATH directly." >&2
  echo "Accepted examples under the root: Qwen/Qwen3.6-35B-A3B, Qwen--Qwen3.6-35B-A3B, Qwen3.6-35B-A3B." >&2
  exit 1
fi

detect_gpu_layout

SERVER_ARGS=(
  --model-path "${TARGET_MODEL_PATH}"
  --served-model-name "${QWEN_MODEL_ID}"
  --gpus "${A2UI_VLLM_GPUS}"
  --host "${VLLM_HOST}"
  --port "${VLLM_PORT}"
  --dtype "${VLLM_DTYPE}"
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
  --max-model-len "${VLLM_MAX_MODEL_LEN}"
  --trust-remote-code
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES:-}"
  --architecture-override "${VLLM_ARCHITECTURE_OVERRIDE}"
  --quantization-mode "${VLLM_QUANTIZATION_MODE}"
  --generation-config "${VLLM_GENERATION_CONFIG}"
)

if python "${REPO_ROOT}/dataset/scripts/serve_qwen_vllm.py" --help 2>&1 | grep -q -- "--swap-space"; then
  SERVER_ARGS+=(--swap-space "${VLLM_SWAP_SPACE}")
else
  echo "Warning: serve_qwen_vllm.py does not expose --swap-space; skipping VLLM_SWAP_SPACE=${VLLM_SWAP_SPACE}."
fi

if is_truthy "${QWEN36_ENABLE_REASONING}"; then
  SERVER_ARGS+=(--enable-reasoning --reasoning-parser "${VLLM_REASONING_PARSER}")
fi

if [[ -n "${VLLM_KV_CACHE_DTYPE}" ]]; then
  SERVER_ARGS+=(--kv-cache-dtype "${VLLM_KV_CACHE_DTYPE}")
fi

if [[ -n "${VLLM_CPU_OFFLOAD_GB}" ]]; then
  SERVER_ARGS+=(--cpu-offload-gb "${VLLM_CPU_OFFLOAD_GB}")
fi

if [[ -n "${VLLM_MAX_NUM_SEQS}" ]]; then
  SERVER_ARGS+=(--max-num-seqs "${VLLM_MAX_NUM_SEQS}")
fi

echo "Starting Qwen3.6 vLLM from Python env"
echo "  model: ${TARGET_MODEL_PATH}"
echo "  served model: ${QWEN_MODEL_ID}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "  tensor_parallel_size=${A2UI_VLLM_GPUS}"
echo "  max_model_len=${VLLM_MAX_MODEL_LEN}"
echo "  generation_config=${VLLM_GENERATION_CONFIG}"
echo "  reasoning=${QWEN36_ENABLE_REASONING} parser=${VLLM_REASONING_PARSER}"
printf 'Command: python %q' "${REPO_ROOT}/dataset/scripts/serve_qwen_vllm.py"
printf ' %q' "${SERVER_ARGS[@]}"
printf '\n'

exec python "${REPO_ROOT}/dataset/scripts/serve_qwen_vllm.py" "${SERVER_ARGS[@]}"
