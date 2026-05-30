#!/usr/bin/env bash
set -euo pipefail

# Start Qwen3.6 35B A3B through vLLM directly from a Python virtualenv.
# No A2UI model-config compatibility wrapper is used. Use
# setup_qwen36_vllm_python_env.sh to install a vLLM version that supports
# Qwen3.6 natively.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ENV_DIR="${ENV_DIR:-${REPO_ROOT}/qwen36_vllm_env}"
if [[ -f "${ENV_DIR}/activate_qwen36_vllm.sh" && -z "${VIRTUAL_ENV:-}" ]]; then
  # shellcheck source=/dev/null
  source "${ENV_DIR}/activate_qwen36_vllm.sh"
elif [[ -f "${ENV_DIR}/bin/activate" && -z "${VIRTUAL_ENV:-}" ]]; then
  # shellcheck source=/dev/null
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
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-}"
VLLM_CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-}"
VLLM_SWAP_SPACE="${VLLM_SWAP_SPACE:-}"
VLLM_GENERATION_CONFIG="${VLLM_GENERATION_CONFIG:-auto}"
VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-qwen3}"
QWEN36_ENABLE_REASONING="${QWEN36_ENABLE_REASONING:-1}"
VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-auto}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export FLASHINFER_DISABLE_VERSION__CHECK="${FLASHINFER_DISABLE_VERSION__CHECK:-1}"
export LOCAL_ALLOW_HTTP_ENDPOINT="${LOCAL_ALLOW_HTTP_ENDPOINT:-1}"
export LOCAL_STRICT_OFFLINE="${LOCAL_STRICT_OFFLINE:-0}"
export LOCAL_VLLM_STRIP_THINKING="${LOCAL_VLLM_STRIP_THINKING:-1}"
export LOCAL_VLLM_USE_HF_GENERATION_CONFIG="${LOCAL_VLLM_USE_HF_GENERATION_CONFIG:-1}"

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

vllm_supports_flag() {
  local flag="$1"
  vllm serve --help 2>&1 | grep -q -- "${flag}"
}

TARGET_MODEL_PATH="$(resolve_target_model || true)"
if [[ -z "${TARGET_MODEL_PATH}" ]]; then
  echo "Qwen3.6 target model not found." >&2
  echo "Set MODEL_ROOT/QWEN_MODEL_ROOT to the parent folder containing Qwen/Qwen3.6-35B-A3B, or set QWEN_MODEL_PATH directly." >&2
  exit 1
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "vLLM executable not found. Run: bash dataset/scripts/setup_qwen36_vllm_python_env.sh" >&2
  exit 1
fi

detect_gpu_layout

cmd=(
  vllm serve "${TARGET_MODEL_PATH}"
  --served-model-name "${QWEN_MODEL_ID}"
  --host "${VLLM_HOST}"
  --port "${VLLM_PORT}"
  --tensor-parallel-size "${A2UI_VLLM_GPUS}"
  --dtype "${VLLM_DTYPE}"
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
  --max-model-len "${VLLM_MAX_MODEL_LEN}"
  --trust-remote-code
)

if [[ -n "${VLLM_MAX_NUM_SEQS}" ]] && vllm_supports_flag "--max-num-seqs"; then
  cmd+=(--max-num-seqs "${VLLM_MAX_NUM_SEQS}")
fi
if [[ -n "${VLLM_MAX_NUM_BATCHED_TOKENS}" ]] && vllm_supports_flag "--max-num-batched-tokens"; then
  cmd+=(--max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}")
fi
if [[ -n "${VLLM_KV_CACHE_DTYPE}" ]] && vllm_supports_flag "--kv-cache-dtype"; then
  cmd+=(--kv-cache-dtype "${VLLM_KV_CACHE_DTYPE}")
fi
if [[ -n "${VLLM_CPU_OFFLOAD_GB}" ]] && vllm_supports_flag "--cpu-offload-gb"; then
  cmd+=(--cpu-offload-gb "${VLLM_CPU_OFFLOAD_GB}")
fi
if [[ -n "${VLLM_SWAP_SPACE}" ]] && vllm_supports_flag "--swap-space"; then
  cmd+=(--swap-space "${VLLM_SWAP_SPACE}")
fi
if [[ -n "${VLLM_GENERATION_CONFIG}" && "${VLLM_GENERATION_CONFIG}" != "none" ]] && vllm_supports_flag "--generation-config"; then
  cmd+=(--generation-config "${VLLM_GENERATION_CONFIG}")
fi
if is_truthy "${QWEN36_ENABLE_REASONING}"; then
  if vllm_supports_flag "--enable-reasoning"; then
    cmd+=(--enable-reasoning)
  fi
  if vllm_supports_flag "--reasoning-parser"; then
    cmd+=(--reasoning-parser "${VLLM_REASONING_PARSER}")
  else
    echo "Warning: this vLLM does not expose --reasoning-parser; install vllm>=0.17.0." >&2
  fi
fi
if [[ "${VLLM_DISABLE_CUSTOM_ALL_REDUCE}" != "0" && "${VLLM_DISABLE_CUSTOM_ALL_REDUCE}" != "false" ]]; then
  if [[ "${VLLM_DISABLE_CUSTOM_ALL_REDUCE}" = "1" || "${VLLM_DISABLE_CUSTOM_ALL_REDUCE}" = "true" || ( "${VLLM_DISABLE_CUSTOM_ALL_REDUCE}" = "auto" && "${A2UI_VLLM_GPUS}" -gt 2 ) ]]; then
    if vllm_supports_flag "--disable-custom-all-reduce"; then
      cmd+=(--disable-custom-all-reduce)
    fi
  fi
fi

if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${VLLM_EXTRA_ARGS} )
  cmd+=("${extra[@]}")
fi

echo "Starting Qwen3.6 vLLM directly"
echo "  model: ${TARGET_MODEL_PATH}"
echo "  served model: ${QWEN_MODEL_ID}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "  tensor_parallel_size=${A2UI_VLLM_GPUS}"
echo "  max_model_len=${VLLM_MAX_MODEL_LEN}"
echo "  generation_config=${VLLM_GENERATION_CONFIG}"
echo "  reasoning=${QWEN36_ENABLE_REASONING} parser=${VLLM_REASONING_PARSER}"
python - <<'PY'
try:
    import importlib.metadata as md
    print("  vllm_version=" + md.version("vllm"))
except Exception:
    pass
PY
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
