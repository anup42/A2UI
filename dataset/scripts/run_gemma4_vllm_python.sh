#!/usr/bin/env bash
set -euo pipefail

# Start Gemma4 31B through vLLM from a normal Python virtualenv.
# No container runtime is used. The main model and assistant draft model can be
# supplied directly or discovered from GEMMA4_MODEL_ROOT.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ENV_DIR="${ENV_DIR:-${REPO_ROOT}/gemma4_vllm_env}"
if [[ -f "${ENV_DIR}/bin/activate" && -z "${VIRTUAL_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "${ENV_DIR}/bin/activate"
fi

GEMMA4_MODEL_ROOT="${GEMMA4_MODEL_ROOT:-}"
GEMMA4_MODEL_PATH="${GEMMA4_MODEL_PATH:-}"
GEMMA4_ASSISTANT_MODEL_PATH="${GEMMA4_ASSISTANT_MODEL_PATH:-}"
GEMMA4_MODEL_ID="${GEMMA4_MODEL_ID:-google/gemma-4-31B-it}"

VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-1}"
export VLLM_USE_FLASHINFER_SAMPLER
VLLM_HAS_FLASHINFER_CUBIN="${VLLM_HAS_FLASHINFER_CUBIN:-1}"
export VLLM_HAS_FLASHINFER_CUBIN
A2UI_PREFER_PYTHON_CUDA="${A2UI_PREFER_PYTHON_CUDA:-1}"

GEMMA4_ENABLE_REASONING="${GEMMA4_ENABLE_REASONING:-0}"
GEMMA4_REASONING_FLAGS_MODE="${GEMMA4_REASONING_FLAGS_MODE:-parser}" # parser|full|off
GEMMA4_REASONING_CHAT_TEMPLATE="${GEMMA4_REASONING_CHAT_TEMPLATE:-}"
GEMMA4_ENABLE_DEFAULT_THINKING="${GEMMA4_ENABLE_DEFAULT_THINKING:-0}"

GEMMA4_SPECULATIVE_MODE="${GEMMA4_SPECULATIVE_MODE:-draft}" # draft|off
GEMMA4_SPECULATIVE_TOKENS="${GEMMA4_SPECULATIVE_TOKENS:-4}"
GEMMA4_REQUIRE_SPECULATIVE="${GEMMA4_REQUIRE_SPECULATIVE:-1}"

A2UI_DISABLE_SSL_VERIFY="${A2UI_DISABLE_SSL_VERIFY:-1}"
if [[ "${A2UI_DISABLE_SSL_VERIFY}" = "1" ]]; then
  export CURL_CA_BUNDLE=""
  export REQUESTS_CA_BUNDLE=""
  export SSL_CERT_FILE=""
  export PYTHONHTTPSVERIFY=0
  export GIT_SSL_NO_VERIFY=1
  export HF_HUB_DISABLE_SSL_VERIFICATION=1
fi

export_nvidia_python_libs() {
  local py_lib_dir
  py_lib_dir="$(python - <<'PY' 2>/dev/null || true
import site
paths = site.getsitepackages()
print(paths[0] if paths else "")
PY
)"
  local joined=""
  local bins=""
  local python_cuda_home=""
  local path
  if [[ -n "${py_lib_dir}" ]]; then
    python_cuda_home="${py_lib_dir}/nvidia/cuda_nvcc"
    while IFS= read -r path; do
      if [[ -d "${path}" ]]; then
        if [[ -z "${joined}" ]]; then
          joined="${path}"
        else
          joined="${joined}:${path}"
        fi
      fi
    done < <(find "${py_lib_dir}/nvidia" -type d -name lib 2>/dev/null | sort || true)
    while IFS= read -r path; do
      if [[ -d "${path}" ]]; then
        if [[ -z "${bins}" ]]; then
          bins="${path}"
        else
          bins="${bins}:${path}"
        fi
      fi
    done < <(find "${py_lib_dir}/nvidia" -type d -name bin 2>/dev/null | sort || true)
  fi
  for path in /usr/local/cuda-13.0/lib64 /usr/local/cuda-13.1/lib64 /usr/local/cuda-13.2/lib64 /usr/local/cuda-13.3/lib64 /usr/local/cuda-13/lib64 /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu; do
    if [[ -d "${path}" ]]; then
      if [[ -z "${joined}" ]]; then
        joined="${path}"
      else
        joined="${joined}:${path}"
      fi
    fi
  done
  if [[ -n "${joined}" ]]; then
    export LD_LIBRARY_PATH="${joined}:${LD_LIBRARY_PATH:-}"
  fi
  for path in /usr/local/cuda-13.0/bin /usr/local/cuda-13.1/bin /usr/local/cuda-13.2/bin /usr/local/cuda-13.3/bin /usr/local/cuda-13/bin /usr/local/cuda/bin; do
    if [[ -d "${path}" ]]; then
      if [[ -z "${bins}" ]]; then
        bins="${path}"
      else
        bins="${bins}:${path}"
      fi
    fi
  done
  if [[ -n "${bins}" ]]; then
    export PATH="${bins}:${PATH}"
  fi
  if [[ "${A2UI_PREFER_PYTHON_CUDA}" = "1" && -n "${python_cuda_home}" && -d "${python_cuda_home}" ]]; then
    export CUDA_HOME="${python_cuda_home}"
    export CUDA_PATH="${python_cuda_home}"
  elif [[ -z "${CUDA_HOME:-}" ]]; then
    for path in /usr/local/cuda-13.0 /usr/local/cuda-13.1 /usr/local/cuda-13.2 /usr/local/cuda-13.3 /usr/local/cuda-13 /usr/local/cuda; do
      if [[ -d "${path}" ]]; then
        export CUDA_HOME="${path}"
        export CUDA_PATH="${path}"
        break
      fi
    done
  else
    export CUDA_PATH="${CUDA_HOME}"
  fi
}
export_nvidia_python_libs

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
    if [[ -d "${root}/${name}" ]]; then
      printf '%s\n' "${root}/${name}"
      return 0
    fi
  done
  return 1
}

resolve_target_model() {
  if is_dir "${GEMMA4_MODEL_PATH}"; then
    printf '%s\n' "${GEMMA4_MODEL_PATH}"
    return 0
  fi
  if is_dir "${GEMMA4_MODEL_ROOT}"; then
    if [[ -f "${GEMMA4_MODEL_ROOT}/config.json" ]]; then
      printf '%s\n' "${GEMMA4_MODEL_ROOT}"
      return 0
    fi
    find_child_dir "${GEMMA4_MODEL_ROOT}" \
      gemma-4-31B-it \
      gemma-4-31b-it \
      google--gemma-4-31B-it \
      google--gemma-4-31b-it \
      gemma4-31b \
      gemma4-31b-it \
      Gemma-4-31B-it \
      Gemma4-31B-it && return 0
  fi
  return 1
}

resolve_assistant_model() {
  if is_dir "${GEMMA4_ASSISTANT_MODEL_PATH}"; then
    printf '%s\n' "${GEMMA4_ASSISTANT_MODEL_PATH}"
    return 0
  fi
  if is_dir "${GEMMA4_MODEL_ROOT}"; then
    find_child_dir "${GEMMA4_MODEL_ROOT}" \
      gemma-4-31B-it-assistant \
      gemma-4-31b-it-assistant \
      google--gemma-4-31B-it-assistant \
      google--gemma-4-31b-it-assistant \
      gemma4-31b-assistant \
      gemma4-31b-it-assistant \
      Gemma-4-31B-it-assistant \
      Gemma4-31B-it-assistant && return 0
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
    ids="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' | paste -sd, - || true)"
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
  echo "Gemma4 target model not found." >&2
  echo "Set GEMMA4_MODEL_ROOT to the parent folder containing gemma-4-31B-it and gemma-4-31B-it-assistant, or set GEMMA4_MODEL_PATH directly." >&2
  exit 1
fi

ASSISTANT_MODEL_PATH="$(resolve_assistant_model || true)"
if [[ "${GEMMA4_SPECULATIVE_MODE}" != "off" && -z "${ASSISTANT_MODEL_PATH}" ]]; then
  echo "Gemma4 assistant model not found, but GEMMA4_SPECULATIVE_MODE=${GEMMA4_SPECULATIVE_MODE}." >&2
  echo "Set GEMMA4_ASSISTANT_MODEL_PATH or set GEMMA4_SPECULATIVE_MODE=off." >&2
  exit 1
fi

detect_gpu_layout

HELP_TEXT="$(vllm serve --help 2>&1 || true)"
if grep -q "libcudart.so.13" <<<"${HELP_TEXT}"; then
  echo "vLLM failed to load libcudart.so.13." >&2
  echo "Fix: rerun setup so the CUDA 13 runtime wheel is installed and LD_LIBRARY_PATH is exported:" >&2
  echo "  bash dataset/scripts/setup_gemma4_vllm_python_env.sh" >&2
  echo "Current LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" >&2
  exit 1
fi
if [[ "${VLLM_USE_FLASHINFER_SAMPLER}" != "0" ]] && ! command -v ninja >/dev/null 2>&1; then
  echo "VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER}, but ninja is not on PATH." >&2
  echo "Rerun setup to install ninja/cmake and FlashInfer cubins:" >&2
  echo "  bash dataset/scripts/setup_gemma4_vllm_python_env.sh" >&2
  exit 1
fi
if [[ "${GEMMA4_SPECULATIVE_MODE}" != "off" ]] && ! grep -q -- "--speculative-config" <<<"${HELP_TEXT}"; then
  echo "This vLLM install does not expose --speculative-config." >&2
  if is_truthy "${GEMMA4_REQUIRE_SPECULATIVE}"; then
    echo "Install a newer vLLM nightly/source build or set GEMMA4_REQUIRE_SPECULATIVE=0 GEMMA4_SPECULATIVE_MODE=off." >&2
    exit 1
  fi
fi
if is_truthy "${GEMMA4_ENABLE_REASONING}" && ! grep -q -- "--reasoning-parser" <<<"${HELP_TEXT}"; then
  echo "This vLLM install does not expose --reasoning-parser, so Gemma4 reasoning cannot be enabled." >&2
  exit 1
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
)

if [[ -n "${VLLM_KV_CACHE_DTYPE}" ]]; then
  cmd+=(--kv-cache-dtype "${VLLM_KV_CACHE_DTYPE}")
fi

if [[ "${GEMMA4_SPECULATIVE_MODE}" != "off" ]]; then
  spec_json="$(python - "${ASSISTANT_MODEL_PATH}" "${GEMMA4_SPECULATIVE_TOKENS}" <<'PY'
import json
import sys
print(json.dumps({
    "model": sys.argv[1],
    "num_speculative_tokens": int(sys.argv[2]),
}))
PY
)"
  cmd+=(--speculative-config "${spec_json}")
fi

if is_truthy "${GEMMA4_ENABLE_REASONING}" && [[ "${GEMMA4_REASONING_FLAGS_MODE}" != "off" ]]; then
  if grep -q -- "--enable-reasoning" <<<"${HELP_TEXT}"; then
    cmd+=(--enable-reasoning)
  fi
  cmd+=(--reasoning-parser gemma4)
  if is_truthy "${GEMMA4_ENABLE_DEFAULT_THINKING}"; then
    cmd+=(--default-chat-template-kwargs '{"enable_thinking": true}')
  fi
  if [[ "${GEMMA4_REASONING_FLAGS_MODE}" = "full" ]]; then
    cmd+=(--enable-auto-tool-choice --tool-call-parser gemma4)
    if [[ -n "${GEMMA4_REASONING_CHAT_TEMPLATE}" ]]; then
      cmd+=(--chat-template "${GEMMA4_REASONING_CHAT_TEMPLATE}")
    else
      echo "Warning: GEMMA4_REASONING_FLAGS_MODE=full without GEMMA4_REASONING_CHAT_TEMPLATE; using model default chat template." >&2
    fi
  fi
fi

if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${VLLM_EXTRA_ARGS} )
  cmd+=("${extra[@]}")
fi

echo "Starting Gemma4 vLLM from Python env"
echo "  model: ${TARGET_MODEL_PATH}"
echo "  assistant: ${ASSISTANT_MODEL_PATH:-disabled}"
echo "  served model: ${GEMMA4_MODEL_ID}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  tensor_parallel_size=${A2UI_VLLM_GPUS}"
echo "  max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS}"
echo "  reasoning=${GEMMA4_ENABLE_REASONING} (${GEMMA4_REASONING_FLAGS_MODE})"
echo "  speculative=${GEMMA4_SPECULATIVE_MODE} tokens=${GEMMA4_SPECULATIVE_TOKENS}"
echo "  VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER}"
echo "  VLLM_HAS_FLASHINFER_CUBIN=${VLLM_HAS_FLASHINFER_CUBIN}"
echo "  CUDA_HOME=${CUDA_HOME:-}"
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
