#!/usr/bin/env bash
set -euo pipefail

# Start Gemma4 31B through vLLM from a normal Python virtualenv.
# No container runtime is used. The main model and assistant draft model can be
# supplied directly or discovered from GEMMA4_MODEL_ROOT.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ENV_DIR="${ENV_DIR:-${REPO_ROOT}/gemma4_vllm_env}"
A2UI_CUDA_TOOLKIT_DIR="${A2UI_CUDA_TOOLKIT_DIR:-${ENV_DIR}/cuda_toolkit}"
if [[ -f "${ENV_DIR}/bin/activate" && -z "${VIRTUAL_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "${ENV_DIR}/bin/activate"
fi

MODEL_ROOT="${MODEL_ROOT:-${LOCAL_MODEL_ROOT:-${A2UI_MODEL_ROOT:-}}}"
GEMMA4_MODEL_ROOT="${GEMMA4_MODEL_ROOT:-${MODEL_ROOT}}"
GEMMA4_MODEL_PATH="${GEMMA4_MODEL_PATH:-}"
GEMMA4_ASSISTANT_MODEL_PATH="${GEMMA4_ASSISTANT_MODEL_PATH:-}"
GEMMA4_MODEL_ID="${GEMMA4_MODEL_ID:-google/gemma-4-31b-it}"
GEMMA4_ASSISTANT_MODEL_ID="${GEMMA4_ASSISTANT_MODEL_ID:-google/gemma-4-31b-it-assistant}"

VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
VLLM_CLEAN_STALE_PROCESSES="${VLLM_CLEAN_STALE_PROCESSES:-1}"
VLLM_CLEAN_STALE_FORCE_AFTER_SECONDS="${VLLM_CLEAN_STALE_FORCE_AFTER_SECONDS:-10}"
VLLM_RESTART_ON_CRASH="${VLLM_RESTART_ON_CRASH:-1}"
# 0 means unlimited restarts. Set a positive integer to cap crash recovery.
VLLM_MAX_RESTARTS="${VLLM_MAX_RESTARTS:-0}"
VLLM_RESTART_BACKOFF_SECONDS="${VLLM_RESTART_BACKOFF_SECONDS:-15}"
VLLM_RUN_LOG="${VLLM_RUN_LOG:-/tmp/a2ui_gemma4_vllm_server.log}"
VLLM_RESTART_ON_ENGINE_CRASH_SIGNATURE="${VLLM_RESTART_ON_ENGINE_CRASH_SIGNATURE:-1}"
VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-1}"
export VLLM_USE_FLASHINFER_SAMPLER
VLLM_HAS_FLASHINFER_CUBIN="${VLLM_HAS_FLASHINFER_CUBIN:-1}"
export VLLM_HAS_FLASHINFER_CUBIN
FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export FLASHINFER_DISABLE_VERSION_CHECK
FLASHINFER_DISABLE_VERSION__CHECK="${FLASHINFER_DISABLE_VERSION__CHECK:-1}"
export FLASHINFER_DISABLE_VERSION__CHECK
A2UI_PREFER_PYTHON_CUDA="${A2UI_PREFER_PYTHON_CUDA:-1}"

GEMMA4_ENABLE_REASONING="${GEMMA4_ENABLE_REASONING:-0}"
GEMMA4_REASONING_FLAGS_MODE="${GEMMA4_REASONING_FLAGS_MODE:-parser}" # parser|full|off
GEMMA4_REASONING_CHAT_TEMPLATE="${GEMMA4_REASONING_CHAT_TEMPLATE:-}"
GEMMA4_ENABLE_DEFAULT_THINKING="${GEMMA4_ENABLE_DEFAULT_THINKING:-0}"

GEMMA4_SPECULATIVE_MODE="${GEMMA4_SPECULATIVE_MODE:-off}" # draft|off
GEMMA4_SPECULATIVE_TOKENS="${GEMMA4_SPECULATIVE_TOKENS:-4}"
GEMMA4_REQUIRE_SPECULATIVE="${GEMMA4_REQUIRE_SPECULATIVE:-0}"

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
  if [[ "${A2UI_PREFER_PYTHON_CUDA}" = "1" && -x "${A2UI_CUDA_TOOLKIT_DIR}/bin/nvcc" ]]; then
    export CUDA_HOME="${A2UI_CUDA_TOOLKIT_DIR}"
    export CUDA_PATH="${A2UI_CUDA_TOOLKIT_DIR}"
    export CUDACXX="${A2UI_CUDA_TOOLKIT_DIR}/bin/nvcc"
    export CMAKE_CUDA_COMPILER="${A2UI_CUDA_TOOLKIT_DIR}/bin/nvcc"
    export PATH="${A2UI_CUDA_TOOLKIT_DIR}/bin:${PATH}"
    export LD_LIBRARY_PATH="${A2UI_CUDA_TOOLKIT_DIR}/lib64:${A2UI_CUDA_TOOLKIT_DIR}/targets/x86_64-linux/lib:${A2UI_CUDA_TOOLKIT_DIR}/targets/x86_64-linux/lib64:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="${A2UI_CUDA_TOOLKIT_DIR}/lib64:${A2UI_CUDA_TOOLKIT_DIR}/targets/x86_64-linux/lib:${A2UI_CUDA_TOOLKIT_DIR}/targets/x86_64-linux/lib64:${LIBRARY_PATH:-}"
  elif [[ "${A2UI_PREFER_PYTHON_CUDA}" = "1" && -n "${python_cuda_home}" && -x "${python_cuda_home}/bin/nvcc" ]]; then
    export CUDA_HOME="${python_cuda_home}"
    export CUDA_PATH="${python_cuda_home}"
    export CUDACXX="${python_cuda_home}/bin/nvcc"
    export CMAKE_CUDA_COMPILER="${python_cuda_home}/bin/nvcc"
  elif [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    export CUDA_PATH="${CUDA_HOME}"
    export CUDACXX="${CUDA_HOME}/bin/nvcc"
    export CMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc"
  else
    unset CUDA_HOME
    unset CUDA_PATH
    unset CUDACXX
    unset CMAKE_CUDA_COMPILER
    for path in /usr/local/cuda-13.0 /usr/local/cuda-13.1 /usr/local/cuda-13.2 /usr/local/cuda-13.3 /usr/local/cuda-13 /usr/local/cuda; do
      if [[ -x "${path}/bin/nvcc" ]]; then
        export CUDA_HOME="${path}"
        export CUDA_PATH="${path}"
        export CUDACXX="${path}/bin/nvcc"
        export CMAKE_CUDA_COMPILER="${path}/bin/nvcc"
        break
      fi
    done
  fi
  if [[ -n "${CUDA_HOME:-}" ]]; then
    export CUDAToolkit_ROOT="${CUDA_HOME}"
    export CUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}"
    local cudart
    cudart="$(find "${CUDA_HOME}" \( -type f -o -type l \) 2>/dev/null | grep -E '/libcudart\.so($|\.)' | head -1 || true)"
    if [[ -n "${cudart}" ]]; then
      export CUDA_CUDART_LIBRARY="${cudart}"
      export CMAKE_ARGS="${CMAKE_ARGS:-} -DCUDAToolkit_ROOT=${CUDA_HOME} -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME} -DCUDA_CUDART_LIBRARY=${cudart} -DCUDA_CUDART_LIBRARY_RELEASE=${cudart} -DCMAKE_CUDA_COMPILER=${CUDA_HOME}/bin/nvcc"
      export SKBUILD_CMAKE_ARGS="${SKBUILD_CMAKE_ARGS:-} -DCUDAToolkit_ROOT=${CUDA_HOME} -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME} -DCUDA_CUDART_LIBRARY=${cudart} -DCUDA_CUDART_LIBRARY_RELEASE=${cudart} -DCMAKE_CUDA_COMPILER=${CUDA_HOME}/bin/nvcc"
    fi
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

resolve_model_from_roots() {
  local model_id="$1"
  shift
  python "${REPO_ROOT}/dataset/scripts/resolve_model_from_root.py" \
    --model "${model_id}" \
    "$@" \
    --quiet 2>/dev/null || true
}

resolve_target_model() {
  if is_dir "${GEMMA4_MODEL_PATH}"; then
    printf '%s\n' "${GEMMA4_MODEL_PATH}"
    return 0
  fi
  local mapped
  mapped="$(resolve_model_from_roots "${GEMMA4_MODEL_ID}" --root "${GEMMA4_MODEL_ROOT}")"
  if is_dir "${mapped}"; then
    printf '%s\n' "${mapped}"
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
  local mapped
  mapped="$(resolve_model_from_roots "${GEMMA4_ASSISTANT_MODEL_ID}" --root "${GEMMA4_MODEL_ROOT}")"
  if is_dir "${mapped}"; then
    printf '%s\n' "${mapped}"
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

  local -a pids=()
  local pid
  while IFS= read -r pid; do
    [[ -n "${pid}" ]] && pids+=("${pid}")
  done < <(user_pids_matching 'VLLM.*Worker_TP|EngineCore|vllm[[:space:]]+serve|vllm\.entrypoints\.openai|multiproc_executor')
  while IFS= read -r pid; do
    [[ -n "${pid}" ]] && pids+=("${pid}")
  done < <(port_listener_pids)

  if (( ${#pids[@]} == 0 )); then
    return 0
  fi
  terminate_pids "Cleaning stale vLLM processes before start/restart" "${pids[@]}"
}

TARGET_MODEL_PATH="$(resolve_target_model || true)"
if [[ -z "${TARGET_MODEL_PATH}" ]]; then
  echo "Gemma4 target model not found." >&2
  echo "Set MODEL_ROOT/GEMMA4_MODEL_ROOT to the parent folder containing model-id folders from dataset/configs/models.yaml, or set GEMMA4_MODEL_PATH directly." >&2
  echo "Accepted examples under the root: google/gemma-4-31b-it, google--gemma-4-31b-it, gemma-4-31b-it." >&2
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
  echo "Disabling speculative decoding for this launch because GEMMA4_REQUIRE_SPECULATIVE=0." >&2
  GEMMA4_SPECULATIVE_MODE="off"
fi
VLLM_HAS_REASONING_PARSER=0
if grep -q -- "--reasoning-parser" <<<"${HELP_TEXT}"; then
  VLLM_HAS_REASONING_PARSER=1
fi
VLLM_HAS_ENABLE_REASONING=0
if grep -q -- "--enable-reasoning" <<<"${HELP_TEXT}"; then
  VLLM_HAS_ENABLE_REASONING=1
fi
VLLM_HAS_DEFAULT_CHAT_TEMPLATE_KWARGS=0
if grep -q -- "--default-chat-template-kwargs" <<<"${HELP_TEXT}"; then
  VLLM_HAS_DEFAULT_CHAT_TEMPLATE_KWARGS=1
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
  if [[ "${VLLM_HAS_ENABLE_REASONING}" = "1" && "${VLLM_HAS_REASONING_PARSER}" = "1" ]]; then
    cmd+=(--enable-reasoning)
    cmd+=(--reasoning-parser gemma4)
  else
    echo "Warning: this vLLM build does not expose Gemma4 server reasoning parser flags." >&2
    echo "Continuing with prompt-side Gemma thinking marker; client will strip thinking before JSON parsing." >&2
  fi
  if is_truthy "${GEMMA4_ENABLE_DEFAULT_THINKING}" && [[ "${VLLM_HAS_DEFAULT_CHAT_TEMPLATE_KWARGS}" = "1" ]]; then
    cmd+=(--default-chat-template-kwargs '{"enable_thinking": true}')
  elif is_truthy "${GEMMA4_ENABLE_DEFAULT_THINKING}"; then
    echo "Warning: this vLLM build does not expose --default-chat-template-kwargs; per-request chat_template_kwargs stay disabled unless LOCAL_VLLM_SEND_CHAT_TEMPLATE_KWARGS=1." >&2
  fi
  if [[ "${GEMMA4_REASONING_FLAGS_MODE}" = "full" ]]; then
    if grep -q -- "--enable-auto-tool-choice" <<<"${HELP_TEXT}" && grep -q -- "--tool-call-parser" <<<"${HELP_TEXT}"; then
      cmd+=(--enable-auto-tool-choice --tool-call-parser gemma4)
    else
      echo "Warning: full Gemma4 tool parser flags are not available in this vLLM build." >&2
    fi
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
echo "  FLASHINFER_DISABLE_VERSION_CHECK=${FLASHINFER_DISABLE_VERSION_CHECK}"
echo "  FLASHINFER_DISABLE_VERSION__CHECK=${FLASHINFER_DISABLE_VERSION__CHECK}"
echo "  CUDA_HOME=${CUDA_HOME:-}"
echo "  clean_stale_processes=${VLLM_CLEAN_STALE_PROCESSES}"
echo "  run_log=${VLLM_RUN_LOG}"
echo "  restart_on_engine_crash_signature=${VLLM_RESTART_ON_ENGINE_CRASH_SIGNATURE}"
if (( VLLM_MAX_RESTARTS > 0 )); then
  echo "  restart_on_crash=${VLLM_RESTART_ON_CRASH} max_restarts=${VLLM_MAX_RESTARTS}"
else
  echo "  restart_on_crash=${VLLM_RESTART_ON_CRASH} max_restarts=unlimited"
fi
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

cleanup_stale_vllm_processes

VLLM_ACTIVE_PID=""
VLLM_LAST_RUN_LOG_START_LINE=0

terminate_active_vllm() {
  local pid="${VLLM_ACTIVE_PID:-}"
  if [[ -z "${pid}" || ! "${pid}" =~ ^[0-9]+$ ]]; then
    cleanup_stale_vllm_processes
    return 0
  fi
  echo "Stopping active vLLM process group for pid=${pid}"
  kill -TERM "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  sleep 5
  if pid_alive "${pid}"; then
    kill -KILL "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
  fi
  cleanup_stale_vllm_processes
}

on_shutdown_signal() {
  echo "Received shutdown signal; cleaning vLLM workers."
  terminate_active_vllm
  exit 130
}
trap on_shutdown_signal INT TERM

run_vllm_once() {
  mkdir -p "$(dirname "${VLLM_RUN_LOG}")"
  if [[ -f "${VLLM_RUN_LOG}" ]]; then
    VLLM_LAST_RUN_LOG_START_LINE="$(wc -l <"${VLLM_RUN_LOG}" 2>/dev/null || printf '0')"
  else
    VLLM_LAST_RUN_LOG_START_LINE=0
  fi
  {
    printf '\n===== A2UI vLLM start %s =====\n' "$(date -Is)"
    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  } >>"${VLLM_RUN_LOG}" 2>/dev/null || true

  if command -v setsid >/dev/null 2>&1; then
    setsid "${cmd[@]}" > >(tee -a "${VLLM_RUN_LOG}") 2> >(tee -a "${VLLM_RUN_LOG}" >&2) &
  else
    "${cmd[@]}" > >(tee -a "${VLLM_RUN_LOG}") 2> >(tee -a "${VLLM_RUN_LOG}" >&2) &
  fi
  VLLM_ACTIVE_PID="$!"
  echo "vLLM child pid=${VLLM_ACTIVE_PID}"
  set +e
  wait "${VLLM_ACTIVE_PID}"
  local rc=$?
  set -e
  VLLM_ACTIVE_PID=""
  printf '===== A2UI vLLM exit %s rc=%s =====\n' "$(date -Is)" "${rc}" >>"${VLLM_RUN_LOG}" 2>/dev/null || true
  return "${rc}"
}

vllm_log_has_engine_crash_signature() {
  is_truthy "${VLLM_RESTART_ON_ENGINE_CRASH_SIGNATURE}" || return 1
  [[ -f "${VLLM_RUN_LOG}" ]] || return 1
  local start_line="${VLLM_LAST_RUN_LOG_START_LINE:-0}"
  tail -n +"$(( start_line + 1 ))" "${VLLM_RUN_LOG}" 2>/dev/null | grep -Eiq \
    'EngineDeadError|AsyncLLM output handler failed|RuntimeError: Executor failed|EngineCore encountered an issue|engine core encountered an issue|Executor failed|DeadError|Worker.*died|Traceback \(most recent call last\)'
}

restart_count=0
while true; do
  start_ts="$(date +%s)"
  if run_vllm_once; then
    rc=0
  else
    rc=$?
  fi
  end_ts="$(date +%s)"
  runtime=$(( end_ts - start_ts ))

  cleanup_stale_vllm_processes

  if [[ "${rc}" = "0" ]] && vllm_log_has_engine_crash_signature; then
    echo "vLLM exited with rc=0 but engine crash signature was found in ${VLLM_RUN_LOG}; treating as crash for restart."
    rc=100
  fi

  if [[ "${rc}" = "0" ]]; then
    echo "vLLM exited normally."
    exit 0
  fi
  if ! is_truthy "${VLLM_RESTART_ON_CRASH}"; then
    echo "vLLM exited with rc=${rc}; restart disabled."
    exit "${rc}"
  fi
  if (( VLLM_MAX_RESTARTS > 0 && restart_count >= VLLM_MAX_RESTARTS )); then
    echo "vLLM exited with rc=${rc}; reached max restarts (${VLLM_MAX_RESTARTS})."
    exit "${rc}"
  fi

  restart_count=$(( restart_count + 1 ))
  sleep_s=$(( VLLM_RESTART_BACKOFF_SECONDS * restart_count ))
  if (( VLLM_MAX_RESTARTS > 0 )); then
    echo "vLLM exited with rc=${rc} after ${runtime}s; cleaned stale workers. Restart ${restart_count}/${VLLM_MAX_RESTARTS} in ${sleep_s}s."
  else
    echo "vLLM exited with rc=${rc} after ${runtime}s; cleaned stale workers. Restart ${restart_count}/unlimited in ${sleep_s}s."
  fi
  sleep "${sleep_s}"
done
