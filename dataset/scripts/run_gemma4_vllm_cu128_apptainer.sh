#!/usr/bin/env bash
set -euo pipefail

# Start Gemma4 31B through the custom CUDA 12.8 vLLM SIF.
# Run inside a Slurm GPU allocation, for example:
#   srun --gres=gpu:2 --cpus-per-task=16 --mem=160G --pty bash
#   module load apptainer
#   VLLM_SIF=/isilonhome/k_anup/containers/a2ui-vllm-cu128-source_gemma4_speculative.sif \
#   MODEL_ROOT=/isilonhome/k_anup/models \
#   bash dataset/scripts/run_gemma4_vllm_cu128_apptainer.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GEMMA4_MODEL_ID="${GEMMA4_MODEL_ID:-google/gemma-4-31b-it}"
GEMMA4_ASSISTANT_MODEL_ID="${GEMMA4_ASSISTANT_MODEL_ID:-google/gemma-4-31b-it-assistant}"
MODEL_ROOT="${MODEL_ROOT:-${LOCAL_MODEL_ROOT:-${A2UI_MODEL_ROOT:-${GEMMA4_MODEL_ROOT:-${HOME}/models}}}}"
GEMMA4_MODEL_ROOT="${GEMMA4_MODEL_ROOT:-${MODEL_ROOT}}"
GEMMA4_MODEL_PATH="${GEMMA4_MODEL_PATH:-}"
GEMMA4_ASSISTANT_MODEL_PATH="${GEMMA4_ASSISTANT_MODEL_PATH:-}"

VLLM_SIF="${VLLM_SIF:-${HOME}/containers/a2ui-vllm-cu128-source_gemma4_speculative.sif}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
VLLM_QUANTIZATION_MODE="${VLLM_QUANTIZATION_MODE:-none}"
VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-}"
VLLM_CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
VLLM_CONTAINER_MODEL_PATH="${VLLM_CONTAINER_MODEL_PATH:-/models/gemma4_31b}"
GEMMA4_ASSISTANT_CONTAINER_PATH="${GEMMA4_ASSISTANT_CONTAINER_PATH:-/models/gemma4_31b_assistant}"

# This container runner is specifically for the Gemma4 reasoning/speculative setup.
# The normal Python runner keeps these off by default for compatibility.
GEMMA4_ENABLE_REASONING="${GEMMA4_ENABLE_REASONING:-1}"
GEMMA4_REASONING_PARSER="${GEMMA4_REASONING_PARSER:-gemma4}"
GEMMA4_ENABLE_DEFAULT_THINKING="${GEMMA4_ENABLE_DEFAULT_THINKING:-0}"
GEMMA4_SPECULATIVE_MODE="${GEMMA4_SPECULATIVE_MODE:-draft}" # draft|off
GEMMA4_SPECULATIVE_TOKENS="${GEMMA4_SPECULATIVE_TOKENS:-1}"
GEMMA4_SPECULATIVE_METHOD="${GEMMA4_SPECULATIVE_METHOD:-mtp}"
GEMMA4_REQUIRE_SPECULATIVE="${GEMMA4_REQUIRE_SPECULATIVE:-1}"

VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-1}"
VLLM_HAS_FLASHINFER_CUBIN="${VLLM_HAS_FLASHINFER_CUBIN:-1}"
FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
FLASHINFER_DISABLE_VERSION__CHECK="${FLASHINFER_DISABLE_VERSION__CHECK:-1}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

A2UI_CONTAINER_USER="${A2UI_CONTAINER_USER:-a2ui_user}"
A2UI_CONTAINER_GROUP="${A2UI_CONTAINER_GROUP:-a2ui_group}"
A2UI_BIND_SYNTHETIC_PASSWD="${A2UI_BIND_SYNTHETIC_PASSWD:-1}"
A2UI_ENABLE_HOST_NSS_WRAPPER="${A2UI_ENABLE_HOST_NSS_WRAPPER:-1}"
A2UI_NSS_WRAPPER_LIB="${A2UI_NSS_WRAPPER_LIB:-}"

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
  if [[ -f "${REPO_ROOT}/dataset/scripts/resolve_model_from_root.py" ]] && command -v python >/dev/null 2>&1; then
    python "${REPO_ROOT}/dataset/scripts/resolve_model_from_root.py" \
      --model "${model_id}" \
      "$@" \
      --quiet 2>/dev/null || true
  fi
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
      "google/gemma-4-31b-it" \
      "google/gemma-4-31B-it" \
      google--gemma-4-31b-it \
      google--gemma-4-31B-it \
      gemma-4-31b-it \
      gemma-4-31B-it \
      gemma4-31b-it \
      gemma4-31B-it \
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
      "google/gemma-4-31b-it-assistant" \
      "google/gemma-4-31B-it-assistant" \
      google--gemma-4-31b-it-assistant \
      google--gemma-4-31B-it-assistant \
      gemma-4-31b-it-assistant \
      gemma-4-31B-it-assistant \
      gemma4-31b-it-assistant \
      gemma4-31B-it-assistant \
      Gemma4-31B-it-assistant && return 0
  fi
  return 1
}

find_nss_wrapper_lib() {
  if [[ -n "${A2UI_NSS_WRAPPER_LIB}" && -f "${A2UI_NSS_WRAPPER_LIB}" ]]; then
    printf '%s\n' "${A2UI_NSS_WRAPPER_LIB}"
    return 0
  fi
  if command -v ldconfig >/dev/null 2>&1; then
    local from_ldconfig
    from_ldconfig="$(ldconfig -p 2>/dev/null | awk '/libnss_wrapper\\.so/ {print $NF; exit}')"
    if [[ -n "${from_ldconfig}" && -f "${from_ldconfig}" ]]; then
      printf '%s\n' "${from_ldconfig}"
      return 0
    fi
  fi
  local candidate
  for candidate in \
    "${A2UI_NSS_WRAPPER_PREFIX:-${HOME}/.local/a2ui-nss-wrapper}"/lib*/libnss_wrapper.so \
    "${HOME}/.local/lib"/libnss_wrapper.so \
    "${HOME}/lib"/libnss_wrapper.so \
    /usr/lib64/libnss_wrapper.so \
    /usr/lib/x86_64-linux-gnu/libnss_wrapper.so \
    /usr/lib/libnss_wrapper.so \
    /lib64/libnss_wrapper.so \
    /lib/x86_64-linux-gnu/libnss_wrapper.so; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

if [[ ! -f "${VLLM_SIF}" ]]; then
  echo "SIF not found: ${VLLM_SIF}" >&2
  exit 1
fi

TARGET_MODEL_PATH="$(resolve_target_model || true)"
if [[ -z "${TARGET_MODEL_PATH}" ]]; then
  echo "Gemma4 target model not found." >&2
  echo "Set MODEL_ROOT/GEMMA4_MODEL_ROOT to the parent folder, or set GEMMA4_MODEL_PATH directly." >&2
  echo "Accepted examples under the root: google/gemma-4-31b-it, google--gemma-4-31b-it, gemma-4-31b-it." >&2
  exit 1
fi

ASSISTANT_MODEL_PATH="$(resolve_assistant_model || true)"
if [[ "${GEMMA4_SPECULATIVE_MODE}" != "off" && -z "${ASSISTANT_MODEL_PATH}" ]]; then
  echo "Gemma4 assistant model not found, but GEMMA4_SPECULATIVE_MODE=${GEMMA4_SPECULATIVE_MODE}." >&2
  echo "Set GEMMA4_ASSISTANT_MODEL_PATH or set GEMMA4_SPECULATIVE_MODE=off." >&2
  exit 1
fi

if ! command -v apptainer >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
  module load apptainer || true
fi

if command -v apptainer >/dev/null 2>&1; then
  RUNTIME="apptainer"
elif command -v singularity >/dev/null 2>&1; then
  RUNTIME="singularity"
else
  echo "apptainer or singularity is required. Try: module load apptainer" >&2
  exit 1
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | paste -sd, -)"
  export CUDA_VISIBLE_DEVICES
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
  A2UI_VLLM_GPUS="${A2UI_VLLM_GPUS:-${#gpu_ids[@]}}"
else
  A2UI_VLLM_GPUS="${A2UI_VLLM_GPUS:-1}"
fi

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
HOST_HOME="${HOME:-/tmp}"
CONTAINER_USER="$(id -un 2>/dev/null || true)"
CONTAINER_GROUP="$(id -gn 2>/dev/null || true)"
CONTAINER_USER="${CONTAINER_USER:-${A2UI_CONTAINER_USER}}"
CONTAINER_GROUP="${CONTAINER_GROUP:-${A2UI_CONTAINER_GROUP}}"
PASSWD_DIR=""
PASSWD_FILE=""
GROUP_FILE=""

cleanup() {
  if [[ -n "${PASSWD_DIR}" && -d "${PASSWD_DIR}" ]]; then
    rm -rf "${PASSWD_DIR}"
  fi
}
trap cleanup EXIT

RUNTIME_ARGS=(exec --nv --cleanenv)
RUNTIME_ARGS+=(--env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}")
RUNTIME_ARGS+=(--env "USER=${CONTAINER_USER}")
RUNTIME_ARGS+=(--env "LOGNAME=${CONTAINER_USER}")
RUNTIME_ARGS+=(--env "HOME=${HOST_HOME}")
RUNTIME_ARGS+=(--env "VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER}")
RUNTIME_ARGS+=(--env "VLLM_HAS_FLASHINFER_CUBIN=${VLLM_HAS_FLASHINFER_CUBIN}")
RUNTIME_ARGS+=(--env "FLASHINFER_DISABLE_VERSION_CHECK=${FLASHINFER_DISABLE_VERSION_CHECK}")
RUNTIME_ARGS+=(--env "FLASHINFER_DISABLE_VERSION__CHECK=${FLASHINFER_DISABLE_VERSION__CHECK}")
RUNTIME_ARGS+=(--env "HF_HUB_OFFLINE=${HF_HUB_OFFLINE}")
RUNTIME_ARGS+=(--env "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}")
RUNTIME_ARGS+=(--env "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE}")
RUNTIME_ARGS+=(--bind "${TARGET_MODEL_PATH}:${VLLM_CONTAINER_MODEL_PATH}:ro")

if [[ -n "${ASSISTANT_MODEL_PATH}" ]]; then
  RUNTIME_ARGS+=(--bind "${ASSISTANT_MODEL_PATH}:${GEMMA4_ASSISTANT_CONTAINER_PATH}:ro")
fi

if [[ "${A2UI_BIND_SYNTHETIC_PASSWD}" != "0" ]]; then
  PASSWD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/a2ui_apptainer_user.XXXXXX")"
  PASSWD_FILE="${PASSWD_DIR}/passwd"
  GROUP_FILE="${PASSWD_DIR}/group"
  {
    printf 'root:x:0:0:root:/root:/bin/bash\n'
    printf 'nobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin\n'
    printf '%s:x:%s:%s:A2UI synthetic user:%s:/bin/bash\n' \
      "${CONTAINER_USER}" "${HOST_UID}" "${HOST_GID}" "${HOST_HOME}"
  } > "${PASSWD_FILE}"
  {
    printf 'root:x:0:\n'
    printf 'nogroup:x:65534:\n'
    printf '%s:x:%s:\n' "${CONTAINER_GROUP}" "${HOST_GID}"
  } > "${GROUP_FILE}"
  RUNTIME_ARGS+=(--bind "${PASSWD_FILE}:/etc/passwd:ro")
  RUNTIME_ARGS+=(--bind "${GROUP_FILE}:/etc/group:ro")
fi

if command -v getent >/dev/null 2>&1 && ! getent passwd "${HOST_UID}" >/dev/null 2>&1; then
  echo "Warning: host NSS cannot resolve UID ${HOST_UID}." >&2
  if [[ "${A2UI_ENABLE_HOST_NSS_WRAPPER}" != "0" ]]; then
    if command -v module >/dev/null 2>&1; then
      module load nss_wrapper >/dev/null 2>&1 || true
    fi
    if NSS_WRAPPER_LIB="$(find_nss_wrapper_lib)"; then
      export LD_PRELOAD="${NSS_WRAPPER_LIB}${LD_PRELOAD:+:${LD_PRELOAD}}"
      export NSS_WRAPPER_PASSWD="${PASSWD_FILE}"
      export NSS_WRAPPER_GROUP="${GROUP_FILE}"
      echo "Enabled host NSS wrapper for UID ${HOST_UID}: ${NSS_WRAPPER_LIB}" >&2
    else
      echo "Host NSS wrapper library was not found." >&2
    fi
  fi
fi

echo "Probing vLLM version inside ${VLLM_SIF}"
VLLM_VERSION_TEXT="$("${RUNTIME}" "${RUNTIME_ARGS[@]}" "${VLLM_SIF}" python - <<'PY' 2>&1 || true
import inspect
try:
    import vllm
except Exception as exc:
    print(f"vllm import failed: {exc!r}")
else:
    print("vllm version:", getattr(vllm, "__version__", "unknown"))
    print("vllm path:", inspect.getfile(vllm))
PY
)"
printf '%s\n' "${VLLM_VERSION_TEXT}"

echo "Probing vLLM flags inside ${VLLM_SIF}"
HELP_TEXT="$("${RUNTIME}" "${RUNTIME_ARGS[@]}" "${VLLM_SIF}" vllm serve --help 2>&1 || true)"
if ! grep -q -- "--host" <<<"${HELP_TEXT}"; then
  echo "Warning: could not verify vLLM serve help. First probe lines:" >&2
  printf '%s\n' "${HELP_TEXT}" | head -80 >&2
fi

has_help_flag() {
  grep -q -- "$1" <<<"${HELP_TEXT}"
}

VLLM_CMD_ARGS=(
  vllm serve "${VLLM_CONTAINER_MODEL_PATH}"
  --served-model-name "${GEMMA4_MODEL_ID}"
  --host "${VLLM_HOST}"
  --port "${VLLM_PORT}"
  --tensor-parallel-size "${A2UI_VLLM_GPUS}"
  --dtype "${VLLM_DTYPE}"
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
  --max-model-len "${VLLM_MAX_MODEL_LEN}"
  --max-num-seqs "${VLLM_MAX_NUM_SEQS}"
  --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}"
  --trust-remote-code
)

case "${VLLM_QUANTIZATION_MODE,,}" in
  ""|none|off|false|0)
    ;;
  int8|bnb-int8|bitsandbytes-int8)
    VLLM_CMD_ARGS+=(
      --quantization bitsandbytes
      --load-format bitsandbytes
      --model-loader-extra-config '{"load_in_8bit":true,"load_in_4bit":false}'
    )
    ;;
  bnb-4bit|bitsandbytes-4bit|4bit)
    VLLM_CMD_ARGS+=(
      --quantization bitsandbytes
      --load-format bitsandbytes
      --model-loader-extra-config '{"load_in_8bit":false,"load_in_4bit":true}'
    )
    ;;
  fp8)
    VLLM_CMD_ARGS+=(--quantization fp8)
    ;;
  *)
    echo "Unsupported VLLM_QUANTIZATION_MODE='${VLLM_QUANTIZATION_MODE}'." >&2
    echo "Supported: none, int8, bnb-4bit, fp8" >&2
    exit 1
    ;;
esac

if [[ -n "${VLLM_KV_CACHE_DTYPE}" ]]; then
  VLLM_CMD_ARGS+=(--kv-cache-dtype "${VLLM_KV_CACHE_DTYPE}")
fi

if [[ -n "${VLLM_CPU_OFFLOAD_GB}" ]]; then
  VLLM_CMD_ARGS+=(--cpu-offload-gb "${VLLM_CPU_OFFLOAD_GB}")
fi

if [[ "${GEMMA4_SPECULATIVE_MODE}" != "off" ]]; then
  if has_help_flag "--speculative-config"; then
    SPEC_JSON="{\"method\":\"${GEMMA4_SPECULATIVE_METHOD}\",\"model\":\"${GEMMA4_ASSISTANT_CONTAINER_PATH}\",\"num_speculative_tokens\":${GEMMA4_SPECULATIVE_TOKENS}}"
    VLLM_CMD_ARGS+=(--speculative-config "${SPEC_JSON}")
  elif has_help_flag "--spec-model" && has_help_flag "--spec-tokens"; then
    # Newer vLLM versions expose these as flattened VllmConfig args instead
    # of the older --speculative-model spelling.
    if has_help_flag "--spec-method"; then
      VLLM_CMD_ARGS+=(--spec-method "${GEMMA4_SPECULATIVE_METHOD}")
    fi
    VLLM_CMD_ARGS+=(--spec-model "${GEMMA4_ASSISTANT_CONTAINER_PATH}")
    VLLM_CMD_ARGS+=(--spec-tokens "${GEMMA4_SPECULATIVE_TOKENS}")
  elif has_help_flag "--speculative-model" && has_help_flag "--num-speculative-tokens"; then
    VLLM_CMD_ARGS+=(--speculative-model "${GEMMA4_ASSISTANT_CONTAINER_PATH}")
    VLLM_CMD_ARGS+=(--num-speculative-tokens "${GEMMA4_SPECULATIVE_TOKENS}")
  elif is_truthy "${GEMMA4_REQUIRE_SPECULATIVE}"; then
    echo "This vLLM install exposes no supported speculative decoding server flags." >&2
    echo "Checked: --speculative-config, --spec-model/--spec-tokens, --speculative-model/--num-speculative-tokens." >&2
    echo "Use the Gemma4 speculative vLLM source image, or set GEMMA4_REQUIRE_SPECULATIVE=0 GEMMA4_SPECULATIVE_MODE=off." >&2
    echo "Speculative-related help lines from the container:" >&2
    grep -Ei 'spec|draft|mtp|eagle|medusa' <<<"${HELP_TEXT}" | head -120 >&2 || true
    exit 1
  else
    echo "Warning: vLLM speculative flags are unavailable; continuing without speculative decoding." >&2
  fi
fi

if is_truthy "${GEMMA4_ENABLE_REASONING}"; then
  if has_help_flag "--enable-reasoning" && has_help_flag "--reasoning-parser"; then
    VLLM_CMD_ARGS+=(--enable-reasoning --reasoning-parser "${GEMMA4_REASONING_PARSER}")
  elif has_help_flag "--reasoning-parser"; then
    # Current vLLM exposes reasoning parsing through StructuredOutputsConfig
    # without a separate --enable-reasoning switch.
    VLLM_CMD_ARGS+=(--reasoning-parser "${GEMMA4_REASONING_PARSER}")
  else
    echo "Warning: this vLLM build does not expose --enable-reasoning/--reasoning-parser." >&2
    echo "Server will start without reasoning parser flags. Use prompt-side <|think|> handling in the client if needed." >&2
  fi

  if is_truthy "${GEMMA4_ENABLE_DEFAULT_THINKING}"; then
    if has_help_flag "--default-chat-template-kwargs"; then
      VLLM_CMD_ARGS+=(--default-chat-template-kwargs '{"enable_thinking": true}')
    else
      echo "Warning: --default-chat-template-kwargs unavailable; default thinking flag was not applied." >&2
    fi
  fi
fi

if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${VLLM_EXTRA_ARGS} )
  VLLM_CMD_ARGS+=("${extra[@]}")
fi

echo "Runtime: ${RUNTIME}"
echo "SIF: ${VLLM_SIF}"
echo "Model path: ${TARGET_MODEL_PATH}"
echo "Assistant path: ${ASSISTANT_MODEL_PATH:-disabled}"
echo "Served model: ${GEMMA4_MODEL_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Tensor parallel GPUs: ${A2UI_VLLM_GPUS}"
echo "GPU memory utilization: ${VLLM_GPU_MEMORY_UTILIZATION}"
echo "Max model length: ${VLLM_MAX_MODEL_LEN}"
echo "Max num seqs: ${VLLM_MAX_NUM_SEQS}"
echo "Max num batched tokens: ${VLLM_MAX_NUM_BATCHED_TOKENS}"
echo "Quantization mode: ${VLLM_QUANTIZATION_MODE}"
echo "KV cache dtype: ${VLLM_KV_CACHE_DTYPE:-auto}"
echo "CPU offload GiB/GPU: ${VLLM_CPU_OFFLOAD_GB:-0}"
echo "Reasoning requested: ${GEMMA4_ENABLE_REASONING}"
echo "Speculative mode: ${GEMMA4_SPECULATIVE_MODE}, method=${GEMMA4_SPECULATIVE_METHOD}, tokens=${GEMMA4_SPECULATIVE_TOKENS}"
echo "FLASHINFER_DISABLE_VERSION_CHECK=${FLASHINFER_DISABLE_VERSION_CHECK}"
echo "FLASHINFER_DISABLE_VERSION__CHECK=${FLASHINFER_DISABLE_VERSION__CHECK}"
echo "vLLM command: ${VLLM_CMD_ARGS[*]}"

exec "${RUNTIME}" "${RUNTIME_ARGS[@]}" \
  "${VLLM_SIF}" \
  "${VLLM_CMD_ARGS[@]}"
