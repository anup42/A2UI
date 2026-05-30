#!/usr/bin/env bash
set -euo pipefail

# Start Gemma4 31B through a Singularity-hosted vLLM OpenAI-compatible server.
# Supports normal generation, thinking/reasoning prompts, and Gemma4 assistant
# MTP speculative decoding when GEMMA4_SPECULATIVE_MODE=mtp.

GEMMA4_MODEL_ID="${GEMMA4_MODEL_ID:-google/gemma-4-31b-it}"
GEMMA4_MODEL_PATH="${GEMMA4_MODEL_PATH:-}"
VLLM_SIF="${VLLM_SIF:-${HOME}/containers/a2ui-vllm-cu128-source_gemma4_speculative.sif}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_QUANTIZATION_MODE="${VLLM_QUANTIZATION_MODE:-none}"
VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-}"
VLLM_CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}"
VLLM_CONTAINER_MODEL_PATH="${VLLM_CONTAINER_MODEL_PATH:-/models/gemma4}"

GEMMA4_ENABLE_REASONING="${GEMMA4_ENABLE_REASONING:-0}"
GEMMA4_ENABLE_SERVER_REASONING_FLAGS="${GEMMA4_ENABLE_SERVER_REASONING_FLAGS:-0}"
GEMMA4_REASONING_PARSER="${GEMMA4_REASONING_PARSER:-}"

GEMMA4_SPECULATIVE_MODE="${GEMMA4_SPECULATIVE_MODE:-mtp}"
GEMMA4_SPECULATIVE_MODEL_PATH="${GEMMA4_SPECULATIVE_MODEL_PATH:-${GEMMA4_DRAFT_MODEL_PATH:-}}"
GEMMA4_ASSISTANT_MODEL_PATH="${GEMMA4_ASSISTANT_MODEL_PATH:-}"
GEMMA4_SPECULATIVE_CONTAINER_PATH="${GEMMA4_SPECULATIVE_CONTAINER_PATH:-/models/gemma4_draft}"
GEMMA4_SPECULATIVE_TOKENS="${GEMMA4_SPECULATIVE_TOKENS:-1}"
GEMMA4_SPECULATIVE_DRAFT_TP="${GEMMA4_SPECULATIVE_DRAFT_TP:-}"
GEMMA4_ALLOW_LEGACY_SPECULATIVE_FALLBACK="${GEMMA4_ALLOW_LEGACY_SPECULATIVE_FALLBACK:-1}"

is_truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

resolve_model_path() {
  local requested="$1"
  shift
  if [[ -n "${requested}" && -d "${requested}" ]]; then
    printf '%s\n' "${requested}"
    return 0
  fi
  local candidate
  for candidate in "$@"; do
    if [[ -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

find_nss_wrapper_lib() {
  if [[ -n "${A2UI_NSS_WRAPPER_LIB:-}" && -f "${A2UI_NSS_WRAPPER_LIB}" ]]; then
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

if ! command -v singularity >/dev/null 2>&1; then
  echo "singularity is required on this machine." >&2
  exit 1
fi

GEMMA4_MODEL_PATH="$(resolve_model_path "${GEMMA4_MODEL_PATH}" \
  "${HOME}/dataset_generation/models/gemma4-31b" \
  "${HOME}/dataset_generation/models/gemma-4-31b-it" \
  "${HOME}/models/gemma4-31b" \
  "${HOME}/models/gemma-4-31b-it" \
  "${HOME}/models/google--gemma-4-31b-it")" || {
    echo "Gemma4 model folder not found. Set GEMMA4_MODEL_PATH." >&2
    exit 1
  }

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
CONTAINER_USER="${CONTAINER_USER:-a2ui_user}"
CONTAINER_GROUP="${CONTAINER_GROUP:-a2ui_group}"

PASSWD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/a2ui_singularity_user.XXXXXX")"
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

if command -v getent >/dev/null 2>&1 && ! getent passwd "${HOST_UID}" >/dev/null 2>&1; then
  echo "Warning: host NSS cannot resolve UID ${HOST_UID}." >&2
  if NSS_WRAPPER_LIB="$(find_nss_wrapper_lib)"; then
    export LD_PRELOAD="${NSS_WRAPPER_LIB}${LD_PRELOAD:+:${LD_PRELOAD}}"
    export NSS_WRAPPER_PASSWD="${PASSWD_FILE}"
    export NSS_WRAPPER_GROUP="${GROUP_FILE}"
    echo "Enabled host NSS wrapper: ${NSS_WRAPPER_LIB}" >&2
  else
    echo "Host libnss_wrapper not found; Singularity may fail with unknown userid." >&2
  fi
fi

RUNTIME_ARGS=(exec --nv --cleanenv)
RUNTIME_ARGS+=(--env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}")
RUNTIME_ARGS+=(--env "USER=${CONTAINER_USER}")
RUNTIME_ARGS+=(--env "LOGNAME=${CONTAINER_USER}")
RUNTIME_ARGS+=(--env "HOME=${HOST_HOME}")
RUNTIME_ARGS+=(--bind "${PASSWD_FILE}:/etc/passwd:ro")
RUNTIME_ARGS+=(--bind "${GROUP_FILE}:/etc/group:ro")
RUNTIME_ARGS+=(--bind "${GEMMA4_MODEL_PATH}:${VLLM_CONTAINER_MODEL_PATH}:ro")

VLLM_CMD_ARGS=(
  vllm serve "${VLLM_CONTAINER_MODEL_PATH}"
  --served-model-name "${GEMMA4_MODEL_ID}"
  --host "${VLLM_HOST}"
  --port "${VLLM_PORT}"
  --tensor-parallel-size "${A2UI_VLLM_GPUS}"
  --dtype "${VLLM_DTYPE}"
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
  --max-model-len "${VLLM_MAX_MODEL_LEN}"
  --trust-remote-code
)

case "${VLLM_QUANTIZATION_MODE,,}" in
  ""|none|off|false|0) ;;
  int8|bnb-int8|bitsandbytes-int8)
    VLLM_CMD_ARGS+=(--quantization bitsandbytes --load-format bitsandbytes)
    VLLM_CMD_ARGS+=(--model-loader-extra-config '{"load_in_8bit":true,"load_in_4bit":false}')
    ;;
  bnb-4bit|bitsandbytes-4bit|4bit)
    VLLM_CMD_ARGS+=(--quantization bitsandbytes --load-format bitsandbytes)
    VLLM_CMD_ARGS+=(--model-loader-extra-config '{"load_in_8bit":false,"load_in_4bit":true}')
    ;;
  fp8)
    VLLM_CMD_ARGS+=(--quantization fp8)
    ;;
  *)
    echo "Unsupported VLLM_QUANTIZATION_MODE=${VLLM_QUANTIZATION_MODE}" >&2
    exit 1
    ;;
esac

if [[ -n "${VLLM_KV_CACHE_DTYPE}" ]]; then
  VLLM_CMD_ARGS+=(--kv-cache-dtype "${VLLM_KV_CACHE_DTYPE}")
fi
if [[ -n "${VLLM_CPU_OFFLOAD_GB}" ]]; then
  VLLM_CMD_ARGS+=(--cpu-offload-gb "${VLLM_CPU_OFFLOAD_GB}")
fi
if [[ -n "${VLLM_MAX_NUM_SEQS}" ]]; then
  VLLM_CMD_ARGS+=(--max-num-seqs "${VLLM_MAX_NUM_SEQS}")
fi

if is_truthy "${GEMMA4_ENABLE_REASONING}"; then
  echo "Gemma4 reasoning mode: enabled"
  if is_truthy "${GEMMA4_ENABLE_SERVER_REASONING_FLAGS}"; then
    VLLM_CMD_ARGS+=(--enable-reasoning)
    if [[ -n "${GEMMA4_REASONING_PARSER}" ]]; then
      VLLM_CMD_ARGS+=(--reasoning-parser "${GEMMA4_REASONING_PARSER}")
    fi
  fi
else
  echo "Gemma4 reasoning mode: disabled"
fi

VLLM_HELP_TEXT="$(singularity "${RUNTIME_ARGS[@]}" "${VLLM_SIF}" vllm serve --help 2>&1 || true)"
VLLM_HAS_SPECULATIVE_CONFIG=0
if grep -q -- "--speculative-config" <<<"${VLLM_HELP_TEXT}"; then
  VLLM_HAS_SPECULATIVE_CONFIG=1
fi
VLLM_HAS_LEGACY_SPECULATIVE=0
if grep -q -- "--speculative-model" <<<"${VLLM_HELP_TEXT}" && grep -q -- "--num-speculative-tokens" <<<"${VLLM_HELP_TEXT}"; then
  VLLM_HAS_LEGACY_SPECULATIVE=1
fi

case "${GEMMA4_SPECULATIVE_MODE,,}" in
  ""|off|none|false|0)
    echo "Gemma4 speculative decoding: disabled"
    ;;
  mtp)
    GEMMA4_ASSISTANT_MODEL_PATH="$(resolve_model_path "${GEMMA4_ASSISTANT_MODEL_PATH}" \
      "${HOME}/dataset_generation/models/gemma-4-31b-it-assistant" \
      "${HOME}/dataset_generation/models/gemma4-31b-assistant" \
      "${HOME}/models/gemma-4-31b-it-assistant" \
      "${HOME}/models/gemma4-31b-assistant" \
      "${HOME}/models/google--gemma-4-31b-it-assistant")" || {
        echo "MTP speculative decoding requested, but Gemma4 assistant model was not found." >&2
        echo "Set GEMMA4_ASSISTANT_MODEL_PATH, or run with GEMMA4_SPECULATIVE_MODE=off." >&2
        exit 1
      }
    RUNTIME_ARGS+=(--bind "${GEMMA4_ASSISTANT_MODEL_PATH}:${GEMMA4_SPECULATIVE_CONTAINER_PATH}:ro")
    if [[ "${VLLM_HAS_SPECULATIVE_CONFIG}" = "1" ]]; then
      spec_json="$(python - "${GEMMA4_SPECULATIVE_CONTAINER_PATH}" "${GEMMA4_SPECULATIVE_TOKENS}" <<'PY'
import json
import sys
print(json.dumps({
    "method": "mtp",
    "model": sys.argv[1],
    "num_speculative_tokens": int(sys.argv[2]),
}))
PY
)"
      VLLM_CMD_ARGS+=(--speculative-config "${spec_json}")
      echo "Gemma4 speculative decoding: mtp=${GEMMA4_ASSISTANT_MODEL_PATH}, tokens=${GEMMA4_SPECULATIVE_TOKENS}"
    elif [[ "${VLLM_HAS_LEGACY_SPECULATIVE}" = "1" ]] && is_truthy "${GEMMA4_ALLOW_LEGACY_SPECULATIVE_FALLBACK}"; then
      echo "Warning: this vLLM install does not expose --speculative-config." >&2
      echo "Using legacy --speculative-model/--num-speculative-tokens with token-1 fallback; this is not true Gemma4 MTP." >&2
      VLLM_CMD_ARGS+=(--speculative-model "${GEMMA4_SPECULATIVE_CONTAINER_PATH}")
      VLLM_CMD_ARGS+=(--num-speculative-tokens "${GEMMA4_SPECULATIVE_TOKENS}")
      echo "Gemma4 speculative decoding: mtp-legacy=${GEMMA4_ASSISTANT_MODEL_PATH}, tokens=${GEMMA4_SPECULATIVE_TOKENS}"
    else
      echo "This vLLM install does not expose --speculative-config." >&2
      echo "True Gemma4 MTP requires --speculative-config. Install the Gemma4-compatible vLLM build or set GEMMA4_SPECULATIVE_MODE=off." >&2
      exit 1
    fi
    ;;
  draft)
    GEMMA4_SPECULATIVE_MODEL_PATH="$(resolve_model_path "${GEMMA4_SPECULATIVE_MODEL_PATH}" \
      "${HOME}/dataset_generation/models/gemma4-2b" \
      "${HOME}/dataset_generation/models/gemma-4-2b-it" \
      "${HOME}/models/gemma4-2b" \
      "${HOME}/models/gemma-4-2b-it" \
      "${HOME}/models/google--gemma-4-2b-it")" || {
        echo "Speculative decoding requested, but draft model was not found." >&2
        echo "Set GEMMA4_SPECULATIVE_MODEL_PATH, or run with GEMMA4_SPECULATIVE_MODE=off." >&2
        exit 1
      }
    RUNTIME_ARGS+=(--bind "${GEMMA4_SPECULATIVE_MODEL_PATH}:${GEMMA4_SPECULATIVE_CONTAINER_PATH}:ro")
    VLLM_CMD_ARGS+=(--speculative-model "${GEMMA4_SPECULATIVE_CONTAINER_PATH}")
    VLLM_CMD_ARGS+=(--num-speculative-tokens "${GEMMA4_SPECULATIVE_TOKENS}")
    if [[ -n "${GEMMA4_SPECULATIVE_DRAFT_TP}" ]]; then
      VLLM_CMD_ARGS+=(--speculative-draft-tensor-parallel-size "${GEMMA4_SPECULATIVE_DRAFT_TP}")
    fi
    echo "Gemma4 speculative decoding: draft=${GEMMA4_SPECULATIVE_MODEL_PATH}, tokens=${GEMMA4_SPECULATIVE_TOKENS}"
    ;;
  ngram)
    VLLM_CMD_ARGS+=(--speculative-model ngram --num-speculative-tokens "${GEMMA4_SPECULATIVE_TOKENS}")
    echo "Gemma4 speculative decoding: ngram, tokens=${GEMMA4_SPECULATIVE_TOKENS}"
    ;;
  *)
    echo "Unsupported GEMMA4_SPECULATIVE_MODE=${GEMMA4_SPECULATIVE_MODE}. Use mtp, draft, ngram, or off." >&2
    exit 1
    ;;
esac

echo "Singularity: $(singularity --version 2>&1)"
echo "SIF: ${VLLM_SIF}"
echo "Gemma4 model path: ${GEMMA4_MODEL_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Tensor parallel GPUs: ${A2UI_VLLM_GPUS}"
echo "vLLM command: ${VLLM_CMD_ARGS[*]}"

exec singularity "${RUNTIME_ARGS[@]}" "${VLLM_SIF}" "${VLLM_CMD_ARGS[@]}"
