#!/usr/bin/env bash
set -euo pipefail

# Start Gemma4 vLLM from inside the CUDA 13.0 Singularity/Jupyter container.
# Use this when the container was launched by jupyterlab/sbatch.sh and you are
# already in a JupyterLab terminal. Do not wrap this script with singularity exec.

GEMMA4_MODEL_ID="${GEMMA4_MODEL_ID:-google/gemma-4-31b-it}"
GEMMA4_MODEL_PATH="${GEMMA4_MODEL_PATH:-}"
GEMMA4_DRAFT_MODEL_PATH="${GEMMA4_DRAFT_MODEL_PATH:-${GEMMA4_SPECULATIVE_MODEL_PATH:-}}"
GEMMA4_ASSISTANT_MODEL_PATH="${GEMMA4_ASSISTANT_MODEL_PATH:-}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_QUANTIZATION_MODE="${VLLM_QUANTIZATION_MODE:-none}"
VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-}"
VLLM_CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}"
GEMMA4_ENABLE_REASONING="${GEMMA4_ENABLE_REASONING:-0}"
GEMMA4_ENABLE_SERVER_REASONING_FLAGS="${GEMMA4_ENABLE_SERVER_REASONING_FLAGS:-0}"
GEMMA4_REASONING_PARSER="${GEMMA4_REASONING_PARSER:-}"
GEMMA4_SPECULATIVE_MODE="${GEMMA4_SPECULATIVE_MODE:-mtp}"
GEMMA4_SPECULATIVE_TOKENS="${GEMMA4_SPECULATIVE_TOKENS:-1}"
GEMMA4_SPECULATIVE_DRAFT_TP="${GEMMA4_SPECULATIVE_DRAFT_TP:-}"

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

GEMMA4_MODEL_PATH="$(resolve_model_path "${GEMMA4_MODEL_PATH}" \
  "/home/k_anup/Storage_gpu/models/gemma-4-31b-it" \
  "/home/k_anup/Storage_gpu/models/gemma4-31b" \
  "/home/k_anup/models/gemma-4-31b-it" \
  "/models/gemma4" \
  "${HOME}/models/gemma-4-31b-it" \
  "${HOME}/dataset_generation/models/gemma4-31b")" || {
    echo "Gemma4 model folder not found. Set GEMMA4_MODEL_PATH." >&2
    exit 1
  }

VLLM_CMD_ARGS=(
  vllm serve "${GEMMA4_MODEL_PATH}"
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

VLLM_HELP_TEXT="$(vllm serve --help 2>&1 || true)"
VLLM_HAS_SPECULATIVE_CONFIG=0
if grep -q -- "--speculative-config" <<<"${VLLM_HELP_TEXT}"; then
  VLLM_HAS_SPECULATIVE_CONFIG=1
fi
VLLM_HAS_SPLIT_SPECULATIVE_CONFIG=0
if grep -q -- "--spec-method" <<<"${VLLM_HELP_TEXT}" && grep -q -- "--spec-model" <<<"${VLLM_HELP_TEXT}" && grep -q -- "--spec-tokens" <<<"${VLLM_HELP_TEXT}"; then
  VLLM_HAS_SPLIT_SPECULATIVE_CONFIG=1
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
      "/home/k_anup/Storage_gpu/models/gemma-4-31b-it-assistant" \
      "/home/k_anup/Storage_gpu/models/gemma4-31b-assistant" \
      "/home/k_anup/models/gemma-4-31b-it-assistant" \
      "/models/gemma4_draft" \
      "${HOME}/models/gemma-4-31b-it-assistant")" || {
        echo "MTP speculative decoding requested, but Gemma4 assistant model was not found." >&2
        echo "Set GEMMA4_ASSISTANT_MODEL_PATH, or run with GEMMA4_SPECULATIVE_MODE=off." >&2
        exit 1
      }
    if [[ "${VLLM_HAS_SPECULATIVE_CONFIG}" = "1" ]]; then
      spec_json="$(python - "${GEMMA4_ASSISTANT_MODEL_PATH}" "${GEMMA4_SPECULATIVE_TOKENS}" <<'PY'
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
    elif [[ "${VLLM_HAS_SPLIT_SPECULATIVE_CONFIG}" = "1" ]]; then
      VLLM_CMD_ARGS+=(--spec-method gemma4_mtp)
      VLLM_CMD_ARGS+=(--spec-model "${GEMMA4_ASSISTANT_MODEL_PATH}")
      VLLM_CMD_ARGS+=(--spec-tokens "${GEMMA4_SPECULATIVE_TOKENS}")
      echo "Gemma4 speculative decoding: mtp=${GEMMA4_ASSISTANT_MODEL_PATH}, tokens=${GEMMA4_SPECULATIVE_TOKENS}"
    else
      echo "This vLLM install does not expose Gemma4 MTP speculative flags." >&2
      echo "Expected either --speculative-config or --spec-method/--spec-model/--spec-tokens." >&2
      echo "Do not use legacy --speculative-model for Gemma4 MTP; install a vLLM build with Gemma4 MTP support or set GEMMA4_SPECULATIVE_MODE=off." >&2
      exit 1
    fi
    ;;
  draft)
    GEMMA4_DRAFT_MODEL_PATH="$(resolve_model_path "${GEMMA4_DRAFT_MODEL_PATH}" \
      "/home/k_anup/Storage_gpu/models/gemma-4-31b-it-assistant" \
      "/home/k_anup/Storage_gpu/models/gemma4-31b-assistant" \
      "/home/k_anup/Storage_gpu/models/gemma-4-2b-it" \
      "/home/k_anup/models/gemma-4-31b-it-assistant" \
      "/home/k_anup/models/gemma-4-2b-it" \
      "/models/gemma4_draft" \
      "${HOME}/models/gemma-4-31b-it-assistant" \
      "${HOME}/models/gemma-4-2b-it")" || {
        echo "Speculative decoding requested, but draft/assistant model was not found." >&2
        echo "Set GEMMA4_DRAFT_MODEL_PATH, or run with GEMMA4_SPECULATIVE_MODE=off." >&2
        exit 1
      }
    VLLM_CMD_ARGS+=(--speculative-model "${GEMMA4_DRAFT_MODEL_PATH}")
    VLLM_CMD_ARGS+=(--num-speculative-tokens "${GEMMA4_SPECULATIVE_TOKENS}")
    if [[ -n "${GEMMA4_SPECULATIVE_DRAFT_TP}" ]]; then
      VLLM_CMD_ARGS+=(--speculative-draft-tensor-parallel-size "${GEMMA4_SPECULATIVE_DRAFT_TP}")
    fi
    echo "Gemma4 speculative decoding: draft=${GEMMA4_DRAFT_MODEL_PATH}, tokens=${GEMMA4_SPECULATIVE_TOKENS}"
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

echo "Python: $(python --version 2>&1)"
python - <<'PY'
import torch, vllm
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("vllm", getattr(vllm, "__version__", "unknown"))
print("cuda_available", torch.cuda.is_available())
PY
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Tensor parallel GPUs: ${A2UI_VLLM_GPUS}"
echo "Gemma4 model path: ${GEMMA4_MODEL_PATH}"
echo "vLLM command: ${VLLM_CMD_ARGS[*]}"

exec "${VLLM_CMD_ARGS[@]}"
