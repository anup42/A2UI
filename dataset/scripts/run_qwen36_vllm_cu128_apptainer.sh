#!/usr/bin/env bash
set -euo pipefail

# Start Qwen3.6-35B-A3B through the custom CUDA 12.8 vLLM SIF.
# Run inside a Slurm GPU allocation, for example:
#   srun --gres=gpu:2 --cpus-per-task=16 --mem=160G --pty bash
#   module load apptainer
#   VLLM_SIF=/isilonhome/k_anup/containers/a2ui-vllm-cu128-source_qwen36.sif \
#   bash dataset/scripts/run_qwen36_vllm_cu128_apptainer.sh

QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3.6-35B-A3B}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${HOME}/models/Qwen--Qwen3.6-35B-A3B}"
VLLM_SIF="${VLLM_SIF:-${HOME}/containers/a2ui-vllm-cu128-source_qwen36.sif}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_QUANTIZATION_MODE="${VLLM_QUANTIZATION_MODE:-none}"
VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-}"
VLLM_CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}"
VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-qwen3}"
VLLM_CONTAINER_MODEL_PATH="${VLLM_CONTAINER_MODEL_PATH:-/models/qwen36}"
A2UI_CONTAINER_USER="${A2UI_CONTAINER_USER:-a2ui_user}"
A2UI_CONTAINER_GROUP="${A2UI_CONTAINER_GROUP:-a2ui_group}"
A2UI_BIND_SYNTHETIC_PASSWD="${A2UI_BIND_SYNTHETIC_PASSWD:-1}"

if [[ ! -f "${VLLM_SIF}" ]]; then
  echo "SIF not found: ${VLLM_SIF}" >&2
  exit 1
fi

if [[ ! -d "${QWEN_MODEL_PATH}" ]]; then
  echo "Model folder not found: ${QWEN_MODEL_PATH}" >&2
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

echo "Runtime: ${RUNTIME}"
echo "SIF: ${VLLM_SIF}"
echo "Model path: ${QWEN_MODEL_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Tensor parallel GPUs: ${A2UI_VLLM_GPUS}"
echo "GPU memory utilization: ${VLLM_GPU_MEMORY_UTILIZATION}"
echo "Max model length: ${VLLM_MAX_MODEL_LEN}"
echo "Quantization mode: ${VLLM_QUANTIZATION_MODE}"
echo "KV cache dtype: ${VLLM_KV_CACHE_DTYPE:-auto}"
echo "CPU offload GiB/GPU: ${VLLM_CPU_OFFLOAD_GB:-0}"
echo "Max num seqs: ${VLLM_MAX_NUM_SEQS:-vLLM default}"

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
HOST_HOME="${HOME:-/tmp}"
CONTAINER_USER="$(id -un 2>/dev/null || true)"
CONTAINER_GROUP="$(id -gn 2>/dev/null || true)"
CONTAINER_USER="${CONTAINER_USER:-${A2UI_CONTAINER_USER}}"
CONTAINER_GROUP="${CONTAINER_GROUP:-${A2UI_CONTAINER_GROUP}}"

RUNTIME_ARGS=(exec --nv --cleanenv)
RUNTIME_ARGS+=(--env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}")
RUNTIME_ARGS+=(--env "USER=${CONTAINER_USER}")
RUNTIME_ARGS+=(--env "LOGNAME=${CONTAINER_USER}")
RUNTIME_ARGS+=(--env "HOME=${HOST_HOME}")
RUNTIME_ARGS+=(--bind "${QWEN_MODEL_PATH}:${VLLM_CONTAINER_MODEL_PATH}:ro")

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
  echo "Binding synthetic passwd/group for UID:GID ${HOST_UID}:${HOST_GID} as ${CONTAINER_USER}:${CONTAINER_GROUP}"
fi

if command -v getent >/dev/null 2>&1 && ! getent passwd "${HOST_UID}" >/dev/null 2>&1; then
  echo "Warning: host NSS cannot resolve UID ${HOST_UID}." >&2
  echo "If ${RUNTIME} fails before the container starts with 'unknown userid', the cluster login/NSS layer must be fixed for this UID." >&2
fi

VLLM_CMD_ARGS=(
  vllm serve "${VLLM_CONTAINER_MODEL_PATH}"
  --served-model-name "${QWEN_MODEL_ID}"
  --host "${VLLM_HOST}"
  --port "${VLLM_PORT}"
  --tensor-parallel-size "${A2UI_VLLM_GPUS}"
  --dtype "${VLLM_DTYPE}"
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
  --max-model-len "${VLLM_MAX_MODEL_LEN}"
  --reasoning-parser "${VLLM_REASONING_PARSER}"
  --trust-remote-code
)

case "${VLLM_QUANTIZATION_MODE,,}" in
  ""|none|off|false|0)
    ;;
  int8|bnb-int8|bitsandbytes-int8)
    # BitsAndBytes INT8 reduces model-weight memory. KV cache memory is still
    # controlled separately by max-model-len/max-num-seqs/kv-cache-dtype.
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

if [[ -n "${VLLM_MAX_NUM_SEQS}" ]]; then
  VLLM_CMD_ARGS+=(--max-num-seqs "${VLLM_MAX_NUM_SEQS}")
fi

echo "vLLM command: ${VLLM_CMD_ARGS[*]}"

exec "${RUNTIME}" "${RUNTIME_ARGS[@]}" \
  "${VLLM_SIF}" \
  "${VLLM_CMD_ARGS[@]}"
