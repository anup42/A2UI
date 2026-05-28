#!/usr/bin/env bash
#SBATCH --job-name=a2ui-qwen36-50
#SBATCH --output=dataset/data/runs/qwen36_stage123_50-%j.out
#SBATCH --error=dataset/data/runs/qwen36_stage123_50-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00

set -euo pipefail

# Submit from the A2UI repo root:
#   sbatch dataset/scripts/qwen36_stage123_50.sbatch
#
# Optional overrides:
#   RUN_ID=my_qwen36_50 sbatch dataset/scripts/qwen36_stage123_50.sbatch

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"

RUN_ID="${RUN_ID:-dataset_qwen36_slurm_50}"
TARGET_COUNT="${TARGET_COUNT:-50}"
MODEL_NAME="${MODEL_NAME:-qwen36_35b_a3b_vllm_reasoning}"
RATE_LIMIT_QPS="${RATE_LIMIT_QPS:-0.2}"
STAGE3_BATCH_SIZE="${STAGE3_BATCH_SIZE:-1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-900}"
QWEN_ENV_ACTIVATE="${QWEN_ENV_ACTIVATE:-${REPO_ROOT}/qwen_vllm_env/activate_qwen_vllm.sh}"

mkdir -p "dataset/data/runs/${RUN_ID}"

if [[ -f "${QWEN_ENV_ACTIVATE}" ]]; then
  # shellcheck source=/dev/null
  source "${QWEN_ENV_ACTIVATE}"
else
  echo "Qwen env activation script not found: ${QWEN_ENV_ACTIVATE}" >&2
  exit 1
fi

export QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${HOME}/models/Qwen--Qwen3.6-35B-A3B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export A2UI_VLLM_GPUS="${A2UI_VLLM_GPUS:-2}"
export LOCAL_ALLOW_HTTP_ENDPOINT="${LOCAL_ALLOW_HTTP_ENDPOINT:-1}"
export LOCAL_STRICT_OFFLINE="${LOCAL_STRICT_OFFLINE:-0}"
export LOCAL_VLLM_ENABLE_THINKING="${LOCAL_VLLM_ENABLE_THINKING:-1}"
export VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-qwen3}"

vllm_ready() {
  curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1
}

echo "Starting vLLM for ${RUN_ID}"
bash dataset/scripts/run_qwen36_vllm_server.sh > "dataset/data/runs/${RUN_ID}/vllm_server.log" 2>&1 &
VLLM_PID="$!"
trap 'kill "${VLLM_PID}" 2>/dev/null || true' EXIT

deadline=$((SECONDS + VLLM_READY_TIMEOUT))
until vllm_ready; do
  if (( SECONDS >= deadline )); then
    echo "vLLM did not become ready within ${VLLM_READY_TIMEOUT}s" >&2
    tail -120 "dataset/data/runs/${RUN_ID}/vllm_server.log" >&2 || true
    exit 1
  fi
  sleep 5
done

echo "Running Stage 1: ${TARGET_COUNT} queries"
STAGE=1 \
RUN_ID="${RUN_ID}" \
MODEL_NAME="${MODEL_NAME}" \
RATE_LIMIT_QPS="${RATE_LIMIT_QPS}" \
MAX_QUERIES_TOTAL="${TARGET_COUNT}" \
bash dataset/scripts/run_qwen36_dataset_stages.sh

echo "Running Stage 2: ${TARGET_COUNT} responses"
STAGE=2 \
RUN_ID="${RUN_ID}" \
MODEL_NAME="${MODEL_NAME}" \
RATE_LIMIT_QPS="${RATE_LIMIT_QPS}" \
MAX_RESPONSES_TOTAL="${TARGET_COUNT}" \
bash dataset/scripts/run_qwen36_dataset_stages.sh

echo "Running Stage 3: ${TARGET_COUNT} IR records"
STAGE=3 \
RUN_ID="${RUN_ID}" \
MODEL_NAME="${MODEL_NAME}" \
RATE_LIMIT_QPS="${RATE_LIMIT_QPS}" \
STAGE3_BATCH_SIZE="${STAGE3_BATCH_SIZE}" \
MAX_GENUI_TOTAL="${TARGET_COUNT}" \
bash dataset/scripts/run_qwen36_dataset_stages.sh

echo "Complete: dataset/data/runs/${RUN_ID}"
