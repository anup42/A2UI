#!/usr/bin/env bash
set -euo pipefail

# Runs inside the Singularity/SIF container submitted by jupyterlab/sbatch.sh.

JUPYTER_PORT="${JUPYTER_PORT:-8811}"
JUPYTER_IP="${JUPYTER_IP:-0.0.0.0}"
A2UI_REPO_DIR="${A2UI_REPO_DIR:-${PWD}}"
JUPYTER_WORKDIR="${JUPYTER_WORKDIR:-${A2UI_REPO_DIR}}"
JUPYTER_TOKEN="${JUPYTER_TOKEN:-}"

unset SSL_CERT_FILE || true
unset REQUESTS_CA_BUNDLE || true
unset CURL_CA_BUNDLE || true
export PYTHONHTTPSVERIFY="${PYTHONHTTPSVERIFY:-0}"
export GIT_SSL_NO_VERIFY="${GIT_SSL_NO_VERIFY:-true}"
export HF_HUB_DISABLE_SSL_VERIFICATION="${HF_HUB_DISABLE_SSL_VERIFICATION:-1}"
export NODE_TLS_REJECT_UNAUTHORIZED="${NODE_TLS_REJECT_UNAUTHORIZED:-0}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | paste -sd, -)"
  export CUDA_VISIBLE_DEVICES
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && -z "${A2UI_VLLM_GPUS:-}" ]]; then
  IFS=',' read -r -a _a2ui_gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
  export A2UI_VLLM_GPUS="${#_a2ui_gpu_ids[@]}"
fi

cd "${JUPYTER_WORKDIR}"

echo "A2UI JupyterLab container"
echo "Workdir: ${JUPYTER_WORKDIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "A2UI_VLLM_GPUS=${A2UI_VLLM_GPUS:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi
python --version

pkill -f "python.*jupyter-lab|python.*jupyterlab" 2>/dev/null || true
sleep 2

exec python -m jupyterlab \
  --ip="${JUPYTER_IP}" \
  --port="${JUPYTER_PORT}" \
  --no-browser \
  --ServerApp.allow_remote_access=True \
  --ServerApp.token="${JUPYTER_TOKEN}" \
  --ServerApp.password="" \
  --IdentityProvider.token="${JUPYTER_TOKEN}"
