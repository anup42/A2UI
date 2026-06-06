#!/usr/bin/env bash
set -euo pipefail

# Install the Gemma4/vLLM Python environment from a copied offline bundle.
# The bundle is created on an internet PC by:
#   bash dataset/scripts/download_gemma4_vllm_offline_bundle.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

BUNDLE_DIR="${1:-${BUNDLE_DIR:-${PWD}/gemma4_vllm_offline_bundle}}"
WHEELHOUSE="${BUNDLE_DIR}/wheelhouse"

if [[ ! -d "${WHEELHOUSE}" ]]; then
  echo "Wheelhouse not found: ${WHEELHOUSE}" >&2
  echo "Copy the full gemma4_vllm_offline_bundle from the internet PC first." >&2
  exit 1
fi

export A2UI_OFFLINE_WHEELHOUSE="${WHEELHOUSE}"
export A2UI_DISABLE_SSL_VERIFY="${A2UI_DISABLE_SSL_VERIFY:-1}"
export A2UI_DISABLE_PROXY="${A2UI_DISABLE_PROXY:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

bash dataset/scripts/setup_gemma4_vllm_python_env.sh

echo "Offline Gemma4 vLLM env ready."
echo "Activate with:"
echo "  source ${ENV_DIR:-${REPO_ROOT}/gemma4_vllm_env}/activate_gemma4_vllm.sh"
