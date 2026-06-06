#!/usr/bin/env bash
set -euo pipefail

# Recreate the Gemma4 vLLM Python environment with the intended current stack:
#   torch 2.11.0 + CUDA 13.0/cu130 + latest vLLM nightly
#
# Use this when an existing env contains a stale/mixed vLLM/Torch stack such as
# torch 2.9.x + older vLLM and fails with undefined symbols in vllm/_C.abi3.so.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ENV_DIR="${ENV_DIR:-${REPO_ROOT}/gemma4_vllm_env}"
ENV_DIR="$("${PYTHON_BIN}" - "${ENV_DIR}" <<'PY'
import os
import sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
)"
REPO_ROOT_ABS="$("${PYTHON_BIN}" - "${REPO_ROOT}" <<'PY'
import os
import sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
)"
HOME_ABS="$("${PYTHON_BIN}" - <<'PY'
import os
print(os.path.abspath(os.path.expanduser("~")))
PY
)"

case "${ENV_DIR}" in
  ""|"/"|"/home"|"/home/"*"/.."*)
    echo "Refusing unsafe ENV_DIR: ${ENV_DIR}" >&2
    exit 1
    ;;
esac
if [[ "${ENV_DIR}" == "${HOME_ABS}" || "${ENV_DIR}" == "${REPO_ROOT_ABS}" ]]; then
  echo "Refusing to delete ENV_DIR=${ENV_DIR}" >&2
  echo "Set ENV_DIR to a dedicated virtualenv directory." >&2
  exit 1
fi

echo "Recreating Gemma4 vLLM env:"
echo "  ENV_DIR=${ENV_DIR}"
echo "  stack=torch 2.11.0 + cu130 + latest vLLM nightly"
if [[ -d "${ENV_DIR}" ]]; then
  rm -rf "${ENV_DIR}"
fi

export ENV_DIR
export A2UI_CLEAN_VLLM_STACK=1
export A2UI_REQUIRE_SPECULATIVE="${A2UI_REQUIRE_SPECULATIVE:-1}"
export A2UI_VLLM_INSTALL_MODE="${A2UI_VLLM_INSTALL_MODE:-nightly}"
export A2UI_TORCH_BACKEND="${A2UI_TORCH_BACKEND:-cu130}"
export TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
export TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
export TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0}"
export PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
export VLLM_NIGHTLY_INDEX="${VLLM_NIGHTLY_INDEX:-https://wheels.vllm.ai/nightly/cu130}"
export FLASHINFER_CUDA_TAG="${FLASHINFER_CUDA_TAG:-cu130}"
export A2UI_DISABLE_SSL_VERIFY="${A2UI_DISABLE_SSL_VERIFY:-1}"

bash "${SCRIPT_DIR}/setup_gemma4_vllm_python_env.sh"

echo
echo "Recreated ${ENV_DIR}"
echo "Start with:"
echo "  ENV_DIR=${ENV_DIR} GEMMA4_REQUIRE_SPECULATIVE=1 GEMMA4_SPECULATIVE_MODE=draft bash dataset/scripts/run_gemma4_vllm_python_env_first.sh"
