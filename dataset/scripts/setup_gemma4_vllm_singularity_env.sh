#!/usr/bin/env bash
set -euo pipefail

# Prepare the host-side A2UI dataset runner environment for Gemma4 31B served by
# vLLM inside a Singularity container. The vLLM server itself should run from
# run_gemma4_vllm_singularity.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ENV_DIR="${ENV_DIR:-${REPO_ROOT}/gemma4_vllm_env}"
VLLM_SIF="${VLLM_SIF:-${HOME}/containers/a2ui-vllm-cu128-source_gemma4_speculative.sif}"
GEMMA4_MODEL_PATH="${GEMMA4_MODEL_PATH:-${HOME}/dataset_generation/models/gemma4-31b}"
GEMMA4_DRAFT_MODEL_PATH="${GEMMA4_DRAFT_MODEL_PATH:-}"
A2UI_DISABLE_SSL_VERIFY="${A2UI_DISABLE_SSL_VERIFY:-1}"
A2UI_SKIP_PIP_INSTALL="${A2UI_SKIP_PIP_INSTALL:-0}"
A2UI_CHECK_CONTAINER="${A2UI_CHECK_CONTAINER:-1}"

if ! command -v singularity >/dev/null 2>&1; then
  echo "singularity is required on this machine." >&2
  exit 1
fi

echo "Singularity: $(singularity --version 2>&1)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "Warning: nvidia-smi not found on PATH." >&2
fi

if [[ ! -d "${ENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${ENV_DIR}"
fi

# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel setuptools

if [[ "${A2UI_SKIP_PIP_INSTALL}" != "1" ]]; then
  PIP_ARGS=()
  if [[ "${A2UI_DISABLE_SSL_VERIFY}" = "1" ]]; then
    export CURL_CA_BUNDLE=""
    export REQUESTS_CA_BUNDLE=""
    export SSL_CERT_FILE=""
    PIP_ARGS+=(
      --trusted-host pypi.org
      --trusted-host files.pythonhosted.org
      --trusted-host download.pytorch.org
      --trusted-host github.com
      --trusted-host codeload.github.com
      --trusted-host raw.githubusercontent.com
    )
  fi
  python -m pip install "${PIP_ARGS[@]}" \
    "pyyaml>=6.0.2" \
    "requests>=2.32.0" \
    "numpy>=1.26" \
    "pillow>=10.0" \
    "matplotlib>=3.8" \
    "jsonschema>=4.23" \
    "referencing>=0.35" \
    "openai>=1.60" \
    "httpx>=0.27" \
    "truststore>=0.10" \
    "tqdm>=4.66"
fi

cat > "${ENV_DIR}/activate_gemma4_vllm.sh" <<EOF
#!/usr/bin/env bash
source "${ENV_DIR}/bin/activate"
export VLLM_SIF="${VLLM_SIF}"
export GEMMA4_MODEL_PATH="${GEMMA4_MODEL_PATH}"
export GEMMA4_DRAFT_MODEL_PATH="${GEMMA4_DRAFT_MODEL_PATH}"
export GEMMA4_MODEL_ID="\${GEMMA4_MODEL_ID:-google/gemma-4-31b-it}"
export CUDA_VISIBLE_DEVICES="\${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export A2UI_VLLM_GPUS="\${A2UI_VLLM_GPUS:-4}"
export VLLM_MAX_MODEL_LEN="\${VLLM_MAX_MODEL_LEN:-8192}"
export VLLM_GPU_MEMORY_UTILIZATION="\${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
export GEMMA4_ENABLE_REASONING="\${GEMMA4_ENABLE_REASONING:-0}"
export GEMMA4_SPECULATIVE_MODE="\${GEMMA4_SPECULATIVE_MODE:-draft}"
export GEMMA4_SPECULATIVE_TOKENS="\${GEMMA4_SPECULATIVE_TOKENS:-4}"
export LOCAL_ALLOW_HTTP_ENDPOINT=1
export LOCAL_STRICT_OFFLINE=0
export LOCAL_VLLM_STRIP_THINKING=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
EOF
chmod +x "${ENV_DIR}/activate_gemma4_vllm.sh"

if [[ "${A2UI_CHECK_CONTAINER}" = "1" && -f "${VLLM_SIF}" ]]; then
  echo "Checking vLLM speculative flags inside ${VLLM_SIF}"
  HELP_TEXT="$(singularity exec --nv "${VLLM_SIF}" vllm serve --help 2>&1 || true)"
  if ! grep -q -- "--speculative-model" <<<"${HELP_TEXT}"; then
    echo "Warning: vLLM help did not expose --speculative-model. Speculative decoding may not be available in this image." >&2
  fi
  if ! grep -q -- "--num-speculative-tokens" <<<"${HELP_TEXT}"; then
    echo "Warning: vLLM help did not expose --num-speculative-tokens. Speculative decoding may not be available in this image." >&2
  fi
fi

echo
echo "Environment ready."
echo "Activate with:"
echo "  source ${ENV_DIR}/activate_gemma4_vllm.sh"
echo
echo "Start Gemma4 vLLM with:"
echo "  bash dataset/scripts/run_gemma4_vllm_singularity.sh"
echo
echo "Run Stage 1/2/3 with:"
echo "  bash dataset/scripts/run_gemma4_dataset_stages.sh"
