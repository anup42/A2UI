#!/usr/bin/env bash
set -euo pipefail

# Host-side dataset runner environment for the CUDA 13.0 Gemma4 speculative SIF.
# The heavy vLLM runtime lives in the SIF; this venv runs dataset Stage 1-5
# commands against the local OpenAI-compatible vLLM server.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ENV_DIR="${ENV_DIR:-${REPO_ROOT}/gemma4_vllm_cu130_env}"
VLLM_SIF="${VLLM_SIF:-${HOME}/containers/a2ui-vllm-cu130-source_gemma4_speculative_ada.sif}"
GEMMA4_MODEL_PATH="${GEMMA4_MODEL_PATH:-${HOME}/dataset_generation/models/gemma4-31b}"
GEMMA4_DRAFT_MODEL_PATH="${GEMMA4_DRAFT_MODEL_PATH:-}"
A2UI_DISABLE_SSL_VERIFY="${A2UI_DISABLE_SSL_VERIFY:-1}"
A2UI_SKIP_PIP_INSTALL="${A2UI_SKIP_PIP_INSTALL:-0}"

if ! command -v singularity >/dev/null 2>&1 && ! command -v apptainer >/dev/null 2>&1; then
  echo "singularity or apptainer is required on this machine." >&2
  exit 1
fi

echo "A2UI repo: ${REPO_ROOT}"
echo "Target env: ${ENV_DIR}"
echo "Target SIF: ${VLLM_SIF}"
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

PIP_ARGS=()
if [[ "${A2UI_DISABLE_SSL_VERIFY}" = "1" ]]; then
  export PYTHONHTTPSVERIFY=0
  export GIT_SSL_NO_VERIFY=true
  export CURL_CA_BUNDLE=""
  export REQUESTS_CA_BUNDLE=""
  export SSL_CERT_FILE=""
  export PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org download.pytorch.org github.com codeload.github.com raw.githubusercontent.com objects.githubusercontent.com release-assets.githubusercontent.com huggingface.co cdn-lfs.huggingface.co"
  PIP_ARGS=(
    --trusted-host pypi.org
    --trusted-host files.pythonhosted.org
    --trusted-host download.pytorch.org
    --trusted-host github.com
    --trusted-host codeload.github.com
    --trusted-host raw.githubusercontent.com
    --trusted-host objects.githubusercontent.com
    --trusted-host release-assets.githubusercontent.com
    --trusted-host huggingface.co
    --trusted-host cdn-lfs.huggingface.co
  )
fi

if [[ "${A2UI_SKIP_PIP_INSTALL}" != "1" ]]; then
  python -m pip install "${PIP_ARGS[@]}" \
    "aiohttp>=3.9" \
    "beautifulsoup4>=4.12" \
    "httpx>=0.27" \
    "imageio>=2.34" \
    "jinja2>=3.1" \
    "jsonschema>=4.23" \
    "lxml>=5.2" \
    "matplotlib>=3.8" \
    "numpy>=1.26" \
    "openai>=1.60" \
    "opencv-python-headless>=4.9" \
    "pandas>=2.2" \
    "pillow>=10.0" \
    "playwright>=1.45" \
    "pyarrow>=15.0" \
    "python-dotenv>=1.0" \
    "pyyaml>=6.0.2" \
    "referencing>=0.35" \
    "requests>=2.32.0" \
    "rich>=13.7" \
    "scikit-learn>=1.4" \
    "scipy>=1.12" \
    "tenacity>=8.3" \
    "tqdm>=4.66" \
    "truststore>=0.10"
fi

cat > "${ENV_DIR}/activate_gemma4_vllm_cu130.sh" <<EOF
#!/usr/bin/env bash
source "${ENV_DIR}/bin/activate"
export VLLM_SIF="${VLLM_SIF}"
export GEMMA4_MODEL_PATH="${GEMMA4_MODEL_PATH}"
export GEMMA4_DRAFT_MODEL_PATH="${GEMMA4_DRAFT_MODEL_PATH}"
export GEMMA4_MODEL_ID="\${GEMMA4_MODEL_ID:-google/gemma-4-31b-it}"
if command -v nvidia-smi >/dev/null 2>&1 && [ -z "\${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="\$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | paste -sd, -)"
fi
if [ -n "\${CUDA_VISIBLE_DEVICES:-}" ] && [ -z "\${A2UI_VLLM_GPUS:-}" ]; then
  IFS=',' read -r -a _a2ui_gpu_ids <<< "\${CUDA_VISIBLE_DEVICES}"
  export A2UI_VLLM_GPUS="\${#_a2ui_gpu_ids[@]}"
fi
export VLLM_MAX_MODEL_LEN="\${VLLM_MAX_MODEL_LEN:-8192}"
export VLLM_GPU_MEMORY_UTILIZATION="\${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
export VLLM_DTYPE="\${VLLM_DTYPE:-bfloat16}"
export GEMMA4_ENABLE_REASONING="\${GEMMA4_ENABLE_REASONING:-0}"
export GEMMA4_SPECULATIVE_MODE="\${GEMMA4_SPECULATIVE_MODE:-draft}"
export GEMMA4_SPECULATIVE_TOKENS="\${GEMMA4_SPECULATIVE_TOKENS:-4}"
export LOCAL_ALLOW_HTTP_ENDPOINT=1
export LOCAL_STRICT_OFFLINE=0
export LOCAL_VLLM_STRIP_THINKING=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export A2UI_DISABLE_SSL_VERIFY=1
export PYTHONHTTPSVERIFY=0
export GIT_SSL_NO_VERIFY=true
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""
export SSL_CERT_FILE=""
EOF
chmod +x "${ENV_DIR}/activate_gemma4_vllm_cu130.sh"

echo
echo "Environment ready."
echo "Activate with:"
echo "  source ${ENV_DIR}/activate_gemma4_vllm_cu130.sh"
echo
echo "Start Gemma4 CU130 vLLM with:"
echo "  bash dataset/scripts/run_gemma4_vllm_cu130_singularity.sh"
echo
echo "Run Stage 1-5 with:"
echo "  STAGE=all bash dataset/scripts/run_gemma4_dataset_stages.sh"
