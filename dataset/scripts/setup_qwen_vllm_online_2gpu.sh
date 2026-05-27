#!/usr/bin/env bash
set -euo pipefail

# Online setup for a 2-GPU A100 machine. This script creates a Python 3.11
# environment, installs dataset/vLLM dependencies, downloads the Qwen model, and
# writes an activation file with the expected A2UI dataset-generation env vars.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
ENV_DIR="${ENV_DIR:-${REPO_ROOT}/qwen_vllm_env}"
REQUESTED_ENV_DIR="${ENV_DIR}"
MINIFORGE_DIR="${MINIFORGE_DIR:-${HOME}/miniforge3}"
QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3.6-35B-A3B}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${REPO_ROOT}/qwen_models/${QWEN_MODEL_ID//\//--}}"
REQ_FILE="${REQ_FILE:-${REPO_ROOT}/dataset/requirements-qwen-vllm.txt}"
A2UI_CA_BUNDLE="${A2UI_CA_BUNDLE:-}"
A2UI_DISABLE_SSL_VERIFY="${A2UI_DISABLE_SSL_VERIFY:-1}"
export A2UI_DISABLE_SSL_VERIFY

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export A2UI_VLLM_GPUS="${A2UI_VLLM_GPUS:-2}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
export VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"

python_supported() {
  "$1" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)
PY
}

python_version_string() {
  "$1" - <<'PY' 2>/dev/null || true
import sys
print(sys.version.split()[0])
PY
}

env_ready() {
  [[ -f "$1/bin/activate" ]] && [[ -x "$1/bin/python" ]] && python_supported "$1/bin/python"
}

if [[ -d "${ENV_DIR}" ]] && ! env_ready "${ENV_DIR}"; then
  old_version="missing"
  if [[ -x "${ENV_DIR}/bin/python" ]]; then
    old_version="$(python_version_string "${ENV_DIR}/bin/python")"
  fi
  ENV_DIR="${REQUESTED_ENV_DIR}_py311"
  echo "Existing env ${REQUESTED_ENV_DIR} is incomplete or unsupported (${old_version}); using ${ENV_DIR} instead." >&2
  if [[ -d "${ENV_DIR}" ]] && ! env_ready "${ENV_DIR}"; then
    ENV_DIR="${REQUESTED_ENV_DIR}_py311_$(date +%Y%m%d_%H%M%S)"
    echo "Fallback env is also incomplete or unsupported; using ${ENV_DIR} instead." >&2
  fi
fi

echo "A2UI repo: ${REPO_ROOT}"
echo "Target env: ${ENV_DIR}"
echo "Target model: ${QWEN_MODEL_ID}"
echo "Target model path: ${QWEN_MODEL_PATH}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found; continue only if this is intentional." >&2
fi

create_with_python311() {
  python3.11 -m venv "${ENV_DIR}"
  # shellcheck source=/dev/null
  source "${ENV_DIR}/bin/activate"
}

create_with_conda() {
  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck source=/dev/null
  source "${conda_base}/etc/profile.d/conda.sh"
  conda create -y -p "${ENV_DIR}" "python=${PYTHON_VERSION}" pip
  conda activate "${ENV_DIR}"
}

install_miniforge_and_create_env() {
  local installer="/tmp/miniforge_a2ui.sh"
  if [[ ! -x "${MINIFORGE_DIR}/bin/conda" ]]; then
    echo "Installing Miniforge to ${MINIFORGE_DIR}"
    local curl_ssl_args=()
    if [[ "${A2UI_DISABLE_SSL_VERIFY}" == "1" ]]; then
      curl_ssl_args=(-k)
    fi
    curl -L "${curl_ssl_args[@]}" -o "${installer}" \
      "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash "${installer}" -b -p "${MINIFORGE_DIR}"
  fi
  # shellcheck source=/dev/null
  source "${MINIFORGE_DIR}/etc/profile.d/conda.sh"
  conda create -y -p "${ENV_DIR}" "python=${PYTHON_VERSION}" pip
  conda activate "${ENV_DIR}"
}

if env_ready "${ENV_DIR}"; then
  # shellcheck source=/dev/null
  source "${ENV_DIR}/bin/activate"
else
  if command -v python3.11 >/dev/null 2>&1; then
    create_with_python311
  elif command -v conda >/dev/null 2>&1; then
    create_with_conda
  else
    install_miniforge_and_create_env
  fi
fi

python - <<'PY'
import sys
if sys.version_info < (3, 10) or sys.version_info >= (3, 13):
    raise SystemExit(
        f"vLLM setup requires Python 3.10-3.12; current is {sys.version.split()[0]}"
    )
print("python:", sys.version.split()[0])
PY

PIP_SSL_ARGS=()
if [[ "${A2UI_DISABLE_SSL_VERIFY}" == "1" ]]; then
  echo "WARNING: A2UI_DISABLE_SSL_VERIFY=1; TLS certificate verification is disabled for setup downloads." >&2
  export PYTHONHTTPSVERIFY=0
  export CURL_SSL_BACKEND=openssl
  PIP_SSL_ARGS=(
    --trusted-host pypi.org
    --trusted-host files.pythonhosted.org
    --trusted-host huggingface.co
    --trusted-host cdn-lfs.huggingface.co
  )
fi

python -m pip install "${PIP_SSL_ARGS[@]}" --upgrade pip setuptools wheel
python -m pip install "${PIP_SSL_ARGS[@]}" -r "${REQ_FILE}"

if [[ -z "${A2UI_CA_BUNDLE}" ]]; then
  for candidate in \
    /etc/ssl/certs/ca-certificates.crt \
    /etc/pki/tls/certs/ca-bundle.crt \
    /etc/ssl/cert.pem; do
    if [[ -f "${candidate}" ]]; then
      A2UI_CA_BUNDLE="${candidate}"
      break
    fi
  done
fi
if [[ "${A2UI_DISABLE_SSL_VERIFY}" == "1" ]]; then
  echo "Skipping CA bundle setup because SSL verification is disabled." >&2
elif [[ -n "${A2UI_CA_BUNDLE}" && -f "${A2UI_CA_BUNDLE}" ]]; then
  export SSL_CERT_FILE="${A2UI_CA_BUNDLE}"
  export REQUESTS_CA_BUNDLE="${A2UI_CA_BUNDLE}"
  export CURL_CA_BUNDLE="${A2UI_CA_BUNDLE}"
  export GIT_SSL_CAINFO="${A2UI_CA_BUNDLE}"
  echo "Using CA bundle: ${A2UI_CA_BUNDLE}"
else
  echo "No CA bundle detected. If SSL fails, set A2UI_CA_BUNDLE=/path/to/company-root-ca.pem" >&2
fi

mkdir -p "$(dirname "${QWEN_MODEL_PATH}")"
export QWEN_MODEL_ID QWEN_MODEL_PATH
python - <<'PY'
import os
if os.environ.get("A2UI_DISABLE_SSL_VERIFY") == "1":
    import requests
    import urllib3
    from huggingface_hub import configure_http_backend

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def backend_factory():
        session = requests.Session()
        session.verify = False
        return session

    configure_http_backend(backend_factory=backend_factory)
else:
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception as exc:
        print(f"truststore injection skipped: {exc}")
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["QWEN_MODEL_ID"],
    local_dir=os.environ["QWEN_MODEL_PATH"],
    resume_download=True,
)
PY

cat > "${ENV_DIR}/activate_qwen_vllm.sh" <<EOF
#!/usr/bin/env bash
source "${ENV_DIR}/bin/activate"
export QWEN_MODEL_PATH="${QWEN_MODEL_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export A2UI_VLLM_GPUS="${A2UI_VLLM_GPUS}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION}"
export VLLM_DTYPE="${VLLM_DTYPE}"
export LOCAL_ALLOW_HTTP_ENDPOINT=1
export LOCAL_STRICT_OFFLINE=0
export LOCAL_VLLM_ENABLE_THINKING=1
export VLLM_REASONING_PARSER=qwen3
EOF
chmod +x "${ENV_DIR}/activate_qwen_vllm.sh"

python - <<'PY'
import importlib.metadata as md
import torch

for package in ["torch", "vllm", "transformers", "huggingface_hub"]:
    print(f"{package}:", md.version(package))
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
for idx in range(torch.cuda.device_count()):
    print(f"gpu {idx}:", torch.cuda.get_device_name(idx))
PY

echo
echo "Setup complete."
echo "Activate:"
echo "  source ${ENV_DIR}/activate_qwen_vllm.sh"
echo
echo "Start vLLM:"
echo "  bash dataset/scripts/run_qwen36_vllm_server.sh"
echo
echo "Run stages:"
echo "  RUN_ID=dataset_qwen36_vllm_reasoning_v0 bash dataset/scripts/run_qwen36_dataset_stages.sh"
