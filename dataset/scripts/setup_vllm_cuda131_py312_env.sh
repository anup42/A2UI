#!/usr/bin/env bash
set -euo pipefail

# Create an internet-machine Python environment for A2UI local vLLM serving.
#
# Target host:
#   - Python 3.12.3
#   - NVIDIA driver/CUDA runtime capable of CUDA 13.1
#   - Internet access
#
# Notes:
#   - vLLM currently publishes CUDA-13 wheels as the cu130 variant. A CUDA 13.1
#     host/driver can run this stack as long as the NVIDIA driver is compatible.
#   - This script installs the latest vLLM by default. Use
#     A2UI_VLLM_CHANNEL=nightly if you specifically need current main-branch
#     features before the next PyPI release.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
REQUIRED_PYTHON_VERSION="${REQUIRED_PYTHON_VERSION:-3.12.3}"
ENV_DIR="${ENV_DIR:-${REPO_ROOT}/vllm_cu131_py312_env}"

# release = latest PyPI release, nightly = latest vLLM nightly wheel.
A2UI_VLLM_CHANNEL="${A2UI_VLLM_CHANNEL:-release}" # release|nightly|source
A2UI_VLLM_NIGHTLY_VARIANT="${A2UI_VLLM_NIGHTLY_VARIANT:-cu130}"
VLLM_NIGHTLY_INDEX="${VLLM_NIGHTLY_INDEX:-https://wheels.vllm.ai/nightly/${A2UI_VLLM_NIGHTLY_VARIANT}}"
VLLM_SOURCE_REF="${VLLM_SOURCE_REF:-main}"
VLLM_SOURCE_DIR="${VLLM_SOURCE_DIR:-${ENV_DIR}/src/vllm}"

A2UI_INSTALL_DATASET_DEPS="${A2UI_INSTALL_DATASET_DEPS:-1}"
A2UI_INSTALL_FLASHINFER="${A2UI_INSTALL_FLASHINFER:-0}"
FLASHINFER_CUDA_TAG="${FLASHINFER_CUDA_TAG:-cu130}"
FLASHINFER_INDEX_URL="${FLASHINFER_INDEX_URL:-https://flashinfer.ai/whl/${FLASHINFER_CUDA_TAG}}"

# SSL bypass defaults to on because several target machines in this project sit
# behind intercepting or incomplete certificate chains. Set to 0 on clean hosts.
A2UI_DISABLE_SSL_VERIFY="${A2UI_DISABLE_SSL_VERIFY:-1}"
A2UI_DISABLE_PROXY="${A2UI_DISABLE_PROXY:-0}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

apply_network_flags() {
  if [[ "${A2UI_DISABLE_SSL_VERIFY}" = "1" ]]; then
    export CURL_CA_BUNDLE=""
    export REQUESTS_CA_BUNDLE=""
    export SSL_CERT_FILE=""
    export PYTHONHTTPSVERIFY=0
    export GIT_SSL_NO_VERIFY=1
    export HF_HUB_DISABLE_SSL_VERIFICATION=1
    git config --global http.sslVerify false >/dev/null 2>&1 || true
  fi
  if [[ "${A2UI_DISABLE_PROXY}" = "1" ]]; then
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy PIP_PROXY UV_HTTP_PROXY UV_HTTPS_PROXY
    export no_proxy="127.0.0.1,localhost,::1${no_proxy:+,${no_proxy}}"
    export NO_PROXY="127.0.0.1,localhost,::1${NO_PROXY:+,${NO_PROXY}}"
  fi
}

python_version="$("${PYTHON_BIN}" - <<'PY'
import sys
print(".".join(map(str, sys.version_info[:3])))
PY
)" || die "Could not execute ${PYTHON_BIN}. Set PYTHON_BIN=/path/to/python3.12"

if [[ "${python_version}" != "${REQUIRED_PYTHON_VERSION}" ]]; then
  die "${PYTHON_BIN} is Python ${python_version}; expected ${REQUIRED_PYTHON_VERSION}. Set PYTHON_BIN to Python 3.12.3 or override REQUIRED_PYTHON_VERSION deliberately."
fi

ENV_DIR="$("${PYTHON_BIN}" - "${ENV_DIR}" <<'PY'
import os
import sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
)"

case "${ENV_DIR}" in
  ""|"/"|"/home"|"/home/"*"/.."*)
    die "Refusing unsafe ENV_DIR=${ENV_DIR}"
    ;;
esac

echo "Creating vLLM CUDA 13.1-host environment"
echo "  repo: ${REPO_ROOT}"
echo "  python: ${PYTHON_BIN} (${python_version})"
echo "  env: ${ENV_DIR}"
echo "  vLLM channel: ${A2UI_VLLM_CHANNEL}"
echo "  nightly index: ${VLLM_NIGHTLY_INDEX}"
echo "  SSL verify disabled: ${A2UI_DISABLE_SSL_VERIFY}"
echo "  proxy disabled: ${A2UI_DISABLE_PROXY}"

apply_network_flags

"${PYTHON_BIN}" -m venv "${ENV_DIR}"
# shellcheck source=/dev/null
source "${ENV_DIR}/bin/activate"
hash -r

export PYTHONNOUSERSITE=1
unset PYTHONHOME

python -m pip install -U pip setuptools wheel
python -m pip install -U uv

UV_INSECURE_ARGS=()
if [[ "${A2UI_DISABLE_SSL_VERIFY}" = "1" ]]; then
  UV_INSECURE_ARGS+=(--allow-insecure-host pypi.org)
  UV_INSECURE_ARGS+=(--allow-insecure-host files.pythonhosted.org)
  UV_INSECURE_ARGS+=(--allow-insecure-host wheels.vllm.ai)
  UV_INSECURE_ARGS+=(--allow-insecure-host download.pytorch.org)
  UV_INSECURE_ARGS+=(--allow-insecure-host flashinfer.ai)
  UV_INSECURE_ARGS+=(--allow-insecure-host github.com)
  UV_INSECURE_ARGS+=(--allow-insecure-host objects.githubusercontent.com)
fi

uv_pip_install() {
  python -m uv pip install "${UV_INSECURE_ARGS[@]}" "$@"
}

case "${A2UI_VLLM_CHANNEL}" in
  release)
    # Official vLLM docs recommend uv and --torch-backend=auto so the vLLM
    # wheel resolves its compatible PyTorch/CUDA stack.
    uv_pip_install -U vllm --torch-backend=auto
    ;;
  nightly)
    # vLLM docs list CUDA-13 nightlies under cu130. There is no separate cu131
    # wheel index at the time this script was added.
    uv_pip_install -U vllm --pre \
      --torch-backend=auto \
      --extra-index-url "${VLLM_NIGHTLY_INDEX}"
    ;;
  source)
    mkdir -p "$(dirname "${VLLM_SOURCE_DIR}")"
    if [[ ! -d "${VLLM_SOURCE_DIR}/.git" ]]; then
      git clone https://github.com/vllm-project/vllm.git "${VLLM_SOURCE_DIR}"
    fi
    git -C "${VLLM_SOURCE_DIR}" fetch --all --tags
    git -C "${VLLM_SOURCE_DIR}" checkout "${VLLM_SOURCE_REF}"
    uv_pip_install -e "${VLLM_SOURCE_DIR}" --torch-backend=auto
    ;;
  *)
    die "Unsupported A2UI_VLLM_CHANNEL=${A2UI_VLLM_CHANNEL}; use release, nightly, or source."
    ;;
esac

if [[ "${A2UI_INSTALL_DATASET_DEPS}" = "1" ]]; then
  tmp_req="$(mktemp)"
  grep -Ev '^[[:space:]]*(vllm|torch|torchvision|torchaudio)([<>=!~ ].*)?$' \
    "${REPO_ROOT}/dataset/requirements-qwen-vllm.txt" > "${tmp_req}"
  uv_pip_install -U -r "${tmp_req}"
  rm -f "${tmp_req}"
fi

if [[ "${A2UI_INSTALL_FLASHINFER}" = "1" ]]; then
  uv_pip_install -U flashinfer-python \
    --extra-index-url "${FLASHINFER_INDEX_URL}" || {
      echo "Warning: FlashInfer install failed from ${FLASHINFER_INDEX_URL}. Continuing without FlashInfer." >&2
    }
fi

cat > "${ENV_DIR}/activate_vllm_cuda131_py312.sh" <<EOF
#!/usr/bin/env bash
source "${ENV_DIR}/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
export A2UI_DISABLE_SSL_VERIFY="\${A2UI_DISABLE_SSL_VERIFY:-${A2UI_DISABLE_SSL_VERIFY}}"
export FLASHINFER_DISABLE_VERSION_CHECK="\${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export FLASHINFER_DISABLE_VERSION__CHECK="\${FLASHINFER_DISABLE_VERSION__CHECK:-1}"
export LOCAL_ALLOW_HTTP_ENDPOINT="\${LOCAL_ALLOW_HTTP_ENDPOINT:-1}"
EOF
chmod +x "${ENV_DIR}/activate_vllm_cuda131_py312.sh"

echo
echo "Environment verification"
python - <<'PY'
import importlib.util
import shutil
import subprocess
import sys

print("python:", sys.executable)
print("python_version:", sys.version.replace("\n", " "))

for name in ("torch", "vllm", "transformers", "jsonschema", "openai"):
    spec = importlib.util.find_spec(name)
    print(f"{name}_origin:", spec.origin if spec else "NOT_FOUND")

try:
    import torch
    print("torch_version:", torch.__version__)
    print("torch_cuda:", getattr(torch.version, "cuda", None))
    print("torch_cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
      print("torch_cuda_device_count:", torch.cuda.device_count())
except Exception as exc:
    print("torch_import_error:", repr(exc))

try:
    import vllm
    print("vllm_version:", getattr(vllm, "__version__", "unknown"))
except Exception as exc:
    print("vllm_import_error:", repr(exc))

if shutil.which("nvidia-smi"):
    try:
        print("nvidia_smi:")
        print(subprocess.check_output(["nvidia-smi"], text=True, timeout=10))
    except Exception as exc:
        print("nvidia_smi_error:", repr(exc))
PY

echo
echo "Checking vLLM server flags"
vllm serve --help | grep -E -- '--speculative-config|--reasoning-parser|--limit-mm-per-prompt|--tensor-parallel-size' || true

echo
echo "Ready."
echo "Activate with:"
echo "  source ${ENV_DIR}/activate_vllm_cuda131_py312.sh"
echo
echo "Start Gemma/Qwen using the existing repo launchers, for example:"
echo "  ENV_DIR=${ENV_DIR} MODEL_ROOT=/path/to/models bash dataset/scripts/run_gemma4_vllm_python_env_first.sh"
