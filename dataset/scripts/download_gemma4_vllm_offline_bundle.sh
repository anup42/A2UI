#!/usr/bin/env bash
set -euo pipefail

# Download Gemma4/vLLM setup wheels on an internet-connected Linux machine.
# Use the same Python minor version and Linux architecture as the target GPU
# machine, then copy the whole bundle directory to the offline machine.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BUNDLE_DIR="${BUNDLE_DIR:-${PWD}/gemma4_vllm_offline_bundle}"
WHEELHOUSE="${BUNDLE_DIR}/wheelhouse"
DOWNLOAD_VENV="${BUNDLE_DIR}/.download_venv"

A2UI_DISABLE_SSL_VERIFY="${A2UI_DISABLE_SSL_VERIFY:-1}"
A2UI_VLLM_INSTALL_MODE="${A2UI_VLLM_INSTALL_MODE:-nightly}" # nightly|release|skip
VLLM_VERSION="${VLLM_VERSION:-0.22.0}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0}"
VLLM_NIGHTLY_INDEX="${VLLM_NIGHTLY_INDEX:-https://wheels.vllm.ai/nightly/cu130}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
FLASHINFER_CUDA_TAG="${FLASHINFER_CUDA_TAG:-cu130}"
FLASHINFER_INDEX_URL="${FLASHINFER_INDEX_URL:-https://flashinfer.ai/whl/${FLASHINFER_CUDA_TAG}}"
CUDA_RUNTIME_PACKAGE="${CUDA_RUNTIME_PACKAGE:-nvidia-cuda-runtime==13.0.96}"
CUDA_NVCC_PACKAGE="${CUDA_NVCC_PACKAGE:-nvidia-cuda-nvcc==13.0.88}"
CUDA_CRT_PACKAGE="${CUDA_CRT_PACKAGE:-nvidia-cuda-crt==13.0.88}"
CUDA_CCCL_PACKAGE="${CUDA_CCCL_PACKAGE:-nvidia-cuda-cccl==13.0.85}"

mkdir -p "${WHEELHOUSE}"

if [[ ! -d "${DOWNLOAD_VENV}" ]]; then
  "${PYTHON_BIN}" -m venv "${DOWNLOAD_VENV}"
fi
# shellcheck disable=SC1091
source "${DOWNLOAD_VENV}/bin/activate"

PIP_SSL_ARGS=()
if [[ "${A2UI_DISABLE_SSL_VERIFY}" = "1" ]]; then
  export CURL_CA_BUNDLE=""
  export REQUESTS_CA_BUNDLE=""
  export SSL_CERT_FILE=""
  export PYTHONHTTPSVERIFY=0
  export GIT_SSL_NO_VERIFY=1
  export HF_HUB_DISABLE_SSL_VERIFICATION=1
  PIP_SSL_ARGS=(
    --trusted-host pypi.org
    --trusted-host pypi.python.org
    --trusted-host files.pythonhosted.org
    --trusted-host download.pytorch.org
    --trusted-host download-r2.pytorch.org
    --trusted-host wheels.vllm.ai
    --trusted-host github.com
    --trusted-host codeload.github.com
    --trusted-host raw.githubusercontent.com
    --trusted-host objects.githubusercontent.com
    --trusted-host release-assets.githubusercontent.com
    --trusted-host huggingface.co
    --trusted-host cdn-lfs.huggingface.co
    --trusted-host flashinfer.ai
  )
fi

download_wheels() {
  python -m pip download "${PIP_SSL_ARGS[@]}" --dest "${WHEELHOUSE}" "$@"
}

python -m pip install "${PIP_SSL_ARGS[@]}" --upgrade pip wheel setuptools

download_wheels \
  pip \
  wheel \
  "setuptools>=77.0.3,<81.0.0" \
  "setuptools-rust>=1.9.0" \
  "setuptools-scm>=8.0" \
  "packaging>=24.2" \
  jinja2 \
  "MarkupSafe>=2.0" \
  "uv>=0.5.0"

download_wheels \
  --extra-index-url "${PYTORCH_INDEX_URL}" \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  "ninja>=1.11" \
  "cmake>=3.28"

download_wheels \
  "${CUDA_RUNTIME_PACKAGE}" \
  "${CUDA_NVCC_PACKAGE}" \
  "${CUDA_CRT_PACKAGE}" \
  "${CUDA_CCCL_PACKAGE}" || {
    echo "Warning: could not download CUDA runtime/NVCC/CRT/CCCL wheels." >&2
  }

case "${A2UI_VLLM_INSTALL_MODE}" in
  nightly)
    download_wheels \
      --pre \
      --extra-index-url "${VLLM_NIGHTLY_INDEX}" \
      --extra-index-url "${PYTORCH_INDEX_URL}" \
      vllm
    ;;
  release)
    download_wheels \
      --extra-index-url "${PYTORCH_INDEX_URL}" \
      "vllm==${VLLM_VERSION}"
    ;;
  skip)
    echo "Skipping vLLM wheel download because A2UI_VLLM_INSTALL_MODE=skip"
    ;;
  source)
    echo "A2UI_VLLM_INSTALL_MODE=source is not supported by this wheelhouse downloader." >&2
    echo "Use nightly/release wheels, or build a container/source checkout separately." >&2
    exit 1
    ;;
  *)
    echo "Unsupported A2UI_VLLM_INSTALL_MODE=${A2UI_VLLM_INSTALL_MODE}" >&2
    exit 1
    ;;
esac

download_wheels "flashinfer-python" "flashinfer-cubin" || {
  echo "Warning: could not download flashinfer-python/flashinfer-cubin from PyPI." >&2
}

python -m pip download "${PIP_SSL_ARGS[@]}" \
  --dest "${WHEELHOUSE}" \
  --index-url "${FLASHINFER_INDEX_URL}" \
  "flashinfer-jit-cache" || {
    echo "Warning: could not download flashinfer-jit-cache from ${FLASHINFER_INDEX_URL}." >&2
  }

download_wheels \
  "pyyaml>=6.0.2" \
  "requests>=2.32.0" \
  "numpy>=1.26" \
  "pillow>=10.0" \
  "matplotlib>=3.8" \
  "jsonschema>=4.23" \
  "referencing>=0.35" \
  "jsonschema-specifications>=2023.12.1" \
  "attrs>=23" \
  "rpds-py>=0.20" \
  "openai>=1.60" \
  "httpx>=0.27" \
  "truststore>=0.10" \
  "tqdm>=4.66" \
  "jupyterlab>=4.2" \
  "bitsandbytes>=0.45"

export BUNDLE_DIR WHEELHOUSE
python - <<'PY'
import sys
from pathlib import Path

wheelhouse = Path(__import__("os").environ["WHEELHOUSE"])
wheel_names = [path.name.lower().replace("_", "-") for path in wheelhouse.glob("*.whl")]
required = {
    "pip": "pip",
    "wheel": "wheel",
    "setuptools": "setuptools",
    "jinja2": "jinja2",
    "MarkupSafe": "markupsafe",
    "uv": "uv",
}
missing = [
    name
    for name, normalized in required.items()
    if not any(wheel.startswith(f"{normalized}-") for wheel in wheel_names)
]
if missing:
    print(f"Missing required offline wheels: {', '.join(missing)}", file=sys.stderr)
    sys.exit(1)
PY

cat > "${BUNDLE_DIR}/README.txt" <<EOF
Gemma4 vLLM offline bundle

Wheelhouse:
  ${WHEELHOUSE}

Copy this whole folder to the offline GPU machine, then run from the A2UI repo:
  bash dataset/scripts/install_gemma4_vllm_offline_env.sh /path/to/gemma4_vllm_offline_bundle

The target machine should use the same Python minor version and compatible Linux/CUDA stack.
EOF

python - <<'PY'
import json
import os
from pathlib import Path

bundle = Path(os.environ.get("BUNDLE_DIR", "gemma4_vllm_offline_bundle")).resolve()
wheelhouse = bundle / "wheelhouse"
manifest = {
    "bundle": str(bundle),
    "wheelhouse": str(wheelhouse),
    "wheel_count": len(list(wheelhouse.glob("*"))),
}
(bundle / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
PY

echo "Offline bundle ready: ${BUNDLE_DIR}"
