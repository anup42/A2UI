#!/usr/bin/env bash
set -euo pipefail

# Build a CUDA 13.0 vLLM image for Gemma4 vLLM serving on RTX Ada GPUs.
# Build this on an internet machine with Docker, copy the generated .tar to the
# GPU/Slurm machine, then convert it to a .sif with
# build_vllm_cu130_gemma4_speculative_sif.sh.

IMAGE_NAME="${IMAGE_NAME:-a2ui-vllm-cu130-source}"
IMAGE_TAG="${IMAGE_TAG:-gemma4_speculative_ada}"
# Pin to the Gemma4-special vLLM ref used by the existing Gemma4 speculative
# setup. Override only after validating Gemma4 + speculative flags.
VLLM_REF="${VLLM_REF:-9b4e83934d895b5f6e488411cd46c8d0915115a1}"
VLLM_VERSION="${VLLM_VERSION:-0.22.0}"
VLLM_NIGHTLY_INDEX="${VLLM_NIGHTLY_INDEX:-https://wheels.vllm.ai/nightly/cu130}"
A2UI_VLLM_INSTALL_MODE="${A2UI_VLLM_INSTALL_MODE:-nightly}" # nightly|source|release
CUDA_BASE_IMAGE="${CUDA_BASE_IMAGE:-nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0}"
MAX_JOBS="${MAX_JOBS:-12}"
NVCC_THREADS="${NVCC_THREADS:-4}"
# RTX Ada / RTX 6000 Ada class GPUs are SM 8.9. Override for a mixed cluster.
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9+PTX}"
CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-89}"
BUILD_DIR="${BUILD_DIR:-${PWD}/.a2ui_vllm_cu130_gemma4_build}"
OUT_DIR="${OUT_DIR:-${PWD}/vllm_cu130_artifacts}"
TAR_NAME="${TAR_NAME:-${IMAGE_NAME}_${IMAGE_TAG}.tar}"
BUILD_LOG_NAME="${BUILD_LOG_NAME:-${IMAGE_NAME}_${IMAGE_TAG}_docker_build.log}"
A2UI_BYPASS_SSL="${A2UI_BYPASS_SSL:-1}"
A2UI_REQUIRE_SPECULATIVE="${A2UI_REQUIRE_SPECULATIVE:-0}"
DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"
BUILDKIT_PROGRESS="${BUILDKIT_PROGRESS:-plain}"
export DOCKER_BUILDKIT
export BUILDKIT_PROGRESS

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required on the internet build machine." >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}" "${OUT_DIR}"

cat > "${BUILD_DIR}/Dockerfile" <<'DOCKERFILE'
ARG CUDA_BASE_IMAGE=nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04
FROM ${CUDA_BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG VLLM_REF=main
ARG VLLM_VERSION=0.22.0
ARG VLLM_NIGHTLY_INDEX=https://wheels.vllm.ai/nightly/cu130
ARG A2UI_VLLM_INSTALL_MODE=nightly
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG A2UI_BYPASS_SSL=1
ARG A2UI_REQUIRE_SPECULATIVE=0
ARG MAX_JOBS=12
ARG NVCC_THREADS=4
ARG TORCH_CUDA_ARCH_LIST=8.9+PTX
ARG CMAKE_CUDA_ARCHITECTURES=89

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV CUDA_HOME=/usr/local/cuda
ENV VLLM_USAGE_SOURCE=a2ui-cu130-gemma4-speculative-ada-docker
ENV MAX_JOBS=${MAX_JOBS}
ENV NVCC_THREADS=${NVCC_THREADS}
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS}
ENV CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}"
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/playwright-browsers
ENV A2UI_REQUIRE_SPECULATIVE=${A2UI_REQUIRE_SPECULATIVE}
ENV A2UI_VLLM_INSTALL_MODE=${A2UI_VLLM_INSTALL_MODE}
ENV PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org download.pytorch.org github.com codeload.github.com raw.githubusercontent.com objects.githubusercontent.com release-assets.githubusercontent.com huggingface.co cdn-lfs.huggingface.co"
ENV UV_INSECURE_HOST="pypi.org,files.pythonhosted.org,download.pytorch.org,github.com,codeload.github.com,raw.githubusercontent.com,objects.githubusercontent.com,release-assets.githubusercontent.com,huggingface.co,cdn-lfs.huggingface.co"

RUN if [ "${A2UI_BYPASS_SSL}" = "1" ]; then \
      printf 'Acquire::https::Verify-Peer "false";\nAcquire::https::Verify-Host "false";\n' > /etc/apt/apt.conf.d/99-a2ui-insecure-ssl; \
    fi

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    ccache \
    cmake \
    curl \
    git \
    jq \
    libasound2t64 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcairo2 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libglib2.0-0 \
    libnspr4 \
    libnss3 \
    libnuma-dev \
    libnuma1 \
    libpango-1.0-0 \
    libpci3 \
    libx11-6 \
    libxcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    libxshmfence1 \
    fonts-liberation \
    fonts-noto-color-emoji \
    ninja-build \
    pciutils \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    wget \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
    && if [ "${A2UI_BYPASS_SSL}" = "1" ]; then \
         export CURL_CA_BUNDLE=""; \
         export REQUESTS_CA_BUNDLE=""; \
         export SSL_CERT_FILE=""; \
         python -m pip install \
           --trusted-host pypi.org \
           --trusted-host files.pythonhosted.org \
           --trusted-host download.pytorch.org \
           --trusted-host github.com \
           --trusted-host codeload.github.com \
           --trusted-host raw.githubusercontent.com \
           --upgrade \
           pip \
           "setuptools>=77.0.3,<81.0.0" \
           wheel \
           uv \
           "setuptools-rust>=1.9.0" \
           "setuptools-scm>=8.0" \
           "packaging>=24.2" \
           jinja2; \
       else \
         python -m pip install \
           --upgrade \
           pip \
           "setuptools>=77.0.3,<81.0.0" \
           wheel \
           uv \
           "setuptools-rust>=1.9.0" \
           "setuptools-scm>=8.0" \
           "packaging>=24.2" \
           jinja2; \
       fi

RUN if [ "${A2UI_BYPASS_SSL}" = "1" ]; then \
      git config --global http.sslVerify false; \
      python -m pip config set global.trusted-host "${PIP_TRUSTED_HOST}"; \
      python -m pip config set global.cert ""; \
    fi

RUN if [ "${A2UI_BYPASS_SSL}" = "1" ]; then \
      export UV_INSECURE_HOST="${UV_INSECURE_HOST}"; \
      export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST}"; \
      export GIT_SSL_NO_VERIFY=1; \
      export CURL_CA_BUNDLE=""; \
      export REQUESTS_CA_BUNDLE=""; \
      export SSL_CERT_FILE=""; \
    fi; \
    python -m uv pip install \
      --index-url "${PYTORCH_INDEX_URL}" \
      "torch==${TORCH_VERSION}" \
      "torchvision==${TORCHVISION_VERSION}" \
      "torchaudio==${TORCHAUDIO_VERSION}"

WORKDIR /opt
RUN if [ "${A2UI_BYPASS_SSL}" = "1" ]; then \
      export UV_INSECURE_HOST="${UV_INSECURE_HOST}"; \
      export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST}"; \
      export GIT_SSL_NO_VERIFY=1; \
      export CURL_CA_BUNDLE=""; \
      export REQUESTS_CA_BUNDLE=""; \
      export SSL_CERT_FILE=""; \
    fi; \
    case "${A2UI_VLLM_INSTALL_MODE}" in \
      nightly) \
        python -m uv pip install -U --reinstall vllm --pre \
          --extra-index-url "${VLLM_NIGHTLY_INDEX}" \
          --extra-index-url "${PYTORCH_INDEX_URL}" \
          --index-strategy unsafe-best-match ;; \
      release) \
        python -m uv pip install -U --reinstall "vllm==${VLLM_VERSION}" \
          --extra-index-url "${PYTORCH_INDEX_URL}" \
          --index-strategy unsafe-best-match ;; \
      source) \
        git clone https://github.com/vllm-project/vllm.git /opt/vllm \
          && cd /opt/vllm \
          && git checkout "${VLLM_REF}" \
          && git submodule update --init --recursive \
          && python -m uv pip install --no-build-isolation -e . ;; \
      *) \
        echo "Unsupported A2UI_VLLM_INSTALL_MODE=${A2UI_VLLM_INSTALL_MODE}; use nightly, source, or release." >&2; \
        exit 1 ;; \
    esac

# Dataset Stage 1-5 dependencies:
# - Stage 1/2/3: prompt generation, asset fetch, JSON/schema metrics.
# - Stage 4/5: Playwright/Chromium HTML rendering.
# - bitsandbytes: local quantized model loading and vLLM bnb modes.
RUN if [ "${A2UI_BYPASS_SSL}" = "1" ]; then \
      export UV_INSECURE_HOST="${UV_INSECURE_HOST}"; \
      export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST}"; \
      export GIT_SSL_NO_VERIFY=1; \
      export CURL_CA_BUNDLE=""; \
      export REQUESTS_CA_BUNDLE=""; \
      export SSL_CERT_FILE=""; \
      export NODE_TLS_REJECT_UNAUTHORIZED=0; \
    fi; \
    python -m uv pip install \
      "accelerate>=0.34.0" \
      "aiohttp>=3.9" \
      "beautifulsoup4>=4.12" \
      "bitsandbytes>=0.45.0" \
      "huggingface_hub[cli]>=0.25.0" \
      "httpx>=0.27" \
      "imageio>=2.34" \
      "jinja2>=3.1" \
      "jupyterlab>=4.2" \
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
      "safetensors>=0.4.5" \
      "scikit-learn>=1.4" \
      "scipy>=1.12" \
      "sentencepiece>=0.2.0" \
      "tenacity>=8.3" \
      "tokenizers>=0.20.0" \
      "tqdm>=4.66" \
      "truststore>=0.10"

RUN if [ "${A2UI_BYPASS_SSL}" = "1" ]; then \
      export NODE_TLS_REJECT_UNAUTHORIZED=0; \
      export CURL_CA_BUNDLE=""; \
      export REQUESTS_CA_BUNDLE=""; \
      export SSL_CERT_FILE=""; \
    fi; \
    python -m playwright install chromium

RUN python - <<'PY'
import os
import subprocess
import sys
import torch
import vllm
import yaml
import requests
import numpy
from PIL import Image
import matplotlib
import jsonschema
import referencing
import openai
import httpx
import truststore
import tqdm
import pandas
import playwright
import jupyterlab
import bitsandbytes

print("python", sys.version.split()[0])
print("torch", torch.__version__, "torch_cuda", torch.version.cuda)
print("vllm", getattr(vllm, "__version__", "unknown"))
print("vllm_install_mode", os.environ.get("A2UI_VLLM_INSTALL_MODE", "unknown"))
print("jupyterlab", getattr(jupyterlab, "__version__", "unknown"))
print("bitsandbytes", getattr(bitsandbytes, "__version__", "unknown"))
assert str(torch.version.cuda).startswith("13.0"), torch.version.cuda
help_text = subprocess.run(
    ["vllm", "serve", "--help"],
    check=False,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
).stdout
has_speculative_config = "--speculative-config" in help_text
has_legacy_speculative_model = "--speculative-model" in help_text
print("vllm_has_speculative_config", has_speculative_config)
print("vllm_has_legacy_speculative_model", has_legacy_speculative_model)
if not has_speculative_config and not has_legacy_speculative_model:
    message = "vLLM build does not expose speculative decoding flags"
    if os.environ.get("A2UI_REQUIRE_SPECULATIVE", "0").lower() in {"1", "true", "yes", "y", "on"}:
        raise SystemExit(message)
    print("WARNING:", message, "- continuing because A2UI_REQUIRE_SPECULATIVE=0")
print("a2ui stage1-5 deps ok")
PY

CMD ["bash"]
DOCKERFILE

IMAGE_REF="${IMAGE_NAME}:${IMAGE_TAG}"
TAR_PATH="${OUT_DIR}/${TAR_NAME}"
BUILD_LOG_PATH="${OUT_DIR}/${BUILD_LOG_NAME}"
MANIFEST_PATH="${OUT_DIR}/${IMAGE_NAME}_${IMAGE_TAG}_manifest.json"

echo "Building ${IMAGE_REF}"
echo "Target host: CUDA 13.0, driver 580.159.03, RTX Ada 49140MiB"
echo "CUDA base: ${CUDA_BASE_IMAGE}"
echo "vLLM install mode: ${A2UI_VLLM_INSTALL_MODE}"
echo "vLLM ref: ${VLLM_REF}"
echo "vLLM nightly index: ${VLLM_NIGHTLY_INDEX}"
echo "PyTorch index: ${PYTORCH_INDEX_URL}"
echo "MAX_JOBS: ${MAX_JOBS}"
echo "NVCC_THREADS: ${NVCC_THREADS}"
echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
echo "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}"
echo "Bypass SSL: ${A2UI_BYPASS_SSL}"
echo "Require speculative: ${A2UI_REQUIRE_SPECULATIVE}"
echo "Docker progress: ${BUILDKIT_PROGRESS}"
echo "Build log: ${BUILD_LOG_PATH}"

docker build --progress="${BUILDKIT_PROGRESS}" \
  --build-arg "CUDA_BASE_IMAGE=${CUDA_BASE_IMAGE}" \
  --build-arg "A2UI_VLLM_INSTALL_MODE=${A2UI_VLLM_INSTALL_MODE}" \
  --build-arg "VLLM_REF=${VLLM_REF}" \
  --build-arg "VLLM_VERSION=${VLLM_VERSION}" \
  --build-arg "VLLM_NIGHTLY_INDEX=${VLLM_NIGHTLY_INDEX}" \
  --build-arg "PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL}" \
  --build-arg "TORCH_VERSION=${TORCH_VERSION}" \
  --build-arg "TORCHVISION_VERSION=${TORCHVISION_VERSION}" \
  --build-arg "TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION}" \
  --build-arg "A2UI_BYPASS_SSL=${A2UI_BYPASS_SSL}" \
  --build-arg "A2UI_REQUIRE_SPECULATIVE=${A2UI_REQUIRE_SPECULATIVE}" \
  --build-arg "MAX_JOBS=${MAX_JOBS}" \
  --build-arg "NVCC_THREADS=${NVCC_THREADS}" \
  --build-arg "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}" \
  --build-arg "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}" \
  -t "${IMAGE_REF}" \
  "${BUILD_DIR}" 2>&1 | tee "${BUILD_LOG_PATH}"

echo "Saving ${IMAGE_REF} to ${TAR_PATH}"
docker save -o "${TAR_PATH}" "${IMAGE_REF}"

cat > "${MANIFEST_PATH}" <<EOF
{
  "image": "${IMAGE_REF}",
  "tar": "${TAR_PATH}",
  "target_host": {
    "driver": "580.159.03",
    "cuda": "13.0",
    "gpu": "NVIDIA RTX Ada Generation",
    "gpu_memory_mib": 49140
  },
  "cuda_base_image": "${CUDA_BASE_IMAGE}",
  "vllm_install_mode": "${A2UI_VLLM_INSTALL_MODE}",
  "vllm_ref": "${VLLM_REF}",
  "vllm_version": "${VLLM_VERSION}",
  "vllm_nightly_index": "${VLLM_NIGHTLY_INDEX}",
  "pytorch_index_url": "${PYTORCH_INDEX_URL}",
  "torch_version": "${TORCH_VERSION}",
  "torchvision_version": "${TORCHVISION_VERSION}",
  "torchaudio_version": "${TORCHAUDIO_VERSION}",
  "max_jobs": "${MAX_JOBS}",
  "nvcc_threads": "${NVCC_THREADS}",
  "torch_cuda_arch_list": "${TORCH_CUDA_ARCH_LIST}",
  "cmake_cuda_architectures": "${CMAKE_CUDA_ARCHITECTURES}",
  "a2ui_stage1_to_stage5_deps": true,
  "playwright_chromium": true,
  "jupyterlab": true,
  "bitsandbytes": true,
  "gemma4_speculative_decoding_ref": true,
  "a2ui_require_speculative": "${A2UI_REQUIRE_SPECULATIVE}",
  "built_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo
echo "Created:"
echo "  ${TAR_PATH}"
echo "  ${MANIFEST_PATH}"
echo "  ${BUILD_LOG_PATH}"
echo
echo "Copy to the GPU machine, then convert to SIF:"
echo "  bash dataset/scripts/build_vllm_cu130_gemma4_speculative_sif.sh /path/to/${TAR_NAME}"
