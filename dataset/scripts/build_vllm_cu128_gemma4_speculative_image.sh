#!/usr/bin/env bash
set -euo pipefail

# Build a CUDA 12.8 vLLM image from the Gemma4 speculative-decoding vLLM ref.
# Copy the generated .tar to the Slurm machine and convert it to a .sif with
# build_vllm_cu128_sif_on_slurm.sh.

IMAGE_NAME="${IMAGE_NAME:-a2ui-vllm-cu128-source}"
IMAGE_TAG="${IMAGE_TAG:-gemma4_speculative}"
# Pin to the Gemma4-special vLLM ref used for Gemma4 speculative decoding.
# Override with VLLM_REF only after validating a newer Gemma4-compatible ref.
VLLM_REF="${VLLM_REF:-9b4e83934d895b5f6e488411cd46c8d0915115a1}"
CUDA_BASE_IMAGE="${CUDA_BASE_IMAGE:-nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0}"
MAX_JOBS="${MAX_JOBS:-8}"
NVCC_THREADS="${NVCC_THREADS:-2}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-80}"
BUILD_DIR="${BUILD_DIR:-${PWD}/.a2ui_vllm_cu128_build}"
OUT_DIR="${OUT_DIR:-${PWD}/vllm_cu128_artifacts}"
TAR_NAME="${TAR_NAME:-${IMAGE_NAME}_${IMAGE_TAG}.tar}"
BUILD_LOG_NAME="${BUILD_LOG_NAME:-${IMAGE_NAME}_${IMAGE_TAG}_docker_build.log}"
A2UI_BYPASS_SSL="${A2UI_BYPASS_SSL:-0}"
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
ARG CUDA_BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
FROM ${CUDA_BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG VLLM_REF=main
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG A2UI_BYPASS_SSL=0
ARG MAX_JOBS=16
ARG NVCC_THREADS=4
ARG TORCH_CUDA_ARCH_LIST=8.0
ARG CMAKE_CUDA_ARCHITECTURES=80
ARG A2UI_INSTALL_STAGE_DEPS=1

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV CUDA_HOME=/usr/local/cuda
ENV VLLM_USAGE_SOURCE=a2ui-cu128-gemma4-speculative-docker
ENV MAX_JOBS=${MAX_JOBS}
ENV NVCC_THREADS=${NVCC_THREADS}
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS}
ENV CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}"
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org download.pytorch.org github.com codeload.github.com raw.githubusercontent.com"
ENV UV_INSECURE_HOST="pypi.org,files.pythonhosted.org,download.pytorch.org,github.com,codeload.github.com,raw.githubusercontent.com"

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
    libnuma-dev \
    libnuma1 \
    ninja-build \
    pciutils \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
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
      python -m pip config set global.trusted-host "pypi.org files.pythonhosted.org download.pytorch.org github.com codeload.github.com raw.githubusercontent.com"; \
      python -m pip config set global.cert ""; \
    fi

RUN if [ "${A2UI_BYPASS_SSL}" = "1" ]; then \
      export UV_INSECURE_HOST="pypi.org,files.pythonhosted.org,download.pytorch.org,github.com,codeload.github.com,raw.githubusercontent.com"; \
      export PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org download.pytorch.org github.com codeload.github.com raw.githubusercontent.com"; \
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
RUN git clone https://github.com/vllm-project/vllm.git /opt/vllm \
    && cd /opt/vllm \
    && git checkout "${VLLM_REF}" \
    && git submodule update --init --recursive

WORKDIR /opt/vllm

# Build against the already-installed CUDA 12.8 PyTorch. This avoids pulling a
# CUDA 12.9/13.0 torch stack during vLLM install.
RUN if [ "${A2UI_BYPASS_SSL}" = "1" ]; then \
      export UV_INSECURE_HOST="pypi.org,files.pythonhosted.org,download.pytorch.org,github.com,codeload.github.com,raw.githubusercontent.com"; \
      export PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org download.pytorch.org github.com codeload.github.com raw.githubusercontent.com"; \
      export GIT_SSL_NO_VERIFY=1; \
      export CURL_CA_BUNDLE=""; \
      export REQUESTS_CA_BUNDLE=""; \
      export SSL_CERT_FILE=""; \
    fi; \
    python -m uv pip install --torch-backend=cu128 --no-build-isolation -e .

# Add the A2UI Stage 1/2/3 dataset-runner dependencies into the same venv.
# This intentionally avoids reinstalling torch/vLLM after the source build.
RUN if [ "${A2UI_INSTALL_STAGE_DEPS}" = "1" ]; then \
      if [ "${A2UI_BYPASS_SSL}" = "1" ]; then \
        export UV_INSECURE_HOST="pypi.org,files.pythonhosted.org,download.pytorch.org,github.com,codeload.github.com,raw.githubusercontent.com"; \
        export PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org download.pytorch.org github.com codeload.github.com raw.githubusercontent.com"; \
        export GIT_SSL_NO_VERIFY=1; \
        export CURL_CA_BUNDLE=""; \
        export REQUESTS_CA_BUNDLE=""; \
        export SSL_CERT_FILE=""; \
      fi; \
      python -m uv pip install \
        "accelerate>=0.34.0" \
        "safetensors>=0.4.5" \
        "huggingface_hub[cli]>=0.25.0" \
        "sentencepiece>=0.2.0" \
        "tokenizers>=0.20.0" \
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
        "tqdm>=4.66" \
        "jupyterlab>=4.2" \
        "bitsandbytes>=0.45.0"; \
    fi

RUN python - <<'PY'
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
import jupyterlab
import bitsandbytes
print("torch", torch.__version__, "torch_cuda", torch.version.cuda)
print("vllm", getattr(vllm, "__version__", "unknown"))
print("jupyterlab", getattr(jupyterlab, "__version__", "unknown"))
print("bitsandbytes", getattr(bitsandbytes, "__version__", "unknown"))
assert str(torch.version.cuda).startswith("12.8"), torch.version.cuda
print("a2ui stage deps ok")
PY

RUN python - <<'PY'
import subprocess
import sys

proc = subprocess.run(
    ["vllm", "serve", "--help"],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
help_text = proc.stdout
has_speculative_config = "--speculative-config" in help_text
has_flat_spec_flags = "--spec-model" in help_text and "--spec-tokens" in help_text
has_legacy_spec_flags = "--speculative-model" in help_text and "--num-speculative-tokens" in help_text
has_reasoning_parser = "--reasoning-parser" in help_text
print("vllm_has_speculative_config", has_speculative_config)
print("vllm_has_flat_spec_flags", has_flat_spec_flags)
print("vllm_has_legacy_spec_flags", has_legacy_spec_flags)
print("vllm_has_reasoning_parser", has_reasoning_parser)
if not (has_speculative_config or has_flat_spec_flags or has_legacy_spec_flags):
    print("ERROR: vLLM build does not expose speculative decoding flags.", file=sys.stderr)
    print("Speculative-related help lines:", file=sys.stderr)
    for line in help_text.splitlines():
        lowered = line.lower()
        if any(token in lowered for token in ("spec", "draft", "mtp", "eagle", "medusa")):
            print(line, file=sys.stderr)
    raise SystemExit(2)
if not has_reasoning_parser:
    print("WARNING: vLLM build does not expose --reasoning-parser.", file=sys.stderr)
print("vllm speculative/reasoning flag validation ok")
PY

CMD ["bash"]
DOCKERFILE

IMAGE_REF="${IMAGE_NAME}:${IMAGE_TAG}"
TAR_PATH="${OUT_DIR}/${TAR_NAME}"
BUILD_LOG_PATH="${OUT_DIR}/${BUILD_LOG_NAME}"
MANIFEST_PATH="${OUT_DIR}/${IMAGE_NAME}_${IMAGE_TAG}_manifest.json"

echo "Building ${IMAGE_REF}"
echo "CUDA base: ${CUDA_BASE_IMAGE}"
echo "vLLM ref: ${VLLM_REF}"
echo "PyTorch index: ${PYTORCH_INDEX_URL}"
echo "MAX_JOBS: ${MAX_JOBS}"
echo "NVCC_THREADS: ${NVCC_THREADS}"
echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
echo "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}"
echo "Bypass SSL: ${A2UI_BYPASS_SSL}"
echo "Docker progress: ${BUILDKIT_PROGRESS}"
echo "Build log: ${BUILD_LOG_PATH}"

docker build --progress="${BUILDKIT_PROGRESS}" \
  --build-arg "CUDA_BASE_IMAGE=${CUDA_BASE_IMAGE}" \
  --build-arg "VLLM_REF=${VLLM_REF}" \
  --build-arg "PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL}" \
  --build-arg "TORCH_VERSION=${TORCH_VERSION}" \
  --build-arg "TORCHVISION_VERSION=${TORCHVISION_VERSION}" \
  --build-arg "TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION}" \
  --build-arg "A2UI_BYPASS_SSL=${A2UI_BYPASS_SSL}" \
  --build-arg "MAX_JOBS=${MAX_JOBS}" \
  --build-arg "NVCC_THREADS=${NVCC_THREADS}" \
  --build-arg "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}" \
  --build-arg "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}" \
  --build-arg "A2UI_INSTALL_STAGE_DEPS=1" \
  -t "${IMAGE_REF}" \
  "${BUILD_DIR}" 2>&1 | tee "${BUILD_LOG_PATH}"

echo "Saving ${IMAGE_REF} to ${TAR_PATH}"
docker save -o "${TAR_PATH}" "${IMAGE_REF}"

cat > "${MANIFEST_PATH}" <<EOF
{
  "image": "${IMAGE_REF}",
  "tar": "${TAR_PATH}",
  "cuda_base_image": "${CUDA_BASE_IMAGE}",
  "vllm_ref": "${VLLM_REF}",
  "pytorch_index_url": "${PYTORCH_INDEX_URL}",
  "torch_version": "${TORCH_VERSION}",
  "torchvision_version": "${TORCHVISION_VERSION}",
  "torchaudio_version": "${TORCHAUDIO_VERSION}",
  "max_jobs": "${MAX_JOBS}",
  "nvcc_threads": "${NVCC_THREADS}",
  "torch_cuda_arch_list": "${TORCH_CUDA_ARCH_LIST}",
  "cmake_cuda_architectures": "${CMAKE_CUDA_ARCHITECTURES}",
  "a2ui_stage123_deps": true,
  "jupyterlab": true,
  "bitsandbytes": true,
  "gemma4_speculative_decoding_ref": true,
  "built_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo
echo "Created:"
echo "  ${TAR_PATH}"
echo "  ${MANIFEST_PATH}"
echo "  ${BUILD_LOG_PATH}"
echo
echo "Copy the Gemma4 speculative-decoding tar to Slurm, for example:"
echo "  scp ${TAR_PATH} <user>@<slurm-host>:/isilonhome/k_anup/containers/"
echo
echo "Then on Slurm:"
echo "  bash dataset/scripts/build_vllm_cu128_sif_on_slurm.sh /isilonhome/k_anup/containers/${TAR_NAME}"
