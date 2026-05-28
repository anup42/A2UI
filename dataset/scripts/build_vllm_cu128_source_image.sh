#!/usr/bin/env bash
set -euo pipefail

# Build a CUDA 12.8 vLLM image from source on an Ubuntu machine with internet.
# Copy the generated .tar to the Slurm machine and convert it to a .sif with
# build_vllm_cu128_sif_on_slurm.sh.

IMAGE_NAME="${IMAGE_NAME:-a2ui-vllm-cu128-source}"
IMAGE_TAG="${IMAGE_TAG:-qwen36}"
# Pin to the same vLLM commit used by the current cu129-nightly image that was
# verified in Docker Hub metadata. Override with VLLM_REF=main only when needed.
VLLM_REF="${VLLM_REF:-626fa9bba5663a5cf6a870debf031ee344ddb822}"
CUDA_BASE_IMAGE="${CUDA_BASE_IMAGE:-nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
MAX_JOBS="${MAX_JOBS:-16}"
NVCC_THREADS="${NVCC_THREADS:-4}"
BUILD_DIR="${BUILD_DIR:-${PWD}/.a2ui_vllm_cu128_build}"
OUT_DIR="${OUT_DIR:-${PWD}/vllm_cu128_artifacts}"
TAR_NAME="${TAR_NAME:-${IMAGE_NAME}_${IMAGE_TAG}.tar}"
A2UI_BYPASS_SSL="${A2UI_BYPASS_SSL:-0}"
DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"
export DOCKER_BUILDKIT

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
ARG A2UI_BYPASS_SSL=0
ARG MAX_JOBS=16
ARG NVCC_THREADS=4

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV CUDA_HOME=/usr/local/cuda
ENV VLLM_USAGE_SOURCE=a2ui-cu128-source-docker
ENV MAX_JOBS=${MAX_JOBS}
ENV NVCC_THREADS=${NVCC_THREADS}
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

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
    && python -m pip install --upgrade pip setuptools wheel uv

RUN if [ "${A2UI_BYPASS_SSL}" = "1" ]; then \
      git config --global http.sslVerify false; \
      python -m pip config set global.trusted-host "pypi.org files.pythonhosted.org download.pytorch.org github.com codeload.github.com"; \
    fi

RUN if [ "${A2UI_BYPASS_SSL}" = "1" ]; then \
      export UV_INSECURE_HOST="pypi.org,files.pythonhosted.org,download.pytorch.org,github.com,codeload.github.com"; \
    fi; \
    python -m uv pip install \
      --index-url "${PYTORCH_INDEX_URL}" \
      torch torchvision torchaudio

WORKDIR /opt
RUN git clone https://github.com/vllm-project/vllm.git /opt/vllm \
    && cd /opt/vllm \
    && git checkout "${VLLM_REF}" \
    && git submodule update --init --recursive

WORKDIR /opt/vllm

# Build against the already-installed CUDA 12.8 PyTorch. This avoids pulling a
# CUDA 12.9/13.0 torch stack during vLLM install.
RUN if [ "${A2UI_BYPASS_SSL}" = "1" ]; then \
      export UV_INSECURE_HOST="pypi.org,files.pythonhosted.org,download.pytorch.org,github.com,codeload.github.com"; \
    fi; \
    python -m uv pip install --torch-backend=cu128 --no-build-isolation -e .

RUN python - <<'PY'
import torch
import vllm
print("torch", torch.__version__, "torch_cuda", torch.version.cuda)
print("vllm", getattr(vllm, "__version__", "unknown"))
assert str(torch.version.cuda).startswith("12.8"), torch.version.cuda
PY

CMD ["bash"]
DOCKERFILE

IMAGE_REF="${IMAGE_NAME}:${IMAGE_TAG}"
TAR_PATH="${OUT_DIR}/${TAR_NAME}"
MANIFEST_PATH="${OUT_DIR}/${IMAGE_NAME}_${IMAGE_TAG}_manifest.json"

echo "Building ${IMAGE_REF}"
echo "CUDA base: ${CUDA_BASE_IMAGE}"
echo "vLLM ref: ${VLLM_REF}"
echo "PyTorch index: ${PYTORCH_INDEX_URL}"
echo "MAX_JOBS: ${MAX_JOBS}"
echo "NVCC_THREADS: ${NVCC_THREADS}"
echo "Bypass SSL: ${A2UI_BYPASS_SSL}"

docker build \
  --build-arg "CUDA_BASE_IMAGE=${CUDA_BASE_IMAGE}" \
  --build-arg "VLLM_REF=${VLLM_REF}" \
  --build-arg "PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL}" \
  --build-arg "A2UI_BYPASS_SSL=${A2UI_BYPASS_SSL}" \
  --build-arg "MAX_JOBS=${MAX_JOBS}" \
  --build-arg "NVCC_THREADS=${NVCC_THREADS}" \
  -t "${IMAGE_REF}" \
  "${BUILD_DIR}"

echo "Saving ${IMAGE_REF} to ${TAR_PATH}"
docker save -o "${TAR_PATH}" "${IMAGE_REF}"

cat > "${MANIFEST_PATH}" <<EOF
{
  "image": "${IMAGE_REF}",
  "tar": "${TAR_PATH}",
  "cuda_base_image": "${CUDA_BASE_IMAGE}",
  "vllm_ref": "${VLLM_REF}",
  "pytorch_index_url": "${PYTORCH_INDEX_URL}",
  "max_jobs": "${MAX_JOBS}",
  "nvcc_threads": "${NVCC_THREADS}",
  "built_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo
echo "Created:"
echo "  ${TAR_PATH}"
echo "  ${MANIFEST_PATH}"
echo
echo "Copy the tar to Slurm, for example:"
echo "  scp ${TAR_PATH} <user>@<slurm-host>:/isilonhome/k_anup/containers/"
echo
echo "Then on Slurm:"
echo "  bash dataset/scripts/build_vllm_cu128_sif_on_slurm.sh /isilonhome/k_anup/containers/${TAR_NAME}"
