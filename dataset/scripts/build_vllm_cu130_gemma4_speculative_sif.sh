#!/usr/bin/env bash
set -euo pipefail

# Convert the Docker archive created by
# build_vllm_cu130_gemma4_speculative_image.sh into a Singularity/Apptainer SIF.

IMAGE_TAR="${1:-${IMAGE_TAR:-}}"
SIF_PATH="${SIF_PATH:-}"

if [[ -z "${IMAGE_TAR}" ]]; then
  echo "Usage: bash dataset/scripts/build_vllm_cu130_gemma4_speculative_sif.sh /path/to/a2ui-vllm-cu130-source_gemma4_speculative_ada.tar" >&2
  exit 1
fi

if [[ ! -f "${IMAGE_TAR}" ]]; then
  echo "Docker archive not found: ${IMAGE_TAR}" >&2
  exit 1
fi

if [[ -z "${SIF_PATH}" ]]; then
  base="$(basename "${IMAGE_TAR}")"
  SIF_PATH="$(cd "$(dirname "${IMAGE_TAR}")" && pwd)/${base%.tar}.sif"
fi

if ! command -v apptainer >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
  module load apptainer || true
fi

if command -v singularity >/dev/null 2>&1; then
  RUNTIME="singularity"
elif command -v apptainer >/dev/null 2>&1; then
  RUNTIME="apptainer"
else
  echo "singularity or apptainer is required. Try: module load apptainer" >&2
  exit 1
fi

echo "Building SIF with ${RUNTIME}"
echo "Input tar: ${IMAGE_TAR}"
echo "Output SIF: ${SIF_PATH}"

"${RUNTIME}" build --force "${SIF_PATH}" "docker-archive://${IMAGE_TAR}"

echo
echo "Created: ${SIF_PATH}"
echo
echo "Smoke test:"
echo "  ${RUNTIME} exec --nv --cleanenv ${SIF_PATH} python - <<'PY'"
echo "import torch, vllm, bitsandbytes"
echo "print('torch', torch.__version__, 'cuda', torch.version.cuda)"
echo "print('vllm', getattr(vllm, '__version__', 'unknown'))"
echo "print('bnb', getattr(bitsandbytes, '__version__', 'unknown'))"
echo "print('gpu_available', torch.cuda.is_available())"
echo "PY"
