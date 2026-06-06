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
MATPLOTLIB_VERSION="${MATPLOTLIB_VERSION:-3.10.9}"
JUPYTERLAB_VERSION="${JUPYTERLAB_VERSION:-4.5.8}"
BITSANDBYTES_VERSION="${BITSANDBYTES_VERSION:-0.49.2}"
TRUSTSTORE_VERSION="${TRUSTSTORE_VERSION:-0.10.4}"
GEMMA4_OFFLINE_TARGET_PLATFORM="${GEMMA4_OFFLINE_TARGET_PLATFORM:-manylinux_2_28_x86_64}"
GEMMA4_OFFLINE_TARGET_PLATFORMS="${GEMMA4_OFFLINE_TARGET_PLATFORMS:-${GEMMA4_OFFLINE_TARGET_PLATFORM},manylinux_2_24_x86_64,manylinux2014_x86_64}"
GEMMA4_OFFLINE_TARGET_PYTHON_VERSION="${GEMMA4_OFFLINE_TARGET_PYTHON_VERSION:-311}"
GEMMA4_OFFLINE_TARGET_IMPLEMENTATION="${GEMMA4_OFFLINE_TARGET_IMPLEMENTATION:-cp}"
GEMMA4_OFFLINE_TARGET_ABI="${GEMMA4_OFFLINE_TARGET_ABI:-cp${GEMMA4_OFFLINE_TARGET_PYTHON_VERSION}}"
GEMMA4_OFFLINE_CLEAN_WHEELHOUSE="${GEMMA4_OFFLINE_CLEAN_WHEELHOUSE:-1}"

mkdir -p "${WHEELHOUSE}"
if [[ "${GEMMA4_OFFLINE_CLEAN_WHEELHOUSE}" = "1" ]]; then
  echo "Cleaning existing wheelhouse to avoid stale wheels for the wrong Python ABI..."
  rm -f "${WHEELHOUSE}"/*.whl
fi

if [[ "${A2UI_DOWNLOAD_USE_SYSTEM_PYTHON:-0}" = "1" ]]; then
  echo "Using system Python for downloads: ${PYTHON_BIN}"
elif [[ ! -d "${DOWNLOAD_VENV}" ]]; then
  if ! "${PYTHON_BIN}" -m venv "${DOWNLOAD_VENV}"; then
    echo "Could not create download venv with ${PYTHON_BIN}; falling back to system Python if pip is available." >&2
    if ! "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
      echo "Selected Python has no venv and no pip: ${PYTHON_BIN}" >&2
      echo "Install python3-venv/python3-pip, set PYTHON_BIN to a Python with pip, or set A2UI_DOWNLOAD_USE_SYSTEM_PYTHON=1 with a pip-capable Python." >&2
      exit 1
    fi
    A2UI_DOWNLOAD_USE_SYSTEM_PYTHON=1
  fi
fi
if [[ "${A2UI_DOWNLOAD_USE_SYSTEM_PYTHON:-0}" != "1" ]]; then
  if [[ -f "${DOWNLOAD_VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${DOWNLOAD_VENV}/bin/activate"
  elif [[ -f "${DOWNLOAD_VENV}/Scripts/activate" ]]; then
    # shellcheck disable=SC1091
    source "${DOWNLOAD_VENV}/Scripts/activate"
  else
    echo "Download venv was created but no activation script was found: ${DOWNLOAD_VENV}" >&2
    exit 1
  fi
fi
DOWNLOAD_PYTHON="python"
if [[ "${A2UI_DOWNLOAD_USE_SYSTEM_PYTHON:-0}" = "1" ]]; then
  DOWNLOAD_PYTHON="${PYTHON_BIN}"
fi
PYTHON_BUNDLE_DIR="${BUNDLE_DIR}"
PYTHON_WHEELHOUSE="${WHEELHOUSE}"
if [[ "${DOWNLOAD_PYTHON}" == *.exe ]] && command -v wslpath >/dev/null 2>&1; then
  PYTHON_BUNDLE_DIR="$(wslpath -w "${BUNDLE_DIR}")"
  PYTHON_WHEELHOUSE="$(wslpath -w "${WHEELHOUSE}")"
fi

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
PIP_TARGET_ARGS=(
  --only-binary=:all:
  --python-version "${GEMMA4_OFFLINE_TARGET_PYTHON_VERSION}"
  --implementation "${GEMMA4_OFFLINE_TARGET_IMPLEMENTATION}"
  --abi "${GEMMA4_OFFLINE_TARGET_ABI}"
)
IFS=' ,' read -r -a GEMMA4_OFFLINE_TARGET_PLATFORM_LIST <<< "${GEMMA4_OFFLINE_TARGET_PLATFORMS}"
for target_platform in "${GEMMA4_OFFLINE_TARGET_PLATFORM_LIST[@]}"; do
  if [[ -n "${target_platform}" ]]; then
    PIP_TARGET_ARGS+=(--platform "${target_platform}")
  fi
done

download_wheels() {
  "${DOWNLOAD_PYTHON}" -m pip download "${PIP_SSL_ARGS[@]}" "${PIP_TARGET_ARGS[@]}" --dest "${PYTHON_WHEELHOUSE}" "$@"
}

download_wheels_no_deps() {
  "${DOWNLOAD_PYTHON}" -m pip download "${PIP_SSL_ARGS[@]}" "${PIP_TARGET_ARGS[@]}" --dest "${PYTHON_WHEELHOUSE}" --no-deps "$@"
}

wheelhouse_has_package() {
  local normalized="$1"
  "${DOWNLOAD_PYTHON}" - "${PYTHON_WHEELHOUSE}" "${normalized}" <<'PY'
import sys
from pathlib import Path

wheelhouse = Path(sys.argv[1])
normalized = sys.argv[2].lower().replace("_", "-")
for path in wheelhouse.glob("*.whl"):
    name = path.name.lower().replace("_", "-")
    if name.startswith(f"{normalized}-"):
        print(path)
        sys.exit(0)
sys.exit(1)
PY
}

download_required_binary_wheel() {
  local normalized="$1"
  shift
  if wheelhouse_has_package "${normalized}" >/dev/null; then
    echo "Required wheel already present for ${normalized}: $(wheelhouse_has_package "${normalized}")"
    return 0
  fi
  local spec
  for spec in "$@"; do
    echo "Downloading required binary wheel: ${spec}"
    "${DOWNLOAD_PYTHON}" -m pip download "${PIP_SSL_ARGS[@]}" \
      "${PIP_TARGET_ARGS[@]}" \
      --dest "${PYTHON_WHEELHOUSE}" \
      --no-deps \
      "${spec}" || true
    if wheelhouse_has_package "${normalized}" >/dev/null; then
      echo "Downloaded required wheel for ${normalized}: $(wheelhouse_has_package "${normalized}")"
      return 0
    fi
  done
  echo "Missing required binary wheel after retries: ${normalized}" >&2
  return 1
}

download_required_target_binary_wheel() {
  local normalized="$1"
  shift
  rm -f "${WHEELHOUSE}/${normalized}"-*.whl "${WHEELHOUSE}/${normalized//-/_}"-*.whl
  local spec
  for spec in "$@"; do
    echo "Downloading required target wheel: ${spec}"
    echo "  platforms=${GEMMA4_OFFLINE_TARGET_PLATFORMS} python=${GEMMA4_OFFLINE_TARGET_PYTHON_VERSION} abi=${GEMMA4_OFFLINE_TARGET_ABI}"
    "${DOWNLOAD_PYTHON}" -m pip download "${PIP_SSL_ARGS[@]}" \
      "${PIP_TARGET_ARGS[@]}" \
      --dest "${PYTHON_WHEELHOUSE}" \
      --no-deps \
      "${spec}" || true
    if wheelhouse_has_package "${normalized}" >/dev/null; then
      echo "Downloaded target wheel for ${normalized}: $(wheelhouse_has_package "${normalized}")"
      return 0
    fi
  done
  echo "Missing required target wheel after retries: ${normalized}" >&2
  return 1
}

"${DOWNLOAD_PYTHON}" -m pip install "${PIP_SSL_ARGS[@]}" --upgrade pip wheel setuptools

download_wheels \
  pip \
  wheel \
  "setuptools>=77.0.3,<81.0.0" \
  "setuptools-rust>=1.9.0" \
  "setuptools-scm>=8.0" \
  "packaging>=24.2" \
  jinja2 \
  "MarkupSafe>=2.0"

download_required_target_binary_wheel "markupsafe" "MarkupSafe>=2.0" "markupsafe>=2.0"

download_wheels \
  --extra-index-url "${PYTORCH_INDEX_URL}" \
  "torch==${TORCH_VERSION}" \
  "ninja>=1.11" \
  "cmake>=3.28"

download_wheels_no_deps \
  --extra-index-url "${PYTORCH_INDEX_URL}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}"

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

"${DOWNLOAD_PYTHON}" -m pip download "${PIP_SSL_ARGS[@]}" \
  "${PIP_TARGET_ARGS[@]}" \
  --dest "${PYTHON_WHEELHOUSE}" \
  --index-url "${FLASHINFER_INDEX_URL}" \
  "flashinfer-jit-cache" || {
    echo "Warning: could not download flashinfer-jit-cache from ${FLASHINFER_INDEX_URL}." >&2
  }

download_wheels \
  "pyyaml>=6.0.2" \
  "requests>=2.32.0" \
  "numpy>=1.26" \
  "pillow>=10.0" \
  "jsonschema>=4.23" \
  "referencing>=0.35" \
  "jsonschema-specifications>=2023.12.1" \
  "attrs>=23" \
  "rpds-py>=0.20" \
  "openai>=1.60" \
  "httpx>=0.27" \
  "tqdm>=4.66"

"${DOWNLOAD_PYTHON}" -m pip download "${PIP_SSL_ARGS[@]}" \
  "${PIP_TARGET_ARGS[@]}" \
  --dest "${PYTHON_WHEELHOUSE}" \
  --no-deps \
  "matplotlib==${MATPLOTLIB_VERSION}" \
  "jupyterlab==${JUPYTERLAB_VERSION}" \
  "bitsandbytes==${BITSANDBYTES_VERSION}" \
  "truststore==${TRUSTSTORE_VERSION}"

"${DOWNLOAD_PYTHON}" - "${PYTHON_WHEELHOUSE}" "${GEMMA4_OFFLINE_TARGET_PYTHON_VERSION}" "${GEMMA4_OFFLINE_TARGET_ABI}" "${GEMMA4_OFFLINE_TARGET_PLATFORMS}" <<'PY'
import os
import re
import sys
from pathlib import Path

wheelhouse = Path(sys.argv[1])
wheel_names = [path.name.lower() for path in wheelhouse.glob("*.whl")]
required = {
    "pip": "pip",
    "wheel": "wheel",
    "setuptools": "setuptools",
    "jinja2": "jinja2",
    "MarkupSafe": "markupsafe",
}
missing = [
    name
    for name, normalized in required.items()
    if not any(wheel.replace("_", "-").startswith(f"{normalized}-") for wheel in wheel_names)
]
target_python = sys.argv[2]
target_abi = sys.argv[3].lower()
target_platforms = [
    platform.strip().lower()
    for platform in sys.argv[4].replace(",", " ").split()
    if platform.strip()
]
markupsafe_wheels = [wheel for wheel in wheel_names if wheel.startswith("markupsafe-")]
compatible_markupsafe = [
    wheel
    for wheel in markupsafe_wheels
    if (
        f"-cp{target_python}-" in wheel
        and f"-{target_abi}-" in wheel
        and any(platform in wheel for platform in target_platforms)
    )
]
if markupsafe_wheels and not compatible_markupsafe:
    print("Found MarkupSafe wheels, but none match target Python/platform:", file=sys.stderr)
    print(f"  target python=cp{target_python} abi={target_abi} platforms={','.join(target_platforms)}", file=sys.stderr)
    for wheel in markupsafe_wheels:
        print(f"  incompatible: {wheel}", file=sys.stderr)
    missing.append("MarkupSafe-compatible-target-wheel")

target_major = int(target_python[0])
target_minor = int(target_python[1:])
incompatible_python_wheels = []
for wheel in wheel_names:
    if not wheel.endswith(".whl"):
        continue
    parts = wheel[:-4].rsplit("-", 3)
    if len(parts) != 4:
        continue
    python_tag, abi_tag, _platform_tag = parts[1:]
    python_tags = python_tag.split(".")
    abi_tags = set(abi_tag.split("."))

    def is_python_compatible(tag: str) -> bool:
        if tag.startswith("py3"):
            return True
        if tag == f"cp{target_python}":
            return True
        match = re.fullmatch(r"cp(\d)(\d+)", tag)
        if match and "abi3" in abi_tags:
            major = int(match.group(1))
            minor = int(match.group(2))
            return major == target_major and minor <= target_minor
        return False

    if not any(is_python_compatible(tag) for tag in python_tags):
        incompatible_python_wheels.append(wheel)

if incompatible_python_wheels:
    print(f"Found wheels incompatible with target Python cp{target_python}:", file=sys.stderr)
    for wheel in sorted(incompatible_python_wheels):
        print(f"  incompatible: {wheel}", file=sys.stderr)
    missing.append("target-python-compatible-wheels")
if missing:
    print(f"Missing required offline wheels: {', '.join(missing)}", file=sys.stderr)
    print("Available wheels:", file=sys.stderr)
    for wheel in sorted(wheel_names):
        print(f"  {wheel}", file=sys.stderr)
    sys.exit(1)
PY

cat > "${BUNDLE_DIR}/README.txt" <<EOF
Gemma4 vLLM offline bundle

Wheelhouse:
  ${WHEELHOUSE}

Target:
  platforms=${GEMMA4_OFFLINE_TARGET_PLATFORMS}
  python=cp${GEMMA4_OFFLINE_TARGET_PYTHON_VERSION}
  abi=${GEMMA4_OFFLINE_TARGET_ABI}

Copy this whole folder to the offline GPU machine, then run from the A2UI repo:
  bash dataset/scripts/install_gemma4_vllm_offline_env.sh /path/to/gemma4_vllm_offline_bundle

The target machine should use the same Python minor version and compatible Linux/CUDA stack.
EOF

"${DOWNLOAD_PYTHON}" - "${PYTHON_BUNDLE_DIR}" "${PYTHON_WHEELHOUSE}" "${GEMMA4_OFFLINE_TARGET_PLATFORM}" "${GEMMA4_OFFLINE_TARGET_PLATFORMS}" "${GEMMA4_OFFLINE_TARGET_PYTHON_VERSION}" "${GEMMA4_OFFLINE_TARGET_ABI}" <<'PY'
import json
import os
import sys
from pathlib import Path

bundle = Path(sys.argv[1]).resolve()
wheelhouse = Path(sys.argv[2]).resolve()
manifest = {
    "bundle": str(bundle),
    "wheelhouse": str(wheelhouse),
    "wheel_count": len(list(wheelhouse.glob("*"))),
    "target_platform": sys.argv[3],
    "target_platforms": sys.argv[4],
    "target_python": f"cp{sys.argv[5]}",
    "target_abi": sys.argv[6],
}
(bundle / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
PY

echo "Offline bundle ready: ${BUNDLE_DIR}"
