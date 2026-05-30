#!/usr/bin/env bash
set -euo pipefail

# Create a non-container Python environment for serving Gemma4 31B with vLLM
# and running A2UI dataset Stage 1/2/3 against that local endpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-}"
ENV_DIR="${ENV_DIR:-${REPO_ROOT}/gemma4_vllm_env}"
A2UI_CUDA_TOOLKIT_DIR="${A2UI_CUDA_TOOLKIT_DIR:-${ENV_DIR}/cuda_toolkit}"
GEMMA4_MODEL_ROOT="${GEMMA4_MODEL_ROOT:-}"
GEMMA4_MODEL_PATH="${GEMMA4_MODEL_PATH:-}"
GEMMA4_ASSISTANT_MODEL_PATH="${GEMMA4_ASSISTANT_MODEL_PATH:-}"

A2UI_DISABLE_SSL_VERIFY="${A2UI_DISABLE_SSL_VERIFY:-1}"
A2UI_SKIP_PIP_INSTALL="${A2UI_SKIP_PIP_INSTALL:-0}"
A2UI_REQUIRE_SPECULATIVE="${A2UI_REQUIRE_SPECULATIVE:-1}"
A2UI_VLLM_INSTALL_MODE="${A2UI_VLLM_INSTALL_MODE:-source}" # source|release|nightly|skip
A2UI_CLEAN_VLLM_STACK="${A2UI_CLEAN_VLLM_STACK:-1}"
VLLM_VERSION="${VLLM_VERSION:-0.22.0}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0}"
VLLM_NIGHTLY_INDEX="${VLLM_NIGHTLY_INDEX:-https://wheels.vllm.ai/nightly/cu130}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
A2UI_TORCH_BACKEND="${A2UI_TORCH_BACKEND:-cu130}"
# Gemma4-special vLLM ref used by the Gemma4 speculative build flow.
VLLM_SOURCE_REF="${VLLM_SOURCE_REF:-9b4e83934d895b5f6e488411cd46c8d0915115a1}"
VLLM_SOURCE_DIR="${VLLM_SOURCE_DIR:-${ENV_DIR}/src/vllm-gemma4-speculative}"
FLASHINFER_CUDA_TAG="${FLASHINFER_CUDA_TAG:-cu130}"
FLASHINFER_INDEX_URL="${FLASHINFER_INDEX_URL:-https://flashinfer.ai/whl/${FLASHINFER_CUDA_TAG}}"
CUDA_RUNTIME_PACKAGE="${CUDA_RUNTIME_PACKAGE:-nvidia-cuda-runtime==13.3.29}"
CUDA_NVCC_PACKAGE="${CUDA_NVCC_PACKAGE:-nvidia-cuda-nvcc==13.3.33}"
A2UI_PREFER_PYTHON_CUDA="${A2UI_PREFER_PYTHON_CUDA:-1}"

if [[ -z "${PYTHON_BIN}" ]]; then
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "No Python executable found. Install Python 3.10-3.12 and retry." >&2
  exit 1
fi

PY_VER="$("${PYTHON_BIN}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
case "${PY_VER}" in
  3.10|3.11|3.12) ;;
  *)
    echo "Warning: detected Python ${PY_VER}. vLLM wheels are normally published for Python 3.10-3.12." >&2
    echo "Set PYTHON_BIN=python3.12 or PYTHON_BIN=python3.11 if install fails." >&2
    ;;
esac

apply_ssl_bypass() {
  if [[ "${A2UI_DISABLE_SSL_VERIFY}" = "1" ]]; then
    export CURL_CA_BUNDLE=""
    export REQUESTS_CA_BUNDLE=""
    export SSL_CERT_FILE=""
    export PYTHONHTTPSVERIFY=0
    export GIT_SSL_NO_VERIFY=1
    export HF_HUB_DISABLE_SSL_VERIFICATION=1
    git config --global http.sslVerify false >/dev/null 2>&1 || true
  fi
}

apply_ssl_bypass

link_dir_children() {
  local src_dir="$1"
  local dest_dir="$2"
  [[ -d "${src_dir}" ]] || return 0
  mkdir -p "${dest_dir}"
  local item
  shopt -s nullglob dotglob
  for item in "${src_dir}"/*; do
    ln -sfn "${item}" "${dest_dir}/$(basename "${item}")"
  done
  shopt -u nullglob dotglob
}

build_python_cuda_toolkit() {
  if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    return 0
  fi

  local py_lib_dir
  py_lib_dir="$(python - <<'PY' 2>/dev/null || true
import site
paths = site.getsitepackages()
print(paths[0] if paths else "")
PY
)"
  [[ -n "${py_lib_dir}" && -d "${py_lib_dir}/nvidia" ]] || return 0

  local toolkit="${A2UI_CUDA_TOOLKIT_DIR}"
  rm -rf "${toolkit}"
  mkdir -p \
    "${toolkit}/bin" \
    "${toolkit}/include" \
    "${toolkit}/lib64" \
    "${toolkit}/nvvm" \
    "${toolkit}/targets/x86_64-linux/include" \
    "${toolkit}/targets/x86_64-linux/lib" \
    "${toolkit}/targets/x86_64-linux/lib64"

  local path
  local canonical_cuda_root=""
  if [[ -x "${py_lib_dir}/nvidia/cu13/bin/nvcc" ]]; then
    canonical_cuda_root="${py_lib_dir}/nvidia/cu13"
  elif [[ -x "${py_lib_dir}/nvidia/cuda_nvcc/bin/nvcc" ]]; then
    canonical_cuda_root="${py_lib_dir}/nvidia/cuda_nvcc"
  else
    canonical_cuda_root="$(find "${py_lib_dir}/nvidia" -path '*/bin/nvcc' -type f 2>/dev/null | head -1 | sed 's#/bin/nvcc$##' || true)"
  fi

  if [[ -n "${canonical_cuda_root}" ]]; then
    link_dir_children "${canonical_cuda_root}/bin" "${toolkit}/bin"
    link_dir_children "${canonical_cuda_root}/include" "${toolkit}/include"
    link_dir_children "${canonical_cuda_root}/include" "${toolkit}/targets/x86_64-linux/include"
    link_dir_children "${canonical_cuda_root}/lib" "${toolkit}/lib64"
    link_dir_children "${canonical_cuda_root}/lib" "${toolkit}/targets/x86_64-linux/lib"
    link_dir_children "${canonical_cuda_root}/lib" "${toolkit}/targets/x86_64-linux/lib64"
    link_dir_children "${canonical_cuda_root}/nvvm" "${toolkit}/nvvm"
  fi

  while IFS= read -r path; do
    [[ -n "${canonical_cuda_root}" && "${path}" == "${canonical_cuda_root}/bin" ]] && continue
    link_dir_children "${path}" "${toolkit}/bin"
  done < <(find "${py_lib_dir}/nvidia" -type d -name bin 2>/dev/null | sort || true)

  # Do not merge every nvidia/*/include into the CUDA toolkit includes. Mixing
  # headers from cuda-nvcc 13.3 with older cuda-runtime 13.0 triggers PyTorch's
  # "FindCUDA says 13.3 but headers say 13.0" build failure.
  while IFS= read -r path; do
    [[ -n "${canonical_cuda_root}" && "${path}" == "${canonical_cuda_root}/lib" ]] && continue
    link_dir_children "${path}" "${toolkit}/lib64"
    link_dir_children "${path}" "${toolkit}/targets/x86_64-linux/lib"
    link_dir_children "${path}" "${toolkit}/targets/x86_64-linux/lib64"
  done < <(find "${py_lib_dir}/nvidia" -type d -name lib 2>/dev/null | sort || true)

  # nvcc resolves helper binaries like cicc relative to ../nvvm, so PATH alone is
  # not enough. Mirror the package's nvvm tree into the combined toolkit root.
  while IFS= read -r path; do
    [[ -n "${canonical_cuda_root}" && "${path}" == "${canonical_cuda_root}/nvvm" ]] && continue
    link_dir_children "${path}" "${toolkit}/nvvm"
  done < <(find "${py_lib_dir}/nvidia" -type d -name nvvm 2>/dev/null | sort || true)

  local lib_dir
  local versioned
  for lib_dir in "${toolkit}/lib64" "${toolkit}/targets/x86_64-linux/lib" "${toolkit}/targets/x86_64-linux/lib64"; do
    for versioned in "${lib_dir}"/libcudart.so.*; do
      [[ -e "${versioned}" ]] || continue
      [[ -e "${lib_dir}/libcudart.so" ]] || ln -sfn "$(basename "${versioned}")" "${lib_dir}/libcudart.so"
      break
    done
  done

  if [[ -x "${toolkit}/bin/nvcc" ]]; then
    export CUDA_HOME="${toolkit}"
    export CUDA_PATH="${toolkit}"
    export CUDACXX="${toolkit}/bin/nvcc"
    export CMAKE_CUDA_COMPILER="${toolkit}/bin/nvcc"
    export CUDAToolkit_ROOT="${toolkit}"
    export CUDA_TOOLKIT_ROOT_DIR="${toolkit}"
    export LIBRARY_PATH="${toolkit}/lib64:${toolkit}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="${toolkit}/lib64:${toolkit}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
    export PATH="${toolkit}/bin:${PATH}"
  fi
}

configure_cuda_build_env() {
  if [[ -z "${CUDA_HOME:-}" ]]; then
    echo "CUDA_HOME is unset; cannot configure CUDA build environment." >&2
    exit 1
  fi

  local cudart
  cudart="$(find "${CUDA_HOME}" \( -type f -o -type l \) 2>/dev/null | grep -E '/libcudart\.so($|\.)' | head -1 || true)"
  if [[ -z "${cudart}" ]]; then
    echo "CUDA_HOME has nvcc but no libcudart runtime library: ${CUDA_HOME}" >&2
    echo "Rerun setup after installing ${CUDA_RUNTIME_PACKAGE}, or set A2UI_CUDA_TOOLKIT_DIR to a complete CUDA toolkit." >&2
    exit 1
  fi
  if [[ ! -x "${CUDA_HOME}/bin/ptxas" ]]; then
    echo "CUDA_HOME has nvcc but no matching ptxas assembler: ${CUDA_HOME}/bin/ptxas" >&2
    echo "Rerun setup with A2UI_CLEAN_VLLM_STACK=1 so ${CUDA_NVCC_PACKAGE} is installed and mirrored into the venv CUDA toolkit." >&2
    exit 1
  fi

  export CUDA_CUDART_LIBRARY="${cudart}"
  export LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
  export CMAKE_ARGS="${CMAKE_ARGS:-} -DCUDAToolkit_ROOT=${CUDA_HOME} -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME} -DCUDA_CUDART_LIBRARY=${cudart} -DCUDA_CUDART_LIBRARY_RELEASE=${cudart} -DCMAKE_CUDA_COMPILER=${CUDA_HOME}/bin/nvcc"
  export SKBUILD_CMAKE_ARGS="${SKBUILD_CMAKE_ARGS:-} -DCUDAToolkit_ROOT=${CUDA_HOME} -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME} -DCUDA_CUDART_LIBRARY=${cudart} -DCUDA_CUDART_LIBRARY_RELEASE=${cudart} -DCMAKE_CUDA_COMPILER=${CUDA_HOME}/bin/nvcc"
  echo "Using CUDA_CUDART_LIBRARY=${CUDA_CUDART_LIBRARY}"
  "${CUDA_HOME}/bin/ptxas" --version | head -5 || true
}
export_nvidia_python_libs() {
  if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    return 0
  fi
  local py_lib_dir
  py_lib_dir="$(python - <<'PY' 2>/dev/null || true
import site
paths = site.getsitepackages()
print(paths[0] if paths else "")
PY
)"
  local lib_candidates=()
  local bin_candidates=()
  local cuda_home_candidates=()
  local python_cuda_home=""
  if [[ -n "${py_lib_dir}" ]]; then
    [[ -x "${A2UI_CUDA_TOOLKIT_DIR}/bin/nvcc" ]] && cuda_home_candidates+=("${A2UI_CUDA_TOOLKIT_DIR}")
    python_cuda_home="${py_lib_dir}/nvidia/cuda_nvcc"
    [[ -x "${python_cuda_home}/bin/nvcc" ]] && cuda_home_candidates+=("${python_cuda_home}")
    while IFS= read -r path; do
      [[ -n "${path}" ]] && lib_candidates+=("${path}")
    done < <(find "${py_lib_dir}/nvidia" -type d -name lib 2>/dev/null | sort || true)
    while IFS= read -r path; do
      [[ -n "${path}" ]] && bin_candidates+=("${path}")
    done < <(find "${py_lib_dir}/nvidia" -type d -name bin 2>/dev/null | sort || true)
  fi
  lib_candidates+=(
    /usr/local/cuda-13.0/lib64
    /usr/local/cuda-13.1/lib64
    /usr/local/cuda-13.2/lib64
    /usr/local/cuda-13.3/lib64
    /usr/local/cuda-13/lib64
    /usr/local/cuda/lib64
    /usr/lib/x86_64-linux-gnu
  )
  bin_candidates+=(
    /usr/local/cuda-13.0/bin
    /usr/local/cuda-13.1/bin
    /usr/local/cuda-13.2/bin
    /usr/local/cuda-13.3/bin
    /usr/local/cuda-13/bin
    /usr/local/cuda/bin
  )
  cuda_home_candidates+=(
    "${A2UI_CUDA_TOOLKIT_DIR}"
    /usr/local/cuda-13.0
    /usr/local/cuda-13.1
    /usr/local/cuda-13.2
    /usr/local/cuda-13.3
    /usr/local/cuda-13
    /usr/local/cuda
  )
  local joined=""
  local path
  for path in "${lib_candidates[@]}"; do
    if [[ -d "${path}" ]]; then
      if [[ -z "${joined}" ]]; then
        joined="${path}"
      else
        joined="${joined}:${path}"
      fi
    fi
  done
  if [[ -n "${joined}" ]]; then
    export LD_LIBRARY_PATH="${joined}:${LD_LIBRARY_PATH:-}"
  fi
  local bins=""
  for path in "${bin_candidates[@]}"; do
    if [[ -d "${path}" ]]; then
      if [[ -z "${bins}" ]]; then
        bins="${path}"
      else
        bins="${bins}:${path}"
      fi
    fi
  done
  if [[ -n "${bins}" ]]; then
    export PATH="${bins}:${PATH}"
  fi
  if [[ "${A2UI_PREFER_PYTHON_CUDA}" = "1" && -x "${A2UI_CUDA_TOOLKIT_DIR}/bin/nvcc" ]]; then
    export CUDA_HOME="${A2UI_CUDA_TOOLKIT_DIR}"
    export CUDA_PATH="${A2UI_CUDA_TOOLKIT_DIR}"
    export CUDACXX="${A2UI_CUDA_TOOLKIT_DIR}/bin/nvcc"
    export CMAKE_CUDA_COMPILER="${A2UI_CUDA_TOOLKIT_DIR}/bin/nvcc"
    export PATH="${A2UI_CUDA_TOOLKIT_DIR}/bin:${PATH}"
    export LD_LIBRARY_PATH="${A2UI_CUDA_TOOLKIT_DIR}/lib64:${A2UI_CUDA_TOOLKIT_DIR}/targets/x86_64-linux/lib:${A2UI_CUDA_TOOLKIT_DIR}/targets/x86_64-linux/lib64:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="${A2UI_CUDA_TOOLKIT_DIR}/lib64:${A2UI_CUDA_TOOLKIT_DIR}/targets/x86_64-linux/lib:${A2UI_CUDA_TOOLKIT_DIR}/targets/x86_64-linux/lib64:${LIBRARY_PATH:-}"
  elif [[ "${A2UI_PREFER_PYTHON_CUDA}" = "1" && -n "${python_cuda_home}" && -x "${python_cuda_home}/bin/nvcc" ]]; then
    export CUDA_HOME="${python_cuda_home}"
    export CUDA_PATH="${python_cuda_home}"
    export CUDACXX="${python_cuda_home}/bin/nvcc"
    export CMAKE_CUDA_COMPILER="${python_cuda_home}/bin/nvcc"
  elif [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    export CUDA_PATH="${CUDA_HOME}"
    export CUDACXX="${CUDA_HOME}/bin/nvcc"
    export CMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc"
  else
    unset CUDA_HOME
    unset CUDA_PATH
    unset CUDACXX
    unset CMAKE_CUDA_COMPILER
    for path in "${cuda_home_candidates[@]}"; do
      if [[ -x "${path}/bin/nvcc" ]]; then
        export CUDA_HOME="${path}"
        export CUDA_PATH="${path}"
        export CUDACXX="${path}/bin/nvcc"
        export CMAKE_CUDA_COMPILER="${path}/bin/nvcc"
        break
      fi
    done
  fi
  if [[ -n "${CUDA_HOME:-}" ]]; then
    export CUDAToolkit_ROOT="${CUDA_HOME}"
    export CUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}"
    local cudart
    cudart="$(find "${CUDA_HOME}" \( -type f -o -type l \) 2>/dev/null | grep -E '/libcudart\.so($|\.)' | head -1 || true)"
    if [[ -n "${cudart}" ]]; then
      export CUDA_CUDART_LIBRARY="${cudart}"
      export CMAKE_ARGS="${CMAKE_ARGS:-} -DCUDAToolkit_ROOT=${CUDA_HOME} -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME} -DCUDA_CUDART_LIBRARY=${cudart} -DCUDA_CUDART_LIBRARY_RELEASE=${cudart} -DCMAKE_CUDA_COMPILER=${CUDA_HOME}/bin/nvcc"
      export SKBUILD_CMAKE_ARGS="${SKBUILD_CMAKE_ARGS:-} -DCUDAToolkit_ROOT=${CUDA_HOME} -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME} -DCUDA_CUDART_LIBRARY=${cudart} -DCUDA_CUDART_LIBRARY_RELEASE=${cudart} -DCMAKE_CUDA_COMPILER=${CUDA_HOME}/bin/nvcc"
    fi
  fi
}

require_nvcc_for_source_build() {
  build_python_cuda_toolkit
  export_nvidia_python_libs
  if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "Using CUDA_HOME=${CUDA_HOME}"
    "${CUDA_HOME}/bin/nvcc" --version | head -5 || true
    configure_cuda_build_env
    return 0
  fi
  if command -v nvcc >/dev/null 2>&1; then
    local nvcc_bin
    nvcc_bin="$(command -v nvcc)"
    CUDA_HOME="$(cd "$(dirname "${nvcc_bin}")/.." && pwd)"
    export CUDA_HOME
    export CUDA_PATH="${CUDA_HOME}"
    echo "Using CUDA_HOME=${CUDA_HOME}"
    "${CUDA_HOME}/bin/nvcc" --version | head -5 || true
    configure_cuda_build_env
    return 0
  fi

  echo "CUDA NVCC compiler was not found, but A2UI_VLLM_INSTALL_MODE=source needs nvcc to build vLLM." >&2
  echo "Expected one of these paths to exist:" >&2
  echo "  ${VIRTUAL_ENV:-<venv>}/lib/python*/site-packages/nvidia/cuda_nvcc/bin/nvcc" >&2
  echo "  /usr/local/cuda-13.0/bin/nvcc" >&2
  echo "  /usr/local/cuda/bin/nvcc" >&2
  echo "The setup script attempted to install: ${CUDA_NVCC_PACKAGE}" >&2
  echo "Current CUDA_HOME=${CUDA_HOME:-unset}" >&2
  echo "Current PATH=${PATH}" >&2
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "Installed CUDA-related packages:" >&2
    python -m pip list 2>/dev/null | grep -E 'nvidia-cuda|torch|vllm|flashinfer' >&2 || true
    echo "CUDA files under virtualenv:" >&2
    find "${VIRTUAL_ENV}" -path '*cuda_nvcc*' -maxdepth 8 2>/dev/null | head -40 >&2 || true
  fi
  exit 1
}

is_dir() {
  [[ -n "${1:-}" && -d "$1" ]]
}

find_child_dir() {
  local root="$1"
  shift
  local name
  for name in "$@"; do
    if [[ -d "${root}/${name}" ]]; then
      printf '%s\n' "${root}/${name}"
      return 0
    fi
  done
  return 1
}

resolve_target_model() {
  if is_dir "${GEMMA4_MODEL_PATH}"; then
    printf '%s\n' "${GEMMA4_MODEL_PATH}"
    return 0
  fi
  if is_dir "${GEMMA4_MODEL_ROOT}"; then
    if [[ -f "${GEMMA4_MODEL_ROOT}/config.json" ]]; then
      printf '%s\n' "${GEMMA4_MODEL_ROOT}"
      return 0
    fi
    find_child_dir "${GEMMA4_MODEL_ROOT}" \
      gemma-4-31B-it \
      gemma-4-31b-it \
      google--gemma-4-31B-it \
      google--gemma-4-31b-it \
      gemma4-31b \
      gemma4-31b-it \
      Gemma-4-31B-it \
      Gemma4-31B-it && return 0
  fi
  return 1
}

resolve_assistant_model() {
  if is_dir "${GEMMA4_ASSISTANT_MODEL_PATH}"; then
    printf '%s\n' "${GEMMA4_ASSISTANT_MODEL_PATH}"
    return 0
  fi
  if is_dir "${GEMMA4_MODEL_ROOT}"; then
    find_child_dir "${GEMMA4_MODEL_ROOT}" \
      gemma-4-31B-it-assistant \
      gemma-4-31b-it-assistant \
      google--gemma-4-31B-it-assistant \
      google--gemma-4-31b-it-assistant \
      gemma4-31b-assistant \
      gemma4-31b-it-assistant \
      Gemma-4-31B-it-assistant \
      Gemma4-31B-it-assistant && return 0
  fi
  return 1
}

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
apply_ssl_bypass
export_nvidia_python_libs

PIP_TRUSTED_ARGS=()
UV_INSECURE_ARGS=()
if [[ "${A2UI_DISABLE_SSL_VERIFY}" = "1" ]]; then
  mkdir -p "${ENV_DIR}/pip_conf"
  cat > "${ENV_DIR}/pip_conf/pip.conf" <<'EOF'
[global]
trusted-host =
    pypi.org
    files.pythonhosted.org
    download.pytorch.org
    wheels.vllm.ai
    github.com
    codeload.github.com
    raw.githubusercontent.com
    flashinfer.ai
disable-pip-version-check = true
EOF
  export PIP_CONFIG_FILE="${ENV_DIR}/pip_conf/pip.conf"
  PIP_TRUSTED_ARGS+=(
    --trusted-host pypi.org
    --trusted-host files.pythonhosted.org
    --trusted-host download.pytorch.org
    --trusted-host wheels.vllm.ai
    --trusted-host github.com
    --trusted-host codeload.github.com
    --trusted-host raw.githubusercontent.com
    --trusted-host flashinfer.ai
  )
  UV_INSECURE_ARGS+=(--allow-insecure-host pypi.org)
  UV_INSECURE_ARGS+=(--allow-insecure-host files.pythonhosted.org)
  UV_INSECURE_ARGS+=(--allow-insecure-host download.pytorch.org)
  UV_INSECURE_ARGS+=(--allow-insecure-host wheels.vllm.ai)
  UV_INSECURE_ARGS+=(--allow-insecure-host github.com)
  UV_INSECURE_ARGS+=(--allow-insecure-host codeload.github.com)
  UV_INSECURE_ARGS+=(--allow-insecure-host raw.githubusercontent.com)
  UV_INSECURE_ARGS+=(--allow-insecure-host flashinfer.ai)
fi

python -m pip install "${PIP_TRUSTED_ARGS[@]}" --upgrade \
  pip \
  wheel \
  "setuptools>=77.0.3,<81.0.0" \
  "setuptools-rust>=1.9.0" \
  "setuptools-scm>=8.0" \
  "packaging>=24.2" \
  jinja2

if [[ "${A2UI_SKIP_PIP_INSTALL}" != "1" ]]; then
  python -m pip install "${PIP_TRUSTED_ARGS[@]}" "uv>=0.5.0"

  if [[ "${A2UI_CLEAN_VLLM_STACK}" = "1" && "${A2UI_VLLM_INSTALL_MODE}" != "skip" ]]; then
    python -m uv pip uninstall -y \
      vllm \
      torch \
      torchvision \
      torchaudio \
      xformers \
      flashinfer-python \
      flashinfer-cubin \
      flashinfer-jit-cache \
      nvidia-cuda-runtime \
      nvidia-cuda-nvcc \
      nvidia-cuda-runtime-cu12 \
      nvidia-cuda-nvcc-cu12 \
      nvidia-cuda-runtime-cu13 \
      nvidia-cuda-nvcc-cu13 \
      || true
  fi

  python -m uv pip install -U --reinstall \
    --torch-backend="${A2UI_TORCH_BACKEND}" \
    --extra-index-url "${PYTORCH_INDEX_URL}" \
    --index-strategy unsafe-best-match \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    "ninja>=1.11" \
    "cmake>=3.28" \
    "${UV_INSECURE_ARGS[@]}"

  python -m uv pip install \
    "${CUDA_RUNTIME_PACKAGE}" \
    "${CUDA_NVCC_PACKAGE}" \
    "${UV_INSECURE_ARGS[@]}" || {
      echo "Warning: could not install CUDA runtime/NVCC wheels (${CUDA_RUNTIME_PACKAGE}, ${CUDA_NVCC_PACKAGE}). If vLLM or FlashInfer JIT fails, install matching CUDA runtime/NVCC packages manually." >&2
    }
  build_python_cuda_toolkit
  export_nvidia_python_libs

  case "${A2UI_VLLM_INSTALL_MODE}" in
    release)
      python -m uv pip install -U --reinstall "vllm==${VLLM_VERSION}" \
        --extra-index-url "${PYTORCH_INDEX_URL}" \
        --index-strategy unsafe-best-match \
        "${UV_INSECURE_ARGS[@]}"
      ;;
    nightly)
      python -m uv pip install -U --reinstall vllm --pre \
        --extra-index-url "${VLLM_NIGHTLY_INDEX}" \
        --extra-index-url "${PYTORCH_INDEX_URL}" \
        --index-strategy unsafe-best-match \
        "${UV_INSECURE_ARGS[@]}"
      ;;
    source)
      python -m uv pip install -U --reinstall \
        --torch-backend="${A2UI_TORCH_BACKEND}" \
        --extra-index-url "${PYTORCH_INDEX_URL}" \
        --index-strategy unsafe-best-match \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" \
        "${UV_INSECURE_ARGS[@]}"
      mkdir -p "$(dirname "${VLLM_SOURCE_DIR}")"
      if [[ ! -d "${VLLM_SOURCE_DIR}/.git" ]]; then
        git clone https://github.com/vllm-project/vllm.git "${VLLM_SOURCE_DIR}"
      fi
      git -C "${VLLM_SOURCE_DIR}" fetch --all --tags
      git -C "${VLLM_SOURCE_DIR}" checkout "${VLLM_SOURCE_REF}"
      git -C "${VLLM_SOURCE_DIR}" submodule update --init --recursive
      require_nvcc_for_source_build
      python -m uv pip install \
        --torch-backend="${A2UI_TORCH_BACKEND}" \
        --no-build-isolation \
        -e "${VLLM_SOURCE_DIR}" \
        "${UV_INSECURE_ARGS[@]}"
      ;;
    skip)
      echo "Skipping vLLM install because A2UI_VLLM_INSTALL_MODE=skip"
      ;;
    *)
      echo "Unsupported A2UI_VLLM_INSTALL_MODE=${A2UI_VLLM_INSTALL_MODE}; use source, release, nightly, or skip." >&2
      exit 1
      ;;
  esac

  python -m uv pip install -U \
    "flashinfer-python" \
    "flashinfer-cubin" \
    "${UV_INSECURE_ARGS[@]}" || {
      echo "Warning: could not install flashinfer-python/flashinfer-cubin from PyPI." >&2
    }
  python -m pip install "${PIP_TRUSTED_ARGS[@]}" --index-url "${FLASHINFER_INDEX_URL}" -U "flashinfer-jit-cache" || {
    echo "Warning: could not install flashinfer-jit-cache from ${FLASHINFER_INDEX_URL}. FlashInfer may JIT compile kernels on first use." >&2
  }

  python -m uv pip install \
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
    "ninja>=1.11" \
    "cmake>=3.28" \
    "${UV_INSECURE_ARGS[@]}"
fi

export_nvidia_python_libs

TARGET_PATH="$(resolve_target_model || true)"
ASSISTANT_PATH="$(resolve_assistant_model || true)"

cat > "${ENV_DIR}/activate_gemma4_vllm.sh" <<EOF
#!/usr/bin/env bash
source "${ENV_DIR}/bin/activate"
export GEMMA4_MODEL_ROOT="${GEMMA4_MODEL_ROOT}"
export GEMMA4_MODEL_PATH="${TARGET_PATH}"
export GEMMA4_ASSISTANT_MODEL_PATH="${ASSISTANT_PATH}"
export GEMMA4_MODEL_ID="\${GEMMA4_MODEL_ID:-google/gemma-4-31B-it}"
export CUDA_VISIBLE_DEVICES="\${CUDA_VISIBLE_DEVICES:-}"
export A2UI_VLLM_GPUS="\${A2UI_VLLM_GPUS:-}"
export VLLM_MAX_MODEL_LEN="\${VLLM_MAX_MODEL_LEN:-8192}"
export VLLM_GPU_MEMORY_UTILIZATION="\${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
export VLLM_MAX_NUM_BATCHED_TOKENS="\${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
export GEMMA4_ENABLE_REASONING="\${GEMMA4_ENABLE_REASONING:-0}"
export GEMMA4_SPECULATIVE_MODE="\${GEMMA4_SPECULATIVE_MODE:-draft}"
export GEMMA4_SPECULATIVE_TOKENS="\${GEMMA4_SPECULATIVE_TOKENS:-4}"
export VLLM_USE_FLASHINFER_SAMPLER="\${VLLM_USE_FLASHINFER_SAMPLER:-1}"
export VLLM_HAS_FLASHINFER_CUBIN="\${VLLM_HAS_FLASHINFER_CUBIN:-1}"
export FLASHINFER_CUDA_TAG="\${FLASHINFER_CUDA_TAG:-${FLASHINFER_CUDA_TAG}}"
export A2UI_VLLM_INSTALL_MODE="\${A2UI_VLLM_INSTALL_MODE:-${A2UI_VLLM_INSTALL_MODE}}"
export VLLM_SOURCE_REF="\${VLLM_SOURCE_REF:-${VLLM_SOURCE_REF}}"
export PYTORCH_INDEX_URL="\${PYTORCH_INDEX_URL:-${PYTORCH_INDEX_URL}}"
export A2UI_TORCH_BACKEND="\${A2UI_TORCH_BACKEND:-${A2UI_TORCH_BACKEND}}"
export A2UI_DISABLE_SSL_VERIFY="\${A2UI_DISABLE_SSL_VERIFY:-1}"
export A2UI_PREFER_PYTHON_CUDA="\${A2UI_PREFER_PYTHON_CUDA:-1}"
export A2UI_CUDA_TOOLKIT_DIR="\${A2UI_CUDA_TOOLKIT_DIR:-${A2UI_CUDA_TOOLKIT_DIR}}"
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""
export SSL_CERT_FILE=""
export PYTHONHTTPSVERIFY=0
export GIT_SSL_NO_VERIFY=1
export HF_HUB_DISABLE_SSL_VERIFICATION=1
export PIP_CONFIG_FILE="${ENV_DIR}/pip_conf/pip.conf"
_a2ui_export_nvidia_python_libs() {
  local py_lib_dir
  py_lib_dir="\$(python - <<'PY' 2>/dev/null || true
import site
paths = site.getsitepackages()
print(paths[0] if paths else "")
PY
)"
  local joined=""
  local bins=""
  local python_cuda_home=""
  if [[ -n "\${py_lib_dir}" ]]; then
    python_cuda_home="\${py_lib_dir}/nvidia/cuda_nvcc"
    while IFS= read -r path; do
      if [[ -d "\${path}" ]]; then
        if [[ -z "\${joined}" ]]; then
          joined="\${path}"
        else
          joined="\${joined}:\${path}"
        fi
      fi
    done < <(find "\${py_lib_dir}/nvidia" -type d -name lib 2>/dev/null | sort || true)
    while IFS= read -r path; do
      if [[ -d "\${path}" ]]; then
        if [[ -z "\${bins}" ]]; then
          bins="\${path}"
        else
          bins="\${bins}:\${path}"
        fi
      fi
    done < <(find "\${py_lib_dir}/nvidia" -type d -name bin 2>/dev/null | sort || true)
  fi
  for path in /usr/local/cuda-13.0/lib64 /usr/local/cuda-13.1/lib64 /usr/local/cuda-13.2/lib64 /usr/local/cuda-13.3/lib64 /usr/local/cuda-13/lib64 /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu; do
    if [[ -d "\${path}" ]]; then
      if [[ -z "\${joined}" ]]; then
        joined="\${path}"
      else
        joined="\${joined}:\${path}"
      fi
    fi
  done
  if [[ -n "\${joined}" ]]; then
    export LD_LIBRARY_PATH="\${joined}:\${LD_LIBRARY_PATH:-}"
  fi
  for path in /usr/local/cuda-13.0/bin /usr/local/cuda-13.1/bin /usr/local/cuda-13.2/bin /usr/local/cuda-13.3/bin /usr/local/cuda-13/bin /usr/local/cuda/bin; do
    if [[ -d "\${path}" ]]; then
      if [[ -z "\${bins}" ]]; then
        bins="\${path}"
      else
        bins="\${bins}:\${path}"
      fi
    fi
  done
  if [[ -n "\${bins}" ]]; then
    export PATH="\${bins}:\${PATH}"
  fi
  if [[ "\${A2UI_PREFER_PYTHON_CUDA}" = "1" && -x "\${A2UI_CUDA_TOOLKIT_DIR}/bin/nvcc" ]]; then
    export CUDA_HOME="\${A2UI_CUDA_TOOLKIT_DIR}"
    export CUDA_PATH="\${A2UI_CUDA_TOOLKIT_DIR}"
    export CUDACXX="\${A2UI_CUDA_TOOLKIT_DIR}/bin/nvcc"
    export CMAKE_CUDA_COMPILER="\${A2UI_CUDA_TOOLKIT_DIR}/bin/nvcc"
    export PATH="\${A2UI_CUDA_TOOLKIT_DIR}/bin:\${PATH}"
    export LD_LIBRARY_PATH="\${A2UI_CUDA_TOOLKIT_DIR}/lib64:\${A2UI_CUDA_TOOLKIT_DIR}/targets/x86_64-linux/lib:\${A2UI_CUDA_TOOLKIT_DIR}/targets/x86_64-linux/lib64:\${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="\${A2UI_CUDA_TOOLKIT_DIR}/lib64:\${A2UI_CUDA_TOOLKIT_DIR}/targets/x86_64-linux/lib:\${A2UI_CUDA_TOOLKIT_DIR}/targets/x86_64-linux/lib64:\${LIBRARY_PATH:-}"
  elif [[ "\${A2UI_PREFER_PYTHON_CUDA}" = "1" && -n "\${python_cuda_home}" && -x "\${python_cuda_home}/bin/nvcc" ]]; then
    export CUDA_HOME="\${python_cuda_home}"
    export CUDA_PATH="\${python_cuda_home}"
    export CUDACXX="\${python_cuda_home}/bin/nvcc"
    export CMAKE_CUDA_COMPILER="\${python_cuda_home}/bin/nvcc"
  elif [[ -n "\${CUDA_HOME:-}" && -x "\${CUDA_HOME}/bin/nvcc" ]]; then
    export CUDA_PATH="\${CUDA_HOME}"
    export CUDACXX="\${CUDA_HOME}/bin/nvcc"
    export CMAKE_CUDA_COMPILER="\${CUDA_HOME}/bin/nvcc"
  else
    unset CUDA_HOME
    unset CUDA_PATH
    unset CUDACXX
    unset CMAKE_CUDA_COMPILER
    for path in /usr/local/cuda-13.0 /usr/local/cuda-13.1 /usr/local/cuda-13.2 /usr/local/cuda-13.3 /usr/local/cuda-13 /usr/local/cuda; do
      if [[ -x "\${path}/bin/nvcc" ]]; then
        export CUDA_HOME="\${path}"
        export CUDA_PATH="\${path}"
        export CUDACXX="\${path}/bin/nvcc"
        export CMAKE_CUDA_COMPILER="\${path}/bin/nvcc"
        break
      fi
    done
  fi
  if [[ -n "\${CUDA_HOME:-}" ]]; then
    export CUDAToolkit_ROOT="\${CUDA_HOME}"
    export CUDA_TOOLKIT_ROOT_DIR="\${CUDA_HOME}"
    local cudart
    cudart="\$(find "\${CUDA_HOME}" \( -type f -o -type l \) 2>/dev/null | grep -E '/libcudart\.so($|\.)' | head -1 || true)"
    if [[ -n "\${cudart}" ]]; then
      export CUDA_CUDART_LIBRARY="\${cudart}"
      export CMAKE_ARGS="\${CMAKE_ARGS:-} -DCUDAToolkit_ROOT=\${CUDA_HOME} -DCUDA_TOOLKIT_ROOT_DIR=\${CUDA_HOME} -DCUDA_CUDART_LIBRARY=\${cudart} -DCUDA_CUDART_LIBRARY_RELEASE=\${cudart} -DCMAKE_CUDA_COMPILER=\${CUDA_HOME}/bin/nvcc"
      export SKBUILD_CMAKE_ARGS="\${SKBUILD_CMAKE_ARGS:-} -DCUDAToolkit_ROOT=\${CUDA_HOME} -DCUDA_TOOLKIT_ROOT_DIR=\${CUDA_HOME} -DCUDA_CUDART_LIBRARY=\${cudart} -DCUDA_CUDART_LIBRARY_RELEASE=\${cudart} -DCMAKE_CUDA_COMPILER=\${CUDA_HOME}/bin/nvcc"
    fi
  fi
}
_a2ui_export_nvidia_python_libs
export LOCAL_ALLOW_HTTP_ENDPOINT=1
export LOCAL_STRICT_OFFLINE=0
export LOCAL_VLLM_STRIP_THINKING=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
EOF
chmod +x "${ENV_DIR}/activate_gemma4_vllm.sh"

# Many manual debug sessions source only the venv's standard bin/activate.
# Patch it as well so CUDA discovery works without remembering the wrapper.
ACTIVATE_FILE="${ENV_DIR}/bin/activate"
if [[ -f "${ACTIVATE_FILE}" ]]; then
  python - "${ACTIVATE_FILE}" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()
text = re.sub(
    r"\n?# >>> A2UI Gemma4 CUDA env >>>.*?# <<< A2UI Gemma4 CUDA env <<<\n?",
    "\n",
    text,
    flags=re.S,
)
path.write_text(text.rstrip() + "\n")
PY
  cat >> "${ACTIVATE_FILE}" <<EOF
# >>> A2UI Gemma4 CUDA env >>>
export A2UI_CUDA_TOOLKIT_DIR="\${A2UI_CUDA_TOOLKIT_DIR:-${A2UI_CUDA_TOOLKIT_DIR}}"
export A2UI_PREFER_PYTHON_CUDA="\${A2UI_PREFER_PYTHON_CUDA:-1}"
if [[ "\${A2UI_PREFER_PYTHON_CUDA}" = "1" && -x "\${A2UI_CUDA_TOOLKIT_DIR}/bin/nvcc" ]]; then
  export CUDA_HOME="\${A2UI_CUDA_TOOLKIT_DIR}"
  export CUDA_PATH="\${A2UI_CUDA_TOOLKIT_DIR}"
  export CUDACXX="\${A2UI_CUDA_TOOLKIT_DIR}/bin/nvcc"
  export CMAKE_CUDA_COMPILER="\${A2UI_CUDA_TOOLKIT_DIR}/bin/nvcc"
  export CUDAToolkit_ROOT="\${A2UI_CUDA_TOOLKIT_DIR}"
  export CUDA_TOOLKIT_ROOT_DIR="\${A2UI_CUDA_TOOLKIT_DIR}"
  export PATH="\${A2UI_CUDA_TOOLKIT_DIR}/bin:\${PATH}"
  export LD_LIBRARY_PATH="\${A2UI_CUDA_TOOLKIT_DIR}/lib64:\${A2UI_CUDA_TOOLKIT_DIR}/targets/x86_64-linux/lib:\${A2UI_CUDA_TOOLKIT_DIR}/targets/x86_64-linux/lib64:\${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="\${A2UI_CUDA_TOOLKIT_DIR}/lib64:\${A2UI_CUDA_TOOLKIT_DIR}/targets/x86_64-linux/lib:\${A2UI_CUDA_TOOLKIT_DIR}/targets/x86_64-linux/lib64:\${LIBRARY_PATH:-}"
  _a2ui_cudart="\$(find "\${A2UI_CUDA_TOOLKIT_DIR}" \( -type f -o -type l \) 2>/dev/null | grep -E '/libcudart\\.so($|\\.)' | head -1 || true)"
  if [[ -n "\${_a2ui_cudart}" ]]; then
    export CUDA_CUDART_LIBRARY="\${_a2ui_cudart}"
    export CMAKE_ARGS="\${CMAKE_ARGS:-} -DCUDAToolkit_ROOT=\${CUDA_HOME} -DCUDA_TOOLKIT_ROOT_DIR=\${CUDA_HOME} -DCUDA_CUDART_LIBRARY=\${_a2ui_cudart} -DCUDA_CUDART_LIBRARY_RELEASE=\${_a2ui_cudart} -DCMAKE_CUDA_COMPILER=\${CUDA_HOME}/bin/nvcc"
    export SKBUILD_CMAKE_ARGS="\${SKBUILD_CMAKE_ARGS:-} -DCUDAToolkit_ROOT=\${CUDA_HOME} -DCUDA_TOOLKIT_ROOT_DIR=\${CUDA_HOME} -DCUDA_CUDART_LIBRARY=\${_a2ui_cudart} -DCUDA_CUDART_LIBRARY_RELEASE=\${_a2ui_cudart} -DCMAKE_CUDA_COMPILER=\${CUDA_HOME}/bin/nvcc"
  fi
  unset _a2ui_cudart
fi
# <<< A2UI Gemma4 CUDA env <<<
EOF
fi

python - <<'PY'
import importlib
mods = ["vllm", "yaml", "requests", "numpy", "PIL", "matplotlib",
        "jsonschema", "referencing", "openai", "httpx", "truststore", "tqdm"]
for name in mods:
    importlib.import_module(name)
import vllm
import torch
print("vllm", getattr(vllm, "__version__", "unknown"))
print("torch", getattr(torch, "__version__", "unknown"))
print("Gemma4 Python env dependency imports OK")
PY

HELP_TEXT="$(vllm serve --help 2>&1 || true)"
if ! grep -q -- "--speculative-config" <<<"${HELP_TEXT}"; then
  echo "Warning: this vLLM install did not expose --speculative-config." >&2
  if [[ "${A2UI_REQUIRE_SPECULATIVE}" = "1" ]]; then
    python - <<'PY' >&2 || true
try:
    import vllm
    print("Installed vLLM:", getattr(vllm, "__version__", "unknown"))
except Exception as exc:
    print("Could not import vLLM to read version:", exc)
PY
    echo "Speculative decoding is required. This setup defaults to the Gemma4-special vLLM source ref ${VLLM_SOURCE_REF}, the same ref used by the Docker Gemma4 speculative build." >&2
    echo "It also pins torch==${TORCH_VERSION}; if you see undefined torch symbols, rerun with A2UI_CLEAN_VLLM_STACK=1." >&2
    echo "If the flag is still missing, rerun with A2UI_VLLM_INSTALL_MODE=source VLLM_SOURCE_REF=9b4e83934d895b5f6e488411cd46c8d0915115a1." >&2
    echo "Set A2UI_REQUIRE_SPECULATIVE=0 only if you intentionally want to run without speculative decoding." >&2
    exit 1
  fi
fi
if ! grep -q -- "--reasoning-parser" <<<"${HELP_TEXT}"; then
  echo "Warning: this vLLM install did not expose --reasoning-parser; Gemma4 reasoning parser may not work." >&2
fi

echo
echo "Gemma4 Python environment ready."
echo "Install mode: ${A2UI_VLLM_INSTALL_MODE}"
echo "vLLM source ref: ${VLLM_SOURCE_REF}"
echo "PyTorch index: ${PYTORCH_INDEX_URL}"
echo "Torch backend: ${A2UI_TORCH_BACKEND}"
echo "Target model: ${TARGET_PATH:-not resolved yet; set GEMMA4_MODEL_ROOT or GEMMA4_MODEL_PATH}"
echo "Assistant model: ${ASSISTANT_PATH:-not resolved yet; set GEMMA4_MODEL_ROOT or GEMMA4_ASSISTANT_MODEL_PATH}"
echo
echo "Activate with:"
echo "  source ${ENV_DIR}/activate_gemma4_vllm.sh"
echo
echo "Start vLLM with:"
echo "  bash dataset/scripts/run_gemma4_vllm_python.sh"
