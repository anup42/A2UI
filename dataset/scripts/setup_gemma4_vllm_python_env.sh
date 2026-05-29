#!/usr/bin/env bash
set -euo pipefail

# Create a non-container Python environment for serving Gemma4 31B with vLLM
# and running A2UI dataset Stage 1/2/3 against that local endpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-}"
ENV_DIR="${ENV_DIR:-${REPO_ROOT}/gemma4_vllm_env}"
GEMMA4_MODEL_ROOT="${GEMMA4_MODEL_ROOT:-}"
GEMMA4_MODEL_PATH="${GEMMA4_MODEL_PATH:-}"
GEMMA4_ASSISTANT_MODEL_PATH="${GEMMA4_ASSISTANT_MODEL_PATH:-}"

A2UI_DISABLE_SSL_VERIFY="${A2UI_DISABLE_SSL_VERIFY:-1}"
A2UI_SKIP_PIP_INSTALL="${A2UI_SKIP_PIP_INSTALL:-0}"
A2UI_REQUIRE_SPECULATIVE="${A2UI_REQUIRE_SPECULATIVE:-1}"
A2UI_VLLM_INSTALL_MODE="${A2UI_VLLM_INSTALL_MODE:-release}" # release|nightly|source|skip
A2UI_CLEAN_VLLM_STACK="${A2UI_CLEAN_VLLM_STACK:-1}"
VLLM_VERSION="${VLLM_VERSION:-0.22.0}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
VLLM_NIGHTLY_INDEX="${VLLM_NIGHTLY_INDEX:-https://wheels.vllm.ai/nightly/cu129}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu129}"
VLLM_SOURCE_REF="${VLLM_SOURCE_REF:-v0.22.0}"
FLASHINFER_CUDA_TAG="${FLASHINFER_CUDA_TAG:-cu130}"
FLASHINFER_INDEX_URL="${FLASHINFER_INDEX_URL:-https://flashinfer.ai/whl/${FLASHINFER_CUDA_TAG}}"

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
  if [[ -n "${py_lib_dir}" ]]; then
    while IFS= read -r path; do
      [[ -n "${path}" ]] && lib_candidates+=("${path}")
    done < <(find "${py_lib_dir}/nvidia" -type d -name lib 2>/dev/null | sort || true)
    while IFS= read -r path; do
      [[ -n "${path}" ]] && bin_candidates+=("${path}")
    done < <(find "${py_lib_dir}/nvidia" -type d -name bin 2>/dev/null | sort || true)
  fi
  lib_candidates+=(
    /usr/local/cuda/lib64
    /usr/local/cuda-13/lib64
    /usr/local/cuda-13.0/lib64
    /usr/local/cuda-13.1/lib64
    /usr/local/cuda-13.2/lib64
    /usr/lib/x86_64-linux-gnu
  )
  bin_candidates+=(
    /usr/local/cuda/bin
    /usr/local/cuda-13/bin
    /usr/local/cuda-13.0/bin
    /usr/local/cuda-13.1/bin
    /usr/local/cuda-13.2/bin
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
  for path in "${py_lib_dir}/nvidia/cuda_nvcc" /usr/local/cuda /usr/local/cuda-13 /usr/local/cuda-13.0 /usr/local/cuda-13.1 /usr/local/cuda-13.2; do
    if [[ -d "${path}" && -z "${CUDA_HOME:-}" ]]; then
      export CUDA_HOME="${path}"
      export CUDA_PATH="${path}"
    fi
  done
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

python -m pip install "${PIP_TRUSTED_ARGS[@]}" --upgrade pip wheel setuptools

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
      || true
  fi

  python -m uv pip install -U --reinstall \
    --extra-index-url "${PYTORCH_INDEX_URL}" \
    --index-strategy unsafe-best-match \
    "torch==${TORCH_VERSION}" \
    "torchvision" \
    "torchaudio" \
    "${UV_INSECURE_ARGS[@]}"

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
        --extra-index-url "${PYTORCH_INDEX_URL}" \
        --index-strategy unsafe-best-match \
        "torch==${TORCH_VERSION}" "torchvision" "torchaudio" \
        "${UV_INSECURE_ARGS[@]}"
      python -m uv pip install -U --reinstall \
        "git+https://github.com/vllm-project/vllm.git@${VLLM_SOURCE_REF}" \
        "${UV_INSECURE_ARGS[@]}"
      ;;
    skip)
      echo "Skipping vLLM install because A2UI_VLLM_INSTALL_MODE=skip"
      ;;
    *)
      echo "Unsupported A2UI_VLLM_INSTALL_MODE=${A2UI_VLLM_INSTALL_MODE}; use release, nightly, source, or skip." >&2
      exit 1
      ;;
  esac

  python -m uv pip install \
    "nvidia-cuda-runtime>=13,<14" \
    "nvidia-cuda-nvcc>=13,<14" \
    "${UV_INSECURE_ARGS[@]}" || {
      echo "Warning: could not install CUDA runtime/NVCC wheels. If vLLM or FlashInfer JIT fails, install nvidia-cuda-runtime and nvidia-cuda-nvcc manually." >&2
    }
  export_nvidia_python_libs

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
export GEMMA4_ENABLE_REASONING="\${GEMMA4_ENABLE_REASONING:-0}"
export GEMMA4_SPECULATIVE_MODE="\${GEMMA4_SPECULATIVE_MODE:-draft}"
export GEMMA4_SPECULATIVE_TOKENS="\${GEMMA4_SPECULATIVE_TOKENS:-4}"
export VLLM_USE_FLASHINFER_SAMPLER="\${VLLM_USE_FLASHINFER_SAMPLER:-1}"
export VLLM_HAS_FLASHINFER_CUBIN="\${VLLM_HAS_FLASHINFER_CUBIN:-1}"
export FLASHINFER_CUDA_TAG="\${FLASHINFER_CUDA_TAG:-${FLASHINFER_CUDA_TAG}}"
export A2UI_DISABLE_SSL_VERIFY="\${A2UI_DISABLE_SSL_VERIFY:-1}"
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
  if [[ -n "\${py_lib_dir}" ]]; then
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
  for path in /usr/local/cuda/lib64 /usr/local/cuda-13/lib64 /usr/local/cuda-13.0/lib64 /usr/local/cuda-13.1/lib64 /usr/local/cuda-13.2/lib64 /usr/lib/x86_64-linux-gnu; do
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
  for path in /usr/local/cuda/bin /usr/local/cuda-13/bin /usr/local/cuda-13.0/bin /usr/local/cuda-13.1/bin /usr/local/cuda-13.2/bin; do
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
  for path in "\${py_lib_dir}/nvidia/cuda_nvcc" /usr/local/cuda /usr/local/cuda-13 /usr/local/cuda-13.0 /usr/local/cuda-13.1 /usr/local/cuda-13.2; do
    if [[ -d "\${path}" && -z "\${CUDA_HOME:-}" ]]; then
      export CUDA_HOME="\${path}"
      export CUDA_PATH="\${path}"
    fi
  done
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
    echo "Speculative decoding is required. This setup pins vllm==${VLLM_VERSION}, whose upstream tag exposes --speculative-config." >&2
    echo "It also pins torch==${TORCH_VERSION}; if you see undefined torch symbols, rerun with A2UI_CLEAN_VLLM_STACK=1." >&2
    echo "If the flag is still missing, rerun with A2UI_VLLM_INSTALL_MODE=source VLLM_SOURCE_REF=v0.22.0." >&2
    echo "Set A2UI_REQUIRE_SPECULATIVE=0 only if you intentionally want to run without speculative decoding." >&2
    exit 1
  fi
fi
if ! grep -q -- "--reasoning-parser" <<<"${HELP_TEXT}"; then
  echo "Warning: this vLLM install did not expose --reasoning-parser; Gemma4 reasoning parser may not work." >&2
fi

echo
echo "Gemma4 Python environment ready."
echo "Target model: ${TARGET_PATH:-not resolved yet; set GEMMA4_MODEL_ROOT or GEMMA4_MODEL_PATH}"
echo "Assistant model: ${ASSISTANT_PATH:-not resolved yet; set GEMMA4_MODEL_ROOT or GEMMA4_ASSISTANT_MODEL_PATH}"
echo
echo "Activate with:"
echo "  source ${ENV_DIR}/activate_gemma4_vllm.sh"
echo
echo "Start vLLM with:"
echo "  bash dataset/scripts/run_gemma4_vllm_python.sh"
