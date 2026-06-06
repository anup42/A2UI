#!/usr/bin/env bash
set -euo pipefail

# Start Gemma4 vLLM while forcing resolution to prefer the selected Python
# virtualenv over user/system Python packages and native libraries.
#
# Use this on shared/offline machines where another vLLM/PyTorch install may be
# visible through PYTHONPATH, user-site packages, or LD_LIBRARY_PATH.
# It sanitizes the environment, prepends venv site-packages/native library paths,
# prints the resolved vLLM/Torch locations, then delegates to run_gemma4_vllm_python.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ENV_DIR="${ENV_DIR:-${REPO_ROOT}/gemma4_vllm_env}"
STRICT_ENV="${A2UI_VLLM_STRICT_ENV:-1}"
KEEP_PYTHONPATH="${A2UI_KEEP_EXISTING_PYTHONPATH:-0}"
KEEP_LD_LIBRARY_PATH="${A2UI_KEEP_EXISTING_LD_LIBRARY_PATH:-1}"

if [[ ! -f "${ENV_DIR}/bin/activate" ]]; then
  echo "Gemma4 venv activation script not found: ${ENV_DIR}/bin/activate" >&2
  echo "Set ENV_DIR to the correct venv or rerun setup_gemma4_vllm_python_env.sh." >&2
  exit 1
fi

if [[ -n "${VIRTUAL_ENV:-}" && "$(cd "${VIRTUAL_ENV}" 2>/dev/null && pwd || true)" != "$(cd "${ENV_DIR}" && pwd)" ]]; then
  echo "Warning: replacing active venv ${VIRTUAL_ENV} with ${ENV_DIR}" >&2
fi

# shellcheck source=/dev/null
source "${ENV_DIR}/bin/activate"
hash -r

export PYTHONNOUSERSITE=1
unset PYTHONHOME
if [[ "${STRICT_ENV}" = "1" ]]; then
  export PIP_REQUIRE_VIRTUALENV=0
  export VIRTUAL_ENV_DISABLE_PROMPT=1
fi

readarray -t ENV_INFO < <(python - <<'PY'
import os
import site
import sys
from pathlib import Path

prefix = Path(sys.prefix).resolve()
print(f"prefix={prefix}")
for p in site.getsitepackages():
    path = Path(p).resolve()
    if path.exists():
        print(f"site={path}")
        torch_lib = path / "torch" / "lib"
        if torch_lib.is_dir():
            print(f"lib={torch_lib}")
        for nvidia_lib in sorted(path.glob("nvidia/*/lib")):
            if nvidia_lib.is_dir():
                print(f"lib={nvidia_lib.resolve()}")
        for nvidia_bin in sorted(path.glob("nvidia/*/bin")):
            if nvidia_bin.is_dir():
                print(f"bin={nvidia_bin.resolve()}")
for rel in ("lib", "lib64"):
    path = prefix / rel
    if path.is_dir():
        print(f"lib={path}")
PY
)

ENV_PREFIX=""
ENV_SITE_PATHS=()
ENV_LIB_PATHS=()
ENV_BIN_PATHS=()
for line in "${ENV_INFO[@]}"; do
  case "${line}" in
    prefix=*) ENV_PREFIX="${line#prefix=}" ;;
    site=*) ENV_SITE_PATHS+=("${line#site=}") ;;
    lib=*) ENV_LIB_PATHS+=("${line#lib=}") ;;
    bin=*) ENV_BIN_PATHS+=("${line#bin=}") ;;
  esac
done

join_by_colon() {
  local result=""
  local item
  for item in "$@"; do
    [[ -n "${item}" && -d "${item}" ]] || continue
    if [[ -z "${result}" ]]; then
      result="${item}"
    else
      result="${result}:${item}"
    fi
  done
  printf '%s' "${result}"
}

ENV_SITE_JOINED="$(join_by_colon "${ENV_SITE_PATHS[@]}")"
ENV_LIB_JOINED="$(join_by_colon "${ENV_LIB_PATHS[@]}")"
ENV_BIN_JOINED="$(join_by_colon "${ENV_BIN_PATHS[@]}")"

if [[ "${KEEP_PYTHONPATH}" = "1" && -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${ENV_SITE_JOINED}${ENV_SITE_JOINED:+:}${PYTHONPATH}"
elif [[ -n "${ENV_SITE_JOINED}" ]]; then
  export PYTHONPATH="${ENV_SITE_JOINED}"
else
  unset PYTHONPATH
fi

export PATH="${ENV_DIR}/bin${ENV_BIN_JOINED:+:${ENV_BIN_JOINED}}:${PATH}"
if [[ "${KEEP_LD_LIBRARY_PATH}" = "1" && -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${ENV_LIB_JOINED}${ENV_LIB_JOINED:+:}${LD_LIBRARY_PATH}"
elif [[ -n "${ENV_LIB_JOINED}" ]]; then
  export LD_LIBRARY_PATH="${ENV_LIB_JOINED}"
else
  unset LD_LIBRARY_PATH
fi

export CUDA_HOME="${CUDA_HOME:-${ENV_DIR}/cuda_toolkit}"
if [[ -d "${CUDA_HOME}" ]]; then
  export CUDA_PATH="${CUDA_HOME}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:${CUDA_HOME}/targets/x86_64-linux/lib64:${LD_LIBRARY_PATH:-}"
fi

export A2UI_DISABLE_SSL_VERIFY="${A2UI_DISABLE_SSL_VERIFY:-1}"
if [[ "${A2UI_DISABLE_SSL_VERIFY}" = "1" ]]; then
  export CURL_CA_BUNDLE=""
  export REQUESTS_CA_BUNDLE=""
  export SSL_CERT_FILE=""
  export PYTHONHTTPSVERIFY=0
  export GIT_SSL_NO_VERIFY=1
  export HF_HUB_DISABLE_SSL_VERIFICATION=1
fi

python - <<'PY'
import importlib.util
import os
import sys
from pathlib import Path

print("A2UI environment-first vLLM launch")
print(f"  python={sys.executable}")
print(f"  sys.prefix={sys.prefix}")
print(f"  PYTHONNOUSERSITE={os.environ.get('PYTHONNOUSERSITE')}")
print(f"  PYTHONPATH={os.environ.get('PYTHONPATH', '')}")
print(f"  LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}")
for module in ("torch", "vllm"):
    spec = importlib.util.find_spec(module)
    print(f"  {module}_origin={spec.origin if spec else 'NOT_FOUND'}")
try:
    import torch
    print(f"  torch_version={torch.__version__}")
except Exception as exc:
    print(f"  torch_import_error={type(exc).__name__}: {exc}")
try:
    import vllm
    print(f"  vllm_version={getattr(vllm, '__version__', 'unknown')}")
except Exception as exc:
    print(f"  vllm_import_error={type(exc).__name__}: {exc}")
PY

exec "${SCRIPT_DIR}/run_gemma4_vllm_python.sh" "$@"
