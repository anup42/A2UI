#!/usr/bin/env bash
set -euo pipefail

# Online setup for a 2-GPU A100 machine. This script creates a Python 3.11
# environment, installs dataset/vLLM dependencies, downloads the Qwen model, and
# writes an activation file with the expected A2UI dataset-generation env vars.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
ENV_DIR="${ENV_DIR:-${REPO_ROOT}/qwen_vllm_env}"
REQUESTED_ENV_DIR="${ENV_DIR}"
MINIFORGE_DIR="${MINIFORGE_DIR:-${HOME}/miniforge3}"
QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3.6-35B-A3B}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${HOME}/models/Qwen--Qwen3.6-35B-A3B}"
REQ_FILE="${REQ_FILE:-${REPO_ROOT}/dataset/requirements-qwen-vllm.txt}"
A2UI_CA_BUNDLE="${A2UI_CA_BUNDLE:-}"
A2UI_DISABLE_SSL_VERIFY="${A2UI_DISABLE_SSL_VERIFY:-1}"
A2UI_DOWNLOAD_QWEN_MODEL="${A2UI_DOWNLOAD_QWEN_MODEL:-0}"
A2UI_VLLM_VERSION="${A2UI_VLLM_VERSION:-0.11.2}"
A2UI_VLLM_INSTALL_BACKEND="${A2UI_VLLM_INSTALL_BACKEND:-uv-pypi}"
A2UI_TORCH_VERSION="${A2UI_TORCH_VERSION:-2.9.0}"
A2UI_ONLY_BINARY="${A2UI_ONLY_BINARY:-1}"
A2UI_VLLM_CUDA_VARIANT="${A2UI_VLLM_CUDA_VARIANT:-126}"
A2UI_PYTORCH_INDEX_URL="${A2UI_PYTORCH_INDEX_URL:-}"
A2UI_TRANSFORMERS_VERSION="${A2UI_TRANSFORMERS_VERSION:-source}"
A2UI_TRANSFORMERS_SOURCE_DIR="${A2UI_TRANSFORMERS_SOURCE_DIR:-${HOME}/transformer}"
A2UI_TRANSFORMERS_SOURCE_URL="${A2UI_TRANSFORMERS_SOURCE_URL:-https://codeload.github.com/huggingface/transformers/zip/refs/heads/main}"
A2UI_DOWNLOAD_TRANSFORMERS_SOURCE="${A2UI_DOWNLOAD_TRANSFORMERS_SOURCE:-1}"
A2UI_TRANSFORMERS_INSTALL_SPEC="${A2UI_TRANSFORMERS_INSTALL_SPEC:-}"
A2UI_HUGGINGFACE_HUB_SPEC="${A2UI_HUGGINGFACE_HUB_SPEC:-huggingface-hub>=1.5.0,<2.0}"
A2UI_TOKENIZERS_SPEC="${A2UI_TOKENIZERS_SPEC:-tokenizers>=0.22.0,<=0.23.0}"
export A2UI_DISABLE_SSL_VERIFY

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export A2UI_VLLM_GPUS="${A2UI_VLLM_GPUS:-2}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
export VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"

python_supported() {
  "$1" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)
PY
}

python_version_string() {
  "$1" - <<'PY' 2>/dev/null || true
import sys
print(sys.version.split()[0])
PY
}

env_ready() {
  [[ -f "$1/bin/activate" ]] && [[ -x "$1/bin/python" ]] && python_supported "$1/bin/python"
}

if [[ -d "${ENV_DIR}" ]] && ! env_ready "${ENV_DIR}"; then
  old_version="missing"
  if [[ -x "${ENV_DIR}/bin/python" ]]; then
    old_version="$(python_version_string "${ENV_DIR}/bin/python")"
  fi
  ENV_DIR="${REQUESTED_ENV_DIR}_py311"
  echo "Existing env ${REQUESTED_ENV_DIR} is incomplete or unsupported (${old_version}); using ${ENV_DIR} instead." >&2
  if [[ -d "${ENV_DIR}" ]] && ! env_ready "${ENV_DIR}"; then
    ENV_DIR="${REQUESTED_ENV_DIR}_py311_$(date +%Y%m%d_%H%M%S)"
    echo "Fallback env is also incomplete or unsupported; using ${ENV_DIR} instead." >&2
  fi
fi

echo "A2UI repo: ${REPO_ROOT}"
echo "Target env: ${ENV_DIR}"
echo "Target model: ${QWEN_MODEL_ID}"
if [[ -n "${QWEN_MODEL_PATH}" ]]; then
  echo "Target model path: ${QWEN_MODEL_PATH}"
else
  echo "Target model path: not set; will ask for existing local model folder."
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found; continue only if this is intentional." >&2
fi

create_with_python311() {
  python3.11 -m venv "${ENV_DIR}"
  # shellcheck source=/dev/null
  source "${ENV_DIR}/bin/activate"
}

create_with_conda() {
  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck source=/dev/null
  source "${conda_base}/etc/profile.d/conda.sh"
  conda create -y -p "${ENV_DIR}" "python=${PYTHON_VERSION}" pip
  conda activate "${ENV_DIR}"
}

install_miniforge_and_create_env() {
  local installer="/tmp/miniforge_a2ui.sh"
  if [[ ! -x "${MINIFORGE_DIR}/bin/conda" ]]; then
    echo "Installing Miniforge to ${MINIFORGE_DIR}"
    local curl_ssl_args=()
    if [[ "${A2UI_DISABLE_SSL_VERIFY}" == "1" ]]; then
      curl_ssl_args=(-k)
    fi
    curl -L "${curl_ssl_args[@]}" -o "${installer}" \
      "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash "${installer}" -b -p "${MINIFORGE_DIR}"
  fi
  # shellcheck source=/dev/null
  source "${MINIFORGE_DIR}/etc/profile.d/conda.sh"
  conda create -y -p "${ENV_DIR}" "python=${PYTHON_VERSION}" pip
  conda activate "${ENV_DIR}"
}

if env_ready "${ENV_DIR}"; then
  # shellcheck source=/dev/null
  source "${ENV_DIR}/bin/activate"
else
  if command -v python3.11 >/dev/null 2>&1; then
    create_with_python311
  elif command -v conda >/dev/null 2>&1; then
    create_with_conda
  else
    install_miniforge_and_create_env
  fi
fi

python - <<'PY'
import sys
if sys.version_info < (3, 10) or sys.version_info >= (3, 13):
    raise SystemExit(
        f"vLLM setup requires Python 3.10-3.12; current is {sys.version.split()[0]}"
    )
print("python:", sys.version.split()[0])
PY

PIP_SSL_ARGS=()
UV_SSL_ARGS=()
if [[ "${A2UI_DISABLE_SSL_VERIFY}" == "1" ]]; then
  echo "WARNING: A2UI_DISABLE_SSL_VERIFY=1; TLS certificate verification is disabled for setup downloads." >&2
  export PYTHONHTTPSVERIFY=0
  export GIT_SSL_NO_VERIFY=true
  export CURL_SSL_BACKEND=openssl
  export PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org huggingface.co cdn-lfs.huggingface.co github.com objects.githubusercontent.com release-assets.githubusercontent.com download.pytorch.org download-r2.pytorch.org"
  export UV_INSECURE_HOST="${UV_INSECURE_HOST:-pypi.org files.pythonhosted.org huggingface.co cdn-lfs.huggingface.co download.pytorch.org download-r2.pytorch.org github.com objects.githubusercontent.com release-assets.githubusercontent.com}"
  PIP_SSL_ARGS=(
    --trusted-host pypi.org
    --trusted-host files.pythonhosted.org
    --trusted-host huggingface.co
    --trusted-host cdn-lfs.huggingface.co
    --trusted-host github.com
    --trusted-host objects.githubusercontent.com
    --trusted-host release-assets.githubusercontent.com
    --trusted-host download.pytorch.org
    --trusted-host download-r2.pytorch.org
  )
  UV_SSL_ARGS=(
    --allow-insecure-host pypi.org
    --allow-insecure-host files.pythonhosted.org
    --allow-insecure-host huggingface.co
    --allow-insecure-host cdn-lfs.huggingface.co
    --allow-insecure-host download.pytorch.org
    --allow-insecure-host download-r2.pytorch.org
    --allow-insecure-host github.com
    --allow-insecure-host objects.githubusercontent.com
    --allow-insecure-host release-assets.githubusercontent.com
  )
fi

python -m pip install "${PIP_SSL_ARGS[@]}" --upgrade pip setuptools wheel
if [[ "${A2UI_DISABLE_SSL_VERIFY}" == "1" ]]; then
  python -m pip config --site set global.trusted-host "${PIP_TRUSTED_HOST}" >/dev/null || true
fi
REQ_RUNTIME_FILE="$(mktemp)"
grep -Ev '^[[:space:]]*(vllm|torch|torchvision|torchaudio|transformers)([<>=!~ ].*)?$' "${REQ_FILE}" > "${REQ_RUNTIME_FILE}"
python -m pip install "${PIP_SSL_ARGS[@]}" -r "${REQ_RUNTIME_FILE}"
rm -f "${REQ_RUNTIME_FILE}"

install_transformers_register_guard() {
  python - <<'PY'
from pathlib import Path
import site
import textwrap

site_dirs = [Path(path) for path in site.getsitepackages() if path.endswith("site-packages")]
if not site_dirs:
    site_dirs = [Path(site.getusersitepackages())]
site_dir = next((path for path in site_dirs if path.exists()), site_dirs[0])
site_dir.mkdir(parents=True, exist_ok=True)

module_path = site_dir / "a2ui_transformers_register_guard.py"
pth_path = site_dir / "a2ui_transformers_register_guard.pth"

module_path.write_text(
    textwrap.dedent(
        """
        # Auto-loaded by a2ui_transformers_register_guard.pth.
        # vLLM 0.9.x registers a few AutoConfig entries that newer Transformers
        # can already include. Suppress only that duplicate-registration case so
        # newer Transformers can still recognize newer Qwen MoE checkpoints.
        try:
            import huggingface_hub as _a2ui_hf_hub

            # Some bleeding-edge Transformers branches import the private
            # _is_offline_mode symbol, while current huggingface_hub exports
            # the public is_offline_mode helper. Keep setup robust across both.
            if (
                not hasattr(_a2ui_hf_hub, "_is_offline_mode")
                and hasattr(_a2ui_hf_hub, "is_offline_mode")
            ):
                _a2ui_hf_hub._is_offline_mode = _a2ui_hf_hub.is_offline_mode
        except Exception:
            pass

        try:
            from transformers.models.auto.configuration_auto import AutoConfig
        except Exception:
            AutoConfig = None

        if AutoConfig is not None and not getattr(AutoConfig, "_a2ui_register_guard_installed", False):
            _orig_register = AutoConfig.register

            def _a2ui_register(cls, model_type, config, exist_ok=False):
                try:
                    return _orig_register(model_type, config, exist_ok=exist_ok)
                except ValueError as exc:
                    if "already used" in str(exc):
                        try:
                            return _orig_register(model_type, config, exist_ok=True)
                        except TypeError:
                            return None
                    raise
                except TypeError:
                    return _orig_register(model_type, config)

            AutoConfig.register = classmethod(_a2ui_register)
            AutoConfig._a2ui_register_guard_installed = True
        """
    ).strip()
    + "\n",
    encoding="utf-8",
)
pth_path.write_text("import a2ui_transformers_register_guard\n", encoding="utf-8")
print(f"Installed Transformers AutoConfig guard: {module_path}")
PY
}

download_transformers_source() {
  if [[ "${A2UI_DOWNLOAD_TRANSFORMERS_SOURCE}" != "1" ]]; then
    return 1
  fi
  echo "Transformers source not found locally; downloading from ${A2UI_TRANSFORMERS_SOURCE_URL}"
  export A2UI_TRANSFORMERS_SOURCE_DIR A2UI_TRANSFORMERS_SOURCE_URL A2UI_DISABLE_SSL_VERIFY
  python - <<'PY'
from pathlib import Path
import os
import shutil
import ssl
import tempfile
import urllib.request
import zipfile

source_url = os.environ["A2UI_TRANSFORMERS_SOURCE_URL"]
requested_dest = Path(os.environ["A2UI_TRANSFORMERS_SOURCE_DIR"]).expanduser()
fallback_dest = Path(str(requested_dest) + "_auto")

def is_transformers_root(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() or (path / "setup.py").is_file()

if is_transformers_root(requested_dest):
    print(f"Using existing Transformers source: {requested_dest}")
    raise SystemExit(0)

dest = requested_dest
if requested_dest.exists() and any(requested_dest.iterdir()):
    dest = fallback_dest

if dest.exists():
    shutil.rmtree(dest)
dest.parent.mkdir(parents=True, exist_ok=True)

ssl_context = None
if os.environ.get("A2UI_DISABLE_SSL_VERIFY") == "1":
    ssl_context = ssl._create_unverified_context()

with tempfile.TemporaryDirectory(prefix="a2ui_transformers_") as tmp_dir:
    tmp_dir_path = Path(tmp_dir)
    zip_path = tmp_dir_path / "transformers.zip"
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
    with opener.open(source_url, timeout=180) as response, zip_path.open("wb") as output:
        shutil.copyfileobj(response, output)

    extract_dir = tmp_dir_path / "extract"
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)

    roots = [path for path in extract_dir.iterdir() if path.is_dir()]
    root = next((path for path in roots if is_transformers_root(path)), None)
    if root is None:
        root = next((path for path in extract_dir.rglob("pyproject.toml")), None)
        root = root.parent if root else None
    if root is None or not is_transformers_root(root):
        raise SystemExit("Downloaded Transformers zip did not contain pyproject.toml/setup.py")

    shutil.move(str(root), str(dest))
    print(f"Downloaded Transformers source to: {dest}")
PY
}

resolve_transformers_source() {
  local candidate
  for candidate in \
    "${A2UI_TRANSFORMERS_SOURCE_DIR}" \
    "${A2UI_TRANSFORMERS_SOURCE_DIR}_auto" \
    "${HOME}/transformer" \
    "${HOME}/transformer_auto" \
    "${HOME}/transformers" \
    "${HOME}/transformers-main" \
    "${HOME}/transformer-main" \
    "${HOME}/transformers-main/transformers"; do
    if [[ -f "${candidate}/pyproject.toml" || -f "${candidate}/setup.py" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

install_transformers_stack() {
  local transformers_spec
  echo "Installing Transformers runtime dependencies: ${A2UI_HUGGINGFACE_HUB_SPEC}, ${A2UI_TOKENIZERS_SPEC}"
  python -m pip install "${PIP_SSL_ARGS[@]}" \
    --upgrade \
    "${A2UI_HUGGINGFACE_HUB_SPEC}" \
    "${A2UI_TOKENIZERS_SPEC}" \
    "safetensors>=0.6.0"

  if [[ "${A2UI_TRANSFORMERS_VERSION}" == "source" ]]; then
    if [[ -n "${A2UI_TRANSFORMERS_INSTALL_SPEC}" ]]; then
      transformers_spec="${A2UI_TRANSFORMERS_INSTALL_SPEC}"
    else
      transformers_spec="$(resolve_transformers_source || true)"
      if [[ -z "${transformers_spec}" ]]; then
        download_transformers_source
        transformers_spec="$(resolve_transformers_source || true)"
      fi
      if [[ -z "${transformers_spec:-}" ]]; then
        echo "Transformers source checkout not found." >&2
        echo "Download: ${A2UI_TRANSFORMERS_SOURCE_URL}" >&2
        echo "Extract it so pyproject.toml is at ${A2UI_TRANSFORMERS_SOURCE_DIR}/pyproject.toml, or set A2UI_TRANSFORMERS_INSTALL_SPEC=/path/to/transformers." >&2
        exit 1
      fi
    fi
  elif [[ -n "${A2UI_TRANSFORMERS_INSTALL_SPEC}" ]]; then
    transformers_spec="${A2UI_TRANSFORMERS_INSTALL_SPEC}"
  elif [[ "${A2UI_TRANSFORMERS_VERSION}" == "managed" ]]; then
    echo "Using vLLM-managed Transformers dependency."
    install_transformers_register_guard
    return
  elif [[ "${A2UI_TRANSFORMERS_VERSION}" == "latest" ]]; then
    transformers_spec="transformers"
  else
    transformers_spec="transformers==${A2UI_TRANSFORMERS_VERSION}"
  fi

  echo "Installing Transformers compatibility stack: ${transformers_spec}"
  python -m pip install "${PIP_SSL_ARGS[@]}" \
    --upgrade \
    --force-reinstall \
    --no-deps \
    "${transformers_spec}"
  install_transformers_register_guard
}

install_vllm_cuda_stack() {
  echo "Installing vLLM ${A2UI_VLLM_VERSION:-latest} with backend ${A2UI_VLLM_INSTALL_BACKEND}"
  # Remove CUDA 13 / mismatched packages from a reused environment.
  python -m pip freeze | awk -F== '/^(torch|torchvision|torchaudio|vllm|triton|nvidia-)/ {print $1}' \
    | xargs -r python -m pip uninstall -y

  local vllm_spec="vllm"
  if [[ -n "${A2UI_VLLM_VERSION}" ]]; then
    vllm_spec="vllm==${A2UI_VLLM_VERSION}"
  fi
  local constraint_args=()
  local binary_args=()
  local torch_spec=()
  local constraint_file=""
  if [[ "${A2UI_ONLY_BINARY}" == "1" ]]; then
    binary_args=(--only-binary :all:)
  fi
  if [[ -n "${A2UI_TORCH_VERSION}" && "${A2UI_TORCH_VERSION}" != "managed" ]]; then
    torch_spec=("torch==${A2UI_TORCH_VERSION}")
    constraint_file="$(mktemp)"
    echo "torch==${A2UI_TORCH_VERSION}" > "${constraint_file}"
    constraint_args=(-c "${constraint_file}")
    echo "Pinning torch to ${A2UI_TORCH_VERSION} to avoid CUDA 12.9 wheels on this CUDA 12.8 driver."
  fi
  local pytorch_index_args=()
  if [[ -n "${A2UI_PYTORCH_INDEX_URL}" ]]; then
    pytorch_index_args=(--extra-index-url "${A2UI_PYTORCH_INDEX_URL}")
  fi

  case "${A2UI_VLLM_INSTALL_BACKEND}" in
    uv-pypi)
      # Default for this cluster: avoid uv's PyTorch CDN backend because
      # download-r2.pytorch.org is blocked here. vLLM >=0.12 Linux wheels are
      # manylinux_2_31 and trigger source builds on this older cluster, so the
      # default pin uses the newest manylinux1-compatible vLLM wheel.
      python -m pip install "${PIP_SSL_ARGS[@]}" --upgrade uv
      if [[ ${#torch_spec[@]} -gt 0 ]]; then
        python -m uv pip install \
          --python "$(command -v python)" \
          "${UV_SSL_ARGS[@]}" \
          "${binary_args[@]}" \
          --upgrade \
          "${torch_spec[@]}"
      fi
      python -m uv pip install \
        --python "$(command -v python)" \
        "${UV_SSL_ARGS[@]}" \
        "${binary_args[@]}" \
        --upgrade \
        "${constraint_args[@]}" \
        "${vllm_spec}"
      ;;
    uv-auto)
      # vLLM's uv installer chooses a compatible torch backend for the local
      # CUDA/driver stack. This is required for newer Qwen MoE checkpoints
      # such as qwen3_5_moe; old vLLM 0.9.x does not know those configs.
      # Use only if the network can reach download-r2.pytorch.org.
      python -m pip install "${PIP_SSL_ARGS[@]}" --upgrade uv
      python -m uv pip install \
        --python "$(command -v python)" \
        "${UV_SSL_ARGS[@]}" \
        "${binary_args[@]}" \
        --upgrade \
        "${vllm_spec}" \
        --torch-backend=auto
      ;;
    pip)
      python -m pip install "${PIP_SSL_ARGS[@]}" \
        "${pytorch_index_args[@]}" \
        "${binary_args[@]}" \
        --force-reinstall \
        "${constraint_args[@]}" \
        "${vllm_spec}"
      ;;
    *)
      echo "Unsupported A2UI_VLLM_INSTALL_BACKEND=${A2UI_VLLM_INSTALL_BACKEND}; use uv-pypi, uv-auto, or pip." >&2
      exit 1
      ;;
  esac
  if [[ -n "${constraint_file}" ]]; then
    rm -f "${constraint_file}"
  fi
  install_transformers_stack
}

install_vllm_cuda_stack

if [[ -z "${A2UI_CA_BUNDLE}" ]]; then
  for candidate in \
    /etc/ssl/certs/ca-certificates.crt \
    /etc/pki/tls/certs/ca-bundle.crt \
    /etc/ssl/cert.pem; do
    if [[ -f "${candidate}" ]]; then
      A2UI_CA_BUNDLE="${candidate}"
      break
    fi
  done
fi
if [[ "${A2UI_DISABLE_SSL_VERIFY}" == "1" ]]; then
  echo "Skipping CA bundle setup because SSL verification is disabled." >&2
elif [[ -n "${A2UI_CA_BUNDLE}" && -f "${A2UI_CA_BUNDLE}" ]]; then
  export SSL_CERT_FILE="${A2UI_CA_BUNDLE}"
  export REQUESTS_CA_BUNDLE="${A2UI_CA_BUNDLE}"
  export CURL_CA_BUNDLE="${A2UI_CA_BUNDLE}"
  export GIT_SSL_CAINFO="${A2UI_CA_BUNDLE}"
  echo "Using CA bundle: ${A2UI_CA_BUNDLE}"
else
  echo "No CA bundle detected. If SSL fails, set A2UI_CA_BUNDLE=/path/to/company-root-ca.pem" >&2
fi

if [[ -z "${QWEN_MODEL_PATH}" ]]; then
  if [[ -t 0 ]]; then
    read -r -p "Enter existing local Qwen model folder path: " QWEN_MODEL_PATH
  else
    echo "QWEN_MODEL_PATH is required because model download is disabled by default." >&2
    echo "Example: QWEN_MODEL_PATH=/path/to/Qwen3.6-35B-A3B bash dataset/scripts/setup_qwen_vllm_online_2gpu.sh" >&2
    exit 1
  fi
fi

if [[ ! -d "${QWEN_MODEL_PATH}" ]]; then
  if [[ "${A2UI_DOWNLOAD_QWEN_MODEL}" != "1" ]]; then
    echo "Model folder does not exist: ${QWEN_MODEL_PATH}" >&2
    echo "Set QWEN_MODEL_PATH to the already-downloaded model folder." >&2
    echo "If you really want this script to download from Hugging Face, rerun with A2UI_DOWNLOAD_QWEN_MODEL=1." >&2
    exit 1
  fi
  mkdir -p "$(dirname "${QWEN_MODEL_PATH}")"
  export QWEN_MODEL_ID QWEN_MODEL_PATH
  python - <<'PY'
import os
if os.environ.get("A2UI_DISABLE_SSL_VERIFY") == "1":
    import requests
    import urllib3
    from huggingface_hub import configure_http_backend

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def backend_factory():
        session = requests.Session()
        session.verify = False
        return session

    configure_http_backend(backend_factory=backend_factory)
else:
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception as exc:
        print(f"truststore injection skipped: {exc}")
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["QWEN_MODEL_ID"],
    local_dir=os.environ["QWEN_MODEL_PATH"],
    resume_download=True,
)
PY
else
  echo "Using existing local model folder: ${QWEN_MODEL_PATH}"
fi

export QWEN_MODEL_PATH
python - <<'PY'
import os
from transformers import AutoConfig

model_path = os.environ["QWEN_MODEL_PATH"]
try:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
except Exception as exc:
    print(
        "WARNING: raw Transformers AutoConfig could not load the local Qwen config. "
        "This is expected for some new Qwen MoE checkpoints when vLLM provides "
        "the model config internally. Continue to start vLLM; if vLLM still fails, "
        "rerun setup with A2UI_VLLM_INSTALL_BACKEND=uv-auto and the latest vLLM. "
        f"Original error: {exc}",
        flush=True,
    )
else:
    print("model config type:", getattr(config, "model_type", "unknown"))
PY

cat > "${ENV_DIR}/activate_qwen_vllm.sh" <<EOF
#!/usr/bin/env bash
_a2ui_qwen_env="${ENV_DIR}"
_a2ui_qwen_activate="\${_a2ui_qwen_env}/bin/activate"
_a2ui_qwen_shim_marker="\${_a2ui_qwen_env}/bin/.a2ui_activate_shim"
if [[ -f "\${_a2ui_qwen_activate}" && ! -f "\${_a2ui_qwen_shim_marker}" ]]; then
  source "\${_a2ui_qwen_activate}"
else
  export PATH="\${_a2ui_qwen_env}/bin:\${PATH}"
  export VIRTUAL_ENV="\${_a2ui_qwen_env}"
fi
export CONDA_PREFIX="\${_a2ui_qwen_env}"
export CONDA_DEFAULT_ENV="\$(basename "\${_a2ui_qwen_env}")"
export CONDA_PROMPT_MODIFIER="(\${CONDA_DEFAULT_ENV}) "
if [[ -n "\${PS1:-}" ]]; then
  if [[ -z "\${_A2UI_QWEN_OLD_PS1:-}" ]]; then
    export _A2UI_QWEN_OLD_PS1="\${PS1}"
  fi
  _a2ui_qwen_prompt="\${PS1}"
  _a2ui_qwen_prompt="\${_a2ui_qwen_prompt#*) }"
  PS1="(\${CONDA_DEFAULT_ENV}) \${_a2ui_qwen_prompt}"
  export PS1
fi
export QWEN_MODEL_PATH="${QWEN_MODEL_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export A2UI_VLLM_GPUS="${A2UI_VLLM_GPUS}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION}"
export VLLM_DTYPE="${VLLM_DTYPE}"
export LOCAL_ALLOW_HTTP_ENDPOINT=1
export LOCAL_STRICT_OFFLINE=0
export LOCAL_VLLM_ENABLE_THINKING=1
export VLLM_REASONING_PARSER=qwen3
EOF
chmod +x "${ENV_DIR}/activate_qwen_vllm.sh"
mkdir -p "${ENV_DIR}/bin"
if [[ ! -f "${ENV_DIR}/bin/activate" ]]; then
  cat > "${ENV_DIR}/bin/activate" <<EOF
#!/usr/bin/env bash
source "${ENV_DIR}/activate_qwen_vllm.sh"
EOF
  touch "${ENV_DIR}/bin/.a2ui_activate_shim"
  chmod +x "${ENV_DIR}/bin/activate"
fi
cat > "${ENV_DIR}/bin/activate_qwen_vllm" <<EOF
#!/usr/bin/env bash
source "${ENV_DIR}/activate_qwen_vllm.sh"
EOF
chmod +x "${ENV_DIR}/bin/activate_qwen_vllm"

python - <<'PY'
import importlib.metadata as md
import torch

for package in ["torch", "vllm", "transformers", "huggingface_hub"]:
    print(f"{package}:", md.version(package))
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
for idx in range(torch.cuda.device_count()):
    print(f"gpu {idx}:", torch.cuda.get_device_name(idx))
PY

echo
echo "Setup complete."
echo "Activate by running one of these exact commands, without a trailing colon:"
echo "  source ${ENV_DIR}/activate_qwen_vllm.sh"
echo "  source ${ENV_DIR}/bin/activate_qwen_vllm"
echo
echo "Start vLLM:"
echo "  bash dataset/scripts/run_qwen36_vllm_server.sh"
echo
echo "Run stages:"
echo "  RUN_ID=dataset_qwen36_vllm_reasoning_v0 bash dataset/scripts/run_qwen36_dataset_stages.sh"
