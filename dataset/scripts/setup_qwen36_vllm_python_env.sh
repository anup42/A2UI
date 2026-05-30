#!/usr/bin/env bash
set -euo pipefail

# Setup a Python environment for Qwen3.6-35B-A3B with a vLLM version that
# supports Qwen3.6 directly. This intentionally does not install A2UI runtime
# compatibility wrappers/shims for the model config.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
ENV_DIR="${ENV_DIR:-${REPO_ROOT}/qwen36_vllm_env}"
REQ_FILE="${REQ_FILE:-${REPO_ROOT}/dataset/requirements-qwen-vllm.txt}"
A2UI_VLLM_SPEC="${A2UI_VLLM_SPEC:-vllm>=0.17.0}"
A2UI_DISABLE_SSL_VERIFY="${A2UI_DISABLE_SSL_VERIFY:-1}"
A2UI_USE_UV="${A2UI_USE_UV:-1}"
QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3.6-35B-A3B}"
MODEL_ROOT="${MODEL_ROOT:-${LOCAL_MODEL_ROOT:-${A2UI_MODEL_ROOT:-${QWEN_MODEL_ROOT:-}}}}"
QWEN_MODEL_ROOT="${QWEN_MODEL_ROOT:-${MODEL_ROOT}}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-}"

if [[ "${A2UI_DISABLE_SSL_VERIFY}" = "1" ]]; then
  echo "WARNING: A2UI_DISABLE_SSL_VERIFY=1; TLS verification is disabled for setup downloads." >&2
  export CURL_CA_BUNDLE=""
  export REQUESTS_CA_BUNDLE=""
  export SSL_CERT_FILE=""
  export PYTHONHTTPSVERIFY=0
  export GIT_SSL_NO_VERIFY=1
  export HF_HUB_DISABLE_SSL_VERIFICATION=1
  export PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org huggingface.co cdn-lfs.huggingface.co github.com objects.githubusercontent.com release-assets.githubusercontent.com download.pytorch.org download-r2.pytorch.org"
  export UV_INSECURE_HOST="${UV_INSECURE_HOST:-pypi.org files.pythonhosted.org huggingface.co cdn-lfs.huggingface.co github.com objects.githubusercontent.com release-assets.githubusercontent.com download.pytorch.org download-r2.pytorch.org}"
fi

PIP_SSL_ARGS=()
UV_SSL_ARGS=()
if [[ "${A2UI_DISABLE_SSL_VERIFY}" = "1" ]]; then
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
    --allow-insecure-host github.com
    --allow-insecure-host objects.githubusercontent.com
    --allow-insecure-host release-assets.githubusercontent.com
    --allow-insecure-host download.pytorch.org
    --allow-insecure-host download-r2.pytorch.org
  )
fi

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python binary not found: ${PYTHON_BIN}. Set PYTHON_BIN to Python 3.10-3.12." >&2
    exit 1
  fi
  "${PYTHON_BIN}" -m venv "${ENV_DIR}"
fi

# shellcheck source=/dev/null
source "${ENV_DIR}/bin/activate"

python - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(f"vLLM requires Python 3.10-3.12 here; current={sys.version.split()[0]}")
print("python:", sys.version.split()[0])
PY

python -m pip install "${PIP_SSL_ARGS[@]}" --upgrade pip setuptools wheel
if [[ "${A2UI_DISABLE_SSL_VERIFY}" = "1" ]]; then
  python -m pip config --site set global.trusted-host "${PIP_TRUSTED_HOST}" >/dev/null || true
fi

if [[ "${A2UI_USE_UV}" = "1" ]]; then
  python -m pip install "${PIP_SSL_ARGS[@]}" --upgrade uv
fi

REQ_RUNTIME_FILE="$(mktemp)"
grep -Ev '^[[:space:]]*(vllm|torch|torchvision|torchaudio|transformers|xformers|flashinfer)([<>=!~ ].*)?$' "${REQ_FILE}" > "${REQ_RUNTIME_FILE}"
python -m pip install "${PIP_SSL_ARGS[@]}" -r "${REQ_RUNTIME_FILE}"
rm -f "${REQ_RUNTIME_FILE}"

if [[ "${A2UI_USE_UV}" = "1" ]] && command -v uv >/dev/null 2>&1; then
  uv pip install "${UV_SSL_ARGS[@]}" --torch-backend=auto --upgrade "${A2UI_VLLM_SPEC}"
else
  python -m pip install "${PIP_SSL_ARGS[@]}" --upgrade "${A2UI_VLLM_SPEC}"
fi

ACTIVATE_FILE="${ENV_DIR}/activate_qwen36_vllm.sh"
cat > "${ACTIVATE_FILE}" <<EOF
#!/usr/bin/env bash
source "${ENV_DIR}/bin/activate"
export QWEN_MODEL_ID="\${QWEN_MODEL_ID:-${QWEN_MODEL_ID}}"
export MODEL_ROOT="\${MODEL_ROOT:-${MODEL_ROOT}}"
export LOCAL_MODEL_ROOT="\${LOCAL_MODEL_ROOT:-\${MODEL_ROOT}}"
export A2UI_MODEL_ROOT="\${A2UI_MODEL_ROOT:-\${MODEL_ROOT}}"
export QWEN_MODEL_ROOT="\${QWEN_MODEL_ROOT:-${QWEN_MODEL_ROOT}}"
export QWEN_MODEL_PATH="\${QWEN_MODEL_PATH:-${QWEN_MODEL_PATH}}"
if [[ -z "\${QWEN_MODEL_PATH}" && -n "\${QWEN_MODEL_ROOT}" ]]; then
  resolved="\$(python "${REPO_ROOT}/dataset/scripts/resolve_model_from_root.py" --model "\${QWEN_MODEL_ID}" --root "\${QWEN_MODEL_ROOT}" --quiet 2>/dev/null || true)"
  if [[ -n "\${resolved}" ]]; then
    export QWEN_MODEL_PATH="\${resolved}"
  fi
fi
export LOCAL_ALLOW_HTTP_ENDPOINT="\${LOCAL_ALLOW_HTTP_ENDPOINT:-1}"
export LOCAL_STRICT_OFFLINE="\${LOCAL_STRICT_OFFLINE:-0}"
export LOCAL_VLLM_STRIP_THINKING="\${LOCAL_VLLM_STRIP_THINKING:-1}"
export LOCAL_VLLM_USE_HF_GENERATION_CONFIG="\${LOCAL_VLLM_USE_HF_GENERATION_CONFIG:-1}"
export VLLM_GENERATION_CONFIG="\${VLLM_GENERATION_CONFIG:-auto}"
export VLLM_REASONING_PARSER="\${VLLM_REASONING_PARSER:-qwen3}"
export LOCAL_VLLM_TIMEOUT_SECONDS="\${LOCAL_VLLM_TIMEOUT_SECONDS:-600}"
export LOCAL_VLLM_MAX_OUTPUT_TOKENS="\${LOCAL_VLLM_MAX_OUTPUT_TOKENS:-8192}"
export LOCAL_STAGE3_PROMPT_MAX_TOKENS="\${LOCAL_STAGE3_PROMPT_MAX_TOKENS:-8192}"
export A2UI_QUERY_MAX_TOKENS="\${A2UI_QUERY_MAX_TOKENS:-8192}"
export A2UI_RESPONSE_MAX_TOKENS="\${A2UI_RESPONSE_MAX_TOKENS:-8192}"
export A2UI_GENUI_MAX_TOKENS="\${A2UI_GENUI_MAX_TOKENS:-8192}"
export A2UI_GENUI_PROMPT_MAX_TOKENS="\${A2UI_GENUI_PROMPT_MAX_TOKENS:-8192}"
export A2UI_DISABLE_SSL_VERIFY="\${A2UI_DISABLE_SSL_VERIFY:-1}"
if [[ "\${A2UI_DISABLE_SSL_VERIFY}" = "1" ]]; then
  export CURL_CA_BUNDLE=""
  export REQUESTS_CA_BUNDLE=""
  export SSL_CERT_FILE=""
  export PYTHONHTTPSVERIFY=0
  export GIT_SSL_NO_VERIFY=1
  export HF_HUB_DISABLE_SSL_VERIFICATION=1
fi
EOF
chmod +x "${ACTIVATE_FILE}"
# Backward-compatible activation filename; this is not a model wrapper.
cp "${ACTIVATE_FILE}" "${ENV_DIR}/activate_qwen_vllm.sh"

python - <<'PY'
import importlib.metadata as md
import subprocess
import sys

print("vllm:", md.version("vllm"))
help_text = subprocess.run(["vllm", "serve", "--help"], text=True, capture_output=True).stdout
for flag in ("--enable-reasoning", "--reasoning-parser", "--generation-config"):
    print(f"{flag}:", "yes" if flag in help_text else "no")
PY

if [[ -n "${QWEN_MODEL_PATH}" && ! -f "${QWEN_MODEL_PATH}/config.json" ]]; then
  echo "Warning: QWEN_MODEL_PATH does not contain config.json: ${QWEN_MODEL_PATH}" >&2
elif [[ -z "${QWEN_MODEL_PATH}" && -n "${QWEN_MODEL_ROOT}" ]]; then
  resolved="$(python "${REPO_ROOT}/dataset/scripts/resolve_model_from_root.py" --model "${QWEN_MODEL_ID}" --root "${QWEN_MODEL_ROOT}" --quiet 2>/dev/null || true)"
  if [[ -n "${resolved}" ]]; then
    echo "Resolved Qwen model: ${resolved}"
  else
    echo "Warning: Qwen model not resolved from QWEN_MODEL_ROOT=${QWEN_MODEL_ROOT}" >&2
  fi
fi

echo "Setup complete."
echo "Activate with: source ${ACTIVATE_FILE}"
echo "Start server: MODEL_ROOT=/path/to/models QWEN36_ENABLE_REASONING=1 bash dataset/scripts/run_qwen36_vllm_python.sh"
