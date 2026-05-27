#!/usr/bin/env bash
set -euo pipefail

# Run this on the offline GPU machine after copying qwen_vllm_offline_bundle.

BUNDLE_DIR="${1:-${BUNDLE_DIR:-${PWD}/qwen_vllm_offline_bundle}}"
ENV_DIR="${ENV_DIR:-${PWD}/qwen_vllm_env}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
REQ_FILE="${BUNDLE_DIR}/requirements-qwen-vllm.txt"
WHEELHOUSE="${BUNDLE_DIR}/wheelhouse"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${BUNDLE_DIR}/models/Qwen--Qwen3.6-35B-A3B}"

if [[ ! -d "${BUNDLE_DIR}" ]]; then
  echo "Bundle folder not found: ${BUNDLE_DIR}" >&2
  exit 1
fi
if [[ ! -d "${WHEELHOUSE}" ]]; then
  echo "Wheelhouse not found: ${WHEELHOUSE}" >&2
  exit 1
fi
if [[ ! -f "${REQ_FILE}" ]]; then
  echo "Requirements file not found: ${REQ_FILE}" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info < (3, 10) or sys.version_info >= (3, 13):
    raise SystemExit(
        f"Use Python 3.10-3.12 for vLLM. Python {sys.version.split()[0]} is not supported here."
    )
print("python:", sys.version.split()[0])
PY

"${PYTHON_BIN}" -m venv "${ENV_DIR}"
# shellcheck source=/dev/null
source "${ENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel --no-index --find-links "${WHEELHOUSE}" || true
python -m pip install --no-index --find-links "${WHEELHOUSE}" -r "${REQ_FILE}"

cat > "${ENV_DIR}/activate_qwen_vllm.sh" <<EOF
#!/usr/bin/env bash
source "${ENV_DIR}/bin/activate"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export LOCAL_ALLOW_HTTP_ENDPOINT=1
export LOCAL_STRICT_OFFLINE=0
export LOCAL_VLLM_ENABLE_THINKING=1
export VLLM_REASONING_PARSER=qwen3
export QWEN_MODEL_PATH="${QWEN_MODEL_PATH}"
EOF
chmod +x "${ENV_DIR}/activate_qwen_vllm.sh"

python - <<'PY'
import importlib.metadata as md
import sys

print("python:", sys.version.split()[0])
for package in ["torch", "vllm", "transformers", "huggingface_hub"]:
    try:
        print(f"{package}:", md.version(package))
    except Exception as exc:
        print(f"{package}: import/version failed: {exc}")

try:
    import torch
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())
    for idx in range(torch.cuda.device_count()):
        print(f"gpu {idx}:", torch.cuda.get_device_name(idx))
except Exception as exc:
    print("torch cuda check failed:", exc)
PY

echo "Offline env ready: ${ENV_DIR}"
echo "Activate with: source ${ENV_DIR}/activate_qwen_vllm.sh"
