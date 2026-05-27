#!/usr/bin/env bash
set -euo pipefail

# Run this on an internet-connected Linux machine with the same Python minor
# version and platform as the offline GPU machine.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3.6-35B-A3B}"
BUNDLE_DIR="${BUNDLE_DIR:-${PWD}/qwen_vllm_offline_bundle}"
REQ_FILE="${REQ_FILE:-${REPO_ROOT}/dataset/requirements-qwen-vllm.txt}"
MODEL_DIR="${MODEL_DIR:-${BUNDLE_DIR}/models/${QWEN_MODEL_ID//\//--}}"
WHEELHOUSE="${BUNDLE_DIR}/wheelhouse"
DOWNLOAD_VENV="${BUNDLE_DIR}/.download_venv"
export BUNDLE_DIR QWEN_MODEL_ID MODEL_DIR
A2UI_CA_BUNDLE="${A2UI_CA_BUNDLE:-}"
A2UI_DISABLE_SSL_VERIFY="${A2UI_DISABLE_SSL_VERIFY:-1}"
export A2UI_DISABLE_SSL_VERIFY

mkdir -p "${BUNDLE_DIR}" "${WHEELHOUSE}" "$(dirname "${MODEL_DIR}")"
cp "${REQ_FILE}" "${BUNDLE_DIR}/requirements-qwen-vllm.txt"

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info < (3, 10) or sys.version_info >= (3, 13):
    raise SystemExit(
        f"Use Python 3.10-3.12 for vLLM wheels, not {sys.version.split()[0]}"
    )
print("python:", sys.version.split()[0])
PY

"${PYTHON_BIN}" -m venv "${DOWNLOAD_VENV}"
# shellcheck source=/dev/null
source "${DOWNLOAD_VENV}/bin/activate"
PIP_SSL_ARGS=()
if [[ "${A2UI_DISABLE_SSL_VERIFY}" == "1" ]]; then
  echo "WARNING: A2UI_DISABLE_SSL_VERIFY=1; TLS certificate verification is disabled for setup downloads." >&2
  export PYTHONHTTPSVERIFY=0
  PIP_SSL_ARGS=(
    --trusted-host pypi.org
    --trusted-host files.pythonhosted.org
    --trusted-host huggingface.co
    --trusted-host cdn-lfs.huggingface.co
  )
fi
python -m pip install "${PIP_SSL_ARGS[@]}" --upgrade pip setuptools wheel
python -m pip install "${PIP_SSL_ARGS[@]}" "huggingface_hub[cli]>=0.25.0" truststore certifi

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

echo "Downloading wheels to ${WHEELHOUSE}"
# PIP_DOWNLOAD_EXTRA_ARGS can be used for custom CUDA/PyTorch indexes.
python -m pip download "${PIP_SSL_ARGS[@]}" --dest "${WHEELHOUSE}" pip setuptools wheel ${PIP_DOWNLOAD_EXTRA_ARGS:-}
python -m pip download \
  "${PIP_SSL_ARGS[@]}" \
  --dest "${WHEELHOUSE}" \
  --only-binary=:all: \
  -r "${BUNDLE_DIR}/requirements-qwen-vllm.txt" \
  ${PIP_DOWNLOAD_EXTRA_ARGS:-}

echo "Downloading model ${QWEN_MODEL_ID} to ${MODEL_DIR}"
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
    local_dir=os.environ["MODEL_DIR"],
    resume_download=True,
)
PY

cat > "${BUNDLE_DIR}/README.txt" <<EOF
Qwen vLLM offline bundle

Model: ${QWEN_MODEL_ID}
Model dir: models/${QWEN_MODEL_ID//\//--}
Requirements: requirements-qwen-vllm.txt
Wheelhouse: wheelhouse/

Copy this whole folder to the offline GPU machine, then run:
  bash dataset/scripts/install_qwen_vllm_offline_env.sh /path/to/qwen_vllm_offline_bundle
EOF

python - <<'PY'
import json, os, platform, sys
from pathlib import Path

bundle = Path(os.environ.get("BUNDLE_DIR", "qwen_vllm_offline_bundle")).resolve()
manifest = {
    "model_id": os.environ.get("QWEN_MODEL_ID"),
    "model_dir": os.environ.get("MODEL_DIR"),
    "python_version": sys.version.split()[0],
    "platform": platform.platform(),
    "wheelhouse": str((bundle / "wheelhouse").resolve()),
    "requirements": str((bundle / "requirements-qwen-vllm.txt").resolve()),
}
(bundle / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
PY

echo "Bundle complete: ${BUNDLE_DIR}"
