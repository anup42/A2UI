#!/usr/bin/env bash
set -euo pipefail

# Print a focused diagnostics report for a Gemma4 vLLM SIF.
# Usage:
#   VLLM_SIF=/path/to/a2ui-vllm-cu128-source_gemma4_speculative.sif \
#   bash dataset/scripts/debug_gemma4_vllm_sif_report.sh

VLLM_SIF="${VLLM_SIF:-${1:-}}"

if [[ -z "${VLLM_SIF}" ]]; then
  echo "Usage: VLLM_SIF=/path/to/file.sif bash dataset/scripts/debug_gemma4_vllm_sif_report.sh" >&2
  echo "   or: bash dataset/scripts/debug_gemma4_vllm_sif_report.sh /path/to/file.sif" >&2
  exit 2
fi

if [[ ! -f "${VLLM_SIF}" ]]; then
  echo "SIF not found: ${VLLM_SIF}" >&2
  exit 1
fi

if ! command -v apptainer >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
  module load apptainer >/dev/null 2>&1 || true
fi

if command -v apptainer >/dev/null 2>&1; then
  RUNTIME="apptainer"
elif command -v singularity >/dev/null 2>&1; then
  RUNTIME="singularity"
else
  echo "apptainer or singularity is required. Try: module load apptainer" >&2
  exit 1
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | paste -sd, - || true)"
  export CUDA_VISIBLE_DEVICES
fi

RUNTIME_ARGS=(exec --nv --cleanenv)
RUNTIME_ARGS+=(--env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}")
RUNTIME_ARGS+=(--env "FLASHINFER_DISABLE_VERSION_CHECK=${FLASHINFER_DISABLE_VERSION_CHECK:-1}")
RUNTIME_ARGS+=(--env "FLASHINFER_DISABLE_VERSION__CHECK=${FLASHINFER_DISABLE_VERSION__CHECK:-1}")
RUNTIME_ARGS+=(--env "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}")
RUNTIME_ARGS+=(--env "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}")
RUNTIME_ARGS+=(--env "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}")

run_in_sif() {
  "${RUNTIME}" "${RUNTIME_ARGS[@]}" "${VLLM_SIF}" "$@"
}

section() {
  printf '\n===== %s =====\n' "$1"
}

section "Host"
echo "date: $(date -Is)"
echo "runtime: ${RUNTIME}"
echo "sif: ${VLLM_SIF}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
fi

section "vLLM Version And Path"
run_in_sif python - <<'PY' || true
import inspect
import os
import sys

print("python:", sys.version.replace("\n", " "))
print("executable:", sys.executable)
try:
    import vllm
except Exception as exc:
    print(f"vllm import failed: {exc!r}")
else:
    print("vllm version:", getattr(vllm, "__version__", "unknown"))
    print("vllm path:", inspect.getfile(vllm))
    root = os.path.dirname(inspect.getfile(vllm))
    print("vllm package root:", root)
PY

section "vLLM CLI Version"
run_in_sif bash -lc 'vllm --version 2>&1 || true'

section "vLLM Serve Help Flag Check"
HELP_TEXT="$(run_in_sif bash -lc 'vllm serve --help 2>&1' || true)"
for flag in \
  "--speculative-config" \
  "--spec-model" \
  "--spec-tokens" \
  "--spec-method" \
  "--speculative-model" \
  "--num-speculative-tokens" \
  "--reasoning-parser" \
  "--enable-reasoning" \
  "--default-chat-template-kwargs"; do
  if grep -q -- "${flag}" <<<"${HELP_TEXT}"; then
    echo "present: ${flag}"
  else
    echo "missing: ${flag}"
  fi
done

section "Speculative/Reasoning Help Lines"
grep -Ei 'speculative|spec-|draft|mtp|eagle|medusa|reasoning|thinking' <<<"${HELP_TEXT}" | head -240 || true

section "Parser-Derived Arg Check"
run_in_sif python - <<'PY' || true
import argparse

checks = [
    "--speculative-config",
    "--spec-model",
    "--spec-tokens",
    "--spec-method",
    "--speculative-model",
    "--num-speculative-tokens",
    "--reasoning-parser",
    "--enable-reasoning",
]

try:
    from vllm.engine.arg_utils import AsyncEngineArgs
except Exception as exc:
    print(f"could not import AsyncEngineArgs: {exc!r}")
else:
    parser = argparse.ArgumentParser()
    AsyncEngineArgs.add_cli_args(parser)
    help_text = parser.format_help()
    for flag in checks:
        print(f"{flag}: {'present' if flag in help_text else 'missing'}")
    print("--- matching parser lines ---")
    for line in help_text.splitlines():
        if any(token in line.lower() for token in ("speculative", "spec-", "draft", "mtp", "eagle", "medusa", "reasoning", "thinking")):
            print(line)
PY

section "Source Code Feature Check"
run_in_sif python - <<'PY' || true
import importlib.util
import inspect
import os
import pathlib

try:
    import vllm
except Exception as exc:
    print(f"vllm import failed: {exc!r}")
    raise SystemExit(0)

root = pathlib.Path(inspect.getfile(vllm)).resolve().parent
repo = root.parent
print("vllm root:", root)
print("candidate repo root:", repo)

files = {
    "arg_utils": repo / "vllm" / "engine" / "arg_utils.py",
    "speculative": repo / "vllm" / "config" / "speculative.py",
    "gemma4_mtp": repo / "vllm" / "model_executor" / "models" / "gemma4_mtp.py",
    "gpu_model_runner": repo / "vllm" / "v1" / "worker" / "gpu_model_runner.py",
    "reasoning_init": repo / "vllm" / "reasoning" / "__init__.py",
    "gemma4_reasoning_parser": repo / "vllm" / "reasoning" / "gemma4_reasoning_parser.py",
}

needles = {
    "arg_utils": ["--speculative-config", "--reasoning-parser"],
    "speculative": ["gemma4_assistant", "gemma4_mtp", "use_gemma4_mtp", 'method == "mtp"'],
    "gemma4_mtp": ["Gemma4MTP"],
    "gpu_model_runner": ["Gemma4Proposer", "use_gemma4_mtp"],
    "reasoning_init": ['"gemma4"', "gemma4_reasoning_parser"],
    "gemma4_reasoning_parser": ["Gemma4ReasoningParser"],
}

for name, path in files.items():
    print(f"{name}: {path} exists={path.exists()}")
    if not path.exists():
        continue
    text = path.read_text(errors="replace")
    for needle in needles.get(name, []):
        print(f"  {needle}: {'present' if needle in text else 'missing'}")
PY

section "Git Metadata Inside Container"
run_in_sif bash -lc '
python - <<PY
import inspect, pathlib
try:
    import vllm
except Exception as exc:
    print("vllm import failed:", repr(exc))
    raise SystemExit(0)
repo = pathlib.Path(inspect.getfile(vllm)).resolve().parent.parent
print(repo)
PY
' | while IFS= read -r candidate; do
  [[ -n "${candidate}" ]] || continue
  run_in_sif bash -lc "cd '${candidate}' 2>/dev/null && git rev-parse HEAD 2>/dev/null && git show -s --format='date=%cI subject=%s' HEAD 2>/dev/null || true"
done

section "Recommended Gemma4 MTP Command Shape"
cat <<'EOF'
Expected current-source Gemma4 MTP flag:
  --speculative-config '{"method":"mtp","model":"/path/to/gemma-4-31b-it-assistant","num_speculative_tokens":1}'

Expected reasoning flag:
  --reasoning-parser gemma4

If --speculative-config is missing, the SIF is not using the expected vLLM source build even if another vLLM package is importable.
EOF
