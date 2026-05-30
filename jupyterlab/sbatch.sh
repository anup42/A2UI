#!/usr/bin/env bash
#SBATCH --job-name=singularity_rag
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=400:00:00
#SBATCH --output=singularity_output.log
#SBATCH --error=singularity_error.log

set -euo pipefail

# Submit with:
#   sbatch jupyterlab/sbatch.sh
# Optional overrides:
#   A2UI_SIF=/path/to/a2ui-vllm-cu130-source_gemma4_speculative_ada.sif \
#   A2UI_REPO_DIR=/home/k_anup/code/A2UI \
#   A2UI_BIND_MOUNTS=/Storage_gpu:/home/k_anup/Storage_gpu \
#   sbatch jupyterlab/sbatch.sh

A2UI_REPO_DIR="${A2UI_REPO_DIR:-${SLURM_SUBMIT_DIR:-${PWD}}}"
A2UI_SIF="${A2UI_SIF:-${HOME}/containers/a2ui-vllm-cu130-source_gemma4_speculative_ada.sif}"
A2UI_BIND_MOUNTS="${A2UI_BIND_MOUNTS:-/Storage_gpu:/home/k_anup/Storage_gpu,${A2UI_REPO_DIR}:${A2UI_REPO_DIR}}"
JUPYTER_SCRIPT="${JUPYTER_SCRIPT:-${A2UI_REPO_DIR}/jupyterlab/jupyter.sh}"
JUPYTER_PORT="${JUPYTER_PORT:-8811}"
JUPYTER_TOKEN="${JUPYTER_TOKEN:-}"

if ! command -v singularity >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
  module load singularity 2>/dev/null || module load apptainer 2>/dev/null || true
fi

if command -v singularity >/dev/null 2>&1; then
  RUNTIME="singularity"
elif command -v apptainer >/dev/null 2>&1; then
  RUNTIME="apptainer"
else
  echo "singularity or apptainer command not found." >&2
  exit 1
fi

if [[ ! -f "${A2UI_SIF}" ]]; then
  echo "SIF not found: ${A2UI_SIF}" >&2
  exit 1
fi
if [[ ! -f "${JUPYTER_SCRIPT}" ]]; then
  echo "Jupyter script not found: ${JUPYTER_SCRIPT}" >&2
  exit 1
fi

echo "Runtime: ${RUNTIME}"
echo "SIF: ${A2UI_SIF}"
echo "Repo: ${A2UI_REPO_DIR}"
echo "Bind mounts: ${A2UI_BIND_MOUNTS}"
echo "Jupyter script: ${JUPYTER_SCRIPT}"
echo "Jupyter port: ${JUPYTER_PORT}"
echo "Node: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"

export A2UI_REPO_DIR
export JUPYTER_PORT
export JUPYTER_TOKEN
export JUPYTER_WORKDIR="${JUPYTER_WORKDIR:-${A2UI_REPO_DIR}}"

# Run one Jupyter server. vLLM uses all allocated GPUs from inside the
# container through CUDA_VISIBLE_DEVICES/A2UI_VLLM_GPUS.
srun --mpi=pmix --ntasks=1 "${RUNTIME}" exec \
  --bind "${A2UI_BIND_MOUNTS}" \
  --nv \
  "${A2UI_SIF}" \
  bash "${JUPYTER_SCRIPT}"
