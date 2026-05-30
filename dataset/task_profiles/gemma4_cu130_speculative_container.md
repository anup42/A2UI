# Gemma4 CUDA 13.0 Speculative Container

Target machine:
- Driver: `580.159.03`
- CUDA runtime: `13.0`
- GPU: NVIDIA RTX Ada Generation, about `49140MiB`
- Runtime: Singularity or Apptainer

Build on an internet Docker machine:

```bash
cd /path/to/A2UI
git pull

A2UI_BYPASS_SSL=1 \
MAX_JOBS=12 \
NVCC_THREADS=4 \
TORCH_CUDA_ARCH_LIST="8.9+PTX" \
CMAKE_CUDA_ARCHITECTURES=89 \
bash dataset/scripts/build_vllm_cu130_gemma4_speculative_image.sh
```

Output:

```text
vllm_cu130_artifacts/a2ui-vllm-cu130-source_gemma4_speculative_ada.tar
```

Copy to the GPU machine:

```bash
scp vllm_cu130_artifacts/a2ui-vllm-cu130-source_gemma4_speculative_ada.tar \
  <user>@<gpu-host>:$HOME/containers/
```

Convert to SIF on the GPU machine:

```bash
cd /path/to/A2UI
module load apptainer 2>/dev/null || true

bash dataset/scripts/build_vllm_cu130_gemma4_speculative_sif.sh \
  $HOME/containers/a2ui-vllm-cu130-source_gemma4_speculative_ada.tar
```

Create the host runner env:

```bash
cd /path/to/A2UI

GEMMA4_MODEL_PATH=/path/to/gemma-4-31B-it \
GEMMA4_DRAFT_MODEL_PATH=/path/to/gemma-4-31B-it-assistant \
bash dataset/scripts/setup_gemma4_vllm_cu130_singularity_env.sh

source gemma4_vllm_cu130_env/activate_gemma4_vllm_cu130.sh
```

Start vLLM with all visible GPUs:

```bash
bash dataset/scripts/run_gemma4_vllm_cu130_singularity.sh
```

JupyterLab / Slurm launch path:

```bash
cd /path/to/A2UI

A2UI_SIF=$HOME/containers/a2ui-vllm-cu130-source_gemma4_speculative_ada.sif \
A2UI_REPO_DIR=/path/to/A2UI \
sbatch jupyterlab/sbatch.sh
```

Inside a JupyterLab terminal, start vLLM directly inside the already-running
container:

```bash
cd /path/to/A2UI

GEMMA4_MODEL_PATH=/home/k_anup/Storage_gpu/models/gemma-4-31b-it \
GEMMA4_DRAFT_MODEL_PATH=/home/k_anup/Storage_gpu/models/gemma-4-31b-it-assistant \
bash dataset/scripts/run_gemma4_vllm_cu130_inside_container.sh
```

Run dataset stages against the local server:

```bash
source gemma4_vllm_cu130_env/activate_gemma4_vllm_cu130.sh

RUN_ID=dataset_gemma4_cu130_v0 \
MAX_QUERIES_TOTAL=50 \
MAX_RESPONSES_TOTAL=50 \
MAX_GENUI_TOTAL=50 \
STAGE=all \
bash dataset/scripts/run_gemma4_dataset_stages.sh
```

Notes:
- `CUDA_VISIBLE_DEVICES` is auto-filled from `nvidia-smi` when unset.
- `A2UI_VLLM_GPUS` is auto-derived from `CUDA_VISIBLE_DEVICES`.
- The image includes dataset Stage 1-5 dependencies, Playwright Chromium, and `bitsandbytes`.
- The default vLLM source ref is `9b4e83934d895b5f6e488411cd46c8d0915115a1`.
