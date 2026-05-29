# Gemma4 31B vLLM Dataset Generation

This profile runs A2UI dataset Stage 1, Stage 2, and Stage 3 against a local
Gemma4 31B model served by vLLM inside a Singularity container.

## Camera-Captured Target Machine Notes

The latest device-camera capture showed:

- Runtime available: `singularity`
- `apptainer` and `module` were not available in the active shell
- NVIDIA driver: `580.159.03`
- CUDA reported by `nvidia-smi`: `13.0`
- GPUs visible in the captured frame: at least four `NVIDIA RTX 6000 Ada`
- GPU memory in the captured frame: `49140 MiB` per visible GPU
- No active GPU process was visible

## Setup Host Dataset Environment

```bash
cd /path/to/A2UI

export VLLM_SIF=$HOME/containers/a2ui-vllm-cu128-source_gemma4_speculative.sif
export GEMMA4_MODEL_PATH=$HOME/dataset_generation/models/gemma4-31b
export GEMMA4_DRAFT_MODEL_PATH=$HOME/dataset_generation/models/gemma4-2b

bash dataset/scripts/setup_gemma4_vllm_singularity_env.sh
source gemma4_vllm_env/activate_gemma4_vllm.sh
```

The draft model is required when `GEMMA4_SPECULATIVE_MODE=draft`. To disable
speculative decoding:

```bash
export GEMMA4_SPECULATIVE_MODE=off
```

## Start vLLM

Non-reasoning mode:

```bash
source gemma4_vllm_env/activate_gemma4_vllm.sh

export GEMMA4_ENABLE_REASONING=0
export CUDA_VISIBLE_DEVICES=0,1,2,3
export A2UI_VLLM_GPUS=4
export VLLM_MAX_MODEL_LEN=8192
export GEMMA4_SPECULATIVE_MODE=draft
export GEMMA4_SPECULATIVE_TOKENS=4

bash dataset/scripts/run_gemma4_vllm_singularity.sh
```

Reasoning/thinking mode:

```bash
source gemma4_vllm_env/activate_gemma4_vllm.sh

export GEMMA4_ENABLE_REASONING=1
export LOCAL_VLLM_ENABLE_THINKING=1
export LOCAL_VLLM_STRIP_THINKING=1

bash dataset/scripts/run_gemma4_vllm_singularity.sh
```

The server does not add vLLM `--enable-reasoning` flags by default, because
Gemma-compatible reasoning parsers are build-dependent. If your vLLM build
supports one, enable it explicitly:

```bash
export GEMMA4_ENABLE_SERVER_REASONING_FLAGS=1
export GEMMA4_REASONING_PARSER=<parser-name>
```

## Run Dataset Stages

With the server already running:

```bash
source gemma4_vllm_env/activate_gemma4_vllm.sh

export RUN_ID=dataset_gemma4_31b_vllm_v0
export GEMMA4_ENABLE_REASONING=0

bash dataset/scripts/run_gemma4_dataset_stages.sh
```

Run stages one by one:

```bash
STAGE=1 RUN_ID=dataset_gemma4_31b_vllm_v0 bash dataset/scripts/run_gemma4_dataset_stages.sh
STAGE=2 RUN_ID=dataset_gemma4_31b_vllm_v0 bash dataset/scripts/run_gemma4_dataset_stages.sh
STAGE=3 RUN_ID=dataset_gemma4_31b_vllm_v0 STAGE3_BATCH_SIZE=1 bash dataset/scripts/run_gemma4_dataset_stages.sh
```

Single-terminal 50-sample smoke run:

```bash
source gemma4_vllm_env/activate_gemma4_vllm.sh

export RUN_ID=dataset_gemma4_31b_vllm_50
export TARGET_COUNT=50
export GEMMA4_ENABLE_REASONING=0
export GEMMA4_SPECULATIVE_MODE=draft

bash dataset/scripts/run_gemma4_stage123_50.sh
```
