# Gemma4 31B vLLM Dataset Generation

This profile runs A2UI dataset Stage 1, Stage 2, and Stage 3 against a local
Gemma4 31B model served by vLLM from a normal Python virtualenv. It does not
require Docker, Apptainer, or Singularity.

## Target Machine Notes

The latest device-camera capture showed:

- Runtime available: `singularity`, but this workflow does not use it
- NVIDIA driver: `580.159.03`
- CUDA reported by `nvidia-smi`: `13.0`
- GPUs visible: at least four `NVIDIA RTX 6000 Ada`
- GPU memory visible: `49140 MiB` per GPU

## Model Folder Layout

Set `GEMMA4_MODEL_ROOT` to the parent folder that contains both checkpoints:

```text
<GEMMA4_MODEL_ROOT>/
  gemma-4-31B-it/
  gemma-4-31B-it-assistant/
```

Equivalent names like `gemma-4-31b-it`, `google--gemma-4-31B-it`, and
`gemma4-31b-it-assistant` are also auto-detected. If your folder names differ,
set `GEMMA4_MODEL_PATH` and `GEMMA4_ASSISTANT_MODEL_PATH` directly.

## Create Python Environment

```bash
cd /path/to/A2UI

export GEMMA4_MODEL_ROOT=/path/to/parent/folder
export PYTHON_BIN=python3.12

bash dataset/scripts/setup_gemma4_vllm_python_env.sh
source gemma4_vllm_env/activate_gemma4_vllm.sh
```

The setup script defaults to the same Gemma4-special vLLM source ref used by
`dataset/scripts/build_vllm_cu128_gemma4_speculative_image.sh`:

```text
9b4e83934d895b5f6e488411cd46c8d0915115a1
```

It builds from source with `--no-build-isolation`, matching the Docker build
approach used for Gemma4 speculative decoding.

```bash
export A2UI_VLLM_INSTALL_MODE=source
export VLLM_SOURCE_REF=9b4e83934d895b5f6e488411cd46c8d0915115a1
export PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
export A2UI_TORCH_BACKEND=cu128
bash dataset/scripts/setup_gemma4_vllm_python_env.sh
```

The special vLLM source ref requires a matching PyTorch C++ ABI. The setup
script defaults to a clean reinstall of the vLLM stack and pins the same Torch
family used in the Docker script:

```bash
export A2UI_CLEAN_VLLM_STACK=1
export TORCH_VERSION=2.11.0
export TORCHVISION_VERSION=0.26.0
export TORCHAUDIO_VERSION=2.11.0
bash dataset/scripts/setup_gemma4_vllm_python_env.sh
```

Use this when you see an import error from `vllm/_C.abi3.so` with an undefined
`torch::jit` symbol. That error means vLLM and PyTorch were installed from
incompatible builds.

The setup script defaults to aggressive SSL bypass because the target machines
have certificate interception issues:

```bash
export A2UI_DISABLE_SSL_VERIFY=1
```

If the machine has working certificates, disable it:

```bash
export A2UI_DISABLE_SSL_VERIFY=0
```

If vLLM fails with missing CUDA runtime libraries, rerun the setup script. The
script installs CUDA 12.8 runtime/NVCC Python wheels by default because the
Gemma4-special Docker build was CUDA 12.8 based:

```bash
export CUDA_RUNTIME_PACKAGE=nvidia-cuda-runtime-cu12==12.8.90
export CUDA_NVCC_PACKAGE=nvidia-cuda-nvcc-cu12==12.8.93
bash dataset/scripts/setup_gemma4_vllm_python_env.sh
```

If vLLM fails inside `flashinfer/jit/cpp_ext.py` with `ninja returned non-zero
exit status 127`, the FlashInfer sampler is enabled but the JIT toolchain is
incomplete. Rerun setup:

```bash
bash dataset/scripts/setup_gemma4_vllm_python_env.sh
```

The setup script keeps FlashInfer sampler enabled by default and installs:

- `flashinfer-python`
- `flashinfer-cubin`
- `flashinfer-jit-cache` from `https://flashinfer.ai/whl/${FLASHINFER_CUDA_TAG}`
- `ninja`
- `cmake`
- CUDA 12.8 runtime and NVCC Python wheels

Defaults:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_HAS_FLASHINFER_CUBIN=1
export FLASHINFER_CUDA_TAG=cu128
```

Use `FLASHINFER_CUDA_TAG=cu129` or `cu130` only if you intentionally switch the
PyTorch/vLLM stack away from CUDA 12.8.

## Start vLLM

Non-reasoning mode with speculative decoding:

```bash
source gemma4_vllm_env/activate_gemma4_vllm.sh

export GEMMA4_MODEL_ROOT=/path/to/parent/folder
export GEMMA4_ENABLE_REASONING=0
export GEMMA4_SPECULATIVE_MODE=draft
export GEMMA4_SPECULATIVE_TOKENS=4
export VLLM_MAX_MODEL_LEN=8192
export VLLM_GPU_MEMORY_UTILIZATION=0.90

bash dataset/scripts/run_gemma4_vllm_python.sh
```

Reasoning/thinking mode:

```bash
source gemma4_vllm_env/activate_gemma4_vllm.sh

export GEMMA4_MODEL_ROOT=/path/to/parent/folder
export GEMMA4_ENABLE_REASONING=1
export LOCAL_VLLM_ENABLE_THINKING=1
export LOCAL_VLLM_STRIP_THINKING=1

bash dataset/scripts/run_gemma4_vllm_python.sh
```

To disable speculative decoding:

```bash
export GEMMA4_SPECULATIVE_MODE=off
```

The vLLM Gemma4 recipe uses the assistant checkpoint with
`--speculative-config '{"model": "...assistant", "num_speculative_tokens": 4}'`
and uses `--reasoning-parser gemma4` for thinking mode. The launcher checks
that these flags exist before starting.

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

export GEMMA4_MODEL_ROOT=/path/to/parent/folder
export RUN_ID=dataset_gemma4_31b_vllm_50
export TARGET_COUNT=50
export GEMMA4_ENABLE_REASONING=0
export GEMMA4_SPECULATIVE_MODE=draft

bash dataset/scripts/run_gemma4_stage123_50_python.sh
```

## Readiness Check

```bash
curl -fsS http://127.0.0.1:8000/v1/models
```

If the call succeeds, the dataset stage scripts can use the local endpoint.
