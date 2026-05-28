# Qwen3.6 35B A3B vLLM Dataset Generation

This profile runs dataset Stage 1, Stage 2, and Stage 3 against a local vLLM
OpenAI-compatible server using `Qwen/Qwen3.6-35B-A3B` with Qwen reasoning
enabled.

## Camera-Captured Target Machine Notes

The latest device-camera capture showed:

- Python: `3.13.13`
- NVIDIA driver: `570.133.20`
- CUDA reported by `nvidia-smi`: `12.8`
- GPUs visible in the captured frame: at least two `NVIDIA A100-PCIE-40GB`
- GPU memory in the captured frame: `40960 MiB` per visible GPU
- No active GPU process was visible in the capture

Do not use Python 3.13 for vLLM. Create/install a Python 3.10, 3.11, or 3.12
environment on the GPU machine first. The scripts default to `python3.11`.

The setup pins vLLM/torch to a CUDA 12-compatible stack for this driver path by
default. Do not use the latest unpinned PyPI stack here because it can pull CUDA
12.9/13.0 packages that require a newer NVIDIA driver. GitHub release wheel
URLs and the PyTorch extra index are avoided by default because they are blocked
or fail certificate validation on this cluster path. Defaults:

```bash
export A2UI_VLLM_VERSION=0.9.2
export A2UI_VLLM_CUDA_VARIANT=126
export A2UI_PYTORCH_INDEX_URL=
```

## Internet 2-GPU Machine: Use Existing Model Folder

This is the primary path for the current machine. It has internet and GPUs, and
the model is already present at `~/models/Qwen--Qwen3.6-35B-A3B`.

```bash
cd /path/to/A2UI
export QWEN_MODEL_PATH=~/models/Qwen--Qwen3.6-35B-A3B
bash dataset/scripts/setup_qwen_vllm_online_2gpu.sh
```

If `QWEN_MODEL_PATH` is not set and the script is run interactively, it will ask
for the path. To opt back into Hugging Face download, explicitly set:

```bash
export A2UI_DOWNLOAD_QWEN_MODEL=1
```

## Internet Machine: Download Offline Bundle

Run this on a Linux machine with internet access and the same Python minor
version/platform as the offline GPU machine:

```bash
git clone <A2UI repo> A2UI
cd A2UI

export PYTHON_BIN=python3.11
export QWEN_MODEL_ID=Qwen/Qwen3.6-35B-A3B
export BUNDLE_DIR=$PWD/qwen_vllm_offline_bundle

bash dataset/scripts/download_qwen_vllm_offline_bundle.sh
```

The setup/download scripts default to bypassing TLS certificate verification
because this cluster path is behind a certificate chain that Python cannot
verify. To re-enable verification, run with:

```bash
export A2UI_DISABLE_SSL_VERIFY=0
```

If verification is enabled and Hugging Face download fails with
`CERTIFICATE_VERIFY_FAILED` or `unable to get local issuer certificate`, point
the scripts at the machine or company CA bundle:

```bash
export A2UI_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
# or, on RHEL/CentOS:
export A2UI_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
# or your company root CA:
export A2UI_CA_BUNDLE=/path/to/company-root-ca.pem
```

Then rerun the setup/download script.

Copy the complete `qwen_vllm_offline_bundle/` folder to the offline GPU
machine.

If the Hugging Face model repo name is different, override `QWEN_MODEL_ID` and
`QWEN_MODEL_PATH` consistently.

## Offline GPU Machine: Install Env

```bash
cd /path/to/A2UI

export PYTHON_BIN=python3.11
export ENV_DIR=$PWD/qwen_vllm_env

bash dataset/scripts/install_qwen_vllm_offline_env.sh /path/to/qwen_vllm_offline_bundle
source qwen_vllm_env/activate_qwen_vllm.sh
```

The setup also writes a convenience helper at
`qwen_vllm_env/bin/activate_qwen_vllm`. Source either helper exactly as printed
by setup; do not include the `Activate:` heading or a trailing colon in the
command.

## Start vLLM Server

Use all visible GPUs:

```bash
cd /path/to/A2UI
source qwen_vllm_env/activate_qwen_vllm.sh

export QWEN_MODEL_PATH=~/models/Qwen--Qwen3.6-35B-A3B
export CUDA_VISIBLE_DEVICES=0,1
export A2UI_VLLM_GPUS=2
export VLLM_MAX_MODEL_LEN=32768

bash dataset/scripts/run_qwen36_vllm_server.sh
```

For 4 or 8 GPUs, update both `CUDA_VISIBLE_DEVICES` and `A2UI_VLLM_GPUS`:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export A2UI_VLLM_GPUS=4
```

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export A2UI_VLLM_GPUS=8
```

## Run Dataset Stages

Run all stages sequentially:

```bash
cd /path/to/A2UI
source qwen_vllm_env/activate_qwen_vllm.sh

export RUN_ID=dataset_qwen36_vllm_reasoning_v0
export RATE_LIMIT_QPS=0.2
export STAGE3_BATCH_SIZE=1

bash dataset/scripts/run_qwen36_dataset_stages.sh
```

Run stages one by one:

```bash
STAGE=1 RUN_ID=dataset_qwen36_vllm_reasoning_v0 bash dataset/scripts/run_qwen36_dataset_stages.sh
STAGE=2 RUN_ID=dataset_qwen36_vllm_reasoning_v0 bash dataset/scripts/run_qwen36_dataset_stages.sh
STAGE=3 RUN_ID=dataset_qwen36_vllm_reasoning_v0 STAGE3_BATCH_SIZE=1 bash dataset/scripts/run_qwen36_dataset_stages.sh
```

Equivalent direct commands:

```bash
export LOCAL_ALLOW_HTTP_ENDPOINT=1
export LOCAL_STRICT_OFFLINE=0
export LOCAL_VLLM_ENABLE_THINKING=1
export VLLM_REASONING_PARSER=qwen3
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python dataset/src/main.py --stage 1 --model qwen36_35b_a3b_vllm_reasoning --run_id dataset_qwen36_vllm_reasoning_v0 --rate_limit_qps 0.2
python dataset/src/main.py --stage 2 --model qwen36_35b_a3b_vllm_reasoning --run_id dataset_qwen36_vllm_reasoning_v0 --rate_limit_qps 0.2
python dataset/src/main.py --stage 3 --model qwen36_35b_a3b_vllm_reasoning --run_id dataset_qwen36_vllm_reasoning_v0 --genui_batch_size 1 --rate_limit_qps 0.2
```

## Auto-Start vLLM From Dataset CLI

If you want `dataset/src/main.py` to start/stop vLLM for one stage:

```bash
python dataset/src/main.py \
  --stage 3 \
  --model qwen36_35b_a3b_vllm_reasoning \
  --run_id dataset_qwen36_vllm_reasoning_v0 \
  --start_vllm \
  --vllm_model_path /path/to/qwen_vllm_offline_bundle/models/Qwen--Qwen3.6-35B-A3B \
  --vllm_gpus 2 \
  --vllm_cuda_visible_devices 0,1 \
  --vllm_dtype bfloat16 \
  --vllm_max_model_len 32768 \
  --vllm_trust_remote_code \
  --vllm_start_timeout 900
```

For long runs, a persistent server is usually better than auto-starting per
stage because it avoids model reload cost.
