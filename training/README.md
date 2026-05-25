# Response-to-IR Training

This folder trains local Stage 3 models that convert Stage 2 response text into the Android flat-spec GenUI IR:

```json
{"root":"...","state":{},"elements":{}}
```

`dataset/` remains responsible for generating queries, responses, assets, and cloud IR. `training/` consumes completed dataset runs and provides a model-agnostic path for data preparation, SFT training, evaluation, export, and Android packaging.

## Main Flow

1. Prepare response-to-IR pairs from a dataset run.

```powershell
python training/scripts/prepare_dataset.py --config training/configs/datasets/dataset_v1_stage3.yaml
```

For Gemma 4 training from a folder that contains one or more Stage 3 `genui.jsonl`
files, prepare a deterministic 90/10 train/validation split with:

```powershell
python training/scripts/prepare_dataset.py --config training/configs/datasets/stage3_folder_90_10.yaml --source-genui-dir dataset/data/runs/dataset_v3 --output-dir training/outputs/datasets/stage3_folder_90_10
```

The folder reader is recursive by default (`**/genui.jsonl`). It writes
`train.jsonl`, `val.jsonl`, `test.jsonl`, and `all.jsonl`; `all.jsonl` is used for
fixed-set evaluation jobs.

Prepare the golden50 set once before training:

```powershell
python training/scripts/prepare_dataset.py --config training/configs/datasets/golden50_stage3_eval.yaml
```

2. Train an adapter model.

```powershell
python training/scripts/train_sft.py --config training/configs/models/gemma_e2b_ir_lora.yaml
```

Start Gemma 4 E2B QLoRA training with golden50 evaluation at the end of every epoch:

```powershell
python training/scripts/train_sft.py --config training/configs/models/gemma4_e2b_ir_lora.yaml
```

This requires a GPU machine with the packages in
`training/requirements-training.txt`. On CPU-only machines, use compile/tests only;
do not run the training command.

For Slurm machines, submit the repo-owned sbatch entrypoint instead of writing an
ad-hoc script:

```bash
sbatch training/scripts/slurm_train_gemma4_e2b.sbatch
```

The sbatch file runs the training command through `srun`, activates
`/home/k_anup/gemma4_env` by default, and requests four GPUs. The training
entrypoint uses all queryable GPUs visible to the process by default. The
training runner also downgrades `bfloat16` to `float16` automatically when the
visible GPU/PyTorch setup does not support BF16. By default, SFT training fails
fast when PyTorch cannot see CUDA, because otherwise the job silently runs on CPU
and appears stuck even when `nvidia-smi` shows idle GPUs. Override paths without
editing the file:

```bash
sbatch --export=ALL,A2UI_REPO_DIR=/home/k_anup/code/GenUI,A2UI_VENV=/home/k_anup/gemma4_env training/scripts/slurm_train_gemma4_e2b.sbatch
```

If a Slurm job fails, inspect both Slurm state and the training log:

```bash
sacct -j <jobid> -o JobID,JobName%30,State,ExitCode,DerivedExitCode,Elapsed,Timelimit,MaxRSS,ReqMem,NodeList,Reason -P
scontrol show job -dd <jobid>
tail -200 training/logs/slurm-<jobid>.err
tail -200 training/logs/slurm-<jobid>.out
```

If `nvidia-smi` shows GPUs but only Xorg/display processes and `0%` utilization,
verify PyTorch from the same environment/container:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY
```

If that reports `cuda available: False`, fix the environment rather than waiting
for training: install a CUDA-enabled PyTorch build and launch Singularity with
GPU passthrough, for example `singularity exec --nv <image> ...`.

The training entrypoint normalizes `CUDA_VISIBLE_DEVICES` before importing
PyTorch. If the shell inherits a multi-GPU value such as `0,1,2,3`, it keeps all
GPUs that `nvidia-smi --query-gpu=index` can query. To use a specific healthy
set, pass it explicitly:

```bash
A2UI_CUDA_VISIBLE_DEVICES=0,1,3 python3 training/scripts/train_sft.py --config training/configs/models/gemma4_e2b_ir_lora.yaml
```

To exclude a known bad device without hardcoding the full set:

```bash
A2UI_EXCLUDE_CUDA_DEVICES=2 python3 training/scripts/train_sft.py --config training/configs/models/gemma4_e2b_ir_lora.yaml
```

For the HF Trainer backend, labels are completion-only: prompt tokens are masked
with `-100`, completion/IR tokens are left trainable, and the preflight fails if
a batch would have zero trainable labels. This prevents apparent zero-gradient
runs caused by truncating away the IR completion.

Shell and sbatch files are forced to LF line endings through `.gitattributes`.
This avoids Linux shebang failures such as `cannot execute: required file not
found` caused by CRLF files copied from Windows.

3. Evaluate generated IR against the flat-spec contract and existing UI metrics.

```powershell
python training/scripts/evaluate.py --predictions training/outputs/eval/predictions.jsonl
```

4. Evaluate with dataset-compatible overall score.

```powershell
python training/scripts/evaluate.py --predictions training/outputs/eval/predictions.jsonl --weights-config dataset/configs/run.yaml --baseline-aggregate dataset/data/runs/<baseline>/aggregates.json
```

The evaluator writes `aggregate_metrics.json` with `overall_score`, `baseline_overall_score`, and `overall_score_delta_vs_baseline` when a baseline is provided.

5. Export a trained Gemma 4 E2B model for Google AI Edge Gallery.

Google AI Edge Gallery imports local LLMs as `.litertlm` files. After training,
merge the LoRA adapter into a Hugging Face model directory and run Google's
LiteRT Torch Hugging Face exporter:

```powershell
python -m pip install -r training/requirements-edge-export.txt
hf auth login
python training/scripts/export_edge_gallery_model.py --config training/configs/export/edge_gallery_gemma4_e2b.yaml --merge-lora
adb push training/outputs/export/gemma4_e2b_ir_edge_gallery/litertlm/<model>.litertlm /sdcard/Download/
```

For CI or CPU-only machines, validate the generated command/manifest without
running conversion:

```powershell
python training/scripts/export_edge_gallery_model.py --config training/configs/export/edge_gallery_gemma4_e2b.yaml --dry-run
```

The export wrapper writes `edge_gallery_export_plan.json`,
`run_litert_export.ps1`, `EDGE_GALLERY_IMPORT.md`, and `model_manifest.json`
under `training/outputs/export/gemma4_e2b_ir_edge_gallery`.

6. Export a legacy Android metadata package.

```powershell
python training/scripts/export_model.py --config training/configs/export/litertlm_gemma.yaml
python training/scripts/package_android_model.py --export-dir training/outputs/export/gemma_e2b_ir_litertlm
```

## Architecture

The training code is intentionally model-agnostic:

- Dataset preparation writes neutral JSONL records with `messages`, `prompt`, and `completion`.
- Model-specific formatting is isolated in `ir_training.models.ModelAdapter` implementations.
- Export targets are registry-driven so future formats can be added without changing data prep or metrics.

Initial target: Gemma E2B-family LoRA/QLoRA SFT. Future adapters can be added for Qwen, Llama, Phi, or any Hugging Face causal LM.

## Output Policy

Generated training artifacts are written under `training/outputs/`, `training/runs/`, or `training/checkpoints/`. These folders should stay uncommitted unless a small manifest/report is intentionally needed for review.
