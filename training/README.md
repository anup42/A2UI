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

2. Train an adapter model.

```powershell
python training/scripts/train_sft.py --config training/configs/models/gemma_e2b_ir_lora.yaml
```

3. Evaluate generated IR against the flat-spec contract and existing UI metrics.

```powershell
python training/scripts/evaluate.py --predictions training/outputs/eval/predictions.jsonl
```

4. Export a model package for Android.

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
