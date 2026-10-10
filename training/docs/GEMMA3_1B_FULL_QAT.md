# Gemma 3 1B full-parameter QAT -> INT8 LiteRT-LM

Entry point: `training/scripts/run_gemma3_1b_full_qat_pipeline.py`.
This is a thin wrapper over the recent shared Golden training/deployment code,
not another trainer or converter. It never generates Stage 1/2/3 data.

## What it runs

1. Check the actual local Gemma 3 1B text architecture, CUDA allocation,
   exporter dependencies, and native GPU runtime prerequisites.
2. Prepare completed supervision with the existing prompt, strict validators,
   tokenizer/source hashes, reserved-test exclusion, and reusable token cache.
3. Full-parameter W8 QAT: train every parameter, including embeddings and norms;
   fake-quantize every Linear/Embedding matrix. No PEFT/LoRA. FP32 master weights,
   BF16 compute where supported, SDPA and non-reentrant gradient checkpointing.
4. Select the best checkpoint using Golden32's 31-unique-source v5.4 reward;
   test both best and final checkpoints with **QAT on** on Golden32, Golden35,
   and Bixby50. The two holdouts never select checkpoints.
5. Copy the entire selected dense checkpoint (no adapter merge), verify full
   parameter/alias/FP32/QAT coverage and provenance, and test that copy.
6. Export **W8 only** using the public `dynamic_wi8_afp32` recipe; require physical
   INT8 weight inspection, then test the actual `.litertlm` on all three cohorts
   with the shared native GPU runner. Publish the scorecard and TensorBoard data.

This is a fresh public Gemma 3 graph with INT8 weights and floating activations.
It is **not** the pinned 270M official-graph transplant, the E2B mixed W2/W4/W8
format, or Google's private training recipe. No MTP/drafter is added. The public
exporter supports Gemma3ForCausalLM and local checkpoints
([Google conversion guide](https://developers.google.com/edge/litert/conversion/pytorch/genai)).
Existing E2B/270M defaults, LoRA and dedicated ZeRO pipelines are unchanged.

## Host and inputs

Run on the Linux CUDA training host, not this review PC. Supply an existing local
`google/gemma-3-1b-it` snapshot (or a compatible dense 1B SFT checkpoint) with
its original tokenizer and config. Do not relabel another model or use a
pre-quantized/adapter-only model. No model download happens in this launcher.

Keep three separate environments:

- Training/HF evaluation: install `training/requirements-gemma3-full-qat.txt`
  in a dedicated CUDA environment. Retain the CUDA PyTorch build appropriate
  for the host; do not install the CPU converter wheels there.
- Conversion and native runtime: use the existing setup in
  [DEPLOYMENT_EXPORT_ENVIRONMENT.md](DEPLOYMENT_EXPORT_ENVIRONMENT.md).
  The converter is CPU-only. Native testing needs working NVIDIA Vulkan,
  not merely a working CUDA/PyTorch installation.

The profile supports single GPU or replicated DDP via `--devices`; it does not
enable ZeRO-2/3. Default microbatch is **1**, effective batch **32** (4 GPUs ->
accumulation 8; 8 GPUs -> 4). Training sequence limit is **4096**, evaluation
prompt limit **5120**, generation cap **2048**, export cache **8192**. Changing
the training limit does not shrink evaluation prompts. Budgets are validated;
oversized training rows are filtered by preparation, never silently truncated.

## Command

From the repository root, set these paths to real local files/directories.
The dataset must contain completed, source-bound `train.jsonl` and `val.jsonl`.
Alternatively replace `--input-dir` with `--source-run-dir` for a completed
Stage 3 run. Keep output/cache paths outside input/model directories.

```bash
TRAIN_PY=/group-volume/k.anup/envs/gemma3-full-qat/bin/python
EXPORT_PY=/group-volume/k.anup/envs/a2ui-export-094/bin/python
RUNTIME_PY=/group-volume/k.anup/envs/a2ui-litert-017/bin/python
MODEL=/group-volume/k.anup/models/gemma-3-1b-it
DATA=/group-volume/k.anup/working_dir/improved_training_data
OUTPUT=/group-volume/k.anup/working_dir/gemma3_1b_full_qat_v1

# Plan only: no weights loaded, training, conversion, or output directory created.
"$TRAIN_PY" -u training/scripts/run_gemma3_1b_full_qat_pipeline.py \
  --model-dir "$MODEL" --input-dir "$DATA" --output-dir "$OUTPUT" \
  --exporter-python "$EXPORT_PY" --runtime-python "$RUNTIME_PY" \
  --devices auto --epochs 2 --learning-rate 1e-5 \
  --max-seq-length 4096 --max-input-tokens 5120 --max-new-tokens 2048
```

Append `--execute` on that host to run the plan. For a bounded first smoke, use
a **different fresh output folder** and append
`--execute --steps 20 --eval-steps 10 --golden-every-steps 10`.
The smoke still performs all required checkpoint/native evaluations and export;
it is not a quality benchmark. For training/evaluation without export, the same
profile is available in `run_golden_training.py --profile 1b`.

The full workflow exports only the selected checkpoint and only W8 to avoid
unrequested precision conversions. Preparation/token caches are enabled by
default and validated before reuse. Increase microbatch only after measuring
memory and throughput on your host; effective batch must remain divisible by
`GPU count * microbatch`. Local backward preflight tests the longest prepared
row with resident gradients, but **does not certify optimizer or NCCL headroom**.
No CUDA execution, real 1B export, Android execution, or speed measurement was
performed on this PC; these scripts do not guarantee hardware fit or accuracy.

## Outputs, recovery and inference boundaries

- `<OUTPUT>/deployment/variants/w8/model.litertlm`
- `<OUTPUT>/deployment_scorecard.json` and `deployment_results.md`
- `<OUTPUT>/evaluations/w8_{golden32,golden35,bixby50}/`
- `<OUTPUT>/<run_id>_training/fit/training/best_golden_checkpoint` and `final_model`
  in that same training directory.

If export/native testing fails **after full training and checkpoint evaluations
completed**, rerun the identical command with `--execute --resume-run`.
The existing recovery logic verifies hashes, reuses completed work, and puts
failed-stage retries in fresh recovery directories; it never restarts training.
Check the manifest for the final artifact path after recovery. This flag is
not mid-training optimizer resume. Do not edit completed files or reuse a failed
pre-training output as a fresh run.

`--skip-litert-evaluation` is an explicit export-only alternative on a host without
Vulkan. It retains checkpoint tests/export validation and reports native tests
as **skipped**, not passed. It cannot establish Android GPU compatibility.
Default native tests run on the host GPU; Android-device correctness/latency is
a separate validation. For later testing use the existing
`evaluate_litertlm_on_golden.py` with the saved training config, prepared cohort,
runtime interpreter, and `--builtin-gpu --require-prepared-contract`.
