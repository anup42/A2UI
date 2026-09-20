# Gemma 4 E2B all-parameter QAT pipeline

This is a separate, experimental training lane for the reconstructed Gemma 4
E2B official-mobile text seed. It does **not** modify or replace the official
retained-scale LoRA pipeline in `official_mobile.py`.

## What this lane does

- Reuses the checked dataset preparation/cache and the same completion-masked
  A2UI Express examples.
- Trains every unique text-model parameter, including the ordinary token
  embedding, per-layer embedding, every fully connected weight, norms, and
  other text-model parameters. LoRA/PEFT and frozen model parameters are
  rejected.
- Keeps master parameters and saved checkpoint tensors in FP32, with BF16 AMP
  for execution.
- Uses Adafactor to avoid AdamW's two full FP32 moment tensors. This is still a
  large replicated DDP job; the live maximum-shape backward/optimizer probe is
  mandatory and planning alone does not establish that memory fits.
- Applies dynamic `ste_ai_edge` weight fake quantization to every Linear and
  both embedding matrices using the experimental W2/W4/W8 allocation.
- Selects the best full-model checkpoint only with Golden32
  `unique_source_generation_reward_v5_4_avg`, then evaluates that fixed
  checkpoint on Golden35 and Bixby50 as final-only holdouts.
- Writes TensorBoard training and evaluation records using the existing
  dashboard contract.
- Exports only experimental `w248` through the existing dense checkpoint
  exporter. The exporter copies the selected full checkpoint into a fresh
  dense HF graph; it does not merge a LoRA adapter.

## What this lane does not claim

This is not Google's official retained-scale/static-A8 mobile topology or
private QAT recipe. It recomputes dynamic fake-quantization scales from the
current weights rather than learning or retaining Google's published scales,
and rebuilds a fresh dense graph during export. MTP is disabled. Desktop export does not prove
native LiteRT-LM/Android quality, memory use, or speed; those remain separate
device gates.

The model directory must be the locally reconstructed official-mobile text
seed and must include its verified `mobile_training_seed_manifest.json`,
`mobile_qparams.json`, and `mobile_qparams.safetensors`. The qparams bind seed
provenance; they are not reused as fixed training scales in this lane.

## Host contract

- Use a separate training environment with
  `pip install -r training/requirements-full-parameter-qat.txt`. The new lane's
  Trainer/Accelerate API pair is pinned independently; do not update the working
  official LoRA environment in place. API compatibility is checked before model
  loading, and live DDP bucket views are checked during training.
- Exactly 2, 4, or 8 selected NVIDIA H100 GPUs, each reporting at least 79 GiB.
- Native BF16 support.
- Per-rank microbatch fixed at 1. The default effective batch is 32, recovered
  through gradient accumulation (16/8/4 steps for 2/4/8 GPUs).
- DDP only; no FSDP/DeepSpeed fallback and no silent CPU fallback.
- A separate compatible CPU LiteRT Torch / AI Edge Quantizer Python for export.

The preflight is intentionally stronger than a forward smoke test. Each rank
first runs the existing independent numeric, coverage, and backward checks.
The disposable one-Adafactor-step memory probe then wraps the model with the
same production DDP settings, including gradient bucket views. Regular training
starts later in a new process and reloads the untouched seed; the disposable
probe is never continued as user training. A passing probe covers those tested
maximum shapes and that host allocation; it does not guarantee every later
kernel shape or eliminate the possibility of a runtime OOM.

## Plan first

Planning is offline and is the default. It validates paths and static options
but does not probe CUDA, load the model, train, or export.

Defaults: 2 epochs, learning rate `1e-5`, 3% warmup, validation/save every 500
optimizer updates, Golden32 every 1,000 updates plus the end, and 2,048 generated
tokens. TensorBoard uses `/tensorboard/<run-id>/`. These are conservative starting
settings, not a measured optimum. Full checkpoints are substantially larger than
LoRA adapters; budget disk space for retained Trainer checkpoints, best/final
copies, and the dense export staging copy. This lane deliberately requires fresh
runs and does not resume LoRA or full-model optimizer state.

```bash
python training/scripts/run_full_parameter_qat_pipeline.py \
  --model-dir /models/gemma4_e2b_mobile_dequantized_text_hf \
  --input-dir /data/a2ui_prepared_source \
  --output-dir /runs/e2b_all_parameter_qat_001 \
  --exporter-python /opt/litert-export/bin/python \
  --devices auto
```

The input directory contains `train.jsonl` and `val.jsonl` in the same reviewed
format accepted by the Golden training pipeline. Existing preparation caches
are reused only when their source, tokenizer, prompt, schema, and code identity
all match.

## Execute explicitly

Use a fresh output directory. Execution refuses to start without the explicit
experimental-export acknowledgement:

```bash
python training/scripts/run_full_parameter_qat_pipeline.py \
  --model-dir /models/gemma4_e2b_mobile_dequantized_text_hf \
  --input-dir /data/a2ui_prepared_source \
  --output-dir /runs/e2b_all_parameter_qat_001 \
  --exporter-python /opt/litert-export/bin/python \
  --devices auto \
  --execute \
  --allow-experimental-export
```

Useful bounded smoke options are `--steps`, `--eval-steps`, and
`--golden-every-steps`. The Golden cadence must remain a positive multiple of
the validation cadence. Do not use a smoke result as final model evidence.

## Outputs and stage order

Every stage runs in a bounded subprocess with console/file progress and a
content-bound receipt:

1. Verify the reconstructed official-mobile seed and qparams provenance.
2. Prepare/cache data and frozen Golden32, Golden35, and Bixby50 cohorts.
3. Detect and bind the 2/4/8-H100 profile; write
   `fit/training_config.yaml` and `fit/preparation_report.json`.
4. Run the disposable full-QAT preflight.
5. Train from a fresh seed and publish `fit/training/best_golden_checkpoint`
   plus `fit/training/final_model`.
6. Re-evaluate the selected checkpoint on Golden32, Golden35, and Bixby50.
7. Export experimental W248 into `experimental_w248_export/`.
8. Write `evaluation_scorecard.json` and the three-cohort `results.md` table.

The selected checkpoint must contain verified all-parameter scope, complete
matrix QAT coverage, successful numeric/backward preflight evidence, a complete
FP32 541-tensor inventory, positive optimizer-step provenance, and immutable
checkpoint/config/data hashes before evaluation or export proceeds.

## Interpreting completion

Pipeline completion means checked full-parameter training, three-cohort HF/QAT
evaluation, and creation of an experimental W248 artifact. It does not mean the
artifact has passed Android GPU execution, native prompt-to-A2UI quality, or
throughput parity. The final report says those gates are untested until they are
run separately on the intended device/runtime.

## Regression evidence and remaining gates

Local CPU validation on 2026-09-20 included a broad existing-pipeline suite
(943 passed, 24 skipped) and a fresh affected-suite rerun (370 passed, 9 skipped).
A real tiny Transformers Trainer/Accelerate test also exercised one Adafactor
update and full safetensor checkpoint save, including changed embeddings and
normalization weights. An additional two-layer Gemma 4 CPU test exercised 22
actual Linear/Embedding QAT wrappers, finite gradients for every parameter,
one Adafactor probe step, and a full FP32 checkpoint round trip. These synthetic
CPU tests are not an E2B training run.
The original official-mobile entry point, orchestration module and recipe YAML
were not edited.

Useful focused checks after installing the new training environment:

```bash
python -m pytest -q -p no:cacheprovider \
  training/tests/test_full_model_contract.py \
  training/tests/test_full_parameters.py \
  training/tests/test_full_parameter_qat_pipeline.py \
  training/tests/test_full_qat_trainer_integration.py \
  training/tests/test_official_mobile_pipeline.py
```

No actual H100/DDP run, full-model conversion, Android inference, or measured
quality/throughput comparison was performed locally. Run the mandatory live
preflight on the target host; do not disable a failed memory, scope, or
provenance check to continue.
