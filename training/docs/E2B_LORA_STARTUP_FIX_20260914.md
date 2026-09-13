# E2B startup failure: LoRA selector matched no layers

## Cause and fix

The reported two-GPU run loaded its weights, then stopped in
`resolve_lora_config_targets`, before adapter construction or optimizer updates.
The dense E2B review recipe assumed only the multimodal model hierarchy:

| Loaded model class | Language projection example |
|---|---|
| `Gemma4ForCausalLM` | `model.layers.0.self_attn.q_proj` |
| `Gemma4ForConditionalGeneration` | `model.language_model.layers.0.self_attn.q_proj` |

The old selector required `language_model` and cannot match the first layout.
The saved checkpoint config and installed Transformers mapping determine the
loaded class; the `auto_causal_lm` loader name alone does not guarantee one
hierarchy. The screenshot did not include an actual module inventory, but
this mismatch is reproducible with a real, randomly initialized HF text model.
See the [supported Transformers implementation](https://github.com/huggingface/transformers/blob/v5.10.1/src/transformers/models/gemma4/modeling_gemma4.py)
and [AutoModel mappings](https://github.com/huggingface/transformers/blob/v5.10.1/src/transformers/models/auto/modeling_auto.py).

Both the ordinary E2B review recipe and its lower-level `--qv-baseline`
ablation now accept `model\.(?:language_model\.)?layers\.\d+\.` as their
language-root pattern. Attention/MLP restrictions remain unchanged. Matching
is anchored: no arbitrary leading prefix, vision/audio tower, per-layer
embedding/projection or vocabulary head. Existing `.linear` wrapper resolution
and exact-target resume checks remain; there is no `all-linear` fallback.

Console logs now show the loaded class/type, target count and example paths
before LoRA attachment. No-match errors include a bounded sample of actual
Linear paths and their total count, without printing weights or data.
Successful full target inventories remain in `training_metadata.json`.

This traceback is not caused by the dataset, hyperparameter tuning, GPU count,
or bfloat16. The earlier Transformer Engine JAX-extension warning is not the
fatal exception; installing that extension is not the fix for this traceback.
Gemma 3 270M full-model SFT and separate retained-scale/official QAT contracts
are unchanged. Normal training and sequential trials share the corrected
recipe; tuning is not required.

## Restart on the training PC

Stop any failed launcher before updating. From the repository checkout:

```bash
git pull --ff-only
python training/scripts/run_golden_training.py \
  --profile e2b \
  --model-dir /models/e2b \
  --input-dir /data/full_data_archive_recovered_v9 \
  --output-dir /runs/e2b-archive-v9-lora-fixed \
  --epochs 1 --execute
```

Replace example paths with your actual model/input/output paths and keep any
other intentional settings from your previous command. Use a NEW output
directory. Do not edit the failed run's bound YAML/manifest or use
`--continue-run` after this implementation/config change. Preparation caches
are revalidated; changed code can require preparation again. Preserve old
logs/run directories for comparison. For a short GPU smoke, use a separate
fresh output directory and replace `--epochs 1` with `--steps 20`; do not
reuse the smoke directory for full training.

Defaults remain all CUDA-visible GPUs with the automatic H100 profile,
validation/save every 500 optimizer updates, Golden32 every 1,000 plus final
evaluation, final-only Golden35, generation limit 2,048 and
`/tensorboard/<run-id>/`. This command does not launch HPO or LiteRT export.

## Regression checks

```bash
python -m pytest training/tests/test_lora_target_resolution.py \
  training/tests/test_review_training_recipe.py \
  training/tests/test_e2b_lora_hf_integration.py -q
```

The HF integration tests require Transformers >=5.10.1 and PEFT >=0.19.0;
they skip if these optional packages are absent. They use tiny random models
with per-layer embeddings, both hierarchies and shared/non-shared KV, then
verify exact targets, real PEFT attachment, finite forward/backward gradients
and adapter save/reload. No model downloads, production checkpoint, training
corpus or GPU is needed. Synthetic tests cover wrapped projections and
adversarial modality paths.

Verified locally on 2026-09-14 with Python 3.12.10 / PyTorch 2.13.0 CPU:

- Transformers 5.10.1 + PEFT 0.19.0: all four real-model integration cases passed.
- Transformers 5.16.1 + PEFT 0.20.0: all four real-model integration cases passed.
- Every trainable adapter received a finite gradient; the old selector's
  text-only failure is explicitly reproduced before checking the corrected one.
- The initial targeted resolver/recipe/HF regression run passed all 48 tests.
- Full training regression suite after the final test strengthening:
  761 passed, 1 skipped in 577.16 seconds. Critical Ruff checks and staged
  whitespace/credential-pattern checks passed.

CPU checks do not establish actual H100/DDP throughput or Golden quality.
The training PC must still complete its preflight and run.
