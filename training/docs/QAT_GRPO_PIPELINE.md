# QAT LoRA GRPO with official LiteRT-LM export

`training/scripts/run_qat_grpo_pipeline.py` continues a verified **QAT LoRA SFT
adapter** with GRPO for Gemma 4 E2B or Gemma 3 270M. It reuses the existing data
preparation, model adapters, QAT wrappers, checkpoint selection, LoRA merge,
official-format exporters, and Android LiteRT-LM quality evaluator. The ordinary
SFT entry points and canonical v5.4 evaluation metric retain their behavior.

This adds a training workflow, not a demonstrated accuracy gain. The reward is
an uncalibrated engineering objective. CPU and mocked tests establish software
contracts; actual GPU training, package conversion, and Android results are
separate evidence. Compare quality against the starting SFT checkpoint before
accepting a trained model.

## Required inputs

- `--model-dir`: the original local base used by the SFT adapter. For E2B this
  is the verified retained-mobile BF16 reconstruction, including its seed
  manifest and retained qparams. Do not substitute the merged SFT model.
- `--sft-config`: the exact original resolved training YAML whose bytes match
  `training_config_sha256` in the SFT adapter's metadata. It must describe
  `training.method: qat_lora_sft` and bind an absolute local `model_source`.
  A checked-in template, a regenerated config,
  or another run's YAML is not interchangeable with that file.
- `--sft-checkpoint`: the verified selected SFT adapter directory, including
  `adapter_config.json`, `adapter_model.safetensors`, and
  `training_metadata.json`. Its original training, QAT, and numeric provenance
  must verify. Full-parameter SFT checkpoints are not supported by this route.
- Exactly one of `--input-dir` (training-pair `train.jsonl` and
  `val.jsonl`) or `--prepared-input-dir` (an existing sealed prepared bundle
  with the required tokenizer, prompt, source-disjoint splits, and benchmarks).
  Each input training pair needs its original source identity, response text,
  strict Express `completion`, and structured chat `messages` ending in the
  matching task-user/target-assistant turns. A raw dataset run containing only
  `responses.jsonl` and `genui.jsonl` is not an `--input-dir`; materialize its
  training pairs with the existing training preparation workflow first.
  Preparation validates existing data; it does not generate Stage 1/2/3 data.
- `--official-litertlm`: the pinned released package for the selected family.
  The pipeline verifies its SHA-256. E2B additionally needs the pinned packed
  mobile source `--source-safetensors` used to establish the retained scales.
- `--exporter-python`: an absolute path to the existing separate exporter
  environment's Python executable. Keep the training and exporter environments
  separate; see `requirements-edge-export-tested.txt` and the existing
  [official-mobile runbook](OFFICIAL_MOBILE_QAT_PIPELINE.md).

Use the GPU host's working PyTorch/CUDA environment. The shared GRPO integration
requires **TRL 0.29.1** and checks the installed source API, stop masks, and import
provenance. `requirements-training.txt` has a lower bound, so explicitly pin
`trl==0.29.1` in this environment. E2B also needs the Gemma 4 architecture support
described by `requirements-gemma4-qat.txt`. Dependency checks do not download a
model; every model and export input must already exist locally. Use a separate
GRPO training environment so pinning TRL does not change a working SFT environment.
The configure stage verifies the starting adapter's actual source lineage and
runs the dependency-only check before the expensive E2B no-op export gate.
Training and preparation both load the tokenizer from the original local base.

## Commands

Run from the repository root in the training environment. Paths below are
examples and must identify the real, matching SFT artifacts. Start with a fresh
20-update smoke directory. Omit `--execute` to inspect a read-only plan first.

E2B, export on the GPU host and explicitly defer Android evaluation:

```bash
python training/scripts/run_qat_grpo_pipeline.py \
  --family e2b \
  --model-dir /models/e2b-retained-mobile-seed \
  --sft-config /runs/e2b-sft/configs/original-resolved-training.yaml \
  --sft-checkpoint /runs/e2b-sft/training/best_golden_checkpoint \
  --input-dir /data/completed-e2b-inputs \
  --official-litertlm /models/official-e2b-mobile.litertlm \
  --source-safetensors /models/official-e2b-mobile/model.safetensors \
  --exporter-python /venvs/litert-export/bin/python \
  --output-dir /runs/e2b-qat-grpo-smoke \
  --devices auto --max-steps 20 --golden-every-steps 20 \
  --num-generations 4 --microbatch 1 --effective-batch 32 \
  --max-seq-length 4096 --max-input-tokens 5120 --max-new-tokens 2048 \
  --defer-native-eval --execute
```

270M, using that family's own verified QAT LoRA SFT adapter and official package:

```bash
python training/scripts/run_qat_grpo_pipeline.py \
  --family 270m \
  --model-dir /models/gemma-3-270m-it \
  --sft-config /runs/270m-sft/configs/original-resolved-training.yaml \
  --sft-checkpoint /runs/270m-sft/training/best_golden_checkpoint \
  --input-dir /data/completed-270m-inputs \
  --official-litertlm /models/official-gemma-3-270m-it-int8.litertlm \
  --exporter-python /venvs/litert-export/bin/python \
  --output-dir /runs/270m-qat-grpo-smoke \
  --devices auto --max-steps 20 --golden-every-steps 20 \
  --num-generations 4 --microbatch 1 --effective-batch 32 \
  --max-seq-length 4096 --max-input-tokens 5120 --max-new-tokens 2048 \
  --defer-native-eval --execute
```

To reuse a sealed prepared bundle, replace the `--input-dir` argument with
`--prepared-input-dir /data/matching-prepared-bundle`; never supply both.
After a successful smoke and review of its quality, use a **new output
directory** with `--max-steps 200 --golden-every-steps 50`. These are conservative
starting settings, not tuned optima. The default learning rate is `1e-6`; adjust
it explicitly with `--learning-rate` only as part of a recorded experiment.

To complete Android evaluation on the same host, replace
`--defer-native-eval` with `--serial ACTUAL_ANDROID_SERIAL` and optionally
`--adb /absolute/path/to/adb`. A device with the existing compatible app and
instrumentation APKs already installed is required; this pipeline does not build
or install them. Omitting both an explicit serial and the defer flag refuses
execution.

After deferred export, the run has status `awaiting_native_evaluation`, with
`export_passed: true` and `native_test_passed: false` in `results.json`.
`native_quality_command.json` contains the exact shared evaluator command.
On the Android host, preserve the bound run paths and artifacts and execute:

```bash
python training/scripts/evaluate_official_mobile_native.py \
  --run-dir /runs/e2b-qat-grpo-smoke \
  --output-dir /runs/e2b-qat-grpo-smoke_native_quality \
  --adb /absolute/path/to/adb --serial ACTUAL_ANDROID_SERIAL --execute
```

Use the corresponding 270M run paths for 270M. The evaluation directory must be
fresh and disjoint from the run. The separate native report establishes device
completion; an earlier GPU-host `results.json` remains a record of its deferred
state. Do not claim a native pass from a successful export alone.

## Training and selection contract

The original base and quantization contract are retained. Only the existing
LoRA parameters are trainable. E2B keeps the verified W2/W4 projection scope and
retained A8 simulation; 270M keeps its SFT weight-only dynamic INT8 contract.
QAT remains active during rollout generation, old/current log-probability
calculation, optimization, and checkpoint evaluation. Dropout is disabled so
these policy calculations use consistent behavior.

The workflow uses HF generation on one GPU or DDP on the selected GPUs. It
requires native BF16, non-reentrant checkpointing, and disabled DDP buffer
broadcasts. DeepSpeed, FSDP, and vLLM are unsupported by this QAT GRPO route.
`beta=0` is deliberate: turning off the active adapter would not create the
correct QAT reference policy. Adding KL regularization would require a separate
verified reference implementation.

The effective batch of 32 counts **generated completions**, not 32 source
prompts. With four completions per prompt this is eight prompt groups per update.
The launcher derives accumulation from selected GPU count and microbatch. The
current recipe uses `dr_grpo`, batch reward scaling, and one optimization
iteration per sampled rollout. Golden checkpoint selection cannot begin before
the first 20-update health window (`--golden-every-steps` must be at least 20).

Training preparation uses a 4096-token sequence ceiling; the GRPO prompt ceiling
is also 4096. Golden/native evaluation has an independent 5120-token input
ceiling, with 2048 output tokens by default. The runtime validates the actual
prompt lengths, completion budget, and extra masked terminal token against the
model context. It refuses prompt truncation. A length rejection should be fixed
through a new compatible preparation/configuration, not by trimming raw targets.

Golden32 is development checkpoint selection only. The selected adapter is
then evaluated with QAT on Golden32, Golden35, and Bixby50 before merge/export.
Golden35 and Bixby50 are final holdouts and never enter reward optimization or
checkpoint selection. Repeatedly tuning against their final scores would turn
them into development data; keep final evaluation separate from recipe tuning.

## Reward and learning evidence

The `qat_source_fidelity_v1` training reward wraps existing grouped v5.4 scoring.
It uses the same renderer-visible evidence, source contracts, reference
restoration, matching, and canonical caps as the established evaluator. It does
not change canonical v5.4 scores and does not compare candidates with the SFT
reference completion. Alternative faithful UIs remain eligible.

Malformed raw Express, wrappers, unsupported components, missing references,
and cycles receive `-1`. For valid candidates, quality is the product of:

- Source-applicable content/order/numeric/table/heading/action/media fidelity.
- Existing canonical caps, retaining real evidence differences among capped
  candidates rather than adding artificial variance.
- Precision and semantic non-duplication, penalizing invented facts/actions and
  repeated content rather than rewarding extra text or extra components.
- A critical number/table/action/media preservation factor, plus bounded
  semantic-role and canonical-structure contributions.

The final reward is `2 * quality - 1`, bounded in `[-1, 1]`. There is no length,
novelty, random, or candidate-rank bonus. Hidden state and unused source copies
cannot substitute for visible source content. Canonical quality is retained in
the audit separately from the optimized reward. These checks reduce tested
forms of reward gaming; they do not prove complete semantic understanding.

Inspect `training/grpo_rewards.rank*.jsonl` for candidate text, restored text,
per-component scores, failure reasons, canonical score, and reward policy SHA.
Inspect `training/grpo_rollouts.rank*.jsonl` for sampled tokens, serving stops,
and loss masks. Audits are bounded to 256 candidates per rank by the current
runtime default. `grpo_dependencies.rank*.json`, `grpo_preflight.rank*.json`, and
`grpo_health.rank*.jsonl` retain dependency, numeric, and optimizer evidence.

The rolling 20-update health gate requires genuine reward variation in more
than 25% of groups, clipping at most 5%, finite gradients and parameters, and
nonzero gradients and sampled adapter updates in more than 25% of updates.
A smoke must pass a complete window. All-invalid or identical-reward groups can
produce no useful GRPO signal; the correct response is to inspect the SFT seed,
data, candidates, and reward reasons. Successful optimizer updates alone do not
demonstrate better held-out quality.

## Export reuse and run evidence

E2B uses `build_gemma4_retained_scale_litertlm.py` with the existing
`--qat-compatible-weights` path and a pretraining no-op export check. The
released retained W2/W4/W8/A8 package structure and frozen/MTP sections remain
subject to the existing exporter gates. MTP is neither trained nor enabled.

270M uses the existing `build_checkpoint_official_topology.py` through the
Gemma 270M deployment planner. It exports into its own official INT8 topology;
it does not reuse E2B's retained-scale or MTP package assumptions.

Both routes share `merge_lora_adapter`, canonical exporter validation, and
`evaluate_official_mobile_native.py` for Android LiteRT-LM inference. The shared
native evaluator checks the locked selected artifact and compares its output
with the saved HF evaluations. A merged BF16 directory is an intermediate,
not the deployable LiteRT-LM artifact.

`qat_grpo_plan.json` records the immutable plan; `qat_grpo_manifest.json` records
stage state and hash-bound receipts. Each run requires a fresh output directory,
disjoint from its base, adapter, and inputs. Source mutations, stale receipts,
failed export certification, or incomplete native evaluation prevent a complete
result. Preserve the entire run and original SFT provenance for review.

Focused CPU checks:

```bash
python -m pytest training/tests/test_qat_grpo_reward.py \
  training/tests/test_qat_grpo_pipeline.py \
  training/tests/test_qat_grpo_training.py \
  training/tests/test_qat_grpo_export.py \
  training/tests/test_qat_grpo_native.py -q
```
