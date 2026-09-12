# GPU-PC handoff: E2B and 270M A2UI Express training

**Current handoff:** follow [the end-to-end quickstart](GOLDEN_E2E_QUICKSTART.md)
for `run_golden_training.py`. One command prepares the tracked default Stage 3
data, aligns a single production prompt for train/validation/Golden32/Golden35,
filters all reserved sources, selects visible GPUs, performs model preflight,
trains, and tests selected-best and actual-final checkpoints on **both sets**.
Supply a local complete dense model/tokenizer bundle and a fresh output path;
add `--execute` to start, or omit it for a plan without model loading or writes.

```bash
python training/scripts/run_golden_training.py \
  --profile e2b --model-dir /models/e2b \
  --output-dir /runs/e2b-new --execute
```

Both Golden datasets/manifests and default `dataset_v1` source files are in Git.
The archive development replacement is **32 occurrences / 31 unique cases**;
Golden35 contains **35 unique passing references**. The failed original and
all 15 excluded sources remain reserved. Raw reference pairs are unchanged;
only prepared prompt views and their provenance/hashes are rebuilt. Selection
uses Golden32; Golden35 is final-only. Defaults are 500-update validation/save,
1,000-update Golden32 generation plus final, 2,048 new tokens, and
`/tensorboard/<run-id>/` logging. A small one-epoch corpus may finish before the
periodic boundary, but final testing still runs.

The old handoff below is historical evidence, not the current clone-and-run
recipe. Its manual GPU lists, prompt preparation, 4,096-token generation budget
and request to regenerate the archive's failed Golden slot are superseded.
Official retained-scale QAT, LiteRT multiformat conversion and optional MTP
remain separate workflows with their existing gates; the new dense capability
command does not imply their numerical/export/runtime validation has passed.

## Historical September 5 handoff

This handoff accompanies the September 5, 2026 review fixes. **No training or real checkpoint inference was run on the review PC.** Run the commands below on the GPU PC from the repository root. The small CPU tests use fixtures and mocked runtime objects; their success does not establish GPU memory fit, distributed stability, model quality, or mobile export parity.

## Instructions for the person or Codex agent on the other PC

Use this checkout's finalized changes, including currently uncommitted/new files. Do not copy only the six original merged Python files. Transfer `training/`, `patches/`, the changed Stage3 code/tests, shared quality policy, generated dataset/Android prompt copies, and generated IR manifest together. Preserve the existing dataset schemas, catalog, renderer-semantics code and tokenizer chat templates. The original package extraction and source merge are already complete; do not reapply older package versions over this code.

Copy large artifacts separately: dense HF base model + tokenizer, original prepared train/val JSONL, original Golden32 membership, original Stage2 response/query/asset records needed for regeneration, and any checkpoints selected for diagnosis. Model weights and dataset runs are not guaranteed to be in Git. Retain file hashes, each run's resolved YAML, environment lock and both rank logs. Never report a partial Golden set as Golden32.

Start with clean-data supervised training for **each model**. E2B's default review profile uses rank32 LoRA on supported language attention/MLP linears; `--qv-baseline` provides a rank16 language q/v control. The 270M profile performs full finetuning. These are starting recipes to compare, not measured optima. Establish capability before quantization adaptation or GRPO. Do not relabel QAT-derived BF16 weights as a verified mobile QAT result.

## What changed and why

| Failure found | Implemented behavior |
|---|---|
| Native EOS list overwritten by tokenizer scalar | Preserve actual EOS/end-of-turn IDs at model alignment, Trainer startup and generation. |
| Repeated `</a2ui>` after a complete program | Shared HF stopping scans only generated tokens and ignores quoted markers. Raw and serving-stopped text remain separate. |
| Wrong few-shot demo used as scoring source | Canonical response/final-user binding and metadata/hash checks fail on mismatches. |
| Regex bottom-up conversion lost references or changed literals | Catalog/parser/renderer-aware serialization preserves IDs and exact graph semantics; strict wire schema and complete root reachability are required. |
| Invalid target types and ambiguous duplicate IDs | Quarantine with original row/reason; no ID repair. Stage3 validates compiled wire output on every repair attempt. |
| Prompt/target truncated during SFT | Whole-row overflow fails. Tokenizer-aware preparation quarantines overlength rows without deleting a prefix or target suffix. |
| “Full finetune” actually used LoRA | Explicit full-model path, full checkpoint manifests and full-model resume validation. |
| GPU list/effective batch disagreement | One parent launcher assigns devices and worker count; workers preserve that selection; expected batch is checked. |
| Resume silently reset optimizer or changed data | Require optimizer, scheduler, RNG, checkpoint hashes and identical saved data/recipe contract. |
| QAT merge checks bypassed with a non-QAT YAML | Saved QAT evidence requires original, matching QAT provenance. |
| GRPO zero reward variance/zero updates and collective mismatch | Version/source preflight, consistent prompts, checked termination masks, explicit DDP buffer policy and rolling learning-health gates. GPU smokes remain mandatory. |

The checked SFT loss averages supervised tokens **within each microbatch**, then averages microbatches during accumulation. It is not a global token-weighted objective across variable-length microbatches. This choice is recorded, and its loss/gradient scaling has a numerical CPU regression test.

## 1. Establish the GPU environment

Use Linux for the multi-GPU workflow. Create a separate training environment; install the CUDA PyTorch build appropriate to that host, then the repository overlay. Do not install the review PC's CPU wheel on the GPU PC.

```bash
python3 -m venv .venv-a2ui-train
source .venv-a2ui-train/bin/activate
python -m pip install --upgrade pip
# Install the host's CUDA PyTorch build first.
python -m pip install -r training/requirements-gemma4-qat.txt
python -c "import torch,transformers,peft; print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.device_count(),transformers.__version__,peft.__version__)"
nvidia-smi
python -m pytest training/tests -q
```

The overlay includes the Gemma4 architecture dependency. The resolved environment must support the actual checkpoint, non-reentrant gradient checkpointing and the configured TrainingArguments. Record `python -m pip freeze > requirements-gpu.lock.txt` after the GPU smokes pass; the broad install requirements alone are not a reproducibility lock. Optional GRPO uses a separate reviewed TRL 0.29.1 contract; follow its own runbook.

## 2. Repair source data and freeze Golden32

The CPU sample audit under strict Express **and production wire-schema** validation found:

| Sample inspected | Accepted | Quarantined |
|---|---:|---:|
| First 300 training rows | 252 | 48 |
| First 100 validation rows | 88 | 12 |
| Golden32 | 31 | 1 |

These are sample counts, not estimates for the full corpus. Reasons include scalar `Table.highlightColumns`/`numericColumns`, invalid Text/Icon property types, `entityMedia`, an incomplete repeat and a detached graph. Valid Tabs and other complex references are retained. Inspect the full quarantine report before choosing a training mixture; permanently dropping every difficult case would weaken coverage. Regenerate source targets through Stage3, then rebuild the splits. Training code itself must not generate Stage1/2/3 data.

The invalid Golden reference is query **q_012053**, response **r_012053_01**, UI **u_012053_01**. Its `highlightColumns` must be an array according to the wire contract. The producer prompt now states this and Stage3 retries invalid wire output. Do not manually edit the generated IR.

On the GPU PC, stage the authoritative original Stage2 response and query rows into a **new** `dataset/data/runs/golden32_reference_regen_v2` folder, preserving its asset mappings. Run the existing dataset pipeline, using the configured Vertex Express credentials or an appropriate configured local teacher:

```bash
python dataset/src/main.py --stage 3 \
  --run_id golden32_reference_regen_v2 --model gemini_2_5_pro \
  --genui_batch_size 1 --max_genui_total 1
```

Rebuild prepared Golden32 using the regenerated target and the unchanged 32 source memberships. Version/hash this reference revision, and rebaseline all checkpoints against it. The old 31 accepted records are insufficient. For the full training quarantine, stage the corresponding authoritative Stage2 rows into a separate repair run and regenerate in batches; keep accepted source identities, teacher provenance and quarantine reasons in the resulting manifest.

Use one saved system/few-shot/task scaffold for each experiment. Do not substitute an abbreviated Android prompt at inference. `prompt_scaffolds.json`, prepared `messages`, tokenizer vocabulary and chat-template fingerprints define the contract.

## 3. Prepare complete examples separately for both tokenizers

Replace the following paths with the GPU PC's actual paths. `SOURCE` contains repaired source JSONL, including exactly 32 Golden records. `WORK` must have room for prepared data, full 270M checkpoints, E2B adapters and logs.

```bash
SOURCE=/absolute/path/to/repaired-source
WORK=/absolute/path/to/a2ui-review-runs
E2B=/absolute/path/to/dense-e2b-base-with-tokenizer
G270=/absolute/path/to/gemma-3-270m-it-with-tokenizer

python patches/build_bottom_up_dataset.py \
  --input train="$SOURCE/train.jsonl" --input val="$SOURCE/val.jsonl" \
  --input golden32="$SOURCE/golden32.jsonl" \
  --output-dir "$WORK/prepared-e2b" --ordering root-first \
  --tokenizer "$E2B" --tokenizer-loader pretrained_tokenizer_fast \
  --local-files-only --max-seq-length 4096 --max-input-tokens 4096 \
  --chat-template-kwargs '{"enable_thinking":false}'

python patches/build_bottom_up_dataset.py \
  --input train="$SOURCE/train.jsonl" --input val="$SOURCE/val.jsonl" \
  --input golden32="$SOURCE/golden32.jsonl" \
  --output-dir "$WORK/prepared-270m" --ordering root-first \
  --tokenizer "$G270" --tokenizer-loader auto_tokenizer \
  --local-files-only --max-seq-length 4096 --max-input-tokens 4096 \
  --chat-template-kwargs '{"enable_thinking":false}'
```

Use a new output directory each time: preparation publishes atomically and refuses overwrites. Inspect `manifest.json`, `quarantine.jsonl`, component/reference counts, accepted token maxima and `prompt_scaffolds.json`. Regenerate any quarantined Golden member and rebuild before proceeding. The launch preparer checks train/val and train/Golden overlap by query/source identity and normalized response; it fails rather than changing the split silently.

Root-first is the control. To test bottom-up, prepare the same source files with `--ordering bottom-up` in a new directory. Require identical `accepted_source_rows_sha256` for every split before attributing a score change to ordering. Do not mix prompt scaffolds or repair predictions to improve this comparison.

## 4. Resolve and inspect separate 20-update smoke runs

The following commands **only inspect files, hash artifacts and write configs**. They do not load model weights into a runtime or train. They copy the tokenizer/template settings from preparation, preserve a 16-example effective batch, and require complete Golden32. Two GPUs × microbatch1 × accumulation8 = 16. With three GPUs, choose a divisible batch such as 12 or 24; do not claim it is 16.

```bash
python training/scripts/prepare_review_training.py \
  --profile e2b --model-dir "$E2B" --dataset-dir "$WORK/prepared-e2b" \
  --golden-file "$WORK/prepared-e2b/golden32.jsonl" \
  --output-dir "$WORK/e2b-smoke" --devices 0,1 --effective-batch 16 --steps 20

python training/scripts/prepare_review_training.py \
  --profile 270m --model-dir "$G270" --dataset-dir "$WORK/prepared-270m" \
  --golden-file "$WORK/prepared-270m/golden32.jsonl" \
  --output-dir "$WORK/270m-smoke" --devices 0,1 --effective-batch 16 --steps 20

python training/scripts/launch_review_training.py --config "$WORK/e2b-smoke/training_config.yaml"
python training/scripts/launch_review_training.py --config "$WORK/270m-smoke/training_config.yaml"
```

Review each `preparation_report.json` and resolved YAML. E2B uses the dense QAT-derived `google/gemma-4-E2B-it-qat-q4_0-unquantized` identity as a capability baseline. It is **not** the retained mobile seed. Verify the local bundle is the intended model. The launcher hashes the bundle, prepared data, scaffold and config; edits after preparation require a new plan.

LoRA selectors are resolved against actual supported linear layers before PEFT attaches adapters, including a wrapper's inner `.linear` where needed. Inspect `resolved_lora_targets`, parameter counts and names in training metadata. The E2B baseline must contain only the intended language projections; the 270M run must report all model parameters trainable and `checkpoint_kind=full_model`.

## 5. On the GPU PC only: forward checks, then smoke training

`--execute` is the explicit runtime boundary. `--preflight-only` loads the actual model and performs numeric/greedy forward checks without an optimizer. Omit it only for the intended smoke training.

```bash
python training/scripts/launch_review_training.py --config "$WORK/e2b-smoke/training_config.yaml" --preflight-only --execute
python training/scripts/launch_review_training.py --config "$WORK/270m-smoke/training_config.yaml" --preflight-only --execute

python training/scripts/launch_review_training.py --config "$WORK/e2b-smoke/training_config.yaml" --execute
python training/scripts/launch_review_training.py --config "$WORK/270m-smoke/training_config.yaml" --execute
```

Retain rank logs. Require finite completion loss/gradients, actual parameter updates, no OOM/collective mismatch, exactly 20 optimizer steps and a completed 32-row evaluation. Stop on failure and diagnose; do not hide it with larger NCCL timeouts or disabled checks. A 20-step run tests wiring and learning activity, not final quality. The final outputs are `training/final_adapter` for E2B and `training/final_model` for 270M; best Golden outputs use `training/best_golden_checkpoint` for either kind.

For longer runs, resolve **new directories** with `--steps 1000` for a bounded pilot, then an agreed epoch schedule. Compare the E2B rank16 q/v control (`--qv-baseline`) with rank32 language attention/MLP using the same data, steps, batch and decoding. For 270M, compare full SFT at LR2e-5 against a lower-LR run if dev quality regresses. The current helper sets the documented starting recipe; add a reviewed profile to change learning rate rather than editing a hash-bound prepared YAML.

To resume an interrupted **new-format** run, pass its checkpoint via `--resume` while retaining its original total step/epoch schedule, data and batch. The saved optimizer/scheduler/RNG and checkpoint manifests must be present. Historical checkpoints without the new resume contract fail clearly; do not bypass this by renaming an adapter or providing a replacement non-QAT config. A full-model checkpoint can be an intentional weight-only initialization for a *new* 270M run via `--model-dir`; this resets optimizer progress and is not called resume.

## 6. Evaluate selected checkpoints with the same prompt and decoder

Use the config-aware evaluator below for adapters and full models. It carries saved chat-template kwargs and supports the fixed-row guard. `--checkpoint-kind merged` also denotes an ordinary HF full-model checkpoint; it does not imply that full finetuning used LoRA.

```bash
python training/scripts/evaluate_checkpoint_on_golden.py \
  --config "$WORK/e2b-smoke/training_config.yaml" \
  --checkpoint "$WORK/e2b-smoke/training/best_golden_checkpoint" --checkpoint-kind adapter \
  --split "$WORK/prepared-e2b/golden32.jsonl" --max-rows 32 --required-rows 32 \
  --max-input-tokens 4096 --max-new-tokens 4096 --metric-version v5_4 \
  --output-dir "$WORK/e2b-smoke/recheck" --run-id e2b-smoke --evaluation-name recheck

python training/scripts/evaluate_checkpoint_on_golden.py \
  --config "$WORK/270m-smoke/training_config.yaml" \
  --checkpoint "$WORK/270m-smoke/training/best_golden_checkpoint" --checkpoint-kind merged \
  --split "$WORK/prepared-270m/golden32.jsonl" --max-rows 32 --required-rows 32 \
  --max-input-tokens 4096 --max-new-tokens 4096 --metric-version v5_4 \
  --output-dir "$WORK/270m-smoke/recheck" --run-id 270m-smoke --evaluation-name recheck
```

Select checkpoints by `generation_reward_v5_4_avg` **with failure-category review**. Retain all-row parse/strict validity, duplicate IDs, root reachability, token-limit rate, source/prompt hashes, visible fact coverage, required action/table coverage, raw output and serving-stopped output. v5.4's weights and caps are unchanged. Root reachability below 90% is a score cap; serialized-but-disconnected text is not visible fidelity. Never count ID-repaired results as model capability.

The supplied step20k output rescoring previously moved from 8.806 to 59.798 using correct source plus stop-only handling (27/32 parseable). This was an offline correction of saved output, not new checkpoint generation. It does not predict this code's fresh-generation score. Early checkpoints had many duplicate IDs; later ones still had disconnections, missing actions/tables and omissions. Falling teacher-forced loss alone is insufficient.

Golden32 is a **development** set because it has already influenced checkpoint choice. Freeze a source-disjoint, deduplicated final test set of roughly 500–1,000 cases covering intents, lengths, state/actions, tables, media and complex references. Keep it out of training, data repair decisions, GRPO and checkpoint selection. Proposed quality targets to agree before final testing: at least 99% strict validity, below 1% truncation, complete required graph references and measurable gains in visible facts/actions. Report confidence/sample size; 32 cases cannot establish a reliable rare-failure rate.

## 7. Quantization and deployment follow capability validation

**270M:** after full SFT produces a useful model, prepare a new run with `--profile 270m --qat --model-dir <selected-SFT-full-checkpoint>`. Reprepare the dataset using that checkpoint's saved tokenizer, then run the same 20-step numeric/learning/evaluation gates. This is full-model dynamic W8/FP32-activation QAT at a lower starting LR5e-6. The full checkpoint is already an HF model; do not send it to a LoRA merge function. Evaluate QAT-on using `evaluate_checkpoint_on_golden.py --qat-mode on`, and dense/QAT-off separately. Conversion must use the matching INT8 numerical recipe and compare the exported runtime on identical prompts; a successful export is not a quality result.

**E2B mobile:** use the separate [retained-scale GPU runbook](gemma4_mobile_qat_remote_pc_runbook.md) and its verified packed seed, BF16 reconstruction, qparams sidecars, architecture checks and exact retained-scale gates. Its fixed Golden100 release gate is separate from this Golden32 capability experiment. The historical dense QAT-derived seed cannot be transplanted into the mobile package by changing labels or using generic abs-max quantization. Keep the seed's zero-adapter/numeric/greedy gates and provenance checks intact. MTP is a separate training/export task; nothing here claims a jointly trained MTP assistant.

**Optional GRPO:** only after useful SFT candidate diversity exists, follow [the GRPO GPU preflight/runbook](grpo_gpu_preflight_runbook.md): dependency/source check, single-device 20-update learning smoke, then two-device smoke. Re-evaluate strict dev quality afterward. Zero reward variance and zero updates are a failed learning smoke; changing score caps does not fix them.

**Android/LiteRT:** use the exact saved full scaffold and tokenizer chat template. The LiteRT diagnostic CLI accepts `--prepared-jsonl` and `--query-id`, validates graph/wire output, and retains raw text. Its envelope stop is explicitly **postdecode only**; native cancellation/latency and end-of-turn token handling remain runtime validation work. Compare native HF, exported LiteRT CPU/GPU and actual Android rendering/action behavior. Do not promote either model until required source facts and actions survive this full path.

## Evidence to return after the GPU run

Return the environment lock; prepared/source manifests and quarantine counts; resolved YAML/report; per-rank logs; optimizer-step/parameter-update evidence; checkpoint metadata and hashes; raw/serving prediction JSONL; Golden aggregate/per-case diagnostics; and exported-runtime/device comparisons. Report results separately for E2B and 270M, and distinguish capability baseline, QAT adaptation and deployed model. Keep large weights/checkpoints outside source commits.
