# Codex handoff: student-error replay and DPO+SFT trials

Branch: `new_ir_changes_20260331`, repository `anup42/A2UI`.
Base reviewed: `1b493f4d317fc8e56426e179db94fce257a5fc9a`.
Read `AGENTS.md`, this file, and `GRPO_RUN_REVIEW_20261010.md` before execution.

## Status and scope

This is an **opt-in experimental workflow**, not a measured improvement or a production-qualified exporter change. The review environment ran 58 CPU tests successfully; one real-v5.4 integration test was skipped because the complete repository scorer was not mounted. Compilation and CLI help checks passed. Several repository IO boundaries are explicitly mocked in the CPU tests. Real Transformers/PEFT/QAT training, CUDA, DDP, complete repository regression tests, and Android/native inference were NOT executed during review.

The existing Muse Glimmer generator, acceptance policy, regeneration recipes, SFT launcher, GRPO launcher, MTP assistant work and renderer are unchanged. This workflow adds:

1. A greedy QAT-on probe of **training-only** sources, using the existing evaluator.
2. Current-v5.4 rescoring of the existing verified reference and the student's actual output on exactly the same source.
3. Bounded replay of original teacher targets for failures, plus eligible preferred/rejected pairs. No teacher calls and no IR repair occur here.
4. Matched finite-run A/B/C exposure schedules and an experimental QAT-LoRA trainer.

Hard-target teacher SFT is already present in the original pipeline. The additions here are checkpoint-specific selection and explicit negative-sequence supervision. Soft-target KD, on-policy distillation, new synthetic generation, and automatic multi-round continuation are not implemented. This first round must start from a verified QAT LoRA **SFT** adapter; do not feed it a GRPO final adapter or this experiment's unqualified output.

## Safety and data boundaries

Use fresh output directories, preserve all original artifacts, inspect `git status`, and pull safely with `git pull --ff-only`. Never reset unrelated work or force-push. No dataset/checkpoint outputs should be committed.

Supported training families in v1: E2B retained-mobile QAT LoRA and Gemma 3 270M QAT LoRA. Full-QAT 1B and sharded FSDP/DeepSpeed are deliberately rejected. The train runner uses one example per device and configurable accumulation; the probe is single-process. CUDA BF16 is required for execution. Both entry points support planning without model loading, and the probe/train paths set scoped HF offline flags: no implicit downloads.

Use the corrected frozen prepared bundle, not the old ~5% retained subset. The tool verifies split/scaffold hashes and requires at least 50% train/val preparation retention. It reserves all nontrain splits, the canonical Golden32/Golden35/Bixby50 cohorts, replaced-source exclusions recognized by the existing helper, and transitive source families including augmentation donor identities. A contaminated source bundle stops the run rather than silently filtering its contents.

Original prepared files are never rewritten or copied into a differently labelled dataset. Replays are indices pointing to original rows. Chosen references are rechecked using the current admission policy. Accepted alternative UIs are not negatives even when their scalar score is lower. Review-only failures need a quality gap (default 0.05 on the 0–1 scale). Truncated, unknown-stop, empty and identical candidates are not preference negatives; they can still motivate ordinary replay when the reference passes. Candidates needing human review remain diagnostic evidence, not a claim of proven semantic error.

## Tests before GPU work

Use the existing dedicated training environment; do not upgrade Torch, Transformers, PEFT, or TRL as part of these experiments. Check `PYTHONPATH` for stale shadowing. From the complete repository root:

```sh
python -m pytest -q training/tests/test_student_feedback.py
python -m pytest -q training/tests/test*grpo*.py
python -m pytest -q training/tests/test_semantic_augmentation.py
python -m compileall -q training/src/ir_training training/scripts
python training/scripts/build_student_feedback.py --help
python training/scripts/train_student_feedback.py --help
```

On the complete host, the real-v5.4 scorer test must RUN, not be accepted as skipped. Also run the preparation/reference-binding and model-loading tests used by the local production recipe. Investigate incompatibilities; do not bypass the gates.

## 1. Plan, then probe the starting student

Set the original checkpoint-bound SFT config, original SFT adapter, and corrected prepared bundle to their real absolute host paths. The prepared bundle must include its frozen `golden32.jsonl` and supporting manifests for the training stage. Keep the original SFT config unchanged.

```sh
export PREPARED=/absolute/path/to/checked/prepared
export SFT_CONFIG=/absolute/path/to/original/resolved_sft.yaml
export SFT_CHECKPOINT=/absolute/path/to/original/sft_adapter
export WORK=/absolute/path/to/new/student_feedback_round1

python training/scripts/build_student_feedback.py probe \
  --prepared-dir "$PREPARED" --sft-config "$SFT_CONFIG" \
  --sft-checkpoint "$SFT_CHECKPOINT" --output-dir "$WORK/probe" \
  --max-rows 256 --seed 42 --max-new-tokens 2048
```

The default is planning only. Inspect family/intent/length coverage, identities and budgets. Then repeat the same command with `--execute`, selecting one GPU with `CUDA_VISIBLE_DEVICES`. It generates greedily with QAT enabled through the shared evaluator, keeps raw and serving output separately, and writes `receipt.json` only after complete generation and hash checks. Inference failure leaves diagnostic partial files, not a successful receipt. Do not hand-edit or complete a partial receipt.

The probe selects at most one row per transitive source family, balanced across intent and length buckets. A 256-row probe is a bounded first collection, not a population-wide error census. For a larger pilot, create a **new** probe directory with a larger declared row budget if the hard pool is too small; do not append new predictions to a sealed probe or sample Golden failures instead.

## 2. Build feedback artifacts on CPU

```sh
python training/scripts/build_student_feedback.py build \
  --probe-dir "$WORK/probe" --output-dir "$WORK/feedback" \
  --max-extra-token-fraction 0.10 --max-replays-per-family 1 \
  --min-quality-gap 0.05 --seed 42
```

Inspect:

- `report.json`: reference eligibility, hard-case and pair counts, failure categories, full-sequence token budgets.
- `diagnostics.jsonl`: original row index/hash, prompt hash, student candidate hash/stop, both scorer decisions, pair exclusions.
- `preferences.jsonl`: exact same-source chosen/rejected strings, reasons, score gap and pair identity. Training text remains in the original masked/reference namespace; only scoring restores references.
- `replay.json`: original training indices; no generated targets.
- `binding.json` and `manifest.json`: source/probe/holdout hashes and publication marker.

Rebuilding/consuming feedback rechecks the original artifacts and rescores the small probe. Code/admission drift cannot silently bless old preferences. Noisy reference rows are not “fixed” by this workflow; send those sources through the existing dataset generation/review process as a separate versioned data change.

## 3. GPU qualification before a comparison

Start on one GPU with separate fresh smoke directories for `sft`, `replay`, and `dpo_sft`. Use a short explicit step budget (for example `--max-steps 4 --eval-steps 4`) and `--execute` only after reviewing the plan. The shared retained-scale loading, source-lineage and forward/greedy/backward numeric gates run before optimization.

Verify actual near-budget chosen/rejected rows fit VRAM, references are finite, and the first paired batch has nonzero finite gradients and measured adapter updates. Baseline Golden32 runs with QAT on before the first optimizer step, with RNG restoration. Check it agrees with a separately run source SFT baseline under identical settings. The standard chat template supplies real assistant turn endings; the code rejects prefix/tokenization mismatches and never truncates a training sequence or fabricates an Express closing tag.

Then repeat a real two-/four-rank smoke in fresh directories. Verify all ranks reach baseline evaluation, the first optimizer step, periodic validation, Golden selection, checkpoint saving and train end without mismatched collectives. The implementation uses DDP controls and fixed-shape health gathers, but that path was NOT exercised in the CPU review. Keep all per-rank preflight/health files and CUDA error logs. Do not consider a plan-only invocation or unit test a GPU qualification.

## 4. Matched A/B/C pilot

All arms start independently from the SAME original verified SFT adapter. Do not chain A into B into C. Use the same feedback bundle, seeds, global example budget, learning rate and evaluation settings.

- `sft`: ordinary completion-only SFT control.
- `replay`: a shared control schedule with a bounded set of hard-example replacements.
- `dpo_sft`: EXACTLY the replay schedule; add DPO only on paired replay occurrences. Unpaired rows keep their supervised loss.

A single-device plan uses WORLD_SIZE=1. To inspect the four-device budget without execution:

```sh
WORLD_SIZE=4 python training/scripts/train_student_feedback.py \
  --feedback-dir "$WORK/feedback" --output-dir "$WORK/pilot_replay" \
  --arm replay --max-steps 100 --accumulation 16 \
  --learning-rate 5e-7 --eval-steps 25 --seed 42
```

After qualifying the real DDP path, the corresponding launch is:

```sh
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  training/scripts/train_student_feedback.py \
  --feedback-dir "$WORK/feedback" --output-dir "$WORK/pilot_replay" \
  --arm replay --max-steps 100 --accumulation 16 \
  --learning-rate 5e-7 --eval-steps 25 --seed 42 --execute
```

Repeat separately with `--arm sft` / `--arm dpo_sft` and different output directories. Defaults `beta=0.1`, `sft_weight=1`, learning rate `5e-7` and 100 steps are pilot choices, not tuned optima. DPO reference log probabilities are computed on the initial QAT SFT model BEFORE optimization, without disabling the trained adapter. They are stored with pair/tokenization/source bindings in rank preflight reports; no second frozen model or persistent reusable log-probability cache is required.

The control and replay schedules have identical example counts and optimizer steps. Hard replacements have a full-sequence-token cap relative to control and at most three hard exposures per family by default (`--max-hard-exposures`). Scarce hard examples may underfill the cap; inspect the actual schedule rather than assuming a 10% hard mixture. Do not weaken family caps just to force a desired ratio.

**Equal steps do not mean equal tokens or compute.** `exposure_schedule.json` reports planned full-sequence token totals; final metadata reports actual chosen/rejected completion tokens, processed sequence tokens, and paired examples gathered across ranks. DPO adds negative forward/backward work. SFT loss is per-example mean completion NLL; microbatch is fixed to one so accumulation semantics are explicit. Report actual token/compute budgets in comparisons and add a compute-matched control before making efficiency claims.

Periodic validation loss uses the first 32 fixed validation rows, not a full stochastic rollout evaluation. Full Golden32 is used for greedy development selection. Golden35/Bixby50 are excluded from feedback; keep their final evaluation once the protocol/model is selected, and reserve a new untouched test cohort for stronger claims after prior benchmark inspection.

## Artifacts and promotion limits

The runner writes `feedback_plan.json`, `exposure_schedule.json`, `feedback_preflight.rank*.json`, `feedback_health.rank*.jsonl`, a fresh `baseline_golden32/`, periodic `golden32/`, ordinary checkpoints, `best_golden_checkpoint/`, `final_adapter/` and final `feedback_training_metadata.json`.

Health logs distinguish last-microbatch loss details from rank-min gradient/update measurements. They are not a GRPO reward/variance report. Final training counts exclude reference precomputation and evaluation; account for those separately in wall-clock measurements.

**Do not feed these experimental checkpoints into automatic production export.** The runner deliberately does not write the production `training_metadata.json` or impersonate an approved SFT/GRPO run. It reuses the GRPO module's verified SFT loading/numeric functions internally, but its objective is not GRPO. A production integration must explicitly recognize and verify this new objective, preserve the official quantization topology, bind the selected adapter and its baseline comparison, and pass native evaluation. Never rename metadata files or relabel the method to bypass that review.

Report raw validity, root reachability, source-content/numeric/table/link preservation, repetition, native quality, output length, latency and memory. Better training loss or preference accuracy is not proof of a better renderer-facing model. Keep original SFT and the ordinary-SFT control as baselines. No post-fix gain is claimed by this commit.
