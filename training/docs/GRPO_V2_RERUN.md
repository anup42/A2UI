# GRPO v2: fresh SFT rerun and safety controls

These changes are isolated to QAT LoRA GRPO (E2B/270M). Ordinary SFT/QAT entry
points, fidelity thresholds and official export topology checks remain unchanged.
Software tests are not evidence of GPU memory fit, accuracy or device speed.

## Rerun from the same original SFT checkpoint

Activate the existing reviewed GRPO environment (TRL 0.29.1). Use the previous
GRPO plan to recover original SFT paths and hyperparameters, not GRPO final weights:

```bash
cd /group-volume/k.anup/code/A2UI
git pull --ff-only
OLD_RUN=/absolute/path/to/previous_e2b_grpo_run
NEW_RUN="${OLD_RUN}_grpo_v2_$(date +%Y%m%d_%H%M%S)"
python -u training/scripts/run_qat_grpo_pipeline.py \
  --rerun-from "$OLD_RUN" --output-dir "$NEW_RUN" \
  --defer-native-eval --execute
```

Omit `--execute` to inspect the plan. If the old plan used prepared data,
add `--input-dir /absolute/path/to/original_training_pairs`, containing original
`train.jsonl` and `val.jsonl`. The helper requires a fresh output directory and
always re-prepares raw pairs using the current prompt. It never generates new data.
Original learning rate, budgets, seed and optimizer batch are retained;
rollout microbatch is reset to 1 and optional weight caching is off.
Other CLI overrides are rejected rather than ignored. To change hyperparameters,
use the explicit new-experiment command in [QAT_GRPO_PIPELINE.md](QAT_GRPO_PIPELINE.md).

## Changes and evidence

- **Reward:** `qat_source_fidelity_v2` fixes source-owned inline action links and
  exact labeled entity-field extraction. Supported unambiguous semicolon entity
  records can split into visible Card fields or reorder labeled fields while
  preserving exact names, values, counts, action labels and destinations. Ambiguous
  records keep existing scoring. Swapped links/facts, omissions, hidden copies and
  duplicates remain penalized. At most a 2% preference rewards certified extraction;
  extra Cards alone earn nothing. No validator or admission threshold is relaxed.
- **Scoring consistency:** generated `configs/grpo_reward.yaml` and
  `configs/grpo_metric_weights.yaml` enable this mode explicitly. Reward, fresh SFT
  baseline, Golden selection, final HF and native rescoring share its fingerprint.
  Old canonical scores must be rescored before comparison; canonical SFT defaults
  remain unchanged. Checkpoint provenance binds scorer source/config and reward code.
- **Selection:** GRPO replaces its best checkpoint only after baseline gain,
  source/scorer identity and fidelity guards pass. An ineligible scalar maximum
  cannot overwrite an eligible checkpoint. The 20-update health gate remains active.
- **Rollout memory:** `--rollout-microbatch 1` limits candidates generated at once
  per rank without changing candidate ordering, groups or effective completion batch.
  Different batching may produce different seeded samples. Exact stopping decisions
  are reused to avoid redundant prefix decoding; token/mask semantics stay intact.
- **Optional cache:** `--rollout-qat-weight-cache` is off by default. Enabling it
  requires exact cached/uncached forward parity and adds a temporary eval-only
  fake-quant weight cache. Modes, gradients and saturation monitoring are restored.
  It trades extra memory for potential generation speed; measure on the GPU host.
- **Real runtime probe:** a separate process runs two disposable GRPO Trainer
  optimizer updates with the actual ranks/accumulation and longest prepared prompt
  groups. Updated weights are discarded; production reloads the original SFT adapter.
  Every rank must pass finite checks and the 90%-reserved/10%-free headroom gate.
  Reports bind config/code and actual sampled completion lengths; they do not
  certify every possible maximum-length completion or a whole learning run.
- **Actual configuration:** optimizer, scheduler, dataloader and precision settings
  are applied and reported; unsupported requests fail. TF32 stays off to preserve
  retained-scale merge arithmetic. Inherited warmup cannot exceed a short GRPO run.
- **Coverage:** `training/grpo_dataset_coverage.rank*.json` reports prepared source,
  intent and prompt-length distribution. `grpo_source_exposure.rank*.json` counts
  train/eval exposure separately across the run, not only the first audit batch.
  Unique counts are rank-local and must not be summed as global unique coverage.

## Native export acceptance

Run the exact command in `native_quality_command.json`, or on the Android host:

```bash
python training/scripts/evaluate_official_mobile_native.py \
  --run-dir "$NEW_RUN" --output-dir "${NEW_RUN}_native_quality" \
  --weights-config "$NEW_RUN/configs/grpo_metric_weights.yaml" \
  --serial ACTUAL_ANDROID_SERIAL --execute
```

Use the copied run's local paths after transfer. `native_quality_gate.json` requires
all three cohorts, trained scorer identity and no regression versus the selected
HF checkpoint in score, strict validity, caps or reachability. Successful inference
alone is recorded as execution success, not quality acceptance. A failed quality
gate preserves diagnostics but fails the stage. Deferred testing is never a pass.

## Safe continuation after a downstream failure

```bash
python training/scripts/run_qat_grpo_pipeline.py --continue-run "$NEW_RUN"
python training/scripts/run_qat_grpo_pipeline.py --continue-run "$NEW_RUN" --execute
```

Only newly sealed runs with completed training qualify. The verifier checks every
completed receipt, original input identities, code and scorer hashes; it never
resumes optimization or preparation. Partial pending-stage files/logs are preserved
and listed: archive those exact pending artifacts explicitly before retrying,
without editing completed receipts. Code/config changes require a fresh rerun.
Old unsealed runs must use `--rerun-from`. A plan that deferred native testing
continues to defer it; execute the standalone device command instead of editing it.

Offline QAT DPO and GRPO + SFT rehearsal are now separate opt-in objectives; see
[QAT_POSTTRAIN_OPTIONS.md](QAT_POSTTRAIN_OPTIONS.md) for commands and input contracts.
Frozen-SFT-reference KL and dynamic group resampling remain future experiments.
The default objective stays Dr. GRPO, beta=0, one iteration per fresh rollout. Actual H100/DDP
training, full export and Android execution must be validated on the target hosts.
