# Codex handoff: next E2B GRPO experiment

Work only on `new_ir_changes_20260331` in `anup42/A2UI`. Read `AGENTS.md` and `training/docs/GRPO_RUN_REVIEW_20261010.md` first. These changes are CPU-tested experiment controls; they are NOT a GPU-qualified training result. Preserve the original run, SFT adapter/config, tokenizer, frozen evaluation cohorts, and all unrelated working-tree changes.

## 1. Sync and inspect

Check `git status` and `git branch --show-current`; use `git pull --ff-only` only when safe. Do not reset, force-push, regenerate held-out cases, or overwrite the original run. The inspected original run is `e2b_v11_grpo_20261009_f9e1e784_full200_v1`, starting from the verified step-20,000 QAT LoRA SFT adapter. Use original SFT weights, NOT the GRPO final adapter. Keep the SFT checkpoint-bound configuration unchanged.

Run in the dedicated GRPO environment with the existing reviewed `trl==0.29.1` stack; do not upgrade dependencies as part of this experiment. Check for stale PYTHONPATH overrides. Run:

```sh
python -m pytest -q training/tests/test_grpo_experiment.py
python -m pytest -q training/tests/test*grpo*.py
python -m compileall -q training/src/ir_training training/scripts
```

The new tests passed in a partial CPU review checkout. Run the existing tests in this complete checkout and investigate regressions before training. The pipeline's configure stage runs the real `train_qat_grpo.py --dependency-preflight-only`; do not skip it or the retained-scale numeric/export gates.

## 2. Audit length coverage before committing a long GPU run

Use the raw V11 source input from the original plan, not the old 10,149-row length-filtered subset. Default revised budgets are:

- `max_seq_length=6144`: prompt plus REFERENCE completion used in preparation/preflight.
- `max_input_tokens=5120`: actual GRPO/Golden prompt bound.
- `max_new_tokens=2048`: unchanged primary generated-answer cap.
- E2B context bound: 5120+2048+1=7169, not 6144+2048+1.

Run preparation and inspect `data_audit.json -> prepared_counts`, per-intent/source coverage, and actual prompt/reference length percentiles. Do not truncate sources or reference targets. The new default requires at least 50% train and validation retention. If it fails, preserve the audit and diagnose token accounting/scaffold overhead; do not lower the guard to 5% merely to make the job run. A deliberately narrow curriculum needs a separate documented experiment. The archive omitted raw rejected rows, so 6144-token retention is not yet known.

Before launching the full pilot, validate an actual long retained row and near-budget generation on the four H100s: finite QAT logits/loss/gradients, unchanged seed identity, non-reentrant checkpointing restored, and safe peak VRAM. Increasing reference lengths can increase memory even though the context is legal. If needed, adjust batch/memory settings explicitly while preserving complete GRPO groups and update the planned experiment; never weaken numeric gates.

## 3. Build a fresh pilot plan from the original paths

The existing original `qat_grpo_plan.json` contains the host-specific paths. Reuse those identities instead of guessing paths or editing the checkpoint-bound SFT config. Example from repo root, with OLD_RUN and NEW_RUN set to absolute host directories; NEW_RUN must not exist:

```sh
export OLD_RUN=/absolute/path/to/e2b_v11_grpo_20261009_f9e1e784_full200_v1
export NEW_RUN=/absolute/path/to/e2b_v11_grpo_coverage_pilot_v2
export PYTHONPATH="$PWD/training/src:$PWD/dataset/src${PYTHONPATH:+:$PYTHONPATH}"
python - <<'PY'
import json, os
from dataclasses import fields
from pathlib import Path
from ir_training.pipeline.qat_grpo import QATGRPOOptions, run_pipeline

old = json.loads((Path(os.environ['OLD_RUN']) / 'qat_grpo_plan.json').read_text())['options']
known = {field.name for field in fields(QATGRPOOptions)}
values = {name: value for name, value in old.items() if name in known}
paths = {'model_dir', 'sft_config', 'sft_checkpoint', 'output_dir', 'official_litertlm',
         'exporter_python', 'input_dir', 'prepared_input_dir', 'source_safetensors'}
for name in paths:
    if values.get(name) is not None:
        values[name] = Path(values[name])
# The original raw source, not its already-filtered prepared output, is required.
assert values.get('input_dir') is not None and values.get('prepared_input_dir') is None
values.update(output_dir=Path(os.environ['NEW_RUN']), max_steps=100,
              learning_rate=5e-7, max_seq_length=6144, max_input_tokens=5120,
              max_new_tokens=2048, golden_every_steps=25, validation_max_rows=32,
              min_retained_fraction=0.5, early_stopping_patience=3,
              quality_min_delta=0.5, audit_every_steps=25, defer_native_eval=True)
options = QATGRPOOptions(**values)
print(json.dumps(run_pipeline(options, execute=False), indent=2))
# After the tests, preparation review and long-row GPU checks above pass,
# use run_pipeline(options, execute=True) in a saved launch script.
PY
```

The 5e-7 pilot is a conservative hypothesis, not a proven optimum. Keep a matched 1e-6 control with the same new prepared data/seed/budgets to separate coverage effects from LR effects. Do not count Golden35/Bixby50 as untouched confirmation after repeatedly using their failure cases for development; reserve fresh sources for the final research claim.

## 4. Required GPU callback/collective smoke

Before any long run, exercise the new callbacks under the real four-rank stack. Confirm that:

1. `training/sft_starting_baseline/step_000000000/aggregate_metrics.json` is generated with QAT on, the same Golden32 split/prompt/stop/cap, and zero optimizer updates. It must not replace or masquerade as `best_golden_checkpoint`.
2. Every rank completes the baseline, an ordinary validation/Golden event, a checkpoint save, and train-end/finalize without a collective mismatch. Early-stop decisions are identical on all ranks. The existing 20-update optimizer-health window must pass before declaring a smoke successful.
3. `grpo_validation_subset.rank*.json` agrees across ranks, records the full val hash, and preserves full val files. `validation_max_rows=0` means full validation, not disabled validation.
4. `grpo_periodic_rewards.rank*.jsonl` includes complete local batches at steps 0/25/50/etc., separately tagged train/eval, with `breakdown.qat_reward`. Group candidates from all ranks using source/prompt/step; do not infer late behavior from the first 256 candidates alone.
5. `grpo_quality_history.jsonl` and `grpo_quality_gate.json` contain finite unique-source metrics and correctly bind config/adapter/aggregate/baseline hashes. The gate requires >0.5 quality points over fresh SFT and no loss of strict validity, CAP, or full root reachability. Patience counts eligible improvements, not healthy gradients. Patience 0 disables early stopping but does not bypass promotion.

Use a separate smoke output directory. A quality gate rejection is a valid experiment outcome, not a reason to remove the check. The code still retains selected/final artifacts for diagnosis; automatic pipeline merge/export stops. The current scalar-best selector is intentionally unchanged, so a structurally regressed scalar-best candidate is conservatively rejected instead of silently swapped for a lower-scoring checkpoint.

## 5. Diagnose remaining model errors without score inflation

Keep `beta=0`, loss `dr_grpo`, reward `qat_source_fidelity_v1`, temperature 0.8, top_p=1, top_k=0, repetition_penalty=1, and one optimization iteration per fresh generation batch. Do not introduce top-k/grammar filtering unless likelihoods and importance correction are reviewed together. Do not enable KL by simply disabling the active adapter: that is not a frozen QAT SFT reference.

Inspect late-run strict-valid fraction, fidelity/precision/number/table penalties, actual within-prompt variance, truncation, and first-token errors. Compare the SFT and candidate policy with QAT-on/off using byte-identical prompts. Investigate the stray `Hungary` prefix/early EOS; do not blacklist that word or strip it from scored output. Check duplicate IDs, unresolved children, disconnected roots, and repetitive tails independently of schema validity.

Keep primary results at 2048 output tokens. As a separately labeled diagnostic, rerun BOTH baseline and candidate at 2816 tokens (5120+2816+1=7937) to distinguish recoverable length failures from prolonged loops. Do not mix different caps in a single improvement table. Any source/table-reference format change requires a new aligned preparation/SFT/renderer contract, not hand-edited GRPO targets.

## 6. Completion and reporting

At the end, report source retention/distribution, runtime/dependency identities, best-versus-final curves, fresh-SFT paired per-source gains, structural guards, early-stop status, sampled-audit coverage, and peak-memory/timing observations. Run complete validation once for the selected policy as appropriate; the small periodic RL subset is not the complete validation result. Run matched SFT comparisons on final cohorts, with the same prompt/render/scoring budgets.

If promotion passes, retain canonical merge/export checks and then execute the emitted native Android QA command on the Android host. `--defer-native-eval` deliberately leaves native testing pending; never label it passed. Archive only small manifests/reports/log summaries in git, not checkpoints, prepared datasets, merged weights, or the multi-gigabyte LiteRT file. Commit any additional verified fixes only on the requested branch with tests and a clear separation between implementation validation and measured quality gains.
