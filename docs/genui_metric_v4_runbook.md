# GenUI metric v4 rollout runbook

GenUI Representation Quality v4.0.0 is a source-conditioned, renderer-aware engineering score. The 0-100 value is **uncalibrated**: it is useful for controlled comparisons and regression detection, but it is not an equal-interval quality percentage and must not be presented as one until human calibration is approved.

## Runtime modes and outputs

Set `evaluation.metric_version` in `dataset/configs/run.yaml` to one of:

- `dual` (default for the migration release): write v4 and the immutable legacy score.
- `v4`: write only v4 evaluation outputs.
- `legacy`: roll reporting back to the old structural-richness score.

During dual-write, the legacy headline is `legacy_structural_richness_score`. The compatibility alias `overall_score` remains until 2026-10-01. Do not rewrite historical run files when the alias is removed.

Each Stage 3 row records the candidate-independent `expected_ui_contract`, its version/source/cache status, the full `genui_quality_v4` breakdown, dimension scores, active caps, metric version, and `renderer_check_result`. Run aggregates contain the per-sample score distribution, bootstrap confidence interval, intent macro/micro means, dimension distributions, cap rates, failure rates, and floor/ceiling rates. Latency, tokens, bytes, and cost remain diagnostics outside the quality score.

Stage 3 refreshes partial aggregates after the first new row and every 250 new rows, then always at clean completion. Set `GENUI_STAGE3_AGGREGATE_EVERY` when a different progress cadence is needed; this avoids repeatedly bootstrapping the full growing run after every record.

## Source contracts

The resolver prefers a schema-valid persisted contract. A contract marked `human benchmark` remains distinguishable from an ordinary persisted contract. Otherwise the deterministic extractor handles headings, Markdown tables, explicit `Action:`/`Media:` directives, links, exact values, code, formulas, and email structure. Explicitly empty requirement lists are authoritative.

Fallback contracts are cached under the run artifacts by SHA-256 of normalized source text plus extractor version. A source or extractor-version change therefore cannot silently reuse an incompatible contract.

## Android-native renderer checks

The renderer-smoke atomic accepts only an Android/native/device adapter result; generic browser screenshots do not count. Capture a run with:

```powershell
python dataset/scripts/capture_android_run_screenshots.py --run_id <run-id>
```

The capture script publishes `native_render_checks.jsonl` beside `genui.jsonl`. Every selected record is marked `attempted: true`; capture failures are `ok: false` and activate the render-failure cap rather than becoming N/A. Re-running the normal aggregate-only path loads this manifest after generic render logs, so native results take precedence.

## Historical shadow scoring and benchmark reports

Shadow scoring writes immutable sidecars and never edits the historical `genui.jsonl` or `aggregates.json`:

```powershell
python dataset/scripts/backfill_genui_metric_v4.py <run-id>
```

Outputs are under `<run>/metric_v4_shadow/`: `scores.jsonl`, `aggregates.json`, the contract cache, and `benchmark_report.md`. Use `--limit N` for a runtime sample. Freeze an owner-approved run and its sidecar as the score-drift benchmark; compare future metric changes against that artifact plus the mutation suite. Human preference agreement remains a separate calibration gate.

## GRPO

GRPO must start from an SFT checkpoint:

```powershell
python training/scripts/train_grpo.py `
  --sft-checkpoint <sft-checkpoint> `
  --dataset <training.jsonl> `
  --prompt-template dataset/prompts/genui_gen_mobile_flatspec_v11.md `
  --output-dir <output-dir>
```

The entry point preserves reward columns, uses a source/model-family/time holdout, defaults to eight generations, validates training and evaluation batch divisibility, estimates completion length from accepted FlatSpec p99 plus 20% headroom, checks model context, masks truncated completions, and defaults to batch reward scaling with DAPO. The installed TRL must expose those controls or startup fails clearly.

Enable reward optimization only after SFT validation, reward red-team review, mutation robustness, and human-preference review are signed off. Disabling the GRPO job is independent of selecting `legacy` reporting.

## Monitoring and alerts

Key monitoring dimensions are `metric_version`, `contract_version`, `intent_bucket`, and model checkpoint. Track:

- reward mean/SD and zero-variance generation-group fraction;
- completion length/truncation and component count;
- parse, root, reference, cycle, and native-render failures;
- role omissions, action/table fidelity, JSON/source economy, dimensions, and cap activation;
- score drift against the frozen benchmark.

Alert when component count or completion length rises across two comparable windows without a fidelity improvement, or when any failure/cap rate changes materially. Component count is diagnostic only and must never become a quality objective.

## Verification

```powershell
python -m py_compile dataset/src/pipeline/genui_quality/*.py
python dataset/tests/test_genui_metric_v4.py -v
python dataset/tests/test_genui_metric_v4_metamorphic.py -v
pytest -q dataset/tests
```

The implementation can move from shadow to internal dashboard use only after the reference and mutation suites pass, runtime is measured on agreed benchmark hardware, and product, research, and renderer owners approve the calibration/rollout report.

## Rollback

Set `evaluation.metric_version: legacy` to restore legacy reporting, or stop the GRPO job to disable reward optimization. Preserve both historical v4 and legacy sidecars for the migration window; never recalculate old values in place.
