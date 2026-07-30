# GenUI Representation Quality v5.4 runbook

## Construct and status

GenUI Representation Quality v5.4.0 measures how faithfully a renderer-compatible
canonical FlatSpec represents the supplied source response. The source is
reference truth; the metric does not judge source factuality or writing quality.
The 0–100 value is an **uncalibrated engineering score**, not an equal-interval
human-quality percentage.

V5.4 is immutable and versioned separately from v5.3. Historical v5.3 records,
readers, fingerprints, and sidecars remain valid under v5.3 semantics.

The metric has two public surfaces:

- `render_artifact_quality_v5_4`: scores canonical renderer semantics and ignores
  raw wrappers around an otherwise identical JSON artifact.
- `generation_reward_v5_4`: applies the same artifact score plus a small,
  generation-only raw JSON envelope utility.

`genui_quality_v5_4` is the migration alias for artifact quality.

## Headline formula

For applicable atomic `i` in dimension `d`, the nominal budget is:

```text
p_i = dimension_weight[d] * within_dimension_weight[i]
```

The deterministic capped water-filling solver finds `lambda` such that:

```text
w_i = min(max_atomic_global_weight, lambda * p_i)
sum_i(w_i) = 1
```

When feasible:

```text
Q_base = sum_i(w_i * u_i)
Q_accessible = Q_base * (1 - alpha * (1 - accessibility_conformance))
Q_artifact = min(Q_accessible, all applicable structured caps)
Q_generation = min(Q_accessible * raw_envelope_utility, all applicable caps)
quality_0_100 = 100 * Q
reward = 2 * Q_generation - 1
```

`alpha <= 0.03`. Accessibility is penalty-only: adding a candidate-created
accessible control cannot increase the score. Raw envelope utilities are also
penalty-only and do not alter artifact quality.

If fewer applicable atomics make the configured hard atomic cap infeasible, the
configured `fail_closed` policy is used. No effective atomic weight may exceed
the configured cap.

## Evidence ownership

Source and output evidence are assigned to one semantic channel before scoring.
Tables, charts, code, formulas, logs, email, media, actions, headings, and forms
own their structured spans. Generic prose receives only unowned narrative text.
An explicitly requested narrative summary is a distinct generic source unit.
Redundant text copied from a structured component therefore cannot improve
generic fidelity and is penalized by cross-channel duplication.

## Renderer-effective semantics

V5.4 centralizes Android-effective Table, Chart, and media interpretation in
`renderer_effective_semantics_v5_4.py`.

- Tables preserve object/string columns, positional/mapping rows, nested
  payloads, wrapper rows, and slash/no-slash state paths.
- Charts use Android precedence (`rows`, `data`, Table-compatible fallback),
  effective axes, and numeric parsing. Renderer-inert metadata cannot create
  positive evidence.
- Image, Video, and Audio use the same bounded recursive media-source lookup.
- Type-contract diagnostics validate only behavior the Android renderer
  actually consumes.

The checked-in Python and Android dynamic parity corpora have identical bytes
and version. Initialization fails closed if required schemas, prompt contract,
capture helper, or parity artifacts are missing.

## Role matching and source contracts

Required semantic instances are matched one-to-one. Duplicate candidate
signatures have explicit capacity and cannot inflate semantic count. Matching
certification, count-only low-specificity requirements, unknown dynamic
expressions, and budget exhaustion are reported separately.

Fallback source extraction recognizes common role phrases including
parenthetical headings, “as a chart”, plural chart lists, and corresponding
Formula, CodeBlock, ConsoleLog, EmailPreview, and Form phrases. Persisted
independently produced contracts remain preferred. Source hash, normalized
intent, extractor version, extraction policy, and relevant assets define cache
identity.

## Identity and stale-score rules

Every v5.4 result carries:

- metric and reward-pipeline fingerprints;
- source, raw candidate, canonical candidate, and expected-contract hashes;
- ownership, renderer-effective, component-contract, role-matching, dynamic
  parity, extraction, raw-envelope, and registry-identity policy versions.

The metric fingerprint covers the resolved configuration, score-defining Python
modules, schemas, Stage 3 prompt, Android renderer semantics, parity vectors,
built-ins, and computed registry behavior. Immutable global dependencies used by
a registered computed function affect its identity. Mutable or unsupported
dependencies are rejected.

Stored scores are reused only when every relevant identity, variant, and render
status matches. V4/v5.0–v5.3 breakdowns are never interpreted as v5.4.

## Stage 3, dashboard, aggregation, and GRPO

Use `evaluation.metric_version: v5_4` for v5.4-only scoring or `dual` during
migration. Stage 3 stores:

```text
generation_reward_v5_4
render_artifact_quality_v5_4
genui_quality_v5_4
metric_identity_v5_4
expected_ui_contract_v5_4
```

Aggregation computes distributions from per-sample scores; it never feeds
aggregate means through caps. The dashboard accepts `--metric-version v5_4`.
Training uses `genui_metric_v5_4.yaml`, validates the full bundle at startup,
retains source/intent/assets/contracts, and maps `Q_generation` to `2Q-1`.

## Immutable rescoring

Never rewrite a historical run:

```powershell
python dataset/scripts/backfill_genui_metric_v5_4.py RUN_ID `
  --output-dir dataset/data/runs/RUN_ID_metric_v5_4_sidecar
```

The command refuses an existing output directory, verifies that source
`genui.jsonl` did not change, and writes `scores.jsonl`, `aggregates.json`, and
`manifest.json` with source/output hashes.

## Verification and benchmark

```powershell
python -m compileall -q dataset/src/pipeline/genui_quality `
  dataset/src/pipeline/flat_spec_contract.py `
  dataset/src/pipeline/flat_spec_semantics.py
python -m pytest -q dataset/tests/test_genui_metric_v5_4*.py
python -m pytest -q dataset/tests/test_flat_spec_contract.py
python -m pytest -q dataset/tests
android\gradlew.bat -p android :app:testDebugUnitTest `
  --tests "*FlatSpecRendererSupportTest*"
python dataset/scripts/benchmark_genui_metric_v5_4.py `
  --iterations 20 `
  --output workspace/genui_metric_v5_4_benchmark.json
```

The benchmark reports source preparation, scalar reward, group-of-eight
latency, matching/dynamic/ownership timing, peak memory, and budget exhaustion.
The production owner must approve an SLO on production hardware; the review
machine observations are reference data, not a relaxed correctness threshold.

## Migration and rollback

Shadow-score v5.4 beside v5.3, compare mutation probes and rendered same-source
human preferences, then switch dashboards and GRPO independently. Roll back by
selecting `v5_3`/`legacy` reporting or disabling v5.4 GRPO. Do not rewrite
historical scores.

## Known limitations

- Initial weights and thresholds are expert priors; no human calibration is
  claimed.
- Python semantic interpretation is not native rendering. `render_ok` must come
  from an attempted Android native render; missing native evidence is explicit.
- Regex source extraction remains a low-specificity fallback.
- Dynamic evidence is intentionally bounded; unsupported expressions produce
  explicit unknown diagnostics and caps.
