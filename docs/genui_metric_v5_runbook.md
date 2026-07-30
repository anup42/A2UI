# GenUI Representation Quality v5.0.0

## Construct and status

GenUI metric v5 measures how well a generated FlatSpec represents the supplied
source response as a renderer-compatible UI. The source response is reference
truth. The metric does not judge its factual accuracy, prose quality, safety, or
usefulness.

The 0–100 value is an **uncalibrated engineering score**. It is suitable for
controlled regressions and reward experiments, but it is not an equal-interval
quality percentage. The checked-in weights and thresholds are expert priors;
they were not fitted to the 20-row audit.

The metric never rewards component count, unique type count, output length,
latency, token use, byte size, or cost. Component count and JSON size are
diagnostics; excessive size can only reduce economy.

## Authoritative evaluation path

Every caller uses the same deterministic path:

1. `candidate_normalization.normalize_and_validate_candidate` parses the raw
   completion, applies the production `coerce_and_validate` canonicalization,
   validates the strict FlatSpec schema, and validates renderer references.
2. `source_contract.resolve_expected_ui_contract` accepts only a contract whose
   source, intent, extractor, policy, and asset identities match. Otherwise it
   records the mismatch and deterministically extracts a fresh contract.
3. `flat_spec_semantics.iter_renderer_references` supplies the shared renderer
   reference inventory to the production contract, graph audit, and evidence
   traversal.
4. Evidence is collected only from the declared root. Bounded Repeat expansion
   resolves supported `$state`, `$item`, `$index`, visibility, text, and action
   bindings. Unsupported expressions are reported as unknown.
5. Applicable atomics receive global capped budgets, the base score is computed,
   and explicit integrity or semantic caps are applied.

There are no network calls or model inference in this path. Stage 3, offline
scoring, aggregate rescoring, and GRPO all call this boundary.

## Source contract and instance semantics

Contract version 2 preserves ordered `content_units`, ordered table headers and
rows, optional row keys, ordered-table flags, requirement IDs, `required`,
`minimum_count`, and exact or representative media policy.

Content units and visible output blocks are assigned one-to-one. Pair similarity
is:

```text
sim(s, o) =
    0.45 * F2(unigram multisets)
  + 0.35 * F1(token bigrams)
  + 0.20 * LCSNorm(tokens)
```

The metric reports source-unit recall, output-block precision, their F2, global
token coverage as a secondary feature, and preservation of matched source order.

Tables use one-to-one column alignment followed by one-to-one row assignment.
Column, row, and associated-cell precision/recall are computed separately.
Associated cells remain attached to their source row; row-value swaps therefore
lose credit. `order_sensitive` tables also receive a row-order score.

Actions are assigned one-to-one by action type, normalized destination, and
label. A wrong destination receives no required-action credit. Media and
semantic role instances are also one-to-one; one candidate cannot satisfy
multiple requirements, and identical charts cannot satisfy distinct chart
instances.

Entries with `required: false` do not enter required recall or semantic gates.
Extra emitted actions, media, or tables still affect output precision.

## Atomics

The versioned configuration is
`dataset/configs/genui_metric_v5.yaml`. Its reporting dimensions and atomics are:

- Integrity: production validity, strict-schema validity, raw-format utility,
  schema contract, declared-root reachability, reference integrity, cycle
  freedom, reachable fraction, and renderer type semantics.
- Fidelity: content-unit F2, order preservation, global visible-token F2, exact
  number/date/unit F2, association-preserving table fidelity, heading
  fidelity/order, action/link fidelity, media fidelity, and output-block
  precision.
- Semantic mapping: required roles, distinct role instances, and component
  appropriateness.
- Hierarchy: root layout, root-distance depth, fan-out, heading grouping, and
  two-sided text chunking.
- Economy: semantic non-duplication, renderer-aware wrapper economy, and
  canonical JSON/source size.
- Accessibility: only the renderer-relevant checks applicable to visible
  controls, media, forms, and tables.

An inapplicable atomic is `null`; it does not receive free perfect credit.
Dimension subscores renormalize configured within-dimension weights for
reporting only. They are not combined to produce the headline.

## Headline formula and anti-domination guarantee

For applicable atomic `i` in dimension `d`, define its nominal budget:

```text
p_i = dimension_weight[d] * within_dimension_weight[d, i]
```

With global atomic cap `c`, find the deterministic water-filling multiplier
`lambda > 0` such that:

```text
w_i = min(c, lambda * p_i)
sum_i(w_i) = 1
```

Then:

```text
Q_base = sum_i(w_i * u_i)
C      = minimum active explicit cap, or 1 when no cap is active
Q      = min(Q_base, C)
quality_0_1   = Q
quality_0_100 = 100 * Q
reward_GRPO   = 2 * Q - 1
```

All atomic utilities are bounded to `[0, 1]`. At runtime the implementation
asserts finite non-negative weights, `w_i <= c`, a unit weight sum when
feasible, and `Q_base` in `[0, 1]`. If the applicable atomic count cannot
satisfy `count * c >= 1`, the configured v5 policy fails closed instead of
silently violating the cap.

There is no geometric term in v5. Consequently, changing one ordinary atomic
from 0 to 1 can move `Q_base` by at most its persisted effective weight.

## Validity and semantic caps

Validity is authoritative and cannot be diluted by valid nodes:

- Parse failure/non-object: quality 0.
- Production canonicalization/validation failure: cap 0.25 by default.
- Strict-schema invalid but production-canonicalizable: cap 0.95 by default,
  plus bounded raw-format evidence.
- Missing references or rooted cycles: cap 0.25 by default.
- Attempted native-render failure: cap 0.30 by default. An attempted failure is
  never N/A.

Required table, chart, special-role, action, and media omissions activate
configured missing or partial caps based on successful one-to-one matches.
Optional requirements do not activate these caps. Exact cap values are part of
the metric fingerprint.

## Score identity and stale-score handling

Every v5 breakdown contains:

- metric name, version, and `metric_fingerprint`;
- source, raw-candidate, canonical-candidate, and expected-contract hashes;
- normalization validity and error codes;
- base score, effective atomic and dimension budgets, applicability count, and
  anti-domination feasibility;
- dimensions, atomics, evidence, active caps, and errors.

The metric fingerprint covers the complete resolved configuration, algorithm
version, strict FlatSpec schema, expected-contract schema, renderer reference
inventory/version, contract/extractor/policy versions, and normalization policy.
Schema hashing is state-aware, so an in-process schema change invalidates reuse.

`aggregate.score_record` reuses a stored v5 breakdown only when the fingerprint,
source hash, raw and canonical candidate hashes, expected-contract hash, metric
version, and render evidence all match. Otherwise it recomputes and records
`stored_score_stale_reasons`. A v4 breakdown is never read as v5.

## Stage 3, aggregation, dashboard, and GRPO

`evaluation.metric_version: dual` shadow-writes legacy, frozen v4, and corrected
v5 fields during migration. Stage 3 preserves `genui_raw_completion`; v5 scores
that raw candidate through the same normalization used by GRPO. The aggregate
headline is the distribution of per-sample scores, never a nonlinear function
of aggregate atomic means.

The dashboard accepts `--metric-version v5` and labels the score uncalibrated.
Latency, cost, tokens, bytes, and component counts remain separate diagnostics.

GRPO keeps `response_text`, `intent_bucket`, `assets`, and
`expected_ui_contract` (`remove_unused_columns=False`). The deterministic reward
is `2Q - 1`. Start from a valid SFT checkpoint, use eight generations initially,
mask truncated completions, and monitor reward dispersion, zero-variance groups,
length/truncation, parse/root/reference/render failure, caps, and dimensions.

## Immutable rescoring and migration from v4

V4 remains frozen at version `4.0.0` for historical comparison. Do not overwrite
old `genui_quality_v4` fields or reinterpret them as v5.

Create a new immutable sidecar:

```powershell
python dataset/scripts/backfill_genui_metric_v5.py `
  dataset/data/runs/<run> `
  --output-dir dataset/data/runs/<run>_metric_v5_sidecar
```

For multiple source runs, list each run and provide one new output directory.
The command refuses to overwrite a non-empty sidecar. It writes `scores.jsonl`,
`aggregates.json`, a contract cache, source-file hashes, and a report without
mutating historical inputs.

Rollback is configuration-only: select `legacy` or explicit frozen `v4` for
reporting, or disable v5 GRPO reward. Never rewrite historical scores in place.

## Verification and benchmark

```powershell
python -m compileall -q dataset/src/pipeline/genui_quality dataset/src/pipeline/flat_spec_contract.py
python -m pytest -q dataset/tests/test_genui_metric_v5.py dataset/tests/test_genui_metric_v5_compliance.py dataset/tests/test_genui_metric_v5_metamorphic.py dataset/tests/test_genui_metric_v5_pipeline.py dataset/tests/test_flat_spec_contract.py
python -m pytest -q dataset/tests
python dataset/scripts/benchmark_genui_metric_v5.py --iterations 50
```

The deterministic benchmark reports p50/p95 time for bounded small, medium, and
large samples. Matching uses cubic assignment only up to
`max_assignment_size`; larger candidate sets are truncated for matching with an
explicit diagnostic and unmatched instances remain in precision/recall
denominators.

## Known limitations

- Python audits renderer semantics and references; it is not a replacement for
  Android native rendering.
- Native-render caps apply only when a real render was attempted and its result
  supplied.
- The bounded Repeat interpreter covers the declared renderer/generation
  subset. Unsupported expressions are diagnosed as unknown.
- Human same-source preference calibration and renderer-owner/research-owner
  sign-off remain required before treating score intervals as calibrated or
  enabling production GRPO.
