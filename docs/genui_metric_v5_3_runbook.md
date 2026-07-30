# GenUI Representation Quality v5.3.0

## Construct and status

GenUI Representation Quality measures how faithfully, appropriately,
robustly, accessibly, and economically canonical FlatSpec represents the
supplied source response as an Android-renderer-compatible UI. It does not
judge source factuality or writing quality.

The 0-100 result is an **uncalibrated engineering score**, not an
equal-interval quality percentage. V5.3 weights and thresholds are expert
priors. No weights were fitted to the ten-row shadow dataset.

V5.3 changes applicability, role extraction, semantic exact identity,
dynamic expression parity, and Repeat evidence. It therefore has a new
metric version and fingerprint. V5.2 files, readers, and sidecars remain
historical artifacts and are never reinterpreted as v5.3.

## Public score surfaces

- `generation_reward_v5_3` scores the raw completion through the shared
  candidate boundary and is the GRPO surface.
- `render_artifact_quality_v5_3` scores the final persisted renderer artifact
  and may include an attempted Android-native render result.
- `genui_quality_v5_3` is the migration alias for the final artifact.

Stage 3, offline scoring, aggregation, stale-score rescoring, and GRPO share
candidate normalization, source-contract resolution, renderer traversal,
evidence extraction, matching, configuration, and identity code.

## Exact formula

For each source-applicable atomic utility \(u_i\), with dimension prior
\(W_{d(i)}\) and normalized within-dimension prior \(v_i\):

\[
p_i=W_{d(i)}v_i
\]

Effective weights use capped proportional water-filling:

\[
w_i=\min(c,\lambda p_i),\qquad \sum_i w_i=1
\]

where \(c=0.10\) by default and \(\lambda\) is the deterministic solution.
If `applicable_count * c < 1`, allocation is infeasible and the configured
fail-closed policy applies. The limit is never silently exceeded.

\[
Q_{base}=\sum_i w_i u_i
\]

For active explicit caps \(C_j\):

\[
C=\min_j C_j,\qquad Q=\min(Q_{base},C)
\]

\[
quality_{0..100}=100Q,\qquad r_{GRPO}=2Q-1
\]

Dimension values are reporting summaries. They are not re-normalized into
the headline and there is no geometric headline term.

## Source-driven applicability and ownership

`AtomicApplicabilityPlan` is prepared before candidate scoring and depends
only on source text, intent, assets, the independently created expected
contract, and versioned extraction policies.

- Table fidelity is N/A without a required source table.
- Source-action fidelity is N/A without an applicable required external
  action, except an explicitly required local action instance.
- Source-media fidelity is N/A without required semantic media.
- Optional absent requirements are N/A.
- Candidate-only Tables, Charts, controls, actions, or media cannot turn a
  source-required-fidelity atomic into zero.

Evidence ownership partitions generic prose, tables, charts, actions, media,
semantic roles, and local UI mechanics. Table cells, chart labels, Button
labels, decorative controls, and role content are not also counted as
unsupported generic prose.

The low-budget
`fidelity.unsupported_external_addition_precision` detects unsupported
external URLs and source-like media. It excludes local state actions, tab or
Modal navigation, validation mechanics, decorative Icons, renderer assets,
and layout controls. Its nominal within-fidelity prior is 0.03 and its
effective headline contribution remains bounded by the global 0.10 guard.

Text chunking evaluates generic semantic segments only. It is N/A for a
source fully owned by Table, Chart, Formula, CodeBlock, ConsoleLog,
EmailPreview, Form, or other dedicated structured evidence.

## Structured representation policy

Each required source table has one canonical policy:

- `table_only`
- `chart_only`
- `structured_equivalent`
- `both_required`

The historical `either_table_or_chart` value migrates to
`structured_equivalent`. A chart directive does not independently require a
Table unless the contract says `both_required`.

## Semantic role extraction

The v5.3 deterministic fallback recognizes ordinary requests using chart,
graph, plot, visualization, equation, formula, code example/block, console
output/log, email/preview, form, controls, and input-form aliases. It
recognizes common create/show/add/include/provide/display/use/visualize/draft
verbs, counts, section lists, “X as a pie chart”, and fenced code.

Role instances carry stable requirement IDs plus only source-supported
payload fields. Generic requirement IDs are diagnostics and do not
participate in semantic exact keys.

\[
specificity=
\frac{\text{required instances with semantic payload}}
{\text{all required instances}}
\]

Count-only requirements reduce specificity and are reported through
`count_only_role_count`. Persisted independently generated contracts are
preferred for GRPO. Regex extraction is a conservative deterministic
fallback, and low-specificity behavior is explicit configuration.

## Exact matching and multiplicity

Prepared source indices are built once per source group. Certified exact
preallocation consumes these indices directly, preserves duplicate
multiplicity with queues, and solves only the residual graph.

Exact keys are:

- Action: canonical action type, target, and normalized label.
- Media: canonical kind and URL, plus normalized alt when required.
- Role: canonical role and semantic signature.

Only an explicit `expected_component_id` constrains a candidate component ID.
Generic contract IDs never block an exact match. One-to-one matching applies
to actions, media, roles, content units, table columns, and table rows.

## Android dynamic semantics

`DynamicEvidenceResolverV53` follows Android type-sensitive equality.
Integral and floating values are unequal for `eq`/`neq`, including nested
Maps and Lists. Numeric comparisons continue to use Android-equivalent
numeric coercion.

Renderer stringification is recursive and Kotlin-compatible:

```text
true
null
[1, null]
{x=true}
```

The checked-in parity corpus is version 2.0.0 and is executed by both Python
and `FlatSpecRendererSupportTest`. Changes to the corpus, resolver, built-ins,
or Android renderer invalidate the metric fingerprint.

## Repeat evidence and budgets

Repeat evidence streams every renderer-visible item instead of stopping at a
fixed count of 32. Work is bounded by:

- `max_repeat_work_units`
- `max_dynamic_evidence_bytes`
- `max_expression_evaluations`
- `max_matching_inputs`
- expression-depth and string-expansion limits

Budget exhaustion emits explicit diagnostics and an incomplete lower/upper
evidence certificate. Affected matching/evidence cannot be silently treated
as complete. Default tests cover 32, 33, 100, and 1,000 items.

## Identity and safe reuse

Every v5.3 breakdown stores:

- `metric_fingerprint`
- `reward_pipeline_fingerprint`
- source, raw-candidate, canonical-candidate, and expected-contract hashes
- computed-registry identity

The metric fingerprint hashes the complete resolved configuration, strict
and expected-contract schemas, renderer-reference inventory, Android
renderer, dynamic parity corpus, built-in registry, source/extractor
policies, scoring modules, and transitive reused semantic modules.

The reward-pipeline fingerprint additionally hashes completion-to-text
conversion, scalar/vector input normalization, grouping behavior, GRPO
dispatch code, and the training entry point.

Aggregate reuse requires exact metric version, both fingerprints, all
payload identities, and render evidence. Otherwise it recomputes and records
specific stale reasons. Historical records are not mutated.

## Migration and rollback

Use `evaluation.metric_version: v5_3` for v5.3-only Stage 3 scoring or `dual`
for migration comparison. V5.3 writes sidecar fields and does not overwrite
v5.2 breakdowns.

Rescore a run immutably:

```bash
python dataset/scripts/backfill_genui_metric_v5_3.py RUN_ID \
  --output-dir dataset/data/runs/NEW_V5_3_SIDECAR \
  --previous-sidecar dataset/data/runs/EXISTING_V5_2_SIDECAR
```

Rollback means selecting v5.2 reporting or disabling v5.3 GRPO. Never rewrite
historical scores in place.

## Verification

```bash
python -m compileall -q dataset/src/pipeline/genui_quality \
  dataset/src/pipeline/flat_spec_contract.py \
  dataset/src/pipeline/flat_spec_semantics.py
python -m pytest -q dataset/tests/test_genui_metric_v5_3*.py
python -m pytest -q dataset/tests/test_flat_spec_contract.py
python -m pytest -q dataset/tests
python dataset/scripts/benchmark_genui_metric_v5_3.py \
  --iterations 20 \
  --output workspace/genui_metric_v5_3_benchmark.json
```

Android release gating runs
`FlatSpecRendererSupportTest*` through `:app:testDebugUnitTest`.

## Known limitations

- Python does not implement the Android UI renderer. A non-attempt is
  recorded as unavailable; a failed attempted native render is a failure.
- The engineering score has not been calibrated against rendered
  same-source human preferences.
- Regex-derived role semantics can remain low-specificity; independently
  persisted contracts are preferred for GRPO.
- Benchmark timings are hardware-specific and are not calibration evidence.
