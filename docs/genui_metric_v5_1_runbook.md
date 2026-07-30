# GenUI Representation Quality v5.1.0

## Status and construct

GenUI metric v5.1.0 is a deterministic, source-conditioned engineering
metric for FlatSpec representation quality. It evaluates how faithfully a
candidate FlatSpec represents the supplied response text through the Android
renderer contract. It does not judge the source response's factual accuracy,
writing quality, safety, or usefulness.

The score is an **uncalibrated engineering score**. The initial configuration
contains expert-prior weights and thresholds. Do not interpret the 0-100 score
as a calibrated percentage and do not tune it to the ten-row regression audit.

The metric gives no positive credit for raw component count, unique type count,
JSON length, latency, token usage, bytes, or cost. Those may be reported only as
diagnostics.

## Two score surfaces

- `generation_reward_v5_1` scores the raw model completion through the shared
  normalization boundary. GRPO uses this surface.
- `render_artifact_quality_v5_1` scores the final Stage 3 `genui_json` after
  deterministic asset rewriting, text normalization, and image repair. Offline
  dashboards use this as the headline.

The two results carry separate raw and canonical candidate hashes. They may be
equal when post-processing does not change the candidate. They must not be
silently substituted for each other.

## Candidate and validator boundary

V5.1 requires `jsonschema`. Initialization compiles the FlatSpec and expected
contract validators and raises `MetricV51InitializationError` if the dependency
or either schema is unavailable or invalid.

Every caller uses the same normalization sequence:

1. deterministic raw parsing;
2. production `coerce_and_validate` canonicalization;
3. strict FlatSpec schema validation;
4. shared renderer-reference validation;
5. representation scoring over canonical renderer semantics, while retaining a
   bounded raw-format diagnostic.

Production-invalid candidates are capped independently of per-element averages.
Strict-schema-invalid but production-canonicalizable candidates use the
configured strict-format policy.

## Matching and dynamic evidence

Small assignments use exact deterministic Hungarian matching over the complete
dense matrix. Larger assignments preserve full precision/recall denominators
and use deterministic sparse candidate coverage for every item. A pair budget
never becomes an implicit failure: incomplete candidate coverage emits
`matching_budget_exceeded`, makes the affected fidelity atomic unavailable, and
activates the configured `matching_incomplete` policy.

Text blocks are normalized and tokenized once in immutable prepared
representations. A prepared source context is reused across an eight-generation
GRPO group.

The bounded dynamic evidence interpreter mirrors the Android
`FlatExprResolver` subset used by the generation contract: `$item`, `$state`,
`$bindState`, `$bindItem`, `$index`, `$cond`, `$template`, registered
`$computed`, literal values, renderer visibility comparisons, and renderer
inline-template forms. Unregistered computed functions and unsupported
expressions are explicit unknown evidence; they are never executed or silently
treated as visible/missing content.

Limits are controlled by:

- `max_repeat_items`
- `max_expanded_evidence_nodes`
- `max_expression_depth`
- `max_string_expansion_length`
- `max_matching_edges`
- `large_matching_top_k`

## Atomic allocation and formulas

For applicable atomic \(i\), with configured dimension \(d(i)\):

\[
p_i=W_{d(i)}v_i
\]

A deterministic capped proportional water-fill finds \(\lambda\):

\[
w_i=\min(c,\lambda p_i),\qquad\sum_{i\in A}w_i=1
\]

\[
Q_{base}=\sum_{i\in A}w_i u_i
\]

\[
C=\min\{c_k:\text{cap }k\text{ is active}\}
\]

\[
Q=\min(Q_{base},C),\qquad S=100Q,\qquad r_{GRPO}=2Q-1
\]

Dimension subscores are diagnostic only:

\[
G_d=\frac{\sum_{i\in d,\,i\in A}v_i u_i}
{\sum_{i\in d,\,i\in A}v_i}
\]

There is no dimension-level headline renormalization and no geometric headline
term. Runtime assertions enforce bounded finite utilities, a maximum ordinary
atomic effective weight, a unit weight sum when feasible, and bounded score and
reward.

Content block similarity is:

\[
0.45F_2(\text{unigram multisets})+
0.35F_1(\text{token bigrams})+
0.20LCSNorm
\]

Tables preserve column, row, associated-cell, row-key, and optional order
associations. Actions, media, and special role instances use one-to-one semantic
matching. Chart/Table substitution is allowed only by the table requirement's
`representation_policy`.

## Cap observability

`active_caps` lists all triggered policies. A cap is binding when:

\[
Q_{base}>C+\tau,\qquad \tau=10^{-12}
\]

`binding_caps` contains only those caps, while `cap_margin` is
`base_quality_before_caps - cap_0_1`. Aggregates report active and binding rates
separately. `capped_rate` remains only as a deprecated alias for historical
active-cap semantics.

## Identity and score reuse

Every score persists:

- metric name and version;
- metric fingerprint;
- source hash;
- expected-contract hash;
- raw candidate hash;
- canonical candidate hash.

The fingerprint covers the complete resolved configuration, both schemas,
renderer-reference inventory, contract/extractor/normalization versions,
explicit scoring/matching/evidence/dynamic/reporting policy versions, and a
hash manifest of score-defining source modules.

Stored v5.1 results are reused only when the fingerprint and all relevant
payload identities match exactly. V4 and v5.0 results remain readable but are
never interpreted or reused as v5.1.

## GRPO

Use a valid SFT checkpoint and keep source columns by setting
`remove_unused_columns=False`. The reward entry point prepares one source
context per identical source/intent/assets/contract group and scores all
completions through the same scalar core:

\[
r_{GRPO}=2Q_{generation}-1
\]

No network or model call occurs in the reward. Training startup calls
`ensure_v5_1_validation_ready()` before workers begin.

## Rescoring and rollback

Create a new immutable sidecar; never edit historical `genui.jsonl`,
v4 results, or v5.0 results:

```powershell
python dataset/scripts/backfill_genui_metric_v5_1.py `
  dataset/data/runs/<run> `
  --output-dir dataset/data/runs/<run>/metric_v5_1_shadow_<timestamp>
```

Run the deterministic benchmark:

```powershell
python dataset/scripts/benchmark_genui_metric_v5_1.py `
  --iterations 20 `
  --output workspace/genui_metric_v5_1_benchmark.json
```

Rollback reporting by configuration to v5.0/v4 without deleting v5.1
sidecars. Disable v5.1 GRPO reward independently. Never rewrite historical
scores in place.

## Monitoring

Key metrics by metric fingerprint, contract version, intent, and checkpoint:

- raw and final score distributions;
- active and binding cap rates and margins;
- parse, production, schema, root, reference, cycle, and native-render failure;
- matching completeness and dynamic-expression unknowns;
- content, table, action, media, and semantic-instance fidelity;
- GRPO group reward mean/SD, zero-variance groups, completion truncation;
- component count and source/JSON length only as descriptive diagnostics.

Alert when representation size grows without source-conditioned fidelity
improvement.

## Known limitations

- Python evidence is not Android-native screenshot or render evidence.
- Only registered pure `$computed` functions can contribute known evidence.
- Large bounded assignments may report explicit incomplete evaluation.
- The score has not been calibrated against rendered same-source human
  preferences.
