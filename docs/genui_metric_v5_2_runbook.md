# GenUI Representation Quality v5.2.0

## Construct and status

GenUI Representation Quality measures how faithfully, appropriately,
robustly, accessibly, and economically canonical FlatSpec represents the
supplied source response as the UI consumed by the Android renderer. It does
not judge source factuality, prose quality, latency, cost, output length,
component count, or type count.

The 0–100 output is an **uncalibrated engineering score**. It is not an
equal-interval human-quality percentage. The configuration contains expert
priors; no weights were fitted to the ten-row shadow audit.

V5.2 is separate from v5.1. Historical v5.1 breakdowns and sidecars remain
readable, but are never interpreted or reused as v5.2.

## Score surfaces

- `generation_reward_v5_2` scores the raw model completion after the shared
  candidate-normalization boundary. This is the deterministic GRPO surface.
- `render_artifact_quality_v5_2` scores the final persisted renderer artifact
  and incorporates native render evidence when an attempt exists.
- `genui_quality_v5_2` is a compatibility alias for the final artifact
  breakdown during migration.

Stage 3, offline scoring, aggregation/rescoring, and GRPO all use the same
normalization, source-contract, graph, evidence, matching, and formula code.

## Exact formula

For applicable atomic utility \(u_i\), dimension prior \(W_{d(i)}\), and
within-dimension prior \(v_i\):

\[
p_i = W_{d(i)}v_i
\]

The effective global weights use capped proportional water-filling:

\[
w_i=\min(c,\lambda p_i),\qquad \sum_i w_i=1
\]

where \(c\) is `max_atomic_global_weight`. If the allocation is infeasible,
the configured fail-closed policy applies; the cap is never silently
violated.

\[
Q_{base}=\sum_iw_iu_i
\]

\[
C=\min_j C_j,\qquad Q=\min(Q_{base},C)
\]

\[
S=100Q,\qquad r_{GRPO}=2Q-1
\]

Dimensions are reporting summaries only. There is no dimension-level
headline renormalization and no geometric headline term.

## Candidate and renderer boundary

`normalize_and_validate_candidate` parses raw JSON, applies the same
production canonicalization as Stage 3, validates the strict FlatSpec schema,
and audits renderer references. Representation atomics score the canonical
renderer semantics; raw-format utility remains a small separate signal.

- parse failure: quality 0;
- production-invalid but canonicalizable: cap at the configured invalid cap;
- strict-invalid but production-canonicalizable: strict-format policy;
- strict canonical FlatSpec: full eligibility.

Renderer references come from `flat_spec_semantics.py`, shared by production
contract validation and metric traversal. Unsupported reachable types,
missing nested references, cycles, and forbidden top-level properties cannot
be diluted by adding valid nodes.

## Matching and certification

V5.2 does not use the v5.1 false-complete Boolean.

1. Domain-specific exact identities are allocated one-to-one first while
   preserving multiplicity.
2. Residual fuzzy candidates use deterministic multi-key blocking.
3. The residual bipartite graph is split into connected components.
4. Bounded components use exact Hungarian assignment.
5. Larger sparse components use exact min-cost flow under deterministic work
   budgets.
6. A greedy fallback is allowed only after budget exhaustion. Such a result
   reports `approximate_matching_used=true`,
   `optimality_certified=false`, makes the affected atomic unavailable, and
   activates `matching_uncertified`.

Candidate edges, Hungarian work, sparse-flow relaxations, and blocking width
are independently bounded by `max_matching_edges`,
`max_matching_hungarian_work`, `max_matching_sparse_relaxations`, and
`large_matching_top_k`. These are resolved configuration inputs and therefore
change the metric fingerprint.

Every breakdown exposes:

- `vertex_edge_coverage_complete`;
- `full_cardinality_matching_exists`;
- `candidate_generation_complete`;
- `optimality_certified`;
- `approximate_matching_used`;
- lower and upper weight bounds;
- evaluated/candidate edge counts;
- exact-preallocated and fuzzy-residual counts;
- diagnostic codes.

Vertex edge coverage is not treated as proof of optimality. Incomplete
candidate generation can still be certified only when the observed weight
reaches the mathematical global upper bound.

## Android dynamic-expression parity

The shared corpus is
`dataset/tests/fixtures/flat_expr_parity_vectors.json`, mirrored into Android
unit-test resources. Android evaluates it through the production
`FlatExprResolver` and default computed functions. Python evaluates the same
JSON values through `DynamicEvidenceResolverV52`.

Covered semantics include optional-leading-slash and root paths, escaped
pointer tokens, slash-only item paths, literal dotted keys, bind paths, index,
condition operators, Boolean numeric conversion, truthiness, deep equality,
inline template syntaxes, literals, ordinary unknown-dollar maps, and the
Android built-ins:

- `concat`;
- `uppercase`;
- `lowercase`;
- `coalesce`;
- `sum`.

Unsupported custom computed functions are explicit unknown evidence. A custom
function registry must provide a stable registry ID, version, manifest hash,
and deterministic callable behavior. The effective registry identity is in
the metric fingerprint and every score identity.

Evidence expansion is bounded by repeat item, expanded node, expression
depth, and string length limits. A limit hit is diagnostic evidence, not a
silent omission.

## Semantic role contracts

V5.2 uses singular canonical role keys and instance requirements for Chart,
Formula, CodeBlock, ConsoleLog, EmailPreview, and form/control semantics.
Deterministic extraction supports explicit numbered labels, cardinality
headings plus lists, and inline create/include/show/render statements.

`role_count_coverage` is separate from
`semantic_role_instance_fidelity`. When only a count is source-supported,
semantic-instance fidelity is N/A rather than a free 1.0.
`contract_semantic_specificity`, extraction diagnostics, and migration
provenance are persisted. Explicit empty human requirements remain
authoritative.

Non-interchangeable requirements cannot be satisfied by repeated candidate
instances with the same semantic signature.

## Source contracts and migration

The v5.2 contract version is 2.2.0. Cache identity includes normalized source,
intent, relevant assets, extractor version, and extraction policy version.
Persisted contracts require exact source, cache, intent, version, and schema
identity.

Source-valid v5.0/v5.1 contracts are migrated immutably. Explicit fields are
preserved, deterministic semantics may fill missing information, and
`contract_migration` records source, confidence, and policy. Heuristic
semantics are never relabeled as human-authored.

## Immutable score identity and reuse

`metric_fingerprint_v5_2` hashes canonicalized score-defining inputs:

- algorithm version and complete resolved config;
- strict FlatSpec and expected-contract schemas;
- renderer-reference inventory;
- normalization, matching, scoring, evidence, dynamic, extraction,
  structure, validation, reporting, and migration policy versions;
- the Android parity vector corpus;
- built-in and custom computed registry identities;
- a transitive semantic source manifest covering the core scorer,
  water-filling, v5.2 scorer, matcher, metrics, evidence, source contract,
  graph, validation, normalization, configuration, aggregation/reuse,
  FlatSpec contract, and renderer semantics.

A stored v5.2 score is reusable only when metric version, fingerprint, source
hash, raw candidate hash, canonical candidate hash, expected-contract hash,
and render evidence match exactly. Otherwise it is recomputed with explicit
stale reasons. V4/v5.0/v5.1 data are never mutated in place.

Callable bytecode, nested code constants, defaults, keyword defaults, and
closure values are serialized without process-specific object addresses.
The test suite verifies that identical registries produce the same fingerprint
in separate Python processes.

## GRPO input shapes

- `response_text`: one string or exactly one string per completion.
- `intent_bucket`: null, one string, or exactly one string/null per
  completion.
- `assets`: a list of asset mappings is one collection and broadcasts,
  regardless of whether its length equals the completion count. Per-completion
  assets must be nested collections or use
  `PerCompletionAssetCollections`.
- `expected_ui_contract`: one mapping or exactly one mapping/null per
  completion.

Malformed or ambiguous shapes raise `ValueError` before reward calculation.
Prepared source contracts, source text blocks, exact indices, and signatures
are reused for completions sharing a source group.

## Observability

Reward logging includes certification coverage/full-cardinality/optimality,
approximation, evaluated edges, preallocated/residual counts, dynamic parity
version and unknowns, contract specificity and count-only roles, input-shape
errors, reward latency, score/cap distributions, parse/root/reference/cycle
failures, per-dimension values, truncation, zero-variance groups, component
count, and completion length. Component/length are diagnostics only.

## Rescoring, rollout, and rollback

Use `dataset/scripts/backfill_genui_metric_v5_2.py` to create a new,
non-existing sidecar directory. It reads source records and never rewrites
them. The sidecar contains v5.1/v5.2 raw and final scores, deltas, identities,
matching and dynamic diagnostics, contract specificity, and an aggregate
report.

Roll out v5.2 in shadow mode first. Enable v5.2 GRPO only after reward
red-team, native-render coverage review, representative hardware benchmarks,
and same-source human preference validation. Roll back by selecting v5.1,
v5.0, v4, or legacy reporting; never rewrite historical scores.

## Verification

```powershell
python -m compileall -q dataset/src/pipeline/genui_quality dataset/src/pipeline/flat_spec_contract.py dataset/src/pipeline/flat_spec_semantics.py
$files = Get-ChildItem dataset/tests -Filter 'test_genui_metric_v5_2*.py' | ForEach-Object FullName
python -m pytest -q $files
python -m pytest -q dataset/tests/test_flat_spec_contract.py
python -m pytest -q dataset/tests
python dataset/scripts/benchmark_genui_metric_v5_2.py --iterations 20 --output workspace/genui_metric_v5_2_benchmark.json
```

Android parity:

```powershell
cd android
.\gradlew.bat :app:testDebugUnitTest --tests com.samsung.genuicraft.renderer.FlatSpecRendererSupportTest.flatExpressionResolver_matchesSharedV52ParityVectors
```

## Known limitations

- The engineering score has no human equal-interval calibration.
- Native-render evidence exists only when an Android render is actually
  attempted; Python does not emulate full Android UI rendering.
- Custom computed functions beyond the registered deterministic manifest are
  intentionally unknown.
- GRPO latency budgets are hardware-specific. Correctness and certification
  are not silently relaxed to meet a latency target.
