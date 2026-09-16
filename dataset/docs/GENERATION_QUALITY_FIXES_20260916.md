# Generation-quality implementation handoff

## Scope

Implemented pipeline, evaluator, renderer and generic training-import safeguards from the Gemma 4 run audit. The existing archive and extracted records remain unchanged. No real generation, Stage 3 regeneration, training, model serving or GPU benchmark was run on this PC. Small mocked unit tests exercise pipeline functions without model calls.

Actual asset bytes are **not a training requirement**. The model still needs consistent source-visible placeholder identities. Asset availability is recorded separately from source correctness and training eligibility.

The complete original issue register is in [the quality report](C:/Users/anupk/Downloads/dataset_gemma4_20260915_quality_analysis/GENERATION_QUALITY_REPORT.md). The new [H100 runbook](GEMMA4_H100X8_GENERATION.md) gives the remote launch commands and measurement plan.

## Issue-by-issue disposition

| Audit issue | Implemented change | Remaining evidence or limitation |
|---|---|---|
| Q01: reference namespace corrupts scores | Explicit reference map restores semantic evidence; raw syntax/hash remains separate; map identity participates in score reuse | Old saved scores have not been overwritten or recalculated as a new official run |
| Q02: importer removes valid actions | Correct `[URL_1]` grammar; reject action-removing repairs; preserve actions through masking round trips | Unsafe/inconsistent historical rows require review |
| Q03: target-only placeholders | Verified reference binding followed by one source/target/asset registry; collision avoidance and grounding, including when URL serialization is disabled | Historical recovery requires an explicit map or matching hash-verified graph; no guessing from suffix numbers |
| Q04: invisible List items | Shared literal List contract; Android renders strings/text-link objects; opaque nested component-call maps rejected with guidance to use children | Full Android build/device rendering remains unverified on this host |
| Q05: arithmetic errors | Typed Decimal calculations against reviewed input packets; check all declared result bindings | Arbitrary prose is not fact-checked; supply reviewed fact packets |
| Q06: omitted constraints | Explicit numeric ranges, required/forbidden text and bounded same-day schedule checks | Natural-language and multi-day constraints beyond the declared schema need review |
| Q07: invented facts/roles | Original query and provenance packet preserved; declared fact/role checks; unverified material marked for review | No automatic proof of externally changing or undeclared claims |
| Q08: misleading actions | Declared navigation/capability/mock modes, exact label/destination binding and explicit mock status | Actual action execution requires remote/device functional tests |
| Q09: grammar failures | Prompt generated from typed catalog signatures/enums, quoted placeholders and List/visibility examples; dataset/Android prompt copies synchronized | Real model acceptance rate must be measured remotely |
| Q10: detached nodes | Renderer-reference reachability required at Stage 3 acceptance; report offending IDs | Existing affected rows were not regenerated |
| Q11: arbitrary minimum node count | Role/content completeness replaces the five-element acceptance floor | Semantic completeness still needs review |
| Q12: fake interactions | Modality/capability packet checks, schema/wire validation and separate dynamic evidence diagnostics | Interactive examples require functioning supported actions, state and device tests |
| Q13: unresolved assets | Offline/unverified asset status is explicit and separate from training eligibility; Stage 3 does not download assets | Visual readiness remains a separate concern |
| Q14: source icon penalties | Source role/provenance matching across reachable Icon/Image content | Matching remains an engineering metric, not visual inspection |
| Q15: equivalent formula wrappers | Normalize balanced outer wrappers while retaining equation content and values | Algebraic equivalence is not inferred |
| Q16: paths falsely require console output | Narrowed console/log source detection with regressions for paths and real console evidence | Inferred roles remain review diagnostics |
| Q17: wire/repair failures | Combined parse, schema, renderer reachability and wire gate; concise errors; incomplete completion detection | Provider/model compatibility still needs remote smoke tests |
| Q18: warning counts used as quality | Explainable acceptance metadata; mechanical defects block; semantic matcher differences require review | Assignment optimality does not prove semantic truth; no arbitrary score threshold added |
| Q19: repair lacks original context | Repairs retain the full masked source and invariants; oversized requests fail explicitly instead of silently clipping source | Long sources may need a larger context budget or intentional source partitioning |
| Q20: undercounted attempts | Immutable per-attempt request/completion records and cumulative token/latency accounting | Provider/server-internal work not exposed by the API cannot be reconstructed |
| Q21: repeated scenarios/templates | Bounded prior-scenario context, lexical duplicate candidates and scenario-family split isolation | Lexical similarity is not full semantic deduplication; coverage quotas still need a curated task plan |
| Q22: rare component/task coverage | Typed prompt and modality metadata enable supported examples; family-aware splits prevent simple variant leakage | A reviewed rare-component/state/event cohort must be authored and generated on the GPU server |
| Q23: overwritten provenance | Immutable phase manifests, effective config, prompt/source hashes, package versions, finish/reasoning/attempt metadata | Keep the exact server environment and model/tokenizer revisions; hashes alone cannot recreate missing dirty source files |
| Q24: misleading diagnostics | Reference metadata excluded from numeric evidence; legacy counter and approximate-token measurements explicitly labeled | Legacy counts remain compatibility diagnostics; exact tokenizer counts depend on the deployed model |
| Q25: missing release validation | Source/semantic/structural/asset statuses separated; scenario-family split support and documented remote evaluation gates | Native rendering/interactions, held-out model training and quality/throughput benchmarks remain to be executed |

## Training admission

The generic `prepare_dataset` importer now rejects format/quality-rejected records, mechanical blocking diagnostics, known failed sources, inconsistent source-audit identities, unbound references and action removal. A newer failed Stage 2 audit cannot be hidden by older Stage 3 metadata.

Legacy rows without a reviewed contract remain explicitly unverified. For a curated dataset whose supplied packets have been reviewed, the optional strict filters are:

```yaml
filters:
  require_source_contract_checks: true
  require_semantic_acceptance: true
url_preprocessing:
  enabled: true
  binding_policy: source_identity
```

The second flag requires explicit semantic admission, including absence of unresolved review diagnostics. These are conservative checks, not proof that the student's future predictions will be correct. Enabling both on an unaudited historical corpus can reject most or all rows; inspect the rejection report rather than lowering safeguards to reach a row count.

The source contract schema and examples are documented in [source_quality_contract.md](source_quality_contract.md). This change targets the generic importer and shared codec/repair layer; frozen evaluation and separately curated recovery pipelines retain their own admission policies.

## Read-only historical compatibility result

The corrected import primitives and strict wire validator were applied to the original 999 Stage 3 records without preparing a training dataset:

| Result | Rows |
|---|---:|
| Pass reference, round-trip, action-preservation and wire checks | 321 |
| Quarantined by the new checks | 666 |
| Already marked format rejected | 12 |

The 321 compatible rows preserve **887 actions** and contain no target-only opaque references after grounding. This is a structural/reference result, not a factual quality certification.

First failure categories among the 666 quarantined rows:

- 444: saved canonical graph changes non-reference text relative to raw output, so strict historical reference alignment cannot be established.
- 184: unsupported opaque component calls/shapes inside `List.items`.
- 18: current repair would remove an unsafe action.
- 6: target references have no source/declared-asset grounding.
- 14: strict wire validation fails after the preceding checks.

These categories are mutually exclusive first failures, not all defects per row. The pipeline previously changed text after decoding (for example, hyphens into bullets). Future Stage 3 records now preserve the decoded graph plus exact reference restoration, avoiding that provenance mismatch. This task does not silently rewrite historical text or declare those rows semantically wrong merely because alignment failed.

Details: [postfix_reference_probe.json](C:/Users/anupk/Downloads/dataset_gemma4_20260915_quality_analysis/postfix_reference_probe.json). Query/response/target/manifest hashes matched before and after the probe.

## Validation and limits

- Final integrated CPU suite: **215 tests passed**, plus **17 subtests**, in 52.11 seconds. It covers reference identity, URL collisions, action preservation, source checks, literal lists, metric false positives, launch planning/grouping/shutdown and mocked pipeline paths. Existing `datetime.utcnow` / `jsonschema.RefResolver` deprecation warnings remain.
- A separate broader evaluator selection passed **214 tests** (overlaps the integrated selection; do not add the counts).
- The new Kotlin list parser compiled against the locally cached Kotlin runtime and passed **3 JUnit tests**.
- Full Android Gradle unit-test execution was blocked because `com.android.application:8.7.3` could not resolve in offline mode. No APK was built or installed; Compose integration/device rendering is still unverified.
- Bash syntax and mocked replica processes were tested on Git Bash. Actual Linux vLLM/CUDA process lifecycle, GPU memory fit and performance require the server pilot.
- All **2,283 extracted files** and the original archive still match the extraction hashes. Four generated static prompt copies match the prompt generator, and the scoped Git whitespace check passed.
- The H100 layout is a measured-later candidate: **4 replicas x TP=2**, BF16, 128 total client requests initially, with reasoning enabled throughout and 8,192 completion tokens per stage. Compare accepted examples/hour with TP=1/4/8 before adopting production settings.

Re-run the focused tests from the repository root using Python/pytest. They use synthetic files and mocked providers; do not run the real generation launcher on this PC. [Integrated JUnit XML](C:/Users/anupk/Downloads/gemma_quality_cpu_tests/final_integrated.xml), [original-file verification](C:/Users/anupk/Downloads/dataset_gemma4_20260915_quality_analysis/final_verification.json), and [validation summary](GENERATION_QUALITY_VALIDATION_20260916.md).
