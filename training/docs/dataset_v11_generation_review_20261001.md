# Dataset v11/v11s import and generation review — 1 October 2026

The supplied release is useful additional supervision, but it is not a fully verified gold dataset. Most new rows were accepted by a revised representation-fidelity policy without editing their target. The current review additionally found 59 incorrect legacy text joins and 1,183 rows with Chart subtypes the Android renderer does not support. The release bytes are preserved; a separate preparation guard holds these targets until Stage 3 regeneration or an independently verified renderer change.

## What was imported

Source package: `C:\Users\anupk\Downloads\dataset_v10_v11_v11s_data_and_changes_20260930`.

| Frozen release | Train | Validation | Total | Relationship |
|---|---:|---:|---:|---|
| v10 | 91,115 | 1,862 | 92,977 | Existing original release remains unchanged |
| v11 | 200,214 | 4,089 | 204,303 | 92,953 screened v10 rows plus all v11s rows |
| v11s | 109,121 | 2,229 | 111,350 | New-only subset already contained in v11 |

Ten previously held v10 rows were recovered by the supplied offline repair policy; 24 rows in 21 v10 families remain excluded from v11. These ten recovered rows include four source arithmetic/unit corrections, so v11 is not simply the old v10 files concatenated with new data. The original v10 files remain immutable.

The training representation is A2UI Express v1 with `response_text`, `completion`, and aligned chat `messages`. The Express decoder lowers to the Android flat graph contract. These are completed training inputs; `training/` must not become a Stage 1/2/3 generator.

The repository keeps gzip shards under `training/data/train/v11` and `training/data/train/v11s`, with restoration helpers/manifests. Large audit ledgers retain their original names inside lossless compressed files such as `new_decisions.jsonl.gz`, `new_family_assignments.jsonl.gz`, and `v10_decisions.jsonl.gz`. See [release import evidence](../reports/dataset_v11_20260930/repository_import.json), [audit restoration instructions](../reports/dataset_v11_20260930/README.md), and each dataset's README. Compression changes storage, not the restored JSONL bytes.

## Repair, reassessment, and quarantine are different outcomes

The supplied source inventory reports 155,000 Stage 1 queries, 148,915 Stage 2 responses, and 131,400 Stage 3 records, including one malformed record. There are 17,516 response IDs without a Stage 3 record. Missing targets were not invented. The original raw source archives are required to reproduce the full source inventory and independent Stage 3 lineage replay.

I independently streamed all 131,400 supplied decision records and reproduced these mutually exclusive new-row outcomes:

| Outcome | Rows | Meaning |
|---|---:|---|
| KEEP | 109,799 | No semantic repair was recorded in this release |
| REPAIR | 1,551 | Recorded offline target/source repair, with before/after evidence |
| QUARANTINE | 20,050 | Excluded by source, provenance, representation, role, or structural gates |
| Retained new total | 111,350 | KEEP + REPAIR; equals the frozen v11s release |

Of the retained new rows, **97,975 originally had `training_acceptance.eligible=false`**. They passed a later reassessment. This count overlaps KEEP/REPAIR and must not be added to them. In particular, these 97,975 rows are not 97,975 new repairs, new model generations, or independently fact-checked examples.

The recorded reassessment checks whether generic source units and numeric anchors survive in renderer-visible evidence. It adds exact Table header/value bindings and native static Card labels, accounts for EmailPreview field roles, and removes a default 4,096-character evidence clipping limit through a finite input-derived bound. Specific action/media/table/role/dynamic failures require a fresh v5.4 check. Original flags, source hashes, and teacher telemetry stay in row metadata. This is a representation policy, not source factual certification.

Example: `dataset_muse_glimmer_100k_r1:u_000005_01:4` was originally ineligible for `content_unit_fidelity` and `exact_numbers_dates_units_fbeta`. Its reassessment finds all 14 generic source units represented, no missing or unsupported numeric anchors, and no semantic edit. The row still records `source_quality.status=needs_review`, unverified added action URLs, and `prose_fact_verification=not_performed`.

## Observed repairs and what caused them

Repair-kind counts overlap when one row has several kinds. I reproduced these row-level counts from the decision ledger:

| Repair kind | Retained rows | Concrete example | Likely upstream cause |
|---|---:|---|---|
| Adjacent uniquely anchored source prose restoration | 1,285 | `r1:u_002305_01:67` omitted `Allocation from $4,800 net` between the 50/30/20 heading and allocation list | Stage 3 treated short explanatory/basis prose as disposable layout scaffolding |
| Short source-literal expansion | 182 | `r1:u_005779_01` heading omitted `– 71 min` from `The Cabinet of Dr Caligari 1920 – 71 min` | Title simplification dropped a source qualifier/unit |
| Legacy midword Text joining | 59 | `r1:u_007863_01:5728` changed `San Lorenzo 50` followed by a separate `Trend:` block into `San Lorenzo 50Trend:` | Legacy repair assumes a 180-character block is split midword; this assumption is false for new generators |
| Unique source-clause restoration | 14 | `r1:u_003498_01` omitted `Four applicants are in the screening pipeline with the latest activity dates supplied.` | Paragraph compaction dropped a complete source sentence |
| Source-proven literal quote unescape | 10 | `r1:u_012131_01` rendered `GRUB_CMDLINE_LINUX_DEFAULT=\"quiet splash\"` instead of literal quotes | Transport escaping leaked into display text |
| Empty layout element removal | 3 | `r3:u_014465_01` contained an empty `List(items=[])` | Generator emitted empty scaffolding |

The omission examples establish an actual before/after defect. The upstream causes above are inferences from the output and repair logic, rather than measurements of the teacher's internal reasoning. Do not infer that every original generator review flag represented a genuine defect.

### Incorrect repair discovered during this review

The legacy `_join_midword_text_chunks` rule checks equal styles, a single inbound edge, an alphanumeric boundary, and a left Text length of exactly 180 characters. It does **not** check a source span proving that the two pieces form a single word. The package nonetheless labels the resulting changes `source_grounded=true`.

I inspected all 59 retained new rows with this repair kind. Their 61 changed boundaries delete actual source separators:

- 43 boundaries are proven against exact raw source strings.
- The remaining 18 are proven after removing only Markdown display syntax: emphasis/code wrappers, headings, and bullet/numbered line prefixes. Words, currency, comparator signs, numbers, and their order are preserved.
- Every offending joined boundary is present in the final decoded target. These are 59 affected rows: 57 train and 2 validation.

Examples include `50Trend:`, `RestWeek 5`, `MXN: 0Offer B`, and `USD 1,944Each additional dollar`. The initial raw-only scan found 43 rows and left the other boundaries unproven; the narrower display-syntax follow-up resolved them. The final catalogue records which proof was used for each boundary.

Two affected rows passed the existing `prepare_row(..., "root-first")` before the new preparation guard, demonstrating that schema, reachability, message alignment, and graph round-trip checks alone do not detect this defect. The reassessment tokenization also splits digit/letter boundaries into independent numeric and word tokens, so `50Trend` can retain the token evidence for both `50` and `Trend` while displaying a malformed boundary.

### Renderer-unsupported Chart labels discovered during this review

I streamed all 111,350 v11s rows, selected Chart candidates, and decoded their completions before inspecting component props. **1,183 rows contain unsupported declared Chart subtypes**: 1,159 train and 24 validation. This is not a regular-expression-only count of mentions in prose.

The current capability manifest permits `bar` and `column`, with `bar_chart` aliasing to `bar`. `FlatChartRender.kt` resolves aliases and explicitly emits `UNSUPPORTED_CHART_TYPE` plus an unsupported placeholder when the subtype is outside that inventory. Declared line/radar/etc. charts therefore do not acquire their requested chart semantics from a valid table of points.

Unsupported component counts include `line` 694, `pie` 180, `stackedbar` 101, `scatter` 90, `groupedbar` 85, `area` 70, `combo` 53, and `radar` 7. There are additional less frequent compound/variant strings; the complete distribution and row identities are in the [review catalogue](../data/quality/v11_review_holds_20261001.json). The supported declared component counts are `bar` 682 and `column` 18. Counts refer to components and overlap within a row.

The original v5.4 `effective_chart` path treated valid points/axes as a complete chart without enforcing the native subtype inventory. The supplied reassessment could consequently clear other diagnostics while preserving an unsupported chart. The current patch adds subtype enforcement to effective semantics; this is distinct from modifying the frozen v5.4 evidence in the release. No Android device rendering was performed during this review; the native behavior is source-established.

## Remaining quarantine: what still fails

The 20,050 supplied quarantined rows have overlapping reason counts. These are diagnostics, not independent defect rates or mutually exclusive totals.

| Recorded reason | Rows |
|---|---:|
| `content_unit_fidelity` | 11,829 |
| `generic_source_unit_not_exactly_represented` | 11,101 |
| `exact_numbers_dates_units_fbeta` | 6,208 |
| `action_and_source_link_fidelity` | 2,303 |
| `missing_or_mismatched_action` | 2,255 |
| `inferred_role_count_gap:chart` | 1,904 |
| `generic_numeric_anchor_not_represented` | 1,379 |
| `unsupported_visible_numeric_anchor` | 1,183 |
| `inferred_role_count_gap:email` | 661 |
| `semantic_role_difference:code` | 623 |
| `inferred_role_count_gap:form` | 518 |
| `source_clause_gap` | 484 |
| `table_content_difference` | 171 |
| `renderer_component_contract` | 132 |
| `dynamic_evidence_incomplete` | 50 |

Examples illustrate why these gates must stay separate:

- `r1:u_000257_01` retains numeric anchors but fails three exact source units describing coworking/software/food savings. Numeric presence alone is insufficient.
- `r1:u_000258_01` passes all 19 generic source units yet retains action/media mismatches. Copying visible prose does not prove a working action or media binding.
- `r1:u_004131_01` lacks source chart declarations `X-Axis: Clinic`, `Y-Axis: Coverage %`, and `Series: CHILD, ADULT`, alongside renderer-contract/role failures.
- `r1:u_000527_01` covers all 37 generic source units but adds unsupported numeric anchors. Some cases may be presentation numbering rather than factual invention; the package's conservative policy holds them, and classification requires source-aware review instead of blanket waiver.
- `r1:u_006184_01` passes a generic visible-text proof but retains `dynamic_evidence_incomplete` from the specific check. A visible-copy proof does not certify conditional/repeated behavior.

The package's 354-row pilot was deliberately stratified, not a random corpus accuracy study. The supplied `verification.json` reports 512 strict/content sample rows and zero sampled residual issues, but the full new-row audit above identifies defects outside that sample. These sample results must not be promoted to a claim that every retained row is semantically correct.

## Current training holds and usable counts

The immutable release remains fully available. The new `release_review.py` preparation guard checks the separately maintained review catalogue and native chart subtype support. Known text holds bind source identity and exact canonical semantic hash, so future regenerated labels can be considered independently rather than permanently excluding a source family. Chart checks apply to imported new-row lineage and use the current capability inventory.

The 59 text rows and 1,183 chart rows overlap in four training rows. Their union is **1,238 rows: 1,212 train and 26 validation**.

| Dataset | Frozen train | Frozen val | Held train | Held val | Expected remaining train | Expected remaining val |
|---|---:|---:|---:|---:|---:|---:|
| v11 | 200,214 | 4,089 | 1,212 | 26 | 199,002 | 4,063 |
| v11s | 109,121 | 2,229 | 1,212 | 26 | 107,909 | 2,203 |

These are expected counts after this review policy alone, not an already-produced tokenizer preparation result. Model/context-specific sequence limits and any other filtering can hold additional complete rows. Validation preparation may intentionally fail if a frozen evaluation cohort loses rows; select an explicit reviewed cohort instead of silently scoring a smaller set.

## How future generation should improve

1. **Preserve source units before optimizing layout.** Share training-fidelity instructions across teacher families. Inventory titles, complete sentences, section lead-ins, list items, qualifiers, numeric/date/unit anchors, table rows/columns, literal code, and required actions/media/semantic roles. Keep `Allocation from $4,800 net` and the film duration visible. Compact IR should reduce repeated structure/prose, not omit unique source content.
2. **Report exact source gaps in generation records.** Add a source/renderer fidelity diagnostic that can identify omitted full units and word boundaries, with bounded evidence and no literal-string clipping. Record the audit and guidance identity in Stage 3 metadata. Keep this diagnostic separate from existing training acceptance until calibrated; moving lexical recovery policy upstream is not proof that all paraphrases or additions are correct.
3. **Enforce native capability semantics.** Use the capability manifest for Chart subtype validation in effective semantics and generation audits. A requested unsupported pie/line/radar chart should cause a review/regeneration path. Do not silently relabel it as a bar chart and claim the original request was satisfied. Preserve the source/data and an explicit limitation where needed.
4. **Regenerate failed targets through Stage 3.** Provide bounded, specific failure evidence to the next generation attempt. Preserve originals, attempt history, prompt/model versions, finish reason, and hashes. Do not transplant the imported offline repair chain into normal generation, manually edit dataset IR, or join blocks on a 180-character heuristic.
5. **Check Stage 2 arithmetic and action provenance separately.** The v10 recoveries include a speed improvement of `100/9%` (about 11.1%), a revenue-per-lead metric mislabeled as conversion rate, a winning offer contradicted by its own totals, and a duration maximum of 30.5 hours with 49.25 hours total. These establish opportunities for declared arithmetic/unit checks before Stage 3. Avoid inventing action destinations to satisfy a role quota. None of these internal fixes verifies medical, legal, product, or externally sourced prose.
6. **Measure modality/role coverage as well as intent coverage.** All new retained rows use the Muse-Glimmer-30B teacher and still have `source_review_status=needs_review`. The package reports 101,329 answer rows, 5,014 documents, 4,655 dashboards, and only 352 interactive tools. New rows have List in 62,840 rows but Button in 1,820, Formula in 15, ChoicePicker in 29, and DateTimeInput in 4. Compared with v10, this is a substantial style/interaction distribution shift, not simply more examples of the same distribution. Generate targeted, source-grounded role cohorts without imposing unsupported UI roles on arbitrary requests.
7. **Preserve split and benchmark isolation.** v11s/train is contained in v11/train and cannot be an unseen evaluation set after v11 training. v11s/val is the unchanged new suffix of v11/val before review holds. Keep same/near source, query, and scenario families in one split. Reserve Golden32, Golden35, and Bixby50. The supplied rare-trigram near-duplicate search is bounded lexical retrieval, not exhaustive semantic equivalence; parameterized scenario variants may still require additional family checks.
8. **Validate model-specific training inputs without truncation.** New source p99 length is 6,077 characters and target p99 7,312 characters; maximum target length is 26,626 characters. Teacher generation token counts are not sequence-token counts for the selected student tokenizer/chat template. Run explicit model/context preparation and hold whole overlength rows.

The current implementation adds `dataset/prompts/stage3_training_fidelity_v1.md` through the common `run_stage3` path for all providers, persists `generation_fidelity_audit` plus guidance hashes, and fingerprints the extra prompt in augmentation resume configuration. The bounded renderer evidence configuration retains long concrete literals while keeping aggregate work/byte budgets unchanged. Unsupported Chart literals enter the bounded teacher-repair path, and final effective renderer validation enforces native subtypes, including defaults/aliases and resolved interpolation. Effective/component-contract provenance advances to 5.4.2. These changes keep the source, role gates, and original attempt evidence; archive repairs remain separate from live Stage 3 generation.

The pipeline implementation reports 174 passing CPU tests across its affected guidance, generation, augmentation, renderer/evidence, and metric paths. New live teacher generation, student training, quality/latency gains, tokenizer fit, and device rendering remain to be measured. Test success supports the implemented guards; it does not establish an empirical improvement in future generated training data.

## Evidence and verification scope

The candidate decode scan counts 2,088 Chart components, matching the supplied complete v11s component statistic. Reproduce the read-only boundary/subtype audit against restored release bytes:

```bash
python training/data/train/v11s/restore.py --output-dir /data/a2ui_v11s
python training/scripts/audit_v11_release_edges.py --dataset-dir /data/a2ui_v11s --output-report /data/a2ui_v11s_review_20261001.json
```

On Windows, use Windows paths. The audit reads the preserved compressed decision ledger by default; `--decision-ledger` accepts an alternate original JSONL/gzip ledger. It requires a new output file outside the dataset directory, refuses overwrites, and never modifies restored rows or the maintained hold catalogue. It records split/ledger/capability hashes. Its preparation samples reflect the guards installed when it runs; the earlier accepted results above were measured before installing the new guard.

- Current import integrity: `repository_import.json` records verification of all 308 package entries (10,617,480,483 bytes) and compressed ledger round-trips. This establishes source-package preservation, not the correctness of its labels.
- Current independent analysis: all 131,400 decision rows counted; all 111,350 v11s completions scanned; selected Chart candidates decoded; all 59 legacy-join rows and 61 boundaries inspected; two defective rows shown to pass the old preparation path. Exact identities, source/target/semantic hashes, original lineage hashes, source boundary fragments, and chart component IDs are in `v11_review_holds_20261001.json`.
- Current imported-tool validation: the task's import verification ran 75 supplied importer/repair/reassessment/verifier tests. The pipeline change reports 174 CPU tests passing. The reproducible review CLI matched all published review counts/classifications, counted the complete 2,088 declared Chart components, and demonstrated that the new preparation guard rejects two formerly accepted defects. Compilation/whitespace checks and overwrite refusal were checked for the audit script. No broad GPU/device inference claim follows from these results.
- Historical supplier verification: `verification.json` reports complete new-row original Stage 3 projection replay, 1,551 repaired-row replays, 97,975 reassessed rows, ten v10 repairs, and zero detected split/benchmark overlaps. The current task preserves this evidence but has not rerun the complete original Stage 3 lineage verification because its raw archives are not part of this import.
- Unverified: independent source fact checking, complete semantic/paraphrase equivalence, exhaustive scenario/benchmark overlap, chosen-tokenizer context fit, student model quality, GPU throughput, and final device rendering.

Primary evidence paths: `training/reports/dataset_v11_20260930/{SUMMARY.txt,REVIEW_NOTES.txt,cleaning_summary.json,statistics.json,verification.json,new_decisions.jsonl.gz,known_v10_repairs.json,policy_snapshot/}`, `training/data/quality/v11_review_holds_20261001.json`, `training/scripts/{space_v11_repairs.py,space_v11_reassessment.py}`, `training/src/ir_training/data/{archive_recovery.py,express_preparation.py,release_review.py}`, `dataset/schema/renderer_capabilities.json`, `dataset/src/pipeline/renderer_effective_semantics_v5_4.py`, and `android/app/src/main/java/com/samsung/genuicraft/renderer/FlatChartRender.kt`.

Final checkout validation passed 369 distinct regression tests: 176 dataset generation/metric/renderer/resume tests, 118 training bundle/review/preparation/cache/shared-prompt tests, and 75 imported release tests. An independent review also checked all 59 text holds through direct and shared-prompt preparation. The repeated-Chart gate now requires every rendered instance to be supported, regardless of instance order. Compilation, staged whitespace checks, and staged byte verification of 404 bundle/audit artifacts passed. See [current validation record](../reports/dataset_v11_20260930/repository_validation.json) and [complete source/message/split binding checks](../reports/dataset_v11_20260930/repository_binding_verification.json).
