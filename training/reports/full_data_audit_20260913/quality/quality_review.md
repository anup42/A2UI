# Full-data semantic and prompt-quality review — 2026-09-13

## Conclusion

Do not train the supplied files unchanged. They contain recoverable transport/envelope problems, confirmed evaluation contamination, and examples whose targets are syntactically valid but lose essential user-visible content. Strict IR validity alone is not an adequate admission rule for this corpus.

This review concerns `C:/Users/anupk/Downloads/training_data/train.jsonl` (150,292 rows) and `val.jsonl` (910 rows), not the smaller checked-in `dataset_v1` source. No source file, Golden reference, model weight, or training pipeline was modified. No teacher/model inference was performed.

## Evidence boundaries

- Exhaustive cheap text checks read all **151,202 physical rows** with their actual UTF-16 encoding. Counts below are exact matches to the implemented heuristics, not estimated prevalence and not factual-correctness scores.
- Detailed inspection selected **40 equally spaced train rows and 20 equally spaced validation rows**, including each file's first and last row. It added six low-lexical-recall train candidates and one eligible validation candidate: **67 sampled rows** total. All 67 passed the current `serialize_checked(..., "root-first")` checks. This is not an assertion that every corpus row passed; the separate exhaustive IR audit owns that result.
- The qualitative examples below were read against their complete source responses and targets. Their concrete omissions are confirmed; other automatically flagged rows require review.
- One additional train/validation near-duplicate pair identified by the inventory was inspected separately (train line 40,160 and validation line 209); both targets also passed strict validation. This pair is not part of the 67 deterministic/risk-selected samples.
- Full source hashes, schema/envelope counts, duplicates, and exact Golden overlaps are recorded by the independent [inventory audit](../inventory/audit_manifest.json). The [sample/heuristic report](sample_quality_summary.json) records the script hash and detailed sample signals.
- Raw sampled pairs and all per-row text signals are local, ignored audit artifacts under `training/outputs/audits/full_data_20260913/quality/`. They are not training candidates or modified references.

### Final structural validation and independent reference parity

The completed [exhaustive IR audit](../ir/summary.json) reports **150,292/150,292 train and 910/910 validation targets strict-valid**, with no binding errors and canonical serialization identical to every raw target. This is the target grammar/wire/binding boundary only: it does not establish source fidelity, usable action destinations, provenance, split isolation, token-budget suitability, or rendering quality.

The accelerated full scan was independently checked against the unmodified production `serialize_checked` function and `Draft202012Validator` on **164 distinct actual targets**: [100 initial checks](../ir/reference_parity_initial.json) plus [64 tail/validation checks](../ir/reference_parity.json). All valid/reason, semantic hash, and canonical-text hashes matched. The samples include 149 train and 15 validation targets, all 20 observed component types, graph-size/depth extremes, empty-layout and layout-only cases, train lines 1–149,758, and validation lines 1–910. No invalid-reason stratum existed in the completed index. These are bounded parity checks, not a claim that the slow production validator was run independently on all 151,202 rows.

The full index flags **1,956 train and 14 validation targets with at least one empty layout leaf**, and **three train targets containing only layout/divider component types**. An empty leaf can be harmless spacing; the 1,970-row queue is not automatically a confirmed defect count. All three layout-only cases were read in full and separately passed through unmodified production validation; each has only Card/Stack components and no component that displays its substantive source data:

| Train line | Source | Nodes / empty layout leaves | Confirmed content failure |
|---:|---|---:|---|
| 93,102 | Loan Cost Analysis Overview | 16 / 7 | Loan comparison figures remain only in unused state; recommendations and named actions disappear. |
| 111,526 | Couple's Monthly Financial Snapshot | 17 / 8 | Budget figures and expenditure rows remain only in unused state; displayed budget, insights, and actions disappear. |
| 121,983 | Game Night Parameters | 15 / 9 | Game-selection and roadmap rows remain only in unused state; visible game options, schedule, snack budget, and actions disappear. |

Thus these three are confirmed empty-content supervision, not harmless empty-leaf false positives. This conclusion follows from the full graph and source; no device-rendered screenshot is claimed. All three target hashes and strict-check results are preserved in the two reference-parity reports.

The exhaustive Text-node length histogram does **not** show a sharp corpus-wide 180-character cutoff: train lengths 179, 180, and 181 occur 2,076, 2,356, and 2,096 times respectively. The bounded clipping/fallback examples below remain valid, but neither this histogram nor those samples supports an all-corpus fallback-origin percentage.

## Exhaustive textual signals

| Signal | Train / 150,292 | Validation / 910 | Interpretation |
|---|---:|---:|---|
| Target contains a tracked mojibake marker | 39,036 | 236 | Encoding review required; not every marker is necessarily corruption |
| Source contains a tracked mojibake marker | 20,196 | 118 | Upstream source may itself be damaged |
| U+FFFD replacement character | 0 | 0 | Does not rule out valid-Unicode mojibake |
| TODO/TBD/Lorem ipsum text | 286 | 1 | Review only; placeholders can be legitimate user-requested content |
| Target has placeholder tokens absent from source | 42,808 | 288 | Requires original URL/media restoration map before judging correctness |
| Source has placeholder tokens absent from target | 47,435 | 323 | May indicate omissions or legitimate role remapping |
| At least one named source action label absent literally | 15,168 | 89 | Out of 144,657 / 873 rows with named source actions |
| Distinct source-word recall below 50% | 936 | 3 | Candidate queue, not a semantic verdict |
| Distinct source-word recall below 75% | 7,347 | 31 | Includes the below-50% queue |
| At least half numeric anchors absent | 3,248 | 23 | Only sources with at least three numeric anchors; formatting/URL numbers can trigger false positives |
| Exact generic `Button("Open Source")` | 6 | 0 | Review in context; the first train row confirms a harmful replacement |

The tracked mojibake markers are `ΓÇ`, `â€`, `Ã`, and U+FFFD. These are deliberately labeled heuristics: `Ã` can be legitimate language text, and other encoding-corruption patterns are not counted. For example, the reviewed `El Ni├▒o` corruption is not independently covered by this marker list, although that row also contains a tracked corrupted dash.

Lexical metrics operate on source text versus target program text for the exhaustive scan. The sampled report additionally examines decoded graph values. Neither method proves renderer-visible preservation: facts can remain in unused state without being displayed, as train row 93,102 demonstrates. Numeric strings may be reformatted or embedded in URLs; action labels may be paraphrased. Do not automatically delete every heuristic match.

## Confirmed semantic defects that pass strict IR checks

| Source location | Confirmed issue | Why it matters |
|---|---|---|
| Train line 1 — *90s Neo-Noir & Psychological Thriller Marathon* | Four film options and four scheduled viewings become the first two films and first two viewings. The second film description ends at `keeping`. Another text chunk splits `one` into `on` / `e of...`. Two named actions become an unbound generic `Open Source` button. Bullets are `ΓÇó`. | Teaches clipping, dropped records/actions, broken text, and decorative controls despite a valid program. |
| Train line 24,112 — *Senior DevOps Interview Guide* | All ten question/ideal-answer/red-flag groups are replaced by ten empty `Column` nodes. Heading, rubric, and footer actions remain. | Root reachability is complete, but the main requested content is gone. |
| Train line 93,102 — *Loan Cost Analysis Overview* | Some loan values remain in state, but the reachable UI tree consists only of empty Card/Column/Row containers: no Text, Table, Button, or Image displays the comparison. | State-value recall can appear better than actual rendered usefulness. Empty structural trees must not be treated as successful supervision. |
| Validation line 18 — Senior Project Manager application letter | Omits the 15% cost reduction, PMP qualification, and most letter paragraphs. Splits `business` into `busin` / `ess goals.` while keeping the signature. | Validation labels themselves reward incomplete answers and would conceal the same training failure. |
| Train line 3,855 — *El Niño Climate Phenomenon* | Both source and target contain `El Ni├▒o` and `DecΓÇôFeb`. | A UTF-16-to-UTF-8 file transcode alone will preserve these already-corrupted Unicode strings; it cannot restore the intended characters. |
| Train line 40,160 / validation line 209 — resignation letter | Same scenario, job, final date, tenure, and almost all wording; only transition/well-wishing wording differs. The final targets are exactly identical because they omit those differing paragraphs. | A confirmed near-duplicate across splits that the exact-source overlap count does not include. Also demonstrates distinct sources collapsing to the same incomplete target. This is one reviewed pair, not an exhaustive fuzzy-overlap count. |

These are not arguments to discard all data. For contrast, the reviewed train line 7,708 (*Project Timeline Overview*) retains the roadmap phases, numeric schedule, critical-path/buffer explanations, and both named actions. Train line 11,562 (*Fundraising Gala Roadmap*) retains its seven roadmap entries, three milestones, priorities, and named actions. Retain demonstrably faithful cases after provenance, split isolation, rendering, and token checks.

### Bounded diagnostic: legacy fallback signatures

The clipping pattern has a concrete code counterpart:

- `dataset/src/pipeline/flat_spec_contract.py:815-833` splits long sentences in fixed **180-character slices**, including mid-word splits.
- The same helper at lines **1034-1050** retains at most **four sections** and **three text chunks per section**.
- Lines **1094-1102** add a generic `Open Source` action when a first URL is available.
- This does **not** establish that the current training pipeline creates these labels. `dataset/src/pipeline/stage4_render.py:1585-1613` isolates the legacy fallback to explicit offline comparison rendering; the active Express render path rejects invalid completions instead.

An independent [bounded fallback-signature diagnostic](fallback_signature_check.json) compared the existing references with the current helper's result, followed by normal URL preprocessing and strict Express materialization. All candidates existed only in memory; no training labels or Golden targets were generated/exported.

Of the **67 saved qualitative samples plus two explicitly selected near-duplicate cases**:

- **3 exact semantic matches:** validation line 18, train line 40,160, validation line 209. That is one match among the original 67 samples and both targeted near-duplicate cases.
- **4 compatible but different** results, including train line 1.
- **62 not comparable through current strict Express materialization:** the current legacy fallback helper emits unsupported Table `sourceFormat`/`sourceText` properties. These diagnostic failures were preserved, not patched to manufacture matches.

The exact matches demonstrate a reproducible fallback signature, not historical provenance or an all-corpus fallback percentage. Nonmatches likewise do not disprove fallback ancestry: historical helper versions, URL maps, and later transformations can differ. Recover lineage and quarantine/deweight confirmed fallback-derived labels unless independently demonstrated to preserve the full source; do not promote rendering placeholders into teacher-quality supervision.

## Prompt, provenance, and Golden compatibility

The inventory confirms one identical short historical scaffold across **all 150,292 train and 910 validation rows**:

`b0ffc734d90f15ed7c5381ab8f91c9c90545ba1657650614b92012d64b85a7ce`

That is the raw archive Golden32 scaffold (32/32 occurrences), but **not** the current production scaffold used by raw Golden35 (35/35):

`0597ae864d7c49c34e472ac3d3d1186c5344824366746d057268eb4d2619dbf2`

These hashes describe the JSON-serialized scaffold message list, not the larger shared-prompt contract hash. The new pipeline intentionally normalizes training and both Golden views to the current production contract. Tokenization must therefore use that normalized prompt, not the shorter historical one. Do not change the Golden targets to accommodate a prompt or context limit.

The source envelopes contain only `messages`. They omit explicit completion/source fields, stable original query/response IDs, generator lineage, and URL/media restoration maps. This has two separate consequences:

1. The current `--input-dir` path cannot consume these files directly: its reader expects UTF-8, and its strict source/preparation guards require identity and completion fields that these envelopes lack. Converting the file encoding alone does not resolve the envelope contract.
2. The missing maps prevent distinguishing valid role aliases (for example, a SOURCE_URL restored through an ACTION_URL) from fabricated media/action references. Recover authoritative metadata by source/target hashes where possible; do not invent URL destinations or pretend hash-derived row IDs recover lost original query grouping.

The independent [Golden overlap report](../inventory/golden_overlap.json) confirms **26/35 Golden35 sources in training**, the excluded Golden32 source `q_012053` in training, and Golden32 source `q_019461` in validation. The [duplicate report](../inventory/duplicates.json) also identifies train/validation source overlap. These findings take priority over quality augmentation: reserve accepted **and excluded** Golden sources, and enforce group isolation before any training. A model already trained on these sources cannot be reported as an unseen Golden35 evaluation merely by removing rows from its later evaluation inputs.

## Prioritized filtering, recovery, and augmentation plan

### P0 — Establish an admissible, isolated data artifact

1. Preserve immutable source files and their SHA-256 inventory. Create a new UTF-8/LF candidate artifact; never overwrite the supplied archive. Encoding conversion changes container representation, not target semantics.
2. Reconstruct `completion`, `response_text`, source/query/response identity, URL map/assets, and provenance from authoritative Stage2/Stage3 records. Record exact join evidence and ambiguity. Quarantine unresolved or conflicting joins; do not bypass the training identity guard.
3. Remove all accepted/excluded Golden source matches from train and ordinary validation, using source aliases plus normalized raw/URL-masked response hashes. Resolve train/validation overlaps at the source group level, preserving all candidates from a source in one split. Review near-duplicate source families as well: exact text hashes alone miss the confirmed resignation-letter pair.
4. Normalize only the instruction scaffold to the saved production contract, then run exact model-tokenizer preparation. Keep complete examples; do not clip labels or sources to pass the 4,096-token limit. Keep every Golden reference and evaluate using the declared 2,048-new-token budget, reporting truncation as a runtime limitation.

### P1 — Add content-preservation gates beyond syntax

1. Fail targets with no renderer-visible semantic content for a substantive source. Inspect empty layout branches and unused state separately from element reachability.
2. For structured lists, comparisons, schedules, recipes, and instructions, check required entity/record coverage, numeric values, and bound named actions. Low lexical/numeric retention prioritizes manual review; it is not sufficient evidence by itself.
3. Recover the original uncorrupted response for mojibake cases. Investigate the importer/encoding path, preserve an auditable original-to-recovered mapping, and regenerate Stage3 targets. Do not apply an unverified global character replacement or manually patch target IR.
4. Quarantine the confirmed clipped/empty examples and similarly diagnosed families. Correct the upstream prompt/pipeline/renderer defect and regenerate from the authoritative source. Regenerated targets must pass strict, visible-content, action/media, and semantic checks before re-entry.
   Recover explicit fallback/repair lineage where possible. Exclude known incomplete legacy fallback labels; a retained fallback label needs an independent content-preservation review rather than merely a successful render or strict parse.
5. Curate validation independently using the same correctness criteria. Bad reference labels are not a legitimate test of fidelity; retain failed-reference evidence and publish a versioned clean validation artifact rather than silently hiding failures.

### P2 — Augment only after the clean retained distribution is measured

Use the exhaustive IR and token reports to compute the surviving unique-source distribution. Raw row count is not evidence of useful diversity. Target these task families with new, independent sources rather than copies/paraphrases of Golden content:

- Lossless multi-record content: four-plus items, long schedules, comparisons, and ten-part interview/checklist examples without clipping.
- Narrative/document preservation: full letters and email previews, qualifications, percentages, dates, monetary values, and signatures.
- Compact tables/cards: all records and visible state bindings, never expanded decorative trees or unused state only.
- Source-grounded controls/media: named actions with preserved destinations and complete restoration maps; no invented placeholder assets or unbound generic buttons.
- Unicode and multilingual typography: correct accents, punctuation, symbols, currencies, and units after verified encoding recovery.
- Golden component/intent gaps identified in the **post-filter, post-token** corpus, not assumed from the small checked-in dataset audit. Do not transfer the earlier `dataset_v1` EmailPreview/intent counts to this much larger archive.

The final [component-presence cross-tab](../synthesis/component_coverage.csv) establishes a useful starting point, before content-quality filtering:

| Component | Train rows / 150,292 | Validation rows / 910 | Golden32 rows / 32 | Golden35 rows / 35 |
|---|---:|---:|---:|---:|
| Table | 147,555 | 889 | 32 | 26 |
| Image | 3,294 | 25 | 0 | 12 |
| EmailPreview | 277 | 1 | 1 | 0 |
| Tabs | 3,550 | 18 | 1 | 0 |
| Formula | 18,689 | 124 | 3 | 1 |

Every component used in either Golden cohort occurs in the archive; there is no demonstrated zero-coverage Golden component. However, Image occurs in only about **2.19%** of raw training rows versus **12/35** Golden35 occurrences, and validation has only **one EmailPreview** example. These are coverage/validation-diversity priorities, not instructions to clone Golden proportions or evidence of a predicted Golden score. The overwhelming Table presence (about **98.18%** of training rows) should be reviewed against source intent: compact tables are useful for real matrices, but should not be the default for narrative/card tasks. Count unique, quality-approved source families after filtering before setting augmentation quotas.

Set acceptance criteria and held-out tests before generation. Keep source identity/group lineage for every augmented case. Golden32 and Golden35 remain frozen evaluation cohorts and must not become templates for near-duplicate training targets.

## Reproduction

From the repository root, with the normal audit dependencies installed:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:PYTHONIOENCODING='utf-8'
python training/scripts/audits/full_data_quality_20260913.py `
  --source-dir C:/Users/anupk/Downloads/training_data `
  --output-dir training/reports/full_data_audit_20260913/quality `
  --evidence-dir training/outputs/audits/full_data_20260913/quality
```

The script validates the expected physical row totals, records its SHA-256, and saves deterministic sample positions. Join `row_text_signals.csv` to the inventory/IR/token indices on split and 1-based physical line. Its output is an audit/review queue, not an automatically approved training dataset.
