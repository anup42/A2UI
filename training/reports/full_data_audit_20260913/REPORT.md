# Full training-data audit and improvement plan

Date: 13 September 2026. Checkout reviewed: `new_ir_changes_20260331`, HEAD `4f0a3347`.

## Verdict

**Do not start a long training run with these files unchanged.** The corpus is substantial, but its immediate problems are evaluation contamination, incomplete supervision, missing provenance, and import compatibility—not simply insufficient row count.

The strongest finding is that **26 of Golden35's 35 source responses already occur in the supplied training file**. An apparent improvement on those cases would not establish generalization. Strict-valid programs can also be poor labels: this audit found blank layouts, omitted records, lost named actions, and clipped sentences.

This is an analysis/reporting deliverable. Original data, Golden references, training code, and model weights were not changed. No training, teacher generation, model inference, LiteRT conversion, or Android rendering was launched. Audit scripts and reports were added; no commit or push was made for this analysis request.

## 1. What was actually audited

The supplied path `C:/Users/anupk/Downloads/training/_data` did not exist. The discovered folder, announced during the analysis and used here, is **`C:/Users/anupk/Downloads/training_data`**.

| Property | Training | Validation |
|---|---:|---:|
| File | `train.jsonl` | `val.jsonl` |
| Bytes | 1,970,021,858 | 11,850,436 |
| Physical records | 150,292 | 910 |
| Correctly decoded JSON/message envelopes | 150,292 | 910 |
| Encoding | UTF-16LE, BOM | UTF-16LE, BOM |
| Distinct exact source responses | 146,065 | 909 |
| Distinct raw target strings | 150,277 | 910 |
| Whole-row duplicates | 0 | 0 |

Combined: **151,202 records, 146,946 distinct exact sources, and 151,186 distinct raw targets**. Validation is only about 0.60% of the supplied rows. Every physical byte was accounted for by the inventory; independent file hashes matched.

SHA-256:

```text
train.jsonl  5c47fb3f07dc6f2711ece12a1a93b4ecd6977fa8060119f329db3189c2d6b910
val.jsonl    04e88d07003803279cea7de093bd9c69bcd35ed9a5fab057d577544b8c90ba19
```

These counts are specific to this folder. The earlier [archived training review](../../docs/reviews/20260905_genui_training_review/REVIEW.md) described 169,898 training rows and a reduced 1,024-row validation set. Those are **not this file population**. Without the earlier checkpoint's dataset hash/manifest, this audit cannot establish exactly which of these records a particular already-trained checkpoint consumed.

### Record and prompt shape

Every row contains only `messages`, with this five-message pattern:

```text
system:    historical short instruction
user:      fixed travel-checklist demonstration input
assistant: fixed demonstration target
user:      TASK_PREFIX + actual source response
assistant: actual A2UI Express target
```

The actual task is the **last user message**, not the first user message. The actual label is the **last assistant message**, not the fixed few-shot answer. The audit consistently uses this binding.

There are no explicit `completion`/`response_text` fields, original query/source/response IDs, run identifiers, assets, URL restoration maps, teacher versions, or generation/repair provenance. Matching source hashes against **211 current repository Stage2/Stage3 files, approximately 481 MB**, recovered only 26 training occurrences, all from Golden35; no validation source was recovered by that matching procedure. This is not proof that the original records no longer exist elsewhere.

All train/validation rows use the same historical scaffold, matching raw archive Golden32 but differing from raw Golden35/current production. Scaffold-list SHA-256:

```text
historical: b0ffc734d90f15ed7c5381ab8f91c9c90545ba1657650614b92012d64b85a7ce
production: 0597ae864d7c49c34e472ac3d3d1186c5344824366746d057268eb4d2619dbf2
```

These are hashes of the message-list representation, not interchangeable with the training manifest's broader shared-prompt contract hash.

### Character lengths

| Measure | Training median | Training p95 | Training p99 | Training max | Validation median | Validation max |
|---|---:|---:|---:|---:|---:|---:|
| Source characters | 2,187 | 3,490 | 4,417 | 9,732 | 2,171 | 4,597 |
| Target characters | 3,283 | 4,404 | 5,071 | 8,727 | 3,268 | 6,557 |

Longer target strings do not prove better content retention: generated syntax, repeated containers, and unused state can inflate length.

Full evidence: [inventory report](inventory/README.md), [file manifest](inventory/files.json), [structure](inventory/structure.json), [provenance match report](inventory/provenance.json).

### Source format and URL modes

The sources are overwhelmingly structured responses: 149,624 training rows contain detected Markdown headings, 141,979 have bullet-list lines, 125,572 have Markdown-table separator rows, and 141,900 have broader pipe-row markup. Only 242 have detected fenced code. These are explicit pattern counts, not intent labels or a full Markdown parse. Check whether this structured-source distribution matches deployment before claiming broad free-form-input coverage.

The corpus also mixes **two substantially different URL input modes**:

| Source mode | Train | Val |
|---|---:|---:|
| Literal HTTP(S) scheme, no typed placeholder | 76,899 | 451 |
| Typed placeholders, no literal HTTP(S) scheme | 72,580 | 452 |
| Both | 134 | 3 |
| Neither | 679 | 4 |

This reinforces why literal placeholder differences cannot automatically be treated as hallucinations. Define one deployment source-normalization contract, retain original and normalized source hashes, and reconstruct the deterministic URL registry only where its mapping can be proven. Already-placeholder-only sources with missing maps cannot recover their original destinations from token names. Do not blindly renumber placeholders or change target semantics to reconcile the two modes.

For Table review, the training cross-tab is:

| Source has detected Markdown table separator | Target contains Table | Target has no Table |
|---|---:|---:|
| Yes | 125,418 | 154 |
| No | 22,137 | 2,583 |

The no-separator/Table quadrant is not an error count: 141,429 of all Table-bearing targets have broader pipe-delimited source data. Reviewed sources at train lines 5 and 10 are real tables without dash separators, demonstrating false negatives. Non-table lists can also legitimately become card-presented Table data. Review representation and content, not regex disagreement alone.

There are 47,258 non-ASCII training sources and 103,034 ASCII-only sources. Unicode-block flags include 27,790 with Greek-range codepoints and 27,119 with box-drawing codepoints, but observed mojibake contributes to those counts. They **must not be reported as Greek-language or multilingual-example counts**. See the [source-structure report](source_structure/README.md) for all character/markup flags and limitations.

### Full-corpus tokenizer measurements

All 151,202 data records and all 67 Golden occurrences were tokenized without truncation or token-audit rejection. Source/target counts below use the exact locally available Gemma3 vocabulary. Chat-sequence counts use an explicitly **reconstructed Gemma3 frame**, because the bundle's real chat template is missing. They are **not E2B token counts or certified deployment-template counts**.

| Measure | Train median | Train p99 | Train max | Validation max |
|---|---:|---:|---:|---:|
| Source tokens | 588 | 1,285 | 2,616 | 1,344 |
| Raw target tokens | 1,013 | 1,542 | 2,474 | 2,014 |
| Old-scaffold reconstructed full sequence | 1,841 | 2,881 | 4,188 | 3,414 |
| Production-scaffold reconstructed prompt | 2,092 | 2,789 | 4,120 | 2,848 |
| Production-scaffold reconstructed full sequence | 3,120 | 4,160 | 5,467 | 4,693 |

- Production normalization adds **1,279 tokens** to each historical-scaffold sequence under this framing. Mean train sequence length rises from **1,829.809 to 3,108.809**. This explains why lengths must be recalculated after prompt normalization.
- **1,945 train rows (1.294%) and six validation rows exceed 4,096 full-sequence tokens**; none exceed 6,144. Only three training rows exceeded 4,096 with the old scaffold.
- **47 training targets exceed 2,048 raw target tokens**; 49 exceed it when the reconstructed end-of-turn suffix is included. No validation target exceeds either budget. A training sequence limit and an inference generation limit are different controls: these counts do not automatically require deleting every longer training label.
- Keep a 4,096-token main lane if the actual model-tokenizer audit confirms this distribution, and review a separate complete-example long lane for the rare overflow cases. Do not increase every batch's maximum merely to accommodate a small tail without measuring padding/memory cost. Length bucketing is worth benchmarking on the training host.

| Golden cohort | Maximum reconstructed prompt | Maximum raw target | Target plus turn suffix | All references fit 2,048 output tokens? |
|---|---:|---:|---:|---|
| Golden32, 32 occurrences | 2,539 | 1,430 | 1,432 | Yes, with this vocabulary/frame |
| Golden35, 35 occurrences | 2,857 | 1,539 | 1,541 | Yes, with this vocabulary/frame |

All Golden prompts fit 4,096. One Golden35 **prompt-plus-reference** sequence reaches 4,398; that is not a reason to drop it from generation evaluation, where the reference is not fed into the prompt. Keep the default 2,048-new-token evaluation budget and report actual generated truncations. Correct-reference length does not guarantee that a model will stop within it.

Tokenizer SHA-256: `a306777fab20c8efd76229cea1566e9ec976059a4f2fb73565bc944ff3c2df4f`. The saved tokenizer contains one added token beyond the model config's stated vocabulary; no audited source/target uses the tested special-token marker strings. Still verify tokenizer/model vocabulary and template alignment on the actual model bundle before training.

See [complete token evidence](tokens/summary.json) and [token methodology](tokens/README.md). The reconstructed framing follows the documented user/model turn structure, but the missing saved template remains a real launch blocker, not something this reconstruction repairs. [Official Gemma prompt structure](https://ai.google.dev/gemma/docs/core/prompt-structure).

## 2. Golden contamination and split leakage

### Golden isolation fails

| Cohort | Confirmed overlap | Location |
|---|---|---|
| Golden35 accepted | **26 of 35 sources in training** | 26 rows between train lines 73,347 and 73,379; exact mapping linked below |
| Golden32 accepted | `q_019461` in validation | val line 85 |
| Golden32 excluded invalid source | `q_012053` in training | train line 131,726 |

The matching uses exact, whitespace-normalized, and production URL-masked source hashes. These are exact implemented content matches, not model similarity guesses. Golden32 is the archive-repeat benchmark: **32 occurrences but 31 unique source cases**. Golden35 contains **35 unique accepted cases**. No benchmark rows were rewritten or expanded during this audit.

**Action:** reserve the complete accepted and excluded source families of both benchmark contracts before training-data filtering or splitting. Repeating a valid case to preserve the declared Golden32 occurrence count does not create a new independent test case; report both occurrence-weighted and unique-source scores.

If a prior checkpoint actually trained on the matched Golden35 responses, removing them from a subsequent data file does not undo what that checkpoint learned. For an honest clean benchmark, restart from an uncontaminated base/seed and use newly isolated data. Do not claim the supplied folder alone proves contamination of every historical checkpoint.

### Train/validation isolation also fails

- **28 exact sources cross the split**, affecting 28 of 910 validation rows (**3.08%**). Different target variants do not make the same source held out.
- Training has **4,210 same-source/multiple-raw-target groups**, affecting **8,437 rows**. Validation has one such group, affecting two rows. Some may be legitimate layout alternatives; others may be quality conflicts. Do not blindly retain the first target or call all alternatives duplicates.
- A separately confirmed near-duplicate, **train 40,160 / val 209**, describes the same resignation-letter scenario with slightly different transition/well-wishing paragraphs. Its target strings are identical because both omit those differing paragraphs. It is not included in the 28 exact-source count.
- A bounded lexical scan of every source recovered all 28 exact overlaps and four nonexact pairs, all involving val 209 (train 4,697; 19,609; 40,160; 51,046). This is **one additional validation source-family case**, not four. Some variants change facts, so grouping them must not erase their factual differences. The [near-duplicate audit](near_duplicates/README.md) documents thresholds and missed-paraphrase/containment risks; it does not certify all other sources independent.
- Original source-family IDs are missing, so exact hashes alone cannot recover all scenario/paraphrase groups.

The row-level [Golden overlap map](inventory/golden_overlap.json), [cross-split source map](inventory/cross_split_sources.json), and [duplicate summary](inventory/duplicates.json) make these findings actionable without changing the source files.

## 3. Structural and semantic supervision

### Completed strict target audit

**Every one of the 150,292 training targets and 910 validation targets passed** strict Express parsing, wire assertions, renderer-reference/reachability checks, canonical emission, and semantic roundtrip. All targets already equal the production root-first canonical serialization. The audit's final-role, task-prefix, and few-shot checks passed throughout; this does not supply the missing production aliases, identities, or preparation manifest. The existing graph-repair probe proposed no changes.

The complete pass validated **151,186 unique raw targets** and mapped the results back to all 151,202 occurrences. Independent unmodified production-validator checks on **164 varied real targets** agreed on validity, reason, semantic hash, and canonical text hash, with zero mismatches. The wire-schema acceleration also agreed with the original validator on all 67 Golden payloads and 67 deliberately invalid copies. This is bounded reference-engine parity plus exhaustive accelerated validation, not a claim that all targets were rerun through both engines.

| Structural measure | Train | Validation |
|---|---:|---:|
| Strict-valid targets | 150,292 / 150,292 | 910 / 910 |
| Already canonical | 150,292 / 150,292 | 910 / 910 |
| Median elements per graph | 35 | 34 |
| p99 elements | 60 | 59 |
| Maximum elements | 112 | 102 |
| Rows with at least one flagged empty layout leaf | 1,956 | 14 |
| Rows containing only layout/Divider types | 3 | 0 |

Empty-layout flags require context: a dynamic or state-backed container can be legitimate. Element reachability alone does not establish useful content. All three layout-only training candidates were independently inspected and confirmed blank for substantive sources: **93,102 (loan comparison), 111,526 (monthly finances), 121,983 (game-night parameters)**. Their facts remain in unreferenced state, with only spacing/layout containers in the UI. This is source/graph verification, not an actual device render. There is no structural justification here for running a blanket ID/schema-repair pass over all labels. The major remaining defects are outside those syntax checks.

Two training source/target pairs are semantically duplicated despite different raw target strings: line pairs **111,716 / 131,190** and **107,845 / 128,149**. After semantic comparison, **4,208** training source groups still have multiple distinct target graphs. Target diversity must be reviewed for fidelity, not confused with byte-level deduplication.

### Component coverage relevant to the Goldens

Counts below mean **records containing a component**, not number of nodes, unique scenarios, or approved clean labels. All raw targets are strict-valid; contamination and semantic warnings remain in these raw denominators.

| Component | Train / 150,292 | Val / 910 | Golden32 / 32 occurrences | Golden35 / 35 |
|---|---:|---:|---:|---:|
| Stack | 150,292 | 910 | 32 | 35 |
| Card | 148,706 | 901 | 32 | 35 |
| Text | 150,282 | 910 | 32 | 35 |
| Table | 147,555 | 889 | 32 | 26 |
| Icon | 147,340 | 887 | 31 | 28 |
| Button | 134,504 | 823 | 29 | 28 |
| Formula | 18,689 | 124 | 3 | 1 |
| Image | 3,294 | 25 | 0 | 12 |
| Tabs | 3,550 | 18 | 1 | 0 |
| EmailPreview | 277 | 1 | 1 | 0 |
| Divider | 1,696 | 11 | 0 | 1 |

All component types used by the two Golden cohorts occur in training. That does **not** prove adequate property combinations, visible fidelity, or source diversity. In particular:

- Image appears in **2.19%** of training records versus **12/35 Golden35 cases (34.29%)**. Image/action restoration maps and independent image-bearing examples deserve targeted review. Do not simply oversample existing corrupted/unbound image labels.
- EmailPreview exists in **277** training records, not zero, but only **one** validation record. Strengthen independent validation for complete email/document content; do not infer capability from one case.
- Table appears in **98.18%** of training records, but this is substantially source-driven: **83.55%** of source responses have detected Markdown-table separator rows and **94.42%** have weaker pipe-row markup. Table frequency alone is not evidence of an unjustified layout habit. The source-structure analysis below identifies more useful review subsets.
- Other supported behaviors are very sparse: CodeBlock 594 rows, ConsoleLog 500, CheckBox 163, Slider 112, TextField 88, ChoicePicker 70, AudioPlayer four, List three, Video one. Most are not exercised by either current Golden cohort. Augment them only if they are relevant to the intended deployment, and add independent tests.

The [complete coverage CSV](synthesis/component_coverage.csv) also shows counts after known-Golden-hash, URL-error, and reconstructed-length screening—for example, Image 3,203 and EmailPreview 277. Those are still **not semantically approved retained counts**. True intent/domain labels and original family metadata are absent, so no authoritative intent-balance chart is claimed.

### Confirmed content problems

The detailed qualitative audit inspected 40 evenly spaced training rows, 20 evenly spaced validation rows, and seven additional risk-selected examples. It separately inspected the near-duplicate pair above. These examples were compared against their complete source responses; **this was not manual review of every row**. Exhaustive automatic signals below cover every row but remain heuristics.

| Confirmed example | Problem despite passing strict target checks |
|---|---|
| Train 1, film marathon | Four films and four schedule entries become two of each. A sentence ends at `keeping`; another splits `one` across chunks. Named actions become an unbound generic `Open Source` button. |
| Train 24,112, DevOps interview guide | Ten question/answer/red-flag groups become ten empty Columns. The main content disappears. |
| Train 93,102, loan comparison | Loan data remains in unused state, but the reachable tree contains only empty layout containers and displays no comparison facts. |
| Val 18, application letter | Drops the 15% cost reduction, PMP qualification, and most body paragraphs; splits `business` into `busin` / `ess goals.` |
| Train 3,855, climate explanation | `El Ni├▒o` and `DecΓÇôFeb` appear in both source and target. The source is already damaged. |

Other reviewed examples retain their full timelines, numeric details, and actions. The corpus is not uniformly bad. The appropriate response is targeted review/regeneration and a stronger admission contract, not deletion of all complex records.

### A reproducible clue behind the clipped labels

The repository's legacy fallback helper slices long sentences every **180 characters**, which can cut words, and keeps at most **four sections with three chunks per section**. It can also add a generic `Open Source` button. See [text chunking](../../../dataset/src/pipeline/flat_spec_contract.py#L815) and [section/chunk limits](../../../dataset/src/pipeline/flat_spec_contract.py#L1034).

A bounded diagnostic compared its in-memory output with the 67 qualitative samples plus the two selected resignation-letter cases. **Three targets matched exactly by semantic hash:** val 18, train 40,160, and val 209. Four other comparisons were compatible but different; 62 could not pass current Express materialization because the legacy helper emits unsupported Table properties. Those failures were retained, not patched to force a match.

This is concrete fallback-signature evidence for those labels, **not proof of historical provenance or a full-corpus fallback percentage**. The current caller is isolated to legacy rendering comparison, not evidence that today's training pipeline invokes it. The full Text-length histogram has 2,356 training Text values of length 180 versus 2,076 at 179 and 2,096 at 181; it does not show a sharp global cutoff that would justify attributing all clipping to this helper.

Recover fallback/repair lineage where possible and exclude known incomplete fallback-derived labels. A successful render placeholder must not be promoted to teacher-quality supervision without independent content verification. [Fallback diagnostic and limitations](quality/fallback_signature_check.json).

### Exhaustive warning queues

| Signal | Training rows | Rate of all training rows | Validation rows |
|---|---:|---:|---:|
| Tracked mojibake marker in target | 39,036 | 25.97% | 236 |
| Tracked mojibake marker in source | 20,196 | 13.44% | 118 |
| Target has placeholders absent from source | 42,808 | 28.48% | 288 |
| Source has placeholders absent from target | 47,435 | 31.56% | 323 |
| At least one named source action label absent literally | 15,168 | 10.09% | 89 |
| Distinct source-word retention below 50% | 936 | 0.62% | 3 |
| Distinct source-word retention below 75% | 7,347 | 4.89% | 31 |
| At least half numeric anchors missing, with at least three in source | 3,248 | 2.16% | 23 |
| TODO/TBD/Lorem ipsum markers | 286 | 0.19% | 1 |

**Do not add these rows together:** the queues overlap. They are not automatic failure counts. Literal word/action matching misses valid paraphrases; numeric formatting changes can be harmless; tutorial text can legitimately contain placeholders. The mojibake marker `Ã` can occur legitimately in some language text, and the marker list also misses other corruption patterns.

Placeholder mismatch is especially ambiguous because original restoration maps are missing. A source URL can legitimately be assigned an action/media role during preparation. Conversely, an unmatched target token can be fabricated. Recover the authoritative maps before deciding which is which; never invent destinations.

### Repair versus filtering

| Condition | Recommended disposition |
|---|---|
| UTF-16 container encoding | Losslessly transcode into a **new** UTF-8 artifact, verify decoded row/content hashes, retain original unchanged. |
| Missing aliases with intact final turns | Audited wrapper/import step may reconstruct aliases from those turns without changing label content. |
| Missing original IDs/maps | Recover exact authoritative provenance; otherwise explicitly mark unresolved lineage and quarantine cases requiring it. A hash ID is not an original scenario ID. |
| Invalid IR or empty output for a substantive source | Quarantine, diagnose upstream generation, regenerate Stage3 from the authoritative response. |
| Missing facts, clipped words, dropped actions | Correct upstream prompt/pipeline/renderer; regenerate and revalidate. Do not manually patch IR labels. |
| Mojibake in response or target | Recover original text/encoding path; only use a verified reversible recovery where unambiguous. Regenerate ambiguous targets. |
| Same source with several good layouts | Keep together in one split; choose/cap verified variants or use source-balanced sampling. |
| Length overflow | Keep full examples in a separately configured long-example lane or quarantine for diagnosis; never truncate required content to manufacture a pass. |
| Golden overlap | Exclude the complete source family from training and ordinary validation, regardless of label quality. |

Full examples, methods, and interpretation limits: [semantic quality report](quality/quality_review.md) and [sample/signals JSON](quality/sample_quality_summary.json).

## 4. Direct pipeline compatibility

The supplied folder is **not a ready input artifact for the checked end-to-end launcher**:

1. The repository JSONL loader uses UTF-8; these UTF-16LE files fail before ordinary row checks. It also skips malformed/non-object JSON rows rather than recording them, which should be replaced by an explicit accounting/quarantine path for audited imports.
2. `prepare_row` requires `completion`, while these envelopes only contain `messages`.
3. Prepared-data verification requires a canonical response plus a query/source identity and tokenizer-bound manifest. Those are absent here.
4. Four otherwise valid envelopes trigger **real production URL-masking exceptions**: train **107,449; 110,047; 127,855; 138,115**. Error classes include `Invalid IPv6 URL` and a bracketed `Your-Public-IP` placeholder being interpreted as an IP address. Catch and quarantine these records with diagnostics; do not let an entire preparation run abort without row accounting.
5. One production scaffold must be used consistently for training, both prepared Golden views, and deployment. Changing only evaluation instructions would measure a prompt shift as well as model ability.

Relevant implementation: [strict row preparation](../../src/ir_training/data/express_preparation.py), [JSONL reader](../../src/ir_training/common/jsonl.py), [prepared recipe verification](../../scripts/prepare_review_training.py).

A deliberate import contract can use a content-addressed source identity **as a new identity scheme**, with original file hash, physical line, source hash, target hash, and lineage status. It must not misrepresent generated hashes as recovered original IDs, nor assume they detect paraphrase families. Preserve a versioned source-family mapping and its review evidence.

### Model bundle caveat

The only local vocabulary available for this audit is the saved Gemma3 270M tokenizer in `C:/Users/anupk/Downloads/training/runs/gemma270m_ir_lora/merged_hf`. Its local bundle has **no usable chat template**; a real `apply_chat_template` call fails. No E2B tokenizer was available locally. Restore/pin the correct original model tokenizer and template before producing the final model-specific prepared dataset. Do not copy a guessed chat template into a training bundle on the strength of this audit's length estimates.

### Trainer issues remain separate from data quality

The current 270M review recipe loads BF16 weights for full-parameter training, and the full-finetune branch enables gradients without creating explicit FP32 master trainable weights. Verify genuine optimizer-step updates and numerics on the training host; finite forward loss alone is insufficient. The E2B LoRA path is a different parameter/precision case. Better labels do not substitute for this check.

For the current checked review route, normal loss evaluation/save cadence is **500 optimizer steps**, Golden32 generation cadence **1,000 optimizer steps**, and final evaluation is enabled; Golden35 is bound as the **final-only holdout**. The generation budget remains **2,048 new tokens**. Default TensorBoard root resolves to **`/tensorboard`**. Use unique run IDs, scorer/prompt/tokenizer fingerprints, and dataset hashes to avoid mixing evidence from different runs.

The existing multiformat recipes still name the older September-3 Golden32 source rather than this archive-repeat32 plus Golden35 pair. A data-cleaning report does not certify the separate QAT/LiteRT export-and-evaluation route as integrated. Complete that handoff before claiming one command covers HF plus every LiteRT variant and both current benchmarks.

## 5. Recommended clean-data plan

### How many records remain after basic screens?

This is an **analytical screening scenario**, not a produced clean dataset or a permission to train these counts. It applies known accepted/excluded Golden source-hash removal, strict target validity, URL-error quarantine, and the reconstructed 4,096-sequence / 2,048-target scenario. It does not resolve source-family splits, missing provenance, semantics, or actual E2B template lengths.

| Cumulative row screen | Train occurrences | Validation occurrences |
|---|---:|---:|
| Supplied | 150,292 | 910 |
| Remove known reserved Golden source-hash matches | 150,265 | 909 |
| Strict target check | 150,265 | 909 |
| Quarantine production URL-preprocessing exceptions | 150,261 | 909 |
| Reconstructed sequence ≤4,096 and target ≤2,048 | **148,318** | **903** |

Within that final technical screen, **82,866 training and 525 validation records have at least one tested content/encoding/placeholder/layout review signal**. The remaining **65,452 / 378** merely have none of these selected signals; they are **not certified faithful or leakage-free**. Blanket-dropping the warning cohort would discard many potentially legitimate URL-role aliases and paraphrases. Review high-confidence failures first and calibrate the rest.

These are occurrence counts without semantic-pair deduplication. Separately, after Golden-hash/strict/URL screens, there are 150,259 distinct train source/semantic-target pairs and 909 validation pairs. Do not substitute that distinct-pair diagnostic into the row funnel. The [joined summary](synthesis/summary.json) documents the exact units, caveats, and overlapping flags; its full local queue contains one coordinate for every input row.

### Priority 0: make measurement trustworthy

1. Freeze source hashes and all accepted/excluded Golden source-family reservations.
2. Recover/import envelopes with explicit lineage; losslessly create a new UTF-8 candidate version.
3. Exclude Golden families from both training and ordinary validation; preserve an exclusion manifest.
4. Group exact, normalized, URL-masked, known original-ID aliases, and confirmed near-duplicate scenario families transitively before splitting. Use run-qualified original IDs when available; IDs such as `q_000001` are not globally unique across dataset runs.
5. Select verified target variants at the source-family level. Rebuild validation from clean families; do not patch individual overlap rows while leaving their siblings behind.

### Priority 1: improve the supervision itself

6. Run strict parser, wire schema, renderer references/reachability, canonical semantic roundtrip, and alias-binding checks with explicit per-row results.
7. Add a **renderer-visible content** gate: required records/entities, numbers/units, dates, ordered steps, named actions and destinations, media mappings, and nonempty semantic content. Facts sitting in unused state are not rendered facts.
8. Triage clipped words, blank layout branches, low retention, mojibake, and ungrounded actions/assets. Diagnose high-confidence failures and regenerate Stage3; use sampled manual adjudication to calibrate automated flags before treating them as hard filters.
9. Apply the same content standard to validation labels. Publish a new validation version with rejected-reference evidence rather than silently shrinking the denominator of an existing benchmark.
10. Normalize the saved production scaffold, canonicalize only with semantic-equivalence checks, and tokenize with the **actual selected model/template**. Enforce complete supervision; do not silently truncate.

### Priority 2: augment independently, after measuring retained coverage

Do not generate more of the same damaged examples first. Create new independent sources for confirmed weak capabilities, with a saved generation manifest and content acceptance checklist:

- Four-plus entity comparisons, long schedules, multi-part guides, and nested content where every requested item must survive.
- Complete letters/document summaries preserving qualifications, exact numbers, body paragraphs, and signatures.
- Compact table/card representations with valid state bindings and every required row visible.
- Grounded named actions, complete asset maps, and semantically useful controls; no decorative unbound replacement buttons.
- Correct Unicode/accents/symbols/currencies and independently generated multilingual sources where deployment needs them.
- Rare component/property combinations identified in the **retained** corpus; no copied or paraphrased Golden scenarios.

Use an audited, stratified initial batch to verify yield before scaling augmentation. Teacher approval alone is not enough: retain strict, content, asset/action, and render checks. Dataset generation belongs in `dataset/`; `training/` should consume the resulting completed Stage3 records.

### Proposed manifest for the next version

Each candidate should bind: source-file SHA and physical line; new stable record ID; original run/query/response IDs when genuinely recovered; source-family ID and grouping method; raw/normalized/masked source hashes; raw and canonical target hashes; semantic hash; prompt/scaffold/tokenizer/template fingerprints; original URL map/assets; validator/scorer fingerprints; token lengths; quality flags; admission/quarantine decision and reason; teacher/regeneration provenance.

Keep the original validation as evidence. A new source-family-disjoint development split of roughly **1–2%** is a reasonable starting proposal for this corpus, with coverage checked rather than relying on a random row split. Separately curate a larger untouched acceptance set, initially perhaps **500–1,000 cases**, because 31 unique Golden32 cases and 35 Golden35 cases cannot robustly cover every supported UI/domain behavior. These are proposed design sizes, not measured optima.

## 6. How to establish that training really improves

Use a controlled sequence: clean supervised baseline first; add verified repaired cases; then add targeted independent augmentation. Keep model seed, tokenizer/template, decoder, scorer, and optimizer budget fixed for each comparison where practical. Report training examples/tokens as well as optimizer steps so GPU count or effective-batch changes do not hide differences in data exposure.

Golden32 remains a development/regression checkpoint-selection set. Use its **unique-source** score for selection and publish the repeated 32-occurrence score separately. Golden35 remains final-only; once its scores guide repeated tuning decisions, it is no longer an untouched final test and a new holdout is needed.

For HF checkpoint and every genuinely supported LiteRT variant, record:

- All-case score and denominator; no silently omitted failures.
- Strict parse, wire validity, reachability, and canonical semantic validity.
- Renderer-visible fact/record recall and numeric fidelity, with invalid outputs contributing failure rather than disappearing from the average.
- Action/media grounding, empty-content failures, and generation truncation rate at 2,048 new tokens.
- Latency, generated tokens, memory, backend/format, model/checkpoint hash, data/prompt/template/scorer hashes, and per-case outputs.
- Paired change from the actual HF checkpoint on the same cases; repaired-output diagnostics separately from raw production scores.

Log training loss, independent validation loss, gradient/update checks, throughput, sequence-length buckets, and checkpoint Golden32 metrics under a unique `/tensorboard/<run-id>/...` tree. Store final Golden35 and export comparisons under separate named series. No improved score, GPU throughput, or LiteRT accuracy is claimed by this data-only audit.

## 7. Interpretation limits

An exhaustive programmatic audit is not exhaustive human verification. Cheap lexical/encoding flags are only candidate queues. Structural validity does not prove a useful rendered UI. No original teacher lineage, authoritative URL maps, deployment render screenshots for these 151,202 targets, actual E2B tokenization, or live model predictions were available for full semantic certification.

The frozen Golden references were checked for their current structural contracts, lengths, components, and source overlap; this audit did not independently certify every reference's factual/rendered completeness. Preserve reference-quality review evidence separately from model scores rather than treating a strict-valid reference as automatically perfect.

The strongest recommendation is therefore **clean isolation and faithful supervision before more epochs, broader LoRA targets, MTP, QAT, or more generated rows**. Those training/export choices should be evaluated after the data and measurement contract is trustworthy, not used to compensate for defective labels.

## 8. Audit artifacts and reproduction

- [Inventory and contamination](inventory/README.md)
- [Semantic quality and examples](quality/quality_review.md)
- [Bounded near-duplicate source-family analysis](near_duplicates/README.md)
- [Source markup, scripts, and URL-mode census](source_structure/README.md)
- [Full structural validator evidence](ir/validator.json)
- [Full structural results](ir/summary.json)
- [Independent production-validator parity, first 100 targets](ir/reference_parity_initial.json)
- [Independent production-validator parity, another 64 targets](ir/reference_parity.json)
- [Token results](tokens/summary.json)
- [Canonical target token results](tokens/canonical_summary.json)
- [Joined filtering/coverage summary](synthesis/summary.json)
- [Component coverage CSV](synthesis/component_coverage.csv)

Reusable scripts live in [training/scripts/audits](../../scripts/audits). Large row-level indices, source examples, and review queues remain local and ignored under `training/outputs/audits/full_data_20260913/`; no large training copy or checkpoint is added to Git. The small reports provide hashes, methods, counts, and exact example locations.

The structural pass compiles the unchanged wire-schema assertion set for speed after checking keyword compatibility and production-validator parity; parser, renderer-reference checks, canonical emitter, and semantic roundtrip remain production implementations. Schema defaults and format checks are disabled to match the reference validator. The report records this acceleration explicitly; it does not pretend every target traversed the original slower schema engine.

Runtime used for this audit: Python **3.12.10**, `jsonschema` **4.26.0**, `fastjsonschema` **2.21.2**, `tokenizers` **0.23.1**, and `transformers` **5.16.1**. These are observed audit-environment versions, not a newly certified GPU training dependency lock. Graph depth in the structural audit means maximum depth observed by its first-visit renderer-reference traversal, not guaranteed longest-path depth of a shared-node DAG.

Verification included **four passing audit regression tests**, critical Ruff checks, script compilation, zero tracked-diff whitespace errors, cross-index row/hash reconciliation, original source SHA-256 rechecks, and local-link validation. A final small [audit manifest](audit_manifest.json) records report and relevant code hashes. The full GPU training test suite was not rerun for these report-only additions, and no GPU execution is claimed.

Run the four primary audit scripts with `--help` for input/output options. Existing SQLite destinations are intentionally not overwritten by the inventory/structural/synthesis passes; use new audit destinations to reproduce. No command in this report starts training.
