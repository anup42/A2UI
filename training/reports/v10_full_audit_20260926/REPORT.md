# v10 complete dataset audit — 26 September 2026

## Assessment

**Do not treat unchanged v10 as a quality-approved dataset for the current Bixby/Perplexity converter.** Its syntax/graph integrity is strong, but source-to-target omissions, a pronounced production-input mismatch, prompt-budget expansion, and incomplete model lineage need attention before another long training run. More epochs alone do not address those issues.

This audit freshly processes **92,977 records: 91,115 train + 1,862 validation**, from `training/outputs/datasets/full_data_archive_recovered_v10`. The restored files match the tracked `training/data/train/v10` bundle byte-for-byte. This audit did not change original datasets, training code, model weights or app code. It covers base v10 before any optional augmentation; no augmented rows were generated or audited. No new model training, export or device inference was performed.

“Complete” means every row received the enumerated machine checks. It does **not** mean every source fact or rendered screen was manually certified. Semantic evidence combines full-corpus candidate checks, 20 newly sampled complete-pair reviews, targeted examples, and verification that the previous audit's known defects remain present.

## Highest-priority findings

| Priority | Finding | Training/runtime consequence | Recommended handling |
|---|---|---|---|
| P1 | Synthetic media/action/URL scaffolding dominates v10; numeric citations are nearly absent. | If v10 is used unchanged, its supervision differs substantially from the measured Bixby50 response style. | Add a separate development cohort resembling real production responses, with faithful generated targets, natural prose, tables, citations without URLs, and no forced media/actions. Keep Bixby50 out of SFT. |
| P1 | Current preparation replaces the old 4,766-character system prompt with the 10,553-character deployed contract. | Many valid rows exceed 4,096 tokens under the local proxy; current preparation is designed to quarantine such rows. Actual retention requires the HF manifest. | Bind the exact training tokenizer and prepared manifest, measure retention, then choose a budget/prompt supported by the app. Never truncate targets to make them fit. |
| P1 | Confirmed targets omit schedules, passenger fields, short instructions, requested charts and entity relationships. | The labels teach incomplete conversions even though they compile successfully. | Add semantic regression gates, repair the generator/source where necessary, regenerate Stage 3 targets, and quarantine unresolved pairs. Do not manually rewrite target IR. |
| P1 | Training uses URL placeholders; the trained SDK takes raw text and has no request-scoped URL mask/restore map. | A correct learned placeholder cannot become the original clickable URL without a binding; raw URL inputs differ from supervision. | Make training and serving use the same reversible reference policy; test generated links and missing-reference behavior. |
| P1 | Remote prepared-v10 variants and the tested checkpoints lack a locally verified common training receipt. | Current raw-v10 or prompt checks cannot prove what the newer checkpoint actually learned. | Bind source hashes, prepared data, tokenizer/template, prompt, training config, checkpoint and export in one manifest. |
| P2 | Identical sources have multiple different canonical targets. | Alternative layouts can be legitimate. Variants need fidelity review for possible omissions/actions, and repeated sources receive extra weight. | Compare fidelity within each source group; retain validated variants intentionally or choose a consistently generated target. |
| P2 | Source arithmetic/consistency errors remain. | Faithful targets can still display bad information; this is distinct from conversion failure. | Correct and verify the source, then regenerate its pair. Do not train the converter to silently change supplied facts. |
| P2 | One valid target meets the app's sequential-ID repetition condition. | The current guard can stop a faithful generation before its children are defined. | Refine serving stop detection and regression-test the complete valid target and genuine loops. |
| P2 | All rows lack original query/intent/generator lineage; Bixby50 has prior prompt-development exposure. | Domain balancing, source truth checks and unbiased generalization claims are limited. | Preserve provenance for new data; add a genuinely unseen final cohort and report results by source family/domain. |

## 1. Integrity and structural validity

| Check | Fresh result |
|---|---:|
| Parsed corpus records | 92,977/92,977 |
| Strict Express/canonical graph + complete reachability + wire-schema + semantic serialization round-trip | 92,977/92,977 passed |
| Structural failures | 0 |
| Message/source/assistant consistency, empty content, envelope, replacement-character or special-token flags | 0 flags |
| Distinct archived system/few-shot scaffolds | 1 |
| Effective source/target hash metadata mismatches | 0 |
| Train/validation assigned-split mismatches | 0 |

The full-corpus structural pass uses the production parser, reference inventory, serializer and unchanged wire-schema assertions. Its compiled schema path is checked against the production validator on Golden originals and deliberately invalid copies; the detailed parity receipt is [schema_parity.json](schema_parity.json). This is structural/compiler validation, **not 92,977 Android screenshot tests**.

All archived messages have matching final input/output bindings. There is no evidence here of prompt tokens accidentally used as labels: the current HF path masks the prompt with `-100`, supervises the completion, pads labels with `-100`, and rejects overflow instead of truncating. See [pipeline review](pipeline_review.md) for code anchors and focused tests.

## 2. Production distribution mismatch

The same lexical tests were applied to training+validation and the frozen 50 Bixby responses. These count recognizable source formatting, not semantic domains. The URL-placeholder regex in this table recognizes URL namespaces; the broader reference census also covers placeholders of other kinds. HTTP substring matching finds 42 rows, whereas the reference census requires a URL word boundary and counts 17. Neither check certifies a working destination; table matching here requires a leading pipe and is not a count of every tabular source.

| Source feature | v10 (92,977 rows) | Bixby50 |
|---|---:|---:|
| Markdown headings | 92,962 (99.98%) | 47/50 |
| Leading-pipe table text | 55,099 (59.26%) | 23/50 |
| Media scaffolding | 92,624 (99.62%) | 0/50 |
| Action/Quick Actions scaffolding | 88,934 (95.65%) | 0/50 |
| URL placeholders | 92,911 (99.93%) | 0/50 |
| Numeric citation markers | 7 (0.01%) | 48/50 |
| HTTP(S) substrings | 42 (0.05%) | 0/50 |

This is strong evidence of a **distribution gap**, not proof of a particular checkpoint's causal failure. v10 largely teaches conversion of sources already organized as titled sections, media descriptors and action blocks. The measured Bixby50 cohort contains numeric citations in 48/50 responses and media/action scaffolding or literal URLs in 0/50. Regeneration should preserve those original response forms rather than append synthetic icon/action sections simply to resemble v10.

The reference/provenance audit separately found **92,911 rows with reference placeholders**, **48,108 rows with nonempty current URL maps**, and **44,803 placeholder-containing rows with empty current maps**. That leaves **305,649 distinct source-placeholder occurrences counted per row without bindings**. Current maps bind URLs encountered during normalization; they do not reconstruct missing historical asset lineage. All records declare missing original query, generator and asset provenance. See [reference census](reference_counts.json) and [leakage/provenance review](leakage_review.md).

## 3. Prompt and sequence-length budget

Token counts below use the locally extracted SentencePiece vocabulary, SHA-256 `e594c8a90eb08d8bda498ff4747977dc827ae0c3c56b5c0d41a605a22d02ef03`, plus explicit BOS and the deployed non-thinking Gemma text template. They are exact for that local counting procedure and a **proxy for the missing remote HF training-tokenizer receipt**. They are not claims about the final accepted row count of an unobserved GPU run. The corpus-wide sequence measurements use stored target text. An independent probe compared 503 deterministic rows spanning both splits against canonical preparation and separately encoded prompt/completion: all token deltas were zero, with no threshold disagreement. This sample check does not substitute for the actual remote tokenizer. See [token probe](pipeline_token_probe.json).

| Quantity | Median | P95 | P99 | Maximum |
|---|---:|---:|---:|---:|
| Source only | 489 | 694 | 810 | 1,720 |
| Target only | 922 | 1,255 | 1,408 | 2,467 |
| Archived generation prompt | 1,997 | 2,202 | 2,318 | 3,228 |
| Deployed generation prompt | 3,489 | 3,694 | 3,810 | 4,720 |
| Archived full sequence | 2,923 | 3,448 | 3,709 | 5,276 |
| Deployed-prompt full sequence | 4,415 | 4,940 | 5,201 | 6,768 |
| Components | 35 | 53 | 60 | 111 |
| Graph depth | 5 | 7 | 7 | 11 |

| Full-sequence limit | Over limit with archived prompt | Over limit with deployed prompt |
|---|---:|---:|
| 4,096 | 143 (0.15%) | 73,316 (78.85%) |
| 6,144 | 0 (0.00%) | 21 (0.02%) |
| 8,192 | 0 (0.00%) | 0 (0.00%) |

The current Golden launcher checks the **normalized** prompt and full supervised sequence against 4,096 tokens. It quarantines the whole row. Consequently, the raw bundle's 91,115 training rows must not be presented as the effective training set. A larger nominal model context does not override the launcher's `max_seq_length`.

At the 4,096 full-sequence threshold, the local proxy leaves **19,276 training rows and 385 validation rows** fitting that budget, before other preparation gates. These are hypothetical retention counts, not a verified GPU training receipt.

The deployed generation prompt alone exceeds 4,096 tokens in **56 rows**. Sequence and prompt thresholds are separate settings and both must be included in the exact-tokenizer preparation report.

For a concrete example, train:1 is structurally valid and measures **3,265 tokens with the archived prompt versus 4,757 with the deployed prompt** under this local tokenizer. The same source/target pair moves from below to above the 4,096 threshold simply because the scaffold changes.

Only **23 (0.02%)** stored targets exceed the SDK's 2,048 output-token budget under this tokenizer. The long repetitive outputs observed on-device therefore cannot be explained simply by claiming that most gold targets require more than 2,048 tokens. Source fidelity, output stopping, model fitting and export/runtime parity remain separate checks.

The archived prompt is consistent across the dataset; the current SDK asset matches the current shared builder. The risk is using or comparing different prepared versions without receipts, not random prompt mixing within this raw v10 corpus.

## 4. Confirmed semantic problems

The new fixed-seed complete-pair sample contains 18 train + 2 validation rows, excluding the previous 100 sampled rows: **16 KEEP, 3 REPAIR, 1 REJECT/REGENERATE**. KEEP means no material defect found in that review, not external factual certification. This small, split-stratified review must not be extrapolated to all 92,977 records. The previous 100-row review's 69/26/5 dispositions are historical evidence; their eight target defects are still present in the current corpus.

| v10 coordinate | Confirmed problem | Type |
|---|---|---|
| train:1702 | Detailed four-week activity schedule is lost; only the broader roadmap survives. | Target content loss |
| train:37244 | Passenger name, flight, seat and gate details are dropped. | Target field loss |
| train:14450 | Property restrictions/features are omitted. | Target qualifier loss |
| train:4964 | Detailed itinerary descriptions disappear. | Target content loss |
| train:60428 | Five source cells sit under four headers; target discards alert-checklist instructions. | Malformed source table plus target loss |
| train:69525 | Explicit chart title and axes produce tables/prose without a Chart component. | Visual-role loss |
| train:64773 | Values sum to 80,500, source says 73,500, percentages sum to 109.6%; target repeats errors and omits two specified charts. | Source contradiction plus target loss |
| train:73806 | Claims 450 remains after meeting all allocation targets even though targets total the entire 3,500 income. | Source contradiction |
| train:30894 | Transposed wearable table mislabels entities and hides `focus` values in state. | Entity/visible-field mismatch; previous finding |
| train:87790 | Hotel C receives unsupported booking/action metadata. | Latent invented action data; previous finding |

Full exact snippets, source identities, fresh verdicts, retained old findings and false-positive examples are in [semantic review](semantic_review.md), [fresh 20 pairs](semantic_fresh20.jsonl), and [fresh verdicts](semantic_fresh20_verdicts.json).

Full-corpus screening found these **review candidates**, not certified failure counts; categories overlap:

| Heuristic | Flagged rows |
|---|---:|
| table_cell_exact_gap | 3,856 |
| source_table_width_mismatch | 285 |
| chart_component_absent | 866 |
| explicit_chart_component_absent | 494 |
| number_exact_gap | 992 |
| option_field_exact_gap | 4,781 |
| short_bullet_word_gap | 813 |
| citation_label_exact_gap | 575 |

Exact-match tests can flag harmless reformatting (`10,000` vs `10000`), equivalent paraphrases, or conceptual chart discussion. Do not automatically discard all flagged rows. Conversely, source tokens being present somewhere in state do not prove that the right entity or visible table column shows them.

## 5. Duplicates, splits and benchmark exposure

- **90,485 unique exact sources** across 92,977 rows.
- **2,478 repeated-source groups**, involving **4,970 rows** and **2,492 extra rows** beyond one per source. Each group has different raw targets and multiple stored `effective_semantic_sha256` hashes. A different graph hash alone does not prove a factual contradiction; layout choice changes it too.
- **0 exact duplicate targets.** Split-family identity is preserved; exact, whitespace-normalized, reference-insensitive and original-source signatures found **0 cross-split groups**.
- Bounded lexical search scored **134,009 candidate train/validation pairs** and found **0** meeting its trigram-Jaccard threshold of 0.60. This is not an exhaustive semantic-paraphrase guarantee.
- No direct source matches to Golden35, the 31 unique accepted Golden32 sources, or Bixby50; rejected Golden source identities were also checked where available. Golden32's repeated case is not an extra independent test source.
- Bixby50 was previously used for prompt development, as its experiment README acknowledges. Absence of SFT row leakage does not make it an untouched generalization benchmark.

See [leakage/provenance report](leakage_review.md), [machine counts](leakage_counts.json), and [repeated-source examples](duplicate_source_examples.csv).

## 6. Representation and serving compatibility

The full-corpus component inventory is below. Rows may contain several component types, so row percentages must not be summed.

| Component | Instances | Rows containing it |
|---|---:|---:|
| Text | 1,132,014 | 92,977 (100.00%) |
| Stack | 1,001,587 | 92,977 (100.00%) |
| Card | 401,467 | 92,908 (99.93%) |
| Button | 349,077 | 89,020 (95.74%) |
| Icon | 273,225 | 92,527 (99.52%) |
| Table | 120,557 | 91,624 (98.54%) |
| Formula | 16,929 | 16,075 (17.29%) |
| Image | 2,396 | 1,744 (1.88%) |
| Tabs | 1,515 | 1,465 (1.58%) |
| Divider | 1,241 | 1,175 (1.26%) |
| ConsoleLog | 632 | 218 (0.23%) |
| CodeBlock | 544 | 342 (0.37%) |
| CheckBox | 385 | 89 (0.10%) |
| EmailPreview | 242 | 150 (0.16%) |
| Slider | 58 | 42 (0.05%) |
| TextField | 55 | 24 (0.03%) |
| ChoicePicker | 43 | 28 (0.03%) |
| AudioPlayer | 1 | 1 (0.00%) |

There are **zero supervised Chart targets** in the entire corpus, although the active contract supports Chart and the source scan finds 494 explicit chart-specification candidates without one. This is a concrete missing capability in the labels, not merely class imbalance.

Table-domain metadata is a layout hint and does not recover missing original intent labels. Rows can contain multiple domains. The final column shows how prompt expansion could bias retention under a 4,096 limit (proxy tokenizer and stored targets).

`formula`, `playlist` and `restaurants` are explicitly supported renderer domains; their presence is not an unsupported-enum defect. Compiler size limits are checked separately from domain routing, and current Python validation does not substitute for on-device visual inspection.

| Table domain | Tables | Rows containing domain | Rows over 4,096 with deployed prompt |
|---|---:|---:|---:|
| comparison | 35,299 | 34,744 | 31,053 (89.38%) |
| generic | 22,385 | 20,116 | 16,858 (83.80%) |
| formula | 21,905 | 15,145 | 11,002 (72.64%) |
| schedule | 19,935 | 18,680 | 15,960 (85.44%) |
| status | 14,127 | 13,687 | 9,364 (68.42%) |
| weather | 2,189 | 2,181 | 1,299 (59.56%) |
| flight | 1,585 | 1,580 | 944 (59.75%) |
| playlist | 1,300 | 1,241 | 609 (49.07%) |
| booking | 1,021 | 1,021 | 761 (74.53%) |
| restaurants | 694 | 680 | 305 (44.85%) |
| <unset> | 69 | 36 | 19 (52.78%) |
| data | 6 | 6 | 5 (83.33%) |
| summary | 6 | 6 | 1 (16.67%) |
| analysis | 5 | 5 | 3 (60.00%) |
| profile | 3 | 1 | 1 (100.00%) |
| activity | 2 | 2 | 0 (0.00%) |
| news | 2 | 2 | 0 (0.00%) |
| distribution | 2 | 2 | 0 (0.00%) |
| trends | 2 | 2 | 2 (100.00%) |
| visualization | 2 | 2 | 2 (100.00%) |
| energy | 1 | 1 | 1 (100.00%) |
| gauge | 1 | 1 | 1 (100.00%) |
| value | 1 | 1 | 1 (100.00%) |
| emissions | 1 | 1 | 0 (0.00%) |
| price | 1 | 1 | 1 (100.00%) |
| correlation | 1 | 1 | 1 (100.00%) |
| trend | 1 | 1 | 1 (100.00%) |
| fare_trend | 1 | 1 | 1 (100.00%) |
| finance | 1 | 1 | 1 (100.00%) |
| sales | 1 | 1 | 1 (100.00%) |
| volatility | 1 | 1 | 1 (100.00%) |
| metrics | 1 | 1 | 1 (100.00%) |
| efficiency | 1 | 1 | 1 (100.00%) |
| utilization | 1 | 1 | 0 (0.00%) |
| funnel | 1 | 1 | 1 (100.00%) |
| spending | 1 | 1 | 0 (0.00%) |
| metric | 1 | 1 | 1 (100.00%) |
| fare | 1 | 1 | 0 (0.00%) |

**Repetition guard conflict:** train:6416 (`archive-train-109871-f518eb549644b798`) has a valid screenplay target with `l=Column([m,n,o,...,af,ag,ah],"sm")`. These are 22 distinct sequential children with actual Text definitions, not a model loop. A streaming callback exposing that list before the closing sentinel meets `GenUiOutputRepetitionGuard`'s 20-reference condition. If the entire completed document arrives in one callback, the closing-tag bypass prevents the stop. Full-text testing alone would therefore miss this risk. One such complete-list candidate was found across the corpus; this check does not claim all possible partial-token/prefix false stops are enumerated or that a device failure was observed. See [serving census](serving_distribution.json) and [pipeline review](pipeline_review.md).

Long repeated literals mostly include valid shared dates/tasks; their count is not a measured decoder-loop rate. Likewise, raw long digit strings can be IDs or phone numbers. The recent model's repeated zeros, changed train numbers and altered percentages cannot be attributed to a specific training row without additional evidence.

## 7. Relation to the recent Fold7 results

The [previous five-case comparison](../../../GenUICraft/validation/20260925_fold7_more_epochs_comparison/REPORT.md) records 0/5 raw strict passes and 0/5 mechanical source-integrity passes for both checkpoints, although generated-output repair rendered 5/5 each. Those are prior device measurements, not new measurements from this audit. This corpus contains valid complete targets, so malformed generated DSL is not explained by widespread malformed gold labels. The observed omissions and citation losses are consistent with the data's verified weaknesses, but the repeated IDs, corrupted numbers and syntax failures also require checking exact training-prompt lineage, dense-checkpoint behavior and export/runtime parity. No supplied training manifest independently verifies the newer model's epoch count or exact prepared dataset.

## 8. Recommended sequence before another long run

1. **Freeze the experiment contract:** identify the exact input bundle, prepared splits, tokenizer/chat template, prompt hash, source-code revision and export pipeline used for each compared checkpoint. Use the current manifest-required launcher.
2. **Fix target fidelity first:** add regressions for the confirmed lost fields/charts/entity bindings and malformed source tables. Regenerate through the corrected Stage 3 pipeline; keep uncertain pairs out of the candidate set. Source factual repairs must update the source first.
3. **Close the serving gap:** adopt the same reversible URL/citation normalization in training and SDK, and cover absent URLs, plain text, markdown and mixed Unicode/numeric formats. Do not invent URLs for numeric citation markers.
4. **Measure the real prepared data:** with the actual HF tokenizer, report accepted/quarantined counts by source family, domain, input length and target length after prompt normalization. Select a budget compatible with deployment; preserve complete closing tags and graphs.
5. **Balance and deduplicate intentionally:** audit same-source variants, weight source families fairly, reduce forced icon/action scaffolding, and add representative production-style development cases independent of Bixby50.
6. **Run a short controlled model evaluation:** compare raw strict validity, repaired validity, numeric/citation/entity fidelity, visual-role coverage and rendered output. Separately evaluate dense checkpoint, quantized export, and on-device generation with MTP off/on to locate regressions. Rendering success through repair/fallback is not model accuracy.
7. **Scale epochs only after those gates improve.** Keep an unseen final evaluation set, and compare checkpoints on the same prompts, tokenizer, decoding settings and device.

## Evidence and reproducibility

- [Structural/token summary](structure_summary.json); [compact per-row CSV](row_metrics.csv); [length/domain slices](length_slices.json).
- [Compressed complete per-row evidence](structure_rows.jsonl.gz); [final integrity and code fingerprints](audit_manifest.json).
- [Pipeline source review](pipeline_review.md); [semantic evidence](semantic_review.md); [split/provenance evidence](leakage_review.md).
- Reproduce structure: `python training/reports/v10_full_audit_20260926/audit_structure.py --workers 12`. Requires the local extracted tokenizer path named in that script plus repository Python dependencies. It writes report artifacts only; interrupted runs can use `--resume`.
- Reproduce serving census: `python training/reports/v10_full_audit_20260926/audit_serving.py`.
- Other audit scripts and their deterministic seeds/methods live beside this report. The existing archived manual100 is retained unchanged.
- Dataset train SHA-256: `707244a5ffe2af459605e8629055c18829c65123294191112016cd69892addf5`.
- Dataset validation SHA-256: `b5024919ea0c3557cad452193f90f7133919bb0c10e4ece05fe1a54278cb0fa7`.

The audit does not establish that v10 is the sole cause of the observed E2B failures. Exact preparation lineage and dense-versus-export inference remain necessary to separate dataset quality from optimization, quantization and serving behavior.
