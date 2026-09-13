# Final offline archive recovery and review: v9

Date: 2026-09-13. **Final export, independent verification, and launcher sample
all passed against this exact v9 copy.** The disk-space blocker is resolved.

## Result and immutable inputs

The new copy contains **115,142 offline-valid candidates** from the 151,202
original records. Use `training/outputs/datasets/full_data_archive_recovered_v9/`;
do not train from an intermediate or `.partial-*` directory.

| Final split | Samples | Source families |
|---|---:|---:|
| Training | 112,842 | 109,893 |
| Rebuilt validation | 2,300 | 2,243 |
| Accepted total | **115,142** | **112,136** |

| Final decision across the original archive | Samples |
|---|---:|
| KEEP: no recovery transformation required | 14,412 |
| REPAIR: accepted after audited recovery | 100,730 |
| QUARANTINE / reserved / duplicate | 36,060 |
| All original rows reconciled | **151,202** |

"Valid" means passed the documented offline schema/content policy. It is not
proof that every fact is rendered correctly, that every row fits the selected
model tokenizer, or that the model will achieve a particular Golden score.

Original files were opened read-only and their SHA-256 hashes checked before
and after export. Neither original was modified or deleted:

| Original input | Rows | SHA-256 |
|---|---:|---|
| `C:/Users/anupk/Downloads/training_data/train.jsonl` | 150,292 | `5c47fb3f07dc6f2711ece12a1a93b4ecd6977fa8060119f329db3189c2d6b910` |
| `C:/Users/anupk/Downloads/training_data/val.jsonl` | 910 | `04e88d07003803279cea7de093bd9c69bcd35ed9a5fab057d577544b8c90ba19` |

The v8 inputs are also hash-verified unchanged. v9 preserves all v8 sample and
split assignments and changes exactly one target, its assistant message, and
the associated provenance hashes. The final copy is approximately 2.70 GiB.
It is ignored by Git; copying this directory or reproducing it is required on
the training host. The frozen Golden32/Golden35 files and manifests are already
Git-tracked and were not changed by this work.

## Repairs retained after the deeper review

The accepted set has 14,412 unchanged rows, 63,771 lossless representation-repair
rows, and 36,959 source-grounded structural-repair rows. These three tiers are
mutually exclusive. Individual transformations below overlap and must not be
added together:

| Retained transformation | Rows |
|---|---:|
| Source CP437/UTF-8 corruption recovered by exact round-trip | 31,285 |
| Target CP437/UTF-8 corruption recovered by exact round-trip | 48,837 |
| Source-identity URL/local-reference normalization | 55,820 |
| Exact-label placeholder rebinding | 26,304 |
| Same-index media namespace rebinding | 15,675 |
| Unique same-kind media rebinding | 553 |
| Source-absent reference elements removed | 798 |
| Source-absent reference fields removed | 382 |
| Empty layout elements removed | 290 |
| Empty table columns removed | 175 |
| Ungrounded `openUrl` events removed | 187 |
| Missing existing-button labels repaired from unique source bindings | 53 |
| Historical 180-character boundary joins retained and rechecked | 3 |
| False join corrected with a source-proven separator | 1 |

The 53 button-label rows contain 65 repaired labels. The corrected rule leaves
secondary source buttons unchanged when the requested action already exists.
It does not create an action or guess a URL. Another 53 rows had numeric
grouping false positives resolved: `2,100` and `2100` can agree without changing
dates, decimal values, quantities, or units. Resolving a review false positive
is not the same as generating new content.

The final join repair is at original coordinate `train:65649`. The old helper
joined two neighboring Text values solely because the left value had 180
characters and both boundary characters were alphanumeric. This produced
`olive oilPrep: Roast...`, even though these were separate instructions.
v9 requires unique surrounding source context and restores the source's `. `
separator: `olive oil. Prep: Roast...`. The two genuine split-word examples,
`train:19598` and `train:38747`, remain unchanged. All three are tested for
production serialization and idempotence; no missing sentence was invented.

### Why the final count is lower than v5

The v5 baseline admitted 118,531 rows. The subsequent all-row review recovered
**130 previously quarantined rows**: 121 now REPAIR and nine now KEEP. It also
excluded **3,519 previously accepted rows** under stronger checks:

| Newly excluded relative to v5 | Rows |
|---|---:|
| Unresolved content/reference problems under the stronger review | 3,442 |
| Incomplete letters detected by the dedicated letter check | 70 |
| Newly excluded accepted Golden-related rows | 5 |
| Newly excluded accepted duplicates | 2 |
| Total | **3,519** |

Thus `118,531 + 130 - 3,519 = 115,142`. Five newly removed accepted Golden rows
is not the full benchmark exclusion count: other Golden-related records were
already excluded by earlier checks. The final reserved total is 36.

The paragraph review excluded 2,951 candidates from v6. A subsequent letter
review checked 84 remaining letter-style candidates, retained 14, and excluded
70 with missing or rewritten wording. In particular, completing a clipped
word did not rescue letters that still omitted requests or clauses. These
screens are deliberately conservative and can flag legitimate paraphrases;
they do not establish that every quarantined row is irreparable.

## Benchmark exclusion and rebuilt validation

Both ordinary splits exclude the frozen Golden cohorts and their reserved
original-source exclusions. The final exclusion file contains **36 archive
rows**: 35 normalized, reference-insensitive matches (33 to Golden35 and two
to Golden32), plus one additional previously reserved/excluded source.

Identity checks are supplemented by source normalization across reversible
encoding damage, Unicode/case/whitespace differences, and literal or symbolic
references. The lexical-neighbor gate uses bounded rare-trigram retrieval
followed by full overlap scoring. It is not an exhaustive semantic-paraphrase
detector. Golden membership itself stays unchanged: Golden32 has 32 occurrences
and 31 unique sources; Golden35 has 35 references.

Validation was rebuilt with seed **20260913**, initially targeting 2% of source
families and stratifying by observed component combinations and source length.
Exact/normalized/identified close neighbors remain in one split. It is not an
intent-balanced split: original intent labels and generator IDs are absent
from this archive. v8/v9 retain the rebuilt assignments while filtering or
repairing records; they do not mix the original validation file back in.

Final movement relative to original files is 112,165 train-to-train, 2,283
train-to-validation, 677 validation-to-train, and 17 validation-to-validation.
The old 910-row validation file is therefore not the final held-out set.
There are 112,138 distinct normalized source variants, grouped into 112,136
families after two near-source unions.

An existing checkpoint trained on the old pool may have seen these newly held
out rows and Golden sources. Cleaning files cannot undo exposure in weights.
Use fresh base weights for a clean experiment, Golden32 for checkpoint
selection, and Golden35 for final-only reporting.

## What remains excluded, and how much more can be repaired

These first reasons are mutually exclusive and reconcile all excluded rows:

| Final first reason | Rows | Treatment |
|---|---:|---|
| Unresolved content or reference | 35,951 | Keep out pending authoritative repair/review |
| Incomplete letter | 70 | Keep out until complete source-preserving target is available |
| Reserved Golden source | 36 | Evaluation-only; do not repair back into training |
| Duplicate source-family target | 3 | Keep one valid occurrence, not repeated replacements |
| Total | **36,060** | |

The warning counts overlap: 27,939 missing named actions; 26,947 requested
actions without matching rendered Button bindings; 13,514 missing declared
media; 3,442 numeric-anchor gaps; 2,951 severe prose-block gaps; 956 low lexical
recall cases; 138 empty interactive panels; 70 incomplete letters; 59 unresolved
target references; eight previously confirmed content defects; and three
layout-only outputs. A row can trigger several warnings.

The follow-up found and applied one additional proven separator repair after
v8. No further automatic repair is currently demonstrated by this review.
That does **not** mean the remaining 36,021 content/reference/letter cases are
all irreparable. Their recoverable count cannot be established without the
missing authoritative mappings, a fuller source-to-layout review, or Stage 3.
Do not fill absent actions, paragraphs, media, or tables with plausible guesses.

There are 244 unresolved-content/reference rows whose source family already
has a retained valid target, plus the three duplicate rows. Copying that good
target over bad siblings would not add new source-family coverage and would
inflate the sample count. The valid sibling is already available to training.

## Remaining risks in the accepted data and improvement priorities

1. **Representation imbalance.** Training contains Table in 111,462 rows
   (98.78%), Image in 2,159 (1.91%), EmailPreview in 166, CodeBlock in 488,
   Slider in 65, ChoicePicker in 46, TextField in 53, and AudioPlayer in one.
   No Video remains. Rebuilt validation mirrors the common patterns but cannot
   provide meaningful coverage for extremely rare components.
2. **Alternative targets.** 2,936 training source families and 56 validation
   families have multiple distinct targets. These are not exact duplicates.
   If needed, compare source-balanced sampling or one-target-per-family in a
   controlled experiment; do not blindly duplicate or remove layouts.
3. **Missing historical metadata.** Original generator IDs, intent labels,
   asset manifests, and mappings for already-masked references are unavailable.
   Source-family hashes are derived identities, not recovered original IDs.
4. **Semantic/visual limits.** Content screens use target properties and bound
   state, not Android screenshots. They do not prove that every bound field
   appears on-screen, that every fact is correct, or that all paraphrased
   omissions are detected. The letter gate only covers its documented opening
   patterns. A future independent render/content audit remains useful.
5. **Exact model preparation is still required.** Source length p99 is 3,339
   characters and target length p99 is 4,735 characters in training; these are
   not token counts. Use the actual E2B/270M tokenizer and chat template, reject
   overlong records whole, and record retained coverage. Do not truncate IR.

For the next training experiment, use the verified v9 copy after that exact
tokenizer gate. Keep the conservative exclusions. When Stage 3 is available,
prioritize independent, complete long documents, grounded actions/media,
image-bearing layouts, and rare supported components. Do not augment from or
paraphrase Golden32/Golden35 examples. No quality improvement or model score
has been measured by this CPU-only data repair.

## Verification and reproducibility

The export reuses already parsed/schema-validated candidates, and strictly
serializes the one changed target. Final verification checks every output
row's envelope, identity, message/hash bindings, placeholders, duplicate keys,
and source separation; it also runs the unaccelerated production preparation
path on 512 deterministic rows, all 14 retained letter-style rows through the
letter gate, and all three historical joined rows through source-bound review.

The launcher admission check passed **184/184** rows: a 128-row deterministic
sample plus all 56 targeted button-label/boundary-repair rows. The tokenizer
was deliberately not stubbed or simulated. The focused regression suite passed
**71 tests**; Ruff and Git whitespace checks passed.

Final verification reported zero exact/normalized Golden matches, zero detected
lexical-near Golden matches, zero train/validation family/normalized/lexical
overlaps, zero duplicate family/semantic-target or effective source/target
pairs, and zero target placeholders absent from their source. The 512-row
content sample had no residual same-policy repairs. All 14 retained letters
and all three historical joins passed their targeted gates.

Evidence:

- [Summary and immutable hashes](summary.json)
- [Final independent verification](verification.json)
- [Complete counts, transitions, coverage, and letter evidence](profile.json)
- [Launcher admission check and inspected joined examples](launcher_sample.json)
- [Rebuild, host transfer, exact-tokenizer preflight, and smoke commands](../../docs/messages_archive_final_review.md)

The final exporter now checks disk capacity before creating a new copy,
requiring the estimated copy size plus 1 GiB headroom. Originals and completed
inputs are never overwritten or automatically deleted. No Stage 3 call, GPU
training, checkpoint inference, device render, conversion, or Golden score was
produced. Dense training, official retained-scale QAT/MTP, and LiteRT-LM export
remain distinct workflows; this data pass does not claim to validate all three.
