# v10 dataset: complete source-grounded repair and filtering pass

Date: 2026-09-16. Status: export complete; independent offline verification passed.

The updated copy is `training/outputs/datasets/full_data_archive_recovered_v10`.
It contains **92,977 candidate pairs: 91,115 training and 1,862 validation**.
Every one of the 115,142 v9 pairs was processed. The original v9 files and the
frozen Golden files were rehashed before and after export and are unchanged.
No Stage 3 generation, model training or deletion was performed.

## Counts for this pass

These categories are mutually exclusive and relative to **v9**, not the raw
archive. KEEP means source/target content unchanged by this pass; provenance
metadata was added to all retained rows.

| Outcome | Training | Validation | Total |
| --- | ---: | ---: | ---: |
| v9 input | 112,842 | 2,300 | 115,142 |
| KEEP: retained unchanged | 80,906 | 1,659 | 82,565 |
| REPAIR: corrected and retained | 10,209 | 203 | 10,412 |
| QUARANTINE: excluded from v10 | 21,727 | 438 | 22,165 |
| **Final retained** | **91,115** | **1,862** | **92,977** |

The original archive contained 151,202 records. Its cumulative decision ledger
now has 58,225 quarantined and 92,977 retained records; 36,060 exclusions predate
this pass. Likewise, cumulative KEEP/REPAIR counts in `manifest.json.categories`
include earlier repair versions. Use `manifest.json.semantic_review` for the
new v9-to-v10 counts above.

## Repairs applied across the complete dataset

| Deterministic repair | Retained rows affected |
| --- | ---: |
| Restore omitted wording from one unambiguous source clause | 3,795 |
| Restore shortened table labels from a uniquely matched source table | 2,980 |
| Escape literal state keys so table JSON pointers resolve | 2,052 |
| Remove unsupported, empty image/imageAlt columns | 1,899 |
| Repair partial encoding using an exact inverse and source evidence | 136 |
| Correct an existing citation action to its uniquely matched source reference | 24 |
| Undo a proven false asset/liability URL mask using its retained map | 4 |

Rows can receive multiple repairs, so this table must not be summed to obtain
the unique repaired count. No missing URL, factual number, control or layout
node was invented. Existing source-family identities and split assignments
were preserved. One duplicate created by repairs was excluded deterministically.

Representative changes, identified by original archive coordinate:

- `train:15259`: restored the source's explicit carpentry and plumbing/electric
  hookups to an existing cabinet-planning Text clause.
- `train:122470`: `/variables` became `/~1variables` because the actual root
  state key is literally `/variables`; the underlying data was not moved.
- `train:94959`: repaired the corrupted summation label to `Σ Hours`, with
  exact reversible encoding and matching source/math evidence.
- `train:146903`: removed empty media columns from a study-week table with
  no supplied media; the actual study data remains intact.
- `train:35551`: corrected an existing citation URL binding to the source
  reference associated with its label; no new action was added.

Each row's before/after evidence is in `semantic_review.jsonl`. Retained rows
also carry `metadata.archive_semantic_review` with repair proofs and hashes.

## Why rows were quarantined

The principal flags are below. Counts overlap; the unique total is **22,165**.

| Flag | Rows |
| --- | ---: |
| Source prose has potentially missing substantive wording | 18,449 |
| Source citation not present in visible/bound target content | 3,748 |
| Source media not bound in the target | 1,336 |
| Important source qualifier lost | 219 |
| Explicitly requested chart missing | 215 |
| Suspicious encoding not safely recoverable | 89 |
| Inline action/reference absent from the target | 29 |
| Hash-bound reviewed source requires verification | 16 |
| Specific manually reviewed source detail still missing | 11 |

The remaining flags cover isolated hidden date fields, unresolved state paths,
description text used as input values, the duplicate, and narrow source-risk
patterns. All reason codes and counts are in [summary.json](summary.json).

**Quarantine is not a claim that every row is factually wrong.** The prose
coverage gate deliberately errs toward holding a pair when source meaning may
be lost. Legitimate paraphrases or redundant introductions can also be held.
These rows remain recoverable in v9. Do not relax the gate globally or silently
copy held rows back into training: review the flagged content first.

Known arithmetic, calendar, factual and safety contradictions require source
review. Missing whole components/actions generally require a generation-pipeline
fix and Stage 3 regeneration. Such work was not replaced with manual IR edits.

## Replay of the manually audited 100 cases

This is a regression replay of the prior random sample (seed 2026091602,
90 train + 10 validation), **not a fresh manual audit of v10**.

| Prior manual disposition | v10 KEEP | v10 REPAIR | v10 QUARANTINE |
| --- | ---: | ---: | ---: |
| KEEP (62) | 57 | 2 | 3 |
| REPAIR candidate (30) | 0 | 7 | 23 |
| REJECT (8) | 0 | 0 | 8 |
| **Total** | **57** | **9** | **34** |

All 38 known defective pairs were repaired or held; none slipped through
unchanged. Repaired cases are 19, 21, 46, 55, 56, 70, 84, 93 and 97. Cases 56
and 97 received small additional improvements despite their earlier KEEP
verdicts. Cases 65, 80 and 100 were earlier KEEP cases now held conservatively
for wording gaps; this demonstrates the false-positive tradeoff, not newly
proven factual errors. The finite source holdlist does not generalize into a
universal arithmetic or factual checker.

See [the per-case replay](manual100_policy_replay.json) and the checked-in
[hash-bound findings](../../data/quality/v9_manual100_findings.json). The original
sample's 92 potentially salvageable pairs were a repair opportunity, not 92
already-valid pairs. The stricter offline-only policy retained 66 here.

## Verification actually performed

- All **92,977** retained rows passed compiled strict schema/canonical checks,
  existing content/reference checks, and second-pass repair idempotence.
- Independent verifier rechecked hashes, identities, placeholder closure and
  duplicate checks across all output rows.
- A deterministic **512-row** sample passed the original production validator
  and the new review policy, with no remaining same-policy repairs/issues.
- **90,485 unique normalized sources** were checked for separation: zero
  exact/normalized Golden matches, zero detected lexical-near Golden matches,
  zero cross-split source-family/normalized-source matches, and zero detected
  lexical-near train/validation matches. Lexical search is bounded, not an
  exhaustive semantic-paraphrase guarantee.
- All 14 retained letter rows and all three historical join-repair rows passed
  their independent checks.
- Full training test suite: **1,311 passed, 3 skipped**. After the final edge-case
  tests/compatibility adjustments, focused archive/scaffold regression suite:
  **188 passed**. The two counts overlap; they are not additive.

Evidence: [independent verification](verification.json),
[export summary and hashes](summary.json), [test record](tests.json).
The manifest stores implementation fingerprints from the actual export; later
generic-URL/punctuation edge-case compatibility fixes do not rewrite that
historical provenance. The independent verifier ran against the final policy.

## Use and remaining work

Copy the **entire** v10 folder to the GPU PC and change the existing training
command's `--input-dir` to that folder. Use a fresh output run. Do not resume an
old v9 optimizer run with a changed dataset. Re-run host verification and exact
model-tokenizer preparation. Detailed commands are in the
[v10 runbook](../../docs/messages_archive_semantic_review.md).

This is an improved, offline-validated **candidate** dataset, not a guarantee
of perfect semantics or higher Golden scores. Original questions and complete
generation/asset lineage are still absent. Remaining source facts, every
arithmetic relationship, on-device appearance, exact-model token limits and
model quality have not been universally certified. No GPU training or Golden
inference was run. A fresh manual audit plus a short GPU smoke run is the next
validation step before a long training run.

The generated dataset is intentionally ignored by Git. This report, policy,
tests and reproduction instructions are repository files; a clone by itself
does not transfer the multi-GB dataset. No commit or push was requested for
this pass.
