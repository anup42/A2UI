# Offline messages-archive recovery v5

Historical intermediate: use the [final v9 report](../offline_recovery_20260913_v9/REPORT.md)
and [current runbook](../../docs/messages_archive_final_review.md) for training.
The v5 measurements below are preserved as baseline evidence, not current totals.

## Verdict

Use the **policy-v5 recovered candidate**, not the original messages-only
archive, for the next tokenizer preflight and smoke run. The offline pass
reconciled all 151,202 source rows and retained **118,531 (78.39%)**:

| Decision | Train | Validation | Combined |
|---|---:|---:|---:|
| KEEP, target content unchanged | 14,583 | 77 | **14,660** |
| REPAIR, reversible or uniquely source-grounded | 103,258 | 613 | **103,871** |
| QUARANTINE | 32,451 | 220 | **32,671** |
| Total | 150,292 | 910 | **151,202** |

The emitted files contain **117,841 train rows** and **690 validation rows**.
This is 75,472 more usable rows than the initial conservative v1 screen
(43,059), while retaining hard gates for missing content, Golden leakage,
split leakage, duplicates, and ambiguous references.

No training, model inference, Stage 3 generation, target truncation, or
Golden evaluation ran during this repair. No quality score is claimed.

## Confidence tiers inside the accepted set

The full candidate records every transformation in `repair.changes` and
`metadata.archive_recovery.transformations`:

| Tier | Rows | Meaning |
|---|---:|---|
| Unchanged | 14,660 | No recovery transformation was needed |
| Lossless representation repair | 64,976 | Exact text-codec/reference normalization only |
| Source-grounded structural repair | 38,895 | A unique source label/media token was rebound, or source-absent optional structure was removed |
| Full accepted candidate | **118,531** | Sum of the three tiers |

For the first practical run, use the complete v5 candidate after the exact
model-tokenizer preflight. If compute permits, a useful controlled ablation is
the 79,636-row unchanged-plus-lossless subset versus the complete candidate,
using equal token/update budgets and Golden32 for selection. Do not repeatedly
tune this choice on Golden35.

The train split contains **114,746 source families**. Of these, 3,080 retain
more than one distinct semantic layout target: 3,065 families have two targets
and 15 have three, for 6,175 rows total. These are not duplicate targets, but
they can give the model multiple valid answers for the same input. Do not
delete them blindly; consider source-balanced sampling or a controlled
one-layout-per-family ablation if training becomes unstable.

## What was repaired

Counts below overlap by row and must not be added together. Operation counts
can exceed row counts when one example contains several corrected references.

| Accepted transformation | Rows | Operations where applicable |
|---|---:|---:|
| Source CP437-bytes-as-UTF8 recovery | 32,457 | — |
| Target CP437-bytes-as-UTF8 recovery | 50,312 | — |
| Source-identity URL/local-reference normalization | 57,039 | — |
| Exact-label placeholder rebinding | 27,234 | 62,057 |
| Same-index media URL/asset namespace rebinding | 16,335 | 58,160 |
| Unique one-to-one same-kind media-token rebinding | 574 | 592 |
| Source-absent reference elements removed | 1,089 | 1,235 |
| Source-absent reference fields removed | 402 | 710 |
| Empty layout elements removed | 896 | 4,316 |
| Ungrounded `openUrl` events removed | 193 | 258 |
| Empty table columns removed | 188 | 269 |
| Exact legacy mid-word chunk joins | 29 | 35 |
| Generic unrequested `Open Source` buttons removed | 1 | 1 |

The important admission rules are:

- Text transcoding is accepted only when the whole string round-trips exactly
  and known corruption markers decrease.
- Literal URLs and local references are masked source-first. Restoring the map
  must reproduce the exact source text and target graph.
- Link roles are never guessed by index.
- Label rebinding requires one exact normalized source label and one compatible
  token.
- The extra media-index repair requires exactly one target-only and exactly one
  source-only token of the same concrete media kind. Ambiguous choices are not
  changed.
- An unsupported reference field/component can be removed only when its token
  is absent from the source. Renderer-specific tab/modal/template references
  are protected.
- Adjacent Text values are joined only at an alphanumeric mid-word boundary
  when the left value is exactly 180 characters. No character is invented.
- Every accepted graph is serialized through the production Express contract;
  the independent verifier also runs a deterministic production-preparation
  sample.

## Why 32,671 rows remain quarantined

These first reasons are mutually exclusive and sum to the full quarantine:

| First reason | Rows | Action |
|---|---:|---|
| Unresolved content warning | **32,544** | Keep excluded until authoritative regeneration/review |
| Unresolved target reference | **59** | Recover original map or regenerate; do not guess link/media destination |
| Validation family present in original training | **29** | Keep out of validation |
| Reserved Golden family | **28** | Keep evaluation-only |
| Confirmed content defect | **8** | Keep excluded; one ninth known coordinate is counted in split overlap first |
| Duplicate source-family/semantic target | **3** | Keep one occurrence only |
| Total | **32,671** | — |

The content-warning signals overlap:

| Warning | Rows |
|---|---:|
| Requested named action missing | 27,966 |
| Declared media missing | 13,465 |
| At least half of numeric anchors missing, with at least three in source | 3,498 |
| Distinct lexical recall under 50% | 958 |
| Layout-only output | 3 |

These rows are not all proved irreparable. They are cases where offline code
cannot restore omitted facts, actions, or media without inventing supervision.
Stage 3 is unavailable now, so quarantine is the correct result. Copying a
sibling answer over them would add no new source coverage and could erase a
legitimate alternative layout.

## Verification evidence

The final independent pass verified:

- all 151,202 inputs have exactly one decision;
- artifact hashes match the completed manifest;
- 117,841 train and 690 validation rows are readable JSON and correctly bound
  to their final user/assistant messages;
- train/validation source-family overlap is zero;
- source-family/semantic-target duplicates are zero;
- exact effective source/target duplicates are zero;
- accepted targets with a placeholder absent from their source are zero; and
- 256 deterministic rows pass the production preparation path.

The exact machine-readable evidence is in [summary.json](summary.json) and
[verification.json](verification.json). The original inputs remained read-only:

| File | Rows | Bytes | SHA-256 |
|---|---:|---:|---|
| `train.jsonl` | 150,292 | 1,970,021,858 | `5c47fb3f07dc6f2711ece12a1a93b4ecd6977fa8060119f329db3189c2d6b910` |
| `val.jsonl` | 910 | 11,850,436 | `04e88d07003803279cea7de093bd9c69bcd35ed9a5fab057d577544b8c90ba19` |

The large recovered files are local ignored artifacts at
`training/outputs/datasets/full_data_archive_recovered_v5/`; `train.jsonl` is
about 2.85 GB and `val.jsonl` about 16.5 MB. They are intentionally not placed
in Git. Copy that directory to the GPU host, or rebuild it from the exact
source hashes using the [recovery runbook](../../docs/messages_archive_recovery.md).

## Recommended next sequence

1. Transfer or rebuild the verified v5 dataset on the GPU host.
2. Run `run_golden_training.py --prepare-only` with the exact local E2B or
   270M model bundle. This is the remaining tokenizer/chat-template/length
   gate; do not truncate failures.
3. Continue the same prepared run for a 20-update smoke. Require finite loss,
   gradients, parameter changes, and complete Golden32/Golden35 final reports.
4. Start a fresh full run only after the smoke passes. Keep all visible H100s;
   the launcher selects its 2/4/8-GPU profile automatically.
5. Use Golden32 unique-source score for checkpoint selection. Keep Golden35
   final-only. Report selected-best and actual-final scores separately on both.
6. Keep the default generation cap at 2,048 new tokens and TensorBoard at
   `/tensorboard/<run-id>/`.
7. When Stage 3 becomes available, regenerate the 32,544 content-warning cases
   from authoritative sources in a separately versioned dataset. Do not edit
   their IR by hand.
8. After measuring retained post-tokenizer coverage, add independent data for
   image-bearing layouts, complete long documents/multi-entity schedules,
   grounded actions/media, and rare supported component combinations. Do not
   paraphrase or copy Golden32/Golden35 cases.

A lightweight scan of the retained canonical Express completions found Table
in 116,179/117,841 train rows (98.59%), Image in 2,255 (1.91%), EmailPreview in
184, and only one Video. This remains a highly structured/table-heavy corpus.
It covers every component used by the current Goldens, but it does not match
Golden35's image-bearing frequency (12/35). Do not oversample questionable old
image labels to imitate that small benchmark; create independently sourced,
fully mapped image/action examples when Stage 3 is available.

The dense Golden32/Golden35 launcher does not train MTP or produce LiteRT-LM
packages. MTP enable/disable and official retained-scale QAT remain in the
separate E2B multiformat pipeline; do not treat dense LoRA as official-format
continued QAT.
