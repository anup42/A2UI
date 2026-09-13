# Clean archive copy with rebuilt validation

**v6 is an intermediate, not the final training copy.** Continue with the
[v7-v9 final review](messages_archive_final_review.md), which catches omitted
paragraphs and letter clauses and corrects button relabeling and join boundaries.

The v6 workflow starts from the verified v5 recovery, reviews every retained
target and every original quarantined row, applies additional provable repairs,
excludes Golden source families, and builds fresh source-grouped splits. It
reads the original archive and v5 files without modifying them. All outputs
must use fresh directories.

The exact completed counts and remaining issues belong in the
[v6 report](../reports/offline_recovery_20260913_v6/REPORT.md).

## Reproduce the new copy

If v5 and its audit index are not available, first follow the
[v5 reconstruction runbook](messages_archive_recovery.md). Then run from the
repository root. The completed pass uses Python 3.12.10; a small CPU dependency
file records the schema/parser dependencies without installing GPU training:

```powershell
python -m pip install -r training/requirements-archive-recovery.txt

python training/scripts/refine_recovered_archive.py `
  --base-dir training/outputs/datasets/full_data_archive_recovered_v5 `
  --source-dir C:/Users/anupk/Downloads/training_data `
  --inventory training/outputs/audits/full_data_20260913/inventory.sqlite `
  --output-dir training/outputs/datasets/full_data_archive_recovered_v6 `
  --audit-dir training/outputs/audits/archive_refinement_v6 `
  --report-dir training/outputs/audits/archive_refinement_report_v6 `
  --workers 16 --val-fraction 0.02 --seed 20260913 --execute

python training/scripts/verify_refined_archive.py `
  --dataset-dir training/outputs/datasets/full_data_archive_recovered_v6 `
  --strict-sample-size 512 `
  --report training/outputs/audits/archive_refinement_report_v6/verification.json
```

On another checkout, choose a fresh report path under `training/outputs/`
because the small completed report is intended to be kept in Git. Omit
`--execute` to inspect the plan. Existing output directories are never
overwritten. `review.sqlite` is local audit evidence, not a training input.

The large UTF-8 training files remain ignored by Git. To train elsewhere,
copy the complete v6 dataset directory and verify it on that host, or reproduce
it from the hash-bound originals and v5 evidence. A clone alone does not include
this external 151,202-record archive.

## Repairs and admission rules

- Preserve the v5 codec, reference, schema, and content checks.
- Recognize conventional thousands separators when all previously missing
  numeric anchors have an equivalent target value. Dates, fractions, comma
  decimals, and percent markers retain their meaning; training values are not
  rewritten by this comparison.
- Unescape a quoted prose value only if the entire corrected value appears in
  the source. Code blocks, console logs, and formulas are protected.
- Complete a trailing clipped word in a 180-character Text value only if its
  full prefix has exactly one source match and the continuation is explicit.
- Restore an existing button label only from one unambiguous source action
  declaration with the same existing URL. Preserve already grounded labels.
- Reject requested links that exist only in hidden state, dead buttons, or the
  wrong URL binding; reject empty interactive tab/modal panels.
- Keep confirmed missing-content examples excluded. The previously confirmed
  El Nino encoding example can return only after the exact codec recovery and
  all current checks pass.

These rules cannot recreate omitted sections, missing table data, asset maps,
or unknown action destinations. They do not synthesize new UI components.

## Benchmark isolation and rebuilt validation

Golden32 and Golden35 and their excluded originals remain reserved. Matching
uses the previous source-family evidence, raw hashes, repaired Unicode/case/
whitespace/reference-normalized source signatures, and a bounded lexical
near-source scan. The fixed prompt/scaffold is excluded from source matching.

The initial validation target is 2% of source families, stratified by observed
component shape and source-length bins. Source families are indivisible.
Further close lexical neighbors are grouped transitively and move together
into validation; the final row fraction can exceed 2%. Membership and output
order are deterministic with seed `20260913`.

Near matching retrieves up to 24 rare word trigrams per reference, requires at
least two shared anchors and a source-length ratio of at least 0.65, then
requires full trigram Jaccard of at least 0.60. Golden matching additionally
checks at least 90% trigram containment for sources with at least 50 words and
length ratio at least 0.40. Iteration must converge with no detected train/val
near matches before export. This is not an exhaustive semantic paraphrase
guarantee; original generator IDs and source lineage are absent from the archive.

Rebuilt validation is held out for a **fresh training run**. A checkpoint already
trained on the original archive may have seen these records and the contaminated
Golden cases. Cleaning the files does not remove that exposure from its weights.

## Continue to the final copy

Use the v7 paths and commands in the [final runbook](messages_archive_final_review.md).
Do not start new training from the intermediate v6 copy.

The tokenizer/template gate can reject additional overlong examples; it must
never clip their sources or targets. Training/evaluation defaults remain:
all visible GPUs with automatic H100 profiles, validation/save every 500
optimizer updates, Golden32 every 1,000 plus final, Golden35 final-only,
2,048 generated tokens, and TensorBoard under `/tensorboard/<run-id>/`.

The dense Golden launcher remains separate from the official retained-scale
QAT/MTP/LiteRT multiformat pipelines. This data refinement does not run or
validate GPU training, model conversion, or checkpoint scores.
