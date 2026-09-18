# v10: renderer-aware and source-grounded review of v9

The v10 workflow scans **every** v9 training and validation pair, writes a new
dataset, and preserves v9 byte-for-byte. It does not call an LLM or Stage 3.
The completed run's exact counts and limitations are in
[the v10 report](../reports/offline_recovery_20260916_v10/REPORT.md).

## Use the updated dataset

The local completed dataset is:

```text
training/outputs/datasets/full_data_archive_recovered_v10/
  train.jsonl
  val.jsonl
  manifest.json
  decisions.csv
  quarantine.csv
  semantic_review.jsonl
  reviewed_findings.json
  ...retained benchmark/split lineage files
```

Copy the **entire folder** to the training PC, not just the two JSONL files.
Multi-GB generated datasets are ignored by Git. A clone alone does not contain
them. Rejected records remain in v9; the review log records their original
archive coordinate and their v9 split/line. Nothing is deleted.

Use a fresh output/run directory and point the existing training or full
deployment command at `--input-dir /data/full_data_archive_recovered_v10`.
Do not resume a v9 optimizer/checkpoint run while changing its dataset.
Preparation caches fingerprint the input; the changed v10 data must get a new
preparation/tokenization cache entry. After that it can be reused normally.

```bash
python training/scripts/verify_refined_archive.py \
  --dataset-dir /data/full_data_archive_recovered_v10 \
  --strict-sample-size 512 \
  --report /data/v10-host-verification.json

python training/scripts/run_golden_training.py \
  --profile e2b --model-dir /models/e2b \
  --input-dir /data/full_data_archive_recovered_v10 \
  --output-dir /runs/e2b-v10-smoke \
  --steps 20 --prepare-only
```

Then use the [training quickstart](GOLDEN_E2E_QUICKSTART.md) or
[full GPU deployment runbook](GOLDEN_GPU_DEPLOYMENT.md), retaining the new v10
input path. Model-tokenizer preflight, training, evaluation and deployment
runtime checks still happen on the GPU PC; offline validity does not replace
them.

## Reproduce from v9

Use Python 3.12 and `training/requirements-archive-recovery.txt`. All destination
paths must be new. Original UTF-16 files, deleted older recovered copies and
historical SQLite indexes are **not** required: completed v9 plus this code and
the frozen Golden files are sufficient.

```powershell
python training/scripts/finalize_archive_semantics.py `
  --base-dir training/outputs/datasets/full_data_archive_recovered_v9 `
  --output-dir training/outputs/datasets/full_data_archive_recovered_v10 `
  --report-dir training/outputs/audits/archive_semantic_report_v10 `
  --workers 8 --execute

python training/scripts/verify_refined_archive.py `
  --dataset-dir training/outputs/datasets/full_data_archive_recovered_v10 `
  --strict-sample-size 512 `
  --report training/outputs/audits/archive_semantic_report_v10/verification.json
```

Without `--execute`, the command only prints a plan. `--workers 0` selects a
CPU/memory-aware worker count. Workers use bounded input queues, deterministic
row order and a schema-compatible compiled validator; the independent verifier
uses the original production validator on a deterministic sample. Progress
includes scanned/retained/quarantined counts, throughput and estimated time.

Exports are staged in a fresh `.partial-<pid>` sibling directory and renamed
only after reconciliation and integrity checks. Never train from a partial
folder. Interrupted partial files are retained for diagnosis and are not
silently reused or deleted. Source hashes are checked before and after export.

## What changed

- Resolve table paths with Android JSON-pointer semantics. A literal root key
  containing `/` can be referenced with its exact escaped pointer; no data is
  invented or moved between unrelated state branches.
- Remove only unsupported, empty `image`/`imageAlt` columns where no image is
  supplied and the columns are not used by a primary/highlight/numeric binding.
- Restore a shortened table label only from one matching source table.
- Restore an existing Text clause only when its words form an ordered
  subsequence of one unambiguous source clause. No new layout nodes are added.
- Correct an existing citation action only from a unique, matching source
  citation label/reference. Missing actions or citations are not fabricated.
- Repair partial encoding only when the byte conversion is reversible and the
  resulting text is explicitly supported by the source.
- Undo the proven `asset/liability` false asset mask only with its retained map,
  and prevent the same normalization error on future preparations.
- Check projected table cells, source media, inline actions, hidden date
  columns, source qualifiers, prose omissions, placeholder closure and field
  values that incorrectly contain UI-description text.
- Preserve valid renderer-local checkbox behavior. Unused state alone is not
  proof that a control is broken.

The hash-bound catalog in `training/data/quality/v9_manual100_findings.json`
also prevents known audited source errors or omitted details being silently
retained. The ordinary rules apply across the complete dataset, not just the
100 audit rows. Source holds apply to all target siblings for a known source.
Narrow GDPR, QR-analytics and allergy-translation risk screens extend coverage
to similar source patterns; this is not a general factual-verification engine.

## Repair versus quarantine

Accepted repairs are deterministic preparation corrections with recorded
before/after evidence, source/target hashes and strict canonical round-trip
validation. The pass checks that they are idempotent. It does not manually edit
individual IR records, synthesize missing content, or run Stage 3 from training.

An arithmetic contradiction in the source is **not** permission to rewrite a
medical/financial scenario. Ambiguous source assumptions, missing whole
components/actions, and factual/safety errors stay quarantined until source
review and, where required, Stage 3 regeneration. Fix the generation pipeline
and regenerate those examples rather than hand-writing replacement IR.

Some conservative content gates can also hold legitimate paraphrases or
redundant introductory wording. `quarantine` means excluded from this candidate
copy pending review, not a claim that every excluded row is wrong forever.

Existing v9 train/validation family assignments are preserved among retained
rows; validation is not randomly reshuffled. Effective duplicates created by
repairs are excluded deterministically. Golden data is never modified or used
to write training targets. The independent verifier checks exact/normalized
and lexical-neighbor Golden exclusion and train/validation separation.

## Remaining limits

Recovered records still lack original questions and complete generator/asset
lineage. These rules cannot verify that a Stage-2 response answered its original
question, certify every factual claim, or guarantee device appearance. They
also do not establish model-token lengths or Golden-score improvements. Keep
the independent validation and a fresh manual audit before a production run.
