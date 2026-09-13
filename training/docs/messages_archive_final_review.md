# Final repaired archive copy: v9

The completed **v9** copy has **112,842 training rows and 2,300 validation
rows, 115,142 total**. Use it for the next exact-model tokenizer preflight.
The final pass corrected the false historical join at original `train:65649`:
`oilPrep` is now `oil. Prep`, with the separator taken directly from its source.
All three retained historical joins are independently rechecked. The earlier
disk-space interruption is resolved; failed `.partial-*` folders are not
training inputs and are not automatically removed.

The review discovered that intermediate
repaired letters still omitted paragraphs or short clauses and that v6
sometimes renamed a secondary source button unnecessarily. v7 corrects the
button rule, checks long prose blocks, filters failures, and rebuilds validation
from source families. v8 then excludes incomplete letters without editing
retained records or moving their final source-family split assignment.

The original UTF-16 files and the v5/v6/v7/v8 copies stay unchanged. Exact counts,
repair outcomes, benchmark exclusions, and remaining issues are recorded in
the [final report](../reports/offline_recovery_20260913_v9/REPORT.md),
[summary](../reports/offline_recovery_20260913_v9/summary.json), and
[profile](../reports/offline_recovery_20260913_v9/profile.json).

## Rebuild from the supplied archive

1. Reproduce the frozen v5 recovery and inventory using the
   [recovery runbook](messages_archive_recovery.md).
2. Build the intermediate v6 all-row review using the
   [refinement runbook](messages_archive_refinement.md).
3. Run the final-review stages below. Use fresh output/audit/report paths. Checked-in
   report folders already exist on another clone, so reproduction reports go
   under ignored `training/outputs/` by default in this example.

```powershell
python -m pip install -r training/requirements-archive-recovery.txt

python training/scripts/finalize_refined_archive.py `
  --base-dir training/outputs/datasets/full_data_archive_recovered_v6 `
  --review-index training/outputs/audits/archive_refinement_v6/review.sqlite `
  --source-dir C:/Users/anupk/Downloads/training_data `
  --inventory training/outputs/audits/full_data_20260913/inventory.sqlite `
  --output-dir training/outputs/datasets/full_data_archive_recovered_v7 `
  --audit-dir training/outputs/audits/archive_final_review_v7 `
  --report-dir training/outputs/audits/archive_final_report_v7 `
  --workers 12 --val-fraction 0.02 --seed 20260913 --execute

python training/scripts/finalize_archive_letters.py `
  --base-dir training/outputs/datasets/full_data_archive_recovered_v7 `
  --output-dir training/outputs/datasets/full_data_archive_recovered_v8 `
  --report-dir training/outputs/audits/archive_final_report_v8 `
  --execute

python training/scripts/verify_refined_archive.py `
  --dataset-dir training/outputs/datasets/full_data_archive_recovered_v8 `
  --strict-sample-size 512 `
  --report training/outputs/audits/archive_final_report_v8/verification.json
```

Complete the final source-bound separator pass:

```powershell
python training/scripts/finalize_archive_boundaries.py `
  --base-dir training/outputs/datasets/full_data_archive_recovered_v8 `
  --output-dir training/outputs/datasets/full_data_archive_recovered_v9 `
  --report-dir training/outputs/audits/archive_final_report_v9 --execute

python training/scripts/verify_refined_archive.py `
  --dataset-dir training/outputs/datasets/full_data_archive_recovered_v9 `
  --report training/outputs/audits/archive_final_report_v9/verification.json
```

The v9 pass preserves v8 membership and splits; one target and its provenance
hashes change. It checks available disk space before creating output, requiring
the estimated copy size plus 1 GiB headroom. The observed output is about
2.70 GiB. Do not use any `.partial-*` directory as training input.

The final review reuses the v6 audit index and rescans every v6 candidate.
It reconstructs the affected button examples from their hash-bound original
source rows, then applies the corrected rule. It does not modify the inputs,
generate missing content, or call Stage 3.

The paragraph screen flags a source prose block only when it has at least
120 characters, at least 16 distinct content words, at least eight absent
words, and less than 50% word recall against target properties plus referenced
state. Tables, code fences, and standalone media/action/header declarations
are not treated as prose blocks. This is a conservative review gate; it can
flag a legitimate paraphrase and is not a factual-correctness score. Referenced
table state is counted whole because native table routes can consume implicit
fields; actual device rendering remains a separate check.

The letter screen applies to responses beginning with `Subject:`, `Dear`, or
`To whom it may concern`. Each paragraph of at least four normalized words
must occur as an ordered word subsequence in target properties plus referenced
state. It tolerates extra headings and punctuation but intentionally rejects
rewritten or shortened letters. It is a conservative completeness policy, not
a guarantee of rendered order or perfect visibility. Other document styles
may still need review. The final pass checked 84 letter-style candidates,
retained 14, and quarantined 70, including both residual clipped-letter cases.

Building v8/v9 verifies the original source paths recorded in its input manifest.
For reproduction on another host, build the lineage there using the local
`--source-dir`, or copy the already completed v9 directory for consumption.
Allow at least 20 GiB for the v5-v9 dataset copies plus audit indexes, in
addition to the original archive and other existing outputs. All stages use
new destinations; they do not remove old copies automatically.

## Copy to the GPU host and train

The multi-GB dataset is ignored by Git. Copy the complete final v9 directory
or rebuild it there; a Git clone by itself does not contain this external
archive. Neither the original archive nor the audit SQLite files are needed
to verify and consume an already copied v9 dataset.

```bash
python training/scripts/verify_refined_archive.py \
  --dataset-dir /data/full_data_archive_recovered_v9 \
  --report /data/v9-host-verification.json

export A2UI_TENSORBOARD_ROOT=/tensorboard
python training/scripts/run_golden_training.py \
  --profile e2b --model-dir /models/e2b \
  --input-dir /data/full_data_archive_recovered_v9 \
  --output-dir /runs/e2b-archive-v9-smoke \
  --steps 20 --prepare-only

python training/scripts/run_golden_training.py \
  --profile e2b --model-dir /models/e2b \
  --input-dir /data/full_data_archive_recovered_v9 \
  --output-dir /runs/e2b-archive-v9-smoke \
  --steps 20 --continue-run --execute
```

After a successful smoke, use a new run directory with `--epochs 1 --execute`.
For Gemma 3 270M, use `--profile 270m` and its complete local model directory.
Prepare each model with its own tokenizer/template and reject overlong records
whole. Offline-valid counts can decrease at that final tokenizer gate.

Validation starts from a 2% source-family target, stratified by component shape
and normalized source length. Close source neighbors move together, so the
final fraction can differ slightly. Membership and row ordering use the saved
seed; all detected exact/normalized/lexical train-val overlaps must be zero.
Goldens and their reserved source families stay out of both ordinary splits.

Training retains the existing defaults: validation/save every 500 optimizer
updates, Golden32 every 1,000 plus final, Golden35 final-only, 2,048 generated
tokens, automatic visible-GPU/H100 profiles, and `/tensorboard/<run-id>/`.
Golden32 selects checkpoints; Golden35 does not. A model already exposed to
the old train pool is not a fresh held-out validation baseline. Start from
base weights for a clean experiment.

The dense Golden launcher is separate from official retained-scale QAT,
MTP, and LiteRT multiformat export. No GPU training, model conversion, device
rendering, or checkpoint quality score is established by this CPU data pass.
