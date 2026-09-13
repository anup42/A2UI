# Recover the external messages-only training archive

For the newest copy with further content review, stronger Golden isolation,
and rebuilt validation, follow the
[final v9 workflow](messages_archive_final_review.md) after this v5 stage.
The v5 counts below are a frozen intermediate, not the latest training split.

This runbook rebuilds the verified offline candidate from the exact external
archive without Stage 3 or model inference. It never overwrites the originals.
The completed analysis and exact counts are in the
[policy-v5 report](../reports/offline_recovery_20260913_v5/REPORT.md).

## Inputs and outputs

Required source directory:

- `train.jsonl`: 150,292 UTF-16LE rows, SHA-256
  `5c47fb3f07dc6f2711ece12a1a93b4ecd6977fa8060119f329db3189c2d6b910`
- `val.jsonl`: 910 UTF-16LE rows, SHA-256
  `04e88d07003803279cea7de093bd9c69bcd35ed9a5fab057d577544b8c90ba19`

The generated UTF-8/LF candidate is intentionally under ignored
`training/outputs/`. A normal Git clone does not contain the 2.8+ GB training
artifact. Either copy the complete verified output directory to the GPU host,
including its manifest/decision files, or rebuild it there from the exact
inputs.

## Rebuild on Windows

Run from the repository root. Use fresh output paths if any destination already
exists; the scripts fail rather than overwrite completed evidence.

```powershell
python -m pip install -r training/requirements-archive-recovery.txt

$Archive = 'C:\Users\anupk\Downloads\training_data'
$Audit = 'training/outputs/audits/full_data_20260913'
$Recovered = 'training/outputs/datasets/full_data_archive_recovered_v5'
$RunEvidence = 'training/outputs/audits/offline_recovery_run_v5'

python training/scripts/audits/full_data_inventory_20260913.py `
  --source-dir $Archive `
  --output-dir "$Audit/inventory" `
  --index "$Audit/inventory.sqlite" `
  --skip-provenance

python training/scripts/audits/full_data_ir_20260913.py `
  --source $Archive `
  --output $Audit `
  --report "$Audit/ir" `
  --workers 16

python training/scripts/recover_full_data_archive.py `
  --source-dir $Archive `
  --audit-root $Audit `
  --source-manifest training/reports/full_data_audit_20260913/audit_manifest.json `
  --near-duplicates training/reports/full_data_audit_20260913/near_duplicates/pairs.json `
  --output-dir $Recovered `
  --report-dir $RunEvidence `
  --workers 16 `
  --execute

python training/scripts/verify_recovered_archive.py `
  --dataset-dir $Recovered `
  --strict-sample-size 256 `
  --report "$RunEvidence/verification.json"
```

The recovery pass should report policy
`messages-archive-recovery-v5-20260913`, 117,841 train rows, 690 validation
rows, and 32,671 quarantined rows. Stop if the source hashes or these frozen
policy counts differ; inspect rather than forcing the run through.

## Use an already copied candidate

Verify it before model preparation:

```bash
python training/scripts/verify_recovered_archive.py \
  --dataset-dir /data/full_data_archive_recovered_v5 \
  --strict-sample-size 256 \
  --report /data/recovery-verification.json
```

The verifier checks all output/artifact hashes, every row envelope and source
identity, source-family split isolation, deduplication, placeholder closure,
and a deterministic production-preparation sample.

## E2B preparation, smoke, and full run

The local model directory must contain the complete dense HF bundle: weights,
`config.json`, tokenizer files, and chat template. Nothing is downloaded
automatically.

First run the exact E2B tokenizer/template gate:

```bash
export A2UI_TENSORBOARD_ROOT=/tensorboard

python training/scripts/run_golden_training.py \
  --profile e2b \
  --model-dir /models/e2b \
  --input-dir /data/full_data_archive_recovered_v5 \
  --output-dir /runs/e2b-archive-v5-smoke \
  --steps 20 \
  --prepare-only
```

If preparation succeeds, continue the same bound run:

```bash
python training/scripts/run_golden_training.py \
  --profile e2b \
  --model-dir /models/e2b \
  --input-dir /data/full_data_archive_recovered_v5 \
  --output-dir /runs/e2b-archive-v5-smoke \
  --steps 20 \
  --continue-run \
  --execute
```

Then use a fresh directory for the full run:

```bash
python training/scripts/run_golden_training.py \
  --profile e2b \
  --model-dir /models/e2b \
  --input-dir /data/full_data_archive_recovered_v5 \
  --output-dir /runs/e2b-archive-v5-epoch1 \
  --epochs 1 \
  --execute
```

## Gemma 3 270M

Use the same verified candidate and a separate output directory:

```bash
python training/scripts/run_golden_training.py \
  --profile 270m \
  --model-dir /models/gemma-3-270m-it \
  --input-dir /data/full_data_archive_recovered_v5 \
  --output-dir /runs/gemma270m-archive-v5-smoke \
  --steps 20 \
  --execute
```

Use `--prepare-only` first when possible, exactly as for E2B. A subsequent
270M W8-QAT run must start from the selected local full SFT checkpoint and use
`--qat`; it is not an official E2B retained-scale conversion.

## Evaluation defaults

- All scheduler-visible H100s are used; 2/4/8-GPU profiles are selected
  automatically.
- Validation loss and checkpoint saves run every 500 optimizer updates.
- Golden32 generation runs every 1,000 optimizer updates and at final weights.
- Golden32's unique-source score selects the best checkpoint.
- Golden35 is final-only and never selects a checkpoint.
- Selected-best and actual-final checkpoints are each tested on Golden32 and
  Golden35.
- Generation defaults to 2,048 new tokens.
- TensorBoard is written below `/tensorboard/<run-id>/`.

The preparation stage can reduce the final row count because only the real
model tokenizer and authoritative chat template can establish the complete
sequence length. Reject overlong examples whole; do not clip labels. No GPU
throughput or quality result is established until the real run finishes.
