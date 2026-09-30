# v11 training dataset bundle

This publishes **200,214 training** and **4,089 validation**
records as ordered gzip parts. The original JSONL bytes, row metadata and split
order are preserved; packaging performs no repair, filtering or regeneration.
The supplied `manifest.json` and `prompt_scaffold.json` are preserved byte for
byte and are verified during restoration.

## Restore and train

Use Python 3.11 or newer; no extra packages or Git LFS are required. Restore to a
fresh directory with at least 5.46 GB free:

```bash
python training/data/train/v11/restore.py --output-dir /data/a2ui_v11
```

On Windows, supply a Windows output path. Restoration verifies each compressed
part plus the complete original split SHA-256, byte size and row count. Existing
directories are refused. An unsuccessful restore retains a separate partial
directory for diagnosis; only a fully verified copy receives the requested name.
Verify without writing an extracted copy:

```bash
python training/data/train/v11/restore.py --verify-only
```

Use the existing [training and deployment command](../../../docs/GOLDEN_GPU_DEPLOYMENT.md)
with `--input-dir /data/a2ui_v11` and a fresh `--output-dir`. Model-specific
tokenizer preparation must run; do not pass this copy as `--prepared-input-dir`.
Preparation still applies holdout exclusion, validation and tokenizer checks.
Training, checkpoint testing and exports can use `--skip-litert-evaluation`
when native inference is unavailable; the model and exporter environment is
still required. Golden evaluation data is already tracked elsewhere.

## Relationship and quality limits

Source: the supplied `v11` splits, policy `space-v11-v10-source-proven-repairs-20260930`,
status `offline_verified_candidate`. v11 combines the retained/repaired v10 base and
newly retained sources; v11s is the new-source subset. **Do not concatenate v11
and v11s:** every v11s split is already included in the corresponding v11 split.
v11s validation is the new-only holdout within v11 validation. v11s training is
not an unseen evaluation set for a v11-trained model.

A subsequent [generation review](../../../docs/dataset_v11_generation_review_20261001.md)
identified confirmed unsafe text joins and unsupported Chart rows. Normal
training preparation can exclude these reviewed defects while preserving the
original release bytes. The row counts in this README and source manifest are
release counts, not the number admitted by training preparation.

Source manifests report offline repair and verification, with tokenizer
preparation deferred and no records truncated. These checks do not certify
source factual correctness, Android rendering or improved trained-model quality.
This publication is an offline candidate and requires normal training preparation
and evaluation. The source manifest references audit files outside this bundle;
those paths are provenance, not portable local prerequisites. The full audit
ledgers and source-generation archive are not duplicated here.
