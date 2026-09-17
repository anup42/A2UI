# v10 training-only dataset

This folder publishes only the two source files consumed by the training
launcher's `--input-dir`: **91,115 training** and **1,862 validation** records.
They are compressed into ordered gzip parts to keep each Git object small.
The bundle preserves the original JSONL bytes, row metadata and split order;
it does not filter, repair or regenerate any sample during restoration.

## Restore after cloning

Use Python 3.11 or newer; no extra packages or Git LFS are required. Choose a
new directory with at least 2.6 GB available:

```bash
python training/data/train/v10/restore.py --output-dir /data/a2ui_v10
```

On Windows, choose a Windows path for `--output-dir`. The helper validates every
compressed part and the complete original-file SHA-256, byte size and row count.
It refuses to overwrite an existing directory. Failed partial copies are kept
separately for diagnosis; only a fully verified copy gets the requested name.
To check the bundle without writing an extracted copy:

```bash
python training/data/train/v10/restore.py --verify-only
```

Then use the existing [deployment command](../../../docs/GOLDEN_GPU_DEPLOYMENT.md)
with `--input-dir /data/a2ui_v10` and a **fresh** `--output-dir`. The existing
`--skip-litert-evaluation` option permits training, checkpoint testing on
Golden32/Golden35/Bixby50, and all four exports without Vulkan/native inference.
The local model and exporter environment are still required.

Normal training preparation performs its current holdout-exclusion, validation
and tokenizer checks and produces a new prepared manifest. This training-only
copy intentionally omits archive audit ledgers, quarantine files and the full
recovery manifest. Do not run the full-archive recovery verifier on this copy.
Golden test files are already tracked elsewhere and are not duplicated here.

## Scope and quality limits

Source: `full_data_archive_recovered_v10`, completed 2026-09-16. The compact
`bundle.json` binds both original split hashes and the source manifest digest.
All recovered row metadata is retained because preparation uses identities and
reference bindings. Original v9/v10 files are unchanged.

This remains an **offline-validated candidate**, not a fully quality-approved
dataset. A subsequent manual review of 100 examples found 69 keep, 26 requiring
repair/review and 5 reject/regenerate dispositions; those later corrections
have **not** been applied. These are sample findings, not a population accuracy
estimate. Original queries/full provenance remain unavailable, and no model
quality improvement is established by publishing this bundle.

Only training inputs and the small files needed to restore/verify them are
published here; review reports and audit artifacts remain local.
