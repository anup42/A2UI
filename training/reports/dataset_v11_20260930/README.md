# Supplied v11/v11s release audit

The original release evidence is preserved byte-for-byte. Large JSONL decision
ledgers and frozen benchmark sidecars are compressed for Git; no row is changed.
The original `audit_files_sha256.json` still refers to their uncompressed names.
`repository_import.json` records local verification of every package entry and
the compressed-to-original hashes. `verification.json` records the supplier's
full lineage replay; the original raw archives are needed to repeat that replay.

Verify every original audit file without extraction:

```bash
python training/reports/dataset_v11_20260930/restore_audit.py --verify-only
```

Restore the complete original audit tree to a new directory (about 239 MB):

```bash
python training/reports/dataset_v11_20260930/restore_audit.py --output-dir /data/a2ui_v11_audit
```

Use the frozen `policy_snapshot` for historical repair-policy evidence. Future
generation must regenerate targets through Stage 3; the imported offline repair
scripts do not become the production generator. The active reassessment test
now resolves this checkout or `A2UI_V10_POLICY_REPO` instead of a fixed D: path;
its original source remains in the frozen policy snapshot.

The supplied release is an offline candidate. The current review found defects
in the legacy text-chunk joining rule; consult
[the generation review](../../docs/dataset_v11_generation_review_20261001.md)
before training. Passing a recorded structural or heuristic fidelity check is
not proof of factual correctness, device rendering or improved model quality.

Restore training inputs with the separate helpers under
[`v11`](../../data/train/v11/README.md) or
[`v11s`](../../data/train/v11s/README.md). v11s is already contained in v11.
