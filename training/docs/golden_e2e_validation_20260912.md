# Shared-prompt dual-Golden workflow validation — September 12, 2026

Scope: the new `run_golden_training.py` HF training/checkpoint-testing entry
point, one versioned production prompt for train/validation/Golden32/Golden35,
strict preparation, source reservations, dual-cohort run bindings, failure
recovery, and TensorBoard run identity. Existing official retained-scale E2B
QAT/LiteRT/MTP workflows remain separate and covered by regression tests.

## Completed checks

- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest training/tests -q`:
  **565 passed in 106.61 seconds**, in one uninterrupted run.
- The three new shared-prompt/dual-contract/workflow test files separately:
  **36 passed in 48.09 seconds**.
- Critical Ruff (`E9,F63,F7,F82,F401`) on changed/new Python files: passed.
- `python -m compileall -q training/src training/scripts patches/build_bottom_up_dataset.py`:
  passed.
- New launcher `--help` executed successfully; documented options checked
  against the parser. `git diff --check`: passed.

CPU integration tests use the actual checked-in Golden records and schema,
with explicitly fake tokenizers, model files, GPU inventory and training/
generation results. They verify orchestration, not trained-model quality:

- All **32 occurrences / 31 unique sources** of archive-repeat Golden32 and
  all **35 unique references** of Golden35 retain their frozen membership,
  source responses, reference semantics and repeat/exclusion provenance.
- All 67 evaluation occurrences share the exact prepared production
  system/few-shot/task scaffold and inference prefix, including task whitespace
  normalization. Previous prompt hashes/scaffolds are preserved for audit.
- Strict-invalid and overlength training rows are quarantined whole. Evaluation
  reference length is not used to discard a Golden case; overlength inference
  prompts or any invalid Golden reference abort publication.
- Both accepted and excluded Golden sources are reserved by identity and
  raw/URL-masked response hash. Source/query/response aliases remain grouped
  transitively across the default train/validation split.
- The completed Stage 3 source branch and existing train/validation branch
  are tested. The former materializes strict Express data before filtering;
  neither branch invokes cloud generation.
- Real configuration/preflight binding logic plus mocked GPU execution covers
  E2B LoRA, 270M SFT and 270M W8 QAT. Each runs four final evaluations:
  best/Golden32, best/Golden35, final/Golden32 and final/Golden35.
- Evaluation checks the saved model/tokenizer, config, split, prompt,
  benchmark and preparation hashes before loading weights. Tests reject
  drift and loaded-tokenizer mismatch. Selected CUDA masks and parent run IDs
  propagate to final testing and `/tensorboard/<run-id>/`.
- Plan-only makes no output and loads no tokenizer/model. Prepare-only loads
  only the tokenizer. Continuation checks completed artifacts and options;
  interrupted training is never silently restarted. Failed final evaluations
  use fresh attempt directories without retraining. Atomic manifest publication
  preserves the prior record on a failed replacement, and missing required
  outputs cannot mark a stage complete.
- Existing regressions cover mocked 2/4/8 H100 auto-selection, distributed
  periodic/final Golden generation, checkpoint selection and actual-final
  provenance, generation stopping/cache restoration, QAT and export gates.

## Tracked inputs and independent pins

The tracked default source has **4,870 Stage 3 records** in
`dataset/data/runs/dataset_v1/genui.jsonl` and **5,069 response records** in
its `responses.jsonl`. These are raw counts, not accepted training counts.
Exact retained counts, quarantine reasons and model-specific token lengths
are produced in `data_audit.json` and `prepared/manifest.json` on the GPU host.
No full-corpus retention or augmentation-quality result is claimed here.

Both Golden datasets and their adjacent `benchmark_manifest.json` files
were verified tracked and byte-identical to these independent pins:

| Artifact | SHA-256 |
|---|---|
| Archive-repeat Golden32 JSONL | `8c7357103e6ea52d99d66430dd4b93242e1c4a4f0bf02f2fcfdf2a2e9c48de4c` |
| Archive-repeat Golden32 manifest | `2b62797db7fd267c3f75e8ab1a7cb1a450d601df3f7785f5b78a62fe2657caa8` |
| Golden35 JSONL | `8fec7fde8c31634f66e4c77dd0ca7a0e3b37d36e453398f167fd30b9604c2d20` |
| Golden35 manifest | `1d468e05015744b8165c03e92ed009e03aac845513471b8f0b99cbe9d93939fb` |

Raw benchmark files were not rewritten by this change. Their Git attributes
preserve byte pins on Windows and Linux. Prepared prompts have a new contract;
new scores must not be compared directly with historical short-prompt scores.

## GPU-host work still required

No real model weights, model tokenizer assets, CUDA training/inference,
conversion or LiteRT runtime were loaded/run here. The real-model forward
preflight and short smoke in the [quickstart](GOLDEN_E2E_QUICKSTART.md) remain
mandatory. CPU tests do not establish H100 memory fit, measured throughput,
distributed runtime stability, convergence, prediction quality or deployment
parity. Periodic Golden generation synchronously pauses training; final tests
run sequentially on one selected GPU. No zero-overhead inference is claimed.

This is the current implementation record. The earlier
[September 12 record](validation_20260912.md) preserves historical results for
the preceding Golden35/GPU changes and the external review archive.
