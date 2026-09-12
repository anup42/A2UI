# September 12 implementation validation

Scope: explicit archive Golden replacement, Golden35 selection, data audit/filtering, automatic
H100/visible-GPU launch profiles, distributed generation efficiency, final and
selected checkpoint provenance, and central TensorBoard integration.

## Earlier September 12 validation before the Golden35 follow-up

The counts and test totals below record the preceding implementation revision.
They are historical validation evidence, not a claim that its old 50-row
configuration remains the active evaluation path.

- `python -m pytest training/tests -q`: **502 passed** (final rerun: 44.25 seconds).
- Critical Ruff (`E9,F63,F7,F82`) on all changed/new Python files: passed.
- `python -m compileall -q training/src training/scripts`: passed.
- `git diff --check`: passed. Git reports existing checkout line-ending
  conversion warnings; hash-bound benchmark files have explicit attributes.
- Repeated archive benchmark: exactly 32 strict-valid occurrences / 31 unique
  sources. Raw SHA-256:
  `8c7357103e6ea52d99d66430dd4b93242e1c4a4f0bf02f2fcfdf2a2e9c48de4c`.
- Raw repeat artifact → tokenized preparation → embedded benchmark manifest →
  review launch binding was tested with a clearly labelled fake tokenizer and
  mocked hardware. Unique-source selection, donor averaging, validation overlap,
  excluded failed-source leakage and manifest tampering have regression tests.
- Original September3 Golden32 was prepared into a new
  `golden32_20260903_eval_prompt_v2` cache without overwriting the old cache or
  source records. Current prompt/LF output SHA-256:
  `eefdfedaca1395477d279712697843fc0324fd41cb669f2c01ecdd098c11ea52`.
  Both canonical pipeline pins agree.
- GPU tests mock 2/4/8 H100s, visibility masks, UUID/MIG devices, incompatible
  effective batches, missing GPUs and stale host plans. They check actual
  torchrun command construction and preserved retained-QAT configuration.
- Generation tests cover scoped cache restoration, final evaluation and
  same-step suppression, resume selector state, distributed failure propagation,
  explicit inference placement, stale external-output rejection, correct
  checkpoint steps and per-artifact metadata/merge evidence.
- Data findings and exact 15 exclusions from the historical 50-case source are in
  [the data audit](data_filtering_audit_20260912.md).

Whole-tree critical Ruff also reports five existing undefined type-annotation
names (`Dataset`/`DatasetDict`) in the unchanged `training/scripts/train_grpo.py`.
They are outside this SFT/QAT update; the changed-file check is clean.

## Golden35 follow-up: current evaluation contract

The user selected the 35 passing references from the original 50-case source
as **Golden35**, stored at `training/data/eval/golden35_v1/golden35.jsonl`.
The active dataset configuration is `training/configs/datasets/golden35_stage3_eval.yaml`;
preparation writes `training/outputs/datasets/golden35_stage3_eval/all.jsonl`.
Standalone final comparisons require `--max-rows 35 --required-rows 35`.

The manifest preserves the 35 retained source identities and all 15 excluded
identities. Training filters reserve both groups through `--reserve-golden35`;
the archive Golden32 development experiment remains a separate reservation and
selection contract. The original raw source folder and the historical failure
list remain available. No failed target is rewritten and no raw source is deleted.

Current Golden35 artifact checks:

- Frozen source: exactly **35 strict-valid rows / 35 unique sources**, with all
  15 excluded source identities and original/URL-masked hashes retained in the
  exclusion manifest. Artifact SHA-256:
  `8fec7fde8c31634f66e4c77dd0ca7a0e3b37d36e453398f167fd30b9604c2d20`.
- The documented `create_golden35_subset.py` command reproduced that exact
  artifact hash from the pinned original sources. Historical source files were
  not edited. Git attributes preserve the source/artifact line endings across hosts.
- The active dataset preparation command completed with **35 accepted / 0
  quarantined**. Prepared `all.jsonl` SHA-256:
  `bfa025ef12cbccb23612621935b3220bd8e32cea43518cb5cd99dca5090b96ac`.
- CPU integration tests cover raw/prepared manifest binding, fixed membership,
  source/target tampering, repeated preparation, excluded-source reservation,
  scoring identity and rejection of partial/mixed/duplicate prediction cohorts.
- Critical Ruff on changed Golden35 Python files, compileall and whitespace
  checks passed.

Regression evidence for this revision (no model loading):

- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest training/tests -q`:
  **525 passed, 3 failed, 1 setup error** in 67.47 seconds. All four remaining
  outcomes were host `MemoryError` failures reading 1–1.5 MB files, not assertion
  failures. The host also reported OpenBLAS/.NET allocation failures during
  earlier attempts; no unrelated processes or system settings were changed.
- Bounded rerun of `test_gemma270m_multiformat_pipeline.py`,
  `test_golden35_pipeline.py` and `test_gpu_training_profile.py`, with the same
  thread limits: **46 passed** in 10.59 seconds. This covers all four memory-error
  tests and the explicit repeated-benchmark hardware fixtures.
- Thus all **529 unique tests** passed across the full attempt and bounded
  rerun, but a single uninterrupted 529-pass full-suite run was not established
  on this memory-constrained host. The original 502-test record remains separate.
- The Golden35 consumer tests include MTP defaults, cohort checks before model
  loading, metadata/context preservation, and mocked latency repeats that do
  not multiply the number of scored cases. Prediction-score tests use fixtures,
  not measured model outputs.

## Not performed or established

No model weights were loaded, no training or GPU inference was launched, and
no conversion, LiteRT runtime/device test, or measured H100 speedup was run.
The real full training corpus and tokenizer length distribution were not
available. The documented profiles are starting points, not proven optimal
microbatches. Full data filtering/token checks and a real GPU smoke remain
mandatory. Golden35's 35 passing references are the current final-evaluation
cohort; the historical 15 exclusions are reserved rather than scored. An actual
LiteRT batch runner is still host-provided. No fabricated model scores or performance
measurements have been added.
