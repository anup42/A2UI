# Augmentation / sequential tuning implementation review

This is code/CPU-test evidence, not a training run or a Golden quality score.
The complete review, data imbalance table, operational limits and GPU-host
commands are in [Augmentation and tuning](../../docs/AUGMENTATION_AND_TUNING.md).

## Decisions and implemented scope

- Retain the unaugmented baseline; optional `rare_components` mode resamples
  only exact, already validated training examples into a new prepared copy.
- Rarity is measured from connected training-source families. Defaults: at
  least five families, at most 3% of families, at most 10% extra rows, at most
  two total occurrences per family. Existing larger families are not removed.
- No Stage 1/2/3 calls, target rewrites, source paraphrases, new model downloads,
  contamination of validation/Goldens, or augmentation of quarantined records.
- Expose validated LR, weight decay, warmup, seed and logging overrides.
  Preserve baseline recipes; the search must supply measured evidence before
  any candidate can be called better.
- Run matched-step trials one at a time, from fresh base weights. Seed before
  model/LoRA initialization as well as in Trainer. Pin code, recipes, data,
  model and tokenizer identities across trials.
- Select on unique-source Golden32 reward; pin the winner before one Golden35
  holdout evaluation. Preserve best and final development scores separately.
- Write unique per-trial TensorBoard runs, suite HParams/scalars, JSON status,
  comparisons, selected checkpoint bindings and a full-training handoff.
- Reduce redundant tokenizer calls without removing vocabulary, prefix,
  masking or overflow checks. Respect CPU affinity/container quotas, bound
  default DataLoader and BLAS/OpenMP worker counts, and show rank/stage progress.
- Preserve all-visible-GPU DDP/H100 profiles and 2,048 generated tokens.
  Periodic Golden generation remains synchronous, not zero-overhead inference.

## Evidence available in this review

The final v9 manifest SHA256 was rechecked:
`cdce0bc4c537ed222a9ad0f5822fbc5772cb0c4df97a15a51e7c88f8ce1c0251`.
Its tracked full-census report records 112,842 train / 2,300 validation rows,
98.78% Table presence in training, and very sparse interactive/media coverage.
The listed candidate component presences sum to 5,584 before overlap, family
caps and tokenizer rejection. **That is only an upper bound, not an actual
augmentation count.** Exact added counts are computed on the GPU host's
tokenizer-admitted prepared training split and saved in `augmentation.json`.

Focused CPU tests cover deterministic copies, source-family caps, validation
and Golden byte preservation, actual Golden32/35 prepared contracts, option
forwarding including QAT overrides, mask/vocabulary checks, early RNG seeding,
hardware metadata and sequential orchestration failure paths.

A real TensorBoard event-file roundtrip passed in an isolated Python virtual
environment with TensorBoard 2.21.0: two synthetic trial runs, HParams plugin
metadata, optimizer steps and scalar values were written and reloaded. These
were explicitly fixture values, not model/benchmark measurements. No global
Python installation was changed.

A separate isolated runtime exercised real Hugging Face `DatasetDict.map`
with `datasets 5.0.1`, `dill 0.4.1` and `multiprocess 0.70.19`. Two synthetic
CPU rows passed formatting, completion-only tokenization and vocabulary
validation with exact expected IDs/masks/labels, six tokenizer calls, no
Arrow cache files and no fingerprint/Python warnings. This verifies the live
dataset API and progress callbacks, not model-tokenizer or CUDA compatibility.

## Final verification

- Full frozen-code suite: **750 passed, 1 skipped in 573.06 seconds** using
  `python -m pytest training/tests -q --tb=short` in the isolated TensorBoard
  environment. The one skip was the OS-privileged Windows symlink-creation
  test; it was not a skipped training or Golden validation failure.
- Separate real TensorBoard roundtrip: **1 passed**; included again in the
  full-suite result above, so do not add it twice to the full test total.
- Separate real Hugging Face DatasetDict preprocessing probe: passed on two
  synthetic CPU rows, as described above.
- Critical Ruff checks and `git diff --check`: passed.
- All three updated/new CLI `--help` entry points: exit code zero.
- Frozen Golden file hashes rechecked unchanged: Golden32
  `8c7357103e6ea52d99d66430dd4b93242e1c4a4f0bf02f2fcfdf2a2e9c48de4c`;
  Golden35 `8fec7fde8c31634f66e4c77dd0ca7a0e3b37d36e453398f167fd30b9604c2d20`.

The temporary TensorBoard and datasets validation environments were created
under repository `tmp/`. Cleanup was blocked by the execution policy, so
`tmp/tensorboard_validation_20260914` and `tmp/datasets_validation_20260914`
remain local and are not intended for Git. Original/repaired datasets,
benchmark files, checkpoints and pre-existing scratch files were not deleted
or modified. No training PC command, model training, real Golden
inference, export, or measured speed/quality benchmark was run here.

## Still requires the training host

- Install the documented training dependencies and supply the complete local
  model bundle plus the external v9 data copy.
- Run exact-tokenizer preparation and real-model forward/preflight checks.
- Measure H100 throughput, peak memory, CPU/RAM use and evaluation overhead;
  static profiles are not proof of optimal hardware utilization.
- Train the baseline/candidates, inspect loss/gradients and per-case errors,
  and only then judge resampling or hyperparameter quality.
- Add independently sourced coverage later for unsupported/rare components;
  resampling cannot fix missing Video or single-example Audio coverage.

Official-format E2B QAT, LiteRT-LM exports and MTP remain separate workflows;
this implementation does not establish their trained/exported runtime parity.
