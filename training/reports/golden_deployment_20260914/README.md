# Full GPU deployment implementation and validation

Date: 2026-09-14. This report records **code/test evidence**, not real model
quality or measured H100 throughput. No real training, LiteRT conversion or
native GPU benchmark was run on this Windows CPU development host.

## Delivered workflow

`training/scripts/run_golden_deployment.py` composes the current dense E2B/270M
training workflow, optional sequential screening followed by fresh full
training, best/final/merged checkpoint evaluation, physical-precision-audited
W32/W16/W8/W4 export, native GPU Golden32/35 evaluation and TensorBoard comparison.
See [the complete runbook](../../docs/GOLDEN_GPU_DEPLOYMENT.md).

The selected checkpoint is chosen using Golden32's 31-unique-source reward,
not the duplicated 32-occurrence average or Golden35. Screening suppresses its
winner's standalone holdout pass; Golden35 first runs after fresh full training.
The required complete output is 14 evaluations, not a partial subset.

## Reviews and fixes

- Standalone checkpoint evaluation now shards cases across isolated processes
  on all selected GPUs. It validates exact original-order coverage before
  publishing predictions. Failures terminate peers and retain partial files.
- Existing periodic Golden32 already uses all training ranks and resident
  models; the new wrapper preserves that implementation and its KV-cache policy.
- Kept conservative H100 batch/accumulation profiles and live backward preflight;
  scoped cuDNN-SDPA avoidance applies to training and HF generation.
- Added bounded subprocess/group teardown, phase/case deadlines, live logs,
  incremental native output and stage timings. No silent failed-case retry.
- Evaluation reuses a content/cohort/implementation-bound source-overlap scan,
  while retaining fresh model/train/validation byte checks. Full-suite review
  caught an overly broad preparation-cache dependency introduced by this change;
  removing the training-orchestrator import restored the intended boundary.
- Preserved Linux virtual-environment executable paths instead of resolving
  their symlinks to the system interpreter.
- Original hash-bound HF snapshot symlinks are supported; generated checkpoint
  inventories remain strict. Export checks actual physical matrix-weight types,
  not filenames or activation types.
- Actual local-tokenizer template parity is checked before training. Post-merge
  checks repeat against bound assets and a genuinely bounded three-row prefix;
  wrapping an eager full-file reader in `islice` was corrected during review.
- Verified native raw-session BOS insertion and adjusted only the proven native
  BOS prefix while checking effective token IDs against HF inputs.
- Native runtime prerequisite checks load its shared library. Successful model
  inference must show PID-specific GPU allocation and allowed UUIDs. Verified
  self `NSpid` identities support visible Jupyter/Kubernetes PID mappings; no
  arbitrary host PID or memory-delta attribution is used.
- Native compiled caches are separated by runtime version and full package
  SHA-256; identically named variant packages cannot collide.
- Independent reviews checked orchestration, cohort selection, executable paths,
  worker teardown, model merge/export contracts and the pinned upstream API.

## Automated verification

Final full suite: **1,114 passed, 3 skipped, 3 warnings in 307.51 seconds**.
No test failures remain. The warnings concern the tiny PEFT fixture's absent
base-config file and CPU-only pinned-memory fixtures; they are not real GPU
training failures. A separate SWIG deprecation message appeared at interpreter
shutdown. Platform skips do not certify Linux native GPU behavior.

Additional verification:

- New native runner suite: **56 passed** (mocked native API/evidence supervision).
- Export suite: **33 passed, 1 skipped**, including real tiny E2B merge/reload.
- Minimum Transformers 5.10.1: the real tiny full-wrapper E2B merge test passed.
- Preparation-cache/prepared-contract focused run after the discovered regression:
  **46 passed**.
- All seven touched/new launcher/evaluator `--help` checks passed.
- New module/test Ruff, affected-code critical lint and tracked whitespace checks passed.

Reproduction on the prepared Windows test environment:

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
& tmp/tensorboard_validation_20260914/Scripts/python.exe -m pytest training/tests -q
```

The environment is Python 3.12 with CPU PyTorch and Transformers 5.16.1.
Tests include real tiny CPU HF/PEFT generation subprocesses, real TensorBoard
HParams/scalar event roundtrips and both cold/warm token-cache training tests.
Native LiteRT/H100 orchestration uses explicit mocks/fixtures, never fake
benchmark results presented as real measurements.

The real tiny E2B test loads the full `gemma4` wrapper, saves/reloads a PEFT
adapter, merges it, saves/reloads the dense model, and verifies wrapper type,
nested `gemma4_text` configuration, tensor inventory and merged values. It also
passed on the minimum Transformers **5.10.1** environment. No configuration
relabeling or silent text-only export fallback is required.

New module/test Ruff checks and affected-code critical checks pass. A broad
repository lint scan still reports five pre-existing undefined annotation names
in the unrelated `training/scripts/train_grpo.py`; that script is not part of
this pipeline and was not changed. CLI help and whitespace checks are included
in final review. Original datasets and unrelated workspace files are untouched.

## Required host validation / limits

1. Install/probe the isolated [export environment](../../docs/DEPLOYMENT_EXPORT_ENVIRONMENT.md)
   and [native runtime](../../docs/LITERTLM_GPU_RUNNER.md). Core package pins
   were checked against release APIs/dependency metadata, not installed and
   exercised together on this development host.
2. Run the actual maximum-length backward preflight and a short real GPU smoke,
   then a fresh full run with the intended data/model/2, 4 or 8 GPU allocation.
3. Every exported package must pass physical-precision inspection and actual
   native GPU generation on all 32/35 cases. Unsupported kernels/driver/PID
   visibility fail explicitly; CPU tests cannot remove hardware uncertainty.
4. Conversion uses CPU. The pinned LiteRT Python API cannot choose a GPU ordinal,
   so it uses one verified native engine; all-GPU HF training/evaluation is not
   falsely advertised as all-GPU LiteRT inference. Allocation evidence does not
   prove every native operator executes on GPU.
5. W16/W4 are experimental; E2B W4 is mixed W4/W8. This is public post-training
   conversion of a dense trained checkpoint, not the official retained-scale
   mobile graph or an MTP model.
6. A `complete` manifest proves all requested evaluations executed with evidence,
   not a required accuracy threshold or production quality promotion. Inspect
   the generated rewards, validity rates, predictions and artifact details.

No checkpoints, model packages, original data edits or synthetic quality scores
are included in this implementation report.
