# Prepared prompt snapshot fix and separate Vulkan host blocker

Date: 2026-09-16. No source dataset, target IR, Golden cohort or checkpoint was
modified. No remote training, driver install or native GPU inference was run.

## What the screenshots establish

- The tuning launch failed in `verify_launch_binding -> verify_prepared ->
  checked_preparation_manifest -> validate_shared_prompt_contract` with a
  shared-prompt mismatch. The remote saved contract was not supplied, so the
  particular changed field/file is not established by the screenshot.
- The old runtime validator regenerated its expectation from the live checkout
  rather than validating the existing, bound prompt snapshot. Local probes
  across six different Python hash seeds returned identical contracts; no
  random hash-order failure was reproduced.
- The later deployment failed at the early Vulkan runtime prerequisite gate:
  `libvulkan.so.1` could not be loaded. This is a separate container dependency
  failure, not a prompt/UUID failure or a measured zero Golden score.

## Changes and safeguards

- Validate saved prompt schema, system/scaffold/contract hashes and file hashes
  without regenerating the prompt at training/evaluation time. Check saved
  scaffold and inference artifacts against that same contract.
- Build inference messages from the saved task prefix and messages. Never
  substitute the current prompt into old prepared examples.
- Keep current-production parity mandatory for fresh preparation. Require
  restored/cache manifests to match the planned contract. Report differing
  fields and saved/current hashes for a stale new plan.
- Validate small prompt artifacts before scanning the training corpus.
- Retain tokenizer, split, contamination, model/config, tuning implementation
  and completed-artifact bindings. No automatic failed-trial/optimizer restart,
  blind manifest rehash, cache deletion or CPU inference fallback was added.

## Verification

Targeted Windows CPU/mock suites passed: **321 distinct tests, 2 skipped**.
Execution was grouped as follows (the 16 new snapshot tests ran twice):

1. Frozen/shared prompt, dual-Golden contract and preparation-cache suites:
   **72 passed**.
2. Training pipeline, sequential experiments, augmentation and GPU-profile
   suites: **138 passed, 1 skipped**.
3. Frozen prompt, Vulkan probe, deployment recovery, UUID handoffs and export
   suites: **127 passed, 1 skipped**.

New coverage checks valid frozen launch/Golden32/Golden35 after live-builder
drift, snapshot tampering, cross-artifact disagreement even with refreshed
outer hashes, fail-fast prompt verification and fresh-plan drift rejection.
Scoped critical Ruff checks, full Ruff for the new tests and whitespace checks
passed. These are not H100/model-quality/native-kernel certification.

## Training PC next steps

See [the prompt and recovery runbook](../../docs/GOLDEN_GPU_DEPLOYMENT.md#prepared-shared-prompt-snapshots)
and [Vulkan setup](../../docs/DEPLOYMENT_EXPORT_ENVIRONMENT.md#linux-vulkan-prerequisites-required-for-native-gpu-testing).
On Ubuntu/Debian, the image owner must install `libvulkan1` (and optionally
`vulkan-tools` for diagnostics), expose the matching host NVIDIA Vulkan driver
with `compute,utility,graphics` container capabilities, and pass the native
preflight in that same container. A repository commit cannot mount missing host
driver libraries. Do not install a replacement NVIDIA host driver in the pod.

Retain the failed tuning output and use a fresh output directory after updating;
post-training `--resume-run` does not resume failed tuning. Do not update an
active checkout. Existing compatible caches remain usable, but this changed
preprocessing implementation invalidates earlier preparation identities once.
