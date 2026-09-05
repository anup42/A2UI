# Implementation and validation — September 5, 2026

The historical [review](REVIEW.md), [merge report](MERGE_REPORT.md), and Golden32 CSV/JSON remain evidence of the supplied run. The fixes below supersede their open implementation recommendations. They do **not** replace the recorded checkpoint scores with unmeasured predictions.

The operational handoff is [GPU_PC_TRAINING_HANDOFF.md](../../GPU_PC_TRAINING_HANDOFF.md). It contains commands for E2B and 270M preparation, smoke training, checkpoint evaluation, QAT adaptation, optional GRPO and deployment gates. All real model operations belong on the GPU PC.

## Completed changes

- Shared native EOS preservation and generated-only, quote-aware closing-tag stopping across HF evaluation, Golden callbacks, SFT greedy probes and GRPO. Trainer startup restores the preserved EOS policy.
- Prediction records bind the actual source, expected target, asset/URL context and prompt fingerprints. Raw and serving-stopped outputs, stop reasons and token counts remain distinct. Frozen v5.4 weights/caps are unchanged; missing legacy metrics no longer shrink the implicit all-row denominator.
- Replaced unsafe regex bottom-up/ID-repair scripts with atomic preparation using the shared catalog, parser, renderer reference inventory and production wire schema. It preserves original IDs/graph semantics and quarantines ambiguous or invalid records.
- Added tokenizer-aware complete-row length checks and vocabulary/chat-template binding. SFT refuses truncation, empty conditioning/targets or mismatched assistant boundaries. Train, validation and Golden source checks fail on overlap.
- Fixed Stage3 repair-loop wire validation and the shared Table array instructions; regenerated matching dataset/Android prompt copies and IR manifest. Targets must be regenerated through Stage3.
- Implemented explicit full SFT/full QAT for 270M, full-model checkpoint provenance, and supported LoRA target resolution against actual Linear modules. Resume uses materialized targets and requires optimizer/scheduler/RNG, model hashes and identical dataset/recipe evidence.
- Respected disabled evaluation, including YAML `no`; corrected epoch-schedule warmup fallback and strategy-dependent cadence checks. Checked loss/gradient accumulation normalization is explicit and numerically tested.
- Added portable, file-only run preparation and print-first GPU launch planning. Device visibility, worker count and effective batch agree. Execution rechecks data/config/model bindings.
- Prevented saved QAT checkpoints from bypassing merge provenance through omitted or relabeled training configs.
- Added source-reviewed TRL 0.29.1 GRPO rollout/termination-mask integration, source/URL/scaffold parity, seed manifests, buffer checks, learning-health gates and actual Linear target resolution. No installed-package source is patched.

## New source-quality evidence

Strict preparation accepted 252 of 300 inspected training rows, 88 of 100 validation rows, and 31 of 32 Golden rows. These are samples, not full-corpus statistics. Failures include array/scalar type mismatches, Text/Icon types, entityMedia, an incomplete repeat and a detached graph. Valid Tabs remain supported. The full corpus needs the same audit and regeneration of quarantined source targets; do not conclude that ordering alone caused the failures.

Golden query `q_012053` / response `r_012053_01` requires Stage3 regeneration. The run preparer refuses to score only the remaining 31. Rebaseline all checkpoints against the versioned, repaired 32-row reference.

## Local validation

- Complete training CPU/mock suite: **442 passed** in 31.06 seconds.
- Related dataset Stage3/Express/prompt/boundary tests: **20 passed** in 2.52 seconds. Only existing deprecation warnings were reported.
- Generated prompt check: **4 files verified**. Generated IR artifact check: **21 files verified**.
- Portable preparation/launch binding was exercised with synthetic files; no model loader or optimizer was invoked. CLI help, Python syntax and scoped whitespace checks passed.
- All **8 pre-existing tracked Android/dataset file diffs** matched the initial review snapshot. Unrelated user artifacts remain in place. No commit or push was performed.

No training, real checkpoint inference, tokenizer/model downloads, remote GPU access or device inference occurred on this PC. Full GPU/DDP learning, fresh Golden scores, environment compatibility and export/device behavior remain unmeasured. LiteRT envelope handling is marked postdecode-only until native cancellation is validated. E2B retained-mobile QAT and MTP remain separate contracts; the capability baseline is not a mobile-ready artifact.
