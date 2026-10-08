# GRPO runtime review — 2026-10-07

Reviewed branch: `new_ir_changes_20260331`.
Review base: `7bda9b0bb225b4558afd41aeb4216012c049ed54`.

## Correctness fixes

### Per-prompt group variance, not between-prompt variance

In TRL 0.29.1, `scale_rewards="batch"` makes the upstream
`frac_reward_zero_std` depend on the global batch standard deviation. Groups
`[-1, -1]` and `[1, 1]` consequently appear diverse even though all four
advantages are zero. The independent gradient/update gates can still catch a
fully dead run; the diversity metric itself was nevertheless misleading.

Both runners now use `build_audited_grpo_trainer`. It observes TRL's already
globally gathered, weighted reward groups, replaces the health-facing
`frac_reward_zero_std` with the true within-group statistic, and records the
original value under `trl/frac_reward_zero_std`. `grpo/group_reward_std` is also
recorded. There are no additional distributed collectives and no changes to
reward values, advantages, reward scaling, or the loss denominator. The
extension deliberately supports the existing `sum_then_normalize` objective.

### Isolated on-policy generation

The shared rollout now builds a fresh generation configuration, rejects
unreviewed filtering/forced-generation controls, and checks that sampling and
likelihood temperatures agree. Checkpoint beam search, token suppression,
forced EOS, and minimum-length settings must not silently affect training.

Passing a fresh config alone is insufficient: Transformers 5.3 fills unset
fields from `model.generation_config`. The rollout therefore temporarily
isolates both the outer model and its PEFT base generation configurations.
Original objects and delegated attribute lookup are restored on success and
failure. No permanent inference defaults are changed.

KV caching is explicitly enabled only for rollout generation, while TRL has
disabled gradient checkpointing. Training cache flags and non-reentrant
checkpointing options are restored afterward. EOS unions, quoted-sentinel
handling, raw sampled token IDs, truncation masks, and loss-masked terminal PAD
semantics are unchanged.

### Standalone sampling and reproducibility

`train_grpo.py` explicitly sets temperature 1, top-p 1, top-k 0, repetition
penalty 1, and dropout disabling, rather than relying on TRL sampling defaults.
The CLI `--seed` is now forwarded into `GRPOConfig`, so trainer initialization
and rank-specific rollout seeding do not silently revert to its default seed.
The QAT runner retains its validated configurable temperature and existing seed.

### Reward and health guardrails

Reward callables must return one finite real number per completion. Short
vectors, `None`, booleans, strings, NaN, and infinity fail clearly, even when
`audit_rollout_limit=0` or the audit budget has already been exhausted. Legitimate
zero and negative rewards are unchanged. The serving-stop and URL-restoration
order is preserved.

Health windows must be positive integers; thresholds must be finite numbers.
Minimum-fraction checks are inclusive: exactly 25% satisfies a 25% minimum.

## Validation performed

From repository root:

```bash
PYTHONPATH=training/src python -m pytest -q training/tests/test_grpo_audit.py
python -m compileall -q training/src training/scripts training/tests
```

Focused regression result in the review environment: **62 passed, 1 skipped**.
The environment had PyTorch 2.10.0 CPU, but no Transformers, TRL, PEFT, datasets,
or CUDA. Tensor statistics are tested with real PyTorch; trainer orchestration,
URL preprocessing, and generation configuration in the unit tests use explicit
mocks. The optional tiny, randomly initialized real-Transformers generation test
was skipped. Compilation covered the locally reviewed files, not a full repo
checkout. The existing rollout test assertions were updated for the explicit
`generation_config` argument; the complete existing test suite was not run.

No GPU training, multi-process DDP run, full reward-scoring pipeline, export, or
held-out quality measurement was executed. These are correctness fixes, not a
claim of measured accuracy or throughput improvement.

## Before a long training run

Run the pinned dependency preflight and the existing complete GRPO/QAT tests.
Then run at least one complete health window on a single GPU, followed by the
same smoke under the intended DDP topology. Inspect corrected group diversity,
clipped-completion fraction, finite gradients, sampled adapter updates, and
checkpoint-specific health provenance. Check saved inference defaults and
export/serving parity after the run.

For quality tuning, compare generation-group sizes and temperature on the
training/validation splits, keeping the final reserved cohorts untouched.
Diagnose whether zero-variance groups are uniformly invalid, uniformly faithful,
or identical samples before changing the reward. Evaluate fidelity to numbers,
URLs, tables, and source content separately from syntax validity. Reward weights,
QAT beta=0, SFT lineage, data split contracts, and quantization/export behavior
were intentionally not changed without that evidence.

## Source review

- [TRL 0.29.1 GRPOTrainer](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/grpo_trainer.py):
  reward gathering, group normalization, sampling kwargs, and telemetry.
- [TRL 0.29.1 model utilities](https://github.com/huggingface/trl/blob/v0.29.1/trl/models/utils.py):
  generation unwrapping and gradient-checkpoint restoration.
- [Transformers 5.3 generation utilities](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/generation/utils.py):
  `_prepare_generation_config` fills unset fields from model defaults.
