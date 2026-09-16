# Bixby50 in the training and deployment pipeline

Bixby50 is the final-only test cohort of captured Bixby/Perplexity responses,
`BXP-001` through `BXP-050`. It now runs alongside **Golden32 and Golden35** in
the current dense E2B / Gemma 3 270M workflows. This does not restore the retired
Golden50: the three cohort names are `golden32`, `golden35` and `bixby50`.

## Bundled inputs and scoring boundary

These files are included in the repository; evaluation does not require the
original local phone logs or screenshots:

- [bixby50.jsonl](../data/eval/bixby50_v1/bixby50.jsonl): frozen source responses,
  IDs, original queries and minimal provenance.
- [benchmark_manifest.json](../data/eval/bixby50_v1/benchmark_manifest.json):
  approved membership, response/query hashes and source-only holdout contract.
- [Artifact README](../data/eval/bixby50_v1/README.md): capture provenance,
  limitations and reproducible extraction instructions.

The response-to-IR model receives each **captured response**, not the original
question, through the same production prompt and tokenizer contract used for
the other cohorts. It generates A2UI Express for scoring.

There is **no reference IR** for Bixby50. Report generated-IR strict validity,
source-grounded v5.4 reward and available runtime/latency metrics. Reference
matching is not applicable: no fabricated IR, empty substitute target, inferred
perfect match or zero reference score is introduced. This tests conversion of
the frozen responses, not factual correctness of their historical claims.

`BXP-038` remains its actual captured refusal. Citation markers without captured
URLs remain unchanged; missing links are not invented. Preparation retains
all benchmark cases or fails with a reason, including over-budget prompts.

All Bixby sources are reserved from training, validation loss and augmentation,
including when final Bixby inference is explicitly deferred. They do not
contribute to checkpoint, hyperparameter or deployment-precision selection.

## Which commands run it?

Existing launch commands need **no new flag** to enable Bixby50:

| Entry point | Bixby50 behavior |
|---|---|
| `run_golden_training.py` | Tests selected-best and actual final checkpoints, after training |
| `run_golden_experiments.py` | No per-trial Bixby inference; tests the already-locked winning best checkpoint once |
| `run_golden_deployment.py` | Tests best/final checkpoints, merged HF, and W32/W16/W8/W4 LiteRT-LM artifacts |

Use the existing commands in the [HF quickstart](GOLDEN_E2E_QUICKSTART.md) or
[full GPU deployment guide](GOLDEN_GPU_DEPLOYMENT.md), with local model/data
paths and a **new output directory**. Both `--profile e2b` and `--profile 270m`
are supported. As before, omitting `--execute` prints a plan and does not train.

HF-only training permits `--no-evaluate-bixby50` to defer its final inference,
but still excludes its sources from training/validation. Full deployment
requires all three cohorts and has no Bixby opt-out.

Tuning selects only by Golden32's unique-source v5.4 reward. Golden35 and
Bixby50 cannot change the locked winner. In full deployment with `--tune`,
both holdouts are deferred until a **fresh full-training run** using the locked
settings; the screening models are not substituted for the final model.

The full deployment scorecard therefore requires **21 verified evaluations**:
seven model/artifact roles times three cohorts. Any incomplete Bixby50 pass
blocks successful completion, just as an incomplete Golden32/Golden35 pass does.

## GPU execution and progress

- Training and periodic Golden32 keep the existing selected-GPU policy.
- Best/final/merged HF tests distribute independent case shards over selected
  CUDA GPUs; run the supervisor once with Python, not torchrun.
- Native LiteRT behavior is unchanged: one warm verified GPU engine per cohort,
  with allowed NVIDIA UUID, tokenizer and native-GPU execution checks. It does
  not claim multi-GPU native inference scaling or fall back to a CPU engine.
- Conversion remains CPU work in its isolated exporter environment.
- Default generation cap remains 2,048 new tokens. Heartbeats, deadlines,
  per-case output and failure preservation remain enabled; more test cohorts
  increase total evaluation work, not the generation budget for each case.

## Results, logs and TensorBoard

For HF-only training, paths are relative to `--output-dir`:

```text
prepared/bixby50.jsonl
evaluations/best_bixby50/attempt_001/
evaluations/final_bixby50/attempt_001/
logs/best_bixby50_001.log
logs/final_bixby50_001.log
evaluation_scorecard.json
pipeline_manifest.json
```

For full deployment, its manifest records the nested training directory and
the exact output paths. Examples under the deployment output are:

```text
evaluations/merged_bixby50/
evaluations/w32_bixby50/
evaluations/w16_bixby50/
evaluations/w8_bixby50/
evaluations/w4_bixby50/
logs/merged_bixby50.log
logs/w4_bixby50.log
deployment_results.md
deployment_scorecard.json
deployment_manifest.json
```

Each completed evaluation retains `predictions.jsonl`, `scored_predictions.jsonl`,
`aggregate_metrics.json` and `evaluation_result.json`. Native passes additionally
retain runner logs/outputs and verified runtime evidence. The final console and
Markdown tables show Bixby50 beside the other cohorts, with missing/failed
results labelled rather than converted into zero scores.

MLP continues to read **`/tensorboard`**. Exact run directories are recorded in
manifests. Important examples:

- Individual evaluation: `evaluation/best_bixby50/generation_reward_v5_4_avg`.
- Deployment comparison under `/tensorboard/deployments/<run-id>/`:
  `evaluation/checkpoint_best/bixby50/generation_reward_v5_4_avg` and
  `evaluation/w4/bixby50/schema_valid_strict_rate`.
- Final comparison HParams: `hparam/w4_bixby50/generation_reward_v5_4_avg`.
- Standalone tuning under `/tensorboard/experiments/<suite-id>/`:
  `selected_holdout/bixby50/generation_reward_v5_4_avg`; results also remain in
  `selected_bixby50/`, `comparison.json`, `comparison.md` and
  `experiments_manifest.json` beneath the experiment output.

Minimal TensorBoard detail stays the default. Complete metrics remain in JSON;
adding Bixby50 does not enable verbose diagnostic charts automatically.

## Upgrading, cache reuse and verification limits

Adding Bixby50 changes cohort membership, exclusion rules and preparation code.
Use a **fresh run output directory after updating**. Keep the same persistent
cache root, but expect one rebuild of affected preparation/token entries;
future exactly matching runs can reuse verified caches. A two-cohort prepared
run is not a valid three-cohort run, even if its training inputs look unchanged.
Do not copy old manifests, rewrite hashes or use `--continue-run`/`--resume-run`
to conceal a changed data contract.

No optimizer restart is automatic. For a compatible new three-cohort deployment
that fails **after** full training and checkpoint tests, the existing explicit
post-training recovery can reuse bound evidence and resume unfinished export/
evaluation stages into new attempt directories. It does not rerun completed
training or erase partial artifacts. See the deployment recovery guide above.

The integration is checked with CPU-only artifact, preparation, orchestration,
reporting and recovery tests. These tests do **not** establish real H100/native
kernel compatibility, throughput, trained-model quality or a measured Bixby50
score. A real model/GPU run still needs to produce and verify those results.
