# Full Golden32/35 training and LiteRT-LM GPU deployment

Entry point: `training/scripts/run_golden_deployment.py`.

This is the current **dense E2B LoRA / Gemma 3 270M full-SFT** workflow with
the repaired archive and current Golden32/35. It adds public W32/W16/W8/W4
export and actual native GPU inference to `run_golden_training.py`. It does not
switch to the older demo Golden32, retained-scale QAT topology, or MTP assistant.

## What runs

```text
GPU / exporter / native runtime prerequisite checks
  -> optional sequential Golden32 hyperparameter screening
  -> lock settings, then fresh full training from the original dense seed
  -> best and final checkpoint tests: Golden32 + Golden35
  -> merge selected-best checkpoint; merged HF tests: Golden32 + Golden35
  -> W32 export + physical-weight audit + GPU tests on both cohorts
  -> W16 export + physical-weight audit + GPU tests on both cohorts
  -> W8  export + physical-weight audit + GPU tests on both cohorts
  -> W4  export + physical-weight audit + GPU tests on both cohorts
  -> evidence-bound JSON/Markdown scorecard + TensorBoard HParams
```

The complete scorecard contains **14 evaluations**: seven model/checkpoint
forms, each on two cohorts. Metrics come from real inference/scoring; no quality
scores are fabricated. Every format must pass export, precision, runtime and
evidence checks. A successful converter exit alone is insufficient. `complete`
means the evaluations executed with complete evidence, **not** that the trained
model achieved a required quality threshold; inspect rewards and validity rates.

Golden32 contains 32 occurrences from **31 unique sources**, including the
user-approved duplicate replacing the invalid archive case. Selection uses
`unique_source_generation_reward_v5_4_avg`. Golden35 contains 35 valid unique
references; its score is `generation_reward_v5_4_avg`. Replaced/omitted source
identities remain reserved from train/validation. Golden35 never selects a
trial, checkpoint or precision. Do not repeatedly tune against its results and
still describe it as an unseen test set.

## GPU and CPU behavior

| Operation | Resource behavior |
|---|---|
| SFT | All selected CUDA GPUs, existing H100-safe profile and backward preflight |
| Periodic Golden32 | All training ranks share cases; resident model/KV cache; synchronous pause |
| Final/merged HF evaluations | One isolated model per selected GPU, disjoint case shards, no DDP collectives |
| Data preparation | CPU affinity/quota-aware workers, verified preparation/token caches |
| Merge and conversion | CPU, isolated exporter environment |
| Native LiteRT inference | One warm GPU engine per cohort, observed GPU evidence, no CPU-engine fallback |

E2B defaults on 80-GB H100s: **microbatch 1 per GPU**, effective batch 32.
Accumulation is 4 on 8 GPUs, 8 on 4 GPUs, 16 on 2 GPUs. 270M defaults to
microbatch 4 on H100. All selected GPUs remain active. A larger E2B microbatch
is not assumed safe merely because the card has 80 GB. Maximum-length backward
preflight, gradient checkpointing and cuDNN-SDPA avoidance remain enabled.

`--devices auto` respects the scheduler CUDA allocation. Explicit devices refer
to visible logical indices or supported UUIDs. Launch standalone HF evaluation
**once with Python, not torchrun**; it supervises its own workers. Inference is
batch-one per GPU, not continuous batching; the slowest shard sets completion.

**Native LiteRT limitation:** the pinned official Python GPU API has no GPU
index/UUID selector. Multiple engines could all choose GPU 0, so the built-in
runner does not claim 8-GPU native scaling. CUDA masks do not prove OpenCL/WebGPU
isolation. The observed native GPU must be among the allowed training GPU UUIDs;
wrong-device or missing evidence fails. This proves selection/allocation, not
every operator's placement or optimal utilization. See the
[native runner contract](LITERTLM_GPU_RUNNER.md).

CPU tests cannot guarantee failure-free H100 execution. The host needs compatible
drivers/delegate libraries, model kernels, RAM and disk. Prerequisite probes run
before training, but cannot certify not-yet-exported model kernels. Unsupported
formats fail explicitly; they are never replaced or silently skipped.

## Setup and E2B command

Use the working CUDA training environment. Create **separate** export/runtime
environments with [the pinned setup commands](DEPLOYMENT_EXPORT_ENVIRONMENT.md).
Do not replace training dependencies with converter dependencies. Provide local
dense model/tokenizer files; weights are never downloaded automatically.

E2B needs a compatible full `gemma4` HF seed. Standalone `gemma4_text` is rejected
unless the exporter proves support for its per-layer embedding graph. Do not
relabel configs to bypass this check. E2B W4 means mixed W4/W8. W32/W16 mean
weight storage precision, not 32-billion/16-billion-parameter models. W16/W4 are
experimental and require `--allow-experimental-formats`; actual weights are audited.

The full repaired v9 archive is external data, **not included by git clone**.
Copy its `train.jsonl` and `val.jsonl` to the host and pass that directory.
Both Golden sets are checked in and automatically integrated. Omitting
`--input-dir` uses the existing smaller checked-in Stage3 source, not full v9.
Original inputs remain untouched. Reserve RAM/disk for dense merge, W32/W16,
converter temporary graphs and separate HPO checkpoints; GPU VRAM is not CPU RAM.

Run from the cloned repo root in the CUDA training environment. Replace paths
and choose a **fresh output folder**:

```bash
python -u training/scripts/run_golden_deployment.py \
  --profile e2b \
  --model-dir /ABSOLUTE/PATH/TO/DENSE_E2B \
  --input-dir /ABSOLUTE/PATH/TO/full_data_archive_recovered_v9 \
  --output-dir /group-volume/k.anup/working_dir/e2b_full_deployment_001 \
  --exporter-python /group-volume/k.anup/envs/a2ui-export-094/bin/python \
  --runtime-python /group-volume/k.anup/envs/a2ui-litert-017/bin/python \
  --devices auto --epochs 1 \
  --tensorboard-root /tensorboard \
  --allow-experimental-formats \
  --execute
```

Omit `--execute` for a read-only plan: paths/options are checked, but models and
GPUs are not loaded or certified. Pass absolute **venv invocation paths**; do not
resolve Python symlinks with `readlink -f`. Use `--profile 270m`, its dense model,
and a new output directory for Gemma 3 270M. This dense workflow does not accept
`--qat`; retained-scale/QAT pipelines remain separate. Append `--steps 20` for a
bounded training smoke followed by all export/tests; that is not a quality run.

## Tuning and augmentation

Append `--tune --trial-steps 1000` to the full command. The default screen runs
four equal-step trials sequentially: baseline, half LR, double LR, stronger
regularization. It locks the Golden32 winner, then starts **fresh full training**
with chosen settings and the requested epochs (or explicit final `--steps`).
Short screening weights are not silently used as the full-epoch model. Unlike
the standalone experiment launcher, this full workflow defers all screening
Golden35 inference until the fresh full run is trained.

`--include-augmentation` adds a capped rare-component resampling trial. Without
tuning, `--augmentation rare_components` enables resampling directly. Only
valid training examples are repeated; no invented content, rewritten targets,
or changed Golden/validation examples. Custom `--trials-file` follows the
[tuning contract](AUGMENTATION_AND_TUNING.md). Short-run selection is not proof
of globally optimal settings or full-epoch quality.

## Defaults, progress and caches

- Validation loss/checkpoint save: every **500 optimizer steps**.
- Periodic Golden32: every **1,000 optimizer steps**, plus final; all ranks.
- Golden35: post-training only, never periodic selection.
- Sequence/prompt budget: **4,096**; generation cap: **2,048 new tokens**.
- Export cache: **8,192**, covering prompt plus generation.
- Training metrics: every **10 optimizer steps**; stage heartbeat: **10 seconds**.
- Subprocess hard limit: **48 hours**, `--stage-timeout-seconds`.
- HF/native generation worker per cohort: **7,200 seconds**, `--generation-timeout-seconds`.
- Native model load: **1,800 seconds**, `--load-timeout-seconds`.
- Native case: **600 seconds**, `--case-timeout-seconds`.

These are configurable total deadlines, not idle-output rules. Owned workers
are stopped on timeout/failure; partial logs remain and incomplete cohorts never
receive a passing score. Periodic training evaluation is synchronous: training
pauses while all ranks generate and rank zero scores/saves the checkpoint.

Caches default to `<output parent>/.golden-preparation-cache` and its `tokens/`
child. Reuse the same paths on fast persistent storage. Content/tokenizer/code
changes invalidate reuse. Evaluation also caches the expensive source-overlap
scan under `<training-output>/fit/evaluation_validation_cache`, while **still
rehashing current train/val and model bytes**. Progress/hash messages are expected,
not a complete filtering rerun. Never delete a live lock to force reuse.

## Results and TensorBoard

Exact paths/run IDs are printed and recorded in `deployment_manifest.json`.
Training is under `<output>/<run-id>_training`; tuning is under
`<output>/<run-id>_tuning`.

```text
<output>/deployment_manifest.json       statuses, timings, bindings, settings
<output>/deployment_scorecard.json      complete evidence-bound 14-result scorecard
<output>/deployment_results.md          readable model/cohort comparison
<output>/logs/                          export, runtime, merged/variant logs
<output>/deployment/merged_hf/          selected-best dense checkpoint
<output>/deployment/variants/w32/       model.litertlm, inspection, export manifest
<output>/deployment/variants/w16/
<output>/deployment/variants/w8/
<output>/deployment/variants/w4/
<output>/evaluations/w4_golden35/        example native predictions/scores/evidence
<training-output>/logs/                 training, preflight, best/final tests
<training-output>/fit/golden_eval/      periodic Golden32 results
```

Each evaluation has predictions, scored predictions, aggregate metrics and
`evaluation_result.json`. HF partials are in
`predictions_workers/rank_NNN/predictions.jsonl.partial`. Native output is flushed
per case to `runner_outputs.jsonl`; `runner.log` retains native logs. Its manifest
records package hash, runtime version, token parity and actual GPU UUIDs.

MLP should read **`/tensorboard`**. Full comparison tags are in
`/tensorboard/deployments/<run-id>/`, including all Golden scores, stage timings
and `final_comparison` HParams. Training, individual evaluations and sequential
trial HParams also live under the same root. Outside MLP, use
`tensorboard --logdir /tensorboard`. Compare matching current prompt contracts,
not historical short-prompt scores.

## Failure/recovery and verification limits

The full launcher needs a fresh output directory. It never automatically
restarts failed optimizers, retries holdouts or overwrites artifacts. Inspect
the manifest's active stage and named log. Completed data/checkpoints/exports
remain. Resolve environment/driver/resource failures first. The plan retains
individual export/evaluation commands for deliberate recovery using retained
checkpoints and fresh destinations. The original training launcher keeps its
separate verified `--continue-run` behavior.

See [local validation evidence](../reports/golden_deployment_20260914/README.md)
for executed tests and unverified real-H100/native-model steps. No actual model
quality or full native GPU run is claimed from CPU/mock tests.
