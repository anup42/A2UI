# Full Golden32/Golden35/Bixby50 training and LiteRT-LM GPU deployment

Entry point: `training/scripts/run_golden_deployment.py`.

**Already trained and only need the four model files?** Use the separate
[checkpoint-only export command](EXPORT_TRAINED_CHECKPOINT.md). It does not
retrain or run HF/native evaluation and does not need Vulkan. The
`--skip-litert-evaluation` mode below still includes training and HF tests.

This is the current **dense E2B LoRA / Gemma 3 270M full-SFT** workflow with
the repaired archive and current Golden32/Golden35/Bixby50. It adds public W32/W16/W8/W4
export and actual native GPU inference to `run_golden_training.py`. It does not
switch to the older demo Golden32, retained-scale QAT topology, or MTP assistant.

## What runs

```text
GPU / exporter / native runtime prerequisite checks
  -> optional sequential Golden32 hyperparameter screening
  -> lock settings, then fresh full training from the original dense seed
  -> best and final checkpoint tests: Golden32 + Golden35 + Bixby50
  -> merge selected-best checkpoint; merged HF tests: all three cohorts
  -> W32 export + physical-weight audit + GPU tests on all three cohorts
  -> W16 export + physical-weight audit + GPU tests on all three cohorts
  -> W8  export + physical-weight audit + GPU tests on all three cohorts
  -> W4  export + physical-weight audit + GPU tests on all three cohorts
  -> evidence-bound JSON/Markdown scorecard + TensorBoard HParams
```

The complete scorecard contains **21 evaluations**: seven model/checkpoint
forms, each on three cohorts. Metrics come from real inference/scoring; no quality
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

Bixby50 contains 50 captured Bixby/Perplexity responses and **no reference IR**.
It is another final-only holdout, never used for tuning/checkpoint/precision
selection. Its generated IR is scored for source-grounded reward, validity and
runtime; reference-match metrics are not applicable. No placeholder or synthetic
IR target is substituted. The current cohorts are Golden32, **Golden35** and
Bixby50, not the retired Golden50. See [Bixby50 details](bixby50_evaluation.md).

### Export without native LiteRT evaluation

Append **`--skip-litert-evaluation`** when Vulkan/native LiteRT inference is
unavailable. This changes only the deployment-testing mode:

```text
GPU + exporter prerequisite checks (no native runtime/Vulkan preflight)
  -> optional sequential Golden32 tuning -> fresh full training
  -> best + final + merged HF tests on Golden32, Golden35 and Bixby50
  -> W32 + W16 + W8 + W4 export, with the same physical-weight/artifact audits
  -> final report: 9 measured HF evaluations; 12 native evaluations skipped
```

Training and HF tests still use all selected GPUs. All four conversion formats,
CPU exporter checks, tokenizer/template contracts and precision audits remain
required. Only native LiteRT generation/scoring and its runtime/Vulkan
requirements are omitted; `--runtime-python` is optional in this mode. The
exporter environment is still required. This does **not** fall back to CPU
inference or claim native compatibility: the exported variants remain untested
on the target runtime. Their scorecard/table rows are explicitly skipped, with
no quality or latency scores invented and no corresponding TensorBoard scores.

Without this flag, the existing complete **21-evaluation** GPU workflow remains
unchanged, including the required `--runtime-python` and native preflight.
The flag also works with `--tune` and the existing augmentation options.

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

CPU tests cannot guarantee failure-free H100 execution. The default full-testing
host needs compatible
drivers/delegate libraries, model kernels, RAM and disk. Prerequisite probes run
before training, but cannot certify not-yet-exported model kernels. Unsupported
formats fail explicitly; they are never replaced or silently skipped.

## Setup and E2B command

Use the working CUDA training environment. Create **separate** export/runtime
environments with [the pinned setup commands](DEPLOYMENT_EXPORT_ENVIRONMENT.md).
For `--skip-litert-evaluation`, create only the training and exporter environments;
the native runtime environment and Vulkan setup are not prerequisites.
Do not replace training dependencies with converter dependencies. Provide local
dense model/tokenizer files; weights are never downloaded automatically.

E2B needs a compatible full `gemma4` HF seed. Standalone `gemma4_text` is rejected
unless the exporter proves support for its per-layer embedding graph. Do not
relabel configs to bypass this check. E2B W4 means mixed W4/W8. W32/W16 mean
weight storage precision, not 32-billion/16-billion-parameter models. W16/W4 are
experimental and require `--allow-experimental-formats`; actual weights are audited.

The full repaired v9 archive is external data, **not included by git clone**.
Copy its `train.jsonl` and `val.jsonl` to the host and pass that directory.
All three evaluation cohorts are checked in and automatically integrated. Omitting
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

To train, test all three cohorts with HF checkpoints and create all four variants
without running LiteRT/Vulkan, use this command instead (runtime Python omitted):

```bash
python -u training/scripts/run_golden_deployment.py \
  --profile e2b \
  --model-dir /ABSOLUTE/PATH/TO/DENSE_E2B \
  --input-dir /ABSOLUTE/PATH/TO/full_data_archive_recovered_v9 \
  --output-dir /group-volume/k.anup/working_dir/e2b_export_no_native_eval_001 \
  --exporter-python /group-volume/k.anup/envs/a2ui-export-094/bin/python \
  --devices auto --epochs 1 --tensorboard-root /tensorboard \
  --allow-experimental-formats --skip-litert-evaluation --execute
```

Use the same flag with `--profile 270m` and its local dense seed for Gemma 3 270M.
This remains the existing LiteRT-LM export format (`model.litertlm`, packaging
the LiteRT/TFLite model); it does not switch the artifact type or quantization
recipe. Review the export manifests and precision audits before device use.

## Tuning and augmentation

Append `--tune --trial-steps 1000` to the full command. The default screen runs
four equal-step trials sequentially: baseline, half LR, double LR, stronger
regularization. It locks the Golden32 winner, then starts **fresh full training**
with chosen settings and the requested epochs (or explicit final `--steps`).
Short screening weights are not silently used as the full-epoch model. Unlike
the standalone experiment launcher, this full workflow defers all screening
Golden35 and Bixby50 inference until the fresh full run is trained. Standalone
`run_golden_experiments.py` instead evaluates both holdouts once for the already
locked winning checkpoint, never for each trial.

`--include-augmentation` adds a capped rare-component resampling trial. Without
tuning, `--augmentation rare_components` enables resampling directly. Only
valid training examples are repeated; no invented content, rewritten targets,
or changed Golden/validation examples. Custom `--trials-file` follows the
[tuning contract](AUGMENTATION_AND_TUNING.md). Short-run selection is not proof
of globally optimal settings or full-epoch quality.

## Defaults, progress and caches

- TensorBoard detail: **minimal**, keeping headline quality/validity, loss,
  learning rate, gradient norm and useful runtime/HParams comparisons. Use
  `--tensorboard-detail full` only when debugging. Complete metrics remain in
  JSON/console logs. This does not remove event files from older runs; filter
  TensorBoard to the current run to avoid seeing old verbose charts.
- Validation loss/checkpoint save: every **500 optimizer steps**.
- Periodic Golden32: every **1,000 optimizer steps**, plus final; all ranks.
- Golden35 and Bixby50: post-training only, never periodic selection.
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

Adding Bixby50 changes the frozen cohort inventory, source exclusions and
preparation identity: the first run after this upgrade rebuilds affected caches.
Retain the same cache root for later matching runs, but use a **fresh output
directory**. Do not attach Bixby50 to an old two-cohort manifest with
`--resume-run`, copy old prepared files, or edit manifest hashes to force reuse.

## Results and TensorBoard

Exact paths/run IDs are printed and recorded in `deployment_manifest.json`.
Training is under `<output>/<run-id>_training`; tuning is under
`<output>/<run-id>_tuning`.

The console prints a final table of **all tuning trials** (if `--tune` was
requested), and **all 21 checkpoint/merged/LiteRT test results**. It includes
row counts, v5.4 reward, strict-valid percentages, and a separate Golden32
31-unique-source selection score. On failure it prints the verified results
already available, marks failed/missing slots explicitly, and does not invent
zero scores. The same tables are saved in `deployment_results.md` and the tuning
folder's `comparison.md`. Without `--tune`, no tuning trials are run. With
`--skip-litert-evaluation`, the same comparison inventory shows nine measured
HF results and twelve explicitly skipped native results; successful export is
not a measured LiteRT score. Native prediction/log directories are not produced
for intentionally skipped evaluations.

```text
<output>/deployment_manifest.json       statuses, timings, bindings, settings
<output>/deployment_scorecard.json      complete evidence-bound 21-result scorecard
<output>/deployment_results.md          readable model/cohort comparison
<output>/logs/                          export, runtime, merged/variant logs
<output>/deployment/merged_hf/          selected-best dense checkpoint
<output>/deployment/variants/w32/       model.litertlm, inspection, export manifest
<output>/deployment/variants/w16/
<output>/deployment/variants/w8/
<output>/deployment/variants/w4/
<output>/evaluations/w4_golden35/        example native predictions/scores/evidence
<output>/evaluations/w4_bixby50/         Bixby50 native predictions/scores/evidence
<training-output>/logs/                 training, preflight, best/final tests
<training-output>/fit/golden_eval/      periodic Golden32 results
```

Each evaluation has predictions, scored predictions, aggregate metrics and
`evaluation_result.json`. HF partials are in
`predictions_workers/rank_NNN/predictions.jsonl.partial`. Native output is flushed
per case to `runner_outputs.jsonl`; `runner.log` retains native logs. Its manifest
records package hash, runtime version, token parity and actual GPU UUIDs.

MLP should read **`/tensorboard`**. Full comparison tags are in
`/tensorboard/deployments/<run-id>/`, including headline Golden/Bixby scores
and `final_comparison` HParams. Training, individual evaluations and sequential
trial HParams also live under the same root. Outside MLP, use
`tensorboard --logdir /tensorboard`. Compare matching current prompt contracts,
not historical short-prompt scores.

Bixby50 deployment tags follow the same minimal-metric policy, for example
`evaluation/checkpoint_best/bixby50/generation_reward_v5_4_avg` and
`evaluation/w4/bixby50/schema_valid_strict_rate`. Final HParams include
`hparam/w4_bixby50/generation_reward_v5_4_avg`; reference-match scores are not
reported for this source-only cohort. Stage timings are logged only in full
TensorBoard detail mode. Full JSON artifacts are retained in either mode.

## Failure/recovery and verification limits

New runs require a fresh output directory. It never automatically restarts failed
optimizers, retries holdouts or overwrites model artifacts. Inspect the
manifest's active stage and named log. Completed data/checkpoints/exports remain.

For the `libvulkan.so.1` / WebGPU `No adapters found` failure, first fix the
[runtime image and Vulkan driver exposure](DEPLOYMENT_EXPORT_ENVIRONMENT.md#linux-vulkan-prerequisites-required-for-native-gpu-testing).
The enhanced `run_litertlm_gpu.py --preflight` checks an actual hardware NVIDIA
Vulkan compute device in an isolated 30-second probe. CUDA and `nvidia-smi`
success alone are insufficient. No package install or container restart is
performed automatically, and the probe does not certify future model kernels.
Alternatively, start a **new output directory** with `--skip-litert-evaluation`
to complete training/HF tests and audited exports while deferring native tests.
Do not add it to a failed full-testing run and attempt to resume that run.

After that probe passes, **reuse your exact original full deployment command**
with the same `--output-dir`, adding:

```bash
--resume-run --tensorboard-detail minimal
```

This is explicitly **post-training recovery only**. It requires completed full
training and all best/final checkpoint evaluations. It rechecks retained file
hashes, merged weights, export precision and result evidence; keeps the saved
hyperparameters/prompt/Golden contracts; reruns host/exporter/Vulkan preflights;
and skips verified completed stages. It never retrains or reruns completed
Golden evaluations. A failed/unfinished export or evaluation uses a fresh
`recovery/attempt_NNNN/` destination, preserving prior partial results/logs.
Remaining commands and paths are recorded in the manifest. A kernel lock rejects
concurrent recovery on the same run. Keep original model/data paths accessible.
An already-complete run is verified and its table printed without rerunning it.

Do not use `--resume-run` to change model, data, token budgets, augmentation,
training settings or tuned trial definitions. Only logging/deadlines may change.
The native-evaluation mode is also bound to the run: a resume must retain the
original `--skip-litert-evaluation` choice. It cannot convert a full-testing run
into export-only mode, or promote an export-only run into native-tested status.
Changing modes requires a fresh output directory. In export-only mode, compatible
post-training recovery rechecks exporter/GPU evidence but does not run a Vulkan
or native-runtime preflight.
If training/tuning itself was incomplete, this option refuses to start it; the
original training launcher retains its separate verified `--continue-run`
behavior. After an intentional code/prompt/schema change, a contract mismatch
must be investigated instead of relabeling old evaluations as comparable.

### Prepared shared-prompt snapshots

`Shared prompt contract is stale or changed` previously compared a prepared
snapshot with a prompt rebuilt from the **live checkout** each time training or
evaluation launched. An unchanged, fully bound preparation could therefore
stop after an unrelated checkout update changed the production prompt/builder.
The screenshot alone does not identify which field changed on the remote host.

Existing prepared runs now validate the saved contract's structure and internal
hashes, plus `shared_prompt.json`, `prompt_scaffolds.json`, `inference_prompt.json`
and their manifest bindings. Training and all three cohorts must agree on the
same saved prompt. Runtime never replaces prepared messages with the current
production prompt. Small prompt checks happen before the full corpus scan.
Tampered artifacts and mixed prompt snapshots still stop the run.

**Fresh preparation still requires the current production prompt.** Cache
identity and restored-manifest checks bind it to that run's planned prompt.
The error reports changed fields and saved/current contract hashes when a new
plan is stale. Do not edit manifest hashes, disable these checks, or compare
scores across prompt revisions as if they used the same input.

Do not `git pull` into a running training/tuning checkout. A tuning suite also
pins its code, recipes, source data and tokenizer across trials; those guards
remain. For the already-failed **tuning** run, retain its files and start the
same command with a **new output directory** after updating. Existing compatible
preparation/token caches can be reused; changed preprocessing identity rebuilds
once. `--resume-run` is not a tuning/optimizer restart mechanism. This change
does not automatically resume failed trials or erase completed checkpoints.

See [local validation evidence](../reports/golden_deployment_20260914/README.md)
for executed tests and unverified real-H100/native-model steps. No actual model
quality or full native GPU run is claimed from CPU/mock tests.
