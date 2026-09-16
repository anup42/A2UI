# E2B / 270M training with Golden32, Golden35 and Bixby50

For the extended **train/tune → W32/W16/W8/W4 LiteRT-LM → Golden32/Golden35/Bixby50 GPU tests**
workflow, see [the full deployment runbook](GOLDEN_GPU_DEPLOYMENT.md) and
`run_golden_deployment.py`. The command below remains the HF-only workflow.

This is the current clone-and-run entry point for **dense E2B LoRA capability
training and Gemma 3 270M full-model training**, followed by checkpoint testing
on all three test cohorts. It aligns training and evaluation to one versioned
production A2UI Express prompt. It does not train here, download model weights,
or establish that a GPU run has passed.

The command performs data preparation, source-leakage filtering, tokenizer
checks, automatic GPU configuration, real-model forward and CUDA backward preflights, training,
and final checkpoint evaluation. Use a short smoke run before a full run.

## 1. Clone and prepare the GPU host

Use Linux and the scheduler/container environment that exposes the intended
H100 GPUs. Run commands from the repository root:

```bash
git clone --branch new_ir_changes_20260331 https://github.com/anup42/A2UI.git
cd A2UI

python3 -m venv .venv-a2ui-train
source .venv-a2ui-train/bin/activate
python -m pip install --upgrade pip
# Install the CUDA-enabled PyTorch build appropriate to this GPU host first.
python -m pip install -r training/requirements-gemma4-qat.txt

python -c "import torch,transformers,peft; print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.device_count(),transformers.__version__,peft.__version__)"
nvidia-smi
export A2UI_TENSORBOARD_ROOT=/tensorboard
```

The Gemma 4 dependency overlay includes the newer architecture APIs needed by
E2B. The package requirements are not a tested lock for every CUDA host. Retain
the actual environment versions after a successful smoke. `/tensorboard` must
exist or be creatable and writable inside the training container; MLP must see
that same mounted location.

A normal clone is sufficient for this workflow. The unrelated `workspace`
gitlink is not a training dependency; do not require recursive submodule setup.

Provide a **local, complete model bundle** containing its dense HF
`config.json`, safetensors weights, tokenizer files and chat template. The
repository review profiles name `google/gemma-4-E2B-it-qat-q4_0-unquantized`
for E2B and `google/gemma-3-270m-it` for 270M. E2B requires that dense
QAT-derived bundle, not a packed mobile checkpoint. Model access, license
acceptance and obtaining those bundles are prerequisites; the launcher never
downloads weights or silently substitutes a model.

The following inputs are included in Git:

- Completed Stage 3 source `dataset/data/runs/dataset_v1/genui.jsonl` and its
  Stage 2 `responses.jsonl`. This is the default training source.
- `training/data/eval/golden32_archive_repeat_v1/golden32.jsonl` and its manifest.
- `training/data/eval/golden35_v1/golden35.jsonl` and its manifest.
- `training/data/eval/bixby50_v1/bixby50.jsonl` and its manifest: source-only
  holdout, with no reference IR. See [Bixby50 evaluation](bixby50_evaluation.md).
- Preparation, filtering, trainer, evaluation and prompt/schema code.

Prepared/tokenizer-bound splits, model weights, training checkpoints and logs
are generated on the GPU host and are not clone prerequisites.

## 2. Run a smoke, then start training

Replace the model and output paths. Every fresh run must use a **new output
directory**; verified continuation is an explicit exception described below.

```bash
python training/scripts/run_golden_training.py \
  --profile e2b --model-dir /models/e2b \
  --output-dir /runs/e2b-smoke --steps 20 --execute
```

The smoke still performs final tests on all three complete cohorts; `--steps 20`
limits optimizer updates, not benchmark membership. Require finite loss and
gradients, real parameter updates, no OOM/distributed failure, and complete
post-training reports. Twenty updates test wiring, not final model quality.

After the smoke passes:

```bash
python training/scripts/run_golden_training.py \
  --profile e2b --model-dir /models/e2b \
  --output-dir /runs/e2b-epoch1 --epochs 1 --execute
```

For Gemma 3 270M, use the same workflow with its own model and output directory:

```bash
python training/scripts/run_golden_training.py \
  --profile 270m --model-dir /models/gemma-3-270m-it \
  --output-dir /runs/gemma270m-epoch1 --epochs 1 --execute
```

E2B uses the review profile's rank-32 LoRA on language attention/MLP projections.
270M defaults to full-model SFT. These are documented starting recipes, not
measured throughput or quality optima.

For optional train-only rare-component resampling, matched-budget sequential
hyperparameter experiments, and TensorBoard comparisons, see
[Augmentation and tuning](AUGMENTATION_AND_TUNING.md). Augmentation is off by
default. The experiment runner defers Golden35 and Bixby50 until the winner is
locked; ordinary single runs test all three cohorts by default. The full
deployment's tuning screen defers both holdouts until its fresh full run.

Execution modes:

- Omit `--execute` to print a plan only: no new files, tokenizer/model load or
  training.
- Use `--prepare-only` instead of `--execute` to materialize/filter inputs and
  prepare examples with the local tokenizer, without model-weight loading or
  training. Continue that prepared run explicitly with `--continue-run` and
  `--execute`, using the same model, data and recipe arguments.
- Use `--execute` for the complete workflow, including model preflight and
  training. A failed stage stops the workflow; later scores are not fabricated.
- HF-only training supports `--no-evaluate-bixby50` to defer Bixby inference;
  its sources remain excluded from train/validation. Full deployment requires
  all three cohorts and does not expose this opt-out.

Use `python training/scripts/run_golden_training.py --help` for supported
overrides. Fresh runs refuse an existing output directory. `--continue-run`
can continue a verified prepared run or retry final evaluation after verified
completed training; it must not restart failed/interrupted training as a new
optimizer run. Preserve failed-run evidence and use the lower-level checked
resume tooling only after diagnosing a training failure and retaining the
original data/recipe contract. No continuation silently overwrites checkpoints.

## 3. Evaluation defaults and outputs

| Setting | Default |
|---|---|
| Epochs | 1 |
| Training / periodic Golden32 GPUs | All scheduler-visible GPUs; H100 2/4/8 profiles selected automatically |
| E2B H100 microbatch / effective batch | 1 per GPU / 32 globally; accumulation 16, 8, 4 for 2, 4, 8 GPUs |
| Validation loss and checkpoint saves | Every 500 optimizer updates |
| Golden32 generation/selection | Every 1,000 optimizer updates, plus final weights |
| Golden35 | Final evaluation only; never used to select a checkpoint |
| Bixby50 | Source-only final evaluation; never used to select a checkpoint |
| Final checkpoint testing | Selected best and actual final weights, each on Golden32, Golden35 and Bixby50 |
| New generation tokens | 2,048 |
| Training full-sequence / evaluation prompt budget | 4,096 tokens with the exact local tokenizer |
| TensorBoard | `/tensorboard/<run-id>/` |

Context overrides must keep `--max-input-tokens` equal to `--max-seq-length`.
The prompt budget plus `--max-new-tokens` must not exceed the recipe context:
8,192 for E2B or 32,768 for 270M. The default 4,096 + 2,048 fits both. Change
both input/sequence options together; increasing only one fails validation.

Golden32's headline selection metric is
`unique_source_generation_reward_v5_4_avg`: its 32 occurrences contain **31
unique sources**, including the requested repeated donor. The ordinary
32-occurrence score is separate. Golden35 has **35 unique strict-valid
references**. Bixby50 has 50 unique captured responses, **without reference IR**:
source-grounded reward, generated-output validity and runtime apply, but
reference-match scores do not. The three sets are scored separately, never as a
mixed average. The retired Golden50 is not restored by this addition.
The original September 3 demo Golden32 used by the separate official
export pipeline is a different cohort.

After training, the workflow tests both the checkpoint selected using Golden32
and the actual final saved checkpoint on **all three** sets. Even if selected-best
and final correspond to the same training step, their artifact roles remain
explicit. Per-case predictions, aggregate metrics, checkpoint identities,
stage logs/status and a combined scorecard are retained in the output run.
TensorBoard receives training, periodic Golden32 and final Golden32/Golden35/Bixby50 metrics.
No score is called complete unless the expected row count is present.

The six post-training cohort/checkpoint evaluations run sequentially; each HF
evaluation shards cases across all selected GPUs using independent model workers.
Launch once with Python, not torchrun: the evaluation supervisor owns its workers.
The full LiteRT deployment keeps its existing one-verified-native-GPU behavior.

Output paths below are relative to the chosen output directory:

| Artifact | Location |
|---|---|
| Stage status, provenance and output inventory | `pipeline_manifest.json` |
| Tokenizer-bound data and prompt contracts | `prepared/` |
| Resolved training configuration | `fit/training_config.yaml` |
| Golden32-selected checkpoint | `fit/training/best_golden_checkpoint/` |
| Actual final weights | `fit/training/final_adapter/` for E2B; `final_model/` for 270M |
| Per-case and aggregate final tests | `evaluations/{best,final}_{golden32,golden35,bixby50}/attempt_001/` |
| Combined final comparison | `evaluation_scorecard.json` |
| Stage command logs | `logs/` |

The `prepared` directory includes `train.jsonl`, `val.jsonl`, `golden32.jsonl`,
`golden35.jsonl`, `bixby50.jsonl`, `manifest.json`, saved shared/inference prompt contracts, and
source prompt-scaffold evidence. Retried evaluations use distinct attempts;
do not treat stale output from a failed attempt as a completed scorecard.

All cadences count **optimizer updates**, not microbatches. The checked-in
`dataset_v1` currently has 4,870 Stage 3 records before filtering; one epoch at
global batch 32 may finish before step 500. In that case, final Golden32 and
Golden35/Bixby50 evaluation still run, but there may be no periodic Golden pass. For a
shorter cadence on a small dataset, explicitly use:

```bash
python training/scripts/run_golden_training.py \
  --profile e2b --model-dir /models/e2b \
  --output-dir /runs/e2b-frequent-eval --epochs 1 \
  --eval-steps 100 --golden-every-steps 100 --execute
```

`--golden-every-steps` must be a positive multiple of `--eval-steps`. Evaluating
more often costs training time. Periodic generation is distributed across
training ranks with a scoped KV cache; it is synchronous and pauses optimizer
work. No zero-overhead concurrent inference or measured H100 speedup is
promised. See [GPU profile details](gpu_training_profiles.md).

```bash
tensorboard --logdir /tensorboard
```

## 4. One prompt, frozen references, clean training data

The launcher materializes the completed default source through the current
strict Express/wire/reachability checks, groups source cases into deterministic
90/10 train/validation splits, and filters reserved Golden source identities
and normalized source responses from both splits, including Bixby50. It also reserves the failed
original Golden32 source and all 15 sources excluded from Golden35. Those
unscored sources do not become training examples.

The full system/few-shot/task scaffold is rebuilt from the versioned shared
production prompt for **training, Golden32, Golden35 and Bixby50** before tokenizer-aware
preparation. The prepared manifest binds prompt, tokenizer vocabulary, chat
template/kwargs, split bytes and benchmark membership. Different input response
text is expected; a different instruction scaffold is not.

The checked-in Golden files remain unchanged. Prompt alignment is a prepared
view with new provenance/hashes, not a rewrite of Golden source responses,
reference outputs, IDs or membership. Deploy/evaluate with the saved full
scaffold and matching chat template; do not substitute an abbreviated prompt.
Historical scores produced with the old archive prompt are not directly
comparable to scores from this aligned setup.

Bixby50 preparation uses an inference-only path. It retains source response and
identity, applies the same shared prompt/token budget, and does not manufacture
an assistant completion or reference target. It is never a loss-validation or
augmentation split. Missing or over-budget Bixby cases fail the whole cohort;
the runner does not silently reduce it to a passing subset.

Strict-invalid or overlong training rows are quarantined whole, with reasons.
No target is silently truncated, manually repaired or synthesized. If even one
fixed Golden occurrence fails strict validation or its **inference prompt**
exceeds the token budget, the workflow
stops instead of reporting a partial benchmark. Inspect corpus retention,
component/intent distribution and token lengths before judging data sufficiency.
The default source is usable input, not a claim that all its legacy records are
clean or that one epoch guarantees high scores.

Golden targets remain complete references: they are not filtered by the
training full-sequence limit or shortened to fit the 2,048-token generation
budget. Report long-reference and generation-limit cases as limitations of the
chosen context/decoder policy, not as missing benchmark cases.

To use another completed Stage 3 run:

```bash
python training/scripts/run_golden_training.py \
  --profile e2b --model-dir /models/e2b \
  --source-run-dir /data/completed-stage3-run \
  --output-dir /runs/e2b-other-source --execute
```

To use authoritative existing `train.jsonl` and `val.jsonl` files, pass
`--input-dir /data/source-bound-train-val` instead. Source/query identities and
canonical source responses must be recoverable; slim review samples lacking
identity/provenance are not suitable production inputs. Existing source splits
are still filtered, prompt-aligned, token-checked and checked for leakage.
Training itself never invokes Stage 1/2/3 generation or a cloud teacher.

The external messages-only archive has a separate offline recovery and
refinement path. Its completed v9 copy includes paragraph, letter, and join-boundary
completeness checks, stronger Golden-source filtering, and rebuilt
source-grouped validation. The
large candidate is ignored by Git, so copy it to the GPU host or rebuild it
from the exact frozen source hashes by following the
[messages-archive final-review runbook](messages_archive_final_review.md). The
[v9 report](../reports/offline_recovery_20260913_v9/REPORT.md) records final counts,
repairs, verification, and remaining limitations. Run the exact-model tokenizer
preflight before training. Do not feed the original UTF-16 messages files
directly to this launcher.

## Startup progress, CPU preparation, and reuse

`run_golden_training.py` now prints preparation progress and streams every
subprocess's stdout/stderr to the console while retaining the same output in
`<output-dir>/logs/`. This includes carriage-return progress displays from
tokenizers, model loading, and training. The files are `prepare.log`,
`configure.log`, `preflight.log`, `training.log`, and per-evaluation logs.
Blocking stages emit a heartbeat every 10 seconds, even before the first
training update. Row phases show completed/total rows, throughput, elapsed
time, and ETA. Hash/copy phases show byte progress. TensorBoard training and
Golden metrics still use `/tensorboard/<run-id>/`; console progress does not
replace those metrics or manufacture scores during preparation.

Preparation is CPU work, not H100 training. Existing `--input-dir` splits are
streamed from disk; strict filtering and target preparation use spawned CPU
workers with bounded batches and deterministic source ordering. The default
`--prepare-workers 0` selects up to 16 workers using CPU affinity, common Linux
container CPU quotas, and available memory. Use `--prepare-workers 1` for the
serial reference path, or specify a measured host-appropriate count. These
workers are separate from `--dataloader-workers` and `--devices`. The actual
tokenizer runs in the parent while CPU workers prepare targets ahead of it;
no worker loads model weights or initializes a training process.

Useful optional flags:

```bash
--prepare-workers 8 --progress-seconds 10
--preparation-cache-dir /runs/.golden-preparation-cache
# Optional separate fast persistent token storage:
--token-cache-dir /fast-data/a2ui-tokens
# Or explicitly turn either/both caches off:
--no-preparation-cache --no-token-cache
```

By default, new runs sharing an output parent reuse completed preparations
through `.golden-preparation-cache/`. The v2 cache owns its prepared-data copy:
earlier run folders no longer have to remain available. Old receipt-only cache
entries require one fresh preparation with this version. A cache hit requires
matching source bytes, Golden memberships/manifests, prompt and token budgets,
tokenizer assets, relevant Python/library versions, and preprocessing/schema
identity. Every saved output is rehashed before reuse and again while copying;
the regular tokenizer/Golden/split contracts are still checked. Changed,
missing, incomplete, or corrupted cache entries trigger full preparation.
The output receives independent copies, not mutable hard links.

After upgrading from a Golden32/35-only run to Bixby50 support, start with a
**fresh output directory**. New cohort membership/exclusions and preparation
code invalidate affected cache entries; one rebuild is expected. Retain the
cache root for later compatible runs. Do not edit old manifests or bypass
contract checks with `--continue-run`: adding this cohort does not authorize an
automatic optimizer restart or relabel old scores as new three-cohort results.

The HF training path also caches the final token IDs, attention masks and
completion-only labels, enabled by `--token-cache` (the default). It uses
`<preparation-cache-dir>/tokens/`, or the explicit `--token-cache-dir`. One
process builds an entry while matching processes wait; preflight, the subsequent
training launch and every matching GPU worker load the verified memory-mapped
tensor dataset. The cache does not save model-forward success or checkpoint
scores. Runtime model/tokenizer compatibility, tensor checks and forward/GPU
preflight still run. The legacy TRL tokenization path does not use this cache.

Changing epochs, learning rate, GPU count or LoRA targets alone no longer
invalidates prepared data or tokens. Changing source data, Golden exclusions,
prompt, tokenizer/template behavior, sequence budgets, or relevant
filtering/formatting/masking implementation invalidates the affected cache.
Augmentation changes training occurrences, so augmented training gets its own
token entry; repeated matching trials share it. Logs show `CACHE HIT`,
`CACHE MISS`, disabled status, and build/wait progress. A hit still reads/hashes
data and performs integrity checks; startup is not zero-cost.

Use fast persistent storage shared by all workers on the host, outside the
model, source and individual run directories. This trades extra disk space for
less repeated CPU processing: the store retains prepared JSONL plus compact
token tensors, and each run retains its independent prepared copy. No automatic
cache eviction is performed. Keep the cache across runs; remove only an unused
cache directory when all jobs using it have stopped, or change its location to
start a clean cache. Both flags are independent: `--no-preparation-cache` reruns
filtering, while an unchanged tokenized dataset may still hit the token cache.
`--no-token-cache` reruns training tokenization even if preparation is reused.

An offline
v9 archive is an input dataset, **not** an exact-model prepared cache; its first
preparation still runs all checks. Cache reuse does not skip GPU preflight,
model loading, training-time tensor checks, or checkpoint evaluation. A normal
training command needs no new flags to enable both caches:

```bash
python training/scripts/run_golden_training.py \
  --profile e2b --model-dir /models/e2b \
  --input-dir /data/source-bound-train-val \
  --output-dir /runs/e2b-next --epochs 1 \
  --preparation-cache-dir /runs/.golden-preparation-cache --execute
```

The generated `fit/training_config.yaml` records `training.token_cache: true`
and an absolute `training.token_cache_dir`, shared by preflight and training.
`prepare_review_training.py` also exposes `--token-cache` / `--no-token-cache`
and `--token-cache-dir`; its default is relative to that command's output parent.
Older/direct SFT YAML recipes without `training.token_cache` retain the uncached
behavior; adding the option is explicit for those separate workflows.
Persistent tokens currently support built-in Transformers fast tokenizers and
the shared `ModelAdapter` formatter. For a custom tokenizer subclass, slow
tokenizer, or custom adapter formatter, use `--no-token-cache`; their arbitrary
runtime behavior cannot safely be inferred from config alone.

`--continue-run` still requires the same saved workflow options and verifies
completed artifacts. It does not resume a partially prepared dataset or
silently restart an interrupted optimizer. Runs created before these new
options were added should finish with their original checkout or use a fresh
output directory with the updated launcher. Do not update the checkout of a
currently running job in place.

If the terminal seems quiet, inspect `pipeline_manifest.json` for `status` and
`active_stage`, then read the matching log. Idle GPUs during `prepare` are
expected; compare the CPU progress/ETA instead. Use the exact flags
`--input-dir` and `--epochs` (not `--input-dit`).

### `ArrowInvalid` followed by pandas `Trailing data` at startup

If startup reports that a nested metadata field changed from boolean to string
while loading `prepared/train.jsonl`, the immediate failure is the generic
Arrow JSON loader trying to infer one fixed schema for every provenance field.
These heterogeneous metadata values do not by themselves mean the JSONL or
training targets are invalid. The later pandas `Trailing data` message can be
a secondary fallback error, not the original cause.

The shared SFT runner now reads each JSONL record before Arrow conversion.
With the token cache enabled, only the validated token IDs, masks and labels
enter Arrow. Without that cache, and for the shared runner's TRL backend, the
loader formats the model inputs first and builds an explicit string-only
training view. Arbitrary nested metadata no longer participates in Arrow
schema inference. Original JSONL, provenance, references and source hashes are
unchanged; malformed JSON and invalid model inputs still fail rather than
being silently skipped or repaired.

After updating the training checkout, rerun the same launcher command with a
**new `--output-dir`** and the same persistent cache location. Do not use
`--continue-run` with the failed old recipe or edit its bound generated YAML.
The first run with the new cache version may rebuild once; subsequent matching
runs reuse its verified data. No source-file rewriting, metadata coercion, or
pandas dependency workaround is required for this failure. This fix applies to
the shared E2B/270M SFT workflow described here; separate experimental runners
have their own loader and validation contracts.

See the [cache/startup fix report](DATA_CACHE_AND_STARTUP_FIX_20260914.md) for
the implementation boundaries, restart example and validation evidence.

### H100 OOM, cuDNN attention backward, or peer/NVLink failure

The September 14 fix lowers automatic E2B microbatch to 1, preserves global
batch 32, and disables only cuDNN SDPA for the complete SFT call. Every non-QAT
HF CUDA worker probes the longest prepared training shape with real backward
and configured checkpointing before optimizer step 1, including on cache hits.
The probe never updates weights. Stateful QAT retains its separate gates.

Pull only after the failed job has exited; use a **fresh output directory** and
the **same preparation/token caches**. Do not resume the failed HPO suite or
edit its bound recipe. The [failure report](H100_CUDNN_BACKWARD_FIX_20260914.md)
includes ordinary/HPO rerun commands and the hardware-health escalation path.
CPU regression tests cannot guarantee the remote CUDA environment is healthy.

## 5. Quantization, LiteRT and MTP are separate

This command closes the **HF training plus dual-Golden checkpoint-testing**
workflow. It does not convert E2B to official retained-scale QAT packages or
automatically export/test W32/W16/W8/W4 LiteRT-LM variants. Do not call dense
QAT-derived LoRA training a verified retained-scale QAT result.

For optional 270M full-model W8 QAT after a successful SFT run, initialize a
fresh run from its selected full checkpoint (including its saved tokenizer):

```bash
python training/scripts/run_golden_training.py \
  --profile 270m --qat \
  --model-dir /runs/gemma270m-epoch1/fit/training/best_golden_checkpoint \
  --output-dir /runs/gemma270m-w8-qat --execute
```

This uses the lower-learning-rate QAT recipe and evaluates with fake
quantization enabled. It remains an HF checkpoint workflow, not a LiteRT
conversion or runtime-parity test.

Those workflows remain available through the
[pipeline inventory](training_pipeline_inventory.md) and the
[retained-scale GPU runbook](gemma4_mobile_qat_remote_pc_runbook.md), with their
own seeds, reference pins, converter environments and runtime runners. Their
existing MTP enable/disable option remains separate; this capability command
does not jointly train an MTP drafter. 270M has no Gemma 4 MTP drafter.

Return the complete scorecard, raw/per-case predictions, manifests, quarantine
counts, resolved config, GPU/environment versions, stage logs and checkpoint
metadata after a real run. CPU tests cannot establish CUDA memory fit,
distributed stability, actual quality or deployed-runtime parity.

Implementation test evidence is in
[the September 12 shared-prompt validation record](golden_e2e_validation_20260912.md).
