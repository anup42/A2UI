# Optional resampling and sequential hyperparameter tuning

Review date: 2026-09-14. This extends the dense E2B LoRA / Gemma 3 270M
Golden training workflow. No GPU training or measured quality improvement is
claimed by this implementation review.

## Review conclusion

Keep the unaugmented recipe as the baseline. Test **capped rare-component
resampling**, disabled by default, against that baseline. It increases exposure
to existing correct examples; it does not create new facts, new layouts, or new
semantic coverage. Do not paraphrase sources, invent actions/media, rewrite IR,
or put quarantined records back into training as augmentation. Those approaches
could recreate the source/target defects repaired in the archive.

The current v9 report records 112,842 training and 2,300 validation examples.
Its manifest was checked again against SHA256
`cdce0bc4c537ed222a9ad0f5822fbc5772cb0c4df97a15a51e7c88f8ce1c0251`.
These are archive counts **before exact local-tokenizer admission**, not a
promise that every example will fit the selected model's token budget.

| Component | Training examples containing it | Validation examples |
|---|---:|---:|
| Table | 111,462 (98.78%) | 2,270 |
| Image | 2,159 | 45 |
| Tabs | 2,055 | 42 |
| CodeBlock | 488 | 10 |
| ConsoleLog | 429 | 12 |
| EmailPreview | 166 | 4 |
| CheckBox | 123 | 4 |
| Slider | 65 | 0 |
| TextField | 53 | 1 |
| ChoicePicker | 46 | 0 |
| AudioPlayer | 1 | 0 |

Counts overlap: one sample can contain several components. Evidence:
[v9 profile](../reports/offline_recovery_20260913_v9/profile.json) and
[final archive review](messages_archive_final_review.md).

The imbalance makes a resampling experiment reasonable, not guaranteed to
improve Golden scores. Many rare-component examples also contain tables, so
resampling will not eliminate table dominance. It cannot teach missing video
coverage or justify repeatedly memorizing the sole audio example. Validation
has little or no support for several rare components; its aggregate loss cannot
establish quality for those components. Keep the held-out split unchanged during
this comparison. A future broader evaluation set needs independent examples,
not copies of the training set or tuning against Golden35.

## What the optional augmentation does

Use `--augmentation rare_components` on the ordinary training launcher.
`--augmentation none` is the default.

- Start only from strictly validated, tokenizer-bound **training** rows.
- Count rarity by connected source family, not repeated target count.
- Restrict to supported non-layout components with at least five training
  families and presence in at most 3% of training families. Common structure,
  Table, Text, Icon and Button are not rarity signals.
- Add no more than 10% extra occurrences by default. This is a ceiling, not a
  target: if fewer families qualify, fewer examples are added.
- Default maximum is two total occurrences per family including originals;
  existing larger families remain intact but receive no additional copies.
- Preserve response, completion, template and token bindings; only occurrence
  identity/provenance distinguishes added copies.
- Preserve validation, Golden32 and Golden35 bytes. Recheck train/validation
  separation and reserved Golden sources before configuring training.
- Write a fresh `augmented/` directory and an augmentation manifest. The base
  prepared copy and the original v9 archive are not modified.

The optional controls are `--augmentation-max-extra-fraction` (default `0.10`,
maximum `0.5`) and `--augmentation-max-family-repeats` (default `2`, allowed
`2`–`5`). Raising these limits can increase memorization. Review actual added
counts and per-component exposure in the saved manifest first.

Example, after copying the complete repaired v9 folder to the GPU host:

```bash
python training/scripts/run_golden_training.py \
  --profile e2b --model-dir /models/e2b \
  --input-dir /data/full_data_archive_recovered_v9 \
  --output-dir /runs/e2b-rare-epoch1 \
  --augmentation rare_components --epochs 1 --execute
```

Use `--prepare-only` instead of `--execute` to inspect augmentation counts
without loading model weights. Use a fresh output directory. The original
multi-GB v9 archive is not included in Git; the Golden cohorts and preparation
code are. See the [archive transfer/run instructions](messages_archive_final_review.md).

## Hyperparameters: test, do not guess an optimum

Current SFT baseline for both profiles: learning rate `2e-5`, weight decay
`0.01`, warmup fraction `0.03`, one epoch, gradient checkpointing enabled.
E2B uses rank-32 LoRA; 270M trains the full model. Their optimal learning rates
need not be the same. The separate 270M W8 QAT mode starts at `5e-6`.

There is no new loss curve, gradient trace, or actual trained checkpoint in this
review that proves these defaults are wrong or identifies an optimum. Start
with a small controlled search around them, holding source data, seed, effective
batch and optimizer-step budget fixed. Fresh model initialization per trial and
an explicit objective follow the [Transformers hyperparameter-search guidance](https://huggingface.co/docs/transformers/hpo_train).

`run_golden_experiments.py` provides a sequential, bounded search. It deliberately
does not launch concurrent trainers or require a distributed tuning service.
The default candidates are the baseline, half learning rate, double learning
rate, and a stronger-regularization candidate. `--include-augmentation` adds a
resampled-data candidate at the baseline hyperparameters. A custom JSON trial
list supports a reviewed search space without editing training code.

```bash
# Plan only: no output files, model loading or training.
python training/scripts/run_golden_experiments.py \
  --profile e2b --model-dir /models/e2b \
  --input-dir /data/full_data_archive_recovered_v9 \
  --output-dir /runs/e2b-tuning-v1 \
  --trial-steps 1000 --include-augmentation

# Add --execute to run the same planned experiment on the GPU host.
```

Use `--profile 270m --model-dir /models/gemma-3-270m-it` and a different output
directory for 270M. Run it after the E2B command finishes to keep both profiles
sequential. `--trial-steps` is mandatory: five trials at 1,000 updates cost 5,000
training updates plus preparation, preflights, saving and evaluation. This is
**screening**, not five completed full-data epochs or a guaranteed optimum.
At fixed epochs, added rows increase training work; fixed steps avoid that
confound in the sweep.

Selection uses the best checkpoint's
`unique_source_generation_reward_v5_4_avg` on Golden32 (31 unique sources in
32 occurrences). Ordinary validation loss and both best/final Golden32
results remain visible. Trials do **not** evaluate Golden35. After all trials
finish, the runner saves the selected trial/checkpoint identity before testing
that checkpoint once on Golden35. Golden35 never participates in ranking.
Treat its score as a final report, not feedback to launch another adaptive
sweep; repeated adaptation would turn it into development data.

For a full follow-up run, use the saved selected settings with a fresh output
directory and an explicit full budget. Screening rankings can change over a
full epoch; inspect loss, per-case failures and parameter updates, not just a
tiny aggregate-score difference. Repeat promising configurations with another
seed on development data when feasible. Do not change many unrelated controls
at once or call a single-seed winner statistically established.

The suite writes `experiments_manifest.json` (status and bindings),
`comparison.json` (all trial results), `selection_locked.json` (winner pinned
before holdout), `selected_golden35/` (holdout evidence), and
`selected_full_training_options.json` (exact plan-only command argument list
for a fresh full run; append `--execute` only on the training host). Failed
trials stop the suite and preserve completed outputs; the runner never silently
restarts a failed optimizer or retries a holdout. Review the saved manifest to
recover deliberately. A new suite requires a fresh directory.

Custom `--trials-file` example (baseline is added automatically):

```json
[
  {"name": "lr_3e5", "learning_rate": 0.00003},
  {"name": "lr_3e5_rare", "learning_rate": 0.00003, "augmentation": "rare_components"}
]
```

Only learning rate, weight decay, warmup ratio and augmentation vary inside a
suite; unsupported keys and duplicate configurations fail early. There are at
most twelve trials including baseline. The built-in regularization candidate
changes weight decay and warmup together; use single-change custom trials if
you need to isolate their individual effects.

The ordinary launcher also accepts `--learning-rate`, `--weight-decay`,
`--warmup-ratio`, `--seed`, and `--logging-steps`. Explicit values are validated,
saved in the bound training config, and honored after QAT profile defaults.

## Comparison, progress and performance

TensorBoard defaults to the MLP mount `/tensorboard`. Every trial has a unique
run identity; compare the same tags across runs for loss, learning rate,
Golden32 metrics and final checkpoint scores. The sweep adds HParams entries
and comparison scalars, backed by persistent JSON records. HParams is the
[PyTorch TensorBoard interface for recording settings with their metrics](https://docs.pytorch.org/docs/2.14/tensorboard.html).

Normal training curves are under `/tensorboard/<unique-trial-id>/training`;
the sweep comparison/HParams group is
`/tensorboard/experiments/<suite-id>/`. Select matching scalar tags across
trial runs or open the HParams view. Run names include a suite path hash to
avoid merging independent suites with the same folder basename.

Logs identify the active trial/stage, effective settings, dataset counts,
optimizer budget, GPU profile and TensorBoard path. Existing subprocess output
is streamed to console and files; blocking stages emit heartbeats. Training
metric cadence is ten optimizer updates by default (`--logging-steps`).

Prepared data and exact HF token IDs, attention masks and completion-only labels
are now persistently cached by default. All sequential trials share
`<suite output parent>/.golden-preparation-cache/` and its `tokens/` child, not a
different cache under each trial. Set `--preparation-cache-dir` and optionally
`--token-cache-dir` to reuse the same persistent storage across separate suites
and normal full runs. The selected full-training handoff preserves those paths
and enable/disable settings. `--no-preparation-cache` and `--no-token-cache`
control the two layers independently.

The first matching process builds and validates each token entry; preflight,
training and other GPU ranks memory-map the completed, checksum-verified entry.
Learning-rate, epoch, GPU and LoRA-only changes do not trigger data rebuilding.
Source/prompt/tokenizer/sequence/masking changes invalidate affected entries;
an augmented training split has a different occurrence identity, so it cannot
reuse unaugmented training tokens. Matching augmented trials can share tokens.
Raw-token admission checks occur during building, and live tensor/model checks
still run on every launch. Forward success, scores and optimizer state are never
reused as cache evidence. The separate legacy TRL path remains uncached.

The owned prepared-data cache survives deletion of earlier run folders; old
receipt-only entries require one fresh v2 preparation. Logs distinguish hits,
miss reasons, build progress and waiting for another builder. Hits still require
hashing and integrity I/O; keep caches on fast persistent storage and budget
disk for prepared data plus token tensors. See the quickstart's
[startup/cache controls](GOLDEN_E2E_QUICKSTART.md#startup-progress-cpu-preparation-and-reuse).

The training seed is now set before model/LoRA initialization as well as passed
to Trainer. Suite inputs, model inventory, recipe and implementation identities
are checked across trials; do not edit or pull into an active training checkout.

The checked GPU path uses all scheduler-visible devices with DDP and selects
H100 2/4/8-GPU batch/accumulation profiles. The global effective batch remains
32 on H100 >=70 GiB rather than silently growing with GPU count. E2B now uses
microbatch 1 with accumulation 16/8/4 for 2/4/8 H100s; 270M stays at microbatch 4.
The shared SFT policy disables cuDNN SDPA during forward and backward, while
retaining the other existing backend choices. Preserve BF16,
SDPA, pinned transfers and bounded DataLoader workers. Explicit CPU worker and
microbatch overrides remain available. These are supported performance
techniques, not an H100 throughput measurement; see the
[PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

`--no-gradient-checkpointing` can trade VRAM for speed; only use it after the
training-mode backward/memory smoke succeeds with the intended sequence length and batch.
An explicit `--microbatch` can reduce accumulation overhead if memory allows.
Keep the global effective batch unchanged for quality comparisons. Do not
enable experimental packing, truncation, unverified compilation or a new
distributed checkpoint format just to improve a nominal throughput number.

The non-QAT HF CUDA path now runs a live longest-shape backward preflight,
including checkpoint recomputation and resident accumulated gradients. It
does not update weights or reuse cached success, and does not certify NCCL or
optimizer-memory headroom. After the September 14 cuDNN/OOM failure, use a new
suite output directory and retain the same cache paths; do not continue the
failed suite across this recipe change. See the
[failure report and fresh-suite command](H100_CUDNN_BACKWARD_FIX_20260914.md).

Periodic Golden generation pauses optimization and is sharded over training
GPUs; it is not free concurrent inference. It defaults to every 1,000 updates,
with validation every 500 and generation capped at 2,048 new tokens. Increase
the Golden interval (a multiple of validation cadence) if measured evaluation
overhead is excessive. Standalone end tests are sequential. Do not start an
unrelated inference service on the same GPUs expecting unchanged training
speed; scheduler-isolate its resources if necessary.

The existing official-format E2B retained-scale QAT/export and MTP workflows
remain separate and unchanged. This search does not train an MTP drafter,
export LiteRT-LM variants, or establish official-format parity. Do not apply
LoRA/SFT sweep settings to those workflows without their own reviewed contract.
