# H100 training and Golden evaluation — September 12, 2026

**Current command:** use the [end-to-end quickstart](GOLDEN_E2E_QUICKSTART.md).
`run_golden_training.py` now prepares the checked-in Stage 3 data with one
production prompt, filters reserved sources, and automatically evaluates the
selected-best and actual final checkpoint on **both Golden32 and Golden35**.
Its default is plan-only; add `--execute` to run on the GPU host. Golden32
remains the development/selection set; Golden35 remains final-only.

```bash
python training/scripts/run_golden_training.py \
  --profile e2b --model-dir /models/e2b \
  --output-dir /runs/e2b-new --execute
```

The manual walkthrough below records the earlier lower-level workflow and
hardware/data rationale. It is **superseded as a clone-and-run recipe**: simply
feeding the historical raw archive messages into its manual preparation steps
can recreate the old prompt mismatch. Use the new launcher to normalize and
bind the full system/few-shot/task scaffold, rather than editing prepared files
or their hashes. Both raw Golden sets and the default `dataset_v1` Stage 3
source are tracked; only model/tokenizer assets and generated outputs must be
provided or created on the GPU host. The old statements about unavailable full
training files refer to the external review archive, not the tracked default
source.

No model training,
model downloads, H100 benchmarks, or LiteRT runtime inference were performed
while implementing these changes. CPU tests and dataset preparation are not
model-quality evidence. Follow the quickstart's short smoke before a long run.

## Benchmark identities — do not mix these scores

| Cohort | Current status | Intended use |
|---|---|---|
| `training/data/eval/golden32_archive_repeat_v1/golden32.jsonl` | 32 strict-valid occurrences, **31 unique sources** | Development/checkpoint selection for the archive-based capability experiment |
| Original September 3 demo Golden32 | 32 unique strict-valid sources; different membership | Existing official/multiformat comparison pipeline |
| `training/data/eval/golden35_v1/golden35.jsonl` | **35 strict-valid references / 35 unique sources** retained from the historical 50-case source | Separate Golden35 final-evaluation lane; all 15 excluded source identities remain reserved |

The archive's invalid `q_012053/r_012053_01` slot was replaced, at the user's
request, by an exact copy of `q_017270/r_017270_01`. The donor's input, target,
assets and source identities were not rewritten. Only the occurrence ID and
explicit benchmark metadata differ. The original archive is untouched. See the
[artifact and manifest](../data/eval/golden32_archive_repeat_v1/README.md).

The duplicate does not add coverage: this revision loses the failed real-estate
case. Evaluation reports both the ordinary 32-occurrence average and
`unique_source_generation_reward_v5_4_avg`. The latter first averages repeated
observations within each source, then gives all 31 sources equal weight, and
is the selection metric in resolved review configs. Never compare these values
directly with the original archive32 or September3 scores. No predicted IDs or
graphs are repaired during scoring. The failed source remains reserved from
training and ordinary validation even though it is not scored.

## 1. Prepare clean, source-disjoint supervision

Read [the measured data audit](data_filtering_audit_20260912.md). Full training
files and tokenizer assets were not available on the review PC. Audit the full
files on the GPU host; the small archive samples do not establish corpus-wide
quality. The slim samples lack source IDs and must not be used as production
training inputs. Recover source/query IDs by joining authoritative Stage2/3
records, or rebuild prepared supervision from completed Stage3 runs with the
existing dataset configs. Training must not generate new Stage1/2/3 data.

Run from the repository root in the GPU host's scheduler/container environment.
Set real paths; choose fresh output directories. Reserve both the Golden32
development set and Golden35. Golden35's adjacent manifest also reserves the
15 excluded original sources; removing a target from scoring never makes its
source eligible for training or ordinary validation.

```bash
export A2UI_TENSORBOARD_ROOT=/tensorboard
SOURCE=/absolute/path/to/full-source-bound-train-val
WORK=/absolute/path/to/new-a2ui-experiment
E2B=/absolute/path/to/dense-e2b-base-with-tokenizer
G270=/absolute/path/to/gemma-3-270m-it-with-tokenizer
G32=training/data/eval/golden32_archive_repeat_v1/golden32.jsonl
G35=training/data/eval/golden35_v1/golden35.jsonl

for SPLIT in train val; do
  python training/scripts/audit_filter_training_data.py \
    --input "$SOURCE/$SPLIT.jsonl" \
    --reserve-golden32 "$G32" --reserve-golden35 "$G35" \
    --require-source-identities --output-dir "$WORK/filtered-$SPLIT"
done
```

Omit `--output-dir` for a read-only report. The filter checks raw and
URL-placeholder-normalized source overlap, strict Express/wire validity,
root reachability, semantic roundtrip, and exact source/target duplicates.
Accepted records remain unchanged. All excluded original rows and reasons are
retained in quarantine. Different valid layouts for the same response are
reported, not indiscriminately dropped. The later launch gate also requires
train/val/Golden source disjointness. Do not randomly split augmented copies
of the same source into different cohorts.

Prepare complete examples with the **actual tokenizer separately for each
model**. Start with root-first ordering; do not change ordering, examples or
prompt between training and inference. The repeated benchmark sidecar is
automatically hash-checked and embedded in its tokenized preparation manifest.

```bash
python patches/build_bottom_up_dataset.py \
  --input train="$WORK/filtered-train/accepted.jsonl" \
  --input val="$WORK/filtered-val/accepted.jsonl" --input golden32="$G32" \
  --output-dir "$WORK/prepared-e2b" --ordering root-first \
  --tokenizer "$E2B" --tokenizer-loader pretrained_tokenizer_fast \
  --local-files-only --max-seq-length 4096 --max-input-tokens 4096 \
  --chat-template-kwargs '{"enable_thinking":false}'

python patches/build_bottom_up_dataset.py \
  --input train="$WORK/filtered-train/accepted.jsonl" \
  --input val="$WORK/filtered-val/accepted.jsonl" --input golden32="$G32" \
  --output-dir "$WORK/prepared-270m" --ordering root-first \
  --tokenizer "$G270" --tokenizer-loader auto_tokenizer \
  --local-files-only --max-seq-length 4096 --max-input-tokens 4096 \
  --chat-template-kwargs '{"enable_thinking":false}'
```

Inspect `manifest.json`, `prompt_scaffolds.json` and `quarantine.jsonl`.
Overlong training examples are quarantined whole; inputs/targets are never
silently truncated. A quarantined Golden occurrence aborts preparation.
Scaffolds must match across training and evaluation; do not fix mismatches by
editing a prepared file or its hash. Select/rebuild one coherent source scaffold
and prepare again. Count target lengths over 2,048 tokens separately: they are
a deployment budget concern, not permission to truncate reference outputs.

## 2. Resolve the detected H100 configuration and smoke-test

`--devices auto` is the default. It selects all scheduler-visible devices,
including UUID masks, without exposing hidden GPUs. The [profile table](gpu_training_profiles.md)
documents 2/4/8 H100 settings: E2B microbatch1, 270M microbatch4, global batch32
with the appropriate accumulation. Batch/learning rate do not silently scale
with GPU count. These are starting profiles, not measured maximum throughput.

```bash
python training/scripts/prepare_review_training.py --profile e2b \
  --model-dir "$E2B" --dataset-dir "$WORK/prepared-e2b" \
  --golden-file "$WORK/prepared-e2b/golden32.jsonl" \
  --output-dir "$WORK/e2b-smoke" --steps 20

python training/scripts/launch_review_training.py \
  --config "$WORK/e2b-smoke/training_config.yaml" --preflight-only --execute
python training/scripts/launch_review_training.py \
  --config "$WORK/e2b-smoke/training_config.yaml" --execute

python training/scripts/prepare_review_training.py --profile 270m \
  --model-dir "$G270" --dataset-dir "$WORK/prepared-270m" \
  --golden-file "$WORK/prepared-270m/golden32.jsonl" \
  --output-dir "$WORK/270m-smoke" --steps 20
python training/scripts/launch_review_training.py \
  --config "$WORK/270m-smoke/training_config.yaml" --preflight-only --execute
python training/scripts/launch_review_training.py \
  --config "$WORK/270m-smoke/training_config.yaml" --execute
```

Inspect all rank logs, memory fit, finite loss, actual parameter updates and a
complete 32-occurrence evaluation. Use a new run directory with `--steps 1000`
for a bounded pilot, then a new epoch run. Explicit `--microbatch`,
`--effective-batch`, `--dataloader-workers`, attention and checkpointing
overrides are available. Do not silently lower batch after OOM or resume an
old optimizer after changing data, tokenizer, benchmark or recipe.

Validation-loss/checkpoint cadence is 500 updates; full Golden generation is
every 1,000 plus the final weights. `--golden-every-steps` must be divisible by
`--eval-steps`. Generation defaults to **2,048 new tokens**. Distributed ranks
evaluate disjoint cases using a scoped KV cache, then gather once for scoring.
The model returns to its training cache/mode afterwards. No training GPU is
reserved for a second inference model. Evaluation is synchronous: it pauses
training, so zero-overhead concurrent inference is not promised. Measure
`evaluation_pause_seconds`, generation throughput and scoring/save time in
TensorBoard before shortening the cadence. A single generated case remains
sequential; adding GPUs cannot eliminate its latency.

Logs and evaluation records use `/tensorboard/<run-id>/`. Standalone rechecks
select CUDA automatically when the training config used `device_map: none`.
They read the selected/final checkpoint step from metadata. Use unique output
directories for every runtime invocation; stale LiteRT outputs are rejected.

## 3. Final and deployed evaluation

The trainer evaluates final weights even when the final update is not on the
periodic boundary. `final_adapter` (E2B) or `final_model` (270M) and
`best_golden_checkpoint` are different artifacts; report their identities and
scores separately. To independently recheck the selected E2B adapter:

```bash
python training/scripts/evaluate_checkpoint_on_golden.py \
  --config "$WORK/e2b-smoke/training_config.yaml" \
  --checkpoint "$WORK/e2b-smoke/training/best_golden_checkpoint" \
  --checkpoint-kind adapter --split "$WORK/prepared-e2b/golden32.jsonl" \
  --max-rows 32 --required-rows 32 \
  --run-id e2b-smoke --evaluation-name selected-recheck \
  --output-dir "$WORK/e2b-smoke/selected-recheck" \
  --max-input-tokens 4096 --max-new-tokens 2048 --metric-version v5_4
```

For 270M use its config/checkpoint and `--checkpoint-kind merged`. Golden35 is
the separate final-evaluation lane. It contains exactly the 35 references that
passed the original 50-case audit; no failed target is repaired or counted.
Prepare and evaluate it with an explicit 35-row guard and a distinct name:

```bash
python training/scripts/prepare_dataset.py \
  --config training/configs/datasets/golden35_stage3_eval.yaml

python training/scripts/evaluate_checkpoint_on_golden.py \
  --config "$WORK/e2b-smoke/training_config.yaml" \
  --checkpoint "$WORK/e2b-smoke/training/best_golden_checkpoint" \
  --checkpoint-kind adapter \
  --split training/outputs/datasets/golden35_stage3_eval/all.jsonl \
  --max-rows 35 --required-rows 35 \
  --run-id e2b-smoke --evaluation-name golden35-final \
  --output-dir "$WORK/e2b-smoke/golden35-final" \
  --max-input-tokens 4096 --max-new-tokens 2048 --metric-version v5_4
```

Before running the second command, verify that the prepared Golden35 scaffold
and tokenizer/chat-template settings match the selected checkpoint. If the
archive capability experiment uses a different scaffold, prepare a separate
version with that saved scaffold while preserving all 35 source/target pairs
and record its hash. Do not silently substitute prompts. Apply the same fixed
Golden35 and decoder to each LiteRT-LM variant using `--max-rows 35
--required-rows 35`, unique output directories and evaluation names.

Keep Golden35 out of checkpoint selection and data tuning to retain its role
as the final comparison set. Golden32 has influenced selection and remains a
development benchmark. Neither small cohort establishes rare-failure rates;
use a larger source-disjoint final test for a release decision.

The retained-scale official E2B and canonical multiformat pipelines keep their
separate seed, data and original September3 benchmark contracts. They now use
the current-prompt preparation at `golden32_20260903_eval_prompt_v2`, preserving
the old cached preparation. Rebuild on the GPU host with:

```bash
python training/scripts/prepare_dataset.py \
  --config training/configs/datasets/golden32_20260903_eval.yaml
```

Do not transplant the archive-repeat benchmark into their original 32-unique
pins. All legacy preparation now applies the production wire/reachability gate
too. Use the retained-scale runbook for official E2B QAT; its numerical seed
and MTP gates remain mandatory. E2B `--mtp` / `--no-mtp` remain supported in the
multiformat launcher; public comparison exports remain target-only, and Gemma3
270M has no Gemma4 MTP drafter. A real host-specific LiteRT batch runner is still
required for deployment scores; no conversion/runtime success is implied here.

## 4. Filtering and augmentation decisions

Filter invalid targets and source leakage **before** considering more data.
Report full-corpus retention by intent, components and token length. Favor
independent Stage2/3-generated training cases covering rare components, valid
table property types, state/actions, long but compact outputs, nested/reference
graphs and realistic media. Sample new entities and wording from training-side
requirements, not from Golden inputs/targets. Revalidate every augmented pair
and retain parent/provenance IDs. Do not mutate targets with string replacements,
duplicate Golden cases into training, or oversample just to inflate an evaluation.
Root-first versus bottom-up is a controlled serialization experiment, not new
semantic training data. Compare augmentation in a fresh run against the same
training-side development split and fixed inference policy; no quality gain is
guaranteed by corpus size alone.

Implementation choices follow PyTorch's [DDP documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
and [performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).
Return environment/package versions, resolved configs, GPU inventory, manifests,
quarantine counts, per-rank logs, timing/peak-memory measurements, checkpoint
hashes and per-case raw/serving scores. Keep large model artifacts outside Git.
