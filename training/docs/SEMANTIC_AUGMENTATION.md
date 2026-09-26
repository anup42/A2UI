# Muse semantic augmentation at training startup

Use `--augmentation` (equivalent to `--augmentation semantic`) on a **fresh**
Gemma 270M Golden-training or official E2B retained-scale QAT LoRA run.
The default remains `none`. Existing `--augmentation rare_components` still
repeats validated examples without generating new labels.

This is a dataset preparation stage, not generation inside a training batch.
After ordinary preparation and train/validation/holdout isolation, the launcher
invokes `dataset/scripts/generate_training_augmentations.py` once. Dataset code
asks the running Muse teacher for a changed synthetic source, separately asks
it to assess source coherence, and uses the current production Stage 3 prompt
to generate a **new** A2UI Express target. Donor IR is never supplied or edited.
Training then validates and tokenizes candidates with the actual student
tokenizer before publishing an immutable augmented dataset.

## Coverage and safeguards

The candidate schedule rotates through nine recipes:

1. Multi-section completeness, including important information near the end.
2. Table shape, record counts and consistent summaries.
3. Coherent entity/numeric substitutions and dependent totals.
4. Explicit action versus informational/no-action contrasts.
5. Alternate wording and source formatting.
6. Missing, uncertain and conditional information.
7. Source length and fact-position variation.
8. Forms, state and less common controls.
9. Exact literals, escaping, identifiers, URLs and media references.

These are requested categories, not a promise that all categories survive.
The report records actual attempted and accepted coverage. Fewer than nine
attempts cannot cover all recipes; at least two rotations are needed for both
sides of action and length contrasts. Candidates must pass all Stage 3 training
admission checks, with no blocking/review reasons, and then training's strict
Express/wire/reachability, source-isolation and tokenizer checks. Oversized
examples are rejected whole, never silently truncated.

- Donors come only from the already-prepared training split. Shared identities
  and exact/URL-normalized source text form transitive source families.
- Validation, Golden32, Golden35 and Bixby50 are never donors, even if a final
  holdout's evaluation was disabled. Their prepared files remain byte-identical.
- At most one candidate is attempted per eligible source family. Families at
  their total exposure cap are not selected; original rows are never removed.
- Default maximum: **500 attempts**, **10% additional rows**, **10% additional
  sequence tokens**, and **two total occurrences per source family**. The most
  restrictive bound wins; this does not attempt to balance every category.
- The teacher process has a two-hour deadline by default. No fallback teacher,
  automatic server launch, model download, or silent switch to resampling.
- Zero accepted/admitted candidates, a deadline, a hash mismatch, or altered
  evaluation artifacts stops the run before student training.

Muse source review is an automated consistency assessment, **not independent
fact checking or human review**. Synthetic examples are marked accordingly.
Passing these checks does not establish rendered quality or improved accuracy;
compare the trained baseline and augmentation trial before adopting the recipe.

The attempt limit is a ceiling, not a throughput promise. Each source that
reaches Stage 3 needs at least three serial model calls, plus metrics and any
bounded regeneration. Start with 18 attempts and size the deadline from measured
latency before requesting hundreds. The teacher context must fit both donor and
candidate during review, and the full Stage 3 prompt. Teacher context failures
are audited rejections; Stage 3's source prompt cap is 16k estimated tokens and
the student uses its own separate exact-token limit.

## Teacher environment

Start the existing Muse SGLang endpoint separately before running the commands.
The registered teacher is `muse_glimmer_30b_sglang_reasoning_dflash`. This version
only accepts that registered Muse HTTP teacher; the model option is an explicit
provenance identifier, not a generic fallback selector.

The existing local adapter accepts these environment settings:

```bash
export LOCAL_VLLM_ENDPOINTS=http://127.0.0.1:30000/v1/chat/completions
export LOCAL_VLLM_SERVED_MODEL=muse-glimmer
```

Use your actual approved endpoint and served model alias. The augmentation
subprocess enables Muse reasoning, bounds individual calls/retries and disables
the shared prompt cache. It does not use a cloud API key or start a teacher.
Use a separate teacher host/GPU allocation: a teacher left resident on the
student GPUs can cause memory preflight or training to fail. Automatic startup
continues into student training after augmentation; it does not pause to stop
an external server.

`--augmentation-python` selects an interpreter with the repository's **dataset**
dependencies installed. By default the current training interpreter is used.
Interpreter symlinks are preserved so an isolated venv stays isolated. No new
training-library dependencies are needed when augmentation is off.

## Gemma 270M command

From the repository root, replace the paths below. This uses the current
Golden-training launcher; the older 270M multi-format wrappers are unchanged.

```bash
python -u training/scripts/run_golden_training.py \
  --profile 270m \
  --model-dir /models/gemma270m \
  --input-dir /data/improved_dataset \
  --output-dir /runs/gemma270m_muse_aug_01 \
  --devices auto --seed 42 \
  --max-seq-length 4096 --max-input-tokens 4096 \
  --augmentation \
  --augmentation-python /envs/dataset/bin/python \
  --augmentation-max-samples 18 \
  --execute
```

For the existing 270M W8 QAT continuation, also add `--qat` and set `--model-dir`
to the selected full SFT checkpoint. The augmentation stage is otherwise the
same. `--prepare-only` prepares and augments without loading student weights,
but **does contact Muse** when semantic augmentation is enabled.

## Official E2B QAT LoRA command

Use the same verified seed, packed source and official LiteRT-LM used by your
existing official run. This changes only the prepared training data; retained
scales, LoRA scope, checkpoint selection, export and packaged drafter are not
changed by augmentation.

```bash
python -u training/scripts/run_official_mobile_pipeline.py \
  --model-dir /models/e2b_mobile_qat_seed \
  --source-safetensors /models/official_packed/model.safetensors \
  --official-litertlm /models/official_e2b.litertlm \
  --input-dir /data/improved_dataset \
  --output-dir /runs/e2b_qat_lora_muse_aug_01 \
  --exporter-python /envs/retained_export/bin/python \
  --devices auto --seed 42 --microbatch 1 --effective-batch 32 \
  --epochs 2 --lora-rank 16 --lora-alpha 16 --learning-rate 1e-5 \
  --max-seq-length 4096 --max-input-tokens 5120 --max-new-tokens 2048 \
  --augmentation \
  --augmentation-python /envs/dataset/bin/python \
  --augmentation-max-samples 18 \
  --execute
```

Omit `--execute` to inspect a plan without teacher calls or training. The examples
request 18 attempts for an initial bounded trial; omitting that override uses
500. Both launchers also accept `--augmentation-timeout-seconds`,
`--augmentation-max-extra-fraction`, and `--augmentation-max-family-repeats`.
The fraction is a row **and token** cap for semantic mode; in rare-component
mode its previous row-resampling meaning is unchanged.

## Artifacts, recovery and experiments

- `prepared/`: original preparation; never overwritten.
- `semantic_augmentation/donors.jsonl`: selected train-only donor sources.
- `semantic_augmentation/generated/`: teacher request outcomes, source reviews,
  raw Stage 3 records, accepted records, category coverage and hash manifests.
- `semantic_augmentation/`: source-filter, tokenizer and token-budget rejection
  evidence. Generation details are in `generation.log` and `generated/run.log`.
- `augmented/`: published train split, copied holdouts and updated manifest.
  `augmentation.json` records added rows/tokens, recipe and provenance.

Only a completely validated bundle is published. Failed generation keeps its
diagnostics; retry with a fresh output directory. Official checkpoint resume
must omit the augmentation flag: it reuses the checkpoint's frozen dataset
(including earlier augmentation) and must not generate new data mid-resume.
Generic `--continue-run` verifies the existing plan and completed stage hashes;
it is not a request to regenerate an interrupted augmentation stage.

Do not compare equal epochs as if the token budgets were equal: augmentation
adds tokens. Match optimizer steps/effective batch and check measured training
tokens. A seeded teacher can still be nondeterministic across server versions,
batching or hardware; compare prepared dataset hashes, not only seeds. For
rank/LR ablations use one frozen, identical augmented preparation across trials
rather than independently regenerating variants and calling them the same data.

No GPU run, live Muse generation, or measured quality improvement is implied
by the offline tests of this feature.
