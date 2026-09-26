# Generate with Muse, stop Muse, then train frozen data

The preferred workflow is **standalone preprocessing before student training**:
prepare and augment once, stop Muse, then train Gemma 270M or official E2B
retained-scale QAT LoRA from the frozen `augmented/` bundle. No teacher is
needed during student training. All commands below are Linux/Bash examples
from the repository root; every `/models`, `/data`, `/runs` and `/envs` path is
a placeholder to replace with an actual local path.

`training/scripts/prepare_semantic_augmentation.py` prepares the original data
and isolates train/validation/holdouts before invoking
`dataset/scripts/generate_training_augmentations.py`. Dataset code
asks the running Muse teacher for a changed synthetic source, separately asks
it to assess source coherence, and uses the current production Stage 3 prompt
to generate a **new** A2UI Express target. Donor IR is never supplied or edited.
The preparation layer then validates and tokenizes candidates with the actual
student tokenizer before publishing an immutable augmented dataset. The
published training split already contains **all original admitted training
rows plus accepted new rows**. Do not append the original rows again, concatenate
raw generated records into it, or edit the frozen bundle.

Startup `--augmentation semantic` remains available as an opt-in alternative;
the training default remains `none`. Existing `rare_components` mode repeats
validated examples without generating new labels.

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

## 1. Start and verify the teacher

The registered teacher is `muse_glimmer_30b_sglang_reasoning_dflash`. This version
only accepts that registered Muse HTTP teacher; the model option is an explicit
provenance identifier, not a generic fallback selector.

Terminal A: start the existing foreground server supervisor with the approved
Muse-capable SGLang environment. This example allocates four H100 GPUs as two
TP=2 replicas on ports 30000 and 30001. Both the teacher and DFlash assistant
must already be downloaded locally. This is a configuration example, not a
measured memory/throughput guarantee; inspect the server logs and readiness
probe before generation. Do not share these GPUs with a trainer.

```bash
/envs/muse/bin/python -u dataset/scripts/run_muse_glimmer_stage3.py servers \
  --gpus 4 --gpu-ids 0,1,2,3 --tp 2 --base-port 30000 \
  --model-path /models/Muse-Glimmer-30B \
  --draft-model-path /models/Muse-Glimmer-30B-assistant \
  --served-model muse-glimmer --context-length 32768 \
  --log-dir /runs/muse_teacher_logs
```

Keep Terminal A running. Terminal B: probe the same two endpoints, then export
their addresses for the standalone augmentation subprocess. The probe checks
the served model, Muse parsers and DFlash configuration; it is not generation
of an augmentation dataset.

```bash
/envs/dataset/bin/python dataset/scripts/run_muse_glimmer_stage3.py probe \
  --gpus 4 --tp 2 --base-port 30000 --served-model muse-glimmer

export LOCAL_VLLM_ENDPOINTS=http://127.0.0.1:30000/v1/chat/completions,http://127.0.0.1:30001/v1/chat/completions
export LOCAL_VLLM_SERVED_MODEL=muse-glimmer
```

Use your actual approved endpoint and served model alias. The augmentation
subprocess enables Muse reasoning, bounds individual calls/retries and disables
the shared prompt cache. It does not use a cloud API key or start a teacher.
For a separate teacher host, supply its explicit endpoints to the probe using
`--endpoints` and export the same reachable URLs; localhost here means the
machine running the client.

`--augmentation-python` selects an interpreter with the repository's **dataset**
dependencies installed. By default the current training interpreter is used.
Interpreter symlinks are preserved so an isolated venv stays isolated. No new
training-library dependencies are needed when augmentation is off.

## 2. Prepare one frozen bundle per student tokenizer

Run the standalone script in the training environment, using
`--augmentation-python` for the separate dataset environment. Its `--model-dir`
requires the local student's `config.json`, `tokenizer_config.json` and all
tokenizer/chat-template assets needed to load that tokenizer. **Student weights
are not required or loaded for this preparation step.** Do not point this option
at the Muse model. The eventual trainer must have its own complete verified
student checkpoint and match the prepared tokenizer/prompt/length contract.

First omit `--execute` to inspect the plan. Add it only when ready to call Muse.
Use a fresh output directory for each execution, including retries. Choose
exactly one raw input source: `--input-dir` for a source-bound train/validation
archive, or `--source-run-dir` for a supported completed dataset run. These
options are alternatives, not extra data to merge.

Official E2B tokenizer-bound bundle (initial bounded trial):

```bash
/envs/training/bin/python -u training/scripts/prepare_semantic_augmentation.py \
  --profile e2b --model-dir /models/e2b_mobile_tokenizer \
  --input-dir /data/improved_dataset \
  --output-dir /runs/preprocess_e2b_muse_01 \
  --seed 42 --max-seq-length 4096 --max-input-tokens 5120 \
  --max-new-tokens 2048 --prepare-workers 0 \
  --augmentation-python /envs/dataset/bin/python \
  --augmentation-max-samples 18 --augmentation-timeout-seconds 7200 \
  --augmentation-max-extra-fraction 0.10 --augmentation-max-family-repeats 2 \
  --execute
```

Gemma 270M needs a **separate** preparation using its own tokenizer and limits:

```bash
/envs/training/bin/python -u training/scripts/prepare_semantic_augmentation.py \
  --profile 270m --model-dir /models/gemma270m_tokenizer \
  --input-dir /data/improved_dataset \
  --output-dir /runs/preprocess_270m_muse_01 \
  --seed 42 --max-seq-length 4096 --max-input-tokens 4096 \
  --max-new-tokens 2048 --prepare-workers 0 \
  --augmentation-python /envs/dataset/bin/python \
  --augmentation-max-samples 18 --augmentation-timeout-seconds 7200 \
  --augmentation-max-extra-fraction 0.10 --augmentation-max-family-repeats 2 \
  --execute
```

Run only the profile(s) you need. Do not reuse the E2B prepared bundle for 270M,
or assume their admitted candidates will match. Defaults are seed 42, sequence
limit 4096, evaluation-prompt limit 5120 for E2B / 4096 for 270M, student
evaluation generation limit 2048,
500 attempts, two-hour augmentation deadline, 10% extra rows **and tokens**, and
two total family occurrences. The examples override attempts to 18 for a pilot.
Review the published manifest and `augmentation.json` for actual admission,
counts, provenance and hashes; an attempt ceiling is not an accepted-row target.
The larger E2B evaluation-prompt limit does not allow a supervised training
sequence to exceed 4096. `--max-new-tokens` is not the Muse teacher's output
budget; teacher generation has its separate bounded configuration.

## 3. Stop Muse before using the same GPUs for training

After every required preparation completes successfully, press **Ctrl-C in
Terminal A** and wait for its supervisor to stop its replicas and exit. It owns
their process groups; do not use a broad process-kill command. Verify your
allocated GPUs no longer contain Muse/SGLang worker processes and that their
memory has been released before starting the student:

```bash
nvidia-smi -i 0,1,2,3
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

If Muse workers remain, inspect the supervisor logs and process ownership; do
not kill unrelated jobs or proceed on the assumption that a closed terminal
freed VRAM. The preparer does not automatically stop an external teacher.

## 4. Train without teacher calls

Both launchers accept `--prepared-input-dir` pointing to the published
`augmented/` directory, with `--augmentation none`. Do not also pass raw
`--input-dir` (or Golden's `--source-run-dir`). Reuse verifies file hashes and the exact
tokenizer, shared prompt and sequence/input limits; a mismatch fails instead of
retokenizing or silently generating replacement examples. A path change alone
does not make an incompatible tokenizer acceptable.

### Gemma 270M full SFT

This uses the current Golden-training launcher; the older multi-format wrappers
are unchanged. The model directory here contains the **full student weights**.

```bash
/envs/training/bin/python -u training/scripts/run_golden_training.py \
  --profile 270m \
  --model-dir /models/gemma270m \
  --prepared-input-dir /runs/preprocess_270m_muse_01/augmented \
  --output-dir /runs/gemma270m_muse_aug_01 \
  --devices auto --seed 42 \
  --max-seq-length 4096 --max-input-tokens 4096 --max-new-tokens 2048 \
  --augmentation none \
  --execute
```

For the existing 270M W8 QAT continuation, also add `--qat` and set `--model-dir`
to the selected full SFT checkpoint. The frozen bundle must still match that
checkpoint's tokenizer and all preparation settings.

### Official E2B QAT LoRA

Use the same verified seed, packed source and official LiteRT-LM used by your
existing official run. This changes only the prepared training data; retained
scales, LoRA scope, checkpoint selection, export and packaged drafter are not
changed by augmentation.

```bash
/envs/training/bin/python -u training/scripts/run_official_mobile_pipeline.py \
  --model-dir /models/e2b_mobile_qat_seed \
  --source-safetensors /models/official_packed/model.safetensors \
  --official-litertlm /models/official_e2b.litertlm \
  --prepared-input-dir /runs/preprocess_e2b_muse_01/augmented \
  --output-dir /runs/e2b_qat_lora_muse_aug_01 \
  --exporter-python /envs/retained_export/bin/python \
  --devices auto --seed 42 --microbatch 1 --effective-batch 32 \
  --epochs 2 --lora-rank 16 --lora-alpha 16 --learning-rate 1e-5 \
  --max-seq-length 4096 --max-input-tokens 5120 --max-new-tokens 2048 \
  --augmentation none \
  --execute
```

Omit `--execute` to inspect either training plan. Student training still runs its
normal memory/backward/contract checks, checkpoint selection, evaluations and
export gates; frozen input does not bypass them. It does not require an active
Muse endpoint. See the [official mobile runbook](OFFICIAL_MOBILE_QAT_PIPELINE.md)
for required seed, packed-source and export artifacts.

### Reuse exactly the same data for rank/LR ablations

Point every E2B rank/LR trial at
`/runs/preprocess_e2b_muse_01/augmented` with `--augmentation none`, unchanged
seed/lengths and the same original verified mobile seed. Use fresh run output
directories. For example, compare rank/alpha 16/16 at `1e-5`, 32/32 at `1e-5`,
and 16/16 at `5e-6` by changing only those flags and the training output path.
Do not rerun Muse for each trial or initialize a trial from another trial's
trained adapter. These are experimental settings, not a proven optimum.

The [official A/B/C commands](OFFICIAL_MOBILE_QAT_PIPELINE.md#controlled-qat-lora-experiments-run-a-run-b-and-run-c)
can use this frozen bundle by replacing their `--input-dir "$INPUT"` with
`--prepared-input-dir /runs/preprocess_e2b_muse_01/augmented --augmentation none`.
The same frozen-data principle applies to 270M learning-rate comparisons, using
the separately prepared 270M bundle.

## Optional legacy startup mode

On a fresh ordinary launcher run, `--augmentation` (equivalent to
`--augmentation semantic`) still prepares raw input and contacts Muse before
student training. Use raw `--input-dir` (or Golden's `--source-run-dir`), not
`--prepared-input-dir`, for this mode. Its augmentation controls and safeguards
are unchanged; Golden's `--prepare-only` with semantic mode **does contact Muse**.

Startup mode continues directly into student training and does not pause to
stop an external teacher. Use a separate teacher host/GPU allocation if taking
that path. For sequential reuse of the same GPUs, prefer the standalone
preparer above. `rare_components` retains its existing row-resampling meaning
and is not an alternative way to label new synthetic sources.

## Artifacts, recovery and experiments

- `prepared/`: original preparation in the preprocessing output; never overwritten.
- `semantic_augmentation/donors.jsonl`: selected train-only donor sources.
- `semantic_augmentation/generated/`: teacher request outcomes, source reviews,
  raw Stage 3 records, accepted records, category coverage and hash manifests.
- `semantic_augmentation/`: source-filter, tokenizer and token-budget rejection
  evidence. Generation details are in `generation.log` and `generated/run.log`.
- `augmented/`: published original-plus-augmented train split, copied holdouts
  and updated manifest. This is the directory passed to `--prepared-input-dir`.
  `augmentation.json` records added rows/tokens, recipe and provenance.

Require a successful command and `augmentation_preparation_manifest.json`
with `status: "prepared"`; the presence of an `augmented/` directory alone is
not proof that final sealing and verification completed. The top-level receipt
records the final bundle's hashes. Stage logs are in
`logs/prepare.log`, `logs/augmentation.log`, and `logs/verify.log`.

For portable reuse, transfer the **whole** `augmented/` directory, including
`augmentation_source_manifest.json`, `augmentation_generation_manifest.json`,
`augmentation_donors.jsonl`, and `augmentation_accepted_genui.jsonl`. These
byte-preserved evidence files bind the original preparation, teacher generation,
donors and accepted Stage 3 targets. Import rejects missing or changed evidence;
copying only train/validation JSONL files is not frozen-data reuse.

Failed generation keeps its diagnostics; retry with a fresh output directory.
Official checkpoint resume
must omit the augmentation flag: it reuses the checkpoint's frozen dataset
(including earlier augmentation) and must not generate new data mid-resume.
With explicit `--resume-from-checkpoint`, omit both `--input-dir` and
`--prepared-input-dir` to reuse that saved, hash-bound dataset. A new
`--prepared-input-dir` override on resume is rejected; start a fresh experiment
to change data. Fresh runs still require an explicit source.
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
