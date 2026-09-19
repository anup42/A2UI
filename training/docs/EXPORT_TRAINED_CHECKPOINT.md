# Export all LiteRT-LM variants from an existing trained checkpoint

Use `training/scripts/export_checkpoint_litertlm.py` for **export only**:
it does not retrain, tune, regenerate/filter data, run Golden/Bixby evaluation,
initialize Vulkan, or require the native LiteRT runtime environment.

Supported inputs are the current **dense E2B LoRA SFT** and **Gemma 3 270M
full-SFT** checkpoints. Retained-scale/QAT checkpoints use their separate
pipelines; a base model name containing `qat` does not make this QAT export.
MTP assistants are not exported.

## One command on the Linux training PC

Run from the cloned repository root, using the same training Python environment
as the original run. Replace the three paths:

```bash
python -u training/scripts/export_checkpoint_litertlm.py \
  --profile e2b \
  --fit-dir /ABSOLUTE/PATH/TO/COMPLETED_RUN/fit \
  --exporter-python /ABSOLUTE/PATH/TO/a2ui-export-094/bin/python \
  --output-dir /ABSOLUTE/PATH/TO/NEW_litertlm_exports \
  --allow-experimental-formats \
  --execute
```

Omit `--execute` to print a read-only plan. Planning checks local paths and
metadata. For resumed checkpoints it also hashes the prepared data and available
resume sources to verify the lineage. If retention removed a source, planning
checks the surviving checkpoint's file inventory and saved resume evidence.
These checks have a progress heartbeat; planning does
not regenerate, filter or tokenize the data. Execution additionally hashes the
selected weights and original base model, verifies prompt and tokenizer
contracts, probes exporter APIs, merges and converts. Neither a
successful plan nor exporter preflight proves a real model will convert or run.
W16/W4 require explicit experimental acknowledgement when executing.

`--fit-dir` is the directory containing `training_config.yaml`,
`preparation_report.json`, and `training/`. In a full deployment run it is
usually `<deployment-output>/<generated-run-id>_training/fit`, **not** the
outer deployment directory or a tuning trial. The default selected checkpoint is
`<fit-dir>/training/best_golden_checkpoint`.

For **270M**, change `--profile e2b` to `--profile 270m` and supply its own fit
directory. To intentionally export another saved checkpoint from the **same
training run or its verified resume lineage**, append:

```bash
--checkpoint /ABSOLUTE/PATH/TO/COMPLETED_RUN/fit/training/final_adapter
```

The 270M final full-model directory is `final_model`, not `final_adapter`.
An explicit checkpoint must retain its matching `training_metadata.json` and
hashed tokenizer/weight inventory. The script never silently falls back to a
different checkpoint.

## Resumed SFT checkpoints

Use the **same command** above after a legitimate training resume. Keep
`--fit-dir` pointing to the original fit directory, not the resume checkpoint.
The exporter resolves the actual config recorded in the selected checkpoint,
such as `training_config_resume_3500.yaml`, automatically:

- `preparation_report.json` remains bound to the original
  `fit/training_config.yaml` SHA256.
- The selected checkpoint remains bound to its actual resumed config SHA256.
  That exact config, including `resume_from_checkpoint`, is passed to merging
  and copied into `merged_hf/training_config.yaml`.
- Each **existing** resume source must have its original metadata, config, weight/tokenizer
  inventory, `trainer_state.json`, optimizer, scheduler and per-rank RNG files.
  A missing file, damaged directory or permission error never activates the
  deleted-source fallback. Multiple resumes are checked back through available
  sources; a missing source uses the separately labelled evidence policy below.
- Only the resume pointer may differ between configs. Model, data hashes, LoRA,
  training recipe, effective batch, and all other config settings must match.
  The saved resume-state metadata hash and optimizer step must also verify.

If a recorded resume config is no longer present, its byte-identical
`training_config.yaml` snapshot inside the checkpoint can be used. If a recorded
copy **exists but was modified**, export fails rather than hiding the change
by using another copy. Missing source evidence or changed contracts are not
bypassed; restore the original files from a backup, not hand-edited hashes.

### Source checkpoint removed by Trainer retention

The same export command automatically supports an already-saved checkpoint
whose entire numbered resume-source directory is absent. For example, a best
checkpoint at step 7000 can reference a deleted `checkpoint-3500` when it retains:

- `resume_state.verified: true`, a positive integer `global_step: 3500`, and a
  well-formed, non-placeholder `metadata_sha256` from the training-time check;
- matching resume paths in its metadata and its hash-bound resume config, with
  the path identifying `checkpoint-3500` in this run's still-accessible training
  output directory;
- the full unchanged resume contract, matching original/resumed configs apart
  from the resume pointer, and both prepared train/validation files whose hashes
  still match that contract;
- a surviving checkpoint at or beyond the recorded resume step, with its full
  weight/tokenizer inventory passing fresh hash checks.

No dummy checkpoint, replacement metadata, rewritten hash, or edited resume
pointer is created. The normal original-model and tokenizer/prompt checks still
run during merging. This is **export only**: resuming training still requires
the physical optimizer, scheduler, RNG, weights and provenance files.

The console warns when this fallback is used. Both export manifests record
`resume_lineage.verification_mode: saved_resume_evidence`,
`physical_chain_complete: false`, the missing source path, the surviving metadata
hash, the checked train/validation hashes, and the original saved resume record.
The missing hop explicitly has `source_files_rechecked: false` and
`source_metadata_rehashed: false`; available hops use `physical_source`.

**Evidence limitation:** a saved digest cannot be recalculated or authenticated
without the deleted metadata. This compatibility policy trusts the surviving
training-time verification record and validates its consistency with the files
that remain. It is not proof that deletion was caused by retention, nor a
cryptographic signature against a coordinated rewrite of saved provenance.
Keep source checkpoints/backups if fresh verification of all historical bytes
is required. An older retained best checkpoint that predates the resume step
still requires the physical source for export under this policy.

`checkpoint_export_manifest.json` and `merged_hf/deployment_source.json` record
both config hashes and the verified `resume_lineage`. No training provenance
files are rewritten. When invoking the lower-level
`deployment_export.py prepare` directly with a resumed config, `--preparation-config` can explicitly
identify the original `fit/training_config.yaml`.

## Required inputs and environment

- Keep the original base model and prepared dataset accessible at the absolute
  paths recorded in `training_config.yaml`. This includes the prepared prompt
  and tokenizer contracts. The merge checks them but does not rebuild the data.
- Keep the original fit configuration, preparation report, checkpoint metadata,
  weights and tokenizer files unchanged. A copied adapter alone is insufficient.
- Use the isolated, compatible CPU exporter environment in
  [DEPLOYMENT_EXPORT_ENVIRONMENT.md](DEPLOYMENT_EXPORT_ENVIRONMENT.md). Do not
  install converter dependencies over the working training environment.
- `--training-python /absolute/path/to/training-env/bin/python` can explicitly
  select the interpreter used for merge; otherwise it is the Python launching
  the script. Both interpreter paths must exist. Venv symlinks are preserved.
- Merge and conversion run on **CPU**, with CUDA hidden and model downloads
  disabled. Allow enough host RAM and disk for merged weights, four packages
  and converter temporary files; H100 VRAM does not replace CPU RAM.
- Default `--cache-length 8192` must cover the saved prompt plus generation
  budgets. Increase it if the original run used a larger budget.

The output directory must be **new**, outside the source model, prepared data,
fit and checkpoint directories. Existing directories are never overwritten.
Do not edit provenance hashes or prompt contracts to force an old checkpoint
through a different training/export contract.

## Execution and results

The six stages run **one at a time**:

```text
Exporter preflight -> verify/merge checkpoint -> W32 -> W16 -> W8 -> W4
```

The console streams child-process logs and shows a heartbeat every 10 seconds
(`--progress-seconds`). Each stage has a hard 48-hour deadline
(`--stage-timeout-seconds`), not an idle-output timeout. No automatic retries,
training resumes, inference fallbacks or package installations occur.

```text
<output>/
  checkpoint_export_manifest.json
  exporter_preflight.json
  merged_hf/
  logs/
    exporter_preflight.log
    merge.log
    export_w32.log
    export_w16.log
    export_w8.log
    export_w4.log
  variants/
    w32/model.litertlm
    w16/model.litertlm
    w8/model.litertlm
    w4/model.litertlm
```

Each variant also has `export_manifest.json` and `package_inspection.json`.
The converter inspects actual physical matrix-weight storage, and the launcher
rechecks package hashes and precision evidence before recording success. E2B
W4 is **mixed W4/W8**, whereas 270M W4 uses block-32 INT4. W32/W16 describe
weight storage precision, not model parameter counts.

At the end, a console table lists each variant's format, size, status and path.
The top-level manifest records source paths, commands, stage timings, logs and
verified artifact hashes. Successful completion is explicitly
`exported_not_evaluated`: **no quality score or native runtime success is
claimed**, and no TensorBoard evaluation entries are generated.

On failure or interruption, remaining stages do not start; the manifest records
the failed stage and previously completed exports/logs remain untouched. This
wrapper has no resume flag: use a new output directory for another full attempt.
For an individual failed format, the existing lower-level
`deployment_export.py convert` can use an already verified `merged_hf` and a
fresh variant output directory. Do not delete a successful export to retry.

## Retry only W16 from an existing `merged_hf`

From the updated repository root on the Linux training PC, replace the two
placeholder paths below. `--model-dir` must be the **existing merged_hf** with
its `deployment_source.json`, hashed dense weights, tokenizer and deployment
template, not the adapter or `best_golden_checkpoint` directory.

```bash
/group-volume/k.anup/envs/a2ui-export-094/bin/python -u \
  training/scripts/deployment_export.py convert \
  --profile e2b --variant w16 \
  --model-dir /ABSOLUTE/PATH/TO/EXISTING_EXPORT/merged_hf \
  --output-dir /ABSOLUTE/PATH/TO/NEW_w16_retry \
  --cache-length 8192
```

Use the original larger cache length if the run required more than 8192 tokens.
Use a **new/empty output directory**; leave previous failure evidence and
successful exports intact. This command hashes/verifies the existing merged source
and runs **only W16 conversion and physical inspection**. It does not retrain,
re-merge, process the dataset, convert other variants, initialize Vulkan, or
evaluate a Golden set. It does not update a previously failed top-level run's
status; its result is a standalone validated variant folder.

W16 now uses real FLOAT16 **weight storage**, with FLOAT32 activations, RMSNorm
and KV cache, via the repository's weight-only float-casting recipe. Do not add
`experimental_use_fp16=True` or `experimental_use_mixed_precision=True`, and do
not modify the installed exporter. The standalone `convert` command is already
an explicit choice of W16; the all-variant wrapper's
`--allow-experimental-formats` is not an option on this lower-level command.
See [W16 implementation and root cause](DEPLOYMENT_EXPORT_ENVIRONMENT.md#w16-weight-casting-not-whole-graph-mixed-precision).

Success produces `model.litertlm`, `package_inspection.json`,
`w16_quantization_recipe.json`, and `export_manifest.json`. The manifest must
show `actual_precision.verified: true` with only `FLOAT16` matrix-weight type
counts. FLOAT32 activations are expected. `exported_not_yet_evaluated` is not a
GPU-runtime or quality-test result.

## Retry only W4 from an existing `merged_hf`

Use the same existing merged model as above, with its unchanged
`deployment_source.json`, weights, tokenizer and deployment template. Replace
both placeholder paths, and choose a fresh W4 output directory:

```bash
/group-volume/k.anup/envs/a2ui-export-094/bin/python -u \
  training/scripts/deployment_export.py convert \
  --profile e2b --variant w4 \
  --model-dir /ABSOLUTE/PATH/TO/EXISTING_EXPORT/merged_hf \
  --output-dir /ABSOLUTE/PATH/TO/NEW_w4_retry \
  --cache-length 8192
```

Use the original larger cache length if needed. This command verifies the
existing merge and runs **only W4 conversion and physical inspection**: no
training, re-merge, data preparation, other formats, Vulkan, or Golden/Bixby
evaluation. Do not delete or overwrite the working W32/W8 packages. W16 and W4
retries can be run separately, one at a time, using their respective commands.
The all-variant wrapper's `--allow-experimental-formats` flag is not accepted
by this explicit lower-level `convert` command.

E2B W4 retains the upstream mixed policy: block-32 INT4 ordinary FC and embedding
weights, with channelwise INT8 for `per_layer` FC projections. The repository
adapts the upstream package-section mapping to the exporter's supported
per-TFLite JSON recipe API; do not patch the installed packages or replace it
with an all-INT8 recipe. See [W4 implementation and root cause](DEPLOYMENT_EXPORT_ENVIRONMENT.md#e2b-w4-adapt-the-package-mapping-to-the-per-tflite-recipe-api).

Before expensive conversion starts, confirm the console shows:

```text
E2B W4 recipe file verified: .../NEW_w4_retry/w4_quantization_recipe.json; sha256=...
```

The actual `export_kwargs.quantization_recipe` must be this JSON path, **not**
the bare string `gemma4_mixed48_b32`. The logical `recipe` label may still use
that upstream name; it is not the argument sent to the converter. If the bare
name is still being passed or the new message is absent, check that this fix is
present in the repository checkout used by the absolute script path in your
launch command. Updating another checkout or only the Python packages is not
enough. No changes to the working W16 implementation are needed.

Success produces `model.litertlm`, `package_inspection.json`,
`w4_quantization_recipe.json`, and `export_manifest.json`. The manifest must
show `actual_precision.verified: true` with INT4/UINT4 matrix weights present;
INT8 is permitted. FLOAT16 block scales and FLOAT32 activations are expected
and are not FLOAT32 model matrices. A standalone retry does not change a
previously failed top-level run's status. Its `exported_not_yet_evaluated`
status is not a claim that native GPU inference or quality testing passed.

To run training and all evaluations instead, use the
[full deployment workflow](GOLDEN_GPU_DEPLOYMENT.md). Its
`--skip-litert-evaluation` flag is a different mode: it still trains and tests HF
checkpoints before exporting. It is not the export-only command above.
