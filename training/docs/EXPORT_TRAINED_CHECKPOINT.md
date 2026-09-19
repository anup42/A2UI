# Export all LiteRT-LM variants from an existing trained checkpoint

Use `training/scripts/export_checkpoint_litertlm.py` for **export only**:
it does not retrain, tune, regenerate/filter data, run Golden/Bixby evaluation,
initialize Vulkan, or require the native LiteRT runtime environment.

Supported inputs are the current **dense E2B LoRA SFT** and **Gemma 3 270M
full-SFT** checkpoints. Retained-scale/QAT checkpoints use their separate
pipelines; a base model name containing `qat` does not make this QAT export.
MTP assistants are not exported. The default variants remain W32, W16, W8 and
W4. E2B additionally has an explicitly selected experimental `w248` PTQ
variant; it is not included by default.

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
W16/W4/W248 require explicit experimental acknowledgement when executing.

To export only the experimental E2B W248 variant, add this option to the same
command:

```bash
--variants w248
```

To request it together with every established variant, specify the complete
list explicitly:

```bash
--variants w32 w16 w8 w4 w248
```

`--variants` accepts one or more names. Omitting it preserves the established
W32/W16/W8/W4 default. Execution with W248 requires all three of
`--profile e2b`, `--allow-experimental-formats`, and `--execute`; a planning run
omits only `--execute`. W248 is rejected for 270M.

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
  disabled. Allow enough host RAM and disk for merged weights, the selected packages
  and converter temporary files; H100 VRAM does not replace CPU RAM.
- Default `--cache-length 8192` must cover the saved prompt plus generation
  budgets. Increase it if the original run used a larger budget.

The output directory must be **new**, outside the source model, prepared data,
fit and checkpoint directories. Existing directories are never overwritten.
Do not edit provenance hashes or prompt contracts to force an old checkpoint
through a different training/export contract.

## Execution and results

The default six stages run **one at a time**:

```text
Exporter preflight -> verify/merge checkpoint -> W32 -> W16 -> W8 -> W4
```

When `--variants` is present, the same preflight and verified merge run first,
followed only by the requested conversion stages in the stated order. For
example, `--variants w248` runs preflight, merge and W248; adding W248 to the
explicit five-name list adds it after W4. The established W32/W16/W8/W4
requests and all checkpoint, tokenizer, prompt and resume-lineage provenance
checks are unchanged.

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
    export_w248.log                 # only when selected
  variants/
    w32/model.litertlm
    w16/model.litertlm
    w8/model.litertlm
    w4/model.litertlm
    w248/model.litertlm             # only when selected
```

Each variant also has `export_manifest.json` and `package_inspection.json`.
The converter inspects actual physical matrix-weight storage, and the launcher
rechecks package hashes and precision evidence before recording success. E2B
W4 is **mixed W4/W8**, whereas 270M W4 uses block-32 INT4. W32/W16 describe
weight storage precision, not model parameter counts.

E2B W248 is a different, experimental dynamic channelwise PTQ export. It is
validated as a mixed W2/W4/W8 artifact, but it is not an official mobile QAT
graph and must not be compared to W4 merely by its directory name.

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

## Experimental E2B W248 PTQ

W248 re-quantizes an ordinary, provenance-bound **dense merged Hugging Face
checkpoint** into new LiteRT/TFLite package sections. It does not modify the
checkpoint or `merged_hf` weights on disk. An existing `.litertlm` is an output
artifact, **not** a supported source for re-quantization: use the original
trained checkpoint through `export_checkpoint_litertlm.py`, or an existing
verified `merged_hf` through the lower-level commands below.

The public bit allocation used by this experiment is:

| E2B text role | W248 weight storage request |
|---|---|
| Token embedding and tied/output `lm_head` | W2 |
| MLP gate/up/down in layers 15-34 | W2 |
| Self-attention in layers 0-34 | W4 |
| MLP gate/up/down in layers 0-14 | W4 |
| Per-layer embedding table | W4 |
| Per-layer input gates and projections, including the model projection | W8 |

The exporter requires the 35-layer E2B text architecture and fails before
conversion if the model/config does not match it. The layer split and most role
widths are derived from Google's public Gemma 4 E2B mobile checkpoint policy.
The model-level per-layer projection is W8 in the released LiteRT-LM text
artifact inspected for this repository, although Google's public Transformers
config excludes that tensor; this distinction is recorded rather than presented
as a private Google recipe. [Google's public packed mobile checkpoint
configuration](https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-ct/blob/main/config.json),
[Google's public mobile-Transformers checkpoint
configuration](https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers/blob/main/config.json).

W248 uses AI Edge Quantizer 0.9.0's public **dynamic channelwise PTQ** recipe
schema. Rules match LiteRT operation output scopes, including distinct token
and per-layer embedder output names; they do not match generic physical weight
names such as `arith.constant`. The pinned API supports W2, W4 and W8 dynamic
integer-weight recipes, and the quantizer serializes two-bit weights as TFLite
`INT2`. [Pinned recipe API](https://github.com/google-ai-edge/ai-edge-quantizer/blob/v0.9.0/ai_edge_quantizer/recipe.py),
[pinned output-scope matcher](https://github.com/google-ai-edge/ai-edge-quantizer/blob/v0.9.0/ai_edge_quantizer/utils/tfl_flatbuffer_utils.py),
[pinned low-bit serialization](https://github.com/google-ai-edge/ai-edge-quantizer/blob/v0.9.0/ai_edge_quantizer/transformations/quantize_tensor.py).

This only establishes a supported experimental serialization route. W248 does
**not** recreate Google's official graph, QAT observations/scales, static INT8
activation contract, or the official per-layer embedding's group-size-256 W4
layout. It is not a claim about Google's private training/calibration process.
Aggressive W2 weight quantization and runtime-dependent dynamic activation
quantization can reduce quality, and W2 kernels may be slow or unsupported on a
target accelerator even when the package is valid.
No MTP assistant/drafter is added.

Before a full export, the following low-level probe screens the installed
exporter's required APIs and reports its versions; it does not enforce the
documented package versions by number. It also checks the exact JSON recipe
loader, all 35 layer boundaries, output-scope matching, tokenizer and
deployment-template compatibility without loading model weights:

```bash
/ABSOLUTE/PATH/TO/a2ui-export-094/bin/python -u \
  training/scripts/deployment_export.py probe \
  --profile e2b \
  --variants w248 \
  --model-dir /ABSOLUTE/PATH/TO/EXISTING_EXPORT/merged_hf \
  --cache-length 8192 \
  --report /ABSOLUTE/PATH/TO/REPORTS/w248_exporter_probe.json
```

When starting from the original checkpoint, first run this command without its
final `--execute` line to obtain a read-only plan. After reviewing the paths and
provenance, rerun the complete command:

```bash
python -u training/scripts/export_checkpoint_litertlm.py \
  --profile e2b \
  --variants w248 \
  --fit-dir /ABSOLUTE/PATH/TO/COMPLETED_RUN/fit \
  --exporter-python /ABSOLUTE/PATH/TO/a2ui-export-094/bin/python \
  --output-dir /ABSOLUTE/PATH/TO/NEW_w248_export \
  --allow-experimental-formats \
  --execute
```

For a standalone retry from an already verified merge, use a fresh output
directory:

```bash
/ABSOLUTE/PATH/TO/a2ui-export-094/bin/python -u \
  training/scripts/deployment_export.py convert \
  --profile e2b --variant w248 \
  --model-dir /ABSOLUTE/PATH/TO/EXISTING_EXPORT/merged_hf \
  --output-dir /ABSOLUTE/PATH/TO/NEW_w248_retry \
  --cache-length 8192
```

The explicit lower-level `convert` command does not accept or need
`--allow-experimental-formats`. It verifies the existing
`merged_hf/deployment_source.json` and source hashes before conversion. It does
not train, re-merge, alter source weights, quantize an existing `.litertlm`, run
other variants, add MTP, initialize a GPU, or evaluate quality.

Before conversion, the canonical recipe is copied to
`w248_quantization_recipe.json`; its exact bytes and SHA-256 are bound into the
variant manifest. After bundling, validation checks every physical FC and
embedding matrix against its unique role and expected layer set, verifies
packed INT2/INT4/INT8 buffer sizes, symmetric channelwise scales and zero
points, serialized FLOAT32 FC inputs/outputs and embedding outputs, all
35-layer coverage, both external embedding sections and the output head. The
manifest must also retain the experimental dense-PTQ, no-official-QAT, no-MTP
and not-GPU-tested labels. Validation requires exactly the target, token-embedder and per-layer-embedder
graphs and rejects any MTP drafter section. Unknown, ambiguous, float or
wrong-width target matrices fail closed. Counts are not forced to match the
official graph because dense-export fusion and prefill/decode aliases differ.
Repeated valid matrix uses are allowed; this is precision-policy and layer
coverage validation, not graph-identity or functional-equivalence certification.

A successful result remains `exported_not_yet_evaluated`. This export-only
workflow performs no native GPU evaluation and makes no speed, delegation or
quality claim. Before deployment, run native target-device/runtime checks and
compare the trained dense checkpoint and W248 package on Golden32, Golden35 and
Bixby50. Treat material regression, incomplete acceleration/delegation, failed
allocation, or unacceptable latency as a failed experiment rather than silently
substituting W248 for a working W8/W4 package.

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
