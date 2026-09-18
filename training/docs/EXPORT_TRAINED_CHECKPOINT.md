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
small metadata files; execution additionally hashes weights, verifies prompt
and tokenizer contracts, probes exporter APIs, merges and converts. Neither a
successful plan nor exporter preflight proves a real model will convert or run.
W16/W4 require explicit experimental acknowledgement when executing.

`--fit-dir` is the directory containing `training_config.yaml`,
`preparation_report.json`, and `training/`. In a full deployment run it is
usually `<deployment-output>/<generated-run-id>_training/fit`, **not** the
outer deployment directory or a tuning trial. The default selected checkpoint is
`<fit-dir>/training/best_golden_checkpoint`.

For **270M**, change `--profile e2b` to `--profile 270m` and supply its own fit
directory. To intentionally export another saved checkpoint from the **same
training configuration**, append:

```bash
--checkpoint /ABSOLUTE/PATH/TO/COMPLETED_RUN/fit/training/final_adapter
```

The 270M final full-model directory is `final_model`, not `final_adapter`.
An explicit checkpoint must retain its matching `training_metadata.json` and
hashed tokenizer/weight inventory. The script never silently falls back to a
different checkpoint.

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

To run training and all evaluations instead, use the
[full deployment workflow](GOLDEN_GPU_DEPLOYMENT.md). Its
`--skip-litert-evaluation` flag is a different mode: it still trains and tests HF
checkpoints before exporting. It is not the export-only command above.
