# Gemma 4 E2B mobile QAT remote-PC runbook

This is the hand-off contract for a fresh training machine and its Codex agent.
It is intentionally fail-closed: plan and preflight first, then start a new run
only after the retained-scale QAT validator passes. Never resume
`checkpoint-20500`, never use its adapter as initialization, and never bypass a
failed gate merely because teacher-forced loss appears good.

## The numerical rule that must not change

Training starts from the manifest-verified BF16 text reconstruction of
`google/gemma-4-E2B-it-qat-mobile-transformers`. For every deployment-mapped
W2/W4/W8 tensor, the packed mobile checkpoint's codes, scales, axis and grouping
are the numerical authority. With a zero LoRA adapter, QAT-on must reproduce the
official codes and scales exactly. Do not recompute weight scales with abs-max,
percentile calibration, or a generic public quantizer.

The production config contract is:

```yaml
model:
  mobile_qparams_contract: outputs/seeds/gemma4_e2b_mobile_dequantized_text_hf/mobile_qparams.json
qat:
  scale_mode: retained_mobile
  fixed_scale_required: true
  fixed_activation_scale_required: true
  effective_lora_only: true
  ste_gradient: clipped
```

The strict gate is:

```text
python training/scripts/validate_gemma4_mobile_scale_preserving_qat.py --config <resolved-config> --strict
```

It verifies retained W2/W4/W8 scale/code coverage, the hash-bound qparams
sidecar, and streamed zero-adapter packed-code identity. The portable launcher
then runs `train_sft.py --preflight-only` on the real model and fixed completion
rows to compare QAT-off versus zero-adapter QAT-on loss/top-token probes and to
repeat a short deterministic zero-adapter greedy generation before it permits
the training command. The QAT-on repeats must match each other and retain at
least the configured eight-token prefix from QAT-off. This is a liveness/parity
gate, not a claim that the untrained base already produces correct Express IR.
Checkpoint provenance remains a promotion requirement after training.
`validate_ai_edge_qat_numeric_contract.py` validates a different, abs-max public
converter contract and is not permission to start this mobile fine-tune.

## Files that must be available on the remote PC

Keep paths project-relative in configs. Large model and dataset outputs are
ignored by Git and therefore must be copied or rebuilt separately:

- the current A2UI checkout containing this launcher and the scale-preserving
  QAT implementation;
- `training/outputs/datasets/stage3_folder_90_10/train.jsonl` and `val.jsonl`;
- `training/outputs/datasets/golden100_stage3_eval/all.jsonl`, containing exactly
  100 unique held-out rows that are absent from the training sources;
- the official packed checkpoint at pinned revision
  `dd693ff40353f057ca5f07e945ad867f4afbf2ec` (`model.safetensors` and
  `config.json`);
- the exact official `.litertlm` package and retained-compiled parity report;
- the generated BF16 reconstruction under
  `training/outputs/seeds/gemma4_e2b_mobile_dequantized_text_hf`, including
  `mobile_training_seed_manifest.json`, `mobile_qparams.json`, and
  `mobile_qparams.safetensors`.

Never copy only the 10 GB dense shards without their manifests/contracts. Hash
verification is part of the gate. The canonical packed input identities are
encoded in `reconstruct_gemma4_mobile_training_seed.py`; do not substitute a
similarly named model.

## Environment setup

Linux is recommended for multi-GPU training. Use a separate virtual environment
from LiteRT conversion. Install a CUDA-enabled PyTorch build appropriate for the
host first, then the training overlay:

```bash
python3 -m venv .venv-gemma4-train
. .venv-gemma4-train/bin/activate
python -m pip install --upgrade pip
# Install the CUDA PyTorch wheel selected for this host from pytorch.org first.
python -m pip install -r training/requirements-gemma4-qat.txt
python -m pip freeze > training/runs/remote_environment_freeze.txt
```

PowerShell activation is:

```powershell
py -m venv .venv-gemma4-train
.\.venv-gemma4-train\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install the CUDA PyTorch wheel selected for this host from pytorch.org first.
python -m pip install -r training/requirements-gemma4-qat.txt
```

Verify the same interpreter that will launch training:

```text
python -c "import torch,transformers,peft; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count(), transformers.__version__, peft.__version__)"
nvidia-smi
```

CUDA must be available. Do not accept the automatic FP16 fallback for this
reproducibility run: the mobile seed and intended run are BF16, so use GPUs and a
PyTorch/CUDA stack that report BF16 support.

If Hugging Face access is gated, authenticate without pasting or committing a
token:

```text
hf auth login
```

## Reconstruct the seed when it was not copied

The reconstruction needs the exact packed checkpoint, config, and retained
compiled-parity report. Plan first; the plan creates no seed:

```text
python training/scripts/reconstruct_gemma4_mobile_training_seed.py --source-safetensors <packed-dir>/model.safetensors --source-config <packed-dir>/config.json --retained-compiled-report <evidence-dir>/gemma4-mobile-retained-compiled-parity.json --output-dir training/outputs/seeds/gemma4_e2b_mobile_dequantized_text_hf
```

Inspect `ready: true` and every identity check, then repeat with `--execute`.
Never reconstruct over an existing non-empty destination.

## Prepare datasets only when the verified outputs are absent

The training dataset comes from completed Stage 3 runs; training code must not
generate Stage 1/2/3 data:

```text
python training/scripts/prepare_dataset.py --config training/configs/datasets/stage3_folder_90_10.yaml --source-genui-dir <completed-stage3-root> --output-dir training/outputs/datasets/stage3_folder_90_10
```

Prepare Golden-100 from its separate immutable source, never from a duplicated
Golden-50 or training source:

```text
python training/scripts/prepare_dataset.py --config training/configs/datasets/golden100_stage3_eval.yaml --source-run-dir <immutable-golden100-run>
```

Review both manifests. The Golden-100 output must have exactly 100 accepted,
unique rows. Preserve and hash these artifacts; do not regenerate them midway
through an experiment.

## Portable launcher workflow

Use a unique run ID. The launcher is cross-platform Python and uses
`python -m torch.distributed.run`, so its torchrun module comes from the same
environment. It never resumes and refuses any existing run directory.

For the requested few-step wiring/export/device test, use this exact source
profile in every launcher command:

```text
training/configs/models/gemma4_e2b_mobile_seed_ir_qat_sft_smoke_3_steps.yaml
```

It performs exactly three optimizer updates (not three microbatches), logs each
update, and evaluates/saves once at step 3 on the full fixed Golden-100 set.
It is explicitly `quality_promotion_eligible: false` and
`serialization_validation_eligible: true`: a passing run may exercise the
retained-scale exporter and Android wiring, but it is not a model-quality
checkpoint. Omit this `--config` only when intentionally starting the reviewed
full production schedule.

1. Print the plan. This creates no directory and loads no model:

```text
python training/scripts/run_gemma4_mobile_qat.py --config training/configs/models/gemma4_e2b_mobile_seed_ir_qat_sft_smoke_3_steps.yaml --run-id e2b_mobile_qat_smoke_plan --num-gpus 4 --source-safetensors <packed-dir>/model.safetensors
```

To select a non-default healthy set, add `--gpu-ids 0,1,3,4`; its count must
equal `--num-gpus`. The resolved CUDA visibility is recorded in the plan and is
inherited by every gate and torchrun worker.

`--source-safetensors` is optional only when the absolute source path recorded
in `mobile_training_seed_manifest.json` still exists on this machine. A copied
seed normally records the old PC's path, so provide the new local path. The
launcher requires the pinned official SHA-256, records the selected file's size
and hash, passes that exact path to the strict validator, and rechecks it before
torchrun.

2. Run all strict preflights but do not train:

```text
python training/scripts/run_gemma4_mobile_qat.py --config training/configs/models/gemma4_e2b_mobile_seed_ir_qat_sft_smoke_3_steps.yaml --run-id e2b_mobile_qat_smoke_preflight --num-gpus 4 --source-safetensors <packed-dir>/model.safetensors --preflight
```

3. Read `training/runs/portable_gemma4_mobile_qat/e2b_mobile_qat_smoke_preflight/launch/preflight_report.json`
and every referenced log. Confirm all gates passed. Because preflight reserves
the run ID to keep its evidence immutable, use a new ID for the real run.

4. Explicitly launch a fresh run:

```text
python training/scripts/run_gemma4_mobile_qat.py --config training/configs/models/gemma4_e2b_mobile_seed_ir_qat_sft_smoke_3_steps.yaml --run-id e2b_mobile_qat_smoke_train --num-gpus 4 --source-safetensors <packed-dir>/model.safetensors --execute
```

The launcher writes one resolved config, launch plan, gate reports, TensorBoard
events, Golden-100 outputs and live training log under that unique run root.
`preflight_report.json` records the size and SHA-256 of every gate log and each
declared JSON report; a missing declared report fails the preflight even if its
command returned zero. Checkpoint provenance hashes this report.
Watch TensorBoard with the exact command printed in the plan. The headline tag
is `eval/golden100/v5_4_score`.

For the commands below, `<run_root>` is therefore
`training/runs/portable_gemma4_mobile_qat/e2b_mobile_qat_smoke_train`.

Do not use `run_gemma4_e2b_mobile_mtp.py --execute-training` for multi-GPU
training; that orchestration path launches plain Python. Do not use
`run_gemma4_e2b_torchrun.sh` without overriding its historical non-mobile
default. The portable launcher exists to avoid both mistakes.

## Stop/restart policy

The current SFT entrypoint has no verified resume interface: it calls
`trainer.train()` without `resume_from_checkpoint`, and the Golden-100 callback's
best state is not restored. Therefore:

- an interrupted run remains evidence and must not be deleted or overwritten;
- do not rerun the same ID and do not pass `checkpoint-20500`;
- diagnose the interruption, then launch a new ID from the official mobile seed;
- do not manually copy an intermediate Trainer checkpoint into
  `best_golden_checkpoint`.

Only the callback-created best adapter selected by
`generation_reward_v5_4_avg` is eligible for later merge. A successful training
process still does not authorize merge, quantization, MTP, Android deployment or
promotion; those remain separate reviewed stages.

Gemma 4 `retained_mobile` now has a dedicated, fail-closed exporter stage in
`run_gemma4_e2b_mobile_mtp.py`. It is plan-only by default. Use the exact
launcher-resolved training config and callback-created best checkpoint; never
substitute the source YAML, `final_adapter`, or an intermediate Trainer
checkpoint. Every merged/export directory and output file must be a fresh path:

```powershell
python training/scripts/run_gemma4_e2b_mobile_mtp.py `
  --training-config <run_root>/launch/resolved_training_config.yaml `
  --best-checkpoint <run_root>/best_golden_checkpoint `
  --base-litertlm <official.litertlm> `
  --merged-model-dir <fresh_export_root>/merged_best_hf `
  --exact-output-dir <fresh_export_root>/retained_scale_export `
  --export-report <fresh_export_root>/retained_scale_export/report.json `
  --output-litertlm <fresh_export_root>/retained_scale_export/gemma4_e2b_retained_scale.litertlm
```

Inspect that plan, the best-Golden metadata, and every bound artifact before
adding `--execute-merge --execute-retained-scale-export`. The stage must prove
the pre-step parity gates, exact 205-projection retained-qparams binding,
best-checkpoint/merge provenance, immutable official scale bytes and frozen
constants, and a passing exporter report before atomically promoting the final
`.litertlm`. The old `--execute-exact-topology-export`, public abs-max export,
and `--compose` routes are blocked for Gemma 4 `retained_mobile` checkpoints.

After the plan passes and serialization validation is explicitly approved, run
the same command with both execution flags (all paths must remain identical):

```powershell
python training/scripts/run_gemma4_e2b_mobile_mtp.py `
  --training-config <run_root>/launch/resolved_training_config.yaml `
  --best-checkpoint <run_root>/best_golden_checkpoint `
  --base-litertlm <official.litertlm> `
  --merged-model-dir <fresh_export_root>/merged_best_hf `
  --exact-output-dir <fresh_export_root>/retained_scale_export `
  --export-report <fresh_export_root>/retained_scale_export/report.json `
  --output-litertlm <fresh_export_root>/retained_scale_export/gemma4_e2b_retained_scale.litertlm `
  --execute-merge `
  --execute-retained-scale-export
```

If either output directory already exists, choose a new
`<fresh_export_root>`; do not use `--force`, delete evidence, or reuse a partial
export. A nonzero exit means no package is eligible for Android testing.

## What Codex must inspect during the run

At startup, stop immediately if the scale-preserving validator, streamed
zero-adapter code parity, QAT-off/QAT-on completion-loss/top-token gate, or
repeated deterministic greedy-generation probe fails. A very high
initial loss, NaN/Inf, explosive gradient norm, severe W2/W4 code collapse, or
near-total zero/saturation occupancy is a numerical failure, not normal warmup.

At every evaluation, confirm:

- all Golden-100 rows were generated and scored;
- strict Express validity and v5.4 score, not eval loss alone;
- TensorBoard received `eval/golden100/v5_4_score`;
- logged loss and gradient norms remain finite and bounded.

The current trainer does not scan all 205 merged matrices for saturation at
every evaluation; do not claim that it does. The immutable scales themselves
are hash-bound, clipped STE limits further outward gradients, and the reset
learning rate is conservative. The dedicated retained-scale exporter performs
the selected merged-adapter code/scale and provenance gates; its report must
pass before the package can proceed to target-only device validation.

After a clean training completion, confirm that `best_golden_checkpoint`
contains local training metadata binding its adapter bytes, seed, qparams
contract, resolved-config hash and parity reports. The Golden callback can write
or replace the best adapter during evaluation before final metadata is copied;
therefore an interrupted best directory without that local metadata is evidence
only and is not eligible for merge, export or promotion.

Keep the released MTP section byte-exact and MTP disabled while validating the
new target. Do not enable or train MTP until the retained-scale exporter report,
target-only semantic checks, and Android GPU target-only gates all pass. Full
`LITERT_CL` delegation proves GPU execution, not model correctness.

First run the host package/graph inspection. This is a structural gate only; it
does not prove that the fine-tuned target generates valid Express IR:

```powershell
python training/scripts/validate_litertlm_mtp_gpu.py `
  <fresh_export_root>/retained_scale_export/gemma4_e2b_retained_scale.litertlm `
  --inspect-graphs `
  --official-artifact <official.litertlm> `
  --output <fresh_export_root>/retained_scale_export/host_validation.json
```

## Minimum Android target-only strict-Express gate

Do not use `android/tools/deploy_gemma4_e2b_trained.ps1` for this gate: that
helper installs the older `gemma-4-e2b-ir-trained-int4.litertlm` profile, while
the strict trained-Express probe resolves
`gemma-4-e2b-trained-express-int4.litertlm`. Use the isolated checked-in
`judgeCapture` build so an installed consumer app and its data remain untouched.
The checked-in wrapper builds and replacement-installs only that isolated test
package, hashes the host and device copies, runs synchronous GPU generation with
`mtp=false`, and fails unless strict Express decoding, canonical validation,
response-fact coverage, and native rendering all pass:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File android/tools/validate_gemma4_e2b_trained_express.ps1 `
  -ModelPath <fresh_export_root>/retained_scale_export/gemma4_e2b_retained_scale.litertlm
```

Pass `-Serial <adb-serial>` when more than one Android device is connected. The
long form below is equivalent and is retained as an auditable fallback; it does
not enable MTP:

```powershell
Set-Location android
.\gradlew.bat -PandroidTestBuildType=judgeCapture `
  :app:installJudgeCapture :app:installJudgeCaptureAndroidTest
Set-Location ..

$validationPackage = "com.samsung.genuicraft.judgecapture"
$runner = "$validationPackage.test/androidx.test.runner.AndroidJUnitRunner"
$model = (Resolve-Path -LiteralPath "<fresh_export_root>/retained_scale_export/gemma4_e2b_retained_scale.litertlm").Path
$deviceDir = "/sdcard/Android/data/$validationPackage/files/on_device_models"
$deviceModel = "$deviceDir/gemma-4-e2b-trained-express-int4.litertlm"

adb shell mkdir -p $deviceDir
adb push $model $deviceModel
$hostHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $model).Hash.ToLowerInvariant()
$deviceHash = ((adb shell sha256sum $deviceModel) -split "\s+")[0].ToLowerInvariant()
if ($hostHash -ne $deviceHash) { throw "Host/device model SHA-256 mismatch" }

$instrumentation = adb shell am instrument -w -r `
  -e mtp false `
  -e class com.samsung.genuicraft.Gemma4E2bTrainedExpressRawProbeTest `
  $runner 2>&1
$instrumentation
if (($instrumentation -join "`n") -notmatch "OK \(1 test\)") {
  throw "Target-only trained-Express instrumentation failed"
}

$reportText = adb shell run-as $validationPackage cat `
  files/result_gemma4_e2b_trained_express_raw_probe/gpu_target_only-report.json
$report = $reportText | ConvertFrom-Json
$required = @(
  $report.starts_with_a2ui,
  $report.strict_decode_succeeded,
  $report.strict_validation_succeeded,
  $report.response_fact_coverage_succeeded,
  $report.native_render_succeeded
)
if (@($required | Where-Object { $_ -ne $true }).Count -gt 0) {
  throw "Target-only strict-Express semantic gate failed: $reportText"
}
```

This probe uses LiteRT-LM's synchronous final target message with GPU,
temperature 0, top-k 1, and `mtp=false`. It must produce strict A2UI Express,
retain the authoritative response facts, and render a native surface. It is the
minimum semantic gate; it is not a throughput comparison.

After the strict target-only semantic probe passes, run the paired official vs.
trained target-only GPU benchmark. The isolated test APK installed above already
contains the required probe:

```powershell
python training/scripts/benchmark_android_litertlm_gpu_parity.py `
  --official <official.litertlm> `
  --candidate <fresh_export_root>/retained_scale_export/gemma4_e2b_retained_scale.litertlm `
  --output-dir <fresh_export_root>/android_gpu_target_only `
  --app-package com.samsung.genuicraft.judgecapture `
  --runner com.samsung.genuicraft.judgecapture.test/androidx.test.runner.AndroidJUnitRunner `
  --max-num-tokens 2048 `
  --output-tokens 64 `
  --warm-runs 3 `
  --top-k 1 `
  --top-p 1 `
  --temperature 0 `
  --seed 42 `
  --prompt "Write exactly one hundred numbered words."
```

If a serial was needed above, add `--serial <adb-serial>`. Require
`comparison.overall_pass=true` in
`android_gpu_target_only/android_litertlm_gpu_parity_report.json`. Full
`LITERT_CL` delegation confirms that the relevant graphs executed through the
Android GPU delegate; the strict probe above is what establishes Express
semantics.

Only after both target-only gates pass may MTP be enabled. The official MTP
section is used byte-for-byte; this run does not train or replace the drafter:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File android/tools/validate_gemma4_e2b_trained_express.ps1 `
  -ModelPath <fresh_export_root>/retained_scale_export/gemma4_e2b_retained_scale.litertlm `
  -Mtp `
  -SkipBuildInstall

python training/scripts/benchmark_android_litertlm_gpu_parity.py `
  --official <official.litertlm> `
  --candidate <fresh_export_root>/retained_scale_export/gemma4_e2b_retained_scale.litertlm `
  --output-dir <fresh_export_root>/android_gpu_mtp `
  --app-package com.samsung.genuicraft.judgecapture `
  --runner com.samsung.genuicraft.judgecapture.test/androidx.test.runner.AndroidJUnitRunner `
  --mtp `
  --max-num-tokens 2048 `
  --output-tokens 64 `
  --warm-runs 3 `
  --top-k 1 `
  --top-p 1 `
  --temperature 0 `
  --seed 42 `
  --max-mtp-success-rate-drop 0.10 `
  --mtp-max-decode-overshoot 4 `
  --prompt "Write exactly one hundred numbered words."
```

Finally bind both device reports into one host validation report:

```powershell
python training/scripts/validate_litertlm_mtp_gpu.py `
  <fresh_export_root>/retained_scale_export/gemma4_e2b_retained_scale.litertlm `
  --inspect-graphs `
  --official-artifact <official.litertlm> `
  --target-only-device-report <fresh_export_root>/android_gpu_target_only/android_litertlm_gpu_parity_report.json `
  --device-report <fresh_export_root>/android_gpu_mtp/android_litertlm_gpu_parity_report.json `
  --output <fresh_export_root>/retained_scale_export/final_validation.json
```

The GPU+MTP IR Demo test may be run after these gates. A three-step smoke model
validates training/export/runtime wiring only; it is never an accuracy or
quality-promotion result.

## Paste-ready prompt for Codex on the training PC

```text
You are on the GPU training PC in the A2UI repository. Prepare and, only after I
explicitly approve the printed preflight evidence, launch one fresh Gemma 4 E2B
mobile retained-scale QAT run.

Read AGENTS.md, training/docs/gemma4_mobile_qat_remote_pc_runbook.md,
training/docs/gemma4_e2b_qat_mtp_knowledge.md, the selected training YAML, and
the scale-preserving validator completely. Inspect git status and preserve all
unrelated files, datasets, model shards, checkpoints, Android data and prior
runs. Do not delete, overwrite, resume, merge, export, deploy, or train MTP.
Never use checkpoint-20500.

First verify the exact Git commit, CUDA-enabled PyTorch, BF16 GPU support,
Transformers/PEFT versions, the manifest-verified mobile BF16 seed, retained
mobile qparams contract, prepared 90/10 dataset, and a separate immutable exact
Golden-100. Do not use validate_ai_edge_qat_numeric_contract.py as the mobile
scale gate. The mandatory strict gate is
training/scripts/validate_gemma4_mobile_scale_preserving_qat.py and it must
prove zero-LoRA official code/scale identity. The launcher's mandatory
`model_numeric_preflight` then proves QAT-off/QAT-on initial-loss and at least
90% fixed top-token agreement, plus repeatable nontrivial zero-adapter greedy
generation with an eight-token QAT-off/QAT-on common prefix on the real model,
without creating an optimizer or training.

Use training/scripts/run_gemma4_mobile_qat.py with the explicit
training/configs/models/gemma4_e2b_mobile_seed_ir_qat_sft_smoke_3_steps.yaml
config and a unique run ID for this requested three-step wiring check. Run it
with --source-safetensors pointing to this PC's exact official packed
model.safetensors (omit it only if the manifest path is verified local), without
--execute first, and show me the plan. Then run --preflight only with the same
packed source and summarize every gate and log path. Stop and ask me before
--execute. Do not bypass a missing or failed gate. If I approve, use a new unique
run ID with the same smoke config, --execute, and the same packed source; monitor the live log and
TensorBoard v5.4 tag. Stop on NaN/Inf,
explosive initial loss/gradients, code collapse, failed Golden-100 generation,
or scale/parity drift. Training completion is not permission to merge or
quantize. Report the best Golden-100 checkpoint and provenance, plus all
remaining target-only/export/device gates, without claiming Android accuracy or
MTP speed. Treat this bounded result as serialization-validation eligible but
never model-quality-promotion eligible. If I separately approve serialization validation, first run the
Gemma 4 mobile MTP pipeline plan-only with the launcher's resolved training
config, callback-created best checkpoint, exact official base LiteRT-LM, and
fresh merged/export/report/output paths. Execute only with both
--execute-merge and --execute-retained-scale-export after the plan passes.
Reject the legacy exact-topology, public abs-max, and compose routes. Preserve
the official MTP section byte-for-byte and keep MTP disabled until target-only
semantic and Android GPU gates pass.
```
