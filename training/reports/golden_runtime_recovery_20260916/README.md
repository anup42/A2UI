# LiteRT Vulkan failure, explicit recovery and concise results

Date: 2026-09-16. Scope: the supplied `w32_golden32` failure screenshot,
current dense E2B/270M deployment workflow, result tables and TensorBoard detail.

## Evidence and cause

The native log says `Couldn't load Vulkan: libvulkan.so.1`, followed by
`Failed to initialize WebGPU environment` and `No adapters found`. The failure
is during W32 LiteRT engine creation, after the workflow reached training,
checkpoint evaluation, merge and W32 export. It is not a CUDA training OOM.
The screenshot command does not request `--tune`, so that invocation should not
be described as having run hyperparameter trials.

The old prerequisite probe checked installed APIs/native library import and
NVIDIA inventory, but not the lazily loaded Vulkan backend. CUDA/NVML can work
in a compute-only container without Vulkan loader/graphics driver exposure.

## Changes

- Probe the actual Vulkan loader, physical NVIDIA adapter, compute queue,
  logical-device creation and queue readiness in an isolated, 30-second child
  **before** training. Reject software-only adapters. Preserve actual native
  errors in the propagated failure message. No automatic CPU fallback.
- Add explicit `--resume-run` for **completed-training** deployment runs. Verify
  retained nested training evidence, merged weights, exported precision and
  evaluation records; reconstruct comparisons from bound files, not summary
  dictionaries. Rerun prerequisite probes; reuse finished training/exports/tests.
  Failed/unfinished deployment stages use fresh `recovery/attempt_NNNN/` paths.
  Previous partial logs and artifacts are not deleted. Concurrent writers fail
  against a kernel lock. Original training hardware HParams stay distinct from
  recovery evaluation hardware.
- Print/save all planned HPO trials and all 14 checkpoint/merged/LiteRT versus
  Golden32/35 result slots on success **and failure**. Show v5.4 reward, strict
  validity, row counts and the separate 31-unique-source Golden32 selection
  metric. Missing scores are unknown, not zero or successful. Golden35 still
  never selects a trial/checkpoint/precision.
- Default `--tensorboard-detail minimal`: retain meaningful training, validation,
  Golden quality/validity and runtime/HParams charts; suppress huge JSON text
  events and duplicate Trainer Golden metric streams. `full` is available for
  debugging. Complete aggregates, predictions, metadata and JSON evidence remain.
  Old TensorBoard event files are not erased by changing the setting.

## Training-host action still required

The repo cannot install system libraries or change mounts on the remote MLP
container. Install `libvulkan1` (`vulkan-tools` for diagnostics) in the runtime
image and ensure the host NVIDIA Vulkan ICD/libraries are exposed. NVIDIA
Container Toolkit requires `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`
at container creation. Setting it in an already-running terminal does not mount
missing libraries. The administrator may need to recreate the pod/container.
Preserve scheduler-assigned GPU isolation. Do not install host NVIDIA drivers
inside the container or accept software Vulkan as GPU success.

After the same-container native preflight passes, rerun the **original command
and output directory** with `--resume-run --tensorboard-detail minimal`. Do not
change model/data/training/prompt/Golden settings during recovery.

See [full run/recovery commands](../../docs/GOLDEN_GPU_DEPLOYMENT.md) and
[runtime image setup](../../docs/DEPLOYMENT_EXPORT_ENVIRONMENT.md).

## Verification and remaining limits

Final integrated suite on the current shared checkout: **1,205 passed,
3 skipped, 3 warnings in 402.49 seconds**. No test failures remain. This includes
the new recovery, result-table and lean TensorBoard tests plus real tiny CPU
model/PEFT and TensorBoard roundtrips. Warnings were the existing tiny PEFT
fixture's absent base config and CPU-only pinned-memory fixtures; an additional
SWIG deprecation message appeared at interpreter shutdown.

Focused Vulkan/native tests: **89 passed**. Initial deployment/recovery tests:
**31 passed, 1 skipped**, followed by expanded recovery coverage in the full run.
Scoped Ruff, affected-code critical lint, CLI help for deployment/tuning/native
preflight, and whitespace checks passed. The full test suite command was:

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
& tmp/tensorboard_validation_20260914/Scripts/python.exe -m pytest training/tests -q
```

Independent review found and corrected recovery summary/evidence mismatches
and separation of original training hardware from recovery evaluation hardware.
Regression coverage rejects changed packages/merged weights/result summaries,
changed semantic options, concurrent recovery and incomplete training; multiple
failed deployment attempts preserve their earlier artifacts.

These are CPU/mock integration tests on Windows, not a real H100/Dawn/model
kernel test. A successful Vulkan compute-device probe is a stronger prerequisite,
not proof that every W32/W16/W8/W4 model kernel will run. The probe explicitly
records `webgpu_adapter_tested=false`, `model_kernel_tested=false`, and no native
GPU affinity claim; actual package execution remains a required gate. No real
model quality scores are invented. Unrelated Android/dataset edits in the shared
workspace are not part of this change.

Primary references: [NVIDIA container capabilities](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#driver-capabilities),
[Vulkan loader/driver discovery](https://vulkan.lunarg.com/doc/view/latest/linux/LoaderDriverInterface.html),
[NVIDIA data-center API/platform support](https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-550-54-15/index.html).
