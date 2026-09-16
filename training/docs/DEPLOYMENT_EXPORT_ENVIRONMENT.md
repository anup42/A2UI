# Isolated environment for current-checkpoint LiteRT-LM export

This environment converts the **selected dense E2B LoRA or 270M full-SFT checkpoint** into public LiteRT-LM variants. It does not retrain the model, recreate Google's official retained-scale QAT graph, or export an MTP assistant. A dense seed whose model name contains `qat` is not proof that subsequent public PTQ conversion retains the official mobile QAT contract.

## Setup on the Linux training host

Keep three independent Python environments: existing CUDA training/HF evaluation; CPU conversion; LiteRT-LM GPU runtime. Do not replace the working training environment with converter dependencies. The converter uses CPU graph conversion; the H100 GPU training and native LiteRT GPU inference stages are separate.

From the cloned repository root, choose new environment paths:

```bash
python3.12 -m venv /group-volume/k.anup/envs/a2ui-export-094
EXPORT_PY=/group-volume/k.anup/envs/a2ui-export-094/bin/python
"$EXPORT_PY" -m pip install --upgrade pip
"$EXPORT_PY" -m pip install torch==2.11.0 torchao==0.17.0 \
  --index-url https://download.pytorch.org/whl/cpu
"$EXPORT_PY" -m pip install -r training/requirements-deployment-export.txt
"$EXPORT_PY" -m pip check

python3.12 -m venv /group-volume/k.anup/envs/a2ui-litert-017
RUNTIME_PY=/group-volume/k.anup/envs/a2ui-litert-017/bin/python
"$RUNTIME_PY" -m pip install --upgrade pip
"$RUNTIME_PY" -m pip install -r training/requirements-litertlm-runtime.txt
"$RUNTIME_PY" -m pip check
```

The core converter pins are deliberate: LiteRT Torch 0.9.4, AI Edge Quantizer 0.9.0, LiteRT 2.2.0, converter 0.4.0, builder 0.17.0, Transformers 5.16.1, and the documented Torch 2.11.0 / TorchAO 0.17.0 pairing. These are **source/API/dependency-metadata checked**, not a claim that this complete environment or all four real E2B exports have been exercised here. Python 3.12 is the recommended deployment setup; the upstream runtime Linux wheel requires glibc 2.27 or newer. [LiteRT Torch release](https://pypi.org/project/litert-torch/0.9.4/), [TorchAO compatibility matrix](https://github.com/pytorch/ao/issues/2919), [LiteRT-LM API release](https://pypi.org/project/litert-lm-api/0.17.0/).

CPU wheels avoid unnecessary CUDA converter allocations. TorchAO documents the CPU wheel index. Do not mix a wheel compiled for another Torch ABI; import failures may abort the process rather than raise a recoverable Python error. [TorchAO installation](https://github.com/pytorch/ao#installation).

Use the **absolute venv executable path**, not `readlink -f "$EXPORT_PY"`: resolving Linux's Python symlink to `/usr/bin/python` can bypass the virtual environment. Pass `--exporter-python "$EXPORT_PY" --runtime-python "$RUNTIME_PY"` to the full deployment launcher. Do not add `--system-site-packages` to these venvs.

Capture the resolved transitive dependency versions beside your run, because the requirements file pins core packages, not every transitive wheel:

```bash
mkdir -p /group-volume/k.anup/working_dir/a2ui_environment_records
"$EXPORT_PY" -m pip freeze --all > /group-volume/k.anup/working_dir/a2ui_environment_records/export-freeze.txt
"$RUNTIME_PY" -m pip freeze --all > /group-volume/k.anup/working_dir/a2ui_environment_records/runtime-freeze.txt
```

For reproducibility across hosts, also retain your platform/container image and wheelhouse or hash-locked requirements generated from this successfully probed environment. Do not call the core pins a full cross-platform lockfile.

## Linux Vulkan prerequisites required for native GPU testing

**The Python wheel and a working `nvidia-smi` are not sufficient.** The native
WebGPU delegate dynamically loads `libvulkan.so.1` and requires a usable hardware
NVIDIA Vulkan driver/ICD. Missing loader errors followed by `No adapters found`
can occur in an otherwise working CUDA training container. The updated launcher
checks these prerequisites before tuning/training, and again for native inference.

For an Ubuntu/Debian runtime image, have the image owner or administrator install
the userspace loader and diagnostic tools (these are manual setup commands, not
commands the pipeline executes):

```bash
sudo apt-get update
sudo apt-get install -y libvulkan1 vulkan-tools
```

`libvulkan1` is the loader; `vulkan-tools` supplies `vulkaninfo`. Neither package
provides the matching NVIDIA host driver by itself. Do **not** install a random
NVIDIA kernel/display driver inside a managed training container or copy an ICD
JSON whose referenced NVIDIA library is absent. The administrator must expose
the host driver and its matching ICD through the platform's NVIDIA container
runtime. [Ubuntu Vulkan loader package](https://packages.ubuntu.com/noble/libvulkan1),
[Ubuntu diagnostic tools](https://packages.ubuntu.com/noble/vulkan-tools).

For NVIDIA Container Toolkit jobs, request this driver capability set **when
creating the container**:

```text
NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
```

Keep GPU visibility restricted to the GPUs allocated by your scheduler. The
`graphics` capability exposes Vulkan dependencies; `compute` retains CUDA/OpenCL
and `utility` retains NVML/`nvidia-smi`. These values replace the capability list,
not append to it. Merely exporting this variable in an already-running
Jupyter/Kubernetes shell cannot mount libraries that were omitted at container
creation: request an updated image/pod from the platform administrator and
restart that container when necessary. The pipeline never changes GPU ownership,
host drivers, or scheduler isolation. [NVIDIA container capabilities](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#driver-capabilities).

Run these checks **inside the same container/environment used for inference**:

```bash
nvidia-smi
vulkaninfo --summary
"$RUNTIME_PY" -u training/scripts/run_litertlm_gpu.py --preflight \
  --report /group-volume/k.anup/working_dir/a2ui_environment_records/runtime-probe.json
```

At least one real NVIDIA GPU must appear and permit creation of a Vulkan compute
device/queue. A software adapter such as `llvmpipe`, `lavapipe`, or `SwiftShader`
does not satisfy this requirement. The code uses the loader directly in an
isolated child with a 30-second deadline, so `vulkaninfo` is optional diagnostics,
not a parser dependency. It requires no graphical window or display server.
If the loader is present but no NVIDIA adapter is visible, have the administrator
check the mounted ICD/library dependencies, driver compatibility, device access,
and container policy; do not mask the problem by enabling CPU fallback.

If the console stops at `runtime_preflight` with **`libvulkan.so.1: cannot open
shared object file`**, the loader is still missing from that container. A Git
update or `pip install` of the Python runtime does not install this OS library.
Use the Ubuntu/Debian commands above, or ask the managed-platform administrator
to add the loader and NVIDIA graphics exposure to the job image. If you have no
`sudo`, ask for the image change rather than installing a different host driver.
Only retry the deployment after the same-container native preflight passes.

When this fails **before tuning/training**, use a fresh output directory for the
retry; no optimizer has started in this new deployment run. `--resume-run` is
only for a run that previously completed full training and its checkpoint
evaluations. Earlier results remain in their original folders; `not reported`
in this run's final table means no score was produced here, not a zero score.

NVIDIA's data-center release notes list Vulkan support and HGX H100 platforms;
that is not certification of a particular managed H100 image or of LiteRT's
model kernels. The actual probe and each exported model's native evaluation are
still mandatory. The probe report deliberately records
`model_kernel_tested=false`, `webgpu_adapter_tested=false`, and
`gpu_affinity_verified=false`; actual inference subsequently verifies its NVIDIA
process allocation against the job's allowed UUIDs. [NVIDIA data-center API and
platform support](https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-550-54-15/index.html).

## Probe before spending GPU training time

```bash
"$EXPORT_PY" -u training/scripts/deployment_export.py probe \
  --profile e2b --model-dir /ABSOLUTE/PATH/TO/LOCAL_DENSE_MODEL \
  --cache-length 8192 \
  --report /group-volume/k.anup/working_dir/a2ui_environment_records/export-probe.json

"$RUNTIME_PY" -u training/scripts/run_litertlm_gpu.py --preflight \
  --report /group-volume/k.anup/working_dir/a2ui_environment_records/runtime-probe.json
```

Use `--profile 270m` for Gemma 3 270M. Export probing reads config and installed APIs/recipes without loading model weights. For E2B it also loads the actual local tokenizer and checks deployment-template parity on Unicode/whitespace, system-plus-user, and few-shot messages before training. Unsupported template behavior fails early. It checks actual `ExportableModuleConfig` fields, including the experimental FP16 option passed through the export API's extra keyword arguments; merely accepting `**kwargs` does not prove option support. [Pinned export configuration](https://github.com/google-ai-edge/litert-torch/blob/v0.9.4/litert_torch/generative/export_hf/core/exportable_module_config.py).

For E2B, the full dense source normally has top-level `model_type: gemma4` and a nested `gemma4_text` configuration. The current upstream Gemma4-specific exporter routes both the main text graph and required per-layer embedding graph by the top-level `gemma4` type. A standalone `gemma4_text` seed is rejected if the installed exporter does not explicitly provide those routes. **Do not fix this by relabeling the config** or placing dense weights in the official QAT graph. Standard Hugging Face snapshot symlinks into its blob cache are accepted for the original, content-hash-bound source. [Published E2B dense config](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized/blob/main/config.json), [pinned exporter routing](https://github.com/google-ai-edge/litert-torch/blob/v0.9.4/litert_torch/generative/export_hf/model_ext/exportables.py).

A passing prerequisite probe does **not** certify all model shapes, native GPU driver kernels, quantization accuracy, or available host RAM/disk. The actual converter, package inspector, merged-HF evaluator and native GPU evaluator remain required gates. Native GPU inference is not CUDA DDP; the current Python API does not select an H100 ordinal, and the built-in runner uses one verified native engine. The training/HF evaluation stages can still use all 2/4/8 allocated GPUs. [Official Python API](https://developers.google.com/edge/litert-lm/python), [pinned engine implementation](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.17.0/python/litert_lm/engine.py).

## Precision and provenance gates

| Requested variant | Conversion request | Required observed matrix-weight storage |
|---|---|---|
| W32 | `quantization_recipe=none` | FP32 |
| W16 | `none` plus `experimental_use_fp16=True` | FP16 |
| W8 | `dynamic_wi8_afp32` | INT8 |
| E2B W4 | `gemma4_mixed48_b32` | INT4 present; INT8 permitted (mixed W4/W8) |
| 270M W4 | `dynamic_wi4b32_afp32` | INT4 |

W16 and W4 require the full launcher's experimental-format acknowledgment. A failed format is not silently replaced by W8 or counted as passing. FP32 activations/shape constants are not counted as FP32 model weights. The inspector follows constant-weight dequantization/cast/reshape/transpose chains and checks physical fully-connected and embedding weight storage. Unknown or mismatched weights fail closed. This is not a guarantee of every operator's arithmetic precision or placement.

The source preparation stage validates saved checkpoint hashes, positive optimizer-step provenance, original dense source hashes and tokenizer/template fingerprints. It copies the checked prompt metadata; E2B's LiteRT-compatible template must render identically to the training template on checked examples, including a bounded three-row prefix of the prepared training data. This check does not read the entire training JSONL into memory. A real, random-weight tiny CPU regression checks that the full Gemma4 wrapper survives the training loader, PEFT adapter save/reload, merge, and full checkpoint save/reload with an unchanged tensor-key inventory and exact merged tensor values. This is not a full E2B model export test. The native GPU runner additionally compares every Golden prompt's runtime token IDs with the HF-formatted input contract. The original checkpoint, data and tokenizer are never overwritten.

Every variant publishes `model.litertlm`, `package_inspection.json`, and `export_manifest.json`, including the artifact SHA-256, original merged-source binding, installed versions, actual precision evidence and the explicit status `exported_not_yet_evaluated`. Only subsequent completed Golden32 and Golden35 GPU evaluations supply quality results. A file named `.litertlm` or a successful converter exit alone is insufficient. [Official Hugging Face conversion guide](https://developers.google.com/edge/litert/conversion/pytorch/genai).
