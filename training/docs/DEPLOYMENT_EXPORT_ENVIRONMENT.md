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

The core converter pins are deliberate: LiteRT Torch 0.9.4, AI Edge Quantizer 0.9.0, LiteRT 2.2.0, converter 0.4.0, builder 0.17.0, Transformers 5.16.1, and the documented Torch 2.11.0 / TorchAO 0.17.0 pairing. These are **source/API/dependency-metadata checked**, not a claim that this complete environment, the four established real E2B exports, or optional W248 has been exercised here. Python 3.12 is the recommended deployment setup; the upstream runtime Linux wheel requires glibc 2.27 or newer. [LiteRT Torch release](https://pypi.org/project/litert-torch/0.9.4/), [TorchAO compatibility matrix](https://github.com/pytorch/ao/issues/2919), [LiteRT-LM API release](https://pypi.org/project/litert-lm-api/0.17.0/).

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
| W16 | Repository `weight_only_fp16.json` / AEQ `float_casting`; both experimental precision flags **false** | FP16; activations and KV cache remain FP32 |
| W8 | `dynamic_wi8_afp32` | INT8 |
| E2B W4 | Repository `gemma4_mixed48_b32_flat.json`, checked against upstream `gemma4_mixed48_b32` | INT4 present; INT8 permitted (mixed W4/W8) |
| E2B W248 (opt-in) | Repository `gemma4_dense_mixed248.json`; public output-scope API, dynamic channelwise PTQ | Role-checked packed INT2/INT4/INT8; no MTP section |
| 270M W4 | `dynamic_wi4b32_afp32` | INT4 |

W16, W4 and W248 require the full launcher's experimental-format
acknowledgment. A failed format is not silently replaced by W8 or counted as
passing. FP32 activations/shape constants are not counted as FP32 model weights.
The inspector follows constant-weight dequantization/cast/reshape/transpose
chains and checks physical fully-connected and embedding weight storage. Unknown
or mismatched weights fail closed. This is not a guarantee of every operator's
arithmetic precision or placement.

W248 is E2B-only and optional; omitting `--variants` keeps the existing
W32/W16/W8/W4 export set unchanged. It applies a public-API, channelwise
W2/W4/W8 PTQ policy to a verified dense checkpoint/`merged_hf`, not to an
existing `.litertlm`, and does not change source weights on disk. It is not the
official QAT graph, static-A8/scales contract, per-layer embedding group-256
layout, or an MTP export. Preflight and output inspection bind the exact recipe
hash, validate all 35 layers and physical packed widths/scales, and reject a
drafter section; they do not run GPU inference or quality evaluation. See
[experimental E2B W248 PTQ, commands and limitations](EXPORT_TRAINED_CHECKPOINT.md#experimental-e2b-w248-ptq).

### W16: weight casting, not whole-graph mixed precision

The current deployment/export-only entry points use
[`weight_only_fp16.json`](../configs/export/weight_only_fp16.json) for both E2B
and 270M W16. AI Edge Quantizer's public `float_casting` algorithm physically
stores FC/embedding constants as FLOAT16 and inserts dequantization to the
unchanged FLOAT32 graph. No calibration set is required. Both
`experimental_use_fp16` and `experimental_use_mixed_precision` are explicitly
false for W16. W32/W8 requests are unchanged; the E2B W4 recipe transport fix is
described below. This is **W16 storage with
FP32 activation/cache contracts**, not a promise of all-FP16 arithmetic, lower
KV-cache memory, or faster native inference. [Pinned float-casting implementation](https://github.com/google-ai-edge/ai-edge-quantizer/blob/v0.9.0/ai_edge_quantizer/algorithms/nonlinear_quantize/float_casting.py).

Why the previous E2B W16 request failed:

1. `experimental_use_fp16=True` controls embedding-input/cache types and bundle
   activation metadata; the HF loader still loads FLOAT32 weights. With recipe
   `none`, that flag alone does not produce W16 weight storage.
2. The additional whole-graph mixed-precision pass is a different operation.
   Gemma4's projection norm uses an `odml.rms_norm` composite, explicit FP32
   normalization, and FP32 scale tensors. The pass protects RMSNorm by casting
   its operands to FP32, skips existing CAST operations, and preserves its
   decomposition without normalizing the preserved function's argument types.
   An FP16 decomposition can therefore disagree with the FP32 composite input.
   The reported cleanup error at `per_layer_projection_norm / mark_tensor_2`
   shows exactly such an F32/F16 interface mismatch. We have not reproduced the
   complete remote E2B MLIR graph locally or claimed to repair the upstream
   mixed-precision optimizer itself.
3. That whole-graph pass is applied to the main text graph, not automatically
   to the token/per-layer embedding exports. Enabling it alone would still be
   insufficient to establish FLOAT16 storage for the complete E2B package.

The weight-only recipe avoids rewriting any RMSNorm composite/interface types.
The standard exporter passes the recipe through its text, token-embedder, and
additional per-layer-embedder conversion paths before bundling. It uses no
runtime monkeypatches, changes no installed package, and leaves the original
merged weights/provenance untouched. [Pinned loader and export paths](https://github.com/google-ai-edge/litert-torch/blob/v0.9.4/litert_torch/generative/export_hf/core/export_lib.py),
[FP16 input flag](https://github.com/google-ai-edge/litert-torch/blob/v0.9.4/litert_torch/generative/export_hf/core/exportable_module.py),
[mixed-precision pass](https://github.com/google-ai-edge/litert-torch/blob/v0.9.4/litert_torch/generative/export_hf/core/mu/mixed_precision.py),
[Gemma4 RMSNorm/projection](https://github.com/google-ai-edge/litert-torch/blob/v0.9.4/litert_torch/generative/export_hf/model_ext/gemma4/patch.py).

Preflight validates the recipe against the installed quantizer for **both FC
and embedding operators**, without loading weights. W16 copies the validated
recipe to `w16_quantization_recipe.json` in its fresh output folder and records
its SHA256 in `export_manifest.json`. Output validation rechecks that file as
well as the unchanged physical precision gate. An unconverted FLOAT32 matrix,
including an external embedding or unsupported GATHER lowering, still fails;
it is never labelled W16 merely because conversion returned successfully.

Focused regression tests perform real float casting of tiny TFLite FC and
embedding graphs, compare physical half-precision bytes and CPU numerical
outputs, check unchanged graph I/O types, and exercise recipe/provenance
failures. They do **not** convert a full E2B model or certify native GPU kernels.
See [retry only W16 from an existing merge](EXPORT_TRAINED_CHECKPOINT.md#retry-only-w16-from-an-existing-merged_hf).

### E2B W4: adapt the package mapping to the per-TFLite recipe API

`gemma4_mixed48_b32()` returns a mapping with three LiteRT-LM section keys:
`tf_lite_prefill_decode`, `tf_lite_embedder`, and `tf_lite_per_layer_embedder`.
The LiteRT Torch HF exporter instead calls `Quantizer.load_quantization_recipe`
separately for each TFLite graph. Passing that mapping directly to its
list-of-rules loader iterates string keys and raises
`TypeError: string indices must be integers, not 'str'`, wrapped as an invalid
recipe error. Constructing the recipe without loading it did not catch this
in the previous preflight. [Pinned Gemma4 recipe](https://github.com/google-ai-edge/ai-edge-quantizer/blob/v0.9.0/ai_edge_quantizer/recipe.py),
[pinned per-graph exporter](https://github.com/google-ai-edge/litert-torch/blob/v0.9.4/litert_torch/generative/export_hf/core/export_lib.py).

The repository now supplies
[`gemma4_mixed48_b32_flat.json`](../configs/export/gemma4_mixed48_b32_flat.json)
through the supported JSON-file recipe API. It preserves the upstream rules:

- Ordinary fully-connected matrices: symmetric INT4, block size 32.
- Fully-connected matrices matching `per_layer`: symmetric INT8, channelwise.
- Token and per-layer embedding tables: symmetric INT4, block size 32.

This is a deliberately limited compatibility adaptation, not an arbitrary
mapping flattener. The current graph sections use disjoint FC/embedding
operators, and both embedding sections have the same policy. Preflight checks
the expected sections, operator separation, equal embedding policies, exact
upstream rule order/content, and the installed recipe manager's selected
configs. If any of these assumptions change, it fails before conversion rather
than silently substituting a different quantization policy. Norm operations are
not targeted. W32/W8, 270M W4, and the W16 weight-casting path are unchanged.

Preflight takes the JSON path from the actual W4 `export_kwargs`, resolves it
through AEQ's public `recipe_utils.resolve_recipe`, then loads the result through
`RecipeManager.load_quantization_recipe`: the same file-resolution and rule-loader
sequence used by `Quantizer.load_quantization_recipe(str_path)` in the pinned
0.9.0 API. It rejects a bare recipe name or a file resolver that returns a mapping
instead of the checked rule list. [Pinned quantizer file loading](https://github.com/google-ai-edge/ai-edge-quantizer/blob/v0.9.0/ai_edge_quantizer/quantizer.py).

The validated recipe is copied to `w4_quantization_recipe.json` in the fresh
variant output folder. That exact destination is reloaded and checked **before
HF graph conversion starts**, logged as `E2B W4 recipe file verified`, and passed
as `export_kwargs.quantization_recipe`. Its path and SHA256 are recorded in
`export_manifest.json`. Validation rechecks the snapshot and the **unchanged
physical precision inspector**: actual
INT4/UINT4 matrix weights must be present, INT8 is allowed for this mixed format,
and all-INT8 or residual FLOAT32 matrix weights fail. FLOAT16 block-scale tensors
are not model matrix weights. No installed environment files are modified.

Tiny real-quantizer tests reproduce the old loader failure, compare flat and
section-specific policies, and verify physical INT4 ordinary FC/token/per-layer
embedding buffers plus the INT8 projection exception. They also exercise
changed upstream mappings, tampered recipe files, and rejected output precision.
An integration regression mocks only HF conversion/package bundling; the exact
snapshot path passed to the exporter is loaded and quantized with real AEQ on
tiny graphs. A separate test proves snapshot-loading failure stops before HF
conversion, even when the upstream factory succeeds.
See [retry only W4 from an existing merge](EXPORT_TRAINED_CHECKPOINT.md#retry-only-w4-from-an-existing-merged_hf).

### Local verification and test commands

Local verification of the combined W16/W4 fixes (2026-09-19): focused
W16/W4/export tests passed **81 tests with 2 skips**; the broader
export/resume/deployment/QAT/inspection regression suite passed
**469 tests with 4 skips** (PEFT and TensorBoard unavailable,
Windows symlink permission, and a POSIX-only test). Ruff and `git diff --check`
passed. Tiny real graph tests used the already-installed AI Edge Quantizer
0.8.0 and LiteRT 2.1.6 on this Windows host; the pinned 0.9.0 quantizer source/API
was separately checked. This does not establish execution of the Linux 0.9.4
exporter stack or of a full E2B conversion.

A subsequent isolated AI Edge Quantizer 0.9.0 run passed **80 focused
tiny-graph, recipe, W248, W4 and W16 tests**. It still did not run a full
Hugging Face conversion because `litert_converter` was unavailable in that
local environment. This is recipe and physical-storage regression evidence,
not a successful full-stack export, GPU result, or quality result.

To rerun the inexpensive focused tests from the repository root:

```bash
PYTHONPATH=training/src:training/scripts:dataset/src \
  python -m pytest training/tests/test_deployment_w16.py \
  training/tests/test_deployment_w4.py \
  training/tests/test_deployment_export.py -q -rs
```

The real tiny-graph tests require AI Edge Quantizer and LiteRT; run them in the
isolated exporter environment, not by installing converter dependencies over
the training environment. They do not import/download E2B weights.

The source preparation stage validates saved checkpoint hashes, positive optimizer-step provenance, original dense source hashes and tokenizer/template fingerprints. It copies the checked prompt metadata; E2B's LiteRT-compatible template must render identically to the training template on checked examples, including a bounded three-row prefix of the prepared training data. This check does not read the entire training JSONL into memory. A real, random-weight tiny CPU regression checks that the full Gemma4 wrapper survives the training loader, PEFT adapter save/reload, merge, and full checkpoint save/reload with an unchanged tensor-key inventory and exact merged tensor values. This is not a full E2B model export test. The native GPU runner additionally compares every Golden prompt's runtime token IDs with the HF-formatted input contract. The original checkpoint, data and tokenizer are never overwritten.

Every variant publishes `model.litertlm`, `package_inspection.json`, and `export_manifest.json`, including the artifact SHA-256, original merged-source binding, installed versions, actual precision evidence and the explicit status `exported_not_yet_evaluated`. Only subsequent completed Golden32 and Golden35 GPU evaluations supply quality results. A file named `.litertlm` or a successful converter exit alone is insufficient. [Official Hugging Face conversion guide](https://developers.google.com/edge/litert/conversion/pytorch/genai).
