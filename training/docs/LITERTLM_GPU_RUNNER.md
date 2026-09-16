# Built-in LiteRT-LM GPU Golden runner

This runner executes actual `.litertlm` packages through **Google's pinned
`litert-lm-api==0.17.0`**, not Transformers, a stub, or an external runner template.
The evaluator is `training/scripts/evaluate_litertlm_on_golden.py --builtin-gpu`;
the isolated runtime worker is `training/scripts/run_litertlm_gpu.py`.

## Important capability boundaries

- Training and HF checkpoint evaluation can use 2/4/8 NVIDIA GPUs. The current
  LiteRT-LM Python `Backend.GPU` API has **no device-index/UUID selector**. Its GPU
  delegate is not CUDA DDP. The built-in runtime intentionally uses **one GPU
  engine**, with native device choice, and refuses `gpu_workers > 1`. Setting
  `CUDA_VISIBLE_DEVICES` does not establish OpenCL/WebGPU adapter isolation.
- Do not launch eight identical LiteRT workers and assume eight-GPU scaling:
  they may all choose one GPU. Truly independent LiteRT GPU workers require
  verified GPU-isolated containers or a runtime exposing an actual GPU selector.
- Linux NVIDIA evidence is required. A successful engine must retain explicit
  GPU selection, produce native output, and have a positive per-process GPU
  allocation reported by `nvidia-smi`. The actual GPU UUID is saved. If that
  evidence is unavailable (including inaccessible PID namespaces), evaluation
  fails rather than reporting CPU/unknown execution as a GPU success.
- For Jupyter/Kubernetes PID namespaces, the runner also accepts its **own**
  kernel-reported `/proc/self/status` `NSpid` identities. It saves the namespace
  chain and the PID matched by NVIDIA. It never guesses by process name, global
  GPU memory changes, or another process's allocation. Multiple conflicting PID
  matches fail closed. If the container's procfs hides the host PID reported by
  NVIDIA, ask the cluster administrator for host-PID-visible runtime execution;
  the runner cannot safely infer a hidden host identity.
- The orchestration layer may set `A2UI_LITERT_ALLOWED_GPU_UUIDS` to a JSON array
  of physical GPU UUIDs assigned to this job. The probe checks that inventory;
  every case rejects a native GPU choice outside the allowed set. This detects
  incorrect device selection; it does not pretend to provide missing affinity.
  Both deployment preflight and evaluation normalize complete bare CUDA UUIDs
  to `GPU-<uuid>` (already-prefixed IDs are not double-prefixed). Original CUDA
  profiles and scheduler masks stay unchanged, including on `--resume-run`.
  Missing, malformed or duplicate selected UUIDs fail before training; they are
  never silently dropped. MIG instance IDs/masks are rejected here rather than
  relabeled as physical GPUs. Native inventory and per-process allocation checks
  still apply; normalization does not fix missing Vulkan libraries.
- This is **not proof that every operator runs on GPU or that GPU utilization is
  optimal**. Native helper/sampling operations may use CPU; native logs are
  preserved. The runner never retries a failed GPU model with a CPU engine.
- Each exported variant must actually load and generate. W32/W16/W8/W4 package
  names do not establish kernel compatibility. This runtime does not fabricate
  exports or certify unsupported quantization/model combinations.
- No H100 or real model-package execution was available in the development
  environment. Mocked API/subprocess tests are not hardware certification.

## Separate runtime environment

From the repository root on the Linux training PC:

```bash
python3 -m venv /YOUR_RUNTIME_ENV
/YOUR_RUNTIME_ENV/bin/python -m pip install -r training/requirements-litertlm-runtime.txt
/YOUR_RUNTIME_ENV/bin/python -u training/scripts/run_litertlm_gpu.py \
  --preflight --report /YOUR_OUTPUT_DIR/litert_runtime_preflight.json
```

The prerequisite probe loads the pinned native shared library and checks its
Python API and NVIDIA inventory before expensive training. It also runs a
**30-second, isolated native Vulkan probe**: load `libvulkan.so.1`, create a
Vulkan 1.1 instance, enumerate physical devices, reject software/non-NVIDIA
adapters, and create a logical NVIDIA GPU device with a usable compute queue.
This catches the CUDA-only-container failure where `nvidia-smi` and Python
imports work but the WebGPU delegate later reports `No adapters found`.
It does not require a desktop, window, swapchain, `vkcube`, or display server.

The report includes `vulkan_compute_device_verified=true` and the Vulkan
device/queue evidence. `status=prerequisites_passed` explicitly includes
`model_kernel_tested=false`; nested `webgpu_adapter_tested=false` makes clear
that this is **not** a Dawn adapter-feature or model-shader test. No shader is
dispatched during this lightweight check. Actual delegate/format/kernel and
GPU UUID allocation checks remain required after export. H100 memory size or
successful CUDA training alone cannot establish these capabilities.

### Fix `libvulkan.so.1` missing / WebGPU `No adapters found`

The screenshot failure is a runtime image/driver prerequisite problem, not a
training loss, checkpoint, or hyperparameter problem. Install the distribution
Vulkan loader in the image and expose the host NVIDIA Vulkan ICD libraries to
the job. See [exact administrator setup and verification](DEPLOYMENT_EXPORT_ENVIRONMENT.md#linux-vulkan-prerequisites-required-for-native-gpu-testing).
The pipeline does not install OS packages, change host drivers, use Mesa CPU
rendering as a substitute, or silently retry inference on CPU. GPU engine
failures retain the native messages and actionable context in `runner.log` and
the final stage exception. A passed Vulkan probe is not a guarantee that every
experimental exported precision is supported by the installed LiteRT delegate.

## Evaluate a produced package

Use the ordinary training/evaluation Python (Transformers and scoring dependencies)
to format prompts and score predictions. `--runtime-python` only selects the
isolated native inference subprocess:

```bash
python -u training/scripts/evaluate_litertlm_on_golden.py \
  --builtin-gpu \
  --runtime-python /YOUR_RUNTIME_ENV/bin/python \
  --model /YOUR_EXPORTED_MODEL.litertlm \
  --model-config /YOUR_TRIAL/fit/training_config.yaml \
  --require-prepared-contract \
  --split /YOUR_TRIAL/prepared/golden32.jsonl \
  --required-rows 32 --max-rows 32 \
  --max-input-tokens 4096 --max-new-tokens 2048 \
  --output-dir /YOUR_FRESH_EVALUATION_DIR \
  --runtime-cache-dir /YOUR_COMPILED_GPU_CACHE \
  --tensorboard-root /tensorboard --run-id YOUR_RUN \
  --evaluation-name selected_w4_golden32 --metric-version v5_4
```

For Golden35, select the prepared `golden35.jsonl` and set both count options to
35. The training config must be the original hash-bound config from that trial,
not a modified copy. This command evaluates an already-exported package; it
does not train or perform conversion. Keep Golden35 out of trial selection.

## Fair prompts, bounded execution, evidence

1. The evaluator checks the saved preparation contract, contamination reservation,
   frozen Golden cohort and tokenizer binding when `--require-prepared-contract`
   is used. This is required by the current end-to-end deployment workflow.
2. The same ModelAdapter and tokenizer used for training/HF evaluation format
   every prompt. The worker receives prompt text, expected input token IDs and
   a prompt hash, **never the expected Golden completion**.
3. Before every prefill, the native input token sequence must equal the HF token
   IDs exactly. The pinned raw-session API automatically prepends its BOS even
   with prompt templates disabled, so the runner removes **exactly that one
   leading native BOS string** from the supplied text and verifies the resulting
   native BOS-plus-tokenized-text sequence against HF. This prevents double BOS;
   missing/different BOS or tokenizer/chat-template assets fail before scoring.
   There is no silent truncation or arbitrary prompt replacement.
4. Native `Engine(..., backend=Backend.GPU())` owns a warm model across cases.
   Every case gets a fresh raw session with `apply_prompt_template=False`, greedy
   sampling, a 2,048-token output limit and the explicit MTP setting. There are no
   tools or executable content in the requests. The existing quoted-text-aware
   `</a2ui>` stop policy cancels generation once the output envelope closes.
5. Results are flushed after **each case** to `runner_outputs.jsonl`. Native
   logs and phase/case progress are streamed to console and `runner.log`; progress
   prints every 10 seconds and per-case completion includes ETA. Native stalls
   are bounded by an external process supervisor, not just a Python thread.

Default deadlines are 1,800 seconds for engine load, 600 seconds per case and
7,200 seconds for the whole runtime worker. Override with
`--load-timeout-seconds`, `--case-timeout-seconds`, and
`--runtime-timeout-seconds`. A failed/expired child is terminated; partial records
and logs remain, and no CPU fallback or automatic failed-case retry occurs.
Use a fresh evaluation output directory when retrying. A compiled-artifact cache
may be reused by the native runtime; Golden predictions are never reused.
The configured cache base is automatically namespaced as
`<base>/<runtime-version>/<complete-package-SHA256>/`. Different variants named
`model.litertlm` cannot share stale compiled entries; Golden32 and Golden35 reuse
the cache for identical package bytes. Without an explicit cache base, an
evaluation-local `runtime_cache` base is used. The manifest records the effective
directory. No prior caches are deleted.

Successful output includes `predictions.jsonl`, `scored_predictions.jsonl`,
`aggregate_metrics.json`, `evaluation_result.json`, and
`external_runner_manifest.json`. The manifest contains the actual package SHA-256
and `gpu_execution` evidence; per-case raw completions, prompt/token hashes, input
token counts, timing, MTP policy, runtime version and GPU UUIDs are retained.
The streaming native API does not expose exact generated token IDs; the runner
does **not** invent exact EOS reasons or token-throughput measurements. TensorBoard
receives scores and observed scalar runtime measurements through the shared logger.

## Primary references checked for this implementation

- [Pinned Engine API (v0.17.0)](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.17.0/python/litert_lm/engine.py)
- [Pinned Backend/Session interfaces](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.17.0/python/litert_lm/interfaces.py)
- [Pinned streaming Session implementation](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.17.0/python/litert_lm/session.py)
- [Pinned raw-session BOS preprocessing](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.17.0/runtime/core/session_utils.cc)
- [Official Python usage](https://developers.google.com/edge/litert-lm/python)
- [NVIDIA nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
- [Linux proc status namespace PID semantics](https://man7.org/linux/man-pages/man5/proc_pid_status.5.html)
- [NVIDIA container graphics driver capability](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#driver-capabilities)
- [Khronos Vulkan logical device creation](https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateDevice.html)
- [Vulkan native ABI structures (header v1.3.290)](https://github.com/KhronosGroup/Vulkan-Headers/blob/v1.3.290/include/vulkan/vulkan_core.h)
