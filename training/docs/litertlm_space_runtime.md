# LiteRT-LM evaluation on the H100 Space

The Space at `10.13.196.24` has an isolated Python runtime at
`/group-volume/k.anup/working_dir/golden_eval_speed_20261001/litert/venv`.
It contains `litert-lm-api==0.17.0`, independent of training and export environments.
Source `litert/activate.sh` before launching its Python interpreter. Evaluation and
scoring still use the training interpreter; `--runtime-python` selects the isolated
native inference worker.

## Problems fixed

- CUDA visibility did not imply Vulkan availability. The image lacked both the
  Vulkan loader and NVIDIA graphics libraries required by the Linux LiteRT-LM
  WebGPU backend. User-local copies of the loader and the exact running driver
  version, 535.129.03, restore hardware Vulkan. A headless ICD using
  `libEGL_nvidia.so.0` works; the package's GLX ICD fails in this container.
- NVML reports zero MiB for every process in this Space. The runner now accepts
  the exact process's NVIDIA graphics context registration (`G` or `C+G`) when
  allocation counters are unavailable. It verifies the physical GPU UUID and
  records that allocation size is unverified. It never attributes another PID's
  context, infers the GPU from a memory delta, or claims all operators use GPU.
- FP16 execution damages the retained-scale QAT package's output. The SDK already
  mitigates this with `GpuFp32ModelCache`. Python evaluation now explicitly requests
  FLOAT32 text activations, matching Android's setting. Model weights and packages
  remain unchanged. Compiled caches separate runtime version, model SHA-256, and
  activation precision. `--activation-dtype float16` is an explicit diagnostic.
- `--gpu-decode-steps-per-sync 8` reduces CPU/GPU synchronization. The engine remains
  loaded across the entire split; native token limits and quote-aware A2UI stopping
  remain active. Closing-tag detection avoids scanning the whole output unless a
  newly received chunk could contain that tag.
- Closing-tag cancellation drains the asynchronous iterator to its final or
  CANCELLED callback before freeing the session. Native callbacks must finish
  before the next case; freeing a still-active session can crash later samples.
- Space benchmark launchers use a SHA-256-verified local-disk copy of the package
  and local compiled caches. The group volume is a network filesystem; keep
  native memory-mapped files and compiler intermediates on stable local storage.

## Recreate the environment

`training/scripts/setup_litertlm_runtime.sh` creates a venv and validates the native
runtime. It can install from PyPI or copy the exact pinned package from an existing
environment when Space network access is unavailable. Optional Debian packages
are extracted under the runtime directory, without sudo, an installer, or system
driver changes. NVIDIA package and running kernel-driver versions must match.

```bash
bash training/scripts/setup_litertlm_runtime.sh /path/to/working_dir/litert \
  --python /path/to/training-env/bin/python \
  --offline-runtime /path/to/existing-litert-017-env \
  --vulkan-loader-deb /path/to/libvulkan1.deb \
  --nvidia-gl-deb /path/to/libnvidia-gl-535.deb
source /path/to/working_dir/litert/activate.sh
```

The matching graphics package used on this Space came from NVIDIA's official
[Ubuntu 22.04 repository](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libnvidia-gl-535_535.129.03-0ubuntu1_amd64.deb).
Its SHA-256 is
`91019b146f60a207c9e920f3a148f16631052e20f5577e598e495ce84cbe6c5a`.
The preflight checks a real NVIDIA Vulkan device and compute queue. A successful
preflight does not replace the subsequent model-kernel test.

## Evaluate and score

```bash
source /path/to/working_dir/litert/activate.sh
export A2UI_ANDROID_REPAIR_RUNTIME=/path/to/android_repair_runtime.json
export A2UI_LITERT_ALLOWED_GPU_UUIDS='["GPU-allocated-physical-uuid"]'
/path/to/training-env/bin/python training/scripts/evaluate_litertlm_on_golden.py \
  --model /path/to/generated.litertlm --builtin-gpu \
  --model-config /path/to/resolved_training_config.yaml \
  --split /path/to/prepared/golden32.jsonl --max-rows 32 --required-rows 32 \
  --max-input-tokens 5120 --max-new-tokens 2048 \
  --runtime-python "$A2UI_LITERT_RUNTIME_DIR/venv/bin/python" \
  --runtime-cache-dir "$A2UI_LITERT_RUNTIME_DIR/cache/compiled" \
  --activation-dtype float32 --gpu-decode-steps-per-sync 8 \
  --output-dir /path/to/fresh/evaluation --tensorboard-root /path/to/tensorboard \
  --run-id model-run --evaluation-name golden32 --metric-version v5_4
```

The native API does not expose a physical device selector. CUDA_VISIBLE_DEVICES
does not constrain its Vulkan adapter. Reserve the native default GPU or provide
an isolated container and verify the observed UUID. The runner intentionally uses
one GPU engine, rather than launching workers that could all select GPU 0.

The outputs include raw scoring and the exact Android repair scorer's independent
post-repair scoring. The scorer has no reference-target or source-text fallback.
See [Android repair scoring](android_repair_scoring.md).

## Interpreting latency

Each runtime row records model load time, prefill time, first-token time, decode
time, total generation time, GPU-evidence overhead, and native token counts/rates.
`generation_seconds` is case wall time, including process verification.
`inference_seconds` and first-token time subtract that verification cost.
`engine_load_seconds` is the same one-time load observation repeated in each row;
do not sum it across rows. Native token counters are reported separately from
token IDs, which this binding does not expose.

The initial 29-token natural-language diagnostic used FP16: warm synchronization
8 took 0.401 seconds versus 0.712 seconds with the default synchronization, with
identical text. Those short-prompt numbers are not the final FP32 Golden latency
or accuracy result. Full results live in `litert/r64_fp32_golden32` and
`litert/r64_fp32_golden35`; the model is the September 27 rank-64 export, not the
newer Run E checkpoint whose export was still pending when evaluation started.

Detached jobs survive an SSH disconnect. They do not survive the Space container
or machine being stopped or restarted.
