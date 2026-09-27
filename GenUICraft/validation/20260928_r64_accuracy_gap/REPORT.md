# Checkpoint / LiteRT accuracy gap: CPU versus GPU isolation

**Follow-up completed:** explicit CPU sampling still reproduces both GPU FP16
failures exactly; forcing GPU FP32 on the unchanged model restores 2/2 valid
raw outputs with correct numbers. See the [native precision controls](../20260928_r64_gpu_controls/REPORT.md).
This narrows the severe additional loss to GPU FP16 execution before sampling.

28 September 2026. **A fresh controlled test localizes a substantial part of
the observed failure to the deployed GPU execution stack. The exact same
quantized model produces 2/2 valid raw outputs on CPU and 0/2 on GPU, with MTP
off in both runs.** The CPU outputs preserve the key weather and timetable
numbers that GPU omits or corrupts. This identifies a backend-dependent
regression, not the precise responsible GPU operation.

[Compare checkpoint, CPU and GPU raw text and screenshots](checkpoint_cpu_gpu.html).

## Measured results

| Case | Supplied CUDA checkpoint | Fresh LiteRT CPU | Fresh LiteRT GPU, MTP off |
|---|---|---|---|
| BXP-001 weather | Raw valid; 312 tokens; correct listed forecast numbers | **Raw valid, no repair**; 302 tokens; 33.36 s; 17.49 decode tokens/s. Entire forecast state line exactly matches the supplied checkpoint output. | Raw invalid; 236 tokens; 10.88 s; 37.75 decode tokens/s. Forecast table is absent; references a missing child. |
| BXP-003 trains | Raw valid; 578 tokens; preserves train numbers/times | **Raw valid, no repair**; 594 tokens; 58.76 s; 17.39 decode tokens/s. Preserves 12007, 20660, 12614 and 10:35–10:45, 11:30, 15:15. | Raw invalid; 586 tokens; 42.64 s; 14.34 decode tokens/s. Changes 12007 to 120007, 20660 to 206066, and 10:35–10:45 to 1:35–0:5. |

Android's raw compiler and the independent Python strict scorer agree: **CPU
2/2, GPU 0/2**. Both GPU cases render only after generated-DSL repair. Each
fresh GPU raw output is byte-identical to its earlier MTP-off run, making this
a repeatable observed failure. Both CPU raw outputs were generated in this
investigation, rather than recovered from the source or supplied checkpoint.

CPU is not a complete quality pass. BXP-001 omits citations. BXP-003 has a
repeated child, unreachable elements and an unsupported placeholder action.
Its supplied-scoring counterpart is also not a perfect source-faithful model.
The CPU v5.4 scores are 91.15 and 40.00, versus 92.41 and 99.01 in the supplied
checkpoint predictions and zero for both GPU outputs. Scores are diagnostic;
schema validity alone is not enough. The SDK's literal-only source checker also
flags values held in state; direct raw-state and screenshot inspection confirms
the CPU numeric values described above.

## Controlled conditions

- Device `R3GL203AKSF`, Samsung `SM-F776U`, Adreno 840, Android 17.
- Both runs use the same installed app, quantized model, frozen two cases,
  source bytes, runtime-rendered prompt hashes, input-token counts, context
  and output limits, greedy sampling, disabled thinking and disabled MTP.
- The only runtime configuration difference is CPU versus GPU. Output is
  captured before compilation/repair. No source-text fallback is allowed.
- Model SHA-256:
  `de60d19c8e1ef06ed1032212d0d4ae5d9b0ec105db72db34a988453004e63e62`.
- Installed main APK SHA-256:
  `6791838cbc0371843670c9a467a9a48891c3899ef20458ed5520fcbd4b6da846`.
  Only the instrumentation APK was updated to allow `-e backend CPU`.
- Screenshots retain the test screen's generic trained-E2B V10 profile label;
  the explicit model path and verified file hash identify the actual rank-64
  QAT-compatible model used in both runs.
- Input counts are 3,174 and 3,394 tokens. Native prompt hashes also agree
  with the supplied CUDA predictions. Native token IDs remain unavailable.
- [Backend log evidence](backend_evidence.log) confirms CPU/XNNPACK and GPU
  OpenCL with MTP false. GPU target decode and prefill are fully delegated.
- [Device identity](device_identity.json) and [installed native library hashes](native_library_hashes.json)
  retain the exact artifacts. The local APK used for library inspection hashes
  exactly to the installed app.

The test APK built and installed successfully; both instrumentation runs
completed. Their broad render assertions pass because GPU repair is allowed;
the separate raw-validity fields and independent scorer expose the failure.
Timing is one run per backend, CPU first; this is an accuracy isolation test,
not a controlled thermal throughput benchmark.

## Ranked explanation

**1. GPU native arithmetic / fused operations / prefill and cache behavior is
the leading explanation for the severe additional corruption.** Backend
selection changes the result while model bytes, prompt and app pipeline stay
fixed. Native GPU precision is a concrete lead: the package requests FP16;
the pinned executor defaults to FLOAT16 and GPU compilation selects FP16.
GPU sampling also selects FP16 logits. Framework evaluation uses BF16 QAT
with Transformers attention; native KV-cache behavior is not simulated.
These differences require measurement before naming a particular faulty
kernel or precision boundary.

**2. GPU sampling and the logits handoff remain candidates.** The installed
APK contains stock Maven LiteRT-LM 0.16.1 JNI with SDK-pinned LiteRT/OpenCL
libraries and a dependency-patched OpenCL sampler. The current sampler links
to `libLiteRt.so`; the older shim-based description in the Android README is
stale. No evidence identifies that dependency patch itself as the defect.

**3. Residual checkpoint/native differences and training quality also remain.**
CPU is closer on numeric fidelity but not identical to the checkpoint. CUDA
adapter matmul versus CPU reconstruction, activation simulation, source
omissions, and graph quality can still matter. The supplied prediction JSONLs
lack the exact evaluation receipt binding their QAT mode and adapter hash.
These caveats cannot account for why an unchanged native model performs so
differently between the freshly tested CPU and GPU paths.

The checked export receipt, identical device hashes and successful CPU
generation reduce the likelihood that damaged copying, wrong adapter selection
or general weight packing failure is the main cause. MTP is not necessary for
the failure. SDK text accumulation directly appends native deltas, and both
backends share it. Increasing the token limit would not fix the early numeric
corruption and missing weather table observed here.

## Source evidence and next isolation

- [QAT activation scale dtype and SRQ](../../../training/src/ir_training/qat/fake_quant.py#L589),
  [projection forward](../../../training/src/ir_training/qat/fake_quant.py#L918),
  and [native cache simulation limitation](../../../training/src/ir_training/qat/fake_quant.py#L1116).
- [Pinned native executor](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.16.1/runtime/executor/llm_litert_compiled_model_executor.cc#L1649)
  and [GPU precision settings](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.16.1/runtime/executor/llm_executor_settings_utils.cc#L76).
- [LiteRT-LM issue 3012](https://github.com/google-ai-edge/LiteRT-LM/issues/3012)
  independently reports malformed numbers/words on GPU with long prompts and
  clean CPU output. It uses E4B on Adreno 750, so it is supporting symptom
  evidence, not proof of the same defect on this E2B/Adreno 840 setup.
- [Checked export receipt](../20260928_r64_qat_reexport/export_audit/REPORT.md)
  establishes successful intended weight reconstruction and package checks,
  without claiming native inference equivalence.

The next focused experiments are GPU execution with `--sampler_backend=cpu`
and, separately, `--force_f32` through a native runner built against the pinned
runtime. [Both options exist in native flags](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.16.1/runtime/engine/shared_flags.h).
They are not exposed by the current Kotlin configuration. CPU sampling also
changes logits precision, so an improvement would isolate a sampling/precision
boundary rather than uniquely implicate the sampler binary. A short/long
prompt control can then test prefill-length sensitivity. None of these further
experiments was run in this investigation.

For present comparisons, CPU is the more reliable diagnostic reference on these
two cases. Preserve the GPU requirement for deployment, but fix or isolate its
runtime behavior before interpreting the GPU results as the model's inherent
quantization loss or deciding to retrain.

## Reproduction artifacts

[CPU run](cpu/REPORT.md), [GPU run](gpu/REPORT.md), [machine-readable comparison](comparison.json),
and [comparison script](compare_backends.py). Both run folders preserve raw
output, prompt/source, repair result, timing and initial/scrolled screenshots.
Large unfiltered diagnostic buffers remain local and are excluded from the commit.

```powershell
python GenUICraft/tools/report_trained_bixby50.py GenUICraft/validation/20260928_r64_accuracy_gap/cpu
python GenUICraft/tools/report_trained_bixby50.py GenUICraft/validation/20260928_r64_accuracy_gap/gpu
python GenUICraft/validation/20260928_r64_accuracy_gap/compare_backends.py --checkpoint-predictions C:/Users/anupk/Downloads/trained_model_data/e2b_runD_predictions/r64_bixby50_scored.jsonl
```
