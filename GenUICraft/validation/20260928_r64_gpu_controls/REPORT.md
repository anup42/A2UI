# Rank-64 LiteRT GPU accuracy gap: precision and sampler controls

28 September 2026. **The severe additional failures on the two tested cases are
caused by the choice of GPU FP16 execution, before sampling.** Moving sampling
to CPU (with FP32 logits) reproduces the same failures. Forcing GPU FP32
computation on the unchanged W4 model restores valid, numerically faithful raw
output in both cases. This identifies a failing runtime setting and a working
mitigation; it does not identify the particular faulty or sensitive operation.

[Compare the actual raw outputs](raw_comparison.html) · [Machine-readable results](comparison.json).

## Controlled results, before repair

| Execution | BXP-001 weather | BXP-003 trains | Raw validity |
|---|---|---|---|
| GPU FP16, GPU sampler | Missing forecast table; invalid reference | Corrupted train numbers/times; invalid DSL | 0/2 |
| GPU FP16, explicit CPU sampler and FP32 logits | Exactly repeats the failing GPU output | Exactly repeats the failing GPU output | 0/2 |
| GPU FP32, GPU sampler | Correct forecast table and numbers | Correct train numbers/times and valid graph | **2/2** |

Both default GPU results match the earlier app/JNI output after trimming outer
whitespace. Both CPU-sampler results also match those outputs exactly. Thus the
native C API probe reproduces the problem without the app, renderer, streaming
adapter, repair code, or fallback.

FP32 weather retains `31/20/57%`, `30/21/55%`, and `30/21/55%`. Its output
matches the earlier CPU result. FP32 trains retain `12007`, `20660`, `12614`,
`10:35–10:45`, `11:30`, and `15:15`. FP16 instead produces values including
`120007`, `206066`, and `1:35–0:5`.

| Case | Supplied checkpoint v5.4 score | Native GPU FP16 | Native GPU FP32 |
|---|---:|---:|---:|
| BXP-001 | 92.41 | 0.00 | **91.15** |
| BXP-003 | 99.01 | 0.00 | **99.01** |

These are independent Python strict/scoring results. No repair was performed.
A score match is not text equivalence or a guarantee of every factual detail.
FP32 weather still omits citations; native/checkpoint text is not identical.
Only two frozen cases were tested, not the full Bixby50 suite. No new renderer
screenshots were generated in this native experiment; the linked comparison
shows the actual raw output.

## What is now isolated

1. **GPU FP16 computation is the demonstrated failure condition.** Keeping the
   same quantized weights and switching the GPU precision setting to FP32
   removes the measured severe failures.
2. **The GPU sampler and FP16 logits handoff are not sufficient explanations.**
   Explicit CPU sampling also changes the compiled logits output to FP32, but
   leaves both raw outputs unchanged and wrong. The remaining discrepancy is
   upstream in GPU FP16 execution.
3. **A broken copy, different prompt, or W4 packing alone cannot explain this
   controlled difference.** All variants use the identical file hash and native
   tokenizer ID sequence. FP32 works with the same quantized model file.
4. **MTP, repair and the renderer are not involved in this reproduction.** MTP
   is disabled and the probe captures native output directly.

The exact lower-level explanation is still open: FP16 rounding/range effects,
precision-sensitive fused kernels, attention/mask/cache handling, and an
OpenCL compiler/driver interaction remain possibilities. No layer activations
or per-step logits were captured, so this report does not claim a specific
kernel bug or general checkpoint-to-LiteRT numerical parity.

The checkpoint's BF16 QAT simulation does not establish parity with native
FP16 GPU execution. That is a relevant mismatch, but these runs do not prove
that BF16-versus-FP16 alone explains every remaining quality difference.

## Reproduction and control integrity

- Device: `R3GL203AKSF`, Samsung `SM-F776U` / Flip8, Android 17, Adreno 840.
- Model SHA-256: `de60d19c8e1ef06ed1032212d0d4ae5d9b0ec105db72db34a988453004e63e62`.
  No model byte was changed. FP32 changes runtime computation, not W4 weights.
- Frozen prompt text matches the earlier app-rendered prompt hashes. Native
  tokenizer IDs are byte-identical across all three variants for each case.
  Tokenizer counts omit the separately inserted BOS; actual prefill counts are
  **3,174 and 3,394** in every variant.
- Context 8,192; output cap 2,048; top-p sampler with k=1, p=1, temperature=0,
  seed=42; MTP disabled; raw session receives the fully rendered prompt without
  applying an additional chat template. Each case/variant has a separate cache.
- Runner uses the official Android C API 0.1.0 distribution in LiteRT-LM
  v0.16.0. Upstream states v0.16.1 shares the same source commit. This is not the
  app's JNI binary; exact reproduction of both JNI GPU outputs establishes its
  relevance to this investigation.
- The optional runtime/OpenCL/sampler libraries match the app's pinned copies.
  Native logs show OpenCL delegate initialization on the GPU path. Auxiliary
  embedding CPU registrations are not a fallback of the main model to CPU.
- The C API exposes FP32 directly. CPU sampler selection required a small,
  diagnostic-only bridge because the C API lacks that setter. The host runner
  verifies the exact native library SHA before execution; the bridge checks
  pinned ARM64 instructions, the GPU backend, and engine/session readbacks.
  Both logs show `sampler_after=CPU(3)` and `session_sampler=CPU(3)`.
- Two probe builds were used: baseline/FP32 before the sampler bridge, then CPU
  sampling after adding that guarded option. Their hashes and full commands are
  in the individual run manifests. The precision setting and ordinary inference
  path are unchanged by the added sampler option.

[Runtime hashes](runtime_manifest.json), [probe source and build instructions](../../tools/native_gpu_probe/README.md),
[baseline run](gpu_default_r2/run_manifest.json), [CPU sampler run](gpu_cpu_sampler/run_manifest.json),
and [FP32 run](gpu_fp32/run_manifest.json) retain the provenance. Two initial
launch attempts failed before inference (linker path and executable mode);
they are retained separately and excluded from all accuracy counts. Large
unfiltered device diagnostics and compiled libraries remain local.
Committed log files use LF endings with trailing whitespace removed; their
[captured and normalized hashes](log_normalization.json) are retained. Raw model
output files are unchanged, and original log bytes remain in local diagnostics.

## Timing observations

| Case / GPU setting | Output tokens | Prefill wall s | Decode wall s | Generation s | Native decode tokens/s |
|---|---:|---:|---:|---:|---:|
| BXP-001 FP16 | 236 | 2.02 | 5.70 | 7.72 | 41.41 |
| BXP-001 FP16 + CPU sampler | 236 | 1.45 | 5.70 | 7.15 | 41.44 |
| BXP-001 FP32 | 302 | 2.73 | 7.67 | 10.40 | 39.37 |
| BXP-003 FP16 | 586 | 2.06 | 22.14 | 24.21 | 26.47 |
| BXP-003 FP16 + CPU sampler | 586 | 1.80 | 43.58 | 45.38 | 13.45 |
| BXP-003 FP32 | 571 | 5.90 | 45.23 | 51.13 | 12.63 |

Generation excludes engine creation. FP32 engine creation took 8.46 s and
11.48 s respectively. These are individual, sequential diagnostic runs without
thermal or frequency control; they do not establish a reliable FP16/FP32 speed
ratio. Native profiling-summary warnings were nonfatal and did not prevent
generation or benchmark counters.

## Practical resolution

Use GPU FP32 computation for this trained W4 model as the demonstrated
mitigation, then validate its quality and latency on a wider sample before
making it the deployment default. No weight re-export was needed for this
improvement. The production Kotlin/SDK integration still needs a supported
precision-control path; the private-ABI sampler bridge must remain diagnostic.
**The installed app and its default precision were not changed by this task.**

If the goal is to keep FP16 performance, the next lower-level investigation is
teacher-forced logits and layer/prefill/decode comparisons to find the first
FP16 divergence. Retraining or repeating the same weight export does not
directly address the demonstrated runtime precision sensitivity.

## Upstream source references

- [v0.16.1 release identity](https://github.com/google-ai-edge/LiteRT-LM/releases/tag/v0.16.1).
- [C API activation-type setter](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.16.1/c/engine.cc#L728).
- [FLOAT32 selects GPU kFp32; other activation settings select kFp16](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.16.1/runtime/executor/llm_executor_settings_utils.cc#L74).
- [GPU sampling uses FP16 logits only with FP16 execution; explicit CPU sampling retains FP32](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.16.1/runtime/executor/llm_litert_compiled_model_executor.cc#L1886).
- [Native settings wrappers used only by the guarded diagnostic bridge](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.16.1/c/engine_internal.h#L65).
