# Why the trained E2B model currently uses GPU FP32

29 September 2026. This audit rechecks saved controlled experiments and the current SDK and pinned LiteRT source. **No new inference, installation, or application-setting change was performed.**

## Finding

**FP32 is an accuracy workaround for this trained W4 model on the deployed LiteRT GPU stack, not a hardware requirement for E2B.** FP16 executes on the GPU, but the tested raw generations corrupt numbers and produce invalid Express graphs. The controlled experiments isolate GPU execution precision as the failure condition; they do not identify the specific sensitive or faulty operator.

The weights remain W4. The production workaround changes the model package's `prefer_activation_type` hint from `fp16` to `fp32`; the verified cached package differs by exactly two bytes. It does not replace the weights, tokenizer, graph, or drafter. See [byte verification](../20260928_gpu_fp32_app/model_byte_verification.json).

## Direct evidence, before any repair

The native probe used the same model SHA, prompt bytes and token IDs, greedy decoding, and MTP disabled. The actual input lengths were 3,174 and 3,394 tokens. It reproduced the app's failing output without the UI or repair pipeline.

| GPU computation and sampling | BXP-001 weather | BXP-003 trains | Raw strict validity |
|---|---|---|---:|
| FP16, GPU sampler | Missing forecast table; invalid reference | Corrupt numbers/times; invalid DSL | 0/2 |
| FP16, CPU sampler with FP32 logits | Same failing output | Same failing output | 0/2 |
| FP32, GPU sampler | Correct forecast table and numbers | Correct train numbers/times and valid graph | 2/2 |

Examples: FP16 changed train `12007` to `120007`, `20660` to `206066`, and `10:35–10:45` to `1:35–0:5`. FP32 preserved these values. Raw v5.4 scores changed from 0.00/0.00 to 91.15/99.01. FP32 was not completely source-faithful: the weather output still omitted citations.

[Controlled report and timings](../20260928_r64_gpu_controls/REPORT.md) · [Actual raw outputs](../20260928_r64_gpu_controls/raw_comparison.html) · [Earlier CPU/GPU comparison](../20260928_r64_accuracy_gap/REPORT.md).

The CPU-sampler control is important: it rules out changing only the sampler or logits handoff as a sufficient fix. The severe discrepancy is already upstream in GPU FP16 execution. MTP, rendering, streaming, and repair are absent from this reproduction. A damaged model copy or changed prompt cannot explain this controlled difference.

## What the runtime setting does

The SDK pins LiteRT-LM Android 0.16.1 and its packaged LiteRT dependency is 2.2.0. `GenUiModelProfiles.trainedE2b()` requests GPU FP32. `Gemma4Runtime` prepares the metadata-only cached copy and uses a separate GPU shader cache while keeping the GPU backend.

In the pinned runtime, activation type FLOAT32 selects GPU compilation precision `kFp32`; other activation types select `kFp16`. The `fp32_fp16` metadata option selects FLOAT32 plus a mixed-precision flag, and the common GPU-options code maps that flag to `kFp32` too. Consequently, merely choosing that metadata label is not an established route to FP16 speed with FP32 accuracy. We have not benchmarked it, and these mappings do not prove every resulting graph or kernel would be identical.

Primary sources: [compilation precision mapping](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.16.1/runtime/executor/llm_executor_settings_utils.cc), [mixed-precision metadata parsing](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.16.1/runtime/engine/engine_settings.cc), [common GPU precision options](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.16.1/runtime/executor/litert_compiled_model_executor_utils.cc).

## Root cause: known boundary, unknown operator

FP16 range/rounding or accumulation, precision-sensitive fused kernels, attention/mask/KV-cache handling, and a GPU compiler/driver interaction remain candidates. **No layer activation or per-step logit comparison was captured, so claiming a confirmed overflow, normalization bug, or particular kernel defect would be premature.**

The training QAT calculation uses BF16 and explicitly does not simulate native KV cache (`training/src/ir_training/qat/fake_quant.py`). Checkpoint/export weight parity therefore does not establish native GPU FP16 execution parity.

A separate upstream report describes long-prompt GPU number/text corruption with official E4B on Adreno 750, clean CPU/short-prompt output, and no improvement from changing MTP. This makes context-dependent GPU behavior a useful lead, but it is a different model/device and does not establish the same root cause here. [Upstream issue 3012](https://github.com/google-ai-edge/LiteRT-LM/issues/3012).

## Speed: precision and thermals must be separated

| Diagnostic case | GPU FP16 decode tokens/s | GPU FP32 decode tokens/s |
|---|---:|---:|
| BXP-001 | 41.41 | 39.37 |
| BXP-003 | 26.47 | 12.63 |

These were sequential single runs with different generated outputs and no thermal/frequency control. They demonstrate observed timings, not a reliable universal FP32 penalty. FP32 can increase arithmetic and intermediate-storage costs, but the exact cost depends on the selected kernels.

The subsequent sustained FP32 benchmark also recorded thermal throttling and GPU maximum-clock caps of 160–342 MHz, versus 1.3 GHz before the run. The slowdown cannot be attributed entirely to FP32. The full Bixby50 MTP-on/off report uses FP32 in both modes and is not a precision comparison. [Saved thermal and timing evidence](../20260928_fp32_bixby50_mtp/REPORT.md).

## Recommended resolution

1. Keep the validated FP32 workaround for this trained model while investigating; FP16 plus repair cannot reliably correct plausible but wrong source numbers.
2. Localize the first divergence with fixed, teacher-forced tokens and runtime instrumentation: compare CPU reference, GPU FP16, and GPU FP32, then distinguish prefill from decode and inspect attention/cache boundaries. The current probe does not supply those intermediate tensors or full per-step logits.
3. Once localized, test selective higher precision or a verified runtime/kernel fix. Merely re-exporting unchanged weights or restoring the FP16 hint is not supported as a remedy by current evidence.
4. Measure the speed improvement on the same cooled device with warm caches, alternating precision, fixed prompts/budgets, repeated trials, and recorded thermal/clock state. Gate any default change on raw numerical fidelity as well as DSL validity and repaired scores.

No production default was changed by this audit. The exact operator-level cause remains unresolved.
