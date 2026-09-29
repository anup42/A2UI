# Rank-64 GPU FP16 RoPE phase defect: mechanism and repair boundary

## Scope and conclusion

The inspected model is the exact rank-64 QAT-compatible W4 package used by the device controls: `de60d19c8e1ef06ed1032212d0d4ae5d9b0ec105db72db34a988453004e63e62` SHA-256 (verified against `.tmp/fp16_rootcause/r64.litertlm`, 2,588,147,712 bytes). Its FlatBuffer graph declares the rotary angle path `FLOAT32`, but the captured LiteRT GPU FP16 OpenCL programs cast the integer token position to `half`, multiply it by a half inverse-frequency vector, store the angle in `half`, and only then convert that rounded angle to float for `native_sin`/`native_cos`. This is a **confirmed numerical defect in the GPU FP16 RoPE phase construction**. It can produce order-one sine/cosine changes at the tested positions. The controlled RoPE-only A/B removed most of the fixed-target loss but still generated invalid raw DSL. The later RoPE + float Q/DQ experiment closed the fixed-target gap; see the [full report](REPORT.md).

The [controlled GPU report](../20260928_r64_gpu_controls/REPORT.md) establishes the execution-level failure condition: with identical model bytes, input token IDs and MTP disabled, GPU FP16 failed both BXP-001 and BXP-003; CPU sampling with FP32 logits reproduced the same bad outputs; GPU FP32 passed both. These controls place the fault before sampling but do not isolate a particular op. The graph and kernel capture below isolate a concrete pre-sampler numerical fault. There is no evidence here for FP16 overflow, and attention accumulation or other FP16 paths remain possible additional contributors.

## Exact graph and runtime path

The graph audit is `.tmp/fp16_export_audit/rope_graph.json`, extracted from the same SHA-verified package. Only the four main signatures among 976 subgraphs contain `SIN`/`COS`; each has both 128-channel and 256-channel rotary branches. The 256-channel inverse-frequency constant contains 64 nonzero values followed by 192 zeros. The graph's position input is `INT32`, its cast/angle/sine/cosine tensors are **declared** `FLOAT32`, and the frequency constants are `FLOAT32`.

| Signature | 128-channel `CAST → MUL → RESHAPE → SIN` ops | 256-channel `MUL → RESHAPE → SIN` ops | Inverse-frequency tensors |
|---|---:|---:|---|
| `decode` | 25 → 26 → 27 → 28 | 319 → 320 → 321 | 296 / 245 |
| `prefill_1024` | 27 → 28 → 29 → 30 | 348 → 349 → 350 | 174 / 119 |
| `prefill_128` | 27 → 28 → 29 → 30 | 348 → 349 → 350 | 174 / 119 |
| `verify` | 26 → 27 → 28 → 29 | 347 → 348 → 349 | 304 / 248 |

For `decode`, `decode_input_pos:0` feeds `RESHAPE` op 24, `CAST` op 25, and `MUL` op 26 by the 128-value inverse-frequency constant. The 256-channel branch shares the position cast. Corresponding `COS` ops follow each `SIN`; the RoPE values then feed Q/K rotation. The input attention mask is Boolean and the KV-cache signature inputs are INT8. Those input dtypes do not themselves round the token position, although they do not rule out downstream half-precision attention errors.

The four files in [evidence/kernels](evidence/kernels/) are exact copies of the OpenCL source captured from this FP16 runtime, not edited pseudocode. The [capture log](trace_fp16.native.log) records `CL_TRACE source=16` onward and successful builds; file numbers and byte counts match the trace. The decisive statements are:

| Captured source | Operation | Decisive code |
|---|---|---|
| [program_00016.cl](evidence/kernels/program_00016.cl) | position cast | `int4 src = read_imagei(...)`; `src_final = convert_half4(src)`; `write_imageh(...)` |
| [program_00017.cl](evidence/kernels/program_00017.cl) | angle multiply | `half4 first_value = read_imageh(...)`; `half4 second_val = read_imageh(...)`; `result = first_value * second_val`; `write_imageh(...)` |
| [program_00018.cl](evidence/kernels/program_00018.cl) | sine | `half4 src = read_imageh(...)`; `convert_half4(native_sin(convert_float4(src)))` |
| [program_00019.cl](evidence/kernels/program_00019.cl) | cosine | `half4 src = read_imageh(...)`; `convert_half4(native_cos(convert_float4(src)))` |

Thus `convert_float4` occurs **after** the angle has lost precision. Declaring intermediate FlatBuffer tensors FLOAT32 is insufficient under the selected GPU FP16 execution mode. The C API's FP32 setting remains the demonstrated working whole-graph control; [pinned LiteRT-LM v0.16.1 executor settings](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.16.1/runtime/executor/llm_executor_settings_utils.cc#L914-L923) map that setting to GPU `kFp32` rather than `kFp16`.

## Numerical magnitude from the model's actual frequencies

[numerical.json](numerical.json) retains all 128 and 256 exact `FLOAT32` inverse-frequency values extracted from the SHA-verified graph and the per-position calculations. This is a NumPy diagnostic, **not a measurement of device kernel output**: reference angle is `float32(position) * original_float32_frequency`; simulated captured path is `float16(float16(position) * float16(frequency))`; ideal sine/cosine are evaluated on each angle and outputs rounded to half. It isolates the angle-rounding effect without modeling the GPU's `native_sin` approximation.

| Position | Half position | 128-channel max angle error | 128-channel max sine error | 256-channel max sine error | Max sine error if F32 sine is rounded only at output |
|---:|---:|---:|---:|---:|---:|
| 1,023 | 1,023 | 0.382 rad | 0.368 | 0.252 | 0.000243 |
| 2,047 | 2,047 | 0.629 rad | 0.555 | 0.551 | 0.000243 |
| 3,173 | 3,172 | 1.705 rad | 0.950 | 1.153 | 0.000244 |
| 3,174 | 3,174 | 0.963 rad | 0.824 | 1.161 | 0.000237 |
| 3,393 | 3,392 | 2.223 rad | 1.788 | 1.788 | 0.000242 |
| 3,394 | 3,394 | 1.083 rad | 0.974 | 0.641 | 0.000243 |

The phase error is not a step function beginning at position 2,048: frequencies and their product are also rounded to half, so substantial error exists before that point. Above 2,048, binary16 spacing is two integers, causing adjacent odd positions such as 3,173 and 3,393 to round to the previous even position. Even positions still suffer large phase error from half-frequency and half-product rounding. Sine/cosine are bounded in [-1, 1], so rounding them to half *after* computing the phase and trig in F32 costs only about 2.5e-4 in this diagnostic.

## Repair and its delegate constraint

The minimally targeted *numerical* repair is to compute both frequency families' sine and cosine from exact integer positions and the model's F32 frequencies in F32, then pass only bounded sine/cosine values through the otherwise FP16 Q/K rotation. An F32-generated sin/cos cache indexed by the original INT32 position is one implementation; [LiteRT-Torch v0.9.3's RoPE cache](https://github.com/google-ai-edge/LiteRT-Torch/blob/v0.9.3/litert_torch/generative/layers/attention_utils.py#L77-L109) uses that order of operations. Patch both families in all four signatures and preserve quantized weights and the rest of GPU FP16 computation. The app's current 8,192 context would require 12 MiB of dense half sine/cosine values for both families; the package's unmodified 32,003-position range would require about 46.9 MiB. These are logical table sizes, not measured GPU residency or package growth. A shorter table requires an enforced matching maximum position and overflow behavior.

The specific 1D-INT32-index lookup repack was compiled and executed on the target Adreno GPU. Captured corrected shaders include the GATHER operations, and the RoPE-only A/B changed the predicted-token scores while retaining GPU execution. This supersedes the earlier source-only concern about [the compatibility checker](https://github.com/google-ai-edge/LiteRT/blob/v2.2.0/tflite/tools/versioning/gpu_compatibility.cc#L782-L799) and [parser](https://github.com/google-ai-edge/LiteRT/blob/v2.2.0/tflite/delegates/gpu/common/model_builder.cc#L1306-L1334) for this particular package. It does not establish support for every gather table layout or backend. The [full report](REPORT.md) records capture and A/B controls.

If a gather variant is built, a rank-2 `[N, C]` table maps to GPU BHWC `[N,1,1,C]` and axis 0 maps to GPU batch; rank-4 `[1,N,1,C]` with axis 1 maps to GPU height ([shape/axis mapping](https://github.com/google-ai-edge/LiteRT/blob/v2.2.0/tflite/delegates/gpu/common/model_builder_helper.cc#L87-L143)). Both gather axes have generated shader branches ([GPU gather task](https://github.com/google-ai-edge/LiteRT/blob/v2.2.0/tflite/delegates/gpu/common/tasks/gather.cc#L29-L78)), but storage limits, shape handling, and full delegation must be checked on the target Adreno GPU. A direct F32 RoPE kernel or runtime-provided per-position F32-computed sin/cos inputs may avoid a dense table but require executor/delegate integration and are not verified here.

The RoPE-only diagnostic produced a fixed-target score of **-40.630646** versus **-226.155121** original FP16 and **-26.047764** original FP32, on the same 570-token target. Its free-running BXP-003 output remained invalid. The independent float-Q/DQ correction combined with RoPE gave **-22.991875** on the same target; one of two raw cases was strict-valid. This supports a second precision-sensitive operation and leaves production readiness unresolved. The current app remains on its working FP32 setting; a native rebuild and broad paired validation are still required before a selective FP16 mode can replace it.
