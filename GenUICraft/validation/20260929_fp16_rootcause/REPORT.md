# GPU FP16 root-cause investigation and experimental correction

Updated 30 September 2026. Device: Flip8 SM-F776U, serial R3GL203AKSF, Adreno 840. All new tests use GPU, greedy sampling, MTP **off**, the same frozen prompts and unchanged original W4 weight payloads. The second device was disconnected during this investigation.

## Conclusion and deployment status

Two concrete numerical faults are confirmed in the captured FP16 path: **RoPE phase construction is rounded to half before sin/cos**, and **fused quantize/dequantize arithmetic uses half intermediates near discrete rounding thresholds**. Half matrix-vector accumulation is a further precision-sensitive path. A diagnostic combination correcting these paths reaches the original FP32 fixed-target score level; individual fixes do not establish a complete quality replacement.

The correction is implemented and executed in the **standalone diagnostic probe**, not deployed in the app/AAR/Bixby. The app remains on its existing FP32 setting. The tested GPU shaders were substituted under exact source-hash and byte-equality guards; this is an experiment, not a shipping runtime integration. Source proposals and a model repacker are provided for a proper rebuild. No native production runtime has been rebuilt on this PC, which had under 5 GB free during this investigation and no configured Linux build environment. The shipped optimized `ml_drift` kernel path also needs verification against any replacement build.

The smaller **RoPE + float Q/DQ** correction also closes the fixed-target deficit without changing GEMV accumulation. This identifies concrete contributors to the FP16/FP32 gap. It does **not** prove every checkpoint-versus-LiteRT difference has the same cause, nor that FP16 is inherently unsuitable for this model.

## Exact mechanisms

1. **RoPE:** although the graph declares FLOAT32 angle tensors, the FP16 delegate generates INT32 → half position, half frequency multiplication, half angle storage, then float sin/cos on that already-rounded angle. Position 3393 becomes 3392. Even exactly representable positions have angle error from frequency/product rounding. Using actual model frequencies, simulated maximum sine error at position 3394 is about 0.974 in the 128-channel branch. Computing the phase and trig in float before rounding the bounded result to half reduces that error to about 0.000243. These error magnitudes are arithmetic simulations; the actual shader sequence is device-captured. See [mechanism and exact kernels](rope_mechanism.md) and [numerical evidence](numerical.json).
2. **Q/DQ:** fused kernels use half subtraction, multiplication and rounding for activation quantization. Small round-off before `round()` can select a different quantization bin, then propagate through subsequent layers. The graph contains QUANTIZE/DEQUANTIZE operators, not standalone FAKE_QUANT operators; “fakequant” in diagnostic filenames refers to this fused generated arithmetic. The tested patch promotes intermediates and delays the GEMV result half cast until after Q/DQ; scalar coefficients remain half-rounded. A production correction should preserve the original float coefficients as well.
3. **Matrix-vector accumulation:** captured decode kernels multiply and accumulate in half, including the local reduction. The diagnostic changes products/accumulators/reductions to float while retaining half inputs, dequantized weights, and output storage. This alone did not close the remaining gap. Do not equate this stronger diagnostic with LiteRT's untested mixed-accumulation enum.

Changing a tensor declaration to FLOAT32, setting the existing `fp32_fp16` metadata, or only changing the sampler does not implement these selective corrections. The current LiteRT-LM integration resolves its existing mixed setting to full FP32. LiteRT enum 3 supports FP32 accumulation for selected matrix operators; it does not itself fix RoPE or Q/DQ. [LiteRT precision definition](https://github.com/google-ai-edge/LiteRT/blob/v2.2.0/litert/c/litert_common.h#L310-L319), [Q/DQ source](https://github.com/google-ai-edge/LiteRT/blob/v2.2.0/tflite/delegates/gpu/common/tasks/quantize_and_dequantize.cc#L23-L50).

## Controlled fixed-target results

All rows score exactly the same 1,696-byte, 570-token train-case continuation after identical prompt tokens. Larger/less-negative is better for this fixed target. These are **not v5.4 scores, percentage accuracy, full-vocabulary logit parity, or Bixby50 results**.

| Controlled run | Same-target score | Target tokens |
|---|---:|---:|
| [Original FP16](original_fp16_score_003/BXP-003.metrics.json) | -226.155121 | 570 |
| [Original FP32](original_fp32_score_003/BXP-003.metrics.json) | -26.047764 | 570 |
| [FP16 + corrected RoPE](rope_fp16_score_003/BXP-003.metrics.json) | -40.630646 | 570 |
| [RoPE + float GEMV](rope_floataccum_score_003/BXP-003.metrics.json) | -48.203625 | 570 |
| [RoPE + float Q/DQ](rope_fakequant_v2_score_003/BXP-003.metrics.json) | -22.991875 | 570 |
| [RoPE + float GEMV + float Q/DQ](rope_floataccum_fakequant_v2_score_003/BXP-003.metrics.json) | -25.938889 | 570 |

See [per-token analysis](fixed_target_score_analysis.md). Original FP16 has large deficits across many positions; correcting RoPE removes most of them. The combined correction brings the aggregate score close to the FP32 control. Autoregressive output can still differ and can still contain DSL mistakes.

## Raw generation results — no repair

| Run | Case | Raw strict valid | Raw v5.4 | Native decode tokens/s | Failure / evidence |
|---|---|---|---:|---:|---|
| Previous gpu_default_r2 | [BXP-001 raw](../20260928_r64_gpu_controls/gpu_default_r2/BXP-001.raw.txt) | False | 0.00 | 41.41 | See score details |
| Previous gpu_default_r2 | [BXP-003 raw](../20260928_r64_gpu_controls/gpu_default_r2/BXP-003.raw.txt) | False | 0.00 | 26.47 | See score details |
| Previous gpu_fp32 | [BXP-001 raw](../20260928_r64_gpu_controls/gpu_fp32/BXP-001.raw.txt) | False | 0.00 | 39.37 | See score details |
| Previous gpu_fp32 | [BXP-003 raw](../20260928_r64_gpu_controls/gpu_fp32/BXP-003.raw.txt) | False | 0.00 | 12.63 | See score details |
| repack_fp16_raw_003 | [BXP-003 raw](repack_fp16_raw_003/BXP-003.raw.txt) | False | 0.00 | 19.37 | raw:ValueError:Unterminated Express string or delimiter |
| rope_fakequant_v2_raw_001 | [BXP-001 raw](rope_fakequant_v2_raw_001/BXP-001.raw.txt) | True | 89.89 | 41.09 | See score details |
| rope_fakequant_v2_raw_003 | [BXP-003 raw](rope_fakequant_v2_raw_003/BXP-003.raw.txt) | False | 0.00 | 12.99 | raw:ValueError:Element 'l' references missing 'n' at children[1] |
| rope_floataccum_fakequant_v2_raw_001 | [BXP-001 raw](rope_floataccum_fakequant_v2_raw_001/BXP-001.raw.txt) | False | 0.00 | 12.47 | raw:ValueError:Element 'root' references missing 'f' at children[5] |
| rope_floataccum_fakequant_v2_raw_003 | [BXP-003 raw](rope_floataccum_fakequant_v2_raw_003/BXP-003.raw.txt) | True | 95.05 | 22.25 | See score details |
| rope_floataccum_raw_001 | [BXP-001 raw](rope_floataccum_raw_001/BXP-001.raw.txt) | False | 0.00 | 25.57 | raw:ValueError:Unbalanced Express delimiter |
| rope_floataccum_raw_003 | [BXP-003 raw](rope_floataccum_raw_003/BXP-003.raw.txt) | False | 0.00 | 27.17 | raw:ValueError:Unknown A2UI Express component 'ColumnRY' |
| rope_fp16_raw_003 | [BXP-003 raw](rope_fp16_raw_003/BXP-003.raw.txt) | False | 0.00 | 14.70 | raw:ValueError:A2UI Express completion must contain exactly one <a2ui> block |
| rope_fp32_raw_003 | [BXP-003 raw](rope_fp32_raw_003/BXP-003.raw.txt) | True | 40.00 | 12.41 | See score details |

The combined correction's train output is strict-valid and scores 95.05. Its weather output preserves the forecast values but references missing element `f`, so its **raw** strict score is zero. The smaller RoPE + Q/DQ correction gives a strict-valid weather output scoring **89.89**, while its train output references missing `n`. These are not silently counted as passes, and each correction has only 1/2 strict-valid raw generations in this tiny test. The RoPE-only FP32 control itself scores 40.0, showing that even a numerically close graph rewrite can change the free-running sequence; it is not identical to the unmodified FP32 control.

**Speed is not a controlled benchmark here.** Rows differ in output length, kernel changes, compilation, and device temperature. Moderate thermal throttling was observed during the sequence. The smaller correction's final weather run measured **41.09 native decode tokens/s**, 327 decode tokens, 1.68 s prefill and 7.96 s decode (9.64 s generation, excluding 5.47 s engine creation). The prior unmodified FP32 weather control measured 39.37 tokens/s under different conditions; no reliable production speedup can be claimed. Score-mode timings are not decode benchmarks.

## Controls and integrity

- Original model SHA-256: `de60d19c8e1ef06ed1032212d0d4ae5d9b0ec105db72db34a988453004e63e62`.
- RoPE diagnostic package SHA-256: `6a6d60e0003349047fb2000148e4c0d21232f869ff7dcde1a737ce943704e428`.
- Repack-only control SHA-256: `0ccf0bbe12ae6c1721e818ae8541dfe8b66d1ed41cc1e3bda7cbfe93232fbce0`. This uses the same repacked layout but keeps the original operations; its raw train output reproduces the original FP16 failure after outer whitespace trimming.
- The RoPE package preserves all 657 original target buffer payloads (777,983,384 bytes), the original target copy, and non-target sections. Small package manifests are under [evidence/model](evidence/model/); large models are intentionally excluded from the repository.
- MTP is disabled for all new tests. The repacker leaves the drafter's own RoPE path unchanged. **Do not enable MTP or use positions ≥8192 with this experimental package**, even under FP32.
- Actual corrected GPU kernels contain GATHER from the original integer position; this operation compiled and executed on the device. A prior source-only concern about constant GATHER compatibility was superseded by that device evidence.
- CPU sampling of the RoPE-only GPU result retained the same failing prefix. Prior original-model controls also reproduced the FP16 failure with CPU sampling, so the GPU top-k sampler is not a sufficient explanation.
- `rope_mixed_capture` requested enum interposition, but no setter interception occurred. It is **not a valid mixed-precision result** and is excluded from the score tables. The runner now rejects that silent non-application.
- The first fakequant source patch failed OpenCL compilation due to a half4-to-float4 cast. The failure is retained; only `v2` completed results are used. One-token kernel-capture runs are not quality results.

## Fix files and remaining integration work

- [Experimental RoPE repacker](../../tools/native_gpu_probe/patch_rope_lookup.py) and [independent verifier](../../tools/native_gpu_probe/verify_rope_lookup.py): four target signatures, 16 RoPE replacements, pinned model and bounds guards; paired repack-only control.
- [Guarded GEMV patch generator](../../tools/native_gpu_probe/patch_fp16_gemv.py), [Q/DQ patch generator](../../tools/native_gpu_probe/patch_fp16_fakequant.py), [OpenCL diagnostic hook](../../tools/native_gpu_probe/opencl_trace.cc), [immutable one-case runner](../../tools/native_gpu_probe/run_precision_case.py).
- [Native probe and reproduction instructions](../../tools/native_gpu_probe/README.md), including teacher-forced selected-token scoring.
- [LiteRT-LM mixed-accumulation source proposal](../../tools/native_gpu_probe/litert_lm_v0.17.1_mixed_accum.patch): explicitly opt-in, selected after common GPU options; **not built or device-tested** and not exposed through current Kotlin/C API.
- [LiteRT Q/DQ source proposal](evidence/upstream/litert_v2.2.0_qdq_fp32.patch): preserves float quantization coefficients and intermediates before the output cast. Source application checks are separate from a native build and device validation; this public-source proposal is not a rebuilt replacement for the optimized binary.

To ship a faster precision path: build a runtime with correct RoPE and Q/DQ precision boundaries plus validated matrix accumulation; expose the explicit policy through C API/JNI and the shared SDK; distinguish its compiled cache identity; fix and validate the drafter as well; enforce context bounds if lookup tables are used; then run paired raw/repaired Bixby50 and MTP on/off with controlled temperatures. Until that validation, the existing FP32 setting remains the working production mitigation.
