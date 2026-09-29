# Corrected FP16 in the Android test app

GenUICraft 0.5.8 adds an explicit `FP16_CORRECTED` GPU policy for the pinned
rank-64 trained E2B export. `MODEL_DEFAULT` is still the original uncorrected
runtime behavior. FP32 remains available for comparison and other exports.

The test app has one shared **FP32 / FP16 (corrected)** setting for the SDK
demo, GenUI pipeline and IR demo. The original selected model path remains
saved. FP16 selects `model-fp16-corrected.litertlm` in the same directory and
requires its `.litertlm.fp16.json` sidecar. The MTP toggle remains separate.

## Corrections

1. `tools/prepare_fp16_model.py` rewrites target and drafter RoPE to gather
   precomputed sin/cos values using the original integer position. This avoids
   rounding positions and unbounded phase angles to half before trigonometry.
   The supported context is 8192 tokens. Original weight buffers are unchanged.
2. `genuicraft/src/main/cpp/fp16_correction.cc` promotes recognized GPU Q/DQ
   arithmetic to float around clamp, bin rounding and dequantization, then casts
   back to half. Tensor storage and matrix arithmetic remain FP16. Quantization
   coefficients supplied by this compiler remain half rounded.

This is an opt-in compatibility adapter for the existing packaged compiler,
not an upstream LiteRT-LM source rebuild. It redirects program-creation symbol
lookup only inside the pinned SDK runtime libraries, including LiteRT-LM JNI's
statically linked GPU compiler. It checks each library Build ID and exact Q/DQ
syntax. Unknown Q/DQ blocks cause initialization to fail. The
SDK serializes precision selection across its generation and engine setup.
Applications must not compile unrelated LiteRT GPU models concurrently outside
the SDK while this experimental policy is active.

The SDK checks the prepared model's pinned SHA-256 and manifest. Corrected
shader caches are separate and start fresh once per app process so a run must
observe actual Q/DQ corrections before reporting `GPU+FP16_CORRECTED`.

## Reproduce

```powershell
python GenUICraft/tools/prepare_fp16_model.py SOURCE.litertlm DEST/model-fp16-corrected.litertlm
python GenUICraft/tools/verify_fp16_model.py SOURCE.litertlm DEST/model-fp16-corrected.litertlm
powershell -File GenUICraft/tools/build_fp16_correction.ps1 -Ndk C:/path/to/android-ndk
```

Use the CLI `--help` output for supported preparation arguments. Preparation
accepts the pinned original rank-64 package only, not arbitrary E2B exports.
Stage the generated model and manifest beside the test app's original model,
then select **FP16 (corrected)**. A custom SDK host can pass the prepared path
to `GenUiModelProfiles.trainedE2b` with
`gpuPrecision = Gemma4GpuPrecision.FP16_CORRECTED`.

`tools/run_fp16_app_comparison.py` runs paired app measurements using identical
prompts and sampling, warmups, alternating FP32/FP16 order, and native token
counters. Raw validity, repaired validity, rendered screenshots and timing are
reported separately. A render pass does not establish semantic accuracy or
checkpoint parity.

## Device pilot result

The Flip8 three-case pilot is in
[the app validation report](../validation/20260930_fp16_app/REPORT.md).
With MTP on, observed native decode throughput was 10.84 tokens/s for FP32 and
16.41 for corrected FP16 (+51.4%); median warm generation time was 52.87 s versus
31.66 s. All three corrected-FP16/MTP outputs compiled and rendered. Repaired
v5.4 means were 93.22 and 93.18, respectively. Thermal throttling and differing
GPU clocks limit causal attribution; this is not a guaranteed speedup.

Without MTP, throughput improved only 2.5% and the baggage case degenerated into
a repeated identifier list. Its repaired rendering is not a useful answer.
FP32 remains the default. The FP16 option is experimental and the app explains
the known MTP-off limitation. The tested device preference is FP16 with MTP on.
