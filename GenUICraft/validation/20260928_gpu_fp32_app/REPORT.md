# GenUICraft 0.5.1: GPU FP32 app update

28 September 2026. The shared library now runs the trained E2B profile with
**GPU FP32 arithmetic**, mitigating the FP16 accuracy loss isolated in the
[native control experiment](../20260928_r64_gpu_controls/REPORT.md). The Android
test app consumes the published **0.5.1 AAR**, and was installed on the connected
Samsung **SM-F776U**, serial `R3GL203AKSF` (the connected Flip8).

The trained rank-64 QAT-compatible model is selected. Both the GenUICraft SDK
screen and the GenUI pipeline / IR Demo use the same library profile. The
official model keeps its existing precision behavior; the MTP enable/disable
option remains available. No Bixby checkout changes were made in this update.

## Device results

These are five fresh model generations: two Bixby50 cases with MTP off, the same
two with MTP on, and one real IR Demo salary case. This is a bounded verification,
not a rerun of all 50 cases.

| Case | MTP | Raw strict valid | Recovery | Render | Native output tokens | Decode token/s | Provider wall time |
|---|---|---|---|---|---:|---:|---:|
| BXP-001 weather | Off | Yes | None | Pass | 302 | 37.62 | 19.28 s |
| BXP-003 trains | Off | Yes | None | Pass | 571 | 13.06 | 48.97 s |
| BXP-001 weather | On | Yes | None | Pass | 301 | 49.37 | 12.99 s |
| BXP-003 trains | On | No | Generated DSL recovery | Pass | 561 | 17.21 | 38.60 s |

With MTP off, both app raw outputs match the native FP32 reference exactly after
trimming outer whitespace. Weather preserves `31/20/57%`, `30/21/55%`, and
`30/21/55%`; trains preserve `12007`, `20660`, `12614`, `10:35–10:45`, `11:30`
and `15:15`. Independent Python strict validation also passes both. The v5.4
scores are **91.15** and **99.01**, matching the earlier native FP32 scores.

MTP still changes generation. Its weather raw output passes strict validation
(score 90.03); its train raw output references `l` from `j=Column([k,l],...)`
without defining `l`. Both Android and Python reject that raw graph. Library
recovery renders the generated content, with no source-text fallback. The raw
train score remains **0**, and is not replaced by the repaired score. All train
numbers/times are preserved in this raw MTP-on output. **Use MTP off when raw
output consistency is the priority**; the retained toggle allows the speed tradeoff.

The real **IR Demo** salary case `q_001374` passes on `GPU+FP32+MTP`: 3,349 input
tokens, 579 output tokens, 13.11 decode token/s, Stage 3 54.65 s, Stage 4 37 ms.
It uses the preloaded response without Gemini, needs no repair, has no missing
numeric facts according to the demo's fact checker, and renders a visible salary
table. The test also verifies debug diagnostics and the MTP runtime identity.

These single sequential runs are not thermally controlled. Provider wall time
includes initialization/prefill/decode and overhead; the first MTP-off request
also prepares the model copy. Decode throughput is not end-to-end throughput.

## Evidence

- [MTP-off raw outputs and screenshots](mtp_off/gallery.html), [MTP-on raw outputs and screenshots](mtp_on/gallery.html).
- Exact raw outputs: [weather off](mtp_off/BXP-001/output.express), [train off](mtp_off/BXP-003/output.express), [weather on](mtp_on/BXP-001/output.express), [train on](mtp_on/BXP-003/output.express).
- [Combined results](comparison.json), [IR Demo results](ir_demo/result.json), [IR Demo screenshot](ir_demo/screenshot.png), [debug screenshot](ir_demo/debug_screenshot.png).
- [Runtime FP32/MTP logs](runtime_evidence.txt), [build hashes](build_manifest.json), [installed APK verification](installed_apk_verification.txt).
- [31 SDK unit tests](jvm_sdk_tests.json) and [20 app unit tests](jvm_app_tests.json) passed; all three device instrumentation tests passed.

## How the fix is delivered

The released LiteRT-LM Kotlin 0.16.1 API cannot directly select activation
precision. The library uses the runtime's supported `prefer_activation_type`
metadata setting in a cached copy, through a bounded schema reader. It does not
bundle the diagnostic private-ABI sampler bridge or change quantized weights.

A full device byte comparison found **exactly two changed bytes**, positions
299 and 300 (one-based): the text executor string changes `fp16` to `fp32`.
Every remaining byte, including weights, graphs, tokenizer, vision preferences
and drafter, is identical. Source SHA-256 remains:

```text
de60d19c8e1ef06ed1032212d0d4ae5d9b0ec105db72db34a988453004e63e62
```

The cached model SHA-256 is:

```text
974b7f921f19aad96405f78073acf4c8f0642b97769110cd46b8c68e8c1332df
```

See [byte comparison](model_byte_verification.json) and [copy manifest](model_cache_manifest.properties).
The one-time copy needs ~2.6 GB extra storage; later requests reuse it. Cache
preparation is serialized and cancellable, retries remove incomplete copies,
and shader caches are separated from FP16. Invalid metadata never silently
falls back to FP16. [Cache lifecycle details](../../ON_DEVICE_RUNTIME.md#trained-model-gpu-precision-051).

The installed APK matches the build hash and is also available on the device at
`Download/GenUICraft-FP32-0.5.1.apk`. The SDK screen was reopened after testing.

## Remaining limits

This mitigates the demonstrated FP16 failure, but does not certify complete
checkpoint-to-LiteRT parity. MTP-on train output still needs recovery, and model
outputs still omit citations and can change wording. Existing source-bound
fidelity diagnostics also flag generated state-backed content; their warnings
are retained alongside independent raw validation and fact checks. This update
does not suppress warnings or claim that render success proves source fidelity.
