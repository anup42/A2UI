# Fold7 token metrics validation — 18 September 2026

The metrics-enabled test app was installed on Galaxy Z Fold7 SM-F966B, serial R3CY30QFWLP, Android 16 (SM8750). The APK was also copied to internal storage at `Download/GenUICraft-TestApp-Fold7.apk`. Host, installed APK, and Download copy have identical SHA-256.

## Feature

- **Token metrics** switch is enabled by default and persists across activity/app restarts. Turning it off hides metrics and disables native benchmarking on the next Gemma engine initialization.
- Actual input/output counts, native decode token/s, conversion total, and per-attempt details. Unknown counters remain unavailable; no character-based estimates.
- Native GPU+MTP/thinking remain enabled. Changing the metrics flag recreates the cached provider before the next generation.
- Gauss uses server-reported prompt/completion usage and explicitly labels output tokens divided by request wall time as request-average speed. Server decode speed is unavailable.
- Renderer-only runs, new conversions, and cancellation clear stale metrics. Details expand within a bounded scroll area.

## Measured Fold run

Three representative cases used the accepted AAR prompt, exact source bindings, 16,384 context, 8,192 output limit, temperature 0, thinking enabled with the default 1,024-token budget, and GPU+MTP. One engine was reused with a fresh conversation for each attempt; there was no uncounted warmup. This is a three-case device smoke benchmark, not a new 50-case sweep or an MTP-on/off comparison.

| Case | Input tokens | Output tokens | Native decode token/s | Conversion seconds | Attempts |
|---|---:|---:|---:|---:|---:|
| BXP-003 | 1,391 | 1,095 | 65.81 | 25.58 | 1 |
| BXP-030 | 3,624 | 2,275 | 34.61 | 69.54 | 2 |
| BXP-047 | 1,742 | 1,519 | 72.73 | 23.01 | 1 |

All **3/3 converted and rendered**, with no fallback. BXP-030 required one model repair after an invalid closing Express tag. Its counts sum both attempts and its decode rate weights both attempts by their actual decode duration (output tokens / native rate). All output counts include native thinking work.

Native per-attempt speeds ranged **34.26–72.73 token/s**. Combined native throughput across all four attempts was **47.35 token/s**. Decode speed excludes startup and prefill; conversion time includes initialization when needed, generation, validation and repair. These numbers do not establish a controlled speed comparison with the previous Flip runs, which used different captured prompts.

Raw measurements, exact generated output and screenshots are in [run](run/). The machine-readable aggregation is [speed_summary.json](speed_summary.json).

The normal user flow was then exercised separately from the instrumentation hook: launcher → GenUICraft SDK → BXP-003 → Gemma → Convert + render. It succeeded with 1,391 input tokens, 1,095 output tokens, **66.21 native decode token/s**, **52.13 request-average token/s**, and a **21.074-second conversion**. The provider call took 21.00 seconds. These rounded UI measurements are not mixed into the three-case native benchmark above. See [the actual screen](manual_ui.png), [its hierarchy](manual_ui.xml), and [expanded details](manual_details.xml). The Fold was left on this rendered result with metrics enabled.

## Validation

- SDK: **304 JVM tests passed**, including native counter validation, opt-in/off propagation and Gauss usage parsing.
- App: **3 focused aggregation/fallback JVM tests passed**.
- Fold UI instrumentation: **1 test passed**, covering the persistent switch, metrics visibility, per-attempt details and renderer-only clearing.
- Real Fold conversion/render instrumentation: **3 cases passed** in 122.527 seconds including rendering and test overhead.
- Final diagnostic log confirms `backend=GPU; MTP=true; modelSupportsMtp=true; thinking=true; metrics=true` for both the benchmark and normal app run. No fatal-exception or ANR markers were found in the collected window. LiteRT emitted warnings that lower-level prefill/decode profile summaries were unavailable; the separate native benchmark token counts and rates were returned successfully for every attempt. Logs are retained under [diagnostics](diagnostics/20260918_105132/summary.md).
- Build/install succeeded. An initial instrumentation-only build had two invalid Compose assertion imports; they were removed and the final instrumentation build passed. The production APK was unchanged by that test-source fix.
- All 25 publication files were copied and hash-verified in `C:/Users/anupk/Downloads/Bixby_18Sep/aars/genuicraft-maven`. Bixby itself was not built because its private dependencies are unavailable.

## Artifact identity

- App APK: `b694f35802df3307ce34d88508e2a1e0ebeb218880b6ea032eccc57c16845ae3` (213,067,308 bytes).
- SDK AAR: `3cb4ceaab5b5bfa749356729e56cf978eb4dce01596b45aff2e53c9eb0683410`.
- Model: `181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c` (2,588,147,712 bytes); transferred from the already verified Flip model and reverified on the Fold.
- Accepted Gemma prompt SHA-256: `154ff4d27d8e37b81279e7979c18b4fd90bbdd5135a4e124917712a85438535b`.
