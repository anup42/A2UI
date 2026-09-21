# Trained E2B integration in GenUI Pipeline and IR Demo

Date: 2026-09-21. Device: Samsung SM-F776U, serial R3GL203AKSF, Android 17.

## Implemented behavior

- Main Settings offers three models: **Trained Gemma 4 E2B (A2UI Mobile)**,
  **official Gemma 4 E2B**, and **Gemma 3 270M A2UI Express**. Hidden historical
  profiles remain recognizable for compatibility; their weights are not deleted.
- The current trained and official packages are reused from `sdk_models`.
  An installed retained successor can replace a stale legacy model selection.
- The main Settings MTP switch shares the SDK demo's persisted switch. Its value
  controls the next engine initialization. Gemma 3 270M does not use a drafter.
- Selecting the trained model for on-device IR routes normal Pipeline Stage 3,
  IR Demo's saved responses, and MCP response paths through `GenUiTrainedConverter`.
- The trained route pins the frozen training prompt, GPU, 8,192 context tokens,
  2,048 maximum output tokens, thinking disabled, and native token metrics.
  CPU/NPU selections fail explicitly for this GPU-only trained route.
- IR Demo skips Stage 2 and branches before cloud credential setup. Its trained
  conversion needs no Gemini request. Normal Pipeline Stage 2 continues to use
  the selected response provider. Optional image URLs may still use networking.
- The demos repair generated DSL, compile it to standard A2UI, and render through
  the existing native renderer. **Source-text fallback is disabled.** Source
  omissions and changes remain visible warnings. A render success is not a
  source-fidelity pass. SDK source-integrity defaults remain unchanged for other
  callers.

## Device test and quality finding

The bounded test opens the real `IrDemoRenderActivity` with the first bundled
sample, `q_001374`, containing the saved response to the $75,000 salary question.
The record has no saved IR, so the Activity must perform fresh inference.
The test observes that same Activity result, validates compiled IR, checks a
native surface and visible generated text, and captures Debug diagnostics.

The initial MTP-on conversion compiled and rendered, with `usedFallback=false`.
It used 3,349 input tokens and reached the 2,048 output-token limit, at **32.92
native decode tokens/s** and **70.436 seconds** for Stage 3. The first test then
failed only because its Debug selector assumed an Android widget class; Compose
exposes that switch as `android.view.View`. The selector was corrected. Initial
evidence is retained under [mtp_on_initial](mtp_on_initial/result.json).

**The generated content was poor.** Repair salvaged six generated string values,
including malformed figures such as `$144.331`, `$284.6`, `$315.0.0` and `$62000.0`.
The response's correct salary/hourly figures, details and URLs were not preserved.
Those values were not replaced with source-derived text. This proves the
integration and repair/render path operate; it does not establish that this
checkpoint produces accurate or attractive IR for the bundled samples.

## Observed runs

All four fresh conversions used the identical frozen rendered prompt (3,349
input tokens), compiled after generated-output repair, returned one native
surface, and reported `usedFallback=false`. **None passed source fidelity.**

| Run | Runtime | Output tokens | Native decode token/s | Stage 3 seconds |
|---|---|---:|---:|---:|
| [Initial on](mtp_on_initial/result.json) | GPU+MTP | 2,048 | 32.92 | 70.436 |
| [Off](mtp_off/result.json) | GPU | 791 | 21.16 | 41.454 |
| [On repeat](mtp_on/result.json) | GPU+MTP | 2,048 | 28.57 | 77.199 |
| [Foreground-interrupted capture](acceptance_initial/result.json) | GPU+MTP | 2,048 | 21.98 | 97.482 |

MTP-on runs reached the output cap. The off run produced more structure but
changed the salary to `$7,00,000`, hourly wage to `$3.666`, and work hours to
`2,800` with `40 hours x 5 weeks`. See the [off generated DSL](mtp_off/generated_ir.json).
The repaired on output is a list of incomplete generated string fragments.
These are model-output defects, not accurate source conversions.

This is functional sampling on a shared device, not a controlled benchmark:
output lengths differed, MTP-on throughput varied, and the last run's screen
capture was interrupted by another foreground activity. Native decode excludes
engine initialization and prompt processing. Stage 3 includes the conversion
and native cleanup.

## Validation and limits

- SDK converter tests: 10 passed, including strict-default rejection and explicit
  generated-only repair with source-fidelity warnings.
- App focused JVM tests: 15 passed across catalog/migration, MTP policy, and the
  trained bridge (configuration, metrics, wire decoding, no-fallback enforcement,
  cancellation cleanup).
- Debug application and instrumentation APKs built with
  `-PgenUiSdkOnlyNative=true`; replacement installation succeeded.
- Shared MTP preference instrumentation test passed.
- Frozen prompt parity check passed:
  `005ef700eee29a4429c895ac6c86580003c43d0ad35e67cbb92a9d4cf0e8c1d2`.
- Main Settings was verified manually: three retained models, trained selected,
  MTP switched off and on, and the matching persisted boolean checked each time.
  See [MTP on](settings_on.png) and [MTP off](settings_off.png).

The complete IR Demo instrumentation test **did not finish green**. The first
three runs passed generation, compilation, native surface and visible generated
text checks, then failed secondary Debug selectors/scroll assumptions. Those
selectors were corrected and scroll-based log checks made explicitly diagnostic.
The fourth run completed conversion, but Android Settings had become the
foreground app before capture, so the visible-output assertion failed. The
harness now guards the foreground package before interacting with the screen;
that final guard was compile-checked only. No further inference was run while
the device was being used elsewhere. Unrelated Android Settings captures are
kept out of this report and Git.

Screenshots of actual IR Demo output: [initial MTP on](mtp_on_initial/screenshot.png),
[MTP off](mtp_off/screenshot.png), [MTP on repeat](mtp_on/screenshot.png).
Instrumentation transcripts are saved as `instrumentation_*.log` here. The
`success` field in per-run JSON describes the pipeline result, not the entire
instrumentation test. Native runtime evidence is in [runtime.log](runtime.log).

Normal Pipeline and MCP Stage 3 routing were checked in code and focused bridge
tests; this task did not make a new cloud Stage 2 request or rerun Bixby50.

## Installed artifacts

- Application APK: `android/app/build/outputs/apk/debug/app-debug.apk`
- Application APK SHA-256:
  `52a94da92adce31555fbe82d37970a1c7f8385903e877be6aefb8021085a32b6`
- Current model SHA-256, verified on device:
  `4675f37353e41c786a3e94f03ba4f64ad366a2e4e5d188bb92f5bef3922a75fe`
- Current model path:
  `/sdcard/Android/data/com.samsung.genuicraft/files/sdk_models/gemma4_e2b_a2ui_mobile.litertlm`
- IR provider was left as on-device LiteRT, trained model selected, GPU selected,
  and MTP enabled. The response provider was preserved.
