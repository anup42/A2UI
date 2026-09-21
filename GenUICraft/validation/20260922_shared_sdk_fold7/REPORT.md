# Shared SDK, demo UI and Bixby integration — 22 September 2026

The SDK and reference app changes are complete and validated on **Galaxy Z Fold7 (SM-F966B, Android 16/API 36)**. The Bixby source integration and local Maven publication are updated. Bixby itself was not built or installed because its private dependencies are unavailable.

## Shared library ownership

SDK **0.4.0** owns the trained conversion route:

`Answer text → frozen trained prompt → GPU model → A2UI Express → generated-output recovery → compiled A2UI → SDK renderer`

`GenUiSession` selects the conversion profile, captures cumulative raw IR and native metrics, applies the recovery policy, and cancels/drains the provider. The trained profile uses best-effort recovery of generated output and **disables source-text fallback**. The immutable raw generation remains available separately from repaired IR. Source-fidelity warnings remain available even when recovery produces a renderable document.

The SDK also owns the generic `LiteRtModelRunner` for official E2B and Gemma 3. App code maps settings and model selections into SDK requests. There are **zero direct LiteRT-LM imports in app production Kotlin sources**. Both SDK native runtimes share one guard for process-global MTP, benchmark and prompt-template flags, restoring them after success or exceptions.

| Consumer | Shared route | Host responsibilities |
| --- | --- | --- |
| GenUICraft SDK / Bixby50 screen | `GenUiSession` and `GenUiView` | Input selection, model setup, progress and inspection UI |
| Pipeline and IR Demo trained model | `TrainedStage3Bridge` → `GenUiSession` | Pipeline state and adapting the SDK document into the existing app graph |
| Bixby | `GenUiIntegrationManager` → `GenUiSession` and `GenUiView` | Answer-stream accumulation, citations, settings and verified model storage |

The frozen training-prompt contract is unchanged. Its synchronization check passes with SHA-256 `005ef700eee29a4429c895ac6c86580003c43d0ad35e67cbb92a9d4cf0e8c1d2`.

## Demo usability changes

- Guided previous/next sample selection, a clear input editor and prominent **Generate UI** action.
- Custom pasted text is labeled **Custom input** instead of retaining an unrelated fixture label.
- Model choices wrap on narrow screens, keeping the trained-model label visible on the Fold7 cover display.
- Model paths and download controls are under **Model setup**; performance metrics and MTP toggles are under **Settings**.
- Generation shows live IR. Success opens **Preview** automatically; **Inspect IR** exposes the original generated IR and repaired IR separately, with copy controls.
- Input/output token counts and native decode speed appear above the preview, where they remain visible during a demo.

Reviewed screenshots: [previous screen](sdk_before.png), [updated input](sdk_input.png), [settings](sdk_settings.png), [live generation](sdk_stream/live.png), [IR inspection](sdk_stream/generated_repaired.png), and [rendered preview](sdk_stream/preview.png).

## Validation results

| Check | Result | Evidence |
| --- | --- | --- |
| SDK JVM tests | **368 passed**, 0 failed/errors/skipped | [counts and suites](jvm_tests.json), [build log](sdk_build.log) |
| Targeted app JVM tests | **29 passed**, 0 failed/errors/skipped | [counts and suites](jvm_tests.json), [build log](app_tests_build.log) |
| Exact Bixby stream/lifecycle/source/action helpers | **34 passed**, 0 failed/errors/skipped | [source hashes and scope](bixby_helper_tests.json), [test log](bixby_helper_tests.log) |
| Exact Bixby model-storage source, isolated Robolectric harness | **4 passed**, 0 failed/errors/skipped | [scope and limitations](bixby_feature_store_validation.md), [JUnit XML](bixby_feature_store_host_tests.xml) |
| Fold7 MTP preference persistence | Passed | [device tests](settings_device_test.txt) |
| Fold7 metrics settings and renderer-only counter clearing | Passed | [device tests](final_apk_device_test.txt) |
| Fold7 live SDK weather generation, streaming, repair and rendering | Passed on final installed APK | [final test](final_layout_device_test.txt), [stream proof](sdk_stream/stream_proof.json) |
| Fold7 trained IR Demo generation without Gemini | Passed; generated output required recovery | [result](ir_demo/result.json), [device tests](ir_recovery_device_test.txt) |
| Fold7 replay of two damaged generated outputs | **2/2 rendered**, no model calls and no source fallback | [summary](recovery/summary.json), [device tests](ir_recovery_device_test.txt) |
| Bixby XML and SDK publication | Six changed XML files parse; new string references resolve; all 20 versioned Maven files match | [manifest](bixby_integration_manifest.json) |

There are **five distinct device test methods**; repeated runs of the same method are not additional coverage. The last repeat followed the narrow-screen layout fix and exercised the real SDK generation screen. The recovery test covers BXP-001 and BXP-003, including scrolling to retained prose and row markers. This is a focused validation, **not a new full Bixby50 model benchmark**.

The package-filtered eight-minute log window after the device suite contained no `FATAL EXCEPTION`, `ANR in`, or `Fatal signal` markers. See [runtime health scope](runtime_health.json). This is not a long-duration stability test.

## Observed GPU+MTP speed

| Fold7 run | Input tokens | Output tokens | Native decode tokens/s | Conversion / stage 3 |
| --- | ---: | ---: | ---: | ---: |
| SDK BXP-001 weather, final APK | 3,174 | 288 | **57.82** | **9.773 s** |
| IR Demo salary `q_001374` | 3,349 | 2,048 | **23.31** | **93.438 s** |

Both report **`LiteRT-LM/Gemma4/GPU+MTP`**. The final weather run records 51 partial-output observations before completion, with the exact raw output retained after repair. The salary run reaches the 2,048 output-token limit and then recovers a renderable document. Native decode throughput excludes initialization and prompt processing; it is not end-to-end throughput. These different workloads are not a controlled performance comparison. No MTP-off comparison was added in this validation.

See [weather metrics](sdk_stream/metrics.json), [weather stream proof](sdk_stream/stream_proof.json), and [IR Demo metrics and fidelity warnings](ir_demo/result.json).

## Bixby delivery and enablement

The actual checkout is `C:\Users\anupk\Downloads\Bixby_18Sep`. Its Renderer consumes `com.samsung.genuicraft:genuicraft:0.4.0` from `aars/genuicraft-maven`. The integration uses the trained GPU profile and the same SDK recovery/renderer path as the test app.

The normal Bixby Settings page now includes **Generative visual answers**. In a Bixby build containing these changes:

1. Open **Generative visual answers → Trained visual model**.
2. Import `Download/GenUICraft/models/gemma4_e2b_a2ui_mobile.litertlm` from the Fold7. The file is already staged there.
3. Enable **Use generative visual answers**. **Faster generation** controls MTP and defaults on for this verified mobile export.

The feature defaults off. Import uses the Android document picker, copies into Bixby's app-scoped storage, verifies exact size and SHA-256, and atomically replaces the model. Verification and import share a mutex. The 2.588 GB model is a separate file, not embedded in the APK or AAR. The isolated host tests cover invalid-import rollback and preference/listener behavior; they do not exercise a successful full-size import through the Bixby Settings Activity.

Search answers identified as Perplexity/internet search and other structured Markdown answers enter the GenUICraft route. Plain native alarm/control responses remain eligible for the existing Bixby renderer. Candidate stream text is buffered so a prose prefix is retained when a later delta establishes Markdown structure. Settings changes cancel active work, invalidate late results, hide the native presentation, and drain/recreate the session. Review also fixed a race between the current-request check and publishing a result.

The legacy WebView still receives commands for conversation state, history and TTS; the SDK view covers its visual answer while enabled. Bixby retains an explicit error/retry presentation when model generation cannot recover. It does not silently replace failed model output with a source-text fallback.

Full Bixby compilation, merged-resource compatibility, real Settings navigation/model import, and end-to-end Bixby rendering remain unverified until its private dependencies are available. The changed source paths and hashes are recorded in [the local integration manifest](bixby_integration_manifest.json); the checkout also contains `docs/genuicraft_bixby_integration.md`. Bixby changes were **not pushed**.

## Artifacts and content limitations

- Final APK: `android/app/build/outputs/apk/debug/app-debug.apk`, installed on Fold7 and copied to `/sdcard/Download/GenUICraft/GenUICraft-demo-0.4.0.apk`.
- APK SHA-256: `ca55cee20860c47a58c8d46ddd1c749e2bb26dbda71333609cc36851454c9ec6`. Installed and internal-storage copies match.
- SDK AAR SHA-256: `b357e81301c863e408f56f9f25a6c2ccc53e5da3caf27558311363b77f0d7a56`. Bixby's copy matches.
- Model: 2,588,147,712 bytes; SHA-256 `4675f37353e41c786a3e94f03ba4f64ad366a2e4e5d188bb92f5bef3922a75fe`. The Fold7 runtime and import copies were verified.

[Artifact manifest](manifest.json) and [model transfer evidence](model_transfer.json) record identities. No model weights, APK, or Bixby proprietary source is included in these committed validation artifacts.

**Rendering success is not factual correctness.** The trained weather output changes a low temperature from `21` to `211` and truncates `Sep 11` to `Sep 1`; recovery preserves those generated values. The salary output also loses or changes numeric facts, documented in its fidelity warnings. Recovery makes damaged generated structure renderable; it cannot establish that the model preserved the answer. No claim of 50/50 factual correctness or production acceptance is made.
