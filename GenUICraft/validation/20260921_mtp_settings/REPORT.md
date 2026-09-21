# MTP settings and connected-device validation

Date: 2026-09-21. Device: SM-F776U, serial R3GL203AKSF, Android 17.
App: `com.samsung.genuicraft`, SDK Bixby50 screen.

## Change

- **Settings → MTP drafter** enables/disables speculative decoding for both
  Gemma profiles. It defaults on and persists across app restarts. The switch is
  disabled during generation. Its new value applies to the next conversion.
- The provider cache key includes MTP, so changing the setting releases the old
  native engine before creating its replacement. Gauss is unaffected.
- The completed run displays its actual engine identity even with token metrics
  off. Runtime logs separate requested MTP, effective MTP and package capability.
- GPU+MTP requests on packages without a drafter fail explicitly. Users can turn
  MTP off for those packages; there is no silent CPU fallback.
- The trained profile retains its existing prompt and disabled thinking setting.

## Correction to the earlier device diagnosis

The prior runtime code evaluated `requestedMtp && modelSupportsMtp(path)` and
logged that result as capability. Thus a disabled setting misleadingly reported
`modelSupportsMtp=false` without inspecting the package. The previous log proved
MTP was off, but did **not** prove the loaded package lacked a drafter.

The current file on this device is named `sdk_models/e2b_v10_w4.litertlm`, but its
SHA-256 is `4675f37353e41c786a3e94f03ba4f64ad366a2e4e5d188bb92f5bef3922a75fe`.
It matches the newer mobile export, whose package audit includes a bundled
44,325,712-byte `tf_lite_mtp_drafter`. No model weights were changed for this task.

## Same-case live UI check

Both conversions used BXP-001, the same model, 3,174 input tokens, 288 output
tokens, GPU, trained prompt, and thinking disabled. Settings were changed through
the actual UI, with no process restart between the two runs.

| Setting | Actual runtime | Native decode | Conversion total |
|---|---|---:|---:|
| On | `LiteRT-LM/Gemma4/GPU+MTP` | 61.39 token/s | 9.033 s |
| Off | `LiteRT-LM/Gemma4/GPU` | 32.68 token/s | 12.396 s |

This is a one-pair functional check, not a controlled throughput benchmark.
Initialization, cache state and device conditions can affect the measurements.
Native decode excludes startup/prefill. Rendering success here is not a new
semantic-quality evaluation of Bixby50 or a no-fallback corpus result.

The on run loaded the drafter and reported `MTP=true; MTPRequested=true;
modelSupportsMtp=true`. Releasing that engine logged `MTP Drafter - Success rate:
0.558642`. The off run reported `MTP=false; MTPRequested=false;
modelSupportsMtp=true`, demonstrating the capability-reporting fix.

Evidence: [runtime logs](runtime_on_off.log), [on screen](run_on.png),
[off screen](run_off.png), [settings screen](settings_on.png).

After the final installation and persistence test, MTP was explicitly turned
back on. A final BXP-001 conversion displayed `GPU+MTP`, 61.38 token/s,
288 output tokens and 12.079 s total. All 198/198 drafter nodes were delegated to
OpenCL. The stored preference is `e2b_mtp_enabled=true`; the app is left showing
that completed MTP run. The final process log contained no fatal-exception/ANR
marker. See [final screen](final_on.png) and [final runtime log](final_runtime.log).

## Build and tests

- SDK focused runtime/provider tests: 14 passed; release AAR published locally.
- App focused model/UI configuration tests: 7 passed.
- Debug application and instrumentation APKs compiled with
  `genUiSdkOnlyNative=true`; application replacement-installed without clearing data.
- The first UI instrumentation attempt exposed the existing Espresso dependency's
  incompatibility with Android 17 (`InputManager.getInstance` missing). The
  settings test uses UI Automator instead of Espresso to exercise the actual UI.
- Final UI Automator instrumentation: **1 passed**. It verifies the absent-key
  default, both setting values across activity recreation, dialog dismissal, and
  restores the original preference. See [test result](settings_test.log).

Final APK SHA-256: `aea81c5ad057ccf072f90d3e44732ff726ea0c1c4e76be0eb1492283c596f787`.
Published AAR SHA-256: `182f980cf7336cd5c6e1634bfb8c7cec292778d05411862f5944312ac52d142b`.
