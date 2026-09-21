# Generated-content recovery fix — 22 September 2026

Recovery now combines generated state, surviving components, and readable fragments instead of accepting the first small renderable subset. The exact captured train and weather failures both retain all three generated data rows and all three surviving Text components. The published AAR is consumed by the installed test app.

## Captured failures before and after

| Case | Previous recovery | Updated recovery | Device check |
| --- | --- | --- | --- |
| BXP-003 trains | 204-character Express; three Text components, no train rows | Three complete rows with all five generated fields, three Text components, and 16 additional readable fragments from damaged components | All three train names and surviving prose visible while scrolling |
| BXP-001 weather | 657-character Express; table points to missing `/forecast_data_data`, no forecast rows | Three complete rows with all four generated fields, three Text components, and four recovered column-label fragments | All three dates and surviving prose visible while scrolling |

The [train preview](device/BXP-003/page_00.png), [train continuation](device/BXP-003/page_01.png), [weather preview](device/BXP-001/page_00.png), and [weather continuation](device/BXP-001/page_01.png) are native device screenshots. Exact generated input, repaired Express, compiled A2UI JSON, diagnostics, and accessibility hierarchies are saved beside each image.

## Recovery behavior

- Recover state and component calls together. State rows keep their original order and repeated rows are retained.
- Recover complete fields and array items from damaged or truncated state. Never finish an unterminated quoted value or invent missing data.
- Resynchronize at subsequent assignments so a broken quote/call cannot swallow later valid Text components.
- Render recovered state as literal tables/lists/text. An unresolved table binding triggers recovery even when its syntax compiles.
- Preserve remaining complete readable strings under **Additional recovered text**, excluding already represented values and recognized layout/binding metadata.
- Keep strict compilation and its resource limits. The broader recovery is opt-in; an over-deep broken layout may be flattened into a bounded valid document.
- Label successful generated-DSL recovery **Recovered** in the SDK demo. Generated raw IR and repaired IR remain separate in the existing streaming view.

This does not add a content-loss rejection gate. Source fallback remains disabled for the tested route. Hosts explicitly supplying `sourceText` to the compiler still opt into the existing separate source-integrity gate.

## Validation

- **350 SDK JVM tests passed**, including ten new content-recovery regressions. See [unit test summary](unit_tests.json).
- **Two captured cases replayed on the connected SM-F776U / Android 17** through the app's published AAR, with all three row markers and surviving prose checked across scroll positions. See [device test output](device_test.txt) and [device results](device/summary.json).
- Exact captured fixture hashes are checked by both JVM and device tests; device and JVM repaired Express are compared byte-for-byte in the validation manifest.
- **Existing captured full50 corpus:** 48/50 compile and JVM-render, zero source fallbacks, 0/48 passing the separate source-integrity audit. See [captured replay report](captured_full50_replay.json). This is deterministic replay of existing model output, not a new inference benchmark or a 50-case device run.
- BXP-009 and BXP-039 in that older corpus contain repeated damaged syntax with no recoverable answer strings or structured values. No answer is fabricated for those inputs.
- The AAR publication, app build, replacement install, and focused device replay completed successfully. Build identity is recorded in [validation manifest](manifest.json).

The model's factual errors remain visible: for example the forecast's `211` low temperature and corrupted train names/times are retained exactly. Recovering data and rendering it is not evidence of factual correctness or source fidelity. Loose fragments can also contain contradictory or incomplete model values; they are shown separately because their original structure cannot be recovered reliably.

## Reproduce the device replay

Build/publish `GenUICraft` and build/install `android` with `-PgenUiSdkOnlyNative=true`. Assemble/install the debug Android-test APK. Stage both exact fixtures using `adb push` to `/data/local/tmp/`, then `run-as com.samsung.genuicraft cp` them to:

```
files/recovery-input/BXP-003.raw.express
files/recovery-input/BXP-001.raw.express
```

Use byte-preserving file transfer and check the fixture SHA-256 values; shell stdin transfer can normalize Windows line endings.

```powershell
adb -s R3GL203AKSF shell am instrument -w -r `
  -e class com.samsung.genuicraft.SdkRecoveryOnDeviceTest `
  com.samsung.genuicraft.test/androidx.test.runner.AndroidJUnitRunner
```

Artifacts are written under app-internal `files/result/sdk_recovery/<run>/`. This test constructs no model/provider, uses no API credentials, and makes no model calls. It uses ActivityScenario and UiAutomator rather than the Compose/Espresso APIs affected by the device's Android 17 input-manager incompatibility.
