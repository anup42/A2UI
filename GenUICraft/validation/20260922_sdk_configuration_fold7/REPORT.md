# SDK demo state restoration — Fold7, 22 September 2026

The GenUICraft SDK page now retains its current result and active generation when Android recreates the Activity for rotation or a theme change.

## Changes

- `GenUiSdkDemoViewModel` retains the provider, generation job, rendered document, raw/repaired IR trace, status, runtime identity and metrics. It uses `Application` context and `viewModelScope`; cleanup runs when leaving the page, rather than on configuration recreation. Conversion, prompts, recovery and model inference continue to use the existing shared SDK APIs.
- Saved Compose state restores the selected Bixby50 sample, edited source, settings dialog, model setup controls, selected tab, code/notes expansion and performance detail expansion.
- Separate saved state holders retain the preview and IR scroll positions when switching tabs and across recreation. Automatic tab selection responds to an actual generation transition, without overriding a restored manual tab selection. A completion that arrives between Activities is still handled.
- Each new generation or explicit document load starts a fresh workspace. The input editor shares the available height so Generate UI and Render A2UI remain reachable in landscape.

This addresses configuration recreation. It does not add a disk-backed session/history store or resume inference after process termination.

## Verification

All device work used **Galaxy Z Fold7, SM-F966B**.

- **10 focused JVM tests passed**, covering existing model-selection/provider-key and token-metric behavior. See [counts](jvm_tests.json) and [build log](test_build.log).
- **Five device tests passed**. See [instrumentation output](device_tests.txt).
  - The selected third sample, exact edited text, sample navigation position and open settings dialog survive Activity recreation.
  - A deterministic streaming provider remains the same instance through recreation, real landscape rotation and a system theme change; there is exactly one generation call and no provider close during those events. Completion preserves raw IR, repaired document, runtime identity, metrics and status. Leaving the page closes the provider.
  - Preview scroll is restored within one pixel after Preview → Inspect IR → Preview and Activity recreation.
  - Existing token-metrics and MTP-settings persistence tests continue to pass.
- One **real trained E2B** run for **BXP-003** survived rotation and a theme change while generating. It completed with GPU+MTP in **one attempt**, showing 3,394 input tokens, 438 output tokens, 44.90 native decode tokens/s and 17.350 seconds for conversion. These are observations from one run with UI transitions, not a performance benchmark. After completion, another theme change retained Inspect IR and the same metrics. See [live assertions](live_check.json).
- The app/test APK builds and replacement installation passed. See [app installation](app_install.log) and [test build](final_test_build.log). Original system theme and automatic rotation settings were restored.

| Evidence | Capture |
| --- | --- |
| Landscape input with visible actions | [Editor](landscape_editor.png) |
| Real generation before recreation | [Streaming](live_generating.png) |
| Completed output after rotation/theme change | [Preview](live_after_rotation_theme.png) |
| Inspect IR retained after theme restoration | [IR view](live_ir_after_theme_restore.png) |
| Final result left open | [Final preview](live_final_preview.png) |

The updated APK is installed and also stored at `Download/GenUICraft/GenUICraft-demo-configuration-fix.apk`. Installed/stored hashes match the local build; see [manifest](manifest.json). This change is in the demo host; the shared AAR remains version 0.4.2. No Bixby changes were needed.
