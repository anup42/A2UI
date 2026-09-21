# IR streaming debug validation - 2026-09-21

## Result

PASS: trained E2B streams real native text in both debug implementations (Pipeline and IR Demo). The IR Demo device acceptance run observed **53 distinct cumulative snapshots before completion**, then retained the exact raw program and displayed **Repaired IR** in the next pane. Both panes scroll independently, and the live pane follows new text.

## Device acceptance

- Connected Samsung SM-F776U (R3GL203AKSF), Android 17.
- Saved fixture: **BXP-001** from the packaged Bixby50 corpus; one inference, no Gemini request.
- Model: `gemma4_e2b_a2ui_mobile`; runtime: `LiteRT-LM/Gemma4/GPU+MTP`.
- Native logs confirm MTP requested and supported, plus actual drafter activity (success rate 0.558642).
- 288 native output tokens; **57.66 tokens/sec** native decode rate in this single run.
- 53 observed updates over 4.62 seconds, before terminal completion.
- Raw output: 923 characters / 929 UTF-8 bytes, unchanged after repair.
- Repaired program: 657 characters / 661 UTF-8 bytes; compiled and produced a native render surface without a render error.
- Test result: `OK (1 test)` in 10.745 seconds. Screenshots visually inspected.
- Installed APK SHA-256: `850e738d4089d9ecccac3257a15c32428059948abe3a21e12229ec4eb286baab`.

[Live generation](live_raw_ir.png) | [Raw and repaired IR](final_repaired_ir.png) | [Full observation evidence](streaming_proof.json) | [Device test](device_test.txt) | [Native runtime evidence](runtime.log)

## Automated checks

- 15 Gemma4 provider tests passed.
- 4 native stream adapter tests passed: cumulative deltas, failure preservation, observer error, and cancellation draining before handle release.
- 9 trained bridge tests passed, including partial delivery before conversion, exact raw preservation on failure, cancellation cleanup, and source-fallback rejection.
- App and instrumentation APK builds passed. App installed with `genUiSdkOnlyNative=true`.
- Main code for both debug screens compiled; the live phone acceptance case exercised IR Demo.

The frozen model prompt, GPU/MTP settings, repair rules, and source-fidelity policy are unchanged. This is one streaming-display acceptance case, not a new full Bixby50 quality evaluation.

## Build environment

The first app build hit a full C: drive after main Kotlin compilation. Compressing the generated Gradle 8.10.2 transform cache and replacing the previous generated APK recovered space; the repeat build and all checks above then passed. No source, dataset, model, or personal files were removed.
