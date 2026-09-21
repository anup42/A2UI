# SDK Bixby50 streaming view - 2026-09-22

## Delivered

The SDK Bixby50 screen now calls the AAR provider's native streaming overload. It shows live Generated IR, retains each complete raw attempt unchanged, and displays Repaired IR separately after conversion. The result area adds an IR/Preview tab bar, progress stages, copy and expand controls, character counts, code scrolling, and expandable conversion notes. Input collapses while generating. Token metrics remain independently optional and available in both views.

The trained SDK screen uses generated-DSL repair with source-text fallback disabled, matching the existing Pipeline and IR Demo policy. The trained prompt and GPU/MTP inference settings are unchanged. Source-fidelity warnings remain available in conversion notes.

## Device verification

- Device: Samsung SM-F776U, Android 17, R3GL203AKSF.
- Case: BXP-001, launched with the actual **Convert + render** button in the SDK screen.
- Actual runtime: `LiteRT-LM/Gemma4/GPU+MTP`; native logs also show the MTP drafter operating.
- **52 distinct cumulative raw snapshots before native completion**.
- Generated program: **923 characters**, retained byte-for-byte after repair.
- Repair: `GENERATED_DSL_REPAIR`; final Express: **657 characters**.
- Final program matches the compiled document rendered in Preview; no source-text fallback.
- Screenshot metrics: 3,174 input tokens, 288 output tokens, **62.12 native decode tokens/sec**, 8.68 seconds conversion total (one measured run).
- Generated/repaired headings and Preview click verified by UiAutomator; all three screenshots inspected visually.
- Final instrumentation: `OK (1 test)` in **11.539 seconds**.

[Live stream](live.png) | [Generated and repaired IR](generated_repaired.png) | [Rendered preview and metrics](preview.png) | [Observation data](stream_proof.json) | [Device test](device_test.txt) | [Runtime logs](runtime.log)

## Checks

All **15 focused JVM tests passed**: 5 streaming-provider cases, 7 model-selection/lifecycle cases, and 3 metrics cases. The app and instrumentation APKs built successfully; the final app was installed on the connected device.

Installed APK SHA-256: `1b55729ce494433f2d7448a6456e2b13e7bb662ce9890bda5c215565419a29f0`.

The initial UI run passed native generation/repair assertions but timed out locating the repaired heading. After tightening the narrow-screen controls and capturing the terminal UI before assertions, the repeat passed end to end. A compilation-only missing-scope import was also fixed before installation.

This validates the SDK screen's streaming, output preservation, repair presentation, and preview navigation using one Bixby50 case. It does not rescore the full corpus or change the model's source-fidelity behavior.
