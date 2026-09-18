# Renderer diagnostics scope

The bundled Android diagnostics tool collected a five-minute log window at 10:27:57 IST on 18 September 2026 during the final v14 corpus replay. Target package: `com.samsung.genuicraft`; observed PID: `29038`; device: `R3GL203AKSF`.

No target-package fatal exception or ANR marker was found in that window. The two fatal markers present in the filtered output belong to unrelated `com.samsung.android.spay`, PID `28505`, and are not attributed to the renderer.

Complete diagnostics remain under `A2UI/tmp/genuicraft_visual_20260918/diagnostics/20260918_102757`. This bounded log window does not establish a device-wide absence of crashes or ANRs.
