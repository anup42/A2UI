# Train comparison renderer — Fold7, 22 September 2026

SDK **0.4.1** adds dedicated train-comparison cards. The current Fold7 screen previously showed generic entity cards with raw labels such as `typical_departure_time` and `seating_class`. The shared renderer now recognizes the recovered `Train comparison` table and presents clear service names, departure stations, prominent departure times, journey durations and seating details.

| Before | After, same generated data |
| --- | --- |
| [Generic cards with machine labels](before_preview.png) | [Train cards](after/page_0.png) |
| [Captured raw/repaired IR view](before.png) | [Third service and retained notes](after/page_1.png) |

## Implementation

- `NativeTrainSemantics` maps rail tables using explicit train/rail title or header evidence. It supports the current snake_case columns, human-readable labels, camelCase and older `departure`, `departure_time`, `duration`, `class` aliases.
- `NativeTrainUiRenderer` adds a train section header and service count, location/seating icons, clear field hierarchy and themed cards. Full names and long values wrap. Wider containers can use two columns; narrow screens use one.
- Direct Table rendering and expanded table rendering both use the train presentation. Explicit requests for a table continue to bypass this new card route.
- Unknown columns, units, duplicate rows and generated values are retained. Mixed or insufficiently identified tables remain on the existing renderer routes. Flight, weather, product and generic comparison exclusions have regression coverage.
- The implementation lives in the GenUICraft AAR. The reference app renderer copy is synchronized for its existing app-native routes. No prompt, inference, repair or generated-IR edits were needed for the visual change.

## Validation

All runtime checks used **Galaxy Z Fold7, SM-F966B, Android 16**.

- **374 SDK JVM tests passed**, including six new rail mapping/fidelity tests. See [counts](jvm_tests.json) and [build log](sdk_build.log).
- AAR publication, app build/install and test APK build passed. See [app installation](app_install.log) and [test build](test_build.log).
- **Two device tests passed**, covering three renderer fixtures: the exact current train output, a previously captured train output using older aliases, and weather recovery/rendering. See [device results](device_tests.txt).
- The current-output test observed all three rail cards, every distinct original cell value and the retained timetable notes while scrolling two viewports. It confirmed the raw snake_case field labels no longer appear in the preview. See [current-output assertions and visible text](after/result.json) and [older-fixture regression results](regression/summary.json).
- The before/after test made **zero model calls** and disabled source fallback. Its captured raw IR SHA-256 is `85f69c512d8df1e871069c2fb3d6c43ccfd283da065e40154e23cc2dd376e1da`. Its repaired IR SHA-256 remains `61abcb5bf5f188560b1b0cb5871849eb208413fdac4c4b97c6fc8181075bc198`, matching the original screen capture exactly.

The screenshots were visually reviewed on the Fold7 cover display. The wider two-column layout has not been physically verified on the unfolded display in this run.

After the deterministic replays, one fresh BXP-003 generation was run and left open in the app.
It uses the new train cards and reports GPU+MTP, 3,394 input tokens, 438 output tokens,
54.38 native decode tokens/s and 13.431 seconds for conversion. This is one observed
run, not a performance comparison. See the [live preview](live_preview.png) and
[captured UI metrics](live_preview.xml).

## Delivery

The updated APK is installed on Fold7 and copied to `Download/GenUICraft/GenUICraft-demo-0.4.1.apk`. Installed and stored hashes match the local build. Bixby's local dependency and all 20 Maven publication files are updated to **0.4.1**. Bixby was not built or pushed. Exact APK/AAR hashes are in [the artifact manifest](manifest.json).

This change improves presentation, not timetable accuracy. Malformed generated times such as `10:35–1004`, `11:000` and `15:5` remain unchanged; the renderer does not guess corrected times, destinations or rankings.
