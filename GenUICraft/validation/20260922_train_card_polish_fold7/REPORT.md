# Train card visual refinement — Fold7, 22 September 2026

SDK **0.4.2** refines the train comparison renderer with a ticket-style service header, prominent full-width departure time, station details, a dashed divider, and compact labeled duration/seating chips. The chips use their natural width and wrap to the next line when needed, so a long duration is no longer squeezed into half a card. The section title and service count can also wrap independently.

| View | Screenshot |
| --- | --- |
| Previous cards on Fold7 | [Before](before.png) |
| Updated cards, dark mode | [After](after/page_0.png) |
| Remaining service and recovered notes | [Scrolled preview](after/page_1.png), [remaining notes](after/page_2.png) |
| Light mode with 130% text size | [Larger-text preview](light_large_text/page_0.png) |

The implementation lives in the shared SDK renderer; the app-native reference copy is synchronized. Colors follow the Material theme. Labels and exact values are both visible and exposed through merged text semantics for the metadata chips. No prompt, inference, repair, generated IR, timetable value, or source note was changed for this visual refinement.

## Verification

- **374 SDK JVM tests passed**, with zero failures, errors, or skips. See [counts](jvm_tests.json) and [build log](sdk_build.log).
- The AAR was published and the demo APK was built and installed on **Fold7 SM-F966B, Android 16**, using replacement installation. See [installation log](app_install.log).
- **Two device tests passed** in dark mode at the existing 100% text size: the exact captured train output, plus the earlier train/weather recovery regression. See [test output](device_tests.txt).
- **One additional device test passed** in light mode at 130% text size. All three train services and their notes remained visible while scrolling. The duration and seating details wrapped without truncation. See [test output](large_text_test.txt) and [captured settings](light_large_text/settings.json). The original dark mode and 100% text size were restored afterward.
- Captured raw and repaired Express hashes are unchanged, and the replay tests disable source fallback and make no model calls. See [current-output assertions](after/result.json), [larger-text assertions](light_large_text/result.json), and [earlier-fixture regression](regression/summary.json).

Screenshots were visually reviewed on the cover display. The existing two-column layout was retained; the unfolded display and spoken TalkBack output were not tested in this run.

After the replays, one fresh BXP-003 generation using the trained model on GPU+MTP completed and was left open on Fold7. The screen reports 3,394 input tokens, 438 output tokens, 53.72 decode tokens/s and 15.411 seconds for conversion. This is a single observed run, not a performance comparison. See the [live preview](live_preview.png) and [UI capture](live_preview.xml).

## Delivery

The APK is installed on Fold7 and copied to `Download/GenUICraft/GenUICraft-demo-0.4.2.apk`. Installed and stored APK hashes match the local build. Bixby's local dependency is updated to **0.4.2**, and all 20 copied Maven publication files match the SDK publication. Bixby was not built or pushed. See [artifact hashes](manifest.json).

The renderer continues to display malformed model-produced times such as `10:35–1004`, `11:000` and `15:5` exactly as supplied; this change does not infer timetable corrections.
