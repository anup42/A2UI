# E2B renderer visual review — 18 September 2026

This update improves the native renderer inside the GenUICraft AAR. It uses the accepted E2B documents already saved in `gemma50_v9_revalidated`; it does not regenerate responses or change the model prompt. The renderer was originally extracted from the A2UI Android reference app. The reference app now exercises the published AAR.

## Review and changes

All 50 historical initial screenshots were reviewed as an overview, with full-size initial and scrolled views for representative weather, travel, list, schedule and comparison cases. Source code was reviewed alongside those captures.

| Finding in the saved screenshots | Renderer change | Representative cases |
| --- | --- | --- |
| Cards and prose had different side margins; the default lavender card color competed with the content. | Shared 16 dp content margins, neutral light/dark surfaces, restrained blue accents, a defined type scale. | BXP-003, 025 |
| Consecutive list entries resembled continuous paragraphs; headings lacked separation. | Hanging bullets for scalar list items without an existing marker; stronger section spacing and heading weights. Original item text, order and links are retained. | BXP-004, 007, 022, 029 |
| Train and schedule cards were very tall, with competing label emphasis. | Clear title/divider/detail hierarchy; muted field labels; short fields share a row when the available width and font size permit. Long values and larger text retain full-width rows. | BXP-003, 025 |
| The weather hero was oversized relative to the forecast. | Smaller icon and temperature typography, tighter internal spacing, and wrapping instead of single-line truncation for the edited hero fields. | BXP-001 |
| Frozen table labels consumed too much space; fades obscured text; a swipe could stop between columns. | Widths use the actual host viewport, frozen labels use at most 34%, each data column fits the remaining space, and frozen tables settle on a column boundary or the scroll end. Shared header/body scrolling and full text wrapping remain. | BXP-030, 032, 037, 047 |
| Some cost descriptions were right-aligned as if they were numbers. | Right alignment is restricted to scalar numeric values; prose stays left-aligned. A small scroll indicator exposes horizontal navigation. | BXP-047 |

These are presentation changes. Source paragraphs, numeric values, table rows, columns, links and actions are not rewritten or summarized. Explicit table/card choices remain intact.

## Final device results

The **v14 saved-JSON replay passes 50/50**, with zero model calls. The independent [full replay audit](full_replay_v14_audit.json) verifies all 50 source-document hashes, **222 screenshot/hierarchy pairs**, **23 tables and 100 columns**, and an observed vertical end for all 50 cases. There are no reported issues or reached vertical limits. Column evidence means a bounded visible header or representative cell; it does not mean every cell has been independently checked at pixel level.

The [before/after gallery](gallery/gallery.html) contains all 50 initial comparisons and matching named scrolled captures. All final initial screenshots were reviewed as an overview; representative initial and scrolled screenshots were inspected at full size, including the corrected numbering in BXP-018 and aligned table columns in BXP-030/047. Useful comparisons: [train cards](gallery/gallery.html#BXP-003%2Finitial), [baggage list](gallery/gallery.html#BXP-004%2Finitial), [schedule cards](gallery/gallery.html#BXP-025%2Finitial), and [scrolled comparison](gallery/gallery.html#BXP-030%2Ftable_1_horizontal_1).

The final [dark/130% replay](gemma_visual_v14_dark_large/replay_summary.json) also passes **5/5**: BXP-001, 003, 004, 025 and 030. Full-size captures confirm readable dark surfaces, wrapping and single-column train details at the larger size: [weather](dark_samples/BXP-001_initial.png), [train](dark_samples/BXP-003_initial.png), [list](dark_samples/BXP-004_initial.png), [schedule](dark_samples/BXP-025_initial.png), [scrolled table](dark_samples/BXP-030_horizontal.png). This checks the selected cases at 130%; it is not exhaustive coverage of every device size or accessibility configuration.

The bounded runtime log check found no target-app fatal exception or ANR marker; see [diagnostic scope](diagnostics_scope.md).

## Build and delivery

The v14 release AAR is **9,376,695 bytes**, SHA-256 `bdd8b06710e4118aa130fb506e181527b31fa6d204b858b47237fa3bff5160fd`. All **294 JVM tests** pass across 32 suites, with no failures, errors or skips. The published AAR is consumed by the installed reference app, and the installed APK hash matches the built file.

The [source and archive audit](host_build_evidence.json) checks all 106 production files against accepted v10: exactly eight renderer/theme/wrapper files changed, and 98 are byte-identical. Provider, converter, compiler, source bindings, prompts and native-library hashes are unchanged. [Source details](source_delta.json) identify every changed file. The test harness changes are outside the AAR.

All [25 publication files copied to Bixby](bixby_publication.json) match the new Maven repository by SHA-256. The Maven version remains `0.1.0` during this local development iteration; refresh Gradle dependencies when consuming the rebuilt version. Bixby itself was not built or installed.

All [four focused device checks](focused-instrumentation.log) pass on v14: host action callback, exact supplied source URL, literal text preservation, and horizontal table access. They make zero live model calls.

## Reproduction

`tools/stage_revalidated_documents.py` stages the accepted JSON and prior reports byte-for-byte with an explicit provenance manifest. It does not invent successful generation records. `GenUiSdkBixby50Test#replaySavedBixbyCorpus`, with `replayMode=json`, renders those documents through the AAR, captures PNG/accessibility pairs, and checks vertical ends and horizontal column access without constructing a model.

The final reference consumer uses `genUiSdkOnlyNative=true` on Samsung SM-F776U (R3GL203AKSF), Android 17. Light-mode corpus replay and selected dark-mode cases at 130% font scale are separate runs. The dark/text-size settings are local to the test view; device-wide settings are unchanged.

`tools/summarize_renderer_replay.py` independently checks all 50 IDs, JSON provenance, capture files, source table fingerprints, bounded visible header or representative-cell evidence, and zero model calls. This is not a complete pixel-level proof of every cell or a new inference acceptance run.

## Development record and limits

- The first visual preview passed seven cases. Inspection of horizontal captures exposed intermediate column fragments, leading to the settling change.
- The superseded v12 full replay was intentionally stopped after BXP-025 to avoid completing a build already replaced by the settling fix. Its instrumentation termination message comes from that explicit force-stop; it is not acceptance evidence.
- The first dark/130% preview rendered four cases successfully; BXP-001 failed when the harness could not inject a vertical gesture. Its visible weather layout was inspected, but that run is not counted as a passing five-case result.
- The v13 corpus replay passed all 50 cases and its dark/130% replay passed five cases. The final overview identified an extra bullet on bold numbered items in BXP-018; marker recognition was corrected without modifying the original text. The v14 build includes that correction and a focused regression test.
- The v13 focused table test captured the expected final column but failed its accessibility lookup. The focused test now uses the same fresh-cache hierarchy capture as the full-corpus harness; the original failure and screenshot are retained.
- Scroll positions with matching capture names can differ because the layout height changed. The gallery preserves original PNG bytes, including the device's existing yellow identifier overlay outside the app content.
- No new model inference, throughput measurement or prompt-quality claim is made in this renderer update. Prior GPU+MTP acceptance remains documented in `../../VALIDATION.md`.
- Bixby cannot be built with the supplied dependencies. Reference-app rendering does not establish full Bixby runtime compatibility.
