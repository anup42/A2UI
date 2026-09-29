# Bixby50: shared renderer and Express recovery improvements

SDK **0.5.6** recovers generated content that was disconnected from the visible root, preserves table labels and row relationships, and improves native card presentation. This is a controlled replay of saved E2B outputs, not a new inference benchmark.

Open the [interactive before/after gallery](before_after.html) for every case's screenshots, original model Express, repaired Express, and frozen source Markdown. The [machine-readable comparison](comparison.json) verifies identical raw bytes across the two SDK versions. The [source audit](audit/REPORT.md), [visual review](audit/VISUAL_REVIEW.md), and [independent code review](audit/CODE_REVIEW.md) separate renderer defects from model omissions and corrupted values.

## Results and scope

| Saved model population | Cases checked | SDK 0.5.5 rendered | SDK 0.5.6 rendered | Cases with newly reachable generated Text |
|---|---:|---:|---:|---:|
| Original E2B v10 W4, GPU + MTP, 21 September | 50 | 46 | 48 | 0 |
| Corrected rank-64, GPU FP32, MTP off, 28 September | 15 | 15 | 15 | 6 |
| Corrected rank-64, GPU FP32, MTP on, 28 September | 12 | 12 | 12 | 4 |

The two old-model blank renders, **BXP-013 and BXP-027**, now show the one-column content the model actually produced. These are partial answers: BXP-013 contains only a bank/SBI fragment, while BXP-027 contains two tip fragments. **BXP-009 and BXP-039** remain unrecoverable because their raw outputs contain repetitive gibberish rather than usable answer content. No source-derived fallback was used.

The older archived report counted 48 recovered documents as rendered. This comparison uses fresh SDK 0.5.5 screenshots and a visible-text check, which exposed the two compiled-but-blank cases and established the 46/50 baseline.

Across the newer rank-64 outputs, 44 additional generated Text components become reachable in ten captured outputs. This is a structural count, not a count of verified facts. All previously reachable Text components remain represented. Repair now also handles some strictly valid programs whose graph hid answer content: generated repair is used in 9/15 MTP-off and 8/12 MTP-on outputs, compared with 3/15 and 4/12 before. The original raw strict-compile rates remain 12/15 and 8/12; a renderer update does not change model generation quality.

**A render is not a faithful conversion.** Mechanical source-integrity acceptance remains 0/48 recovered historical outputs, 3/15 rank-64 off, and 1/12 rank-64 on. That diagnostic is deliberately strict and is not a semantic accuracy score. The independent source review also confirms real missing caveats, lost citations/units, changed digits, and incomplete answers.

## Shared library changes

- **Recover hidden generated content.** Opt-in generated DSL repair detects static components disconnected from the root, reconnects eligible subtrees, and removes duplicate root references/cycles. Conditional and repeated templates stay guarded. Legitimate identical values and shared elements across separate cards remain intact.
- **Keep usable generated structure.** Complete repaired graphs with covered state bindings take precedence over lossy flattening. Table recovery retains authored column labels, order, title, and presentation metadata when generated data matches the schema. Unused generated state still goes through salvage rather than silently disappearing.
- **Make comparisons easier to scan.** Phone prices and requested highlights lead each card. Long values use full-width labeled detail rows instead of crowded pills. Bank, restaurant, and app entity rows use responsive cards; explicit table requests remain tables.
- **Fix missing table content.** One-column tables display their nonblank values. Ranked text rows use entity cards. Metric cards retain all nonblank fields and their labels, including columns beyond the primary number. Entity and metric card routes also show the generated table title.
- **Improve text hierarchy.** Long explanatory sentences incorrectly tagged as headings use body typography, while short titles remain headings. Existing embedded scrolling, compact railway layout, and citation behavior are preserved.

These changes live in the shared SDK. The Android app's native renderer mirror receives the corresponding presentation changes. Model inference, prompt, GPU FP32 configuration, and MTP configuration were not changed. No manually rewritten case IR or additional recovered-text card was introduced.

## Useful examples

| Case | Improvement |
|---|---|
| Rank-64 BXP-002, air quality | PM2.5, citations, and outdoor guidance were generated but invisible; they are now reachable and visible. |
| Rank-64 BXP-003, trains | The long introduction uses body text; compact train details remain readable. |
| Rank-64 BXP-004/005, baggage and metro | Generated fare caveats, pass options, interchanges, and rider guidance become visible. |
| Rank-64 BXP-008, restaurants | Full column labels and title order survive repair; readable cards replace a clipped wide table. |
| Rank-64 BXP-011, phones | Price leads each comparison card; specifications and long price qualifications remain visible. |
| Rank-64 BXP-013, deposits | The answer heading precedes the content, and bank cards retain full rate/size/date labels. The quick-view section still follows the model's long caveats. |
| Historical BXP-018, music chart | The old metric route hid artist names and dates. Entity cards now show the generated title, artist, rank, chart value, and date. Corrupted generated date spelling is preserved rather than guessed. |

## Remaining problems

Repair cannot reconstruct facts that the model never generated. Rank-64 MTP-on BXP-001 loses percent units; BXP-009 omits a neighborhood downside and nests one area within another; BXP-012 omits an uncertainty caveat. These remain visible in the report.

The historical model has much larger problems: truncated answers, corrupted amounts/percentages/dates, merged words, and malformed grouping. Examples include BXP-029/032/047 numerical changes and BXP-034/041/048 quantity/date corruption. Rendering such text successfully does not make it correct. Several explicitly requested wide tables still require horizontal scrolling. Reconnecting static orphans also infers intent from graph structure; ambiguous grouping is not fully solvable without better generation.

## Validation and delivery

- **428 SDK tests passed**, with no failures/errors, including the captured graph/table cases, guarded/shared references, one-column tables, chart field completeness, and genuine multi-column metrics.
- Release AAR passed the Compose Flow ABI guard. Final artifact hashes and build evidence are in [build_validation.json](build_validation.json).
- Demo app built and installed on **Fold7 SM-F966B, R3CY30QFWLP**. Device captures use the cover display, light theme, and font scale 1.0. The replay harness adds 48 dp top/bottom clearance and 16 px side padding; these are not Bixby's embedded view margins. No independent dark-theme, unfolded-layout, rotation, latency, or new-inference claim is made here.
- All 77 saved raw outputs are byte-checked against their input manifests and replayed before and after. Each replay records no model calls and no source-text fallback. Screenshots include bounded vertical/horizontal traversal; visual and source-fidelity reviews are separate from the render status.
- The Bixby checkout at `C:\Users\anupk\Downloads\Bixby_18Sep` now references the local 0.5.6 Maven AAR. Bixby was **not rebuilt or installed** here and its source was not pushed. Rebuild its APK to receive these changes.

The connected Fold7 has older E2B files, but not the latest corrected rank-64 artifact (`de60d19c8e1ef06ed1032212d0d4ae5d9b0ec105db72db34a988453004e63e62`), which was last tested on the other device. Therefore this work evaluates saved outputs from those models. A fresh full50 score for the latest rank-64 model remains unmeasured.

To reproduce, use `tools/stage_presentation_review.py` to stage the archived bytes, push the resulting input run directories into the demo app's external `sdk_benchmark` directory, and run `tools/run_presentation_replays.py` against each installed SDK version. Run `tools/report_presentation_review.py` after both phases. The first 0.5.6 replay exposed the music-chart defect; its intermediate captures were retained locally and excluded from the final comparison.
