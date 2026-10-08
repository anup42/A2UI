# Fold7 sequential Bixby50 renderer review

In progress. Only cases with an explicit completed review are visually verified. [All-50 progress report](PROGRESS.md) · [Progress JSON](progress.json) · [Prepared source mapping](SOURCE_MAP.md).

Device: Samsung Fold7 SM-F966B, serial R3CY30QFWLP. Native SDK view, current trained R64 corrected FP16 model, MTP enabled. For each case, retain the actual model output and replay identical bytes before/after renderer changes. No manual edits to generated Express/JSON. Historical R32 outputs are not evidence for the current model.

The replay test's automated table checks are supplementary: they originally did not recognize high/low temperatures reformatted with units. Every captured page is reviewed visually; an instrumentation pass alone does not establish content fidelity or visual quality.

## BXP-001 — Bengaluru three-day forecast (improved and verified)

Input: exact accepted output from the October 8 corrected FP16 run, SHA-256 `59671e3b384c5410c8dfe6883c1d6b18a51009c8c69b54737c9b4c5a4af4b5db`.

Before: large first-day hero repeats the high/low pair; a chart repeats those values; remaining days occupy individual cards. Default clouds imply daily conditions that were not supplied. The planning advice requires a scroll even with only three days.

Design: put the supplied day/date, high, low, and rain probability in a compact aligned forecast panel. Preserve all other metrics and planning text; show condition imagery only for supplied conditions. Current conditions and a daily forecast are different data shapes. Use a hero only when the shape supports it.

Reference: [Apple Weather daily forecast guidance](https://support.apple.com/en-in/guide/iphone/iph1ac0b35f/26/ios/26) groups daily high/low and precipitation. This is a presentation reference, not a source for weather facts.

Model limitations: the generated Express omits the source citations present in the original answer. Renderer changes cannot recover missing model content.

After: the complete answer fits in the first viewport. All 12 date/high/low/rain cells, the summary, and both planning tips are visible. The same original Express bytes were used for before and after. Dark mode at 1.3 font scale was checked separately without clipping. All 483 shared-library unit tests passed, including five new weather policy tests; the updated coverage-checker instrumentation test also passed on Fold7. Single partial high/low forecast rows now use the panel immediately, avoiding a hero-to-panel switch as streaming adds rows.

## BXP-002 — Delhi AQI (improved and verified)

Generated on Fold7 using `LiteRT-LM/Gemma4/GPU+FP16_CORRECTED+MTP`, without source fallback. The four detail/value rows currently produce four oversized generic cards. The source already contains the reading, scale, observation time, category, and pollutant. A coherent observation panel can prioritize the reading and group its metadata, keeping the full outdoor guidance below.

The supplied answer says `AQI-IN 197` and `Poor`. Preserve these explicit source values; do not infer a category or danger color from an assumed numeric scale. The runtime's literal-content fidelity audit incorrectly treats state-backed table values as missing, so those warnings are recorded but are not treated as evidence of omission without checking the actual output.

After: one AQI panel replaces four large generic cards. The exact reading, three metadata fields, and complete guidance are visible in the first viewport. All eight table cells pass displayed-text coverage and were visually checked. Before/after document bytes match. SDK 0.6.2; 487 shared-library tests passed, including four guarded AQI/classification tests.

## BXP-003 — Train comparison (improved and verified)

The native rail route is already used, but departure time and journey duration have the same weight as every other label. Keep the compact grouped list while prioritizing these two fields. Preserve the full source wording, including `Typical departure time`, `10:35–10:45`, and `About 2 h 15 m to 2 h 25 m`; these are not exact departure/arrival endpoint pairs. No arrival, fare, destination, availability, or booking action is supplied by this table.

After: SDK 0.6.4 presents exact departure ranges prominently, complete duration alongside, and full seating strings in labeled chips. All three services and all four timetable notes plus the offer are retained. Narrow-screen, dark mode and 1.3 font scale were checked; metrics stack at larger text sizes. An older independent malformed-fixture test stopped before rendering because its expected repaired-output hash changed under the intentional no-additional-recovered-text policy. That original test remains unchanged as a separate known failure. The current BXP-003 same-output replay preserved every note and passed.

## BXP-004 — Airline baggage (reviewed and accepted)

SDK0.6.5 groups the existing cabin allowance, checked allowance, charges and exceptions into four compact information panels with smaller headings. Every original paragraph, dimension, amount, per-kg qualifier and fare exception remains readable across captured pages. Exact Express bytes unchanged. This source contains prose, so no invented weight badges, bag counts or fare actions were added.

## BXP-005 — Bengaluru Metro (reviewed and accepted)

The section-panel improvement from004 also organizes this metro guide into operating hours, ticket options, interchanges and rider guidance. Full text is readable across both viewport positions; no map/live service status or ticket actions were invented. No further renderer change needed. MODEL LIMITATION: generated Source: A2UI Express v1 generated model contract is false provenance present in the accepted document; missing citations remain. Visual acceptance is not answer/source-fidelity approval. The first raw replay attempt was a harness-input error (unrepaired raw DSL);r02 and after use the actual accepted A2UI document.

## BXP-006 — Driving routes to Coorg (reviewed and accepted)

Existing route option cards are appropriate: complete waypoint names, blue travel-time emphasis, labeled distance, road conditions and useful stop. Reviewed every scroll page and notes; no need to invent map geometry, live traffic or route selection controls. Source qualifiers and numbered citations retained in model document. No new renderer changes for this sample. The model generated a Continue button without an action; renderer correctly leaves it disabled, which remains a model-output usability limitation.

## BXP-007 — Mysuru sightseeing (reviewed and accepted)

SDK0.6.6 replaces the narrow sightseeing grid with labeled place cards: all three names, opening hours including Tuesday closure, visit durations, and explicit unavailable-fee strings/citations are readable without horizontal scrolling. Full visit order and caveat retained. Table view/Card view switch passed on Fold7 with capture evidence in visual_007_toggle_20261009; table remains available. Same A2UI bytes before/after. Model-generated View all attractions has no action and remains disabled; no navigation/fees were invented.

## BXP-008 — Vegetarian breakfast restaurants (reviewed and accepted)

SDK0.6.7: visually inspected initial and both scroll captures on Fold7. Restaurant name and neighborhood form one header; signature dish now has a restrained highlighted panel; exact cost and full split opening hours/Friday and Sunday closures remain readable, including citations. All three restaurants and final recommendation retained with unchanged A2UI bytes. No photo, rating, open-now state or booking action invented. The source-generated unusual price range is preserved; visual acceptance does not certify its factual accuracy.

## BXP-009 — Jaipur stay areas (reviewed and accepted)

SDK 0.6.8 groups each neighborhood's complete access, transport, atmosphere and drawback paragraphs under its area heading. Card boundaries replace plain divider whitespace; all text/order and saved document bytes remain unchanged across inspected pages. The model omits citations [3], [6] and [4]; the renderer improvement does not restore those.

Evidence: [review](BXP-009/review.json) · [after receipt](BXP-009/after/r01/receipt.json).

## BXP-010 — Wireless earbuds (reviewed and accepted)

Existing product cards already associate each model with its price and complete features/trade-offs. No case-specific renderer change was needed. The later SDK 0.6.9 regression retained all three models, nine table cells, five prose values, live-pricing caveat and complete recommendation without clipping or empty chip space. Same saved bytes; no inferred ANC/battery fields, photos, ratings or purchase actions.

Evidence: [review](BXP-010/review.json) · [latest after receipt](BXP-010/after/r02/receipt.json).

## BXP-011 — Smartphone comparison (reviewed and accepted)

SDK 0.6.9 keeps Display, Battery, Software support and Best for in the same field order across all three phones. A short battery value no longer jumps above Display into a chip, and empty chip-row space is removed. All 18 table values, full price qualifiers, citations and three quick-pick paragraphs remain readable with identical saved bytes.

Evidence: [review](BXP-011/review.json) · [after receipt](BXP-011/after/r01/receipt.json).

## BXP-012 — Note-taking apps (reviewed and accepted)

SDK 0.6.9's existing comparison cards keep every app's offline, synchronization, export and restriction fields readable in consistent order; no main-comparison renderer change was needed. All three recommendations and final export caveat remain visible. The generated graph places the Joplin recommendation outside its Best fit card; that model-authored grouping remains a recorded limitation.

Evidence: [review](BXP-012/review.json) · [after receipt](BXP-012/after/r01/receipt.json).

## BXP-013 — Bank fixed deposits (reviewed and accepted)

SDK 0.6.10 presents bank/rate/deposit information prominently while retaining all 12 table cells, source-date wording, tenure and premature-withdrawal qualifications. All three exact 6.25% rates and Below ₹3 crore buckets remain associated with their banks. Default and independent dark-mode/font-1.3 captures were inspected without clipping; saved bytes match. No new date, ranking or penalty fact was derived, and acceptance does not certify current rates.

Evidence: [review](BXP-013/review.json) · [after receipt](BXP-013/after/r01/receipt.json).

## BXP-014 — Foreign-exchange quotes (reviewed and accepted)

SDK 0.6.11 introduced the compact aligned currency/rate panel; the latest SDK 0.6.12 regression preserves the same content pixels. Exact rates 95.5614, 111.2956 and 129.8867 and the shared Quoted As Of timestamp 28-Aug-2026 14:23:47 IST remain clear, with the complete reference-versus-traveler explanation below. All nine supplied cells survive; repeated timestamps required manual scope checking. The model omits citations [18], [12] and [3]. Visual acceptance does not certify current rates or full source fidelity.

Evidence: [review](BXP-014/review.json) · [latest after receipt](BXP-014/after/r02/receipt.json).

## BXP-015 — Cricket match summary (reviewed and accepted)

SDK 0.6.12 groups the supplied date/venue, opponent and format, emphasizes both teams' exact score strings and result, and keeps both standout paragraphs readable. Default and independent dark-mode/font-1.3 captures passed explicit visual/prose checks for all ten Text values with identical saved bytes. The source itself supplies only July 2026 and the unusual score 193/48.5; preserving them is not a model conversion defect or renderer error. Sporting facts and match recency are not certified.

Evidence: [review](BXP-015/review.json) · [after receipt](BXP-015/after/r01/receipt.json).

## BXP-016 — Recent Hindi streaming films (reviewed and accepted)

SDK 0.6.13 honors the declared cards/primary-column intent: three full film cards replace a horizontal grid that hid synopses. Titles, platforms, exact streaming dates, complete synopses with [33]/[21]/[22] and the provenance paragraph with [1] are visible without horizontal scrolling. Saved JSON is unchanged; no artwork, ratings or playback actions were added. The unused platformReleaseDate field duplicates dates in state; acceptance does not verify factual recency.

Evidence: [review](BXP-016/review.json) · [after receipt](BXP-016/after/r01/receipt.json).

## BXP-017 — Book-query limitation (reviewed and accepted)

No renderer change was needed on SDK 0.6.13: the limitation heading and all three explanatory paragraphs fit without clipping or scrolling. Inspected before/after content pixels match. The original source declined to provide a verified book list, so the graph contains no books, covers, authors or actions for book cards. Presentation acceptance does not mean the recommendation query was answered.

Evidence: [review](BXP-017/review.json) · [after receipt](BXP-017/after/r01/receipt.json).

## BXP-018 — Billboard Global 200 chart (reviewed and accepted)

SDK 0.6.14 replaces the playlist view that dropped chart dates and split combined song/artist text incorrectly with a compact ranked chart. Every rank and full unsplit Song Title, the exact shared Chart Week Date and both observation paragraphs are visible. The same generated JSON is used before/after. The model's rank-4 entry loses source uncertainty, repeats rank 1 and supplies a date; citations [4], [13] and [1] are absent. The renderer preserves that data without certifying chart accuracy.

The failed before receipt remains failed with `renderedWithoutIssues=false` and `allTableColumnsObserved=false`. The new explicit `--allow-before-render-failures` option is limited to after comparisons against a preserved baseline whose known failures are render/column checks. Its receipt records exactly those two overridden baseline checks. Every after check remains required and passed; the waiver does not accept the baseline or waive after failures. Both before and after captures were visually inspected.

Evidence: [review](BXP-018/review.json) · [failed before receipt](BXP-018/before/r01/receipt.json) · [strict after receipt](BXP-018/after/r01/receipt.json).

## BXP-019 — Family co-op PC games (reviewed and accepted)

SDK 0.6.14's existing game cards require no renderer change. All three games retain full online/account age caveats, local/online/solo co-op details, player counts and exact indicative/past-promotion/US-dollar price qualifiers. Three before and three after captures were inspected; all 15 table cells, six literal Text values, ten citations, recommendations and pricing caveat remain readable with identical saved bytes. The two repeated Players=2 values were manually checked in their own cards. This does not verify current prices or authoritative age ratings.

Evidence: [review](BXP-019/review.json) · [after receipt](BXP-019/after/r01/receipt.json).

## BXP-020 — Planned ISRO launch (reviewed and accepted)

The mission panel emphasizes the supplied identity, pairs date and vehicle at normal width, and stacks them at larger text sizes. Complete payload purpose, provisional-schedule uncertainty and official-versus-third-party qualification remain readable. Both normal pages and both dark-mode/font-1.3 pages were reviewed; replay checks pass and source JSON bytes are unchanged. The provisional date was retained, with no confirmed-launch claim added. The model omits citations [2], [10], [1] and [5]; acceptance does not certify mission recency or source fidelity.

Evidence: [review](BXP-020/review.json) · [after receipt](BXP-020/after/r01/receipt.json) · [dark/font-1.3 result](BXP-020/visual_020_dark_large_20261009/BXP-020/replay_result.json).

## BXP-021 — Rakhigarhi archaeology (reviewed and accepted)

Existing shared section panels already give discovery, dating evidence, significance and established-versus-speculative findings a clear hierarchy; no further renderer change was needed. All three before and three after captures were reviewed, retaining every paragraph and uncertainty sentence through the end with identical saved bytes. No structured date or image was invented from prose. The generated document omits source citations [2] and [10]; this remains separate from rendering acceptance and factual recency was not evaluated.

Evidence: [review](BXP-021/review.json) · [after receipt](BXP-021/after/r01/receipt.json).

## BXP-022 — Bengaluru balcony tomatoes (reviewed and accepted)

SDK 0.6.17, after/r02, presents all five care sections as consistent compact cards. Container size and common problems no longer sit in detached blocks; redundant heading/wrapper padding is removed. All three captures and all 15 original Text values were visually checked, retaining inch/litre/hour ranges, qualifications and citations. Original wire bytes and unreachable x/y/z definitions remain unchanged; the first unchanged after/r01 attempt is preserved. Fourteen SDK tests passed, including the saved wire through production host-gutter normalization.

The [gardening test fixture](../../genuicraft/src/test/resources/rendering/bxp022_gardening.a2ui.json) is an exact 3,225-byte copy of the [archived native A2UI JSON](native_generation/visual_prep_016_022_20261009/BXP-022/a2ui.json), not a manually rewritten IR or the raw Express file. Both have SHA-256 `c092340f4d4c8467d8b6683126f4761be782becbbe807f4793b717da9e8f3ce2`, matching the reviewed before/after input. The fixture retains original padding and unreachable definitions so the test exercises actual normalization behavior.

Evidence: [review](BXP-022/review.json) · [latest after receipt](BXP-022/after/r02/receipt.json) · [preserved first after receipt](BXP-022/after/r01/receipt.json).

## BXP-023 — Indoor-cat preventive care (reviewed and accepted)

Existing guide sections on the same recorded SDK 0.6.17 APK require no additional renderer change. All three before and three after captures retain all four sections, complete paragraphs, interval/risk/veterinarian qualifiers and citations with identical saved bytes. The renderer adds no imagery, completed checkmarks/state, vaccine dates or reminder actions. The model adds “Checklist complete.” and “Please consult your veterinarian for personalized advice.” absent from source; these remain plain generated text and are not evidence of completed care. Clinical accuracy was not evaluated.

Evidence: [review](BXP-023/review.json) · [after receipt](BXP-023/after/r01/receipt.json).
