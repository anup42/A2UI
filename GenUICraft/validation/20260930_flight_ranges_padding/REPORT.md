# GenUICraft 0.5.7: flight ranges and horizontal spacing

The shared AAR now renders the captured combined-time flight table through native
flight cards and removes the extra nested horizontal inset in the current Bixby
answer. The Android demo mirror and its SDK dependency are updated. The local
Bixby checkout consumes 0.5.7; its proprietary APK still needs rebuilding.

## Changes

- Parse explicit two-time ranges separated by hyphen, en/em dash, arrow, or `to`.
  Preserve AM/PM, airport codes and explicit `+1` offsets. Reject invalid clocks,
  alternatives such as `04:40 or 07:10`, and ranges with more than two times.
- Let combined-time flight tables enter the dedicated renderer without requiring
  separate Departure and Arrival headers. Preserve supplied separate arrival data.
- Keep duration and stops when both occur in Details. Flight numbers containing
  four digits are no longer mistaken for bare fares. Do not infer nonstop service
  merely because two times exist.
- Retain unused fields such as Date and citation-only Booking values in the card.
  Citation IDs use source previews and never become booking URLs.
- Remove horizontal padding/margins from the outer full-width layout wrappers,
  including nested sections beneath an unpadded root. Preserve vertical spacing,
  fixed-width panels, card interiors and side-by-side cells. Flight-card horizontal
  content padding is 10 dp across native, ranked and itinerary routes.

## Device verification

Device: Flip8 SM-F776U (`R3GL203AKSF`), 1080 px wide, density 480 dpi.
The separate `com.samsung.genuicraft.compatqa` app uses the same Compose versions
as the previous Bixby compatibility QA. It replays **exact stored repaired Express
and A2UI bytes**, with GenUiView embedded mode, hidden Sources section and a 24 dp
host gutter. These are renderer checks, not new model accuracy measurements.

| Check | SDK 0.5.6 | SDK 0.5.7 |
|---|---|---|
| Combined-time example | Generic entity cards | 3 native flight cards |
| First flight | Departure `04:40–07:10` chip | Depart `04:40`, arrive `07:10` |
| Text/card left edge | 120 px | 72 px |
| Text/card right edge | 960 px | 1008 px |
| Usable width | 840 px | 936 px (+11.4%) |
| First paragraph height | 398 px | 329 px |
| Extra nested horizontal padding | 16 dp per side | 0 dp |
| Source `[5]` | Generic field | Tapping opens the supplied easemytrip.com source preview |

See [before/after screenshots](index.html), [source preview](source_preview.png),
and [verification hashes and bounds](verification.json). Saved IR was not rewritten.
The source website was not opened and no booking action was performed.

## Build and tests

- SDK targeted regression suites: **71/71 passed**.
- Release AAR and versioned Maven publication: built successfully.
- Compose FlowRow ABI scan: passed (548 classes).
- Android demo source and unit tests: compiled; **70/71 passed**. The unchanged
  `extractDirectTableModel_routesUiStateTableToProcessCards` test expects process
  cards despite explicitly requesting `preferredPresentation="table"`; the
  renderer returns `TABLE_HORIZONTAL_SCROLL`. All flight and padding tests pass.
  This unrelated test/policy mismatch was left unchanged.
- QA APK built and installed with 0.5.6 and then 0.5.7; both captured inputs replayed.
- Bixby APK was **not rebuilt or replaced**. The existing installed Bixby will keep
  the old rendering until it is rebuilt with 0.5.7 and installed.

## Bixby handoff

`GenUICraft/artifacts/Bixby-GenUICraft-Flight-Padding-0.5.7.zip` contains only
`Renderer/build.gradle.kts` and the 20 files in the 0.5.7 Maven publication under
`aars/genuicraft-maven/com/samsung/genuicraft/genuicraft/0.5.7/`. Archive entries
preserve the Bixby-relative paths and were byte-verified. It is an incremental
update for the already integrated Bixby checkout, not the complete earlier patch.
Merge at the Bixby source root and rebuild/install using the private dependencies.
No Bixby commit or push was performed.
