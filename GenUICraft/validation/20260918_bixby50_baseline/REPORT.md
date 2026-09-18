# Bixby50 device baseline

This run records the user-visible Bixby output for every query in the Bixby50 corpus so it can be compared with the GenUICraft A2UI replay.

## Setup

| Item | Value |
| --- | --- |
| Device | Samsung SM-F776U (`R3GL203AKSF`) |
| Bixby package | `com.samsung.android.bixby.agent` |
| Bixby version | `5.0.10.28` |
| Corpus | `android/app/src/main/assets/genuicraft_bixby50.jsonl` |
| Corpus SHA-256 | `fc46aa381957bed206f9e0f53e28edbb21b5096ca093985ad184fda73faaea0a` |
| Cases | 50 exact queries, each submitted in a fresh Bixby conversation |
| Completion timeout | 90 seconds per query |
| Capture | Device-side PNG after completion plus a 6-second render allowance, or at timeout |

## Result

| Visible result state | Cases |
| --- | ---: |
| Backend completion marker observed, visible renderer still on progress | 49 |
| Completion timeout (`BXP-040`) | 1 |
| Visible completed answer | 0 |

For the 49 completed backend runs, completion took 7.000–23.297 seconds. The median was 12.171 seconds, the mean was 12.213 seconds, and the nearest-rank p95 was 17.922 seconds.

The screenshots are the actual visible result on this device and installed Bixby build. They show the submitted prompt followed by the persistent “Generating an answer with AI” / search progress state. The backend completion marker alone does not prove that a usable answer reached the renderer. This run therefore supports a usability comparison—GenUICraft displays the generated A2UI while this Bixby build remained on progress—but it does not provide completed Bixby answer cards for an aesthetic comparison.

## Evidence

- `screenshots/BXP-001.png` through `screenshots/BXP-050.png`: active, unobstructed 1080×2520 captures.
- `comparison_gallery.html`: searchable side-by-side gallery using these Bixby captures and the existing GenUICraft A2UI `after/initial.png` captures.
- `contact_sheet.png`: overview of all 50 active Bixby captures. Amber means backend completion with the renderer still on progress; red means timeout.
- `manifest.jsonl`: one record per case with query hash, timestamps, completion state, render state, and artifact hashes.
- `bixby50_queries.jsonl`: exact corpus copy used by the run.
- `dom/` and `ui/`: captured WebView text and UI hierarchy for prompt-to-capture verification.
- `summary.json`: aggregate result and timing summary.
- `audit.json`: artifact audit result.

The final audit found 50 ordered records, 50 valid PNGs, 50 unique screenshot hashes, and an exact corresponding query in all 50 DOM captures. It found no active artifact issues. Early keyboard-obstructed attempts were discarded and are excluded from the manifest, contact sheet, and comparison gallery.

Raw logcat was not retained because it can contain account and service authentication data. It was consumed only as a live completion signal.
