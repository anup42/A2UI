# GenUICraft SDK 0.6.45 validation snapshot

28 October 2026 review; additional pre-publish checks completed 29 October 2026. This folder is a compact, portable review of the completed SDK validation. All links below resolve inside this folder. The original report, raw responses, logs, preferences, install packages, and model weights are intentionally absent. Counts and measurements are transcribed from the saved validation results; the five images are selected final device replay viewports.

## What the review established

- The phone comparison's malformed generated Express output is repaired without losing its model-produced table. Its raw output hash stayed identical across baseline, pilot, and final runs. The final A2UI has three rows and four columns, and the saved full-page replay found all three phone cards.
- Complete morning flight offers render as native route, time, and fare cards. A qualified 12:40 option outside the morning window remains a generic details card. Missing fare is shown as missing, rather than invented.
- The copied three-city weather response renders as city cards; the distinct completed Bixby Markdown response renders as a 3 × 5 table. Train comparisons default to cards while an explicit feature matrix retains its grid.
- SDK 0.6.45 Maven publication completed. The local delivery check matched the APK, AAR, and corrected FP16 model hashes; the installed Bixby app was not updated in this review.

## Tests

| Test set | Passed | Interpretation |
| --- | ---: | --- |
| Core JVM suite | 131 | Compiler, recovery, and native domain semantics |
| Flight JVM suite | 23 | Offer, fare, and presentation semantics |
| Presentation JVM suite | 17 | Repair and weather, phone, and flight presentation |
| **Distinct JVM cases** | **142** | 171 suite case counts minus 29 repeated cases; 12 distinct test classes |
| Initial device instrumentation | 7 of 8 | One train UI Automator capture failed with `StaleObjectException` at `UiObject2.getText` |
| Targeted train recheck | 2 of 2 | Both train cases passed, including the initial failure; all 8 distinct device test names passed across the two runs |
| Later card scroll check | 2 of 2 | Flight and phone checks passed again; these repeat existing device tests |

The initial device run was **not** a clean 8 of 8 pass. The train recheck establishes recovery of the capture test, and the card scroll check verifies saved content in additional viewports. Captured replay tests made zero model calls.

An additional pre-publish SDK check passed **59 of 59 tests across 9 classes** in 54 seconds, with zero failures, errors, or skips. It is a separate check and is **not added** to the 142 distinct-case total because overlap has not been reconciled. Class counts: `Gemma270PromptContractTest` 4; `GenUiPresentationTest` 4; `GenUiSessionTest` 11; `EntityListPresentationTest` 6; `NativeWeatherArtworkTest` 5; `Gemma270MProviderTest` 3; `Gemma4StreamingTest` 12; `ManagedGemmaRuntimePoolTest` 10; `GenUiResourceSummaryTest` 4.

The pre-publish app check also passed **29 of 29 tests** in 1 minute 5 seconds, with zero failures or skips: `GenerationMetricsUiTest` 7; `GenUiSdkModelUiTest` 15; `OnDeviceModelCatalogTest` 7. This check is reported separately from the JVM and device totals above.

## Final measured SDK runs

These are five completed conversions of four distinct saved inputs on the corrected FP16 GPU/MTP SDK path. All five had a valid repaired document, successful screenshot smoke check, and `usedFallback=false`. Conversion time excludes separate engine warmup; these individual values are not a controlled speed benchmark.

| Run | Input | Conversion | Raw strict | Repair | Source warnings |
| --- | --- | ---: | --- | --- | ---: |
| `after_first` | Completed weather Markdown | 6.793 s | Yes | None | 0 |
| `after_first` | Copied weather text | 7.677 s | Yes | None | 0 |
| `after_first` | Morning flights | 6.092 s | Yes | None | 3 |
| `after_live` | Morning flights, same input | 7.624 s | Yes | None | 3 |
| `after_live` | Phone comparison | 19.525 s | No | Generated DSL repair | 0 |

The screenshot smoke check means the case view appeared; it does not grade every pixel or prove complete source fidelity. The saved flight warnings are real even though the output rendered.

## Selected final screenshots

These are SDK app replays of captured output, labeled “no inference” in the app. They illustrate displayed presentation, while the measured conversions above were separate model runs. Each image is a viewport, so later cards may require scrolling.

| View | Image |
| --- | --- |
| Copied weather text, first city card | [weather-copy-cards.png](weather-copy-cards.png) |
| Morning flight, first native card | [flight-first-card.png](flight-first-card.png) |
| Later flights and qualified 12:40 option | [flight-later-qualified-card.png](flight-later-qualified-card.png) |
| Phone comparison, first card | [phone-first-card.png](phone-first-card.png) |
| Phone comparison, later card | [phone-later-card.png](phone-later-card.png) |

## Remaining limits

- Both final flight conversions retain **three source-fidelity warnings**: the model's raw output omitted introductory citation markers `[5][6][11]`. Rendering did not resolve those omissions.
- Historical `BXP-047` captured output still contains malformed citation text `NMC[26`; the strict source gate rejects it. No repair is claimed.
- In the observed live Bixby weather request, Bixby used its native Weather Channel result and bypassed GenUICraft. The Mysore itinerary request hit the 8,192-token context guard with an estimated 13,127-token input. The host's View-only payload handling is a possible contributor, but the exact installed-host prompt was not verified.
- The updated Bixby integration was not installed or tested end to end. The earlier live Bixby observations used a different device/model path from these SDK replays. Flight, price, and weather facts from the upstream answer were not independently verified.

The source verification record identifies SDK 0.6.45 artifacts by SHA-256: AAR `164f5efbd5e0e0ee5f4ef99d7f32f56ed3bc60d4388908f98c45eda6dac639c6`, demo APK `a1e179cd4b56f66203ad2851032d6a9ba67303d4aba8cb1b8b1df767b7a05a3a`, and corrected FP16 model `26aa7eedbdfc4fe80321edc7a24aa1cfe85057739e03ffb1b7de7ac5bdb8fdb3`. These binaries are not part of this folder.
