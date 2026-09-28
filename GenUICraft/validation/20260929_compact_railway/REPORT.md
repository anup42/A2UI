# Compact railway cards — SDK 0.5.3

The same three-train fixture occupies **687 px instead of 1,074 px (36.0% less height)** on the connected SM-F776U at font scale 1.0. All displayed train names, numbers, departure values, journey values and the inline citation remain visible. See the [visual comparison](comparison.html).

## Cause and change

The current `Service / Train / Departure / Journey` table missed the native rail route because only one detail heading matched its required semantic roles. Generic comparison cards then repeated `Service`, boxed each field as a padded chip and left large gaps between trains. The older native rail cards also used separate ticket surfaces and oversized headers.

The shared SDK now uses one grouped surface with compact train rows, small train icons, clear service names, labeled wrapping details and thin separators. It omits a redundant default heading when the enclosing card already supplies the route. Clear railway tables with generic details can use this route; explicit rail evidence, service identity and a station/departure field are still required. Flight, weather and product identities remain excluded. Bare `Departure` is preserved when its meaning is ambiguous. Generated values are never interpreted as timetable facts, corrected or sorted.

The app's reference renderer is synchronized, and its demo consumes the published **0.5.3 AAR**.

## Evidence

| Check | Result |
| --- | --- |
| Same synthetic source fixture, same device and font scale | Train block 1,074 → 687 px; 36.0% shorter |
| Exact fixture values | All retained and visible in the UI hierarchy |
| Tap `[1]` inside the compact departure field | Correct Indian Railways source preview opened |
| Text scale 1.3 | Details wrap without horizontal clipping; all three trains visible |
| System text scale after QA | Restored to the original 1.0 |
| Previously captured, richer model output | Three services and every supplied field value retained |
| Targeted JVM tests | 19 passed: train semantics 8, explicit table presentation 8, citations 3 |
| SDK release publication | Built successfully |
| QA harness and main A2UI demo | Built and installed on the connected device |

The historic model document was replayed unchanged from `20260922_train_card_polish_fold7/repaired.express`; no inference or manual IR editing was used for this layout check. Its malformed generated times remain unchanged. This is a renderer comparison, not a model accuracy benchmark.

Measurement uses the table/container bounds from Android's UI hierarchy. Before and after screenshots have different scroll offsets and source disclosure states; the 36% comparison covers only the same three-train block, not the entire screen. Raw measurements are in [comparison.json](comparison.json).

- [Before screenshot](before.png)
- [After screenshot](after.png)
- [Citation preview](citation.png)
- [130% text size](large_text.png)

## Integration and limits

The Bixby checkout is updated to the complete SDK 0.5.3 Maven publication. The cumulative `Bixby-GenUICraft-Compact-Railway-20260929.zip` also retains the earlier answer-slot, source preview and nullable conversation-ID fixes. Merge its directories into the Bixby source root and rebuild in the private Bixby build environment.

The rebuilt Bixby APK has **not** been tested in this change. Device checks above use the actual published SDK in the standalone native QA host; the main A2UI demo build/install also succeeded. No model settings, prompt, repair behavior or inference runtime settings changed.

AAR SHA-256: `8be3df8f21b28633b92fe040a1a220cba16e1918ae04e66e9a52f06d2bb6ab03`.
