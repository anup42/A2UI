# Mysore place cards — SDK 0.6.46

Implemented and installed in the GenUICraft test app on Flip8 (SM-F776U, R3GL203AKSF). Replayed the captured Bixby answer to **Plan a relaxed two-day trip to Mysore from Bengaluru** through the public SDK and real SDK demo page.

## What changed

- Added an SDK-owned importer for supported structured Bixby View responses. Existing View data is converted directly to A2UI Express and validated A2UI JSON; ordinary Markdown continues through the model pipeline.
- Added native PlaceCard rendering: rounded cards, title, gold rating star and review count, linked address, horizontally swipeable photos, and Call/Directions/Website chips. Vertical scrolling belongs to the parent screen.
- Preserved source order and exact photo URLs, optional photo click targets, ratings, review counts, addresses, and action labels/URLs. Missing ratings stay missing; Mysore Palace does not gain an invented rating.
- Fixed Markdown headings being mistaken for shell output. Explicit Markdown stays prose; actual fenced code and shell/Python samples remain supported.
- Added a demo button under **GenUICraft SDK · Bixby50 → Try Mysore place cards · captured Bixby answer**. It needs no model download and labels the operation Import / Validate / Render.

## Verified result

| Check | Result |
|---|---|
| Place cards | 6/6 visible |
| Photo URLs retained exactly | 18/18 |
| Photo click URLs retained exactly | 9/9 |
| Action labels and URLs retained exactly | 11/11 |
| First photo loaded | 6/6 cards |
| First card carousel | Third photo reached |
| Directions | Exact source URL reached host callback |
| Model calls / provider creation | 0 / 0 |
| Repair or fallback | None |
| Conversion status | Place cards ready · 32 ms · no model inference |
| SDK JVM tests | 47 passed |
| App model-readiness tests | 16 passed |
| Python IR format tests | 12 passed; 21 generated artifacts verified |
| Device instrumentation | 2 passed |

The conversion time above measures local import and compilation, not image-download completion. All 18 URLs were checked for preservation; the device test verifies each card's first photo and the first card's third photo, rather than claiming every photo was downloaded. Actions were intercepted for validation, without placing calls or opening external sites.

## Evidence and boundaries

- [Device results](device_final/result.json), [imported Express](device_final/sdk_import.express), [A2UI JSON](device_final/sdk_import.a2ui.json), [captured input](device_final/captured_bixby_view.json), [binary hashes](provenance.json).
- [Final SDK build](sdk-delivery-build.log), [app build](app-delivery-build.log), [install](app-delivery-install.log), [device test](device-delivery-test.log).
- This is a real device replay of captured Bixby content, not a fresh Bixby query or an E2B quality score. Photos and facts come from that captured answer; no image search or generation was added.
- Bixby itself was not changed or installed in this task. Its host must consume SDK 0.6.46 and pass the structured View payload rather than flattening it to plain text.
- The new PlaceCard route is in the shared SDK renderer used by this demo/AAR; the app's separate legacy renderer is not the tested route.
- Remaining text-style limitation: the existing general Text renderer can leave inner Markdown heading markers visible in a multiheading paragraph (for example, Course highlights). The place-card fields and photos are unaffected.
- The extra page_12 capture is a manual follow-up from the demo button, showing the complete zoo card; other device_final pages are from the automated replay.

## Card screenshots

- [Sri Ranganathaswamy Temple](device_final/page_03.png)
- [Brindavan gardens mysore](device_final/page_05.png)
- [Aishwarya Le Royal (Best Budget Hotel in Mysore)](device_final/page_07.png)
- [Chamundi Hill](device_final/page_09.png)
- [Mysore Palace](device_final/page_10.png)
- [Mysuru Zoo](device_final/page_12.png)
