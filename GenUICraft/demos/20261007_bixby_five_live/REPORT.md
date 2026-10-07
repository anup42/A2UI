# Five live Bixby GenUICraft demos — 7 October 2026

The clearest benefit is turning wide, multi-field answer tables into cards that fit the phone width. Flights and trains are the strongest demonstrations: fare and timing fields stay visible together. Product and seasonal-weather rows also become labeled groups. Restaurant cards bring cost, hours and address into view, with the visual limitations below.

All five selected videos contain actual device footage: a fresh Bixby request, on-device A2UI Express streaming, generated output, and a same-answer switch to the original Bixby view. No model output was manually rewritten and no screenshot slideshow was substituted for the recording.

| Scenario | Video | Observed benefit | SDK repair | Warnings |
|---|---|---|---|---|
| Flights | [Bixby_GenUICraft_Flights.mp4](Bixby_GenUICraft_Flights.mp4) | Airline logos, fare, duration and departure/arrival times together. | NONE | 10 |
| Trains | [Bixby_GenUICraft_Trains.mp4](Bixby_GenUICraft_Trains.mp4) | Four compact rail services with fares and journey times visible. | GENERATED_DSL_REPAIR | 13 |
| Phone comparison | [Bixby_GenUICraft_PhoneComparison.mp4](Bixby_GenUICraft_PhoneComparison.mp4) | One card per phone with labeled price, battery, camera and update support. | GENERATED_DSL_REPAIR | 12 |
| Seasonal weather | [Bixby_GenUICraft_WeatherComparison.mp4](Bixby_GenUICraft_WeatherComparison.mp4) | Month, high/low temperatures, rainfall and conditions within the phone width. | NONE | 10 |
| Restaurant comparison | [Bixby_GenUICraft_RestaurantComparison.mp4](Bixby_GenUICraft_RestaurantComparison.mp4) | Cost, opening hours and address grouped with each restaurant. | GENERATED_DSL_REPAIR | 11 |

## Code review before testing

- `GenUICraft/genuicraft/src/main/java/com/samsung/genuicraft/sdk/GenUiTrainedConverter.kt:62` runs the model then calls the shared SDK compiler/recovery. The frozen `e2b_v10_shared_prompt.json` remains the trained-model prompt; no prompt changes were made for these recordings.
- `sdk/internal/renderer/flat/compose/FlatDirectTableRender.kt:250` honors explicit table presentation; specialized train, itinerary, flight and adaptive routes follow. This is why a strict table output does not automatically become cards.
- `sdk/internal/renderer/native/intents/train/NativeTrainSemantics.kt:26` recognizes rail identity and travel fields. The visible selected rail layout groups services compactly.
- `sdk/internal/renderer/FlatSpecRenderer.kt:5632` selects adaptive feature/entity/metric/timeline presentations based on shape and screen width. Entity comparisons are more promising than unstructured prose.
- `sdk/internal/renderer/FlatRestaurantBookingCards.kt:174` recognizes restaurant rows with location/rating information and provides a dedicated card route.
- Native weather, itinerary, recipe and checklist routes were candidates. A rich existing Bixby places response is a weaker conversion candidate than a concise Markdown answer because of context size and existing native presentation.

## Device and native execution evidence

Installed Bixby: **5.0.10.50**, package `com.samsung.android.bixby.agent`, connected **SM-F776U**, portrait 1080 × 2520. GenUICraft was enabled, precision `fp16_corrected`, MTP enabled. The first run's native engine log explicitly confirms `backend=GPU; precision=FP16_CORRECTED; MTP=true; MTPRequested=true; modelSupportsMtp=true`; subsequent conversions reuse the same engine and emit the scoped FP16 policy evidence.

Corrected model SHA-256: `7f01bdf1c6ba9bdf658e57c75001fc35a42edad88238bc5dab0a5f7dcf3de373`. The Q/DQ adapter reports 206 blocks, zero rejections. All selected conversions completed in **one model attempt**, with `NONE` or `GENERATED_DSL_REPAIR`, rather than a source fallback. Per-video success evidence is in [runtime_evidence.txt](runtime_evidence.txt). Tokens/s is not claimed: Bixby's integration disables detailed native metrics.

Selected key names, prices, times and units were checked against the same original-answer UI tree. This is a bounded preservation check, not a model accuracy score or independent verification of Bixby's travel, weather, product or restaurant facts. Compiler warning counts are retained above; a successful conversion does not mean perfect fidelity.

## Candidates not selected

There were **11 live capture/test runs across 10 distinct queries**. Five were selected for a visible presentation benefit. Other runs remain in local `evidence/`:

- Phone feature matrix: conversion logged success but the GenUICraft view was empty. Replaced with an entity-row phone comparison; the successful phone run was recorded again after fixing capture automation's UI-idle timeout.
- Two-day Mysuru itinerary: Bixby returned a much larger native places response. The converter rejected the context budget: estimated input 7,337 + output reserve 2,048 + template reserve 256 exceeds 8,192. It was not a successful model-generation demo.
- Energy comparison with explanation: appliance cost chips were useful, but a prose paragraph was rendered as a clipped `Formula` element. Excluded.
- Energy table-only comparison and apartment document table: rendered successfully as horizontally scrolling tables. They did not show enough additional presentation value to replace one of the five selected card scenarios.

## Visible gaps in selected cases

- Restaurant card cost, hours and address are easier to find. Its large gradient is a **decorative placeholder**, not a fetched restaurant photo. Signature dish survives in accessible row data but is not shown as a visible field; a long address is ellipsized. These need renderer refinement and are not presented as solved in this demo.
- The seasonal-weather output adds the title “Weather Forecast” to a typical-climate comparison. The month values were checked; the title should be made more appropriate by the generation/recovery pipeline.
- View switching caused several brief empty intervals. Those verified post-generation waits are removed in the edited videos, with the exact source intervals recorded in each metadata JSON. Native generation/progress is retained. No elapsed-time claim should be inferred from the edited playback.

## Video validation and editing

Five silent H.264 videos, 1080 × 1080 at 30 fps. Each raw recording and export passed strict FFmpeg decoding. Captions, arrows and transitions were reviewed against actual source/export frames. Only the longer flight Express stream is accelerated; its actual recorded spinner patch plays at normal speed. Other streams remain at normal speed. No timer, source-time text or speed label appears in the videos.

Each `.metadata.json` records source/export SHA-256, dimensions, frame count, crop, timing edits and annotations. Contact sheets and gallery posters contain actual exported frames. Original recordings and full device traces remain locally in ignored `evidence/`; concise runtime proof, selected videos, metadata and review artifacts are retained with the demo package.

Open [index.html](index.html) for all five players. Recreate exports with `render_demo.py --video evidence/<case>/raw.mp4 --plan <scenario>.plan.json --out <video>.mp4`, then run `build_gallery.py`.

## Queries

**Flights**

Show morning flights from Bengaluru to Varanasi for tomorrow. Compare 3 options in a compact table with airline, departure and arrival times, duration and fare.

**Trains**

Compare 4 trains from Delhi to Agra in a compact table with train number, departure, travel time and fare. Keep the introduction to one short sentence.

**Phone comparison**

Help me compare Samsung Galaxy S25, iPhone 16 and Google Pixel 9. Give one short introduction then a table with one row per phone and columns for India price, camera, battery and software updates. Keep it under 150 words.

**Seasonal weather**

Compare typical Bengaluru weather in January, April, July and October. Give one short introduction then a compact table with Month, High, Low, Rainfall and Conditions. Keep it under 130 words.

**Restaurant comparison**

Compare MTR, Vidyarthi Bhavan and Brahmins Coffee Bar for vegetarian breakfast in Bengaluru. Show the signature dish, approximate cost, address and opening hours for each. Keep it under 140 words.
