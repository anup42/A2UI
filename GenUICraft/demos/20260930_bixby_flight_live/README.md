# Bixby + GenUICraft: live Varanasi flight run

## Current revision: faster streaming, normal spinner

The current preview uses `Bixby_GenUICraft_Varanasi_Demo.mp4` (48.8 seconds). Streaming text is accelerated while the circular progress indicator is composited from the actual recording at its original speed. The video has no on-screen speed labels. The query and rendered cards stay at normal speed. The clock retains source recording time.

The spinner is a 24 × 24 pixel recorded patch at (82, 242), replayed at 1× from source second 17 until output second 24.844444, when the completed UI removes it. Nine sampled spinner frames match the original-speed source more closely than accelerated playback: mean pixel error 2.55 versus 11.85. Evidence is in `review_final/spinner_comparison.json`. Strict decode and the 1,464-frame / 48.8-second check pass. The JSON sidecar records the edits; `render_stream_only_demo.py` reproduces them.

The earlier labelled revision, `Bixby_GenUICraft_Varanasi_Streaming3x.mp4`, is also retained. It plays original recording seconds 17.0–41.3 at 3×, occupying playback seconds 17.0–25.1. All other sections stay at normal speed.

The faster revision passed full strict decode and the 1,464-frame / 48.8-second check. Scene frames and half-second contact sheets are in `review_3x/`. Its JSON sidecar records source intervals and SHA-256; `render_fast_demo.py` reproduces the edit.

## Original live capture

This demo records a fresh request submitted to Bixby on the connected Flip8:

> show morning flights from bengaluru to varanasi for tomorrow

`Bixby_GenUICraft_Varanasi_Live.mp4` shows one continuous 65-second section from the start of the actual recording. The original pace is retained throughout query submission, Bixby's text answer, selection of GenUICraft, growing A2UI Express output, conversion to cards, and scrolling through the flight options.

`Bixby_Varanasi_Full_Raw_Recording.mp4` is the original 133.53-second Android screenrecord file, copied without transcoding. It also includes later idle time and comparison interactions after the 65-second demo ends.

## What is visible

- About 2 seconds: submit the prepared flight query.
- About 5–8 seconds: Bixby searches for the answer.
- About 9–16 seconds: the original Bixby flight answer is visible.
- About 17 seconds: select GenUICraft while its circular progress indicator is active.
- About 17–40 seconds: A2UI Express text grows on screen, including flight data and layout instructions.
- By 41.3 seconds: generated flight cards replace the stream.
- About 52 seconds: scroll through the generated flight cards.

The elapsed counter measures time in the recording, not model-only inference time.

## Device and runtime evidence

- Capture date: 30 September 2026, local time approximately 01:54–01:57.
- Samsung Flip8 SM-F776U, serial R3GL203AKSF.
- Actual Bixby package: `com.samsung.android.bixby.agent`, version 5.0.10.38 (501038000).
- Installed APK last update: 2026-09-30 01:42:12. This is a newer installation than the preceding Lucknow recording.
- Runtime log at 01:55:00.998: Gemma4, GPU, FP32, MTP=true, MTPRequested=true, modelSupportsMtp=true.
- Bixby renderer log at 01:55:28.990: conversion succeeded, attempts=1, repair=GENERATED_DSL_REPAIR, warnings=14.
- Runtime metrics were disabled in this run; no token-speed claim is made.

Relevant log excerpts are saved in `runtime_evidence.txt`; capture interactions and screenshot timestamps are in `events.json`.

## Editing and review

The annotated video uses the first 65 seconds of the raw recording. Only trailing footage is removed: there are no interior cuts, speed changes, inserted screenshots, simulated streaming, or added frame holds. Variable-frame-rate device footage is resampled to 30 fps while preserving elapsed time. The phone view is cropped and resized; arrows and captions are placed outside its border. App content is not rewritten.

The MP4 is silent H.264, 1080 × 1080, 30 fps. Full strict decode passed. Half-second contact sheets and full-size scene/transition frames were checked for readable text, arrow placement, visible streaming, a continuous phone border, and the transition to rendered cards. Review images are evidence only and are not inputs to the video renderer.

The displayed flight schedules and fares are Bixby's answer, not independently verified booking availability. The original answer says fares may change.

Editable annotations: `live_plan.json`. Renderer: `render_live_demo.py`. Source capture and the 65-second intermediate are retained in `.tmp/bixby_flight_live/` in this checkout. File hashes and frame/duration checks are in `manifest.json`.
