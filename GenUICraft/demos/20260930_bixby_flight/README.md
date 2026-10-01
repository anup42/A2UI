# Bixby + GenUICraft flight demo

`Bixby_GenUICraft_Flight_Demo.mp4` is a 41-second, silent, 1080 × 1080 H.264 video at 30 fps. It uses the light background, phone border, colored arrows and short callouts of the earlier CARO router demo.

## Actual capture

- Device: Samsung Flip8, SM-F776U, serial R3GL203AKSF.
- App: `com.samsung.android.bixby.agent`, version 5.0.10.38 (501038000).
- Installed APK last update: 2026-09-30 00:15:07. This installed APK predates the SDK 0.5.7 flight-range and spacing update.
- Query: “show morning flight from bengaluru to lucknow for tomorrow”.
- Both presentations belong to the same saved Bixby answer. The recording shows the actual selector change and scroll to the flight cards.
- Callouts highlight airline labels, aligned departure/arrival times, durations, and supplied fares. It is a presentation demonstration, not an inference-latency or accuracy benchmark.

## Edit

The device footage was cropped to remove system bars, resized into the left panel, and annotated outside the phone border. No answer values were rewritten or overlaid. Source interaction speed is preserved. Android's variable-frame-rate capture stops producing frames when the screen is static; the final unchanged frame is held to allow time to read the callouts.

The installed renderer displays a “Best value” badge on the first Air-India card, although that row has no supplied fare. The demo does not endorse that ranking. This is a known limitation visible in the captured build.

## Review

- Full video strict FFmpeg decode: passed, no errors.
- H.264 / 1080 × 1080 / 30 fps / 1,230 frames / 41 seconds: verified.
- Reviewed half-second contact sheets covering the entire video and full-size scene/transition frames.
- Checked caption readability, preserved phone border, selector transition, scroll, and arrow alignment.
- `review/` contains the contact sheets, key frames and metadata. `manifest.json` records artifact hashes.
- A copy is stored on the recording device at `/sdcard/Download/Bixby_GenUICraft_Flight_Demo.mp4`.

`flight_scene_plan.json` and `render_flight_demo.py` contain the editable annotation plan and rendering code. Raw recording and normalized edit source are retained under `.tmp/bixby_flight_video/` in this checkout.
