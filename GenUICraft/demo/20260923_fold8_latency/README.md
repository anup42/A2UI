# Fold8 GenUICraft demo slide

This folder contains the latest Fold8 capture pulled from serial `R3GL203AKSF`
and a one-slide PowerPoint demo artifact.

The slide uses a white background for clearer demo-room readability while keeping
the embedded device captures and editable latency text.

- `output/GenUICraft_Fold8_Demo_Latency.pptx` is the final slide. It embeds the
  17.75-second MP4, so the video travels with the presentation.
- `output/slide_preview.png` is a native PowerPoint render used for visual QA.
- `source/Screen_Recording_20260923_003626_GenUICraft.mp4` is the original device
  recording, SHA-256 `2dcccfd6ff808cebe7b8364104fa57b05913aaee74dfb40f9bfef8e62c2b1975`.
- `source/Screenshot_20260923_003645_GenUICraft.jpg` is the original device
  screenshot, SHA-256 `57f2e787db70bb98a087837da5a802bf0740dccbf4781a023c5edcc14e34db82`.

The slide records BXP-001 on trained E2B W4 with GPU + MTP: 9.53 seconds complete
conversion, 9.48 seconds provider wall time, 2.67 seconds engine initialization,
1.45 seconds prompt prefill, 5.34 seconds native decode, 9 milliseconds runtime
and callback overhead, 50 milliseconds validation and recovery, 53.91 token/s
native decode, and 55.86% MTP acceptance.

The presentation package passed structural validation and first-party Artifact Tool
import. A native PowerPoint export was also inspected after embedding the MP4.
