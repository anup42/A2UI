# GenUI metric v5.3 ten-row shadow audit

This immutable comparison scores the same ten Gemini 3.1 Flash Lite samples
as the v5.2 shadow sidecar. Both audits use source `genui.jsonl` SHA-256
`47aa3d9a25b6a7e9bffcbcac12f22cba2b747e2d597acfd2010934594990c6cc`.

| Surface | v5.2 mean | v5.3 mean | Delta |
|---|---:|---:|---:|
| Raw generation | 77.6863 | 77.0255 | -0.6608 |
| Final artifact | 79.5878 | 78.8296 | -0.7583 |

The v5.3 final median is 80.6741, SD is 14.6273, q05 is 55.7500, and
q95 is 92.5775. Six of ten final artifacts activate at least one cap.

These ten rows are a non-gating smoke comparison only. No weights were
fitted, no human calibration was performed, and no Android-native render was
attempted for this sidecar. The 0-100 values remain uncalibrated engineering
scores.
