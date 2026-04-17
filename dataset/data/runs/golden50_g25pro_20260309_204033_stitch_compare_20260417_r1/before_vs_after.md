# Before vs After: Stitch-Guided Golden50 Recovery

- Generated: 2026-04-16 20:48 UTC
- Baseline: `golden50_g25pro_20260309_204033_android_promptsync_20260413_010634`
- Candidate: `golden50_g25pro_20260309_204033_stitch_compare_20260417_r1`

## Metric Delta

| metric | baseline | candidate | delta |
|---|---:|---:|---:|
| overall_score | 69.7749 | 80.6994 | +10.9245 |
| markdown_leakage_rate_avg | 0.0587 | 0.0000 | -0.0587 |
| table_cell_coverage_avg | 0.0312 | 0.4612 | +0.4301 |
| table_pattern_detected_rate | 0.0000 | 0.7600 | +0.7600 |
| information_chunking_score_avg | 0.5762 | 0.7146 | +0.1383 |
| image_presence_rate | 0.7400 | 0.7800 | +0.0400 |
| icon_presence_rate | 0.5600 | 0.6800 | +0.1200 |
| intent_expectation_pass_rate | 0.0200 | 0.4000 | +0.3800 |
| intent_score_avg | 0.3867 | 0.7133 | +0.3267 |

## Gap Delta (Count)

| gap | baseline | candidate | delta |
|---|---:|---:|---:|
| dense_text_layout | 12 | 0 | -12 |
| fallback_generated | 4 | 0 | -4 |
| markdown_leakage | 8 | 0 | -8 |
| missing_media_signal | 7 | 0 | -7 |
| weak_table_coverage | 13 | 0 | -13 |

## Category Gap Delta

| category | baseline_gap_count | candidate_gap_count | delta |
|---|---:|---:|---:|
| Booking | 9 | 0 | -9 |
| Comparison | 9 | 0 | -9 |
| Travel | 4 | 0 | -4 |
| Weather | 22 | 0 | -22 |

## Improved Examples

- No per-UI gap-count reduction detected in shared UI ids.
