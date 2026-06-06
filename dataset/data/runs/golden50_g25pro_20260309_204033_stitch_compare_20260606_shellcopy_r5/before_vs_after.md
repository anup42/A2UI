# Before vs After: Stitch-Guided Golden50 Recovery

- Generated: 2026-04-29 17:51 UTC
- Baseline: `golden50_g25pro_20260309_204033_stitch_compare_20260417_r1`
- Candidate: `golden50_g25pro_20260309_204033_stitch_compare_20260429_hybrid_r4`

## Metric Delta

| metric | baseline | candidate | delta |
|---|---:|---:|---:|
| overall_score | 80.6994 | 79.4298 | -1.2696 |
| markdown_leakage_rate_avg | 0.0000 | 0.0000 | +0.0000 |
| table_cell_coverage_avg | 0.4612 | 0.4612 | -0.0000 |
| table_pattern_detected_rate | 0.7600 | 0.7600 | +0.0000 |
| information_chunking_score_avg | 0.7146 | 0.7085 | -0.0061 |
| image_presence_rate | 0.7800 | 0.7800 | +0.0000 |
| icon_presence_rate | 0.6800 | 0.7000 | +0.0200 |
| intent_expectation_pass_rate | 0.4000 | 0.3800 | -0.0200 |
| intent_score_avg | 0.7133 | 0.7067 | -0.0067 |

## Gap Delta (Count)

| gap | baseline | candidate | delta |
|---|---:|---:|---:|
| dense_text_layout | 3 | 0 | -3 |
| missing_media_signal | 2 | 0 | -2 |
| weak_table_coverage | 15 | 0 | -15 |

## Category Gap Delta

| category | baseline_gap_count | candidate_gap_count | delta |
|---|---:|---:|---:|
| Booking | 8 | 0 | -8 |
| Comparison | 3 | 0 | -3 |
| Travel | 2 | 0 | -2 |
| Weather | 7 | 0 | -7 |

## Improved Examples

- No per-UI gap-count reduction detected in shared UI ids.
