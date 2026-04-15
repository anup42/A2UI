# Before vs After: Stitch-Guided Golden50 Recovery

- Generated: 2026-04-14 22:53 UTC
- Baseline: `golden50_g25pro_20260309_204033_android_promptsync_20260413_010634`
- Candidate: `golden50_g25pro_20260309_204033_stitch_compare_20260415`

## Metric Delta

| metric | baseline | candidate | delta |
|---|---:|---:|---:|
| overall_score | 69.7749 | 80.8035 | +11.0286 |
| markdown_leakage_rate_avg | 0.0587 | 0.0000 | -0.0587 |
| table_cell_coverage_avg | 0.0312 | 0.4626 | +0.4315 |
| table_pattern_detected_rate | 0.0000 | 0.7600 | +0.7600 |
| information_chunking_score_avg | 0.5762 | 0.7189 | +0.1427 |
| image_presence_rate | 0.7400 | 0.7800 | +0.0400 |
| icon_presence_rate | 0.5600 | 0.6800 | +0.1200 |
| intent_expectation_pass_rate | 0.0200 | 0.4000 | +0.3800 |
| intent_score_avg | 0.3867 | 0.7133 | +0.3267 |

## Gap Delta (Count)

| gap | baseline | candidate | delta |
|---|---:|---:|---:|
| dense_text_layout | 12 | 3 | -9 |
| fallback_generated | 4 | 0 | -4 |
| markdown_leakage | 8 | 0 | -8 |
| missing_media_signal | 7 | 2 | -5 |
| weak_table_coverage | 13 | 15 | +2 |

## Category Gap Delta

| category | baseline_gap_count | candidate_gap_count | delta |
|---|---:|---:|---:|
| Booking | 9 | 8 | -1 |
| Comparison | 9 | 3 | -6 |
| Travel | 4 | 2 | -2 |
| Weather | 22 | 7 | -15 |

## Improved Examples

| ui_id | category | baseline_gaps | candidate_gaps | screenshot |
|---|---|---|---|---|
| `u_000002_01` | Comparison | markdown_leakage | - | `rendered\u_000002_01.png` |
| `u_000004_01` | Weather | dense_text_layout, weak_table_coverage | weak_table_coverage | `rendered\u_000004_01.png` |
| `u_000008_01` | Weather | dense_text_layout, fallback_generated, markdown_leakage, missing_media_signal | weak_table_coverage | `rendered\u_000008_01.png` |
| `u_000009_01` | Weather | dense_text_layout, fallback_generated, markdown_leakage, missing_media_signal | weak_table_coverage | `rendered\u_000009_01.png` |
| `u_000012_01` | Comparison | markdown_leakage, weak_table_coverage | weak_table_coverage | `rendered\u_000012_01.png` |
| `u_000015_01` | Weather | dense_text_layout, fallback_generated, markdown_leakage, missing_media_signal | - | `rendered\u_000015_01.png` |
| `u_000016_01` | Booking | missing_media_signal | - | `rendered\u_000016_01.png` |
| `u_000021_01` | Travel | markdown_leakage | - | `rendered\u_000021_01.png` |
| `u_000023_01` | Comparison | missing_media_signal | - | `rendered\u_000023_01.png` |
| `u_000024_01` | Comparison | dense_text_layout | - | `rendered\u_000024_01.png` |
| `u_000031_01` | Comparison | weak_table_coverage | - | `rendered\u_000031_01.png` |
| `u_000034_01` | Weather | dense_text_layout, fallback_generated, markdown_leakage, missing_media_signal | weak_table_coverage | `rendered\u_000034_01.png` |
| `u_000039_01` | Comparison | markdown_leakage, missing_media_signal | missing_media_signal | `rendered\u_000039_01.png` |
| `u_000045_01` | Travel | dense_text_layout | - | `rendered\u_000045_01.png` |
| `u_000050_01` | Weather | dense_text_layout | - | `rendered\u_000050_01.png` |
