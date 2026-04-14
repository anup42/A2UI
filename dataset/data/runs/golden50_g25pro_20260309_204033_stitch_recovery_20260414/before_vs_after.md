# Before vs After: Stitch-Guided Golden50 Recovery

- Generated: 2026-04-14 02:39 UTC
- Baseline: `golden50_g25pro_20260309_204033_stitch_refine_r2_20260413`
- Candidate: `golden50_g25pro_20260309_204033_stitch_recovery_20260414`

## Metric Delta

| metric | baseline | candidate | delta |
|---|---:|---:|---:|
| overall_score | 70.6185 | 75.9196 | +5.3011 |
| markdown_leakage_rate_avg | 0.1596 | 0.0000 | -0.1596 |
| table_cell_coverage_avg | 0.4897 | 0.4965 | +0.0069 |
| table_pattern_detected_rate | 0.5600 | 0.6000 | +0.0400 |
| information_chunking_score_avg | 0.6463 | 0.7090 | +0.0628 |
| image_presence_rate | 0.0000 | 0.0000 | +0.0000 |
| icon_presence_rate | 0.0000 | 0.0000 | +0.0000 |
| intent_expectation_pass_rate | 0.4400 | 0.5200 | +0.0800 |
| intent_score_avg | 0.7133 | 0.7633 | +0.0500 |

## Gap Delta (Count)

| gap | baseline | candidate | delta |
|---|---:|---:|---:|
| dense_text_layout | 7 | 2 | -5 |
| fallback_generated | 3 | 0 | -3 |
| markdown_leakage | 18 | 0 | -18 |
| missing_media_signal | 9 | 7 | -2 |
| weak_table_coverage | 1 | 0 | -1 |

## Category Gap Delta

| category | baseline_gap_count | candidate_gap_count | delta |
|---|---:|---:|---:|
| Booking | 1 | 0 | -1 |
| Comparison | 6 | 5 | -1 |
| Education | 2 | 0 | -2 |
| Event_schedule | 4 | 0 | -4 |
| General | 1 | 0 | -1 |
| Productivity | 1 | 0 | -1 |
| Recipe | 1 | 0 | -1 |
| Travel | 17 | 2 | -15 |
| Weather | 5 | 2 | -3 |

## Improved Examples

| ui_id | category | baseline_gaps | candidate_gaps | screenshot |
|---|---|---|---|---|
| `u_000001_01` | Travel | markdown_leakage | - | `android_device_rendered\01_u_000001_01.png` |
| `u_000007_01` | Booking | markdown_leakage | - | `android_device_rendered\07_u_000007_01.png` |
| `u_000008_01` | Weather | markdown_leakage | - | `android_device_rendered\08_u_000008_01.png` |
| `u_000009_01` | Weather | markdown_leakage | - | `android_device_rendered\09_u_000009_01.png` |
| `u_000013_01` | Travel | dense_text_layout, fallback_generated, markdown_leakage, missing_media_signal | - | `android_device_rendered\13_u_000013_01.png` |
| `u_000015_01` | Event_schedule | markdown_leakage | - | `android_device_rendered\15_u_000015_01.png` |
| `u_000016_01` | Event_schedule | markdown_leakage | - | `android_device_rendered\16_u_000016_01.png` |
| `u_000018_01` | Recipe | markdown_leakage | - | `android_device_rendered\18_u_000018_01.png` |
| `u_000019_01` | Travel | markdown_leakage | - | `android_device_rendered\19_u_000019_01.png` |
| `u_000020_01` | Travel | markdown_leakage | - | `android_device_rendered\20_u_000020_01.png` |
| `u_000021_01` | Travel | dense_text_layout, fallback_generated, markdown_leakage, missing_media_signal | - | `android_device_rendered\21_u_000021_01.png` |
| `u_000026_01` | Event_schedule | markdown_leakage | - | `android_device_rendered\26_u_000026_01.png` |
| `u_000030_01` | Comparison | markdown_leakage | - | `android_device_rendered\30_u_000030_01.png` |
| `u_000033_01` | Travel | markdown_leakage | - | `android_device_rendered\33_u_000033_01.png` |
| `u_000034_01` | Travel | dense_text_layout, fallback_generated, markdown_leakage, missing_media_signal | - | `android_device_rendered\34_u_000034_01.png` |
