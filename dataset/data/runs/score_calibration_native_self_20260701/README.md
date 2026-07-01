# Native Score Calibration Set

This run is generated locally for structural-score calibration. It uses 10 self-authored response texts that follow the Stage 2 response prompt shape, then creates seven strict flat-spec IR variants per response following the Stage 3 contract rules.

No external LLM is used for the IR variants in this run. The purpose is controlled human calibration: compare native Android screenshots across structural metric score bands.

Target buckets: 40, 50, 60, 70, 80, 90, 100.

Important: `target_score` is the intended structural richness band; `actual_score` and `aggregates_by_bucket.json` are the authoritative computed scores from the current metric formula.

## Generated Artifacts
- `genui.jsonl`: 70 strict flat-spec IR records, 10 scenarios x 7 target buckets.
- `android_device_rendered/`: 70 screenshots captured from the connected Android app native renderer.
- `review_index.html`: browser-friendly visual review page grouped by target bucket.
- `rating_sheet.csv`: sheet for collecting human visual ratings against each screenshot.
- `aggregates_by_bucket.json`: computed aggregate score per bucket using the current dataset metric formula.

## Verified Bucket Aggregate Scores
| Target bucket | Computed aggregate score | Screenshots |
| --- | ---: | ---: |
| 40 | 42.01 | 10 |
| 50 | 50.29 | 10 |
| 60 | 60.11 | 10 |
| 70 | 71.10 | 10 |
| 80 | 81.41 | 10 |
| 90 | 92.33 | 10 |
| 100 | 100.00 | 10 |

