# Gemini 2.5 paired GenUI v4 score audit

Date: 2026-07-21

## Scope

- Fixed paired set: 10 query IDs.
- Models: `gemini-2.5-flash-lite` and `gemini-2.5-flash`.
- Each model generated its own Stage 2 response and Stage 3 FlatSpec.
- Stage 3 used the production flat-spec prompt and `metric_version: dual`.
- Express multiplexed Stage 3 batching failed, so final FlatSpecs were generated sequentially with batch size 1.

## Results

| UI ID | Intent | Flash-Lite v4 | Flash v4 | Flash - Lite |
|---|---|---:|---:|---:|
| u_self_001_01 | weather | 95.73 | 96.94 | +1.21 |
| u_self_002_01 | travel | 91.24 | 98.67 | +7.43 |
| u_self_003_01 | booking | 93.44 | 89.90 | -3.54 |
| u_self_004_01 | shopping | 70.00 | 93.39 | +23.39 |
| u_self_005_01 | entertainment | 78.22 | 82.25 | +4.03 |
| u_self_006_01 | travel | 82.79 | 77.44 | -5.35 |
| u_self_007_01 | calculation | 40.00 | 99.59 | +59.59 |
| u_self_008_01 | writing | 75.00 | 72.26 | -2.74 |
| u_self_009_01 | tech_support | 65.94 | 78.00 | +12.06 |
| u_self_010_01 | planning | 98.75 | 74.13 | -24.61 |

Flash-Lite: mean 79.11, median 80.50, population SD 16.83, range 40.00-98.75.

Flash: mean 86.26, median 86.08, population SD 10.07, range 72.26-99.59.

Flash won 6 of 10 paired query IDs; its mean advantage was 7.15 points. This is an end-to-end comparison because each model generated a different Stage 2 source response.

## Numerical and structural checks

- 20/20 strict schemas valid.
- 20/20 metric parses used the JSON path.
- All scores and rewards are finite and bounded.
- `quality_0_100 == 100 * quality_0_1` and `reward == 2 * quality_0_1 - 1` for every sample.
- No sample exceeds its active cap.
- Aggregate means and distributions exactly match the per-sample records.
- No missing-root, missing-reference, or cycle signal occurred.

## Interpretation

The cap behavior is directionally correct. For example, Flash-Lite `u_self_007_01` received 40.00 because only 11 of 17 elements were reachable and the required Formula role was absent. Flash generated a fully reachable Formula and Table for its corresponding source response and received 99.59.

The scores are valid representation-quality scores, but they should not yet be treated as calibrated end-to-end model quality scores:

1. Source response quality is explicitly out of scope. Flash `u_self_010_01` produced a malformed 40,600-character Markdown table separator with no data rows. Its FlatSpec still scored 74.13 because v4 evaluates response-to-IR representation, not whether the Stage 2 response was good.
2. Required `Icon` media directives are excluded from `media_fidelity`; all icon-only media cases received N/A for that atomic. Consequently, unresolved icon URLs did not lower v4 scores.
3. Native Android rendering was not attempted, so native renderer smoke is N/A for all 20 samples.
4. Economy was 1.0 for all 20 samples, and accessibility was either 1.0 or N/A. Those dimensions had no discriminative power in this small run.

## Verdict

The arithmetic, aggregation, caps, structural validation, and most rank directions are working properly. The results are suitable for engineering diagnostics and paired regression checks. They are not sufficient for calibrated model ranking until source-quality evaluation, icon/media handling, native-render evidence, and human calibration are added.
