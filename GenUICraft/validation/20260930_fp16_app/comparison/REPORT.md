# Gemma 4 corrected FP16 versus FP32 app comparison

Complete matched protocol: **no**. Measured case results: **6**. Warmups excluded.

The arms use the same frozen Bixby50 source cases and trained SDK prompt. FP16_CORRECTED labels a separately prepared model package; its model hash is recorded in `protocol.json`. This experiment measures GPU execution with the app's SDK conversion and render path.

| Arm | Cases | Raw strict | Repaired strict | SDK render smoke | Native decode tok/s (weighted) | Median provider ms | Median TTFT s |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16_CORRECTED_mtp_off | 3 | 1 | 3 | 3 | 12.87 (3/3) | 50494.00 | 4.41 |
| FP32_mtp_off | 3 | 3 | 3 | 2 | 12.55 (3/3) | 40981.00 | 6.40 |

Weighted native decode throughput is `sum(native output tokens) / sum(native output tokens / native decode tokens per second)` over cases with both native measurements. Missing measurements are excluded and counted in the table. A median of per-case native rates is shown below for each matched case.

| Case | MTP | FP32 / FP16 samples | FP16 wall speedup | FP16 decode speedup | Raw strict FP32 / FP16 | Repaired strict FP32 / FP16 | SDK render FP32 / FP16 |
|---|---|---:|---:|---:|---:|---:|---:|
| BXP-001 | off | 1 / 1 | 1.05× | 1.05× | 1 / 1 | 1 / 1 | 1 / 1 |
| BXP-003 | off | 1 / 1 | 1.04× | 1.03× | 1 / 0 | 1 / 1 | 1 / 1 |
| BXP-004 | off | 1 / 1 | 0.25× | 1.02× | 1 / 0 | 1 / 1 | 0 / 1 |
| BXP-001 | on | 0 / 0 | n/a× | n/a× | 0 / 0 | 0 / 0 | 0 / 0 |
| BXP-003 | on | 0 / 0 | n/a× | n/a× | 0 / 0 | 0 / 0 | 0 / 0 |
| BXP-004 | on | 0 / 0 | n/a× | n/a× | 0 / 0 | 0 / 0 | 0 / 0 |

## Evidence and limits

Each run directory preserves exact raw output, frozen source, generation prompt, repaired Express/JSON, per-case native token/timing metrics, finish reason, runtime identity, SDK render screenshot, and status. `telemetry.jsonl` records host-sampled thermal, battery, and GPU state; check it before attributing small timing differences to precision. First-case warmups are recorded separately and excluded.

Raw strict validity means the exact native output compiled unchanged. Repaired strict validity means the SDK returned a document that compiles. SDK render smoke means the app displayed that document without a visible error and captured a screenshot. These mechanical checks do not prove content fidelity. The per-case source fidelity warning count in `summary.json` provides another diagnostic, not a score.

The sample has only weather, train, and airline baggage cases from a development-informed corpus. It does not establish a 50-case success rate or production readiness.

Incomplete or mismatched runs: r02_fp16_corrected_mtp_off (partial; config=nativeWarmup,actualMtpState), r02_fp32_mtp_off (not_run), r01_fp32_mtp_on (not_run), r01_fp16_corrected_mtp_on (not_run), r02_fp16_corrected_mtp_on (not_run), r02_fp32_mtp_on (not_run)
