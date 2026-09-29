# Gemma 4 corrected FP16 versus FP32 app comparison

Complete matched protocol: **yes**. Measured case results: **6**. Warmups excluded.

The arms use the same frozen Bixby50 source cases and trained SDK prompt. FP16_CORRECTED labels a separately prepared model package; its model hash is recorded in `protocol.json`. This experiment measures GPU execution with the app's SDK conversion and render path.

| Arm | Cases | Raw strict | Repaired strict | SDK render smoke | Native decode tok/s (weighted) | Median provider ms | Median TTFT s |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16_CORRECTED_mtp_on | 3 | 3 | 3 | 3 | 16.41 (3/3) | 31660.00 | 6.77 |
| FP32_mtp_on | 3 | 2 | 3 | 3 | 10.84 (3/3) | 52868.00 | 10.50 |

Weighted native decode throughput is `sum(native output tokens) / sum(native output tokens / native decode tokens per second)` over cases with both native measurements. Missing measurements are excluded and counted in the table. A median of per-case native rates is shown below for each matched case.

| Case | MTP | FP32 / FP16 samples | FP16 wall speedup | FP16 decode speedup | Raw strict FP32 / FP16 | Repaired strict FP32 / FP16 | SDK render FP32 / FP16 |
|---|---|---:|---:|---:|---:|---:|---:|
| BXP-001 | on | 1 / 1 | 2.22× | 1.93× | 1 / 1 | 1 / 1 | 1 / 1 |
| BXP-003 | on | 1 / 1 | 1.49× | 1.37× | 0 / 1 | 1 / 1 | 1 / 1 |
| BXP-004 | on | 1 / 1 | 1.35× | 1.40× | 1 / 1 | 1 / 1 | 1 / 1 |

## Evidence and limits

Each run directory preserves exact raw output, frozen source, generation prompt, repaired Express/JSON, per-case native token/timing metrics, finish reason, runtime identity, SDK render screenshot, and status. `telemetry.jsonl` records host-sampled thermal, battery, and GPU state; check it before attributing small timing differences to precision. First-case warmups are recorded separately and excluded.

Raw strict validity means the exact native output compiled unchanged. Repaired strict validity means the SDK returned a document that compiles. SDK render smoke means the app displayed that document without a visible error and captured a screenshot. These mechanical checks do not prove content fidelity. The per-case source fidelity warning count in `summary.json` provides another diagnostic, not a score.

The sample has only weather, train, and airline baggage cases from a development-informed corpus. It does not establish a 50-case success rate or production readiness.
