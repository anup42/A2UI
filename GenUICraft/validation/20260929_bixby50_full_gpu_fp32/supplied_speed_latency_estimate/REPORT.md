# Bixby50: supplied comparison scores and decode-speed latency estimates

| Model / mode | Comparison score /100 | Decode speed tokens/s | Average generated tokens |
| --- | --- | --- | --- |
| Original FP32 checkpoint (supplied reference) | 85.90 | Not supplied | Not estimated here |
| LiteRT MTP off, after SDK repair | 83.26 | 35 | 615.42 |
| LiteRT MTP on, after SDK repair | 80.89 | 52 | 568.70 |

The FP32 checkpoint score 85.90 and decode speeds 35/52 tokens/s are supplied comparison values. The native repaired scores 83.26/80.89 match the completed full 50-case report. The checkpoint differences are 2.64 points without MTP and 5.01 points with MTP.

## Average tokens across all 50 cases per mode

The average model input is **3468.62 tokens** in either mode, including the model prompt. Total generated tokens are **30,771 off** and **28,435 on**. Dividing each by 50 gives **615.42 off** and **568.70 on**. Pooling all 100 attempts gives **592.06 generated tokens** per attempt. Repair adds no model-generated tokens.

## Estimated latency in seconds

| Component | MTP off | MTP on |
| --- | --- | --- |
| Generated tokens / supplied decode speed | 17.58 | 10.94 |
| Prefill (input tokens / recorded native prefill rate) | 6.60 | 6.16 |
| Other provider/dispatch and output recording | 0.03 | 0.03 |
| Converter + compile/repair + benchmark recording | 0.13 | 0.12 |
| Estimated warm conversion total | 24.35 | 17.25 |
| Additional cold engine initialization | 3.98 | 5.63 |
| Estimated cold conversion total | 28.33 | 22.88 |

Formula: `warm conversion = output tokens / supplied decode tokens per second + measured prefill + measured remaining conversion overhead`. Cold conversion adds the measured cold-start initialization mean. The endpoint is conversion completion or failure; Android render/display time is outside this estimate.

At the benchmark's actual batch restart frequency (11 cold calls off and 12 on, out of 50 each), the hybrid mean estimates would be **25.22s off** and **18.60s on**. That restart frequency is a test-harness property, not a recommended app behavior.

## Effect of repetition-stopped outputs

The full 50 includes 6 guard-stopped outputs without MTP and 7 with MTP. Some stopped outputs were repaired successfully, while 9 attempts across both modes had no recoverable UI content. Short stopped generations lower the token and latency averages. Excluding REPETITION_LIMIT attempts gives:

| Population | Count | Average output tokens | Decode estimate seconds | Warm conversion estimate seconds |
| --- | --- | --- | --- | --- |
| MTP off, excluding repetition stops | 44 | 675.61 | 19.30 | 26.12 |
| MTP on, excluding repetition stops | 43 | 638.40 | 12.28 | 18.54 |

The secondary population contains runs labelled COMPLETED by the app; it does not guarantee natural EOS, perfect content, or visual success.

## Measurement limits and timing definitions

These are estimates, not new measurements at 35/52 tokens/s. Prefill, initialization and residual overhead come from the existing two-device run with mixed temperatures and generation dates. Holding those costs fixed while replacing decode speed is an explicit assumption.

Do not add time-to-first-token to prefill: it already includes prefill and first-token work. Do not also add the native initialization phase total: its phase sums can overlap the wall initialization timer. The converter overhead includes timing/benchmark file writes, so it is not an isolated repair measurement. `caseElapsedMs` includes renderer setup, screenshot work and artificial test waits and is intentionally excluded.

No decode speed was supplied for the original FP32 checkpoint, so its latency is not inferred. V5.4 is a UI-representation score, not factual-answer accuracy.

[Summary JSON](summary.json) · [Per-case latency calculations](per_case_latency.csv) · [Full 50-case raw/repaired report](../full_native_repaired_report/REPORT.md)

Input manifest SHA-256: `ee02577f983e5665a9f7626ac03227dff2879116dc62eb0d545376971af862af`.
