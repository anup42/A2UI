# Bixby50 MTP on/off comparison

**INCOMPLETE — no winner or full-50 claim.** 15 completed off cases and 12 completed on cases; **12 matched pairs**, 3 off-only, 0 on-only. The user stopped the run; the interrupted batch contributes only finished records. Same frozen source, prompt, model path, GPU FP32 settings, and device provenance were checked in each included batch. The paired five-case batches alternate in ABBA order across adjacent groups. The speed and score rows below use only the matched cases, since the all-completed mode sets have different case counts.

| Measure | MTP off | MTP on |
|---|---:|---:|
| Raw Android strict valid | 12/15 | 8/12 |
| Python strict valid, raw output | 12/15 | 8/12 |
| Raw Android strict valid, matched cases | 10/12 | 8/12 |
| Android strict after generated-DSL repair | 15/15 | 12/12 |
| Generated-DSL repairs | 3 | 4 |
| Renderer contract valid | 15/15 | 12/12 |
| Screen confirmed | 14/15 | 11/12 |
| Source fallback used | 0 | 0 |
| Output at/above 2048 token cap | 0 | 0 |
| Repetition finish reason/detail | 0 | 0 |
| Paired Warm median native decode tokens/s | 11.64 | 17.14 |
| Paired Warm median native prefill tokens/s | 440.43 | 571.19 |
| Paired Warm median time to first token, s | 7.59 | 5.85 |
| Paired Warm median provider wall, ms | 52,710.00 | 36,114.00 |
| Paired Cold median engine initialization, s | 2.61 | 3.17 |
| Paired Cold median native initialization phase, s | 5.27 | 6.39 |
| Paired Median v5.4 generation reward, scored outputs | 70.00 | 40.00 |
| Paired Median v5.4 render artifact quality, scored outputs | 70.00 | 40.00 |
| Paired Median legacy lexical content coverage, scored outputs | 1.00 | 1.00 |

The legacy lexical `content_coverage` scans the canonical graph, including unreachable content. It does not measure visible-content accuracy. The paired-set v5.4 fidelity atomics below are means of **observed numeric values only**; null, absent, and inapplicable values are excluded. Denominators show observed/scored outputs, and raw composite v5.4 reward keeps its scorer-defined zero for parse-failed raw outputs. These are source-comparison diagnostics, not certification of cited-claim truth.

| v5.4 fidelity atomic | MTP off observed mean (n/scored) | MTP on observed mean (n/scored) |
|---|---:|---:|
| `content_order_preservation` | 1.000 (10/12) | 1.000 (8/12) |
| `content_unit_fidelity` | 0.661 (10/12) | 0.654 (8/12) |
| `exact_numbers_dates_units_fbeta` | 0.660 (9/12) | 0.580 (8/12) |
| `heading_fidelity_and_order` | 0.920 (8/12) | 0.914 (7/12) |
| `markdown_table_fidelity` | 1.000 (5/12) | 0.968 (4/12) |
| `output_block_precision` | 0.554 (10/12) | 0.550 (8/12) |
| `unsupported_external_addition_precision` | 1.000 (10/12) | 1.000 (8/12) |
| `visible_content_multiset_fbeta` | 0.758 (10/12) | 0.754 (8/12) |

Warm paired native decode: **9 on wins, 0 off wins, 0 ties** across 9 comparable cases; median on/off ratio 1.38. Warm provider wall: 9 on shorter, 0 off shorter, 0 ties across 9 pairs. Paired generation reward: 1 on wins, 5 off wins, 6 ties across 12 scored pairs. These are descriptive paired observations, not a significance test.

Native `decodeTokensPerSecond` measures decode throughput; `providerCallMs` includes request work and depends on output length. The first case of **each** five-case batch is cold, giving 3 off and 3 on heads in the matched set. Cold initialization is reported separately. The run's time-to-first-token and prefill fields are native measurements where available. Session MTP drafter acceptance is recorded **per on batch** in `comparison.json`; its unweighted median across available sessions is 0.713. The interrupted session has no acceptance value. No accepted/drafted token counts were recorded, so a global acceptance ratio cannot be inferred.

| MTP-on batch | Session drafter acceptance |
|---|---:|
| `group_01_mtp_on` | 0.7004 |
| `group_02_mtp_on` | 0.7261 |
| `group_03_mtp_on` | n/a |

Finish reasons: off `{"COMPLETED": 15}`; on `{"COMPLETED": 12}`. Token-cap counts use observed output tokens; a finish detail is preserved per case in the CSV/JSON. A cap, repetition finish, or damaged output can affect quality and wall time independently of decode throughput.

Thermal status before cases ranges [0.0, 2.0] off and [2.0, 2.0] on; battery temperature ranges [35.1, 36.7] °C off and [35.1, 36.7] °C on. The **observed GPU clock cap during cases** ranges [160.0, 382.0] MHz off and [160.0, 342.0] MHz on, with host samples in 15/15 off and 12/12 on cases. Falling clock caps under sustained load are a plausible contributor to lower decode speed; the per-case samples in the CSV make this checkable. ABBA balances order across adjacent groups but cannot make paired device heat, clock limits, or timing identical. These observations do not isolate FP32 or MTP as the sole cause of a speed difference.

An exploratory subset has 5 warm pairs whose **observed median GPU clock caps** differ by at most 10%: 5 on wins, 0 off wins, 0 ties. Fifteen-second host samples can miss clock changes; this subset is not a normalized performance estimate.

Raw Android and Python strict validity use separate implementations. Renderer contract validity and screen confirmation are separate: a capture failure does not by itself establish invalid model output. Likewise, a rendering success or v5.4 score does not prove full source/content fidelity. SDK literal-only source warnings are diagnostic and may overcount omissions in reachable table `statePath` data; they are not a factual-accuracy numerator. An exact-number atomic can also include citation digits, and a source-link atomic does not certify which claim a citation supports. Scores exclude unavailable outputs rather than assigning zeros; review the full raw outputs and source in [side_by_side.html](side_by_side.html). MTP is a draft-and-verify decoding mechanism; different observed outputs alone do not show that it changed target weights or is inherently lower quality.

Evidence: [comparison.json](comparison.json), [per_case_comparison.csv](per_case_comparison.csv), [side_by_side.html](side_by_side.html), [off mode](mtp_off/REPORT.md), [on mode](mtp_on/REPORT.md), [SPEED_CONTEXT.md](../SPEED_CONTEXT.md), [QUALITY_REVIEW.md](../QUALITY_REVIEW.md). Missing batches: group_04_mtp_on, group_04_mtp_off, group_05_mtp_off, group_05_mtp_on, group_06_mtp_on, group_06_mtp_off, group_07_mtp_off, group_07_mtp_on, group_08_mtp_on, group_08_mtp_off, group_09_mtp_off, group_09_mtp_on, group_10_mtp_on, group_10_mtp_off. Interrupted batches: group_03_mtp_on.
