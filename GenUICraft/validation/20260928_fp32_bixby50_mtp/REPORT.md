# GPU FP32 MTP comparison: completed results

28 September 2026. **Stopped immediately at the user's request.** Completed **27 generations**: 15 with MTP off and 12 with MTP on. The fair comparison below uses only **BXP-001 through BXP-012**, the 12 matching pairs. Three additional MTP-off cases are listed separately. MTP-on BXP-013 was interrupted and is excluded. No further inference was run.

**MTP on is faster in this sample; MTP off has fewer raw IR failures.** Native token-weighted decode throughput was **10.75 versus 15.67 tokens/s**, a 45.8% increase. Total provider generation time across the matched cases fell **29.9%**. MTP was faster on 11 of 12 pairs. Raw strict validity was 10/12 off versus 8/12 on; content preservation remains mixed. Prefer off when raw IR reliability is the priority, and on when faster output plus recovery is the priority. Neither mode is fully source-faithful yet.

[Actual raw outputs and screenshots side by side](report/side_by_side.html) · [Detailed metrics](report/REPORT.md) · [Per-case CSV](report/per_case_comparison.csv) · [Content review](QUALITY_REVIEW.md) · [Speed explanation](SPEED_CONTEXT.md)

## Matched comparison: 12 cases per mode

| Measure | MTP off | MTP on |
|---|---:|---:|
| Raw Android strict-valid IR | 10/12 | 8/12 |
| Generated-DSL repair needed | 2/12 | 4/12 |
| Strict-valid A2UI after recovery | 12/12 | 12/12 |
| Renderer contract accepted | 12/12 | 12/12 |
| Original screenshots confirmed | 11/12 | 11/12 |
| Source-text fallback | 0 | 0 |
| Input tokens | 39,866 | 39,866 |
| Output tokens | 6,096 | 5,766 |
| Token-weighted native decode tokens/s | 10.75 | 15.67 |
| Warm median decode tokens/s (9 cases each) | 11.64 | 17.14 |
| Warm median provider time, seconds | 52.71 | 36.11 |
| Total provider time, seconds | 663.43 | 465.38 |

Native MTP drafter acceptance was **70.0391%** for cases 001–005 and **72.6133%** for 006–010. The interrupted 011–015 session did not reach normal teardown, so its acceptance is unavailable. These are batch ratios, not an exact pooled acceptance rate.

BXP-002's screenshots were not captured because the test's case-label accessibility check timed out. Both saved documents pass the renderer contract; this is **unconfirmed capture**, not proof of a blank render. Independently, both raw outputs genuinely leave pollutant/advice content unreachable. No additional rendering replay was run after the stop request.

## Why generation has become slower

The app now requests **GPU FP32 arithmetic** to mitigate the measured FP16 accuracy problem, while retaining the same W4 quantized weights. It still uses GPU inference. FP32 changes the computation and can cost throughput; this benchmark does not isolate that cost from device conditions.

The device also reported light/moderate thermal throttling. Its maximum GPU clock was 1.3 GHz before the run, then commonly **160–342 MHz** during sustained generation; it rose again late in the interrupted batch. Thermal state and GPU caps varied between cases. The results describe this device run, not cold peak performance or an isolated MTP/FP32 speed ratio.

Native token/s excludes initialization and prompt prefill. For example, MTP-on BXP-003 took **38.559 s**: about **32.728 s decode** (561 / 17.141), **5.807 s prefill**, and **0.024 s residual request overhead/counter difference**. It reused its engine. Time to first token is an overlapping measurement, not another duration to add. See [timing semantics and primary references](SPEED_CONTEXT.md).

## Every matched result

Entries below are **off / on**. “Repair” means raw compilation failed and the library used generated-output recovery; it is not a raw pass. Generation seconds exclude screenshot waits. Raw links preserve the actual pre-repair text.

| Case / domain | Raw status | Output tokens | Native decode tokens/s | Provider seconds | Actual raw output |
|---|---|---:|---:|---:|---|
| BXP-001 / Weather | Pass / Pass | 302 / 301 | 24.55 / 15.89 | 17.32 / 27.18 | [off](batches/group_01_mtp_off/BXP-001/output.express) / [on](batches/group_01_mtp_on/BXP-001/output.express) |
| BXP-002 / Air quality | Pass / Pass | 208 / 208 | 12.84 / 18.21 | 21.44 / 16.57 | [off](batches/group_01_mtp_off/BXP-002/output.express) / [on](batches/group_01_mtp_on/BXP-002/output.express) |
| BXP-003 / Rail travel | Pass / Repair | 571 / 561 | 12.71 / 17.14 | 51.00 / 38.56 | [off](batches/group_01_mtp_off/BXP-003/output.express) / [on](batches/group_01_mtp_on/BXP-003/output.express) |
| BXP-004 / Airline baggage | Pass / Pass | 433 / 430 | 12.69 / 19.47 | 40.15 / 27.56 | [off](batches/group_01_mtp_off/BXP-004/output.express) / [on](batches/group_01_mtp_on/BXP-004/output.express) |
| BXP-005 / Urban transport | Pass / Pass | 586 / 614 | 12.65 / 16.60 | 52.71 / 42.79 | [off](batches/group_01_mtp_off/BXP-005/output.express) / [on](batches/group_01_mtp_on/BXP-005/output.express) |
| BXP-006 / Road trips | Repair / Pass | 1054 / 753 | 10.50 / 17.39 | 110.46 / 52.79 | [off](batches/group_02_mtp_off/BXP-006/output.express) / [on](batches/group_02_mtp_on/BXP-006/output.express) |
| BXP-007 / Sightseeing | Pass / Pass | 442 / 441 | 7.89 / 17.45 | 64.50 / 31.04 | [off](batches/group_02_mtp_off/BXP-007/output.express) / [on](batches/group_02_mtp_on/BXP-007/output.express) |
| BXP-008 / Restaurants | Repair / Repair | 468 / 488 | 10.41 / 18.15 | 59.49 / 33.01 | [off](batches/group_02_mtp_off/BXP-008/output.express) / [on](batches/group_02_mtp_on/BXP-008/output.express) |
| BXP-009 / Accommodation | Pass / Repair | 508 / 485 | 11.64 / 16.01 | 51.18 / 36.11 | [off](batches/group_02_mtp_off/BXP-009/output.express) / [on](batches/group_02_mtp_on/BXP-009/output.express) |
| BXP-010 / Consumer audio | Pass / Pass | 451 / 449 | 9.38 / 10.21 | 56.68 / 52.77 | [off](batches/group_02_mtp_off/BXP-010/output.express) / [on](batches/group_02_mtp_on/BXP-010/output.express) |
| BXP-011 / Smartphones | Pass / Pass | 536 / 537 | 7.91 / 13.97 | 75.60 / 49.41 | [off](batches/group_03_mtp_off/BXP-011/output.express) / [on](batches/group_03_mtp_on/BXP-011/output.express) |
| BXP-012 / Productivity software | Pass / Repair | 537 / 499 | 10.22 / 13.23 | 62.91 / 57.58 | [off](batches/group_03_mtp_off/BXP-012/output.express) / [on](batches/group_03_mtp_on/BXP-012/output.express) |

## Extra completed MTP-off cases

These are not included in the paired comparison because their MTP-on counterparts did not complete. All three compiled after recovery and received confirmed screenshots; 013 needed generated-DSL repair.

| Case / domain | Raw status | Output tokens | Native decode tokens/s | Provider seconds | Output |
|---|---|---:|---:|---:|---|
| BXP-013 / Bank deposits | Repair | 1025 | 9.30 | 127.61 | [raw](batches/group_03_mtp_off/BXP-013/output.express) |
| BXP-014 / Foreign exchange | Pass | 343 | 9.12 | 47.57 | [raw](batches/group_03_mtp_off/BXP-014/output.express) |
| BXP-015 / Cricket | Pass | 301 | 10.84 | 35.25 | [raw](batches/group_03_mtp_off/BXP-015/output.express) |

Across all completed cases, off had **12/15 raw passes**, 3 repairs, 15/15 renderer-contract passes and 14/15 confirmed screenshots; on had **8/12 raw passes**, 4 repairs, 12/12 renderer-contract passes and 11/12 confirmed screenshots. These unequal totals must not replace the matched comparison above.

## Content quality still matters

- Weather: off retains `57%/55%/55%`; on renders bare `57/55/55`, losing percentage units. Both drop citations.
- AQI: both hide pollutant and outdoor advice in unreachable components despite raw strict compilation.
- Trains: on references an undefined component and needs repair; off raw IR passes. Both preserve the principal train numbers/times, but drop citations.
- Routes: on is better: valid, shorter output without the off mode's extra generated contract/hash text.
- Jaipur: off is raw-valid but hides an entire neighborhood section; on needs repair, which exposes more source content. Neither is fully faithful.
- Earbuds and smartphones: both preserve the material tabular facts and citations in the reviewed reachable graph.

The [12-pair content review](QUALITY_REVIEW.md) distinguishes raw graph, repaired graph, and inspected screenshots. SDK source-warning counts can over-report missing bound-table data and are not accuracy percentages. The independent v5.4 raw score handles reachable table bindings, but is still a composite heuristic rather than proof of factual truth, complete pixel visibility, or citation correctness.

## Reproduction and provenance

Connected device: **Samsung SM-F776U / Galaxy Z Flip8**, serial `R3GL203AKSF`, Android 17. Model: `gemma4_e2b_a2ui_mobile_r64_qat_compatible.litertlm`; original SHA-256 `de60d19c8e1ef06ed1032212d0d4ae5d9b0ec105db72db34a988453004e63e62`. The main APK, test APK, and model hashes match before and after the run; app inference is stopped.

GPU FP32, context 8,192, output limit 2,048, temperature 0, thinking off, generated-DSL repair enabled, source fallback disabled. Five-case groups alternate mode order; each batch starts a new engine. Matched cases have three cold heads and nine warm requests per mode. Prompts and input token counts match. This is one generation per completed case/mode, not a repeated statistical quality study or a full Bixby50 result.

Evidence: [paired statistics](PAIRED_STATS.json), [stop receipt](STOPPED.json), [original protocol](protocol.json), [before hashes](provenance.json), [after hashes](final_verification.json), [thermal/GPU timeline](telemetry.jsonl), and original files under [batches](batches/). Raw artifacts were not repaired or edited for this report.
