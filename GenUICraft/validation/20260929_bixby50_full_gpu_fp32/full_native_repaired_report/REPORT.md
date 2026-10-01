# Bixby50 full native scores after GenUICraft 0.5.6 repair

Generated: 2026-09-29T17:44:56.362628+00:00

This report scores **100 native attempts in 50 pairs**: 27 verified outputs collected on September 28 and 73 newly generated outputs from September 29. Both modes cover the same 50 frozen Bixby50 cases, once per mode, with each case's MTP-off and MTP-on runs on the same device. Every captured raw output was sent through the same published GenUICraft 0.5.6 generated-output repair; any runtime no-output attempt has an explicit failure record. No source text or fallback was supplied to repair.

| Population | Cases | Raw mean /100 | After repair mean /100 | Raw strict | Repair accepted | Repaired strict |
| --- | --- | --- | --- | --- | --- | --- |
| LiteRT W4 / GPU FP32 / MTP off | 50 | 37.56 | 83.26 | 26 | 46 | 46 |
| LiteRT W4 / GPU FP32 / MTP on | 50 | 40.05 | 80.89 | 27 | 45 | 45 |
| Checkpoint LoRA R32 | 50 | 37.80 | 83.74 | 30 | 49 | 49 |
| Checkpoint LoRA R64 | 50 | 32.37 | 77.11 | 29 | 43 | 43 |

The full paired repaired mean difference (on minus off) is **-2.37 points**. These are v5.4 source-to-UI representation-quality scores, not factual-answer accuracy or visual certification. A failed repair contributes zero to the all-attempt mean; missing runtime output has null individual scores and also contributes zero to the aggregate.

## Hardware and generation origin

| Device | Paired cases | Off repaired mean | On repaired mean | On − off | Reused off / on |
| --- | --- | --- | --- | --- | --- |
| SM-F776U (R3GL203AKSF) | 31 | 89.23 | 87.59 | -1.64 | 15 / 12 |
| SM-F966B (R3CY30QFWLP) | 19 | 73.52 | 69.96 | -3.55 | 0 / 0 |

Generation origins: MTP off 15 prior verified and 35 new; MTP on 12 prior verified and 38 new. The archived 27 raw outputs were checked byte for byte against their earlier saved records. Each case's source response and query were checked against the frozen Bixby50 corpus.

## Native speed

| Mode | Completed measured cases | Native output tokens | Token-weighted native decode tokens/s |
| --- | --- | --- | --- |
| LiteRT W4 / GPU FP32 / MTP off | 44 | 29727.0 | 16.55 |
| LiteRT W4 / GPU FP32 / MTP on | 43 | 27451.0 | 24.68 |

| Device | Mode | Completed measured cases | Token-weighted native decode tokens/s |
| --- | --- | --- | --- |
| SM-F776U (R3GL203AKSF) | off | 30 | 13.75 |
| SM-F776U (R3GL203AKSF) | on | 30 | 21.45 |
| SM-F966B (R3CY30QFWLP) | off | 14 | 22.70 |
| SM-F966B (R3CY30QFWLP) | on | 13 | 32.51 |

Decode throughput uses only completed cases with native output-token counts and native decode-token rates. It is computed as total native output tokens divided by the sum of per-case token-count/rate times. It excludes engine initialization, prefill, screen capture, and host repair time. These rows mix archived September 28 and new September 29 generations and were not collected under controlled, equal thermal conditions. They are descriptive diagnostics, not a causal MTP speed comparison; different output lengths and device heat also affect the rate.

## Generation finish and failures

| Finish reason | MTP off | MTP on |
| --- | --- | --- |
| COMPLETED | 44 | 43 |
| REPETITION_LIMIT | 6 | 7 |

The native repetition guard stopped 6 off and 7 on generations after its fixed 20-reference limit. Of those, SDK repair accepted 2 off and 2 on; the remaining guard-stopped outputs were rejected. A repetition stop alone does not assign a zero score. The unchanged native output cap was 2,048 tokens; observed outputs at or above that cap were 1 off and 0 on.

The table below lists each failed SDK repair, runtime no-output attempt, or Android render/capture failure separately. `SCREEN_CAPTURE_FAILED` means Android reported a valid strict/renderer contract but did not confirm a screenshot or visible case. That affects screenshot evidence, not the host v5.4 Express score. `SDK_REPAIR_REJECTED` means the published SDK returned no generated-only candidate; its repaired aggregate contribution is zero. Full errors and capture flags remain in scored JSONL and the interactive case browser.

| Case | Mode | Device | Failure evidence | Finish | Raw chars | Native tokens | Android status | SDK repair | Score effect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BXP-002 | off | SM-F776U | SCREEN_CAPTURE_FAILED | COMPLETED | 607 | 208.0 | render_invalid | GENERATED_DSL_REPAIR | v5.4 Express score retained |
| BXP-028 | off | SM-F776U | SDK_REPAIR_REJECTED | REPETITION_LIMIT | 59 | 49.0 | strict_invalid | REJECTED | zero: no repaired artifact |
| BXP-040 | off | SM-F966B | SDK_REPAIR_REJECTED | REPETITION_LIMIT | 59 | 49.0 | strict_invalid | REJECTED | zero: no repaired artifact |
| BXP-042 | off | SM-F966B | SDK_REPAIR_REJECTED | REPETITION_LIMIT | 59 | 49.0 | strict_invalid | REJECTED | zero: no repaired artifact |
| BXP-046 | off | SM-F966B | SDK_REPAIR_REJECTED | REPETITION_LIMIT | 59 | 49.0 | strict_invalid | REJECTED | zero: no repaired artifact |
| BXP-002 | on | SM-F776U | SCREEN_CAPTURE_FAILED | COMPLETED | 607 | 208.0 | render_invalid | GENERATED_DSL_REPAIR | v5.4 Express score retained |
| BXP-022 | on | SM-F776U | SCREEN_CAPTURE_FAILED | COMPLETED | 1755 | 551.0 | render_invalid | GENERATED_DSL_REPAIR | v5.4 Express score retained |
| BXP-028 | on | SM-F776U | SDK_REPAIR_REJECTED | REPETITION_LIMIT | 59 | 49.0 | strict_invalid | REJECTED | zero: no repaired artifact |
| BXP-040 | on | SM-F966B | SDK_REPAIR_REJECTED | REPETITION_LIMIT | 59 | 49.0 | strict_invalid | REJECTED | zero: no repaired artifact |
| BXP-042 | on | SM-F966B | SDK_REPAIR_REJECTED | REPETITION_LIMIT | 59 | 49.0 | strict_invalid | REJECTED | zero: no repaired artifact |
| BXP-043 | on | SM-F966B | SDK_REPAIR_REJECTED | REPETITION_LIMIT | 59 | 49.0 | strict_invalid | REJECTED | zero: no repaired artifact |
| BXP-046 | on | SM-F966B | SDK_REPAIR_REJECTED | REPETITION_LIMIT | 59 | 49.0 | strict_invalid | REJECTED | zero: no repaired artifact |

## Repair and measurement policy

`GenUiCompiler.compileWithRepair(raw, null, false, true)` receives only the generated Express text. `NONE`, `STRUCTURAL`, and `GENERATED_DSL_REPAIR` are accepted routes. `REJECTED` and `RUNTIME_NO_OUTPUT` provide no repaired candidate. The raw output is scored unchanged; accepted repaired Express is scored unchanged with `score_prediction(source, None, text, metric_version='v5_4')` and cross-checked against `generation_reward_v5_4`. Output-free runtime failures are explicitly labeled and retained in the denominator.

The model, frozen prompt, 20-reference repetition guard, 2,048-token output cap, temperature zero, and no-source-fallback policy stayed fixed. This scoring pass performs no holdout tuning or response regeneration.

The checkpoint R32/R64 rows come from the previous complete 50-case published-AAR repair report. They are comparison populations, not new device generations. The 27 archived LiteRT raw outputs are included here: every selected output was repaired and rescored exactly once into its new 50-case mode mean. The earlier partial LiteRT aggregate means were never averaged into this report.

Expected model SHA-256: `de60d19c8e1ef06ed1032212d0d4ae5d9b0ec105db72db34a988453004e63e62`. Verified published AAR SHA-256: `c59eeae4debf4e8428a806ac7b7d7cb1b13f2d3559a12a655a56ef2f0be723b1`. Metric fingerprint: `3472befab7a179db4302b95a1ba3fa7cbd67858db86cc9f8ba0d96e3a4034a95`. Raw/repaired source-contract equality was verified for every accepted repair; a mismatch aborts scoring and creates `contract_mismatch.json` for review.

## Per-case scores

Each cell is raw → after repair. A repaired zero with no artifact is the stated aggregate policy; inspect the JSONL record for its null individual score and error.

| Case | Device | MTP off | MTP on | Off / on origin |
| --- | --- | --- | --- | --- |
| BXP-001 | SM-F776U | 91.15 → 91.15 | 90.03 → 90.03 | prior_verified / prior_verified |
| BXP-002 | SM-F776U | 40.00 → 97.28 | 40.00 → 97.28 | prior_verified / prior_verified |
| BXP-003 | SM-F776U | 99.01 → 99.01 | 0.00 → 99.01 | prior_verified / prior_verified |
| BXP-004 | SM-F776U | 70.00 → 90.64 | 40.00 → 90.64 | prior_verified / prior_verified |
| BXP-005 | SM-F776U | 40.00 → 70.00 | 40.00 → 96.72 | prior_verified / prior_verified |
| BXP-006 | SM-F776U | 0.00 → 89.98 | 89.26 → 89.26 | prior_verified / prior_verified |
| BXP-007 | SM-F776U | 70.00 → 98.08 | 70.00 → 98.08 | prior_verified / prior_verified |
| BXP-008 | SM-F776U | 0.00 → 95.25 | 0.00 → 96.52 | prior_verified / prior_verified |
| BXP-009 | SM-F776U | 40.00 → 98.80 | 0.00 → 91.63 | prior_verified / prior_verified |
| BXP-010 | SM-F776U | 95.73 → 95.73 | 95.73 → 95.73 | prior_verified / prior_verified |
| BXP-011 | SM-F776U | 90.61 → 90.61 | 90.61 → 90.61 | prior_verified / prior_verified |
| BXP-012 | SM-F776U | 70.00 → 97.68 | 0.00 → 94.86 | prior_verified / prior_verified |
| BXP-013 | SM-F776U | 0.00 → 97.91 | 0.00 → 97.91 | prior_verified / new_run |
| BXP-014 | SM-F776U | 91.62 → 91.62 | 91.38 → 91.38 | prior_verified / new_run |
| BXP-015 | SM-F776U | 98.51 → 98.51 | 98.51 → 98.51 | prior_verified / new_run |
| BXP-016 | SM-F776U | 96.71 → 96.71 | 96.71 → 96.71 | new_run / new_run |
| BXP-017 | SM-F776U | 96.51 → 96.51 | 96.51 → 96.51 | new_run / new_run |
| BXP-018 | SM-F776U | 92.53 → 92.53 | 92.40 → 92.40 | new_run / new_run |
| BXP-019 | SM-F776U | 40.00 → 98.02 | 70.00 → 98.02 | new_run / new_run |
| BXP-020 | SM-F776U | 0.00 → 95.99 | 0.00 → 95.99 | new_run / new_run |
| BXP-021 | SM-F776U | 0.00 → 93.77 | 40.00 → 92.36 | new_run / new_run |
| BXP-022 | SM-F776U | 0.00 → 70.00 | 40.00 → 40.00 | new_run / new_run |
| BXP-023 | SM-F776U | 96.36 → 96.36 | 40.00 → 70.00 | new_run / new_run |
| BXP-024 | SM-F776U | 0.00 → 70.00 | 0.00 → 70.00 | new_run / new_run |
| BXP-025 | SM-F776U | 94.41 → 96.07 | 40.00 → 96.07 | new_run / new_run |
| BXP-026 | SM-F776U | 40.00 → 93.40 | 95.01 → 95.01 | new_run / new_run |
| BXP-027 | SM-F776U | 40.00 → 78.00 | 40.00 → 70.00 | new_run / new_run |
| BXP-028 | SM-F776U | 0.00 → 0.00 | 0.00 → 0.00 | new_run / new_run |
| BXP-029 | SM-F776U | 0.00 → 94.24 | 95.76 → 95.76 | new_run / new_run |
| BXP-030 | SM-F776U | 0.00 → 94.41 | 0.00 → 95.21 | new_run / new_run |
| BXP-031 | SM-F776U | 40.00 → 97.77 | 0.00 → 93.00 | new_run / new_run |
| BXP-032 | SM-F966B | 0.00 → 68.46 | 0.00 → 95.94 | new_run / new_run |
| BXP-033 | SM-F966B | 0.00 → 70.00 | 0.00 → 95.07 | new_run / new_run |
| BXP-034 | SM-F966B | 40.00 → 93.34 | 0.00 → 92.74 | new_run / new_run |
| BXP-035 | SM-F966B | 0.00 → 80.13 | 0.00 → 81.09 | new_run / new_run |
| BXP-036 | SM-F966B | 0.00 → 96.92 | 40.00 → 96.92 | new_run / new_run |
| BXP-037 | SM-F966B | 70.00 → 70.00 | 86.03 → 90.31 | new_run / new_run |
| BXP-038 | SM-F966B | 97.51 → 97.51 | 97.51 → 97.51 | new_run / new_run |
| BXP-039 | SM-F966B | 0.00 → 93.74 | 0.00 → 40.00 | new_run / new_run |
| BXP-040 | SM-F966B | 0.00 → 0.00 | 0.00 → 0.00 | new_run / new_run |
| BXP-041 | SM-F966B | 97.40 → 97.40 | 97.40 → 97.40 | new_run / new_run |
| BXP-042 | SM-F966B | 0.00 → 0.00 | 0.00 → 0.00 | new_run / new_run |
| BXP-043 | SM-F966B | 0.00 → 91.50 | 0.00 → 0.00 | new_run / new_run |
| BXP-044 | SM-F966B | 0.00 → 95.67 | 94.51 → 97.62 | new_run / new_run |
| BXP-045 | SM-F966B | 0.00 → 95.12 | 0.00 → 68.93 | new_run / new_run |
| BXP-046 | SM-F966B | 0.00 → 0.00 | 0.00 → 0.00 | new_run / new_run |
| BXP-047 | SM-F966B | 40.00 → 89.35 | 94.98 → 94.98 | new_run / new_run |
| BXP-048 | SM-F966B | 0.00 → 92.04 | 0.00 → 92.12 | new_run / new_run |
| BXP-049 | SM-F966B | 0.00 → 95.69 | 0.00 → 96.28 | new_run / new_run |
| BXP-050 | SM-F966B | 0.00 → 70.00 | 0.00 → 92.42 | new_run / new_run |

## Artifacts

[Interactive case browser](index.html) · [Summary JSON](summary.json) · [Per-case CSV](per_case_scores.csv) · [Scored MTP-off JSONL](scored/litert_mtp_off.jsonl) · [Scored MTP-on JSONL](scored/litert_mtp_on.jsonl) · [Input hashes](scoring_input_hashes.json).

Repair success and a high v5.4 score do not establish complete source fidelity. MTP may change both output text and generation time; this single paired pass does not establish a statistically stable quality difference.
