# Bixby50 v5.4 scores after GenUICraft SDK repair

Generated: 2026-09-29T17:05:55.954809+00:00

Applied the same **published GenUICraft 0.5.6 AAR** to every available raw prediction: 50 R32 checkpoints, 50 R64 checkpoints, 15 LiteRT MTP-off outputs, and 12 LiteRT MTP-on outputs. **119/127** returned repaired/compiled artifacts. All **27/27** native results exactly match the earlier on-device repaired Express bytes.

These are **post-processing pipeline scores**, not raw model accuracy. V5.4 is an uncalibrated 0–100 source-to-UI representation-quality score; it does not measure factual-answer accuracy or human visual quality.

## All available outputs

| Population | Cases | Raw v5.4 | After SDK repair | SDK accepted | Python strict valid |
| --- | --- | --- | --- | --- | --- |
| Checkpoint LoRA R32 | 50/50 | 37.80 | 83.74 | 49/50 | 49/50 |
| Checkpoint LoRA R64 | 50/50 | 32.37 | 77.11 | 43/50 | 43/50 |
| LiteRT W4 / GPU FP32 / MTP off | 15/50 | 59.77 | 93.48 | 15/15 | 15/15 |
| LiteRT W4 / GPU FP32 / MTP on | 12/50 | 46.30 | 94.20 | 12/12 | 12/12 |

Repair failures remain in the denominator and contribute zero because the pipeline returned no usable artifact. Missing native cases are **unmeasured**, not zero. The 15-case and 12-case native means cannot be compared directly with full 50-case checkpoint means.

## Fair comparison on the same 12 cases

Cases: BXP-001 through BXP-012.

| Population | Same cases | Raw v5.4 | After SDK repair | Repair failures |
| --- | --- | --- | --- | --- |
| Checkpoint LoRA R32 | 12 | 52.75 | 88.94 | 0 |
| Checkpoint LoRA R64 | 12 | 43.10 | 80.83 | 1 |
| LiteRT W4 / GPU FP32 / MTP off | 12 | 58.87 | 92.85 | 0 |
| LiteRT W4 / GPU FP32 / MTP on | 12 | 46.30 | 94.20 | 0 |

On these same cases, after-repair MTP-on minus MTP-off is **+1.35 points**. The repair stage can change the ordering seen in raw scores. This is a single saved generation per case/mode, and 38 native paired cases remain unmeasured.

## Exact repair policy

```java
GenUiCompiler.compileWithRepair(rawOutput, null, false, true)
```

- The repair receives only generated output. `sourceText=null`, source fallback is disabled, and generated-DSL recovery is enabled.
- The source answer is supplied only to the official scorer, after repair. No model, external API, source reconstruction, hand-edited IR or model-specific fix is used.
- The library runs strict compilation first, then its existing syntax and generated-content/graph recovery. `NONE` can still return canonicalized Express; therefore string changes alone do not imply a content repair.
- All outputs, including previously valid outputs, go through the same call so disconnected generated graphs can be recovered consistently.
- Returned Express is recompiled with the SDK and scored unchanged using `score_prediction(source, None, repairedExpress, metric_version='v5_4')`. Its score is cross-checked against `generation_reward_v5_4`.
- For a failed repair there is no invented empty candidate: `repaired_score=null` in JSON, with `repaired_score_for_aggregate=0` under the explicit pipeline-failure policy.
- The original raw report and predictions are unchanged. Full generated and repaired text, errors, cap evidence and content-fidelity diagnostics are available in `scored/*.jsonl` and `index.html`.

## Repair outcomes

| Population | NONE | STRUCTURAL | GENERATED_DSL_REPAIR | REJECTED | Score improved / same / worse |
| --- | --- | --- | --- | --- | --- |
| Checkpoint LoRA R32 | 11 | 0 | 38 | 1 | 36 / 13 / 1 |
| Checkpoint LoRA R64 | 7 | 0 | 36 | 7 | 36 / 14 / 0 |
| LiteRT W4 / GPU FP32 / MTP off | 6 | 0 | 9 | 0 | 9 / 6 / 0 |
| LiteRT W4 / GPU FP32 / MTP on | 4 | 0 | 8 | 0 | 8 / 4 / 0 |

`NONE` means the SDK accepted the generated document without recovery; `STRUCTURAL` and `GENERATED_DSL_REPAIR` identify its repair routes. The scores are computed for the final returned Express, regardless of route.

## Failed repairs

| Population | Case | Raw score | After repair | Recorded generation stop | Raw characters |
| --- | --- | --- | --- | --- | --- |
| Checkpoint LoRA R32 | BXP-040 | 0.00 | 0 (no artifact) | max_new_tokens | 3998 |
| Checkpoint LoRA R64 | BXP-008 | 0.00 | 0 (no artifact) | closing_sentinel | 7 |
| Checkpoint LoRA R64 | BXP-028 | 0.00 | 0 (no artifact) | max_new_tokens | 3051 |
| Checkpoint LoRA R64 | BXP-033 | 0.00 | 0 (no artifact) | max_new_tokens | 3051 |
| Checkpoint LoRA R64 | BXP-040 | 0.00 | 0 (no artifact) | max_new_tokens | 3051 |
| Checkpoint LoRA R64 | BXP-042 | 0.00 | 0 (no artifact) | max_new_tokens | 3051 |
| Checkpoint LoRA R64 | BXP-043 | 0.00 | 0 (no artifact) | max_new_tokens | 3051 |
| Checkpoint LoRA R64 | BXP-046 | 0.00 | 0 (no artifact) | max_new_tokens | 3051 |

R64 BXP-008 contains only `</a2ui>`, with no generated UI content. The other seven failed inputs reached their recorded 2,048-token output limit and were incomplete. The current SDK returned no valid generated-only candidate for these cases. This does not prove that every fragment is unrecoverable; it records the behavior of this exact library version without inventing missing content. See [repair_failures.md](repair_failures.md) for every error and raw ending.

## Remaining limits after successful repair

Compiling a repaired UI does not restore facts that were never generated, prove complete source coverage, or certify a good screenshot. A high repaired score must therefore remain labeled as a source-grounded engineering score. This run did not perform new inference, device rendering or visual review. The JVM repair path was verified against all 27 earlier Android repairs by exact Express equality; checkpoint outputs have SDK compile validation and official Python scoring here, not fresh on-device screenshots.

| Population | Fully reachable after repair | Repair accepted | Successful-output-only mean (diagnostic) |
| --- | --- | --- | --- |
| Checkpoint LoRA R32 | 41/50 | 49/50 | 85.45 |
| Checkpoint LoRA R64 | 36/50 | 43/50 | 89.66 |
| LiteRT W4 / GPU FP32 / MTP off | 14/15 | 15/15 | 93.48 |
| LiteRT W4 / GPU FP32 / MTP on | 12/12 | 12/12 | 94.20 |

The successful-output-only mean excludes failures and is supplied only as a diagnostic. Use the all-attempt mean above for reporting pipeline performance.

## Provenance and validation

AAR SHA-256: `c59eeae4debf4e8428a806ac7b7d7cb1b13f2d3559a12a655a56ef2f0be723b1`.

LiteRT native generations are the saved corrected R64 QAT-compatible W4 / GPU FP32 runs, with MTP selected by batch. Native model SHA-256: `de60d19c8e1ef06ed1032212d0d4ae5d9b0ec105db72db34a988453004e63e62`. Repair on the host does not alter those generation/runtime labels.

Metric version: `5.4.0`.  
Metric fingerprint: `3472befab7a179db4302b95a1ba3fa7cbd67858db86cc9f8ba0d96e3a4034a95`.  
Reward pipeline fingerprint: `24e352841c215e3f88f0347fe836ffae3354e6505e96fcae4c1ce3cd0071375a`.

The raw and repaired scores use the same scorer identity and per-case source contract. Source hashes are checked against the prior report. Java runs only the published AAR and its Gson/Kotlin dependencies; the driver catches candidate validation errors while letting missing classes or runtime failures abort the run.

Files: [interactive report](index.html), [per-case CSV](per_case_scores.csv), [50-case matrix](bixby50_matrix.csv), [summary](summary.json), [repair outcomes](repair_outcomes.jsonl), [AAR/runtime provenance](repair_provenance.json), [device parity checks](device_repair_parity.json), and `scored/*.jsonl`.

## Per-case score matrix

Each cell is **raw → after repair**. N/A means the native output is unavailable. Failed repair has a reported pipeline score of 0.

| Case | R32 checkpoint | R64 checkpoint | LiteRT MTP off | LiteRT MTP on |
| --- | --- | --- | --- | --- |
| BXP-001 | 92.41 → 92.41 | 92.41 → 92.41 | 91.15 → 91.15 | 90.03 → 90.03 |
| BXP-002 | 93.30 → 93.30 | 40.00 → 95.46 | 40.00 → 97.28 | 40.00 → 97.28 |
| BXP-003 | 40.00 → 98.73 | 99.01 → 99.01 | 99.01 → 99.01 | 0.00 → 99.01 |
| BXP-004 | 70.00 → 70.00 | 40.00 → 93.38 | 70.00 → 90.64 | 40.00 → 90.64 |
| BXP-005 | 40.00 → 70.00 | 40.00 → 70.00 | 40.00 → 70.00 | 40.00 → 96.72 |
| BXP-006 | 0.00 → 68.77 | 0.00 → 40.00 | 0.00 → 89.98 | 89.26 → 89.26 |
| BXP-007 | 70.00 → 98.08 | 70.00 → 96.40 | 70.00 → 98.08 | 70.00 → 98.08 |
| BXP-008 | 0.00 → 95.90 | 0.00 → 0.00 | 0.00 → 95.25 | 0.00 → 96.52 |
| BXP-009 | 40.00 → 97.67 | 95.81 → 95.81 | 40.00 → 98.80 | 0.00 → 91.63 |
| BXP-010 | 95.73 → 95.73 | 40.00 → 95.47 | 95.73 → 95.73 | 95.73 → 95.73 |
| BXP-011 | 91.57 → 91.57 | 0.00 → 94.42 | 90.61 → 90.61 | 90.61 → 90.61 |
| BXP-012 | 0.00 → 95.10 | 0.00 → 97.60 | 70.00 → 97.68 | 0.00 → 94.86 |
| BXP-013 | 70.00 → 97.45 | 40.00 → 98.33 | 0.00 → 97.91 | N/A |
| BXP-014 | 91.62 → 91.62 | 0.00 → 91.62 | 91.62 → 91.62 | N/A |
| BXP-015 | 40.00 → 95.19 | 70.00 → 97.90 | 98.51 → 98.51 | N/A |
| BXP-016 | 95.50 → 95.50 | 0.00 → 96.71 | N/A | N/A |
| BXP-017 | 96.51 → 96.51 | 96.51 → 96.51 | N/A | N/A |
| BXP-018 | 92.54 → 92.54 | 40.00 → 89.36 | N/A | N/A |
| BXP-019 | 0.00 → 97.06 | 0.00 → 94.55 | N/A | N/A |
| BXP-020 | 40.00 → 95.99 | 40.00 → 96.35 | N/A | N/A |
| BXP-021 | 40.00 → 93.77 | 40.00 → 93.77 | N/A | N/A |
| BXP-022 | 40.00 → 70.00 | 0.00 → 40.00 | N/A | N/A |
| BXP-023 | 40.00 → 97.77 | 0.00 → 96.36 | N/A | N/A |
| BXP-024 | 0.00 → 77.64 | 0.00 → 92.92 | N/A | N/A |
| BXP-025 | 94.30 → 93.84 | 0.00 → 96.07 | N/A | N/A |
| BXP-026 | 0.00 → 94.73 | 40.00 → 94.73 | N/A | N/A |
| BXP-027 | 40.00 → 78.00 | 70.00 → 70.00 | N/A | N/A |
| BXP-028 | 0.00 → 87.05 | 0.00 → 0.00 | N/A | N/A |
| BXP-029 | 0.00 → 70.00 | 40.00 → 97.24 | N/A | N/A |
| BXP-030 | 0.00 → 69.32 | 0.00 → 95.24 | N/A | N/A |
| BXP-031 | 0.00 → 89.68 | 0.00 → 93.92 | N/A | N/A |
| BXP-032 | 0.00 → 57.35 | 40.00 → 95.39 | N/A | N/A |
| BXP-033 | 40.00 → 93.89 | 0.00 → 0.00 | N/A | N/A |
| BXP-034 | 40.00 → 70.00 | 40.00 → 70.00 | N/A | N/A |
| BXP-035 | 0.00 → 82.17 | 40.00 → 92.09 | N/A | N/A |
| BXP-036 | 0.00 → 93.83 | 40.00 → 96.92 | N/A | N/A |
| BXP-037 | 40.00 → 97.16 | 70.00 → 98.15 | N/A | N/A |
| BXP-038 | 99.17 → 99.17 | 97.51 → 97.51 | N/A | N/A |
| BXP-039 | 40.00 → 70.00 | 40.00 → 91.47 | N/A | N/A |
| BXP-040 | 0.00 → 0.00 | 0.00 → 0.00 | N/A | N/A |
| BXP-041 | 97.40 → 97.40 | 97.40 → 97.40 | N/A | N/A |
| BXP-042 | 40.00 → 70.00 | 0.00 → 0.00 | N/A | N/A |
| BXP-043 | 40.00 → 40.00 | 0.00 → 0.00 | N/A | N/A |
| BXP-044 | 0.00 → 68.55 | 0.00 → 97.08 | N/A | N/A |
| BXP-045 | 0.00 → 68.93 | 40.00 → 70.00 | N/A | N/A |
| BXP-046 | 40.00 → 97.22 | 0.00 → 0.00 | N/A | N/A |
| BXP-047 | 0.00 → 94.98 | 0.00 → 87.24 | N/A | N/A |
| BXP-048 | 0.00 → 94.27 | 40.00 → 70.00 | N/A | N/A |
| BXP-049 | 0.00 → 73.06 | 40.00 → 97.65 | N/A | N/A |
| BXP-050 | 0.00 → 68.33 | 40.00 → 93.17 | N/A | N/A |
