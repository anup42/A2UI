# Bixby50: checkpoint and LiteRT-LM v5.4 score report

Generated: 2026-09-29T16:49:48.048040+00:00

**The two supplied checkpoints have complete 50-case scores. The latest saved corrected LiteRT-LM GPU FP32 run is partial: 15 cases with MTP off and 12 with MTP on. A full 50-case LiteRT score is not yet measured.**

V5.4 is **GenUI Representation Quality**, a source-grounded engineering score from 0 to 100. It measures how the generated UI represents the captured Bixby answer; it is **not factual-answer accuracy, exact-match accuracy, or a human visual-quality rating**. The scorer configuration marks it `uncalibrated_engineering_score`.

## 1. All available raw outputs

| Model / runtime | Completed / 50 | Raw v5.4 mean /100 | Median | Strict valid | Zero-score cases |
| --- | --- | --- | --- | --- | --- |
| Checkpoint LoRA R32 | 50/50 | 37.80 | 40.00 | 30/50 | 20 |
| Checkpoint LoRA R64 | 50/50 | 32.37 | 40.00 | 29/50 | 21 |
| LiteRT W4 / GPU FP32 / MTP off | 15/50 | 59.77 | 70.00 | 12/15 | 3 |
| LiteRT W4 / GPU FP32 / MTP on | 12/50 | 46.30 | 40.00 | 8/12 | 4 |

The 50-case checkpoint means and partial native means have different denominators. **Do not compare those means as an export accuracy gain/loss.** Every completed zero-scoring output stays in its denominator. Missing native cases remain N/A, not zero.

## 2. Fair comparison on the same completed cases

Paired cases: BXP-001, BXP-002, BXP-003, BXP-004, BXP-005, BXP-006, BXP-007, BXP-008, BXP-009, BXP-010, BXP-011, BXP-012.

| Model / runtime | Same cases | Raw v5.4 mean /100 | Strict valid | Zero-score cases |
| --- | --- | --- | --- | --- |
| Checkpoint LoRA R32 | 12 | 52.75 | 9/12 | 3 |
| Checkpoint LoRA R64 | 12 | 43.10 | 8/12 | 4 |
| LiteRT W4 / GPU FP32 / MTP off | 12 | 58.87 | 10/12 | 2 |
| LiteRT W4 / GPU FP32 / MTP on | 12 | 46.30 | 8/12 | 4 |

On these 12 cases, MTP-on minus MTP-off is **-12.57 score points**. MTP on scores higher in 1 cases, lower in 5, and ties in 6. This is one deterministic-generation run per mode, not repeated trials or a full-cohort result.

These numbers measure observed output quality, not checkpoint-to-native numerical parity. Different arithmetic, stopping/repetition guards, the speculative drafter, and export can all affect output. They do not isolate quantization as the cause. Bixby50 is a final-only holdout and should not be used to choose a checkpoint or tune hyperparameters.

## 3. Method and provenance

- Inputs: `r32_bixby50_scored.jsonl` and `r64_bixby50_scored.jsonl` in the supplied `e2b_runD_predictions` folder. Both contain 50 unique cases. Raw and serving-output fields are identical in every supplied row.
- Native inputs: completed `output.express` files from `20260928_fp32_bixby50_mtp/batches`. No new inference was performed for this report. Incomplete case directories are excluded and listed in `validation.json`.
- Native runtime: corrected R64 QAT-compatible LiteRT-LM, **W4 weights, GPU FP32 arithmetic**, temperature 0, 8,192 context tokens, 2,048 output-token limit, thinking disabled, MTP switched by batch. Device: Samsung SM-F776U / R3GL203AKSF / Android 17. FP32 here does not mean FP32 weights.
- Native model SHA-256: `de60d19c8e1ef06ed1032212d0d4ae5d9b0ec105db72db34a988453004e63e62` from the saved device provenance. The supplied checkpoint JSONL records identify CUDA inference and rank by filename but do not supply an exact checkpoint weight/step hash; this report cannot independently certify the export lineage.
- Every source text is byte-equivalent after UTF-8 text decoding to the frozen Bixby50 source in training and Android. All completed native rendered-prompt hashes match both supplied checkpoint prompt hashes for the same case.
- All raw outputs are scored unchanged by `score_prediction(..., expected=None, metric_version='v5_4')`. Results are cross-checked against the underlying `generation_reward_v5_4` breakdown. The same source-derived expected contract is used for every model. There is no reference IR and no invented reference/exact-match score.
- No repair, fallback, source reconstruction, Android render result, or repaired graph is passed into the headline raw score. Strict validity is the official Python Express syntax/catalog result; it is separate from v5.4 quality and Android screen success.
- Default dimension weights: integrity 25%, fidelity 40%, semantic mapping 15%, hierarchy 10%, economy 7%, accessibility 3%. Applicability reweighting and quality caps also apply, so the result is not a simple correctness percentage.

## 4. Archived-score reproducibility

| Population | Supplied/archived mean | Current mean | Changed cases | Maximum absolute delta |
| --- | --- | --- | --- | --- |
| Checkpoint LoRA R32 | 37.80 | 37.80 | 0 | 0 |
| Checkpoint LoRA R64 | 32.37 | 32.37 | 0 | 0 |
| LiteRT W4 / GPU FP32 / MTP off | 59.77 | 59.77 | 0 | 0 |
| LiteRT W4 / GPU FP32 / MTP on | 46.30 | 46.30 | 0 | 0 |

All populations were re-scored using one current identity. The archived fingerprint differs from the current one; the identity also hashes renderer sources and other contract artifacts. Numeric equivalence is checked case by case above, rather than assumed from the version number.

Current metric version: `5.4.0`  
Metric fingerprint: `3472befab7a179db4302b95a1ba3fa7cbd67858db86cc9f8ba0d96e3a4034a95`  
Reward pipeline fingerprint: `24e352841c215e3f88f0347fe836ffae3354e6505e96fcae4c1ce3cd0071375a`

## 5. Why scores are lost

### Checkpoint LoRA R32

Strict-invalid cases (20): BXP-006, BXP-008, BXP-012, BXP-019, BXP-024, BXP-026, BXP-028, BXP-029, BXP-030, BXP-031, BXP-032, BXP-035, BXP-036, BXP-040, BXP-044, BXP-045, BXP-047, BXP-048, BXP-049, BXP-050.

Zero-score cases (20): BXP-006, BXP-008, BXP-012, BXP-019, BXP-024, BXP-026, BXP-028, BXP-029, BXP-030, BXP-031, BXP-032, BXP-035, BXP-036, BXP-040, BXP-044, BXP-045, BXP-047, BXP-048, BXP-049, BXP-050.

Recorded termination reasons: closing_sentinel: 37; max_new_tokens: 13.

Active cap counts (a case can trigger several; the final binding cap is in its JSONL record):

| Cap | Cases |
| --- | --- |
| reachability_below_90 | 16 |
| partial_reachability | 3 |
| parse_failure | 20 |
| missing_special_role | 1 |

Lowest-scoring examples:

| Case | Raw v5.4 | Strict valid | First reported issue |
| --- | --- | --- | --- |
| BXP-006 | 0.00 | False | raw:ValueError:A2UI Express completion must contain exactly one <a2ui> block |
| BXP-008 | 0.00 | False | raw:ValueError:A2UI Express completion must contain exactly one <a2ui> block |
| BXP-012 | 0.00 | False | raw:ValueError:A2UI Express completion must contain exactly one <a2ui> block |
| BXP-019 | 0.00 | False | raw:ValueError:A2UI Express completion must contain exactly one <a2ui> block |
| BXP-024 | 0.00 | False | raw:ValueError:A2UI Express completion must contain exactly one <a2ui> block |

### Checkpoint LoRA R64

Strict-invalid cases (21): BXP-006, BXP-008, BXP-011, BXP-012, BXP-014, BXP-016, BXP-019, BXP-022, BXP-023, BXP-024, BXP-025, BXP-028, BXP-030, BXP-031, BXP-033, BXP-040, BXP-042, BXP-043, BXP-044, BXP-046, BXP-047.

Zero-score cases (21): BXP-006, BXP-008, BXP-011, BXP-012, BXP-014, BXP-016, BXP-019, BXP-022, BXP-023, BXP-024, BXP-025, BXP-028, BXP-030, BXP-031, BXP-033, BXP-040, BXP-042, BXP-043, BXP-044, BXP-046, BXP-047.

Recorded termination reasons: closing_sentinel: 43; max_new_tokens: 7.

Active cap counts (a case can trigger several; the final binding cap is in its JSONL record):

| Cap | Cases |
| --- | --- |
| reachability_below_90 | 19 |
| parse_failure | 21 |
| partial_reachability | 4 |
| missing_table | 1 |
| missing_special_role | 1 |

Lowest-scoring examples:

| Case | Raw v5.4 | Strict valid | First reported issue |
| --- | --- | --- | --- |
| BXP-006 | 0.00 | False | raw:ValueError:Element 'root' references missing 's' at children[6] |
| BXP-008 | 0.00 | False | raw:ValueError:A2UI Express completion must contain exactly one <a2ui> block |
| BXP-011 | 0.00 | False | raw:ValueError:Element 'e' references missing 'h' at children[2] |
| BXP-012 | 0.00 | False | raw:ValueError:A2UI Express completion must contain exactly one <a2ui> block |
| BXP-014 | 0.00 | False | raw:ValueError:Element 'e' references missing 'g' at children[1] |

### LiteRT W4 / GPU FP32 / MTP off

Strict-invalid cases (3): BXP-006, BXP-008, BXP-013.

Zero-score cases (3): BXP-006, BXP-008, BXP-013.

Recorded termination reasons: COMPLETED: 15.

Active cap counts (a case can trigger several; the final binding cap is in its JSONL record):

| Cap | Cases |
| --- | --- |
| reachability_below_90 | 3 |
| partial_reachability | 3 |
| parse_failure | 3 |

Lowest-scoring examples:

| Case | Raw v5.4 | Strict valid | First reported issue |
| --- | --- | --- | --- |
| BXP-006 | 0.00 | False | raw:ValueError:Element 'm' references missing 'q' at children[3] |
| BXP-008 | 0.00 | False | raw:ValueError:Element 'root' references missing 'd' at children[3] |
| BXP-013 | 0.00 | False | raw:ValueError:Element 'i' references missing 'k' at children[1] |
| BXP-002 | 40.00 | True | See source fidelity, graph reachability and cap details |
| BXP-005 | 40.00 | True | See source fidelity, graph reachability and cap details |

### LiteRT W4 / GPU FP32 / MTP on

Strict-invalid cases (4): BXP-003, BXP-008, BXP-009, BXP-012.

Zero-score cases (4): BXP-003, BXP-008, BXP-009, BXP-012.

Recorded termination reasons: COMPLETED: 12.

Active cap counts (a case can trigger several; the final binding cap is in its JSONL record):

| Cap | Cases |
| --- | --- |
| reachability_below_90 | 3 |
| parse_failure | 4 |
| partial_reachability | 1 |

Lowest-scoring examples:

| Case | Raw v5.4 | Strict valid | First reported issue |
| --- | --- | --- | --- |
| BXP-003 | 0.00 | False | raw:ValueError:Element 'j' references missing 'l' at children[1] |
| BXP-008 | 0.00 | False | raw:ValueError:Element 'd' references missing 'f' at children[1] |
| BXP-009 | 0.00 | False | raw:ValueError:Element 't' references missing 'v' at children[1] |
| BXP-012 | 0.00 | False | raw:ValueError:A2UI Express completion must contain exactly one <a2ui> block |
| BXP-002 | 40.00 | True | See source fidelity, graph reachability and cap details |

Do not interpret the generic exact-number channel alone as table accuracy: native table data has its own table-fidelity channel. Legacy lexical coverage and Android source-integrity warning counts are not used as headline accuracy. Dimension means in `summary.json` are observed-only and carry their own denominators; absent diagnostics on parse failures are not treated as perfect scores. See [QUALITY_FINDINGS.md](QUALITY_FINDINGS.md) for concrete failures and structural-loss counts.

## 6. SDK repair: separate application-output result

| SDK output population | Cases | Raw mean | After SDK repair mean | Strict valid after repair |
| --- | --- | --- | --- | --- |
| MTP-off output after SDK 0.5.6 repair | 15 | 59.77 | 93.48 | 15/15 |
| MTP-on output after SDK 0.5.6 repair | 12 | 46.30 | 94.20 | 12/12 |

This appendix re-scores the already captured SDK 0.5.6 repaired Express outputs from the same native generations. It performs no new generation and uses no source-text fallback. It is **post-processing quality**, not raw model accuracy. Successful compilation/rendering does not establish that all source content survived. Checkpoint outputs were not replayed through this SDK here, so repair scores must not be compared with unrepaired checkpoint scores as model accuracy.

## 7. Complete 50-case score matrix

All entries are raw v5.4 /100; N/A means no completed native output in this saved run.

| Case | R32 checkpoint | R64 checkpoint | LiteRT MTP off | LiteRT MTP on |
| --- | --- | --- | --- | --- |
| BXP-001 | 92.41 | 92.41 | 91.15 | 90.03 |
| BXP-002 | 93.30 | 40.00 | 40.00 | 40.00 |
| BXP-003 | 40.00 | 99.01 | 99.01 | 0.00 |
| BXP-004 | 70.00 | 40.00 | 70.00 | 40.00 |
| BXP-005 | 40.00 | 40.00 | 40.00 | 40.00 |
| BXP-006 | 0.00 | 0.00 | 0.00 | 89.26 |
| BXP-007 | 70.00 | 70.00 | 70.00 | 70.00 |
| BXP-008 | 0.00 | 0.00 | 0.00 | 0.00 |
| BXP-009 | 40.00 | 95.81 | 40.00 | 0.00 |
| BXP-010 | 95.73 | 40.00 | 95.73 | 95.73 |
| BXP-011 | 91.57 | 0.00 | 90.61 | 90.61 |
| BXP-012 | 0.00 | 0.00 | 70.00 | 0.00 |
| BXP-013 | 70.00 | 40.00 | 0.00 | N/A |
| BXP-014 | 91.62 | 0.00 | 91.62 | N/A |
| BXP-015 | 40.00 | 70.00 | 98.51 | N/A |
| BXP-016 | 95.50 | 0.00 | N/A | N/A |
| BXP-017 | 96.51 | 96.51 | N/A | N/A |
| BXP-018 | 92.54 | 40.00 | N/A | N/A |
| BXP-019 | 0.00 | 0.00 | N/A | N/A |
| BXP-020 | 40.00 | 40.00 | N/A | N/A |
| BXP-021 | 40.00 | 40.00 | N/A | N/A |
| BXP-022 | 40.00 | 0.00 | N/A | N/A |
| BXP-023 | 40.00 | 0.00 | N/A | N/A |
| BXP-024 | 0.00 | 0.00 | N/A | N/A |
| BXP-025 | 94.30 | 0.00 | N/A | N/A |
| BXP-026 | 0.00 | 40.00 | N/A | N/A |
| BXP-027 | 40.00 | 70.00 | N/A | N/A |
| BXP-028 | 0.00 | 0.00 | N/A | N/A |
| BXP-029 | 0.00 | 40.00 | N/A | N/A |
| BXP-030 | 0.00 | 0.00 | N/A | N/A |
| BXP-031 | 0.00 | 0.00 | N/A | N/A |
| BXP-032 | 0.00 | 40.00 | N/A | N/A |
| BXP-033 | 40.00 | 0.00 | N/A | N/A |
| BXP-034 | 40.00 | 40.00 | N/A | N/A |
| BXP-035 | 0.00 | 40.00 | N/A | N/A |
| BXP-036 | 0.00 | 40.00 | N/A | N/A |
| BXP-037 | 40.00 | 70.00 | N/A | N/A |
| BXP-038 | 99.17 | 97.51 | N/A | N/A |
| BXP-039 | 40.00 | 40.00 | N/A | N/A |
| BXP-040 | 0.00 | 0.00 | N/A | N/A |
| BXP-041 | 97.40 | 97.40 | N/A | N/A |
| BXP-042 | 40.00 | 0.00 | N/A | N/A |
| BXP-043 | 40.00 | 0.00 | N/A | N/A |
| BXP-044 | 0.00 | 0.00 | N/A | N/A |
| BXP-045 | 0.00 | 40.00 | N/A | N/A |
| BXP-046 | 40.00 | 0.00 | N/A | N/A |
| BXP-047 | 0.00 | 0.00 | N/A | N/A |
| BXP-048 | 0.00 | 40.00 | N/A | N/A |
| BXP-049 | 0.00 | 40.00 | N/A | N/A |
| BXP-050 | 0.00 | 40.00 | N/A | N/A |

## 8. Files and reproduction

- `index.html`: all 50 queries, source answers, raw outputs, errors, caps and dimension breakdowns side by side; SDK output in a separate appendix per case.
- `CHECKPOINT_REPORT.md`: full 50-case R32/R64 checkpoint results.
- `LITERT_MTP_REPORT.md`: partial native results and paired MTP comparison.
- `QUALITY_FINDINGS.md`: invalid-output categories, disconnected graphs and representative raw-output examples.
- `per_case_scores.csv`: all raw and separately labeled repaired records, hashes, runtime termination and score deltas.
- `bixby50_matrix.csv`: raw 50-case matrix; missing native values are empty.
- `scored/*.jsonl`: unmodified raw strings, source answers, metrics and complete official v5.4 breakdowns.
- `summary.json`, `validation.json`, `input_manifest.json`, `evidence/`: aggregates, coverage, input hashes, runtime/scorer provenance.

```powershell
python GenUICraft/tools/report_bixby50_v54_models.py `
  --checkpoint-dir "C:\Users\anupk\Downloads\trained_model_data\e2b_runD_predictions" `
  --output-dir "GenUICraft/validation/bixby50_v54_rescore_NEW"
```

Use a new output directory. Reproducing the same identity requires the same scorer and renderer contract sources. No checkpoint weight loading or native inference is performed by this report command.
