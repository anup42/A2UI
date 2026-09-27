# Trained E2B v10 W4 Bixby50 report

## Run coverage

| Measure | Count |
|---|---:|
| Selected | 2 |
| Completed | 2 |
| Returned output | 2 |
| Android strict valid | 2 |
| Python strict valid | 0 |
| Rendered valid | 2 |
| Runtime errors | 0 |
| Python-scored outputs | 2 |

Statuses: `not_selected`=48, `valid`=2.

The Android and Python strict counts come from separate implementations and are reported independently. A schema-valid artifact is not labelled a quality pass. No reference IR exists for this source-only holdout, and neither the scorer nor this report independently verifies the factual truth of the supplied response or generated UI.

## Performance

| Measure | Value |
|---|---:|
| Native decode tokens/s median | 26.90 |
| Native decode tokens/s min | 14.70 |
| Native decode tokens/s max | 39.11 |
| Native decode tokens/s, token-weighted overall | 17.91 |
| Input tokens total | 6,568 |
| Output tokens total | 822 |
| Outputs at or above 2048 tokens | 0 |
| Cold first end-to-end elapsed | 10,957 ms |
| Cold first provider call | 10,908 ms |
| Warm median end-to-end elapsed | 41,677 ms |
| Warm median provider call | 41,594 ms |

Native decode throughput excludes model initialization and prompt prefill. End-to-end elapsed time includes the converter/provider request path; the first case is the cold call. Warm medians include non-first cases that returned `output.express`.

## Source-fidelity metrics

These averages use scored outputs only. Each displayed mean excludes null, absent, and inapplicable atomics, and the table shows its observed/scored denominator. Runtime failures and cases without `output.express` are unavailable and do not contribute fabricated zero rewards. The official `aggregate_scores` fields below retain their training-compatible denominator unchanged.

| Metric | Observed-only average | Observed/scored | N/A excluded |
|---|---:|---:|---:|
| _No source-fidelity metric available_ | unavailable | 0/2 | 2 |

The v5.4 aggregate contains 2 scored outputs. `generation_reward_v5_4` average: 0; `render_artifact_quality_v5_4` average: 0.

## Configuration and provenance

- Run: `flip8_r64_qat_mtp_off_20260928_r1`
- Profile: `trained_e2b_v10_w4`
- Model: `gemma4_e2b_a2ui_mobile_r64_qat_compatible.litertlm` (2,588,147,712 bytes)
- Device: `samsung SM-F776U`, Android SDK `37`, hardware `qcom`
- Runtime: GPU=`True`, context `8192`, output cap `2048`, thinking `False`, MTP `False`, native metrics `True`
- Prompt contract SHA-256: `005ef700eee29a4429c895ac6c86580003c43d0ad35e67cbb92a9d4cf0e8c1d2`
- Prompt asset SHA-256 recorded/current match: `True`
- Prompt contract SHA-256 recorded/current match: `True`
- Android corpus SHA-256 recorded/current match: `True`
- Frozen evaluation corpus SHA-256/manifest match: `True`
- Verified case sources: `2/2` completed cases
- Native rendered-prompt SHA-256 parity: **2/2 reported hashes verified; 0 mismatches; 0 selected cases unavailable**
- Rendered-prompt expected-hash method: `SHA-256 of UTF-8 BOS/turn serialization from shared_prompt.json plus the frozen response; reconstructed prompt.json is the fallback`
- Missing result IDs: `BXP-002, BXP-004, BXP-005, BXP-006, BXP-007, BXP-008, BXP-009, BXP-010, BXP-011, BXP-012, BXP-013, BXP-014, BXP-015, BXP-016, BXP-017, BXP-018, BXP-019, BXP-020, BXP-021, BXP-022, BXP-023, BXP-024, BXP-025, BXP-026, BXP-027, BXP-028, BXP-029, BXP-030, BXP-031, BXP-032, BXP-033, BXP-034, BXP-035, BXP-036, BXP-037, BXP-038, BXP-039, BXP-040, BXP-041, BXP-042, BXP-043, BXP-044, BXP-045, BXP-046, BXP-047, BXP-048, BXP-049, BXP-050`
- Duplicate result IDs: `none`
- Unexpected result IDs: `none`
- `model_audit.json` is absent; this report makes no model-audit assumption.

See [per_case.csv](per_case.csv), [scored_predictions.jsonl](scored_predictions.jsonl), [aggregate_metrics.json](aggregate_metrics.json), and [gallery.html](gallery.html). Visual-quality conclusions remain a manual review step.
