# Trained E2B v10 W4 Bixby50 report

## Run coverage

| Measure | Count |
|---|---:|
| Selected | 12 |
| Completed | 12 |
| Returned output | 12 |
| Android strict valid | 12 |
| Python strict valid | 8 |
| Rendered valid | 11 |
| Runtime errors | 0 |
| Python-scored outputs | 12 |

Statuses: `not_selected`=38, `render_invalid`=1, `valid`=11.

The Android and Python strict counts come from separate implementations and are reported independently. A schema-valid artifact is not labelled a quality pass. No reference IR exists for this source-only holdout, and neither the scorer nor this report independently verifies the factual truth of the supplied response or generated UI.

## Performance

| Measure | Value |
|---|---:|
| Native decode tokens/s median | 16.87 |
| Native decode tokens/s min | 10.21 |
| Native decode tokens/s max | 19.47 |
| Native decode tokens/s, token-weighted overall | 15.67 |
| Input tokens total | 39,866 |
| Output tokens total | 5,766 |
| Outputs at or above 2048 tokens | 0 |
| Cold first end-to-end elapsed | 27,256 ms |
| Cold first provider call | 27,184 ms |
| Warm median end-to-end elapsed | 36,284 ms |
| Warm median provider call | 36,114 ms |

Native decode throughput excludes model initialization and prompt prefill. End-to-end elapsed time includes the converter/provider request path; the first case is the cold call. Warm medians include non-first cases that returned `output.express`.

## Source-fidelity metrics

These averages use scored outputs only. Each displayed mean excludes null, absent, and inapplicable atomics, and the table shows its observed/scored denominator. Runtime failures and cases without `output.express` are unavailable and do not contribute fabricated zero rewards. The official `aggregate_scores` fields below retain their training-compatible denominator unchanged.

| Metric | Observed-only average | Observed/scored | N/A excluded |
|---|---:|---:|---:|
| `content_coverage` | 0.9902 | 8/12 | 4 |
| `content_order_preservation` | 1 | 8/12 | 4 |
| `content_unit_fidelity` | 0.6539 | 8/12 | 4 |
| `exact_numbers_dates_units_fbeta` | 0.5804 | 8/12 | 4 |
| `heading_fidelity_and_order` | 0.9137 | 7/12 | 5 |
| `markdown_table_fidelity` | 0.9680 | 4/12 | 8 |
| `output_block_precision` | 0.5501 | 8/12 | 4 |
| `unsupported_external_addition_precision` | 1 | 8/12 | 4 |
| `visible_content_multiset_fbeta` | 0.7543 | 8/12 | 4 |

The v5.4 aggregate contains 12 scored outputs. `generation_reward_v5_4` average: 46.30; `render_artifact_quality_v5_4` average: 46.30.

## Configuration and provenance

- Run: `mtp_on`
- Profile: `trained_e2b_v10_w4`
- Model: `gemma4_e2b_a2ui_mobile_r64_qat_compatible.litertlm` (2,588,147,712 bytes)
- Device: `samsung SM-F776U`, Android SDK `37`, hardware `qcom`
- Runtime: GPU=`True`, context `8192`, output cap `2048`, thinking `False`, MTP `True`, native metrics `True`
- Prompt contract SHA-256: `005ef700eee29a4429c895ac6c86580003c43d0ad35e67cbb92a9d4cf0e8c1d2`
- Prompt asset SHA-256 recorded/current match: `True`
- Prompt contract SHA-256 recorded/current match: `True`
- Android corpus SHA-256 recorded/current match: `True`
- Frozen evaluation corpus SHA-256/manifest match: `True`
- Verified case sources: `12/12` completed cases
- Native rendered-prompt SHA-256 parity: **12/12 reported hashes verified; 0 mismatches; 0 selected cases unavailable**
- Rendered-prompt expected-hash method: `SHA-256 of UTF-8 BOS/turn serialization from shared_prompt.json plus the frozen response; reconstructed prompt.json is the fallback`
- Missing result IDs: `BXP-013, BXP-014, BXP-015, BXP-016, BXP-017, BXP-018, BXP-019, BXP-020, BXP-021, BXP-022, BXP-023, BXP-024, BXP-025, BXP-026, BXP-027, BXP-028, BXP-029, BXP-030, BXP-031, BXP-032, BXP-033, BXP-034, BXP-035, BXP-036, BXP-037, BXP-038, BXP-039, BXP-040, BXP-041, BXP-042, BXP-043, BXP-044, BXP-045, BXP-046, BXP-047, BXP-048, BXP-049, BXP-050`
- Duplicate result IDs: `none`
- Unexpected result IDs: `none`
- `model_audit.json` is absent; this report makes no model-audit assumption.

See [per_case.csv](per_case.csv), [scored_predictions.jsonl](scored_predictions.jsonl), [aggregate_metrics.json](aggregate_metrics.json), and [gallery.html](gallery.html). Visual-quality conclusions remain a manual review step.
