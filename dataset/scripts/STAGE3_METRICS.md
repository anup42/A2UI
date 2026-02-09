# Stage 3 Metrics Reference

This document explains all Stage 3 metric fields produced in `genui.jsonl` and aggregated into `aggregates.json`.

Code source:
- `src/pipeline/stage3_genui.py`
- `src/pipeline/metrics.py`

## Where Metrics Live

Per row (one UI candidate):
- `genui.jsonl` -> `metrics`
- `genui.jsonl` -> `validation`
- `genui.jsonl` -> `gen`

Per run summary:
- `aggregates.json`

## Per-Row Core Metrics

These are written in `stage3_genui.py` and computed in `metrics.py`.

### Content / Quality

- `content_coverage`
  - Fraction of unique non-stopword tokens from `response_text` that appear in serialized GenUI JSON.
  - Range: `0..1` (higher is better).

- `dup_rate`
  - Duplicate-fragment ratio computed from serialized GenUI JSON chunks.
  - Lower is better.

- `lint_score`
  - Heuristic score that penalizes empty children/lists and empty text/label fields.
  - Range: `0..1` (higher is better).

- `output_tokens_toon`
  - Token count of encoded `toon` string (`count_tokens`).

- `output_tokens_json`
  - Token count of serialized `genui_json` (`count_tokens`).

### UI Structure / Richness

- `component_count`
  - Number of components in `updateComponents`.

- `unique_component_types`
  - Number of distinct component types used.

- `max_tree_depth`
  - Max depth of component graph inferred from `child` and `children`.

- `avg_tree_depth`
  - Mean depth across component nodes.

- `container_to_text_ratio`
  - `container_count / max(1, text_count)`.
  - Container types include `Column`, `Row`, `List`, `Card`, `Tabs`, `Tab`, `Modal`, `Grid`, `Stack`, `Section`, `Container`.

- `information_chunking_score`
  - `1 - (max_text_block_len / total_text_len)`.
  - High value means content is distributed across multiple text nodes.

- `ui_modularity_score`
  - Fraction of components that are `Card`, `List`, or `Row`.

- `ui_decomposition_score`
  - Weighted blend:
  - `0.35 * container_norm + 0.25 * ui_modularity_score + 0.2 * information_chunking_score + 0.2 * variety_norm`
  - `container_norm = min(1, container_to_text_ratio / 1.5)`
  - `variety_norm = min(1, unique_component_types / 8.0)`

### Actions / URLs / Table Realization

- `actionable_elements`
  - Count of `Button` components with `action.functionCall.call == "openUrl"`.

- `action_coverage`
  - `unique_openUrl_buttons / expected_action_urls`.
  - Expected action URLs are URLs found on lines in response text that contain the word `button`.
  - Defaults to `1.0` when no expected action URLs are detected.

- `url_as_text_rate`
  - URL leakage into text:
  - `urls_found_in_text_nodes / urls_found_in_response_text`.
  - Lower is better.

- `table_pattern_detected`
  - `1.0` if table-like IR pattern is detected:
  - vertical `List` containing `Row` children with consistent cell counts.
  - Else `0.0`.

- `table_cell_coverage`
  - Coverage of table cells extracted from response markdown-like table text vs cells represented in detected IR table rows.
  - Range: `0..1`.

- `section_heading_coverage`
  - Match rate between expected headings inferred from response text and heading-like text nodes in IR (`variant` in heading variants).
  - Range: `0..1`.

- `markdown_leakage_rate`
  - Fraction of text-node lines that still match markdown-like patterns (`-`, `#`, `|`, etc.).
  - Lower is better.

### Reference Integrity

- `missing_ids`
  - Number of child references that point to non-existent component IDs.

- `missing_ids_rate`
  - `missing_ids / total_child_references`.

- `dangling_components`
  - Number of components unreachable from inferred roots.

- `dangling_components_rate`
  - `dangling_components / total_components`.

## Per-Row Intent-Aware Metrics

Intent and tags are normalized into buckets. Expected requirements are inferred from:
- intent bucket
- normalized tags
- response content heuristics (URLs, headings, table-like text)

Fields:
- `intent_require_table` (`0/1`)
- `intent_require_actions` (`0/1`)
- `intent_require_sections` (`0/1`)
- `intent_table_ok` (`0/1`, when required)
- `intent_actions_ok` (`0/1`, when required)
- `intent_sections_ok` (`0/1`, when required)
- `intent_expectation_pass` (`0/1`, all required checks passed)
- `intent_score` (mean pass rate over required checks; `1.0` if no checks required)

## Validation Fields (Per Row)

These are not in `metrics`, but are important Stage 3 quality signals:
- `validation.json_parse_ok`
- `validation.schema_valid_strict`
- `validation.schema_valid_lenient`
- `validation.toon_roundtrip_ok`
- `validation.errors`
- `validation.repair_attempts`
- `validation.repair_needed`

## Generation Metadata (Per Row)

From `gen`:
- `provider`
- `model`
- `prompt_version`
- `latency_ms`
- `input_tokens`
- `output_tokens`
- `cost_usd`
- `error`

## Aggregates (`aggregates.json`)

`aggregate_metrics(rows)` computes:
- `counts`
- `schema_valid_strict_rate`
- `content_coverage_avg`
- `lint_score_avg`
- `dup_rate_avg`
- `component_count_avg`
- `unique_component_types_avg`
- `max_tree_depth_avg`
- `avg_tree_depth_avg`
- `container_to_text_ratio_avg`
- `information_chunking_score_avg`
- `ui_modularity_score_avg`
- `ui_decomposition_score_avg`
- `actionable_elements_avg`
- `action_coverage_avg`
- `url_as_text_rate_avg`
- `table_pattern_detected_rate`
- `table_cell_coverage_avg`
- `section_heading_coverage_avg`
- `markdown_leakage_rate_avg`
- `missing_ids_avg`
- `missing_ids_rate_avg`
- `dangling_components_avg`
- `dangling_components_rate_avg`
- `intent_expectation_pass_rate`
- `intent_score_avg`
- `intent_stats` (per intent bucket breakdown)
- `render_ok_rate` (if render info exists in rows)
- `latency_ms_avg`
- `latency_ms_p95`

## Overall Score

`overall_score` is computed with `compute_overall_score(aggregate, weights)`:
- For each weight key `k` in `configs/run.yaml` -> `evaluation.weights`:
  - Tries `aggregate[k + "_rate"]`, then `aggregate[k + "_avg"]`, then `aggregate[k]`.
  - Adds `value * weight`.

This means negative weights are penalties and positive weights are rewards.

## Recompute Commands

Recompute one run:

```powershell
python scripts/recompute_aggregates_stream.py --run-id <RUN_ID>
```

Recompute/rank all runs:

```powershell
python scripts/report_model_rankings.py --recompute --format table
```
