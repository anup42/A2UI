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
  - Whitespace-insensitive character count of encoded `toon` string (size proxy).

- `output_tokens_json`
  - Whitespace-insensitive character count of serialized `genui_json` (size proxy).

- `output_chars_toon` / `output_chars_json`
  - Explicit whitespace-insensitive character-count aliases for TOON and JSON payloads.

### UI Structure / Richness

- `component_count`
  - Number of components in `updateComponents`.

- `component_count_norm` 
  - Normalized component count: `min(1, component_count / 60)`.
  - Range: `0..1` (diagnostic normalization).

- `component_count_capped`
  - Capped raw component count used for scoring: `min(component_count, 60)`.
  - Range: `0..60` (higher means richer UI up to cap).

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

### Intent Check Thresholds

When a check is required, pass/fail is computed with these thresholds:

- `table_ok`: `table_pattern_detected >= 1.0` OR `table_cell_coverage >= 0.4`
- `actions_ok`: `action_coverage >= 0.7`
- `sections_ok`: `section_heading_coverage >= 0.6`

### Per-Intent Baseline Requirements

Baseline means requirement from intent bucket only, before tag/content heuristics are applied.

| Intent (from `intents.info`) | Normalized bucket | Baseline required checks | Metrics used |
|---|---|---|---|
| Information Retrieval | `information_retrieval` | none | heuristic-triggered only |
| Entertainment | `entertainment` | none | heuristic-triggered only |
| product_lookup | `product_lookup` | actions | `action_coverage` |
| Booking | `booking` | actions | `action_coverage` |
| weather | `weather` | none | heuristic-triggered only |
| data visualisation | `data_visualization` | table | `table_pattern_detected`, `table_cell_coverage` |
| Planning | `planning` | sections | `section_heading_coverage` |
| productivity | `productivity` | none | heuristic-triggered only |
| Recipe | `recipe` | sections | `section_heading_coverage` |
| Localization | `localization` | none | heuristic-triggered only |
| Technical support | `technical_support` | sections | `section_heading_coverage` |
| Creating Writing | `creative_writing` | none | heuristic-triggered only |
| Event Schedule | `event_schedule` | actions + sections | `action_coverage`, `section_heading_coverage` |
| Research Analysis | `research_analysis` | none | heuristic-triggered only |
| Comparison | `comparison` | table | `table_pattern_detected`, `table_cell_coverage` |
| calculation | `calculation` | table | `table_pattern_detected`, `table_cell_coverage` |
| Travel | `travel` | actions + sections | `action_coverage`, `section_heading_coverage` |
| Navigation | `navigation` | actions | `action_coverage` |
| Education | `education` | sections | `section_heading_coverage` |
| Documentation | `documentation` | sections | `section_heading_coverage` |
| media playback | `media_playback` | none | heuristic-triggered only |
| qr scanner | `qr_scanner` | none | heuristic-triggered only |
| status check | `status_check` | actions | `action_coverage` |

### Heuristic Requirement Triggers (Apply To Any Intent)

These can add required checks even if baseline is `none`:

- Table required if response is table-like (markdown/piped table), or tags indicate table-heavy intents.
- Actions required if response contains URLs and has button-like cues (for example `Quick Actions` or `[Button ...]`), or tags indicate action-heavy intents.
- Sections required if response contains multiple heading-like blocks, or tags indicate section-heavy intents.

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
- `output_tokens_toon_avg`
- `output_tokens_json_avg`
- `output_chars_toon_avg`
- `output_chars_json_avg`
- `component_count_avg` 
- `component_count_capped_avg` 
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




## Metric Interpretation Bands

These bands are practical defaults for leaderboard interpretation, not hard pass/fail rules. For intent-specific checks, use the intent-aware metrics (`intent_*`).

### Friendly Metric Names (Suggested)

| Current field | Suggested display name |
|---|---|
| `content_coverage` | `semantic_coverage` |
| `dup_rate` | `duplication_rate` |
| `lint_score` | `schema_hygiene_score` |
| `output_tokens_toon` | `toon_chars_nowhitespace` |
| `output_tokens_json` | `json_chars_nowhitespace` |
| `component_count` | `ui_component_count_raw` |
| `component_count_norm` | `ui_component_count_norm` |
| `component_count_capped` | `ui_component_count_capped` |
| `unique_component_types` | `component_type_variety` |
| `max_tree_depth` | `layout_depth_max` |
| `avg_tree_depth` | `layout_depth_avg` |
| `container_to_text_ratio` | `container_text_balance` |
| `information_chunking_score` | `content_chunking_score` |
| `ui_modularity_score` | `layout_modularity_score` |
| `ui_decomposition_score` | `ui_decomposition_score` |
| `actionable_elements` | `openurl_button_count` |
| `action_coverage` | `action_extraction_coverage` |
| `url_as_text_rate` | `url_leakage_rate` |
| `table_pattern_detected` | `table_structure_detected` |
| `table_cell_coverage` | `table_content_coverage` |
| `section_heading_coverage` | `heading_structure_coverage` |
| `markdown_leakage_rate` | `markdown_leakage_rate` |
| `missing_ids_rate` | `missing_reference_rate` |
| `dangling_components_rate` | `unreachable_component_rate` |
| `intent_expectation_pass` | `intent_contract_pass` |
| `intent_score` | `intent_contract_score` |

### Per-Metric Ranges and Bands

| Metric | Theoretical range | Direction | Bad | OK | Good | Excellent |
|---|---|---|---|---|---|---|
| `content_coverage` | `0..1` | Higher | `<0.60` | `0.60-0.75` | `0.75-0.90` | `>=0.90` |
| `dup_rate` | `0..1` | Lower | `>0.35` | `0.20-0.35` | `0.10-0.20` | `<=0.10` |
| `lint_score` | `0..1` | Higher | `<0.70` | `0.70-0.85` | `0.85-0.95` | `>=0.95` |
| `output_tokens_toon` / `output_chars_toon` | `>=0` | Lower (efficiency) | `>8000` | `4000-8000` | `1500-4000` | `<1500` |
| `output_tokens_json` / `output_chars_json` | `>=0` | Lower (efficiency) | `>10000` | `5000-10000` | `2000-5000` | `<2000` |
| `component_count` | `>=0` | Diagnostic only | `<8` | `8-15` | `16-35` | `>35` |
| `component_count_norm` | `0..1` | Diagnostic only | `<0.30` | `0.30-0.50` | `0.50-0.75` | `>=0.75` |
| `component_count_capped` | `0..60` | Higher | `<12` | `12-24` | `24-35` | `>35` |
| `unique_component_types` | `>=0` | Higher | `<3` | `3-4` | `5-7` | `>=8` |
| `max_tree_depth` | `>=0` | Mid-range best | `<=1 or >=10` | `2-3 or 8-9` | `4 or 7` | `5-6` |
| `avg_tree_depth` | `>=0` | Mid-range best | `<1.0 or >4.5` | `1.0-1.4 or 3.6-4.5` | `1.5-1.9 or 3.1-3.5` | `2.0-3.0` |
| `container_to_text_ratio` | `>=0` | Mid/high best | `<0.20` | `0.20-0.50` | `0.50-1.20` | `1.20-2.00` |
| `information_chunking_score` | `0..1` | Higher | `<0.30` | `0.30-0.50` | `0.50-0.70` | `>=0.70` |
| `ui_modularity_score` | `0..1` | Higher | `<0.08` | `0.08-0.15` | `0.15-0.28` | `>=0.28` |
| `ui_decomposition_score` | `0..1` | Higher | `<0.35` | `0.35-0.50` | `0.50-0.65` | `>=0.65` |
| `actionable_elements` | `>=0` | Higher when actions expected | `0` | `1` | `2-3` | `>=4` |
| `action_coverage` | `0..1` | Higher | `<0.50` | `0.50-0.70` | `0.70-0.90` | `>=0.90` |
| `url_as_text_rate` | `0..1` | Lower | `>0.40` | `0.20-0.40` | `0.05-0.20` | `<=0.05` |
| `table_pattern_detected` | `0 or 1` | Higher (if table expected) | `0` | `-` | `-` | `1` |
| `table_cell_coverage` | `0..1` | Higher | `<0.30` | `0.30-0.50` | `0.50-0.75` | `>=0.75` |
| `section_heading_coverage` | `0..1` | Higher | `<0.30` | `0.30-0.60` | `0.60-0.85` | `>=0.85` |
| `markdown_leakage_rate` | `0..1` | Lower | `>0.35` | `0.20-0.35` | `0.08-0.20` | `<=0.08` |
| `missing_ids` | `>=0` | Lower | `>0` | `0` | `0` | `0` |
| `missing_ids_rate` | `0..1` | Lower | `>0` | `0` | `0` | `0` |
| `dangling_components` | `>=0` | Lower | `>0` | `0` | `0` | `0` |
| `dangling_components_rate` | `0..1` | Lower | `>0` | `0` | `0` | `0` |
| `intent_require_table/actions/sections` | `0 or 1` | Requirement flags | informational | informational | informational | informational |
| `intent_table_ok/actions_ok/sections_ok` | `0 or 1` | Higher (if required) | `0` | `-` | `-` | `1` |
| `intent_expectation_pass` | `0 or 1` | Higher | `0` | `-` | `-` | `1` |
| `intent_score` | `0..1` | Higher | `<0.40` | `0.40-0.65` | `0.65-0.85` | `>=0.85` |

Notes:
- Size metrics are whitespace-insensitive character counts, not tokenizer-based token counts.
- Count/depth bands are corpus-dependent; tune them after collecting a few runs.
- For table/action/section quality, rely on intent-aware checks for fair comparison.

## Overall Score Bounds

`overall_score` is a weighted linear sum from `configs/run.yaml -> evaluation.weights`.

Current default weights:
- Positive: `schema_valid_strict(5.0)`, `content_coverage(3.0)`, `lint_score(2.0)`, `component_count_capped(0.4)`, `ui_decomposition_score(1.0)`, `action_coverage(1.5)`, `table_pattern_detected(0.8)`, `table_cell_coverage(1.0)`, `section_heading_coverage(0.8)`, `intent_expectation_pass(1.2)`, `intent_score(0.8)`
- Negative penalties: `dup_rate(-1.0)`, `markdown_leakage_rate(-1.0)`

Because weighted metrics are bounded (`component_count_capped` is capped at `60`), score bounds are strict:
- `overall_score_min = -2.0` (max penalties, no rewards)
- `overall_score_max = 41.1` (all rewards maxed, no penalties)

### Overall Score Bands (raw scale)

| Overall score | Interpretation |
|---|---|
| `< 12.0` | Bad |
| `12.0 - 24.0` | OK |
| `24.0 - 35.0` | Good |
| `> 35.0` | Excellent |

Optional normalized score for dashboards:
- `overall_score_norm = (overall_score - (-2.0)) / (41.1 - (-2.0))`
- Range: `0..1` (clamp outside values).
