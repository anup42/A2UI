# Stage 3 JSON IR Prompt Quality, Optimization, and KV-Cache Controls (Dataset + Android Flat-Spec Pipeline)

## 1) What this document covers

This note maps prompt quality and generation robustness for Stage 3 JSON IR (flat-spec) to exact code paths in this repository, then states best-practice checks for reliable JSON output.

Relevant files and canonical paths:
- Stage 3 orchestrator: `dataset/src/pipeline/stage3_genui.py`
- Stage 3 metrics + overall score: `dataset/src/pipeline/metrics.py`
- Prompt cache: `dataset/src/pipeline/cache.py`
- Local/llm adapter path: `dataset/src/llm/local_adapter.py`
- Main CLI wiring: `dataset/src/main.py`
- Default Stage 3 prompt: `dataset/prompts/genui_gen_mobile_flatspec_v11.md`
- Android runtime prompt copy: `android/app/src/main/assets/pipeline_prompts/genui_gen.md`
- Stage 3 config: `dataset/configs/run.yaml`

---

## 2) How JSON IR quality is currently measured

### 2.1 Per-sample metrics used for scoring

Stage 3 records store metrics in `record["metrics"]` via `run_stage3()` and `compute_ui_metrics(...)`.

- Ingestion and scoring call chain:
  - `run_stage3()` builds `record` and computes sample metrics before writing: `dataset/src/pipeline/stage3_genui.py` (`_compute_sample_overall_score()` around lines near `965`).
  - `compute_overall_score(...)` combines aggregate feature rates/averages: `dataset/src/pipeline/metrics.py` (`compute_overall_score` around lines near `1153`).
  - Run-level aggregation uses `aggregate_metrics(...)` then score composition: `dataset/src/pipeline/metrics.py` (`aggregate_metrics` around `1190`).

- Important quality components currently used:
  - Schema strictness (`schema_valid_strict`)
  - Coverage (`content_coverage`)
  - Structural/cleanliness metrics (`lint_score`, `dup_rate`)
  - Structural richness (`component_count`, `component_count_capped`, `unique_component_types`)
  - Layout/readability (`information_chunking_score`, `ui_modularity_score`, `ui_decomposition_score`, `container_to_text_ratio`)
  - Content affordances (`action_coverage`, `url_as_text_rate`, `table_pattern_detected`, `table_cell_coverage`, `section_heading_coverage`)
  - Safety/quality penalties (`markdown_leakage_rate`, `dangling_components_rate`, `missing_ids_rate`)
  - Intent compliance (`intent_expectation_pass`, `intent_score` + intent-specific table/action/section signals)

These are aggregated in `aggregate_metrics(...)` and then passed through weighted scoring in `compute_overall_score(...)`.

### 2.2 Stage 3 quality warnings (non-blocking diagnostic)

`run_stage3()` runs `_stage3_quality_warnings(...)` (`dataset/src/pipeline/stage3_genui.py`, near lines around `188`) for diagnostics like:
- `low_component_count`
- `sparse_ir`
- `low_heading_preservation`
- `table_cell_loss_risk`
These are stored as validation warnings and **do not currently change score weights**.

### 2.3 Aggregate score visibility

Each run writes `aggregates.json` in `_write_aggregates()` (`stage3_genui.py` around `935`).
This includes:
- `overall_score`
- `media_score` (separate post-aggregation metric)
- `counts` and other aggregated rates
- It is built from all generated records and includes render-log merge when available.

---

## 3) How prompt quality is optimized today

### 3.1 Prompt loading and context shaping

- Stage 3 prompt is loaded via `load_prompt(prompt_path)` and passed through:
  - `_maybe_compact_prompt_template(...)` (line-range ~659 in `stage3_genui.py`) which applies compact variant for Gemma-adapter family to reduce prompt token footprint.
  - `_prepare_prompt_context(...)` (~713) which can switch into system-prefix mode when a provider expects structured system/user split.

- Effective output path can be a compact, system+user split with fixed conversion prompt text and strict constraints.

### 3.2 Prompt hard rules in the canonical prompt file

`dataset/prompts/genui_gen_mobile_flatspec_v11.md` contains the production guidance used for most runs:
- Flat-spec only output shape: `{"root":"...","state":{...},"elements":{...}}`
- No prose mode
- Strict JSON-only instruction
- Markdown cleanup rules (no markdown control tokens in text)
- Table compression rule: one compact `Table` with `columns` + `statePath/rows`
- Domain metadata (`domain`, `preferredPresentation`, optional `primaryColumn`, etc.)
- Media policy, including `Image` vs `Icon` handling and detached media section suppression
- Recipe/instrumentation for formulas, tables, emails, console logs, etc.

### 3.3 Runtime prompt budget controls

Token control is done in `run_stage3()`:
- `prompt_token_multiplier` from env:
  - `STAGE3_PROMPT_TOKEN_MULTIPLIER` (fallback `A2UI_STAGE3_PROMPT_TOKEN_MULTIPLIER`)
- `_resolve_context_limited_prompt_max()` computes model-safe prompt caps (`VLLM_MAX_MODEL_LEN`, `LOCAL_VLLM_MAX_MODEL_LEN`, `A2UI_VLLM_MAX_MODEL_LEN` for local only) and safety margin via `STAGE3_CONTEXT_SAFETY_TOKENS`.
- `_clip_prompt_to_effective_budget(...)` trims oversized prompts and emits warning logs.

This is how you prevent input truncation cascades and 400-context errors from prompt growth.

### 3.4 Adaptive truncation inside Stage 3 prompt build

In `_build_prompt_for(...)` (`stage3_genui.py` around `988+`), prompt text is assembled with:
- response content
- asset policy section
- optional asset mapping block
and clipped if needed:
- drop optional asset_context first
- then drop trailing table context
- then replace response with `{..._truncated...}` fallback text.

### 3.5 Retry + repair strategy for JSON validity

Stage 3 pipeline has multi-step repair:
- primary generation
- parse -> schema validation
- `build_flat_spec_repair_prompt(...)` flow with repair temperature
- `STAGE3_FINAL_REGEN_ATTEMPTS` fallback path with final regeneration
- final structured fallback to minimal flat-spec on repeated failure

This is the key JSON-resilience loop preventing partial/non-parse outputs from leaking into final runs.

---

## 4) Where prompt cache is enabled and how it is used

### 4.1 Repository-level prompt cache

`dataset/src/pipeline/cache.py` implements a file-backed JSONL cache with `PromptCache.get/set`.

Cache is disabled by default unless:
- `A2UI_ENABLE_PROMPT_CACHE`
- or `DATASET_ENABLE_PROMPT_CACHE`

is `1|true|yes|on`.

### 4.2 Stage 1 / Stage 2 / Stage 3 cache integration

- Stage 1: `dataset/src/pipeline/stage1_queries.py`
  - cache key = `hash_text(f"{adapter.spec.name}:{prompt}:seed={seed_value}")`
- Stage 2: `dataset/src/pipeline/stage2_responses.py`
  - cache key = `hash_text(f"{adapter.spec.name}:{prompt}")`
- Stage 3: `dataset/src/pipeline/stage3_genui.py`
  - cache key = `hash_text(f"{adapter.spec.name}:{system_prompt or ''}\n---\n{prompt}")`

Cache short-circuits expensive LLM calls for repeated prompts and improves inter-run throughput. Aggregates still update after each generation/repair.

### 4.3 “KV cache” term clarification

The project has two distinct caching notions:
1) Prompt cache (on/off via env + `.jsonl` lookup) in pipeline code (`PromptCache`).
2) Model decode key-value cache (KV cache) behavior in model runtime.
   - For local HuggingFace direct generation, KV cache is explicitly controlled via `LOCAL_MODEL_USE_CACHE` in `dataset/src/llm/local_adapter.py`.
   - For reasoning-family local models, code currently sets `use_cache=False` by default to reduce memory pressure (large prompts), unless explicitly overridden.

So “KV cache” in model runtime is not always on by default; it is environment controlled for local transformers path.

---

## 5) Best practices for JSON generation in this codebase

### 5.1 Prompt engineering and instruction hygiene
- Keep prompt compact and deterministic:
  - strict output contract
  - no prose and no markdown literal tokens
  - one flat-spec object only
- Separate output policy from media policy (mapping local assets separately from response text)
- Prefer compact table-first representation for comparative content
- Avoid repeated semantics duplication (hero + table + summary copy of same rows)

### 5.2 Model/runtime configuration
- Use `json_mode=True` where supported for cleaner JSON compliance.
- Keep temperature moderate for structure tasks; avoid extreme values unless necessary.
- Use conservative `top_p`/`top_k` values (prompted by model-specific defaults and tested stability).
- Cap `max_tokens` appropriately for domain and prompt size; monitor truncation logs.

### 5.3 Validation + repair loop
- Keep fallback/repair logic enabled rather than trusting first pass.
- Always validate with schema before accepting output.
- Preserve failed sample diagnostics to allow targeted prompt revision later.

### 5.4 Schema-first and table-first metrics guard
- Require `Table` output where tabular data exists rather than duplicated Card/Text trees.
- Avoid random image/icon URLs; if unverifiable, prefer no media than wrong media.
- Preserve all numeric/date/units; avoid lossy conversions in compact tables.

---

## 6) Suggested metrics for better JSON prompt quality governance

Use this dashboard set in addition to the current score:

1. `schema_valid_strict_rate`
2. `markdown_leakage_rate_avg`
3. `table_cell_coverage_avg`
4. `section_heading_coverage_avg`
5. `url_as_text_rate_avg`
6. `intent_score_avg`
7. `content_coverage_avg`
8. `component_count_capped_avg`
9. `missing_ids_rate_avg` + `dangling_components_rate_avg`
10. `table_pattern_detected_rate`

If your regression target is “quality not just validity,” keep these as acceptance gates, not just `overall_score`.

---

## 7) Exact command paths to inspect or tune during runs

- Enable prompt cache: set `A2UI_ENABLE_PROMPT_CACHE=1`.
- Tune Stage 3 batching and prompt size:
  - `A2UI_GENUI_TEMPERATURE`, `A2UI_GENUI_REPAIR_TEMPERATURE`, `A2UI_GENUI_FINAL_REGEN_TEMPERATURE`
  - `A2UI_GENUI_PROMPT_MAX_TOKENS`
  - `STAGE3_PROMPT_TOKEN_MULTIPLIER`
  - `VLLM_MAX_MODEL_LEN`
  - `LOCAL_VLLM_MAX_OUTPUT_TOKENS`
- For local adapter JSON stability:
  - `LOCAL_VLLM_TOP_P` (default 0.95 for gemma family)
  - `LOCAL_VLLM_TOP_K` (default 64 for gemma family)
  - `LOCAL_MODEL_USE_CACHE`
  - `LOCAL_MODEL_MAX_NEW_TOKENS`

---

## 8) Code path map (quick reference)

- Stage 3 main loop, prompt creation, clipping, caching, batch/fallback: `dataset/src/pipeline/stage3_genui.py`
- Cache implementation: `dataset/src/pipeline/cache.py`
- JSON metrics + score: `dataset/src/pipeline/metrics.py`
- Local model generation + generation-time cache/sampling controls: `dataset/src/llm/local_adapter.py`
- Run flags and env mapping: `dataset/src/main.py`
- Default run configuration including weights: `dataset/configs/run.yaml`
- Default prompt and schema config: `dataset/configs/run.yaml`

---

## 9) Where this is wired into Android

The same flat-spec contract is consumed by Android renderer path:
- `android/app/src/main/java/com/samsung/genuicraft/renderer/FlatSpecRenderer.kt`
- Wrapper parsing and payload handling: `android/app/src/main/java/com/samsung/genuicraft/renderer/GenUiNativeRenderer.kt`
- Android prompt copy file should be kept aligned with dataset prompt updates by your project practice:
  - `android/app/src/main/assets/pipeline_prompts/genui_gen.md`

---

## 10) Practical baseline recommendation

Before changing prompt structure again:
1. Keep Stage 3 base prompt unchanged in core contract rules.
2. Tune only section-domain modules and small token-budget clauses.
3. Measure with: `schema_valid_strict_rate`, `table_cell_coverage_avg`, `section_heading_coverage_avg`, `markdown_leakage_rate_avg`, `overall_score`.
4. Keep repair warnings enabled for trend visibility.
5. Preserve stable cache semantics to avoid noisy A/B signal from repeated generation.

## 11) Direct answer: how prompt quality is measured, optimized, cached, and stabilized for JSON

### 11.1 How JSON IR prompt quality is measured

- Per-sample quality (before run-level aggregation) is computed in:
  - `dataset/src/pipeline/stage3_genui.py::_compute_sample_overall_score(...)` (around line 965): applies `compute_overall_score(...)` to an aggregated row feature set.
  - `dataset/src/pipeline/metrics.py::compute_ui_metrics(...)` (around line 928): derives JSON/IR quality signals such as:
    - `content_coverage`
    - `section_heading_coverage`
    - `table_pattern_detected`
    - `table_cell_coverage`
    - `component_count`, `component_count_norm`, `component_count_capped`
    - `action_coverage`, `url_as_text_rate`
    - `markdown_leakage_rate`, `dup_rate`, `lint_score`
    - `missing_ids_rate`, `dangling_components_rate`, `intent_score`
- Run-level aggregation in `dataset/src/pipeline/metrics.py::aggregate_metrics(...)` (around line 1190) produces averages and rates (e.g., `schema_valid_strict_rate`, `table_cell_coverage_avg`, `section_heading_coverage_avg`, `markdown_leakage_rate_avg`).
- Final run score is produced in `dataset/src/pipeline/metrics.py::compute_overall_score(...)` (around line 1153), with weights from `dataset/configs/run.yaml -> evaluation.weights`.

Relevant stage-level score fields in each Stage 3 record:
- `metrics`: full per-record feature vector.
- `validation`: parse/schema flags (`json_parse_ok`, `schema_valid_strict`, `validation_errors`, retry counts).
- `record["sample_overall_score"]`: current sample score written from `_compute_sample_overall_score`.
- `aggregates.json`: `overall_score` plus all aggregate feature means/rates for the run.

### 11.2 Prompt quality optimizations already present

The Stage 3 pipeline currently optimizes prompt and output quality through these exact paths:

1. **Model-aware compact prompt shaping**
   - `dataset/src/pipeline/stage3_genui.py::_maybe_compact_prompt_template(...)` (line 659) trims / compresses prompt format for Gemma paths when enabled.
2. **System/user prompt split handling**
   - `dataset/src/pipeline/stage3_genui.py::_prepare_prompt_context(...)` (line 713) handles adapter capability differences and uses strict system-prefix flow where needed.
3. **Prompt length governance**
   - `STAGE3_PROMPT_TOKEN_MULTIPLIER` and `A2UI_STAGE3_PROMPT_TOKEN_MULTIPLIER` control a safe budget estimate in Stage 3.
   - `_resolve_context_limited_prompt_max()` (line 804) uses local/remote max-model-length env settings and `STAGE3_CONTEXT_SAFETY_TOKENS` (line 823) to keep prompt under context limits.
4. **Adaptive clipping**
   - `_clip_prompt_to_effective_budget(...)` (line 863) trims non-critical prompt sections in a controlled order before truncating response text if needed.
5. **Resilient JSON recovery**
   - parse/schema loop in Stage 3:
     - initial parse and schema validation
     - repair prompt path (`build_flat_spec_repair_prompt`) (`run_stage3` line 1106 path)
     - final regeneration attempts (`STAGE3_FINAL_REGEN_ATTEMPTS`, line 1188)
     - deterministic fallback flat-spec when needed.
6. **JSON output mode when supported**
   - Local/open endpoint JSON mode is requested via local adapter with `response_format={"type": "json_object"}` in `dataset/src/llm/local_adapter.py::_http_generate(...)` (line 532) when supported.
7. **Asset + markdown hygiene**
   - Stage 3 rewrites/restricts URLs, normalizes text content, repairs images, and removes markdown leak patterns through:
     - `_rewrite_genui_asset_urls(...)` (line 557),
     - `_normalize_flat_spec_text_content(...)` (line 622),
     - `repair_flat_spec_images(...)` (line 1334),
     - `_stage3_quality_warnings(...)` (line 212).

### 11.3 KV cache: what it is and where it is enabled

There are two distinct caches:

- **Prompt cache (dedupe by prompt hash)**  
  - `dataset/src/pipeline/cache.py` implements `PromptCache`.
  - Controlled by run config/environment:
    - `run.yaml` `enable_prompt_cache`
    - `A2UI_ENABLE_PROMPT_CACHE` / `DATASET_ENABLE_PROMPT_CACHE`
  - Pipeline passes cache object to stage runs from `dataset/src/main.py::_build_prompt_cache(...)` (line 64).

- **Model KV cache (generation-time attention cache)**  
  - Local generation path:
    - `dataset/src/llm/local_adapter.py::_local_generate(...)` (around line 855) sets `gen_kwargs["use_cache"]`.
    - Default behavior is `use_cache=False` unless adapter/model profile explicitly allows otherwise.
    - Override in env via `LOCAL_MODEL_USE_CACHE`.
  - Remote vLLM path (`_http_generate`) is governed by server-side flags/runtime, not by this local `use_cache` flag path.

### 11.4 Where to look when changing prompt quality

- **Prompt inputs**
  - `dataset/prompts/genui_gen_mobile_flatspec_v11.md`
  - Android mirror: `android/app/src/main/assets/pipeline_prompts/genui_gen.md`
- **Prompt assembly and constraints**
  - `dataset/src/pipeline/stage3_genui.py` (`_build_prompt_for`, `_clip_prompt_to_effective_budget`, warning/repair paths)
- **Validation and scoring**
  - `dataset/src/pipeline/metrics.py`
- **Run-level output**
  - `aggregates.json` and `run_manifest.json` per run folder

### 11.5 Best-practice checklist for JSON IR generation (non-breaking)

Keep these as mandatory checks during experiments:

1. Maintain schema strictness as hard gate:
   - `schema_valid_strict_rate` should stay stable; never allow regressions just to increase token density.
2. Improve structure before aesthetics:
   - target higher `table_cell_coverage_avg`, `section_heading_coverage_avg`, lower `markdown_leakage_rate_avg`, lower `url_as_text_rate_avg`.
3. Maintain intent integrity:
   - monitor `intent_score_avg`, `action_coverage_avg`, `section_heading_coverage_avg`.
4. Preserve readability:
   - keep `component_count` in a useful range (not empty, not over-fragmented).
5. Track compactness side effects:
   - watch `output_tokens_json_avg` and `output_chars_json_avg` against prompt size changes.
6. Use repair telemetry:
   - inspect warning fields (`repair_needed`, `repair_attempts`, `schema_repaired`, warnings list from `_stage3_quality_warnings`) before changing prompts.

### 11.6 Recommended env variable map (Stage 3 JSON-focused)

- Prompt size and context:
  - `A2UI_GENUI_PROMPT_MAX_TOKENS`
  - `STAGE3_PROMPT_TOKEN_MULTIPLIER`
  - `A2UI_STAGE3_PROMPT_TOKEN_MULTIPLIER`
  - `STAGE3_CONTEXT_SAFETY_TOKENS`
  - `VLLM_MAX_MODEL_LEN`
  - `LOCAL_VLLM_MAX_MODEL_LEN`
  - `A2UI_VLLM_MAX_MODEL_LEN`
- Decoding:
  - `A2UI_GENUI_TEMPERATURE`
  - `A2UI_GENUI_REPAIR_TEMPERATURE`
  - `A2UI_GENUI_FINAL_REGEN_TEMPERATURE`
  - `A2UI_GENUI_MAX_TOKENS`
  - `LOCAL_VLLM_TOP_P`, `LOCAL_VLLM_TOP_K`, `LOCAL_VLLM_REPETITION_PENALTY`
- Caching:
  - `A2UI_ENABLE_PROMPT_CACHE`
  - `LOCAL_MODEL_USE_CACHE`
- Retry behavior:
  - `max_repair_attempts` from `run.yaml`
  - `STAGE3_FINAL_REGEN_ATTEMPTS` (fallback control in `run_stage3()`)

### 11.7 How to send this document

This file was written at:
`docs/stage3_prompt_quality_and_json_ir_generation.md`

Share by attaching that file path to:
`anupkushwaha@gmail.com`
