# A2UI – Dataset / GenUICraft Pipeline Work Report

Date: **2026-02-16**  
Scope: `dataset/` folder (generation pipeline, prompts, metrics, renderer, visualizer, run artifacts)

## Executive Summary

Over this development cycle, the project was transformed into a reproducible, multi-stage benchmark pipeline for evaluating LLMs on generating **GenUICraft** UI IR (JSON + TOON) and producing **rendered** HTML/PNG outputs for inspection.

Key outcomes:

- Built/maintained an end-to-end pipeline: **Queries → Responses → GenUICraft IR → Metrics/Aggregates → HTML/PNG rendering**.
- Added/updated multi-provider LLM support (Gemini, OpenAI, Gauss, OpenRouter, Perplexity, plus local Qwen/DeepSeek).
- Improved reliability (per-item error handling, retries, recompute-only mode, safer caps/limits for local models).
- Expanded and documented Stage-3 metrics (including intent requirement checks) and improved reporting/visualization.
- Significantly improved Stage-4 output quality (tables, cards, buttons, link clickability, icons/logos, layout fixes).
- Added repo-level versioning for releases and per-component versions.

## Deliverables (What Exists Now)

### 1) Dataset pipeline (Stages 1–4)

- **Stage 1 – Queries**
  - Generates and stores `queries.jsonl`.
  - Writes run manifests (`run_manifest.json`) for reproducibility.
- **Stage 2 – Responses**
  - Generates `responses.jsonl` with structured outputs designed to preserve downstream UI fidelity.
  - Emphasis on keeping actions/URLs near the element they belong to (supports later linking).
- **Stage 3 – GenUICraft IR**
  - Converts responses into `genui.jsonl` with:
    - JSON IR (`genui_json`)
    - TOON representation (`toon`)
    - Validation outcomes (strict/lenient + repair)
    - Per-item metrics and aggregate computations.
- **Stage 4 – Rendering**
  - Produces `/rendered/*.html` and `/rendered/*.png` for each run.
  - Uses Lit renderer + theme tokens; screenshot capture via Playwright when available.

### 2) Provider integrations (LLM adapters)

Implemented/maintained adapters under `dataset/src/llm/` and configuration in `dataset/configs/models.yaml`:

- Gemini (including Gemini 3 JSON mode + batch workflows).
- OpenAI.
- OpenRouter.
- Gauss (including endpoint/env hardening and prompt/token clamps).
- Perplexity (API integration for stages 1–3, including “gpt5” flow).
- Local Qwen/DeepSeek support via `transformers`, with strict offline controls.

### 3) Prompt system and prompt experiments

Prompts are maintained in `dataset/prompts/`:

- `query_gen.md`: query generation
- `response_gen.md`: response generation
- `genui_gen.md`: IR generation

Work completed:

- Prompt versioning/experiments to improve IR quality and/or reduce overhead.
- Multiple experimental runs saved under `dataset/data/runs/` (e.g., promptlite/system-prefix/prompttuned/promptv6).
- Updates to response prompt for better “action link locality” (URLs near the element they describe).

### 4) Metrics, scoring, and documentation

Core metrics pipeline improvements in `dataset/src/pipeline/metrics.py`:

- Added/standardized Stage-3 structural metrics (coverage, duplication, lint, tree stats, modularity/decomposition signals).
- Table detection + table quality signals (e.g., pattern detection and cell coverage).
- Intent requirement metrics (table/actions/sections expected vs OK), plus intent scoring and expectation pass rates.
- Added output size comparisons for IR encodings:
  - `output_tokens_*` and `output_chars_*` (JSON vs TOON).
  - Component count normalization/capping support (`component_count_norm`, `component_count_capped`).
- Aggregate computation and `overall_score` support (with clarified bounds and banding guidance).

Documentation:

- `dataset/scripts/STAGE3_METRICS.md` documents:
  - Each metric meaning and interpretation,
  - Per-intent “baseline requirements” and which checks apply,
  - Practical banding (bad/ok/good/excellent) guidance.

### 5) Reporting and comparison utilities

Scripts under `dataset/scripts/` support recomputation, backfills, and comparisons:

- `report_model_rankings.py`: ranks model runs across a folder of runs; exports CSV/JSON.
- `recompute_aggregates_stream.py`: recompute aggregates without regenerating everything.
- `backfill_genui_metrics.py`: backfill missing genui metrics on existing runs.
- `refresh_bin_assets.py`: refresh renderer/bin assets used in output.

### 6) Visualizer

The visualizer provides an inspection UI over `dataset/data/runs/`:

- Run server: `python dataset/visualizer/app.py --port 8008`
- Open: `http://127.0.0.1:8008`

Work completed:

- Improved metrics presentation (including metric band visualization and coloring).
- Added/extended per-item display fields as needed for debugging (intent/IR surfaced in the UI).

### 7) Renderer / UI quality improvements (Stage 4)

The stage-4 renderer was iteratively improved to match “manual HTML quality” more closely:

- Buttons:
  - Cursor/pointer behavior, press/hover styling, compact heights and padding.
  - Better consistency between theme styles and runtime-injected styles.
- Tables:
  - Improved list/column detection into proper table-like layouts.
  - Better spacing, borders, and header styling.
- Cards:
  - Consistent corner radius + clipping to prevent corner artifacts.
- Icons/logos:
  - Icon sizing normalization for airline logos (avoid cropping / show full logos).
  - Support for local icon catalogs and mapping (license tracking included).
- Layout:
  - “Media rail” layout (place a relevant image to the right of a list when useful).
  - Fixed cases where media-rail grid behavior created large whitespace gaps.
  - Fixed cases where right-side image appeared with blank left column by pinning list+image to the same row.

## Timeline / Milestones (Recent, since 2026-02-01)

High-level milestones taken from commit history:

- **2026-02-02**
  - Rebrand pipeline/UI text to GenUICraft; embed local schema in Stage-3 prompt; stage-4 renders for OpenRouter.
- **2026-02-03 → 2026-02-04**
  - GenUI/TOON encoding changes; stage-3 schema and Gauss hardening (env parsing, endpoints, max token clamps).
  - Gemini batch support for stages 1–3 and JSON mode enablement.
- **2026-02-05 → 2026-02-08**
  - Local model support improved (vLLM, then transformers), strict offline enforcement, per-item network error handling.
  - Recompute-only stage-3 mode; improved OOM handling.
- **2026-02-09 → 2026-02-10**
  - Metrics pipeline updates; stage-3 metrics mapping documentation; remove unused assets; stage-4 refresh/regeneration for runs.
- **2026-02-11 → 2026-02-13**
  - Metric band visualization; Perplexity support for stages 1–3; prompt versioning + tuned prompt runs; stage-4 reruns for prompt variants.

## Work In Progress / Local Changes (Not Yet Committed)

At the time of writing (2026-02-16), the working tree includes additional improvements not yet committed upstream:

- Icon catalog + mapping configuration:
  - `dataset/assets/…` catalog contents
  - `dataset/configs/icon_map.json`
  - License/source tracking files (e.g., `SOURCES_AND_LICENSES.md`).
- Stage-4 renderer refinements:
  - Further compact button sizing.
  - Logo/icon sizing tweaks (airline logos).
  - Media-rail behavior fixes (reduce whitespace; pin list+image).
- Additional experimental runs under `dataset/data/runs/subset10_*` and `subset50_*` used for quick validation.

## Risks / Notes

- **API latency and rate limits**: prompt size is only one factor; network, provider load, and batch sizing can dominate latency.
- **Asset licensing**: icon catalogs require explicit license tracking (implemented via `SOURCES_AND_LICENSES.md`).
- **Reproducibility**: run manifests + version tracking are critical to compare prompt/provider changes fairly.

## How to Run (Quick Reference)

From `dataset/`:

```bash
python src/main.py --stage 1 --model <model_name> --run_id <run_id>
python src/main.py --stage 2 --model <model_name> --run_id <run_id>
python src/main.py --stage 3 --model <model_name> --run_id <run_id>
python src/main.py --stage 4 --model <model_name> --run_id <run_id>
```

Model comparison report example:

```bash
python scripts/report_model_rankings.py --runs-dir data/runs --include-glob "subset50_*" --format table
```

Visualizer:

```bash
python visualizer/app.py --port 8008
```

