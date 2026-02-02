# Dataset Generation + LLM Benchmark Harness

This folder contains a resumable, multi?stage pipeline to generate user queries, responses, A2UI JSON, and TOON outputs, plus metrics and benchmarking.

## Structure

```
dataset/
  intents.info
  configs/
    run.yaml
    models.yaml
  prompts/
    query_gen.md
    response_gen.md
    a2ui_gen.md
  schema/
    a2ui.schema.json
  src/
    main.py
    pipeline/
      stage1_queries.py
      stage2_responses.py
      stage3_a2ui.py
      toon_convert.py
      metrics.py
      storage.py
    llm/
      base.py
      openai_adapter.py
      gemini_adapter.py
      local_adapter.py
    utils/
      hashing.py
      retry.py
      rate_limit.py
      logging.py
  data/
    runs/
      RUN_ID/
        queries.jsonl
        responses.jsonl
        a2ui.jsonl
        metrics.jsonl
        aggregates.json
        artifacts/
```

## Setup

Set your API keys in the environment:

PowerShell:

```
$env:OPENAI_API_KEY="..."
$env:GEMINI_API_KEY="..."
$env:OPENROUTER_API_KEY="..."
$env:OPENROUTER_API_BASE="https://openrouter.ai/api/v1"
$env:OPENROUTER_SITE_URL="https://your-site.example"
$env:OPENROUTER_APP_NAME="DatasetRunner"
$env:GAUSS_ENDPOINT="https://your-host"
$env:GAUSS_CLIENT_KEY="..."
$env:GAUSS_OPENAPI_TOKEN="Bearer ..."
$env:GAUSS_USER_EMAIL="you@example.com"
$env:GAUSS_MODEL_ID="your-model-id"
```

## Run stages

From the `dataset/` folder:

```
python src/main.py --stage 1 --model openai_gpt4o
python src/main.py --stage 2 --model openai_gpt4o
python src/main.py --stage 3 --model openai_gpt4o
python src/main.py --stage 4 --model openai_gpt4o
```

Outputs are written to `data/runs/<run_id>/`.

Stage 4 renders A2UI JSON to HTML + PNG using the Lit renderer assets copied
from `renderers/lit/dist` into `renderer/lit`. Image capture uses Playwright
if installed; if not, HTML is still generated and rendering errors are logged.

## Print configured limits

```
python src/main.py --print_limits
```

Configure per-model limits in `configs/models.yaml` under each model's `limits` block.

## Benchmark mode

```
python src/main.py --benchmark_models openai_gpt4o gemini_1_5_pro
```

This runs a fixed subset of queries against each model and stores per?model aggregates.

## Notes

- A2UI schema is in `schema/a2ui.schema.json`.
- `jsonschema` is optional; if missing, strict validation is marked false with a warning.
- TOON is encoded in `src/pipeline/toon_convert.py` and can be swapped for a custom spec.

