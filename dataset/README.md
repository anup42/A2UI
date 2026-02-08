# Dataset Generation + LLM Benchmark Harness

This folder contains a resumable, multi?stage pipeline to generate user queries, responses, GenUICraft JSON, and TOON outputs, plus metrics and benchmarking.

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
    genui_gen.md
  schema/
    genui.schema.json
  src/
    main.py
    pipeline/
      stage1_queries.py
      stage2_responses.py
      stage3_genui.py
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
        genui.jsonl
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

## Local Qwen3 / DeepSeek (direct Transformers, no vLLM)

Qwen and DeepSeek local models now run directly via `transformers` in `LocalAdapter`.
You do not need to run a vLLM server for these model entries.

Set model paths and visible GPUs:

```
$env:QWEN_MODEL_PATH="/path/to/Qwen3-Coder-30B-A3B-Instruct"
$env:DEEPSEEK_MODEL_PATH="/path/to/DeepSeek-Coder-V2-Lite-Instruct"
$env:CUDA_VISIBLE_DEVICES="0,1,2,3"   # 4x V100 example
```

Run stages:

```
python src/main.py --stage 1 --model qwen3_coder_30b_a3b_vllm_4x --run_id qwen3_4x
python src/main.py --stage 2 --model qwen3_coder_30b_a3b_vllm_4x --run_id qwen3_4x
python src/main.py --stage 3 --model qwen3_coder_30b_a3b_vllm_4x --run_id qwen3_4x

python src/main.py --stage 1 --model deepseek_coder_v2_lite_vllm_4x --run_id deepseek_4x
python src/main.py --stage 2 --model deepseek_coder_v2_lite_vllm_4x --run_id deepseek_4x
python src/main.py --stage 3 --model deepseek_coder_v2_lite_vllm_4x --run_id deepseek_4x
```

Optional per-run model path override (without editing env):

```
python src/main.py --stage 3 --model qwen3_coder_30b_a3b_vllm_4x --local_model_path /path/to/Qwen3-Coder-30B-A3B-Instruct
python src/main.py --stage 3 --model deepseek_coder_v2_lite_vllm_4x --local_model_path /path/to/DeepSeek-Coder-V2-Lite-Instruct
```

Optional local inference tuning env vars:

- `LOCAL_MODEL_DEVICE_MAP` (default `auto`)
- `LOCAL_MODEL_DTYPE` (`float16`, `bfloat16`, `float32`)
- `LOCAL_MODEL_LOAD_IN_4BIT` (`1` to enable)
- `LOCAL_MODEL_MAX_MEMORY` (example `14GiB`)
- `LOCAL_MODEL_OFFLOAD_DIR` (default `.offload`)

Stage 4 renders GenUICraft JSON to HTML + PNG using the Lit renderer assets copied
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

- GenUICraft schema is in `schema/genui.schema.json`.
- `jsonschema` is optional; if missing, strict validation is marked false with a warning.
- TOON is encoded in `src/pipeline/toon_convert.py` and can be swapped for a custom spec.



