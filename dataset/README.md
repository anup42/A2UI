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

## Local Qwen3-Coder-30B-A3B-Instruct (vLLM)

To run the pipeline on a local multi-GPU server (4x or 8x V100), start a vLLM
OpenAI-compatible server with the provided script, using your local model folder:

```
python scripts/serve_qwen_vllm.py --model-path /path/to/Qwen3-Coder-30B-A3B-Instruct --gpus 4
```

For 8 GPUs:

```
python scripts/serve_qwen_vllm.py --model-path /path/to/Qwen3-Coder-30B-A3B-Instruct --gpus 8
```

Then run stages with one of these model names from `configs/models.yaml`:

```
python src/main.py --stage 1 --model qwen3_coder_30b_a3b_vllm_4x --run_id qwen3_4x
python src/main.py --stage 2 --model qwen3_coder_30b_a3b_vllm_4x --run_id qwen3_4x
python src/main.py --stage 3 --model qwen3_coder_30b_a3b_vllm_4x --run_id qwen3_4x
```

### Auto-start vLLM (recommended)

Instead of starting vLLM manually, you can let the stage runner auto-start it
with `--start_vllm`. It will launch vLLM (if not already running), wait for it,
run the stage, and shut it down when the stage completes.

Example (Qwen, 4x V100):

```
python src/main.py --stage 1 --model qwen3_coder_30b_a3b_vllm_4x --run_id qwen3_4x ^
  --start_vllm --vllm_model_path /path/to/Qwen3-Coder-30B-A3B-Instruct --vllm_gpus 4
```

## Local DeepSeek-Coder-V2-Lite-Instruct (vLLM)

Start vLLM with your local DeepSeek model folder:

```
python scripts/serve_qwen_vllm.py --model-path /path/to/DeepSeek-Coder-V2-Lite-Instruct --gpus 4 --served-model-name deepseek-coder-v2-lite-instruct
```

Then run stages:

```
python src/main.py --stage 1 --model deepseek_coder_v2_lite_vllm_4x --run_id deepseek_4x
python src/main.py --stage 2 --model deepseek_coder_v2_lite_vllm_4x --run_id deepseek_4x
python src/main.py --stage 3 --model deepseek_coder_v2_lite_vllm_4x --run_id deepseek_4x
```

Example (DeepSeek, auto-start):

```
python src/main.py --stage 1 --model deepseek_coder_v2_lite_vllm_4x --run_id deepseek_4x ^
  --start_vllm --vllm_model_path /path/to/DeepSeek-Coder-V2-Lite-Instruct --vllm_gpus 4
```

### Auto-start flags (important)

- `--start_vllm`: enable auto-start for local Qwen/DeepSeek models
- `--vllm_model_path`: **required** local model folder path (or set `QWEN_MODEL_PATH`)
- `--vllm_gpus`: tensor parallel size (4 or 8 for V100)
- `--vllm_cuda_visible_devices`: pin specific GPU IDs (e.g., `0,1,2,3`)
- `--vllm_port`: port for vLLM (default `8000`)
- `--vllm_gpu_mem_util`: GPU memory utilization (default `0.90`)
- `--vllm_max_model_len`: override max model length if needed
- `--vllm_swap_space`: enable CPU swap space for KV cache (GB)

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



