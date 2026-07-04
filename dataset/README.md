# Dataset Generation + LLM Benchmark

This folder contains pipeline to generate user queries, responses, GenUICraft JSON, metrics and benchmarking.

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
      gauss_adapter.py
      openrouter_adapter.py
      perplexity_adapter.py
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
$env:PERPLEXITY_API_KEY="..."
$env:PERPLEXITY_API_BASE="https://api.perplexity.ai"
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

## Versioning

The repo tracks release and component versions.

- `VERSION`: current release version.
- `versions/components.yaml`: per-component versions and compatibility flags.
- `data/runs/<run_id>/run_manifest.json`: auto-written manifest for each run.

Print current version info:

```
python src/main.py --version
```

Bump release/component versions:

```
python scripts/bump_version.py --bump patch
python scripts/bump_version.py --release 0.2.0 --component pipeline=0.2.0 --component renderer=0.2.0
python scripts/bump_version.py --set-compat genui_schema=v0.9
```

Use `--dry-run` to preview changes.
## Run stages

From the `dataset/` folder:

```
python src/main.py --stage 1 --model openai_gpt4o
python src/main.py --stage 2 --model openai_gpt4o
python src/main.py --stage 3 --model openai_gpt4o
python src/main.py --stage 4 --model openai_gpt4o

# Perplexity (sonar-pro via Perplexity API; gpt5 alias maps to sonar-pro)
python src/main.py --stage 1 --model perplexity_gpt5 --run_id perplexity_gpt5
python src/main.py --stage 2 --model perplexity_gpt5 --run_id perplexity_gpt5
python src/main.py --stage 3 --model perplexity_gpt5 --run_id perplexity_gpt5
```

Outputs are written to `data/runs/<run_id>/`.

## Dataset dashboard

Use the generated dataset dashboard to inspect all local runs and optionally mirror runs from other machines.
The local checkout is scanned automatically; remote sources are configured separately so credentials are not committed.

Create a private source config:

```
Copy-Item configs\dataset_dashboard.sources.example.json configs\dataset_dashboard.sources.json
```

Edit `configs/dataset_dashboard.sources.json` and enable any required source:

- `local`: copies from another local folder.
- `ssh`: lists files with `ssh` and copies changed files with `scp`.
- `command`: uses custom list/copy commands for non-standard storage.

For `ssh` sources, prefer passwordless key auth with `identity_file`. For password auth, install
the optional Python package `paramiko`:

```
python -m pip install paramiko
```

Then either set `password` in `configs/dataset_dashboard.sources.json`, or set
`"ask_password": true` to type the password once per dashboard process. Prompted passwords are kept
only in memory. Passwords stored in config are plaintext, so keep
`configs/dataset_dashboard.sources.json` uncommitted. If you explicitly set
`"ssh_backend": "openssh"` with a password, `sshpass` is required.

SSH sources also support jump hosts through either `proxy_jump` or `proxy_command`. Use
`proxy_command` when your working command uses `-o ProxyCommand="ssh ... -W %h:%p jump-host"`;
the dashboard passes the same option to both `ssh` and `scp`.

`path` can be a concrete directory or a glob pattern. For example, to sync only runs whose folder
starts with `dataset_v1`, set:

```
"path": "/home/anup/A2UI/dataset/data/runs/dataset_v1*",
"path_match": "glob"
```

The dashboard auto-detects glob mode when `path` contains `*`, `?`, or `[...]`, so `path_match`
can usually be omitted. The matched folder name is preserved in the local mirror.

For true regex matching, set `path_base` to the parent folder and match immediate child directory
names under it:

```
"path_base": "/home/anup/A2UI/dataset/data/runs",
"path": "^dataset_v1.*",
"path_match": "regex"
```

Start the dashboard from the `dataset/` folder:

```
python scripts/dataset_dashboard.py --sync-on-start
```

Then open:

```
http://127.0.0.1:8765
```

The UI shows separate query, response, and IR counts for every source. Use the source dropdown,
IR version dropdown, text filter, score filter, or date range filters to narrow the visible runs;
the top stats, source cards, and day-wise section recompute counts from the filtered runs. Date
filters use each run's daily buckets, and IR version filters use each run's `gen.prompt_version`
metadata, so query/response/IR counts reflect only matching records where available. With all
sources selected, day-wise counts are aggregated across every matching source. The dashboard auto
refreshes the local scan every 30 seconds by default; use the Auto refresh dropdown to change the
interval or turn it off.

Source cards include a health badge based on the latest sync result and a `Test` action that checks
the configured local/SSH/command source without copying files. The dashboard also shows response and
IR backlog (`queries - responses`, `responses - IR`) for the current filters, plus a model comparison
table that groups runs by dominant Stage 2 response model and Stage 3 IR model with score and quality
signals such as content coverage, section coverage, table coverage, action coverage, and media usage.

One-off commands:

```
python scripts/dataset_dashboard.py --sync-once
python scripts/dataset_dashboard.py --summary-once
```

Remote files are mirrored into `data/dashboard_mirror/`. The sync manifest stores source `size`
and `mtime`, so repeat syncs copy only new or updated files. While sync is running, the dashboard
shows the active source, current file, listed/processed/copied/skipped/error counters, and recent
sync messages.

## Local Qwen3 / DeepSeek

Qwen and DeepSeek local models run directly via `transformers` in `LocalAdapter`.
Set model paths and visible GPUs:

```
$env:QWEN_MODEL_PATH="/path/to/Qwen3-Coder-30B-A3B-Instruct"
$env:DEEPSEEK_MODEL_PATH="/path/to/DeepSeek-Coder-V2-Lite-Instruct"
$env:CUDA_VISIBLE_DEVICES="0,1,2,3"   # 4x V100 example
$env:LOCAL_STRICT_OFFLINE="1"
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
$env:HF_DATASETS_OFFLINE="1"
$env:DATASET_OFFLINE_MODE="1"
```

Run stages:

```
python src/main.py --stage 1 --model qwen3_coder_30b_a3b_local_4x --run_id qwen3_4x
python src/main.py --stage 2 --model qwen3_coder_30b_a3b_local_4x --run_id qwen3_4x
python src/main.py --stage 3 --model qwen3_coder_30b_a3b_local_4x --run_id qwen3_4x

python src/main.py --stage 1 --model deepseek_coder_v2_lite_local_4x --run_id deepseek_4x
python src/main.py --stage 2 --model deepseek_coder_v2_lite_local_4x --run_id deepseek_4x
python src/main.py --stage 3 --model deepseek_coder_v2_lite_local_4x --run_id deepseek_4x
```

Optional per-run model path override (without editing env):

```
python src/main.py --stage 3 --model qwen3_coder_30b_a3b_local_4x --local_model_path /path/to/Qwen3-Coder-30B-A3B-Instruct
python src/main.py --stage 3 --model deepseek_coder_v2_lite_local_4x --local_model_path /path/to/DeepSeek-Coder-V2-Lite-Instruct
```

Optional local inference tuning env vars:

- `LOCAL_MODEL_DEVICE_MAP` (default `auto`)
- `LOCAL_MODEL_DTYPE` (`float16`, `bfloat16`, `float32`)
- `LOCAL_MODEL_LOAD_IN_4BIT` (`1` to enable)
- `LOCAL_MODEL_MAX_MEMORY` (example `14GiB`)
- `LOCAL_MODEL_GPU_MEMORY_UTILIZATION` (default `0.88`; auto cap when `LOCAL_MODEL_MAX_MEMORY` is unset)
- `LOCAL_CUDA_ALLOC_CONF` (optional; sets `PYTORCH_CUDA_ALLOC_CONF` only if you explicitly provide it)
- `LOCAL_MODEL_OFFLOAD_DIR` (default `.offload`)
- `LOCAL_MODEL_MAX_INPUT_TOKENS` (default `8192` for Qwen/DeepSeek)
- `LOCAL_MODEL_MAX_NEW_TOKENS` (default `1024` for Qwen/DeepSeek)
- `LOCAL_MODEL_USE_CACHE` (`0`/`1`; default `0` for Qwen/DeepSeek to reduce memory)
- `LOCAL_STAGE3_PROMPT_MAX_TOKENS` (default `8192` for local stage-3 prompt truncation)
- `LOCAL_STRICT_OFFLINE` (`1`/`0`; default auto-on for qwen/deepseek)
- `LOCAL_FILES_ONLY` (`1` to enforce local files for all local models)
- `LOCAL_ALLOW_HTTP_ENDPOINT` (`1` to allow HTTP endpoint even in strict offline mode)
- `DATASET_OFFLINE_MODE` (`1` disables stage-2 URL asset downloads)

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







