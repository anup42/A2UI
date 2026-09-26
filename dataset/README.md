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

Source sync runs in parallel by default. Set `max_parallel_sources` in
`configs/dataset_dashboard.sources.json` to change the saved default, or use the `Parallel sources`
control in the page toolbar before clicking `Sync sources`. The default is `10`.
SSH sources default to `transfer_mode: "auto"`, which archives all changed files into one remote
`tar.gz`, downloads that archive once, and extracts it into the local mirror. Set
`transfer_mode: "per_file"` on a source to force the older one-file-at-a-time copy behavior. If
archive creation or download fails, the dashboard reports `archive fallback` and retries with
per-file copy for that source.

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
Use the run-health filter to focus on backlog, quality warnings, low content coverage, low media
usage, broken media references, data integrity issues, or log issues, and use the sort dropdown to
rank runs by freshness, score, IR count, backlog, or issue count. The runs table is paginated with
selectable page size, while the top stats, source cards, backlog, quality alerts, metrics overview,
intent quality, model comparison, distributions, exports, and trend chart continue to use the full
filtered run set.

Source cards include a health badge based on the latest sync result and a `Test` action that checks
the configured local/SSH/command source without copying files. The dashboard also shows response and
IR backlog (`queries - responses`, `responses - IR`) for the current filters, plus a model comparison
table that groups runs by dominant Stage 2 response model and Stage 3 IR model with score and quality
signals such as content coverage, section coverage, table coverage, action coverage, and media usage.
The IR version quality panel groups filtered runs by Stage 3 prompt/IR version and shows count, score,
date span, source coverage, and associated Stage 3 models so prompt changes can be compared directly.
The sync configuration panel shows the effective local checkout plus configured sources with redacted
credential indicators, path matching mode, SSH/proxy settings, include/exclude globs, and mirror path.
The last-sync panel shows per-source listed/copied/skipped/error counts and mirror destinations.
The freshness panel highlights the latest matching run update, last sync age, sources with no
matching update in 7 days, runs older than 7 or 30 days, and runs that still have response or IR
backlog. Freshness follows the same source/date/score/text/IR-version filters as the rest of the
dashboard, so it can be used to identify which remote generator or filtered slice stopped updating.
The action-items panel ranks the most urgent source/run problems from the filtered view, combining
sync errors, stale sources, backlog, data integrity, sampled duplicate content, sampled quality
failures, and artifact gaps into a short list with links back to run details.
The completion funnel panel shows exact query-to-response, response-to-IR, and query-to-IR
conversion rates, then combines them with sampled readiness estimates for strict-valid IR and
score-threshold-ready records. Its source table makes it clear which generator/source is losing data
at each pipeline step.
The run-logs panel scans bounded tails from `*.log`, `logs/*.log`, and nested `run.log` files to
surface recent errors, exceptions, HTTP failures, rate-limit messages, progress lines, and stale log
updates. Use the run-health filter `Log issues` or sort by log issues to isolate failing workers.
The throughput panel uses the filtered day buckets to show last-7-day query, response, and IR volume,
response/day and IR/day rates, source-level bottlenecks, and backlog ETA. If a filtered slice has
backlog but no recent response or IR rate, the ETA reports `no recent rate` so stalled workers are
easy to distinguish from slow-but-active workers.
The metrics overview panel shows score bands, weighted average score, weighted core metric averages,
and sampled validation failure/repair/fallback rates for the current filters. Weighted averages use
filtered IR count so larger runs contribute proportionally more than smoke tests.
The metric risk panel applies heuristic thresholds to per-run sampled metric averages and lists the
most common failing quality dimensions plus the highest-risk runs. Use the `Metric risk` run-health
filter or `Sort: metric risk` to isolate runs likely failing due to low coverage, missing
headings/tables/actions/media, markdown leakage, or sparse structure.
The training readiness panel estimates how many response-to-IR pairs are usable for SFT/evaluation
at common score thresholds. It combines paired response/IR counts with sampled Stage 3 gates: JSON
parse success, strict schema pass, no generation error, no fallback, no markdown leakage, and score
`>=70` or `>=80`. It also lists weakest training slices and largest ready runs so training data can
be selected without manually inspecting every run.
The IR structure panel samples flat-spec records and reports component type mix, table domains,
presentation hints, action types, image/icon usage, special components, and uncommon component names.
Use it to quickly catch prompt drift, unsupported components, missing media usage, or renderer coverage
gaps before inspecting individual records.
The media and asset health panel samples `responses.jsonl` asset records plus flat-spec Image/Icon
props to show declared asset records, present/missing local files, remote media references left in IR,
media host distribution, file extension mix, and sample broken references. Use the run-health filter
`Broken media refs` to isolate runs whose copied assets are incomplete or whose IR still points at
remote media instead of localized assets.
The data integrity panel scans core JSONL IDs and links for duplicate IDs, missing IDs, parse errors,
count anomalies, and broken stage links such as IR records whose `response_id` is not present in
`responses.jsonl`. Use the run-health filter `Data integrity issues` to isolate affected runs.
The content duplicates panel checks sampled normalized query text, response text, and IR payloads for
exact duplicate content. Use the run-health filter `Content duplicates` or sort by content duplicates
to isolate runs that may reduce training diversity even when IDs are unique.
The intent quality panel aggregates sampled `genui.jsonl` rows by intent/domain, showing largest
intent buckets and lowest-scoring intents with coverage, heading, table, action, image, and icon
signals. This makes it easier to identify weak domains even when the overall score looks acceptable.
The token/cost/latency panel aggregates available `gen` metadata by stage and model, including input
tokens, output tokens, total tokens, average latency, known cost, and generation errors. CSV exports
include Stage 2 and Stage 3 token, latency, and cost fields for each filtered run.
The prompt provenance panel aggregates Stage 1 query prompt versions, Stage 2 response prompt
versions, and Stage 3 IR prompt versions. It shows full prompt lineages, highlights runs that mix
multiple prompt versions inside one stage, and makes prompt-version strings searchable from the main
text filter. CSV exports include all three stage prompt-version columns.
The regression watch panel compares the latest run against the previous run with the same source,
dominant Stage 2 model, dominant Stage 3 model, and Stage 3 prompt version. It highlights score
drops, recent weak runs, and links directly to latest/previous run details for inspection.
The storage and artifacts panel shows filtered disk usage, file counts, asset/screenshot bytes,
storage by source, largest runs, runs missing core files, and runs with generated IR but no captured
screenshots. CSV and JSON exports also include storage and missing-core metadata for the filtered run
view.
Use `Details` on any run to inspect model counts, IR prompt versions, intent mix, sampled validation
warnings, prompt provenance, training readiness, IR structure, media health, duplicate examples,
sampled records, and metric breakdowns.
Sampled records show issue-prioritized query text, response previews, score, validation notes, and
compact IR component summaries so failing examples can be inspected without opening JSONL files.
`Export CSV` and `Export JSON` download the currently filtered run view, which is useful for sharing
source-specific or score-thresholded slices. CSV exports include media health counters such as
missing response assets, missing local IR media refs, remote IR media refs, sample-record count, and
training readiness estimates. The run detail pane also shows parsed `run_manifest.json` provenance,
core artifacts, sample assets, and sample screenshots with copyable local paths, and the trend chart
shows filtered daily volume with average IR score over time. Filtered distribution panels show top
intents, Stage 2 models, Stage 3 models, IR versions, and source volume for the current view.
The worst-sampled-records panel highlights low-score, repaired, fallback, markdown-leaking, sparse,
or schema-failing IR rows from sampled `genui.jsonl` records and links back to the owning run details.

One-off commands:

```
python scripts/dataset_dashboard.py --sync-once
python scripts/dataset_dashboard.py --summary-once
```

Remote files are mirrored into `data/dashboard_mirror/`. The sync manifest stores source `size`
and `mtime`, so repeat syncs copy only new or updated files. While sync is running, the dashboard
shows aggregate counters plus per-source phase, transfer mode, changed-file count, archive size,
current file, copied/skipped/error counts, and recent sync messages.

## Muse Glimmer 30B dataset generation

The `muse_glimmer_30b_sglang_reasoning_dflash` model entry supports cyclic
Stage 1 query, Stage 2 response, and Stage 3 A2UI Express generation in one
Muse run. It can also generate Stage 3 alone from another run's Stage 1/2 data.
It uses the BF16 text decoder, Muse reasoning parser, and the official DFlash
assistant. Dataset generation is text only, so the launcher omits the
vision tower to leave more H100 memory for concurrent requests.

Install a Muse-capable SGLang build on the Linux GPU host. At the time this
profile was added, Meta and SGLang document the `muse-glimmer` branch or
`lmsysorg/sglang:dev-muse-glimmer` image, not a released SGLang wheel. The
launcher uses the Python branch installation:

```bash
python3 -m venv ~/venvs/muse-sglang
source ~/venvs/muse-sglang/bin/activate
python -m pip install --upgrade pip uv huggingface_hub
git clone -b muse-glimmer https://github.com/sgl-project/sglang.git
cd sglang
uv pip install --prerelease=allow -e "python[all]"
hf download meta-models/Muse-Glimmer-30B --local-dir /models/Muse-Glimmer-30B
hf download meta-models/Muse-Glimmer-30B-assistant --local-dir /models/Muse-Glimmer-30B-assistant
```

Activate that Python environment and change into this A2UI repo in each shell.
Use two shells on the H100 host. Substitute `--gpus 8` for an
8-GPU node; the default is one replica per H100, so 4 GPUs create 4 endpoints
and 8 GPUs create 8. Both commands must use the same GPU count, tensor
parallel setting, and base port.

```bash
python dataset/scripts/run_muse_glimmer_stage3.py plan --gpus 4
python dataset/scripts/run_muse_glimmer_stage3.py servers --gpus 4 \
  --model-path /models/Muse-Glimmer-30B \
  --draft-model-path /models/Muse-Glimmer-30B-assistant

# In another shell, after the servers start:
python dataset/scripts/run_muse_glimmer_stage3.py probe --gpus 4
python dataset/scripts/run_muse_glimmer_stage3.py cycle --gpus 4 \
  --run-id dataset_muse_glimmer_v1 --cycle-size 1000 --total 10000

# For Stage 3 only, using existing Stage 1/2 records:
python dataset/scripts/run_muse_glimmer_stage3.py generate --gpus 4 \
  --source-run-id dataset_v1 --run-id dataset_v1_muse_glimmer
```

`cycle` fills a shared target in order: 1,000 queries, then 1,000 responses,
then 1,000 GenUI records; the next cycle fills each stage to 2,000. Supply
`--cycle-size` and `--total` explicitly. The last cycle stops at the total even
when it is smaller than a full chunk. For just one 1,000-record cycle, set both
to 1,000. Rerun the same command to resume an interrupted cycle; it counts
existing records and refuses a run that contains another model. Stage 1 and 2
use 8,192-token output budgets by default, while Stage 3 uses 12,288. Change
them with `--query-output-tokens`, `--response-output-tokens`, and
`--output-tokens` if needed.

`cycle` and `generate` check every endpoint's served model, active DFlash
configuration, reasoning channel, and final answer before writing any records.
Stage 3-only `generate` reads Stage 1/2 data from the source run, resumes missing
records in the Muse output run, and processes all available Stage 2 responses by
default. Use
`--max-genui-total 10` for an initial ten-record run, then rerun without the
limit to continue. The Stage 3 prompt and schema remain the repo's A2UI
Express configuration.

The starting profile uses a 32K context, 12,288 completion tokens (including
reasoning), a 16K prompt cap, eight concurrent requests per server, BF16,
high reasoning, and a 16-token DFlash block. These are candidate settings,
not measured H100 throughput. If a single-GPU replica runs out of memory,
run both commands with `--tp 2` to use two GPUs per replica. For the fastest
setting on a particular host, compare valid Stage 3 records per minute and
truncation/OOM counts at `--tp 1` versus `--tp 2` and
`--requests-per-server 4`, `8`, and `16`; keep the best measured setting.

References: [Meta SGLang deployment](https://dev.meta.ai/docs/muse-glimmer/sglang),
[Meta prompting guide](https://dev.meta.ai/docs/muse-glimmer/prompting), and
[SGLang Muse Glimmer recipe](https://docs.sglang.io/cookbook/autoregressive/Meta/MuseGlimmer).

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







