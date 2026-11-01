# GenUI generation studio

GenUI adds a **separate ChatGPT-authenticated Stage 3 route** and a local dashboard. It uses the official Codex app-server, with `clientInfo.name=genui` and `clientInfo.title=GenUI`. OpenAI's hosted sign-in page keeps its official Codex/OpenAI branding; this is not a newly registered OAuth application.

This route consumes the account's **Codex allowance**, subject to its model availability and limits. It does not authenticate the OpenAI Batch API, promise free/unlimited generation, or replace API-key billing. The existing OpenAI Batch script, Muse generation, SFT, QAT and evaluation entrypoints keep their defaults.

## Setup and prepare

Use the existing dataset environment and install the official [Codex CLI](https://learn.chatgpt.com/docs/cli). GenUI selects the newest version of `codex` found on PATH, so an old installation cannot shadow the current desktop runtime. Use `--codex-executable /absolute/path/to/codex` on `dashboard`, `login`, `account`, or `run` to pin a particular installation. No new Python or frontend dependency is required beyond the existing pilot environment.

The account's model catalog depends on that runtime. On this machine, version 0.133 returned GPT-5 models, while the installed 0.166.0-alpha.5.1 runtime returned GPT-6.1 Sol, GPT-6 Astra, GPT-6.1 Terra and other current account models. A tiny authenticated GPT-6.1 Sol response was verified on the newer runtime. Existing runs still require the Codex version recorded when their generation began; pin that version when resuming them.

From the repository root, sign in and inspect the account's available model IDs:

```bash
codex --version
python dataset/scripts/genui_stage3.py login
python dataset/scripts/genui_stage3.py account
```

For a remote/headless generation machine, use `login --device`. Complete the official device flow on your own browser; enable device authentication in your account if required. `login --no-browser` prints the normal login link instead. A successful login does **not** start generation.

Prepare a **fresh** directory using existing training data. This is offline:

```bash
python dataset/scripts/genui_stage3.py prepare \
  --input-dir training/data/train/v11 \
  --output-dir /path/to/genui_stage3_100_r1 \
  --samples 100 \
  --selection-profile failure_mix_v1 \
  --model gpt-6.1-sol \
  --reasoning-effort medium \
  --reasoning-summary auto \
  --workers 2
```

Use an exact model ID printed by `account` if this account does not offer `gpt-6.1-sol`. Model and reasoning effort are checked before generation; there is no fallback or silent rerouting. A model change requires a fresh preparation. The CLI also supports `--seed` (default 123) and `--selection-profile random` for other pilot sizes. `failure_mix_v1` supports 100 or 1,000 sources; the default here is 100.

The new route explicitly selects `fidelity_v2`. It reuses the existing source selection, Golden/Bixby/validation exclusion, reference preflight, immutable source/request bindings, canonical Stage 3 prompt, reference restoration, and strict training-admission checks. Prompt capture and local validation do not call a model. Preparation must pass before generation can start.

## Dashboard

```bash
python dataset/scripts/genui_stage3.py dashboard \
  --output-dir /path/to/genui_stage3_100_r1 \
  --port 8767 --open
```

Open `http://127.0.0.1:8767`. Click **Check sign-in**, or **Sign in with ChatGPT** and then follow the OpenAI sign-in link. **Start generation** explicitly starts the prepared run with its selected worker count and automatically validates the completed run. Closing the tab does not stop generation; stopping the server interrupts it and may leave uncertain attempts.

The header provides a visible account link. The **New generation run** panel loads available models and reasoning efforts from the signed-in account; **Refresh models** queries the catalog again. It also exposes reasoning summaries and, under **Data and execution settings**, sample count, selection profile, seed, training-data folder, a fresh output folder, per-source timeout and **Parallel workers** (1–8, default 2). Supported effort levels such as `max` and `ultra` appear only for models that advertise them. Unsupported models or effort combinations fail before preparation.

Click **Prepare new run** to run local selection, prompt capture and validation in the new folder. Preparation does not make model requests. After it succeeds, the dashboard switches to the prepared run; click **Start generation** separately when ready. Changing settings never edits the current contract or existing archive. The timeout is stored in the new contract and restored when reopened; CLI `run --timeout` can explicitly override the wait limit. API-only sampling/output-cap/cache settings are not exposed for this Codex route.

The dashboard shows:

- Prepared/generated counts, stage, elapsed time, failures and uncertain/skipped attempts.
- Strict acceptance and held records, per-category coverage, citation-gap reporting, and rejection reasons.
- Source/output inspection with category, decision and text search. All source/output text is rendered as text, never executable HTML.
- Input, output, cached-input and reasoning tokens **when reported**; reasoning tokens are already included in output.
- Active/configured workers, the initial cache-warming phase, reported cache reuse percentage and cache-write tokens when available. Missing or partial usage is identified explicitly.
- Codex sign-in, plan and reported allowance windows. ChatGPT usage is not converted into an API dollar estimate.

The server binds only to `127.0.0.1`, checks Host/Origin and a session token for actions, and serves only the three dashboard assets and fixed evidence endpoints. For a remote machine, keep it on loopback and use an SSH tunnel, for example `ssh -L 8767:127.0.0.1:8767 user@host`. Do not expose it as a public web service.

To inspect an existing API pilot without any submission, polling or modification:

```bash
python dataset/scripts/genui_stage3.py dashboard \
  --output-dir /path/to/existing_openai_pilot \
  --rescore-dir /path/to/its_separate_offline_rescore \
  --port 8767 --open
```

`--rescore-dir` is optional and verifies the reassessment's contract and summary hash. Its acceptance count is shown separately from the original decisions. API Batch progress comes from locally collected status files; refresh those through the existing Batch `status`/`collect` commands when needed. The dashboard has no API Batch submit action.

## CLI generation and recovery

The CLI provides the same explicitly triggered generation and offline collection without a dashboard:

```bash
python dataset/scripts/genui_stage3.py run --output-dir /path/to/genui_stage3_100_r1
python dataset/scripts/genui_stage3.py collect --output-dir /path/to/genui_stage3_100_r1
```

Do not run a CLI worker and dashboard worker concurrently. The output lock rejects concurrent writers. Resuming `run` reuses completed records and processes only requests that were never attempted. The actual Codex CLI version, model, prompt, code, source and contract are bound to the run.

### Parallel execution and caching

New CLI/dashboard preparations default to **two workers**, with a maximum of eight. Use `--workers 1` for sequential generation. Contracts without a worker setting retain a default of one. CLI `run --workers 4` explicitly overrides the prepared count for that invocation, and the chosen count is recorded in `generation.json` under `execution_history`; model, prompt and source bindings remain fixed.

```bash
# Uses the prepared model/data and records the explicit concurrency override.
python dataset/scripts/genui_stage3.py run \
  --output-dir /path/to/genui_stage3_100_r1 --workers 4
python dataset/scripts/genui_stage3.py collect --output-dir /path/to/genui_stage3_100_r1
```

The first **real pending source** completes before parallel dispatch begins, giving the shared prefix a chance to enter the provider cache. No synthetic warm-up or repeated generation is added. Each invocation, including a resume after an idle period, warms with its first still-unattempted source. Remaining requests share one app-server, workspace, instruction prefix and model/reasoning settings. Each source gets an independent thread and event queue, so prompts, output and token accounting cannot be mixed between workers. Previous samples are never added to another sample's conversation history.

Caching remains provider-managed, including cache routing across independent threads. GenUI preserves the stable prefix and reports observed hits; it does not force a shared cross-thread cache key, promise a hit rate, set an unsupported TTL, or infer API discounts from subscription usage. The first-source warm-up is best effort. `usage.json` includes `reported_cache_hit_rate`, `cache_write_input_tokens`, `records_without_cache_write_usage`, and `records_with_partial_usage`. Cache percentage uses reported input only; missing/partial accounting means actual consumption can be higher. These are measurement fields, not quality gates.

Only the configured number of requests may be in flight. A reported exhausted allowance prevents new dispatch. A timeout, uncertain result, provider failure, or contract error stops new dispatch; already-issued requests drain and save their results. Uncertain calls are never automatically repeated. More workers may improve throughput, but shared account limits and model latency still apply. Parallel regressions use offline providers and the real preparation/replay path; no live multi-source throughput improvement is claimed.

Prepare a fresh run after updating generation code: existing prepared runs deliberately pin their source/prompt/code fingerprints and cannot silently adopt this implementation. Existing sealed archives remain read-only.

An interruption or provider error may consume allowance even if no response reached the client. GenUI stops dispatch, saves any already-running results, and does not retry. Inspect `generation.json` and `chatgpt_records/`, then explicitly skip an uncertain attempt if you want to continue:

```bash
python dataset/scripts/genui_stage3.py run \
  --output-dir /path/to/genui_stage3_100_r1 --skip-uncertain
python dataset/scripts/genui_stage3.py collect --output-dir /path/to/genui_stage3_100_r1
```

Skipping preserves its original uncertain record and never regenerates that source. A malformed or fidelity-rejected final output is also never automatically regenerated. Collection refuses incomplete/uncertain runs and rechecks every available final output through canonical Stage 3 and the strict guard. Once collected, evidence hashes are sealed and repeat collection verifies them.

Outputs are `pilot_accepted.jsonl`, `rejected.jsonl`, `assessments.jsonl`, `report.json`, `usage.json`, canonical Stage 3 artifacts, per-attempt records, and `chatgpt_collection.json`. **Accepted pilot records are not automatically merged into training**. A review and existing dataset-composition procedure are still required.

## Authentication and measurement boundaries

- Credentials are owned by Codex's managed credential store under a dedicated `~/.genui/codex` home (or `--auth-home` outside the repository and separate from the run directory). GenUI does not read/copy OAuth tokens, export them to API Batch, or use an API-key fallback. Existing normal Codex configuration and credentials are not changed.
- The app-server runs in an empty temporary workspace, with read-only sandboxing, no approval requests, disabled shell/web/apps/plugins/hooks/memory/subagents, and an explicit conversion-only policy. On Codex 0.166+, a verified named permission profile restricts filesystem reads to that workspace and disables command network access. Older runtimes retain their existing restricted-read request format. Account-managed plugin caches do not prevent reconnecting, but their tools and skills stay disabled. Unexpected tool activity, local instruction files, model rerouting, compaction, ambiguous final responses and incomplete turns fail closed.
- API `max_output_tokens`, explicit cache TTL and Batch pricing do **not** apply to this transport. Existing request files are canonical prompt templates; their API-only fields are not forwarded to Codex. `--timeout` (30–3,600 seconds, default 600) limits how long GenUI waits per sample, not the provider's billed tokens. There is also a local response-storage size limit.
- GenUI starts at most one Codex turn per selected source and never automatically retries that turn or repairs a rejected sample. Codex itself owns network-level connection recovery; this integration keeps its built-in OpenAI provider unchanged. Reported usage and the account allowance, rather than the number of attempted sources, determine actual consumption.
- `--reasoning-summary auto` requests a concise summary; availability is model-dependent. Returned summaries are stored with the per-attempt record. Raw/private reasoning content is never written by GenUI. `none` disables summary requests.
- Reported token counts may be missing after a disconnect or unsupported event. Missing usage is explicitly counted, never advertised as zero usage. The account's own usage dashboard remains authoritative for allowance/billing.
- Offline/mock tests cover replay, rejection, resume, accounting, model/runtime selection, preparation and dashboard security. The tiny signed-in connection test establishes that model generation works; it does not establish pilot quality. No new pilot/batch is submitted by installing or opening this dashboard.

References: [Codex authentication](https://learn.chatgpt.com/docs/auth), [official app-server API](https://learn.chatgpt.com/docs/app-server), [prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching).
