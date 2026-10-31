# OpenAI Stage 3: separate 100/1,000-source Batch pilot

Use **GPT-6.1 Sol (`gpt-6.1-sol`), medium reasoning** for the first pilot. It
supports Responses and Batch and provides a useful quality/cost starting point.
This is a recommendation, not a measured improvement over Muse: compare accepted
outputs on identical sources and manually review samples before expanding.

This standalone command relabels existing **training responses**. It does not
generate queries or new response facts, run training, change Muse configuration,
or replace any existing dataset. It supports a prepared directory containing
`train.jsonl` and `val.jsonl`, including the repository's verified v11 gzip bundle
without unpacking the full dataset. It is not a raw Stage 2 run-folder importer.

## Setup and offline preparation

Run from the repository root on the generation machine. No GPU or Muse server is
required. Use a dedicated Python 3.12 environment so other environments stay intact:

```bash
cd /group-volume/k.anup/code/A2UI
python3 -m venv /group-volume/k.anup/envs/a2ui-openai-batch
source /group-volume/k.anup/envs/a2ui-openai-batch/bin/activate
python -m pip install -r dataset/requirements-openai-batch.txt

INPUT=training/data/train/v11
PILOT=/group-volume/k.anup/working_dir/openai_stage3_sol_100_mix_r1

python -u dataset/scripts/openai_stage3_batch.py prepare \
  --input-dir "$INPUT" --output-dir "$PILOT" \
  --samples 100 --seed 123 \
  --selection-profile failure_mix_v1 \
  --model gpt-6.1-sol --reasoning-effort medium --reasoning-summary auto --max-output-tokens 8192 \
  --cache-ttl 12h

cat "$PILOT/estimate.json"
cat "$PILOT/selection_report.json"
```

`prepare` makes no generation/API calls and needs no API key. The tokenizer may
download its public encoding file on first use. It streams and verifies the
original splits, excludes validation and pinned Golden/Bixby inputs by identity
and normalized source, deduplicates sources, and selects reproducibly by seed.
It also applies the existing Stage 3 source-quality check before selecting a
source, including rejection of unbound action destinations. Such sources are
excluded, not rewritten or assigned invented URLs. The exclusion counters for
quality checks describe sampled candidates, not a full factual audit of v11.
The old teacher label is retained locally for comparison but is not in the
OpenAI request. A fresh output directory is required; failed preparation cannot
be submitted without its completed, hash-checked contract.

## Opt-in sample mix from existing training data

`--selection-profile failure_mix_v1` supports exactly **100 or 1,000** sources.
Start with `--samples 100` to measure real usage and review quality before scaling.
The CLI default remains 1,000; specify 100 explicitly. Omitting the profile
keeps the existing seeded random selector. This profile changes only source
selection in this standalone pilot, not Muse, training, prompts, caching, or the
canonical Stage 3 admission/fidelity checks.

| Primary source category | 100-source trial | 1,000-source pilot |
| --- | ---: | ---: |
| Repeated entities with separately extractable fields in lists | 30 | 300 |
| Comparison tables, including pipe-delimited sources | 20 | 200 |
| Long answers with multiple sections | 15 | 150 |
| Procedures, recipes, plans and checklists | 10 | 100 |
| Numerical values, units and qualifications | 10 | 100 |
| Citations and supplied URL references/actions | 10 | 100 |
| Short plain answers and uncertainty controls | 5 | 50 |

Features are detected from the original response after canonical reference
restoration, never from old teacher completions or benchmark answers. Category
recognition uses documented English/Markdown heuristics, not a semantic or factual
certification. Entity recognition reuses the existing bounded extraction-policy
source parser, rather than counting colons (which mistakes timestamps for fields).
Sources can match several categories; deterministic matching
assigns each selected source to only one primary bucket. Reassignment prevents
overlap from producing an avoidable shortage. No source is rewritten or truncated.

The reference target is **6 numbered-citation sources plus 4 URL-bearing sources**
for 100, or **60 plus 40** for 1,000. The citation heuristic ignores fenced/inline code, legal years, and
table/form row labels, and looks for citations at the end of a prose claim.
It is conservative and still requires manual review. `[SOURCE_URL_*]` placeholders
are not numbered citations; URL slots require restored URLs and unchanged source
checks. Citation destinations are never invented, fetched or borrowed from a
benchmark. Numeric citations without captured URLs remain literal source text.

If citations are scarce, eligible URL-bearing sources fill those reference slots.
If any primary bucket still lacks candidates, remaining eligible training sources
fill explicit `coverage_backfill` slots. `selection_report.json` records requested
and achieved categories, citation shortfall, URL substitutions, other category
shortfalls, backfill counts, and word-length diagnostics. It never labels those
substitutions as meeting the missing feature. Fewer than the requested number of distinct eligible
sources fails preparation. Existing-source citation scarcity is a coverage gap,
not permission to synthesize references or use Golden/Bixby data.

`detected_before_quality_checks` counts source features after holdout/dedup
exclusion, before sampled source-quality checks; it is **not** a count of all
eligible sources. Every selected row passes the existing source checks. Word
counts describe the selected length mix; a 50/50 length split is not enforced,
and word counts are not student-tokenizer context limits. Review the pilot's
length and category distribution before paying for generation.

Each `selected.jsonl` entry retains its source identity/hash, feature evidence,
primary bucket and any backfill reason. The selection report is hash-bound in
`contract.json`; changing it invalidates submission/collection. A new profile or
code version requires fresh preparation; do not reuse an older prepared pilot.

Both sizes scan and verify the original training/validation inputs; a smaller
pilot reduces paid requests, not necessarily preparation time. The 100-source
mix uses smaller deterministic candidate pools and is not guaranteed to be a
subset of the 1,000-source mix. Do not trim or edit a prepared request file: use
a fresh output directory and `--samples 100` so request counts, source bindings,
selection quotas and cost estimates are sealed consistently.

After explicit submission and collection, review `usage.json` for actual token
counts and token-based cost estimates, `report.json` for acceptance and failure
counts, and `assessments.jsonl`/`pilot_accepted.jsonl` against `selected.jsonl` for
source fidelity and list-to-card extraction. Inspect all 100 outputs when practical,
including rejects and truncations. Accepted-output fidelity is not certification
of source factuality, rendered UI quality or downstream training accuracy.
Use returned usage rather than the old 1,000-source estimate when deciding
whether to expand; cache hit rates and acceptance rates need not scale linearly.

Validation and pinned Golden/Bixby sources remain excluded by identity and
normalized original/restored source hash. Explicit evaluation-only records fail
closed. The broad failure categories guide selection; benchmark inputs, labels,
and paraphrases are not a candidate source pool. Only accepted regenerated labels
can pass the existing downstream gates; this profile does not certify model gains.

Offline v11 selection audit (2026-11-01, seed 123): verified all 200,214 training
and 4,089 validation rows; selected 1,000 distinct training sources meeting the
seven primary quotas above, with no general category backfill. The conservative
detector found **0 numbered-citation candidates**, leaving a **60-source citation
gap**; 60 additional eligible URL-bearing sources filled those reference slots.
This is not proof that every possible citation style is absent. Length diagnostics
were 223 sources below 250 words and 777 at or above 250 words, so the suggested
even length balance is not achieved. The audit performed selection only, without
request capture, API submission, label generation or training. Validation passed
233 regression tests plus 29 subtests, including source preservation, holdout
exclusion, reference binding, replay fidelity and report-tampering checks.

## Submit once, then collect later

Set `OPENAI_API_KEY` using your shell/secret manager. For an interactive Bash
session, this avoids putting the key in shell history:

```bash
read -rsp 'OpenAI API key: ' OPENAI_API_KEY; echo
export OPENAI_API_KEY

# This command incurs API usage. Inspect estimate.json first.
python -u dataset/scripts/openai_stage3_batch.py submit \
  --output-dir "$PILOT" --max-estimated-cost-usd 6

# Run separately whenever convenient; no GPU is occupied while OpenAI works.
python dataset/scripts/openai_stage3_batch.py status --output-dir "$PILOT"

# After completion (or a terminal partial/failed batch):
python -u dataset/scripts/openai_stage3_batch.py collect --output-dir "$PILOT"
```

The allowance is a **local estimated-cost check, not an OpenAI spending cap**.
The example uses a $6 allowance for the 100-source trial; check its actual
`estimate.json` first. For a separately prepared 1,000-source run, choose a new
allowance after reviewing that run's estimate (the earlier pilot used $60).
Account model access and queued-token limits must accommodate the pilot; the
estimate includes its full repeated instruction prompts. Batch has a 24-hour
completion window. No synchronous generation fallback or automatic paid repair
retries occur. Retrying `submit` after success reports the saved Batch ID.

If the network fails after Batch creation, the script refuses another paid
submission. Find the Batch ID in the OpenAI Batch dashboard, then bind it back:

```bash
python dataset/scripts/openai_stage3_batch.py status \
  --output-dir "$PILOT" --batch-id batch_YOUR_EXISTING_ID
```

Recovery checks the input file ID, endpoint, completion window, and contract
metadata. `collect` reuses downloaded results after interruption and is read-only
after successful collection. Do not delete submission journals to retry a job.
Keep the same code, prompt, schema and source-policy version until collection;
changes are rejected rather than mixed into the pilot.

## Shared prompt caching

New preparations use **explicit-only caching**. The canonical Stage 3 instructions
and extraction supplement are preserved verbatim in the first `developer`
message, with one `prompt_cache_breakpoint` on its `input_text` block. The
source-specific user message follows it with **no breakpoint**. This avoids
paying to cache changing source text, IDs or reference bindings. The original
reference masking/restoration and fidelity gates are unchanged.

`--cache-ttl 12h` is the default: the shared instructions can be reused across
asynchronous Batch scheduling gaps. Use `--cache-ttl 30m` when a shorter cache
lifetime and lower write fees suit your run. The 12-hour write rate is higher,
so it is not always cheaper; actual scheduling and reuse determine savings.
No cache warm-up request or extra paid API call is made.

The request sets `prompt_cache_options={"mode":"explicit","ttl":"12h"}`.
Sol handles routing automatically; no per-source/run cache key fragments the
shared prefix. Preparation verifies that every request has the identical
instruction prefix, records its hash and estimated token count, and reports
whether it meets the **1,024 visible-token minimum**. It never pads or truncates
the prompt to force caching. Exact-prefix matches, model/settings, routing and
lifetime affect reuse; cache hits are **not guaranteed**. `store:false` remains
set and is independent of prompt-cache lifetime.

Use a **new output directory** after this cache-policy change. For a batch
already prepared/submitted with an older version, keep that original checkout
for `status`/`collect`; do not edit its requests, contract or submission journal.
Cache settings are bound into the preparation contract and request hashes.

## Usage and price

As verified **2026-11-01**, Sol Batch short-context pricing is **$1.00 per million
input tokens and $5.00 per million output tokens**. Output includes billed
reasoning tokens, not just visible Express code. Batch is 50% below Standard
prices; it reduces dollars, not token usage.

Cached reads cost **$0.05/M**. Cache writes cost **$1.25/M for 30m** or
**$2.50/M for 12h**; these are total write prices, not surcharges on ordinary input.
Tokens after the explicit breakpoint cost the ordinary input rate without a
cache-write charge.

`estimate.json` measures the captured prompts using tiktoken and labels whether
the model tokenizer or the `o200k_base` proxy was used. It reports total input,
per-request maximum input, and output scenarios at 2,048, 4,096 and the configured
cap. The default pilot permits at most **8,192,000 output tokens** across 1,000
requests. Each scenario gives both a **no-cache-hit cost including prefix writes**
and an illustrative **one-prefix-write then all hits** cost. The latter is an
assumption, not a prediction. The budget assumes a prefix write on every request
at the selected TTL, adds 25% input-token margin and conservatively prices message
framing at the write rate. It does not assume any cached-read savings.
It excludes taxes/regional premiums and is not a guaranteed invoice maximum.

Re-estimating the 2026-10-31 offline v11 pilot's 1,000 captured prompts with this
request format on 2026-11-01 (seed 123, medium reasoning) measured **6,132,695
estimated input tokens**, including a shared **5,409-token prefix per request**,
using the `o200k_base` proxy. All 1,000 selected sources had passed the existing
source-quality gate. No API batch was submitted and no cache hits were measured.

| Output assumption (includes reasoning) | Total estimated tokens | 12h, no cache hits | 12h, one write then all hits |
| --- | ---: | ---: | ---: |
| 2,048 per source | 8.18 million | $24.49 | $11.25 |
| 4,096 per source | 10.23 million | $34.73 | $21.49 |
| Every request reaches the 8,192 cap | 14.32 million | $55.21 | $41.97 |

The conservative local budget estimate is **$58.83 with 12h**, or **$50.33 with
30m**; the example command allows $60. These scenarios are **not measured output
usage or a predicted acceptance rate**. Source/prompt changes change the estimate.

`usage.json` reports returned input/output/reasoning counts, cache reads, writes
by lifetime, ordinary input, token cache-hit fraction and **net input savings
including write charges**. Failed/rejected responses with usage are counted.
Reasoning and cache writes are not counted twice. Missing cache accounting makes
the cost unknown (`null`), rather than assuming free writes. Net savings can be
negative when cold writes outweigh reads. Missing usage/partial results remain
explicit; the OpenAI billing record is authoritative for charges.

Official references: [Sol model](https://developers.openai.com/api/docs/models/gpt-6.1-sol),
[Batch guide](https://developers.openai.com/api/docs/guides/batch),
[prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching),
[pricing](https://developers.openai.com/api/docs/pricing).

## Validation and outputs

The workflow captures requests from the **existing `run_stage3`** using the
canonical Express prompt plus the existing opt-in v11 extraction supplement.
On collection it replays returned text through that same runner, with repairs
disabled, then applies `admission_errors`, canonical reference restoration,
the existing strict v11 source/candidate guard, and training `prepare_row`.
Source facts, field ownership, numbers, qualifiers, actions, tables, references,
schema/wire validity and review holds retain their existing thresholds.
The evaluator's required parity/conformance files are checked during offline
preparation and bound to the pilot contract, so missing evaluator dependencies
are detected before submission. Use the full repository checkout.

The model sees only bound opaque references. Restoration first uses the Stage 3
reference map and canonical graph hash, then rebinds to the original training
source's tokens. There is no asset downloading, invented reference binding,
handwritten label repair, or original-target fallback on rejection.

Results are joined by `custom_id`, never output order. Duplicate/unknown IDs stop
collection. Refusals, incomplete/truncated outputs, API errors and missing results
cannot enter the accepted file. Expired/cancelled batches may contribute valid
partial results; they are never reported as a complete 1,000-label dataset.

| File | Purpose |
| --- | --- |
| `selected.jsonl` | Immutable original rows, split and source provenance; baseline labels for review |
| `requests.jsonl` | Exact Responses Batch requests |
| `estimate.json`, `contract.json` | Usage estimate and sealed preparation identity |
| `submission.json`, `remote_status.json` | Durable Batch association/status |
| `output_file_id.jsonl`, `error_file_id.jsonl` | Unmodified remote output/error evidence, when present |
| `pilot_accepted.jsonl` | Only candidates passing all gates; preserves original row IDs and sources |
| `assessments.jsonl`, `rejected.jsonl` | Per-source admission decisions and rejection reasons |
| `report.json`, `usage.json` | Acceptance/partial-result summary and reported API token usage |
| `reasoning_summaries.jsonl` | Optional returned reasoning summaries with source bindings and admission status; never training targets |
| `v-*/stage3/` | Canonical Stage 3 records, metrics and generation audit artifacts |

The accepted file is intentionally marked a **pilot subset**, not a complete
training dataset. Review 50–100 samples across lists/cards, tables, qualifiers,
actions/references, and long responses. Compare with the original labels on the
same sources. Later replace those training labels by original ID in a separately
reviewed dataset composition step; do not append duplicate source/target pairs
or incorporate Golden/Bixby examples. Source factuality, rendered/device quality
and downstream training gains are not certified by structural/fidelity checks.

## Keep reasoning summaries for later review

Add `--reasoning-summary auto` to **prepare** to request the most detailed
summary supported by the model. The default is `none`, which omits the API
summary option and preserves the existing request and training-target behavior.
OpenAI does not expose raw internal reasoning tokens as readable text; this
option saves **provider reasoning summaries**, not the full private reasoning
trace. See the [official reasoning summary documentation](https://developers.openai.com/api/docs/guides/reasoning#reasoning-summaries).

Prepare a fresh folder; existing prepared requests and contracts must not be
edited. Code is also fingerprinted, so use the original checkout to operate an
older pilot, or prepare a new one with the updated code. For this 1,000-source
pilot, the same seed and unchanged v11 inputs preserve the source selection:

```bash
python -u dataset/scripts/openai_stage3_batch.py prepare \
  --input-dir training/data/train/v11 \
  --output-dir /group-volume/k.anup/working_dir/openai_stage3_failure_mix_v1_summaries_r1 \
  --samples 1000 --seed 123 --selection-profile failure_mix_v1 \
  --model gpt-6.1-sol --reasoning-effort medium --reasoning-summary auto \
  --max-output-tokens 8192 --cache-ttl 12h
```

Preparation remains offline and does not submit a Batch. Inspect the new
`estimate.json` before explicit submission. Submit/status/collect use their
existing commands with the new output directory; no collection flag is needed.
No live summary-enabled API run is required by the offline tests.

When collection completes, `reasoning_summaries.jsonl` contains one audit row
per requested source, joined by `custom_id`, including missing, rejected, failed
and incomplete results. Each row records original/prepared/API IDs, returned
summary fragments and combined text, API status and usage, and the final
canonical admission decision. Availability is independent of admission: an
empty summary does not reject valid Express, and a summary cannot admit invalid
Express. Unknown or malformed summary parts are flagged and remain available in
the unchanged raw Batch evidence. No rationale is invented when none is returned.

The sidecar is written atomically and sealed into `collection.json`; interrupted
collection reuses verified downloads, and completed collection verifies the
summary hash. Summaries are excluded from `pilot_accepted.jsonl`, prompts,
fidelity scoring, and training targets. They can be reviewed separately later;
do not treat summaries from rejected results as approved training examples.
Summary availability depends on model/account support and may require OpenAI
organization verification. The script does not silently retry without summaries
or submit replacement requests on API errors. Actual usage remains in `usage.json`.
