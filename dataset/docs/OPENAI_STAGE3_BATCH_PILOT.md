# OpenAI Stage 3: separate 1,000-source Batch pilot

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
PILOT=/group-volume/k.anup/working_dir/openai_stage3_sol_1000_r1

python -u dataset/scripts/openai_stage3_batch.py prepare \
  --input-dir "$INPUT" --output-dir "$PILOT" \
  --samples 1000 --seed 123 \
  --model gpt-6.1-sol --reasoning-effort medium --max-output-tokens 8192

cat "$PILOT/estimate.json"
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

## Submit once, then collect later

Set `OPENAI_API_KEY` using your shell/secret manager. For an interactive Bash
session, this avoids putting the key in shell history:

```bash
read -rsp 'OpenAI API key: ' OPENAI_API_KEY; echo
export OPENAI_API_KEY

# This command incurs API usage. Inspect estimate.json first.
python -u dataset/scripts/openai_stage3_batch.py submit \
  --output-dir "$PILOT" --max-estimated-cost-usd 60

# Run separately whenever convenient; no GPU is occupied while OpenAI works.
python dataset/scripts/openai_stage3_batch.py status --output-dir "$PILOT"

# After completion (or a terminal partial/failed batch):
python -u dataset/scripts/openai_stage3_batch.py collect --output-dir "$PILOT"
```

The allowance is a **local estimated-cost check, not an OpenAI spending cap**.
Account model access and queued-token limits must accommodate the pilot; the
estimate includes its full repeated system prompts. Batch has a 24-hour
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

## Usage and price

As verified **2026-10-31**, Sol Batch short-context pricing is **$1.00 per million
input tokens and $5.00 per million output tokens**. Output includes billed
reasoning tokens, not just visible Express code. Batch is 50% below Standard
prices; it reduces dollars, not token usage.

`estimate.json` measures the captured prompts using tiktoken and labels whether
the model tokenizer or the `o200k_base` proxy was used. It reports total input,
per-request maximum input, and output scenarios at 2,048, 4,096 and the configured
cap. The default pilot permits at most **8,192,000 output tokens** across 1,000
requests. Input is charged for every request; no cached-input discount is assumed.
The budget estimate adds 25% input-token margin and a 25% cache-write allowance.
It excludes taxes/regional premiums and is not a guaranteed invoice maximum.

An offline preparation against the repository's v11 bundle on 2026-10-31
(1,000 sources, seed 123, medium reasoning) measured **6,132,695 estimated input
tokens** using the `o200k_base` proxy. All 1,000 sources passed the existing
source-quality gate before requests were prepared. No API batch was submitted.

| Output assumption (includes reasoning) | Total estimated tokens | Batch cost estimate |
| --- | ---: | ---: |
| 2,048 per source | 8.18 million | $16.37 |
| 4,096 per source | 10.23 million | $26.61 |
| Every request reaches the 8,192 cap | 14.32 million | $47.09 |

The local budget estimate with input/cache-write allowances was **$50.54**;
the example command allows $60. These scenarios are **not measured output usage
or a predicted acceptance rate**. Source/prompt changes will change the estimate.

For example, **10 million input + 3 million output tokens is about $25** at the
uncached Batch rates. Use the actual `estimate.json` for your data instead of
assuming this example. `usage.json` reports actual returned input/output/cached/
reasoning token counts, including rejected responses, and a price-snapshot cost
estimate. Reasoning is not added twice. Missing usage and partial results are
explicit; the OpenAI billing record remains authoritative for charges.

Official references: [Sol model](https://developers.openai.com/api/docs/models/gpt-6.1-sol),
[Batch guide](https://developers.openai.com/api/docs/guides/batch),
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
| `v-*/stage3/` | Canonical Stage 3 records, metrics and generation audit artifacts |

The accepted file is intentionally marked a **pilot subset**, not a complete
training dataset. Review 50–100 samples across lists/cards, tables, qualifiers,
actions/references, and long responses. Compare with the original labels on the
same sources. Later replace those training labels by original ID in a separately
reviewed dataset composition step; do not append duplicate source/target pairs
or incorporate Golden/Bixby examples. Source factuality, rendered/device quality
and downstream training gains are not certified by structural/fidelity checks.
