# Source quality contracts and generation audit safeguards

The Stage 1/2 source checks are CPU-only, deterministic and read-only with
respect to text. They add metadata when new records are written. They do not
invoke a verifier model, retry generation, fetch evidence, rewrite existing
datasets, or require media downloads. Unit tests use mock adapters only.

## Why there is a separate source check

An IR can preserve a wrong source perfectly. The September 2026 audit found
incorrect totals, weighted averages, loan comparisons and schedules in accepted
targets. Source correctness and IR fidelity must remain separate. A `Sources`
link does not prove a claim was verified; a button label does not prove that the
action performs its advertised task.

## Metadata and compatibility

New Stage 2 records include:

```json
{
  "query_text": "the original query",
  "source_quality": {
    "version": "source-quality-v1",
    "status": "checks_passed",
    "training_eligibility": "eligible",
    "verification_scope": "declared_contract_only",
    "prose_fact_verification": "not_performed",
    "contract_reviewed": true,
    "modality": "answer",
    "original_query_sha256": "...",
    "contract_sha256": "...",
    "checks": [],
    "findings": [],
    "asset_downloads_required_for_training": false
  }
}
```

The example above illustrates fields; an actual empty check list is always
`needs_review`, never `checks_passed`.

| Status | Eligibility hint | Meaning |
|---|---|---|
| `checks_passed` | `eligible` | A supplied reviewed contract has nonempty checks and every supported check passed; no unresolved review finding |
| `needs_review` | `review` | Legacy/no contract, incomplete provenance, unbound added actions, interactive task without a capability packet, near-duplicate candidate, or no checks |
| `failed` | `exclude` | A declared calculation/constraint/fact/action failed, or the contract/output has a deterministic defect |

**Eligible means eligible under the declared checks. It does not certify all
free-form prose or externally changing facts.** Consumers must retain
`verification_scope` and `prose_fact_verification`. Domain review remains
necessary for medical/legal/financial/procedural claims not represented by
the packet. Old datasets remain readable; absence of metadata is not a pass.
Stage 2 preserves failed examples with failure metadata for diagnosis instead
of silently deleting or rewriting them. Downstream training admission decides
how to handle `review` records and must exclude known failures.

## Supply a reviewed packet with the query

The optional `source_contract` is input data in `queries.jsonl`. It is supplied
by a fixture author or verified-data preparation step, not trusted because a
model emits the word `reviewed`. Stage 1 does not promote model-generated
self-attestations into a reviewed packet. Existing manually supplied query
fields are preserved through Stage 2. The `verification.reference` should
identify an immutable fixture/review artifact or source snapshot.

```json
{
  "query_id": "fixture_bill_01",
  "intent": "calculation",
  "query_text": "Subtotal is $342.50, service is $51.375 and tax is $27.40. Give the total to two decimal places.",
  "source_contract": {
    "version": 1,
    "verification": {
      "status": "reviewed",
      "reference": "fixtures/billing-v1#fixture_bill_01"
    },
    "modality": "answer",
    "calculations": [{
      "id": "total",
      "op": "sum",
      "values": ["342.50", "51.375", "27.40"],
      "result_label": "Total",
      "tolerance": "0.005"
    }]
  }
}
```

The generated response must include `Total: $421.28` or a two-cell table row
`| Total | $421.28 |`. Every matching result binding is checked; a contradictory
summary alongside a correct table fails. A missing binding fails rather than
being called correct. Arbitrary prose, formulas and LaTeX are not evaluated.
Labels should be unique and simple. Units, taxation rules, selected inputs and
tolerance are the fixture author's responsibility; use required facts/text for
explicit units and assumptions. A tolerance must reflect the requested
rounding, not a convenient acceptance range.

### Supported calculations

- `sum`, `mean`, `product`: `values` array.
- `weighted_mean`: equal-length `values` and nonnegative `weights` arrays.
- `difference`, `ratio`, `percentage`: numeric `a`, `b`; percentage is `a/b*100`.
- `loan_payment`, `loan_interest`: `amount`, **periodic** `period_rate`, integer
  `periods`. Interest is unrounded payment times count minus principal.
- `annuity_future_value`: periodic contribution `amount`, `period_rate`,
  `periods`, `payment_timing: "start" | "end"`.

All use bounded decimal arithmetic. They reject unknown operations, invalid
inputs and zero denominators. No expression strings are executed. Array and
period bounds prevent unexpectedly expensive checks.

### Facts, roles and constraints

```json
{
  "facts": [{
    "id": "affected_host",
    "value": "192.168.1.45",
    "label": "Affected IP",
    "provenance": {"kind": "query"}
  }],
  "constraints": [
    {"id": "budget", "kind": "number_range", "label": "Cost", "min": 0, "max": 2000},
    {"id": "outdoor", "kind": "forbidden_text", "value": "Indoor/Hybrid"},
    {"id": "color", "kind": "required_text", "value": "red"},
    {"id": "day", "kind": "schedule", "window_start": "09:00", "window_end": "18:00"}
  ]
}
```

Fact `provenance.kind=query` requires exact presence in the query, plus exact
preservation in the response by default. Optional `label` binds the value to
its role (`Affected IP: ...`), catching replacement with `Source IP`. Reference
facts use `kind=reference` and a `reference`; they still require a reviewed
packet. Hypothetical facts use `kind=hypothetical` and must be clearly understood
as fixture data, not live account state. These checks cannot find all invented
facts outside the packet.

`number_range` validates explicit numeric result bindings. `required_text` and
`forbidden_text` are literal checks, not semantic classifiers. `schedule`
checks nonoverlapping `HH:MM-HH:MM` intervals inside one same-day window. It
does not infer unlabeled dates, overnight intervals, travel time or natural
language clock expressions. For multi-day schedules, use separate fixtures or
additional checks; do not claim this one-day check proves a full itinerary.

### Actions and modality

```json
{
  "modality": "interactive_tool",
  "actions": [{
    "label": "Copy Text",
    "destination": "action://copy/scanned-text",
    "mode": "capability",
    "capability_id": "copy_text",
    "parameters": {"text": "fixture text"},
    "required": true
  }]
}
```

The response must contain an actual `Action: [Button: Copy Text]
action://copy/scanned-text` declaration. Mentioning that button in prose does
not satisfy the contract. Destinations and labels must match; capabilities
require IDs and parameter dictionaries. The renderer/runtime must independently
support the capability; source validation does not execute it or prove runtime
support. `mode=navigation` means a link; `mode=mock` permits explicit synthetic
actions without pretending they execute. Placeholder hosts such as
`example.com` are **not blanket-rejected**. Unbound added destinations request
review; deterministic URL syntax defects fail. No HTTP request is made.

Modalities are `answer`, `document`, `dashboard`, `interactive_tool`. Without a
packet, the source check records a conservative lexical inference and requests
review for interactive tasks without supported/mock action declarations.

## Assets: readiness is separate from training eligibility

Existing `asset_stats` fields remain for compatibility. Additional fields:

- `verification_state`: `not_applicable`, `skipped_offline`, `resolved`,
  `unverified`, or `processing_error`.
- `visual_ready`: based on resolution state, not a visual screenshot test.
- `training_assets_required: false`.
- `training_policy: symbolic_placeholders_allowed`.
- `verification_scope: asset_resolution_only_not_source_quality`.

Actual images/icons need not exist or download for placeholder-based training.
Skipping assets cannot exclude a training row. An offline `asset_quality_ok`
continues to mean the old permissive policy, while `skipped_offline` explicitly
states what was not verified. The source-quality module never checks local
asset existence. An asset-processing exception also remains independent of
source eligibility.

## Scenario coverage and split safeguards

Stage 1 attaches `query_quality` and `scenario_family_id` to **every** append
path, including fallbacks. It records the real prompt template SHA-256/path and
version rather than the old hardcoded `query_gen_v1` label. The generation
prompt includes up to six recent scenarios for the current intent, two recent
other-intent scenarios and current intent/modality counts. This adds bounded
context, not another model call.

A word-trigram index flags Jaccard similarity at least0.65 across intents and
groups matching candidates into a scenario family for conservative split
isolation. Numbers remain in the signature so actual changed constraints are
visible. Short queries have their own nonempty signatures. Near duplicates
are **review candidates**, not automatic rejection or proven equivalence.
This does not replace embedding-based diversity review or dataset-wide
balancing. Existing query files are indexed in memory but never rewritten.

## CPU-only verification

```powershell
python -m unittest discover -s dataset/tests -p test_source_quality.py -v
python -m unittest discover -s dataset/tests -p test_stage2_asset_validation.py -v
```

Tests cover the audited bill/GPA/loan/schedule errors, source-role preservation,
mock/capability action binding, malformed/legacy contracts, duplicate-family
metadata, and mocked Stage1/Stage2 integration. They do not invoke real
generation, training, inference or Stage3.

## Throughput notes for an 8-H100 launch plan

These are source-code observations and starting values to validate on the GPU
host, not measured speedups:

1. `LOCAL_VLLM_BATCH_PARALLELISM` defaults to at most4 HTTP workers. Raising
   `query_batch_size` alone cannot exceed that client concurrency.
2. Stage1's multi-intent `generate_batch` path requires
   `stage1_intent_batch_size > 1` **and** `stage1_intent_cycle_size <= 0`.
   The older Gemma launcher exports cycle size20, which selects the sequential
   cyclic path. A throughput profile should select cycle0 and32 intent slots.
3. Start Stage1 with4 queries/request and query token budget2048–4096; Stage2
   with one response/query and16–32 concurrent query requests, then increase to
   32–64 only if serving metrics show spare capacity and no queue/memory issue.
   These are request counts, not model tensor parallel settings.
4. Set both `rate_limit_qps=0` and `call_sleep_seconds=0` for an unconstrained
   private local endpoint. The latter still sleeps even when QPS is zero.
5. Stage2 token budget2048–4096 is a starting point for non-reasoning concise
   outputs; reasoning mode consumes additional completion tokens, so measure
   completion reasons and increase the cap for long samples rather than
   accepting truncated answers. Do not derive a hard cap from whitespace word
   counts. Keep one candidate and bounded retries; avoid duplicate per-query
   fallback after an already failed batch unless specifically needed.
6. Placeholder training should use offline mode, no Commons enrichment, no
   asset retries and minimal asset workers. It gains nothing from downloading
   the same icons concurrently. Leave real-media rendering validation as a
   separate selected-sample job.
7. The source checks run locally when each completed response is finalized;
   they do not add a model verification pass, network fetch or global barrier.
   Query similarity uses an inverted trigram index, with at most100 candidate
   comparisons per appended query. Stage2 batches currently return after all
   requests in the group complete, so long-tail completions still delay the
   next group. Streaming scheduling would require a separate implementation.
8. Benchmark sustained accepted examples/hour and input/output tokens/sec,
   retries, truncation rate, queue length and KV-cache pressure on the actual
   host. GPU memory size/topology, serving dtype and reasoning policy determine
   whether replication or tensor parallelism is appropriate; no topology or
   speed assertion is established by these CPU tests.
