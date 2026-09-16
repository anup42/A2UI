# Bixby50: frozen response-only test cohort

This committed artifact contains the 50 captured Bixby/Perplexity responses
`BXP-001` through `BXP-050`, collected on 9 September 2026. It is an
**evaluation-only, final-only holdout**, not training data or a checkpoint/
hyperparameter-selection benchmark. Cloning the repository supplies the test
inputs; the original local device logs are not required to run evaluation.

## What is scored

The response-to-IR model receives the exact captured response through the shared
production prompt and generates A2UI Express. The evaluation pipeline can report
generated-output validity, source-grounded quality/reward, and latency. **No
reference IR was captured or generated for these inputs**, so reference matching
is not applicable and must never be reported as zero, perfect, or inferred from
a fabricated target. Metrics do not establish that historical source facts are
correct or still current. Do not use original queries as model input: this tests
response-to-IR conversion, not question answering.

## Data guarantees and limitations

- All 50 IDs and source responses are unique and nonempty. Collection evidence
  records an exact query match and completed Perplexity response for each case.
- `response_text` preserves the delivery's `markdown_response` exactly. Original
  query, domain and minimal collection provenance are retained in metadata.
- `BXP-038` is the provider's actual plain-text refusal about unavailable
  verification tools. It remains unchanged as a robustness case; it is not a
  completed careers comparison.
- Numeric citation markers exist without captured destination URLs. Do not
  invent hyperlinks or provider facts to fill those gaps.
- There are no completions, reference graphs or synthetic target answers.
- Device serials, request IDs, local absolute paths, raw logs and screenshots
  are intentionally excluded from the committed rows.
- `benchmark_manifest.json` freezes approved IDs, exact UTF-8 response/query
  hashes, source-delivery digest and the output digest. Preparation must preserve
  these identities and evaluation-only flags. Source inputs must be excluded
  from training and validation, just as other held-out test sources are.

## Reproduction

The source was the successful final delivery, including BXP-009's reconciled
retry, rather than an earlier partial capture. The original local file is not
committed. If that exact delivery is available, reproduce into a **new** folder:

```sh
python training/scripts/create_bixby50.py \
  --source /path/to/run_50_delivery/responses.jsonl \
  --output-dir /path/to/new/bixby50_v1
```

The script refuses an existing destination and a delivery whose SHA-256 differs
from `5c8d92953589b346591c62f3dae240cefe22498ced23c5d50e5153b5ba9e9f71`.
It only copies and validates approved source fields; it does not call Stage 3,
generate targets, alter source text, or access device logs.

Artifact checks:

```sh
python -m pytest training/tests/test_bixby50_artifact.py -q
```
