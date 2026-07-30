# Gemini 3.1 Flash-Lite — GenUI metric v5 review

- Requested display name: Gemini 3.6 Flash Lite
- Executed Vertex model ID: `gemini-3.1-flash-lite`
- Reason: no official Vertex `gemini-3.6-flash-lite` model was found; the repository and current Vertex documentation identify `gemini-3.1-flash-lite`.
- Samples: 10 responses and 10 FlatSpecs on the existing fixed 10-query benchmark.
- Metric: GenUI Representation Quality `5.0.0`
- Metric fingerprint: `eae1ab47a6441a8753fbb2ed7ff70ead505f80cbfeb3d7da12324a0b3d330c04`

## Scores

| UI ID | Intent | v4 | v5 | v5 reward | Active caps |
|---|---|---:|---:|---:|---|
| u_self_001_01 | weather | 75.000 | 75.000 | 0.500 | missing_action |
| u_self_002_01 | travel | 40.000 | 40.000 | -0.200 | reachability_below_90, strict_format, partial_action |
| u_self_003_01 | booking | 85.192 | 83.177 | 0.664 | strict_format |
| u_self_004_01 | shopping | 75.000 | 75.000 | 0.500 | missing_action |
| u_self_005_01 | entertainment | 75.000 | 75.000 | 0.500 | missing_action |
| u_self_006_01 | travel | 72.168 | 65.106 | 0.302 | strict_format, partial_action |
| u_self_007_01 | calculation | 98.085 | 94.848 | 0.897 | strict_format |
| u_self_008_01 | writing | 84.856 | 86.532 | 0.731 | strict_format |
| u_self_009_01 | tech_support | 93.424 | 87.483 | 0.750 | strict_format, partial_action |
| u_self_010_01 | planning | 97.838 | 95.000 | 0.900 | strict_format |

Production raw-path v5 distribution:

- Mean: `77.7146`
- Median: `79.0884`
- Population SD: `15.4881`
- Minimum: `40.0000`
- Maximum: `95.0000`

Canonical semantic-path v5 distribution:

- Mean: `79.5561`
- Median: `80.5134`
- All 10 canonical FlatSpecs pass the strict schema.

## Integrity checks

- Production-valid candidates: 10/10
- Current metric fingerprint: 10/10
- Finite bounded scores and rewards: 10/10
- Effective atomic weights sum to 1: 10/10
- Maximum effective atomic weight is within 0.10: 10/10
- Exact stored-score identity reuse: 10/10
- Native Android render attempts: 0/10

Seven raw model completions omitted required `children` fields on one or more
elements. Production canonicalization added the empty arrays, so these remain
production-valid but correctly activate the bounded `strict_format` policy.

Three candidates omitted all required source actions, three achieved only
partial action matching, and one candidate left four recommendation child
elements unreachable. These explain the active semantic/integrity caps.

Stage 2 retained all responses. Three external image downloads associated with
one travel response were unresolved (two HTTP 404 responses and one HTTP 429);
the URLs and failure diagnostics remain in the run log.

These 10 rows are a non-gating engineering audit. No weights or thresholds were
calibrated from them.
