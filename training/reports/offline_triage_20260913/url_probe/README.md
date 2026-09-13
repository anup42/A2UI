# Offline URL normalization probe

This is a **read-only dry run**, not a repaired training dataset. It tests one narrow class of mechanical repair: replacing explicit URLs/assets with typed placeholders while proving that the exact original source and target graph can be restored from the generated map. It does **not** restore clipped words, omitted facts, empty layouts, encoding corruption, or missing historical URL maps.

## Admission before this probe

Candidates come from the completed full-corpus audit's `triage` table. They must be strict-valid, nonreserved, free of recorded URL parser exceptions, fit the original reconstructed Gemma3 4,096-sequence / 2,048-target-token gates, and have none of the selected empty-layout, layout-only, mojibake, placeholder mismatch, missing action, low lexical recall, or missing numeric-anchor warnings.

Those checks do not certify that every retained answer is semantically complete. The parent policy adds split protection, confirmed-failure exclusions and semantic-pair deduplication. Counts in this subreport are therefore **probe counts before those additional exclusions**, not the final dataset size.

## Status meanings

| Status | Meaning |
|---|---|
| `already_symbolic_closed` | Target placeholders are declared in the source and neither side has explicit references. Keep the existing representation; no original destination is inferred. |
| `no_url_transform_needed` | No existing typed placeholders or explicit references need conversion. |
| `verified_reversible_url_normalization` | Neither original side has existing typed placeholders; production preprocessing makes a map; exact source and graph restoration succeeds; normalized output passes strict catalog/schema/renderer validation; every normalized target placeholder appears in normalized source. |
| `ambiguous_not_counted` | Mixed literal/symbolic representation, missing target bindings, failed restoration, or another conversion failure. Do not guess a map or silently renumber placeholders. |

For example, a normalizer can assign a source URL `[SOURCE_URL_1]` but assign that same URL `[ACTION_URL_1]` in a button. Both can restore to the same URL, yet the target token is absent from the model's input. **Reversibility alone is insufficient.** This probe excludes that pair until a consistent source/target binding contract is proven.

Existing symbolic examples may be usable as source-to-IR supervision, but original destinations cannot be recovered from token names. This probe makes no claim that they have complete historical asset provenance or can independently reproduce original device rendering.

## Complete URL-probe counts

| URL status | Training | Validation | Combined |
|---|---:|---:|---:|
| Already symbolic and closed | 18,197 | 93 | 18,290 |
| No URL transform needed | 52 | 0 | 52 |
| Verified reversible URL normalization, before post-normalization length gate | 24,559 | 166 | 24,725 |
| Ambiguous, not counted as repair | 22,644 | 119 | 22,763 |
| **All probed candidates** | **65,452** | **378** | **65,830** |

The ambiguous group contains 55 mixed symbolic/literal training pairs and 22,708 pairs whose normalized target uses placeholders absent from normalized input (22,589 train + 119 validation). There were no restoration or strict-validation failures among the placeholder-free conversions; the exclusion is an input/target binding ambiguity, not evidence that all those original labels were wrong.

**All 24,725 verified normalization pairs also pass the complete post-normalization length check:** 24,559 training and 166 validation pairs; zero length rejections. Maximum normalized target / reconstructed full-sequence lengths are 1,549 / 3,978 for training and 1,372 / 3,579 for validation. These retain the tokenizer/template qualifications below.

## Validation and reproducibility

- Original input records are read by recorded physical UTF-16 byte offsets and checked against the prior source/target SHA-256 hashes.
- The target graph is decoded with the production Express codec. The generated URL map must restore the source string and canonical target graph by exact equality.
- The normalized target is emitted and checked with `serialize_checked`, including strict graph/renderer checks and a semantic-preserving emitter roundtrip.
- The full-corpus audit's accelerated wire validator is used only in this analysis process, with unchanged compatible schema assertions and no defaults/formats. [Validator parity](validator.json) records agreement on 67 real Goldens and 67 deliberately invalid copies. Production source files are unchanged.
- Every repair candidate is re-created in the length probe and must match the exact normalized source/target hashes from the structural probe.
- A bounded reference-engine check on 32 real normalized training targets also passed the **unmodified** production validator/serializer with identical canonical output; the per-row length CSV marks those checks. This is bounded parity evidence, not a claim that all 24,725 targets used both validation engines.
- Lengths use the same pinned local Gemma3 vocabulary, shared prompt fingerprint and explicitly reconstructed Gemma3 frame as the earlier full-corpus audit. **They are not E2B token counts or proof of the missing deployed 270M chat template.** Every measured pair must pass the final reconstructed 4,096/2,048 gates to enter the parent repair category.
- No transformed source, target, URL map, model weights or training dataset is written. Outputs contain statuses, hashes, lengths and diagnostics only.

The completion markers are [URL summary](summary.json) and [post-normalization length summary](length_summary.json). Each contains the corresponding per-row CSV SHA and completed row count. Large CSVs live under ignored `training/outputs/audits/offline_triage_20260913/url_probe/` and are keyed by original split and one-based line.

Reproduce from repository root, with the completed prior audit indices and the original data available:

```powershell
python training/scripts/audits/offline_url_probe_20260913.py --workers 4
python training/scripts/audits/offline_url_length_probe_20260913.py --workers 4
python -m pytest training/tests/test_offline_url_probe_audit.py -q
```

The commands refuse to overwrite their CSV outputs; use separate `--output` and `--report` directories for another run. Parallel CSV arrival order is not guaranteed, so compare keyed row content/counts rather than expecting identical CSV byte order across runs.
