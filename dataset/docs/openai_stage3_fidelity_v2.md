# Explicit OpenAI Stage 3 fidelity_v2 pilot

Select `--pilot-profile fidelity_v2` on `dataset/scripts/openai_stage3_batch.py prepare`
to opt in. Omission or `--pilot-profile legacy` retains the legacy prompt and
collection behavior. The profile is independent of `--selection-profile` and
`--reasoning-summary`. Preparation is offline; submission remains a separate
explicit paid command. No existing sealed batch should be edited or resubmitted.

The v2 adapter receives source and asset fields directly from Stage 3 before
prompt interpolation. It JSON-serializes `source_response_text` and
`asset_mapping`; it never searches for delimiters or policy headings in source
text. Asset instructions live in the shared developer prefix. Source text that
contains those same headings or JSON examples is preserved as source data.
The prefix is identical across sources with and without asset declarations.

Before selection and capture, v2 requires every reserved source reference token
to have an explicit valid donor. Existing donor URLs and Windows/local paths
retain their bytes. This is an identity check only: no asset is fetched, and
remote accessibility or local file existence is not asserted. The pilot-only
hold catalog contains the reviewed source hashes for zero-based audit indices
33, 37, 38 and 84 of the November 1 sealed 100-sample audit. Renaming a held source does not evade
the hold; changing content under its held identity requires review. The shared
v11 source holds and the source records themselves remain unchanged.

The static pilot supplement reinforces entity field ownership, domain-appropriate
cards and compact feature matrices, unique Card headings, and exact Formula
semantics with adjacent Text for long explanations. It prohibits invented facts
and silent source repairs. Preparation's local evaluator probe, canonical Stage 3
v5.4 scoring, its fidelity audit and the final collection guard all use the
selected profile. Stage 3 records the explicit v2 flag in generation provenance.
Passing does not certify source factuality, remote assets or device rendering.

Prepared contracts without `pilot_profile` continue to mean legacy. Existing
code/prompt/schema fingerprints remain mandatory: changed code refuses replay
or submission rather than silently upgrading an old contract. Use its original
checkout for a sealed legacy contract; read-only rescore is separate. New v2
contracts bind the profile, policy module, prompt supplement and hold catalog.
Standard SFT/QAT settings and scoring defaults are unchanged. Historical
GRPO/DPO evidence that binds the scorer or dataset code still requires its
original checkout or an explicit reassessment; this change does not bypass
those provenance checks.


## Offline review of an already collected pilot

From the repository root, use a fresh sibling output directory:

```bash
python dataset/scripts/rescore_openai_stage3_pilot.py \
  --pilot-dir /path/to/sealed-100-sample-pilot \
  --output-dir /path/to/sealed-100-sample-pilot-rescore-v2
```

This command uses no API key and cannot submit, replay or regenerate requests.
It verifies the archive seals and every declared hash, checks source/request
identities, binds each canonical candidate to its exact paid response and
reference-restored semantics, and verifies that the archive stayed unchanged.
It records the original and current code fingerprints separately. Ordinary
submission/collection still refuses code drift; rescore does not relax that
contract.

`summary.json` compares archived decisions, a fresh legacy assessment, v2
assessment and original labels assessed with v2 on the same sources. Category
counts and rejection reasons are included. `assessments.jsonl` provides the
per-source evidence; `rescore.json` binds report hashes and before/after archive
hashes. These are diagnostic reports, not training-ready data. No target is
rewritten and no accepted sample is merged into training.

V2 retains all original thresholds and adds checks for visible Alert and Formula
prose, exact action/source role and destination ownership, source reference
completeness and reviewed source holds. Literal multiline splitting consumes
whole ordered source units; matched headings keep their dedicated ownership.
Card field grouping requires the existing explicit named-entity/labeled-field
grammar, contiguous fields in one Card, exact text and an unambiguous match.
Explicit Action lines support both bare and angle-wrapped destinations. Source
links require their own labeled source occurrence; sharing a URL does not let a
button satisfy a citation, or let one occurrence satisfy multiple requirements.
Source-card header suppression requires proof that native link interception
actually applies. No-link fallback cards retain their visible headers;
uncertified URL/table extraction routes remain incomplete and held for review.
It does not infer unlabeled free-prose fields or erase semantic pipe operators.
Uncertain or unsupported representations remain held for review. Formula
normalization is formatting-only, with mathematical symbols and value changes
still checked. Android unit tests cover the supported Formula rendering changes;
these tests do not replace device rendering or model/training evaluation.

## Future preparation (offline only)

When a new pilot is explicitly authorized, prepare it in a new directory using
both independent profiles:

```bash
python dataset/scripts/openai_stage3_batch.py prepare \
  --input-dir /path/to/existing-train-val-data \
  --output-dir /path/to/new-100-sample-pilot \
  --samples 100 \
  --selection-profile failure_mix_v1 \
  --pilot-profile fidelity_v2
```

Preparation excludes validation and pinned Golden/Bixby data as before. Review
its selection and cost report before any separately authorized submission.
The prompt changes need a new authorized generation to measure their effect;
offline rescoring cannot remove policy leakage or repair existing outputs.
