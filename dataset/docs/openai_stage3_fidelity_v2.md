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

The separate ten-sample ChatGPT pilot also quarantines sample 10,
`dataset_muse_glimmer_100k_r2:u_005711_01:8086`: it gives children's ages as
4 and 7, then calls the seven-year-old the youngest. The catalog pins the exact
source hash and the manual review hash. This is an inherited source defect,
not an instruction to rewrite its output or remove it from existing datasets.

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
It supports both sealed API Batch archives and fully collected ChatGPT sign-in
archives. ChatGPT archives require `generation.json`, `chatgpt_collection.json`,
and every completed request's sealed record; model, request, source, thread,
turn and status evidence remain mandatory. Uncertain/partial generations are
not rescored as successful outputs. Historical client fingerprints are recorded
as evidence, without loading a client or asking the current runtime to replay.
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
grammar or an explicit `Option 1: Name | field | field` record, contiguous
fields in one Card, complete ordered text and an unambiguous match.
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

The ten-sample follow-up adds narrowly certified layout transformations:

- Adjacent Text fragments can form a whole source sentence only under the
  same rendered parent occurrence. Approximation marks, blanks and all other
  non-whitespace characters must survive; duplicate occurrences remain counted.
- A plain pipe table header is credited only with a complete matching source
  run and all ordered headers, cells and rows. Arbitrary pipe expressions do
  not become tables.
- A source equation in a Formula can retain generic-content ownership only
  when its entire visible field group matches that source equation. Supported
  spelling normalization preserves words, units, operands, signs and structure;
  Indian numeric grouping is retained.
- Each exact heading occurrence has one owner, including for numeric counts.
  Missing, changed or duplicated headings remain failures.
- Unsupported Formula commands such as `\mathord` and `\sim`, and unbalanced
  groups, make pilot renderer evidence incomplete. They are not stripped.

These projections apply only under `pilot_fidelity_v2`. Existing entity
certificates retain their own occurrence indexing; uncertain overlaps,
paraphrases and unsupported layouts stay held. No score threshold changes,
general fuzzy matching, or post-hoc acceptance override were added.

For the completed ten-sample ChatGPT pilot, the read-only Windows command is:

```powershell
python dataset/scripts/rescore_openai_stage3_pilot.py `
  --pilot-dir C:/Users/anupk/Downloads/genui_chatgpt_sol_10_parallel4_20261101_r1 `
  --output-dir C:/Users/anupk/Downloads/genui_chatgpt_sol_10_parallel4_20261101_r1_rescore_fixes
```

Choose a fresh sibling output name if that directory already exists. This
produces reports only; generation and training admission remain separate.

The November 1 ten-sample offline check accepted 6/10 versus 1/10 in the sealed
collection. Samples 1, 2, 3, 4, 7 and 9 pass. Samples 5 and 6 remain held for
uncertified paraphrasing/structure changes; sample 8 contains unsupported Formula
commands; sample 10 is the source quarantine above. The original labels for the
same sources pass 4/10 under the current pilot policy. All 90 archive files,
including the 48 files in the original collection seal, stayed unchanged.
This corrects evaluator mismatches, not newly generated model quality. No
samples were regenerated or added to training, and no device test was run.

## Follow-up fixes for the 100-sample ChatGPT pilot

These fixes remain exclusive to `pilot_fidelity_v2`; default SFT, QAT, GRPO,
Muse and legacy-pilot scoring paths retain their existing behavior and gates.
The pilot supplement clarifies complete list-to-card extraction and citation
versus action rendering. Prompt improvements require a separately authorized
generation to measure; rescoring never repairs existing IR.

- Clock and drive-letter colons do not create labeled entity fields. Split
  property bundles such as Age/Sex must retain every field under their unique
  source-section Card. Named Option/date records retain ordered fields, labels,
  qualifiers and complete source text, with no reuse across Cards.
- Balanced prose emphasis and a source heading rendered as a Card title do
  not introduce extra claims. Code, blanks, math operators, literal pipes,
  source multiplicity and field association remain significant.
- Complete pipe tables certify their headers and every ordered source row,
  including optional Markdown borders and genuinely empty trailing cells.
  Table-only contracts additionally require exact cell/column ownership;
  changed comparisons, numbers, qualifiers and duplicate tables fail.
- Explicit source formulas compare against their renderer-supported spelling.
  The `Formula:` source marker may own the matching Formula title only when
  the whole expression is exact. Extra results, changed operands/operators,
  scripts, units and unsupported commands remain failures. Complete numbered
  equation tables must preserve each row identity and its full calculation.
- Labeled source citations accept bare or angle-wrapped HTTPS destinations.
  Labels, destination bytes, occurrence counts and source/action roles remain
  mandatory; standalone action buttons cannot satisfy source citations.

The read-only command for the collected 100-sample archive is:

```powershell
python dataset/scripts/rescore_openai_stage3_pilot.py `
  --pilot-dir C:/Users/anupk/Downloads/genui_chatgpt_sol_100_parallel4_fidelity_v2_20261101_r1 `
  --output-dir C:/Users/anupk/Downloads/genui_chatgpt_sol_100_parallel4_fidelity_v2_20261101_r1_rescore_fixes_r2
```

The output directory must be a new sibling. Reports record current and archived
decisions and fingerprints without modifying the sealed run or admitting data
to training. Existing fingerprint-bound resumes still require their original
code or explicit reassessment; these fixes do not bypass provenance checks.

The November 1 offline rescore of this 100-sample archive accepts 66/100,
compared with 44/100 at collection: 22 false rejections resolved, no previously
accepted sample newly rejected, and 34 still held. The original labels for the
same sources pass 45/100 under the updated policy. This is an evaluator
correction on unchanged outputs, not a newly measured generation improvement.

| Category | Samples | Archived accepted | Rescored accepted |
|---|---:|---:|---:|
| Entity lists | 30 | 12 | 18 |
| Comparisons | 20 | 10 | 14 |
| Long sections | 15 | 3 | 8 |
| Procedures | 10 | 4 | 6 |
| Numerical | 10 | 7 | 8 |
| References | 10 | 3 | 7 |
| Controls | 5 | 5 | 5 |

Remaining holds include citation/action-role errors, unsupported media/prose,
and content/numeric or unproven layout transformations. These are review flags,
not a claim that all 34 outputs lose facts. For example, standalone buttons for
source citations remain invalid, while some rearranged course/profile records
still lack a complete exact ownership certificate. No hold is waived.

Verification covers 688 evaluator/generation/GRPO regression tests and 250
SFT/QAT pipeline contract tests (seven skips), plus 100 byte-for-byte equivalent
default-policy assessments before and after the changes. All 628 archived files
remain unchanged. No GPU training, device evaluation or new generation was run.

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
