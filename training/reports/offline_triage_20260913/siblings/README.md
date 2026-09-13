# Exact-source sibling alternatives: offline dry run

This analysis makes **no changes to training records or target IR**. It asks a
narrow question: when a record fails the existing warning/budget screens, is
there already another target for the **identical source response** that passes
those screens?

## Policy and result

A candidate sibling must be non-reserved, strictly valid, free of URL
preprocessing errors, within the reconstructed 4,096-token sequence and
2,048-token target budgets, and have none of these existing flags:
empty layout, layout-only output, mojibake, placeholder mismatch, missing action
label, low lexical recall, or missing numeric anchors.

| Measurement | Training | Validation | Combined |
|---|---:|---:|---:|
| Original rows | 150,292 | 910 | 151,202 |
| Rows passing these screens, before source deduplication | 65,452 | 378 | 65,830 |
| Distinct exact source responses among passing rows | 63,014 | 378 | 63,375 |
| Flagged rows with a passing exact-source sibling | 358 | 0 | 358 |
| Same-split sibling available | 357 | 0 | 357 |
| Only an other-split sibling available | 1 | 0 | 1 |
| Exactly one distinct passing semantic target among all siblings | 355 | 0 | 355 |
| Multiple distinct passing semantic targets among siblings | 3 | 0 | 3 |

These 358 flagged records represent 358 exact source responses. **They recover
zero additional source coverage**, because their screened siblings already
exist. After verifying a sibling, keep the existing good record and discard its
bad alternative; do not duplicate the good target to inflate training counts.

Passing automated warnings **does not establish semantic completeness**. Thus
358 is a reproducible *review-candidate* count, not 358 approved repairs. Source
matching does not justify choosing blindly among alternate UI descriptions.

Known benchmark rows remain excluded even if their other checks pass. The
reserved cohort is 27 train and one validation row; 20 reserved train rows would
otherwise have passed this screen. They must not be restored to training.

## Concrete coordinates

- Blank-layout train **111526** has screened sibling train **131024**. This is
  an example where retaining an existing target may avoid regeneration after
  content verification.
- Train **116647** has a screened sibling only at validation **817**. Do not
  transfer that target into training while continuing to report the same
  source as independent validation. Rebuild source-family-disjoint splits.
- Train **80176** has two distinct screened alternatives: **99283**, **123910**.
- Train **99199** has two distinct screened alternatives: **80089**, **123826**.
- Train **123905** has two distinct screened alternatives: **80171**, **99278**.

The complete coordinate mapping and overlapping reasons are in
[candidate_rows.csv](candidate_rows.csv); counts and policy are in
[summary.json](summary.json).

## Other deterministic repair limits

The completed full structural audit found all 151,202 targets already strict
valid and canonical, and the existing graph-repair layer proposed **zero**
changes. There is no measured backlog of schema-alias/container fixes here.
Empty or incomplete containers are a missing-content problem, not a syntax
problem, and cannot be filled without an authoritative existing target or
regeneration.

The URL preprocessor assigns placeholders using a shared registry keyed by
role and raw URL while processing response, graph, and asset metadata together.
It is unsafe to reconstruct unknown historical placeholder mappings by simply
matching numeric suffixes or order. Present literal URLs can be mapped
deterministically for a new representation, but absent URLs/assets cannot be
invented and existing target-only placeholders are not thereby verified.

The token limits above use the prior audit's reconstructed Gemma3 framing, not
a recovered actual E2B tokenizer/template. Any final dataset build must repeat
the budget check using its actual model configuration.

## Reproduction

From the repository root:

```powershell
python training/scripts/audits/full_data_sibling_repair_20260913.py
```

The script opens the completed synthesis SQLite index read-only. It writes only
this audit's summary and candidate CSV. Original source files, model code,
Goldens, and target labels are untouched. Python compilation and critical Ruff
checks passed.
