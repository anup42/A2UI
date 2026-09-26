# Muse Stage 1/2/3 archive quality review

Reviewed 2026-09-26. **UI conversion is mostly faithful; upstream source quality
is not yet ready for bulk training admission.** Improvements below change future
generation. No supplied response or generated IR was rewritten or regenerated.

## Scope and integrity

The supplied `dataset_muse_glimmer_100k_r1_stage123.tar.gz` is an intact partial
snapshot: **5,000 queries, 5,000 responses and 259 Stage3 records**, not 100,000
completed examples. SHA-256:
`46d6a6345b367df4088a0f2caf375225ab6fb1553f77f9e22911f0e0add4922c`.
Only its three named JSONL members were extracted into a separate local audit
directory. Original archive/files were preserved. No duplicate IDs, exact
normalized query/response duplicates, missing joins, or Stage2-to-Stage3 source
text mismatches were found.

| Whole-archive check | Result |
| --- | ---: |
| Actual completed Stage3 outputs | 256 |
| Fresh strict Express decoding and standard-wire compilation | 256/256 pass |
| Source-rejected records without UI | 3 |
| Saved generation `record_status=accepted` | 255 |
| Saved conservative `training_acceptance.eligible=true` | 0/259 |
| Balanced URL destinations truncated by old reference processing | 14 in 6 UI records |
| Newly detected truncated source action links | 51 in 44 responses |
| Sources failing revised deterministic checks, including existing errors | 56/5,000 |

The remaining 4,944 sources are **review-needed**, not independently verified.
All 5,000 lack a reviewed source contract and independent prose-fact verification.
The stored representation-quality mean is 90.61/100 over 256 outputs (median
91.51); this is not factual accuracy, training eligibility, or a recomputed score.
Schema/wire validity also does not prove correct facts or good on-device layout.

## Manual review of 100 random records

Sampling: Python `random.Random(20260926).sample(linked_stage3_rows, 100)`, retaining
original file order before sampling. Population is the 259 archived Stage3 rows.
Three Astra reviewers read disjoint groups of 34, 34 and 32 complete
query/response/Express packets; canonical graphs were inspected for ambiguous
escaping, action and reference behavior. Cross-review reconciled classifications,
checked arithmetic and removed unsupported assertions about external facts.

| Source response verdict | Count | Meaning |
| --- | ---: | --- |
| Internally usable | 38 | No material internal defect found; not externally fact-certified |
| Needs review | 43 | Unsupported/ambiguous claims, minor defects or missing evidence requiring verification/correction |
| Reject current content | 19 | Demonstrable arithmetic, scheduling, hard-constraint or malformed-action defect; regenerate after source correction |

Separately, **93 UIs preserve their source**, **6 need regeneration** for actual
reference/presentation defects, and **1 has no generated UI** because its source
was rejected. There are **36 pairs** with both an internally usable source and a
faithful UI. These human verdicts do not override the existing training gate.
“Reject” does not mean the original question is irreparable or should be deleted.

Examples from the actual review:

- `u_001802_01`: twelve listed track durations total **52:50**, not the claimed
  approximately 58 minutes. Stage3 faithfully copies the incorrect source.
- `u_000781_01`: listed pages total **2,448**, while the source claims 2,424.
- `u_001289_01`: a 180-minute film window is filled by 181 minutes and finishes
  at 17:01 instead of 17:00.
- `u_000520_01`: January 1,200 to June 1,450 is **20.83%** growth, not 32%.
- `u_001804_01`: correct opaque tokens restore to Wikipedia links missing the
  final parenthesis. This is a pipeline bug, not Muse inventing a destination.
- `u_000007_01`: an incomplete Glassdoor link ending `...` produces a malformed
  live token `[URL_2...]`.
- `u_003848_01`: a media declaration becomes both an Icon and visible raw-URL
  Text. The source's useful narrative is otherwise retained.

Many source errors involve unsupported live prices, admissions/eligibility,
legal specifics or capabilities such as “create playlist” pointing only to a
search/homepage. These are verification gaps unless the supplied facts prove a
contradiction; this review does not invent authoritative replacement facts.

The all-review automatic gate is not a finding that every UI is bad. Examples
include chart *recommendations* being treated as required charts, citations in
Text being inferred as missing buttons, and exact-content scores penalizing
faithful restructuring. Those metric thresholds were **not relaxed**. Android
already parses Markdown headings/bold; cosmetic `#`/`**` observations were removed
as defects after checking the renderer. No device-render inspection was performed.

## Implemented prevention

1. **Muse-only Stage2 prompt.** Replaces contradictory generic live-data/table/
   action/media quotas for the exact registered local Muse model. It requires
   checking totals, durations, hard constraints, rankings and conclusions;
   exposes infeasibility and uncertainty; disallows invented live verification,
   fake cut lengths and incomplete links; keeps all requested records. Its
   documented Action syntax is tested against the actual parser.
2. **Optional media remains optional.** Muse does not retry simply to invent an
   icon. Declared invalid media still fails/retries, with explicit permission to
   omit unsupported media. Other models retain existing prompt/retry behavior.
3. **Prompt provenance/cache.** The effective Muse Stage2 template and selected
   request (including a successful retry) receive hashes and a version. Changing
   the template changes the cache key. Serial and batched calls use the same policy.
4. **Balanced URL preservation.** Shared boundary handling keeps parentheses in
   destinations such as `Anastasia_(1997_film)` while removing unmatched prose
   delimiters. Tests check actual restored action destinations, not just text
   round trips. Stage2 media extraction uses the same boundary handling.
5. **Hard source admission.** `source-quality-v1.1` rejects literal truncation
   markers in action URL hosts/paths and unresolved action tokens. Stage3
   rechecks deterministic failures using the original query/contract before a
   provider call, even if saved response quality is missing or stale. A generated
   response cannot declare its own trusted contract.
6. **Output reference admission.** Malformed/unbound recognized reference tokens
   in media destinations/openUrl actions fail the existing repair/regeneration
   gate. Inert quoted text, table data and Event context are not treated as live
   destinations. This is not a general URL existence/network verifier.
7. **Targeted Stage3 guidance.** Compiler-checked examples show a compact
   comparison table with complete record-labelled narrative notes, and an asset
   declaration rendered once as media without exposing transport text. Source
   facts remain unchanged; Stage3 must not silently repair faulty arithmetic.

The same registered Muse launch commands pick up these changes automatically;
no new training or generation flag is required. Sampling, reasoning, model,
training, holdouts and core Express grammar are unchanged. Shared URL/validation
fixes also protect other generators; non-Muse prompt selection is unchanged.

## Next run and limits

No generation endpoint was available on this machine, as confirmed by the user.
**Zero records were regenerated or declared repaired.** Code tests do not prove
the new prompt eliminates semantic errors. In particular, arbitrary prose math,
current facts and action capabilities remain unverified without suitable trusted
inputs/tools; model self-checking is not independent validation.

On the generation machine, with the existing Muse servers ready, start a fresh
small candidate run (match `--gpus` to that server setup):

```bash
python dataset/scripts/run_muse_glimmer_stage3.py cycle --gpus 4 \
  --run-id dataset_muse_quality_pilot_v2_20260926 \
  --total 100 --cycle-size 100
```

Use a new run ID to avoid mixing old/new sources; existing IDs are skipped on
resume and a new prompt does not retroactively fix them. For this archive's bad
source responses, regenerate **Stage2 then Stage3** on the generation machine.
For verified-good sources with only UI/reference defects, regenerate **Stage3**.
Never hand-patch IR/provenance or automatically promote the 36 human-reviewed
pairs. Check a fresh pilot's arithmetic, constraints, actions and rendered layout
before scaling, and compare identical fixed non-holdout inputs for a genuine
before/after experiment. Keep Golden35/Bixby50 out of teacher examples.

## Evidence files

- [summary.json](summary.json): counts, archive/member hashes, sample IDs and limits.
- [manual_review_100.jsonl](manual_review_100.jsonl): one annotated record per sample;
  adjudications preserve original reviewer notes and corrections.
- [reference_corruption.json](reference_corruption.json): exact affected source and
  saved destinations; diagnostics only, not patched labels.
- [source_destination_scan.json](source_destination_scan.json): all 56 deterministic
  source failures, including 44 newly detected sources.
- [lint_comparison.json](lint_comparison.json): existing touched-file lint baseline.

## Validation

The focused offline suite covers real Stage2/Stage3 paths with scripted providers,
cache/retry provenance, unchanged non-Muse routing, source-contract rechecks,
reference admission, prompt examples, launcher behavior and canonical-prompt drift.
No network generation, training or device deployment was launched.

```bash
python -m pytest dataset/tests/test_muse_stage2_prompt_quality.py \
  dataset/tests/test_muse_stage3_prompt_quality.py \
  dataset/tests/test_muse_reference_admission.py \
  dataset/tests/test_reference_boundaries.py \
  dataset/tests/test_stage2_asset_validation.py dataset/tests/test_source_quality.py \
  dataset/tests/test_generation_quality_regressions.py \
  dataset/tests/test_stage3_a2ui_express.py dataset/tests/test_muse_glimmer_stage3.py \
  dataset/tests/test_muse_latency_logging.py \
  dataset/tests/test_a2ui_express_prompt_generation.py \
  -q -p no:cacheprovider --basetemp .muse_review_tests
```

Final integrated result: **179 tests passed, 29 subtests passed**. The 113 warnings
are existing `datetime.utcnow` and `jsonschema.RefResolver` deprecations.
New files and the changed Stage3 prompt test pass Ruff; the wider legacy touched
files retain 76 pre-existing lint diagnostics with no additional code/message
categories. Python compilation and scoped whitespace checks pass.
