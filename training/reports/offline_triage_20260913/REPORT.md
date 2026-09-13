# No-Stage-3 data triage plan and exact policy counts

## Recommendation

Build a separate, conservative candidate dataset from the existing answers. Keep answers with no detected warning, permit only demonstrated lossless representation conversions, and quarantine everything unresolved. Do not try to reconstruct missing paragraphs, rows, or actions by editing IR.

**No clipped or missing-content label has a proved offline repair in this analysis.** The nonzero repair category below is exclusively reversible URL/reference normalization, not recovery of omitted content.

This is a completed full-archive classification dry run, not a training-ready export. No Stage 3, teacher/model inference, training, source-file edits, Golden edits, or production-pipeline changes were performed. The actual analyzed files are `C:/Users/anupk/Downloads/training_data/train.jsonl` and `val.jsonl`; their full SHA-256 hashes were rechecked against the prior audit.

## Exact totals under this conservative policy

Every original row appears in exactly one category, after known Golden reservation, split protection, quality/reference checks and deduplication.

| Category | Original train | Original validation | Total |
|---|---:|---:|---:|
| **KEEP — provisional, existing target unchanged** | **18,249** | **93** | **18,342** |
| **REPAIR — verified reversible URL normalization candidate** | **24,559** | **158** | **24,717** |
| **REJECT / QUARANTINE — exclude from this run** | **107,484** | **659** | **108,143** |
| **All original rows** | **150,292** | **910** | **151,202** |
| **Potentially usable after import/repair and final gates** | **42,808** | **251** | **43,059** |

The 43,059 candidate records cover **41,704 distinct exact source responses**: 41,453 training sources and 251 validation sources. Multiple different layouts for the same input remain together in one split; they are not automatically bad duplicates. No admitted exact source/semantic-target duplicate or identical effective source/target text pair remains under the tested rules.

These are **exact policy decisions, not exact counts of factually perfect or permanently irreparable labels**. Passing automatic checks does not establish that every required fact is visibly rendered. Reject means retain the original in a traceable exclusion queue, not delete it.

## What each category means

### KEEP

Keep the existing target content unchanged. It passed the completed strict IR checks, selected token-budget screen, known-family isolation and all selected warning gates.

- 18,290 combined rows already use consistent symbolic reference tokens; every target reference is present in the input.
- 52 have no references requiring URL conversion.
- Existing symbolic records lack historical destination maps. They are provisional symbolic response-to-IR training candidates, **not evidence of working real-world links or reproducible device rendering**.
- The initial placeholder gate is intentionally stricter than target-subset closure: it rejects a source/target placeholder-set mismatch, including potentially harmless unused input references.

### REPAIR

The 24,717 admitted candidates contain explicit references that can be normalized without inventing information. The dry run proved:

1. Neither original input nor original output already contained typed placeholders, avoiding collisions with missing historical maps.
2. Production URL preprocessing created an explicit restoration map from the literal references.
3. Restoring that map reproduced the **exact original input string and original output graph**.
4. The normalized output passed strict graph/catalog/schema/renderer-contract checks.
5. Every normalized target placeholder occurred in the normalized input.
6. The exact transformed pair, bound by both hashes, passed the reconstructed 4,096-sequence / 2,048-target-token screen.

The preliminary URL probe found 24,725 such rows. All passed post-conversion lengths; **eight validation rows were subsequently excluded for training-family overlap**, leaving 24,717 in the final repair category. Additionally, 32 real normalized targets matched the unmodified production validator/emitter. No normalized training records or maps were exported.

All 151,202 original targets were already canonical and strict-valid, so **required schema/graph repair = 0**. No verified mojibake/content restoration is counted.

### REJECT / QUARANTINE

The following are first-match reasons: they are mutually exclusive and sum exactly to 108,143. A row can have additional overlapping warnings in its audit record.

| First exclusion reason | Train | Validation | Total |
|---|---:|---:|---:|
| Unresolved content/encoding/placeholder/action/number warnings | 82,859 | 513 | 83,372 |
| URL binding not admissible under current normalization contract | 22,644 | 109 | 22,753 |
| Outside the selected original token-budget profile | 1,943 | 6 | 1,949 |
| Validation family also occurs in original training | 0 | 29 | 29 |
| Reserved accepted/excluded Golden family | 27 | 1 | 28 |
| Confirmed content defect, not already excluded above | 7 | 1 | 8 |
| URL preprocessing exception | 4 | 0 | 4 |
| **Total** | **107,484** | **659** | **108,143** |

The confirmed-defect list contains nine original coordinates. Validation line 209 is counted under split overlap first, hence eight in the dedicated defect row. Examples include clipped/incomplete letters and film schedules, empty interview answers, three layout-only targets, and the corrupted El Nino example. Their evidence is in the [full semantic review](../full_data_audit_20260913/quality/quality_review.md).

The 83,372 warning exclusions are **not 83,372 proved bad answers**. For example, accents can resemble a tracked mojibake marker, numbers can be reformatted, actions can be paraphrased, and source references can legitimately be omitted. This first-run policy quarantines uncertainty instead of guessing. The numeric gate means at least half the anchors missing when the source has at least three; the lexical gate is distinct-word recall below 50%.

Similarly, a raw URL can become `SOURCE_URL_1` in input but `ACTION_URL_1` in output. Even if both map back to the same URL, the output token is not declared in the normalized input. These rows are withheld under this contract, not declared semantically wrong. A separately tested source-consistent reference-normalization change could recover additional rows without Stage 3. That unimplemented possibility is not included in the repair count.

## Handling a bad answer when another answer already exists

The exact-source sibling audit found 358 flagged training rows with another initially warning-free answer for the same input. After the final reference/split checks, **170 still have an eligible existing training sibling**; the other 188 do not have a finally eligible donor under this policy.

Keep the eligible existing answer and exclude the defective alternative. Do not copy it over the bad row and count that as an additional recovered example: it adds **zero new source coverage**. Do not copy a validation target into training while continuing to call that validation example held out. Details: [sibling audit](siblings/README.md).

## Implementation plan — no Stage 3 required

1. **Freeze originals and create a new versioned artifact.** Keep the source hashes, original split/physical line, original target hash, every exclusion reason and policy version. Never overwrite the archive.
2. **Perform universal packaging separately from content repair.** Decode UTF-16LE/BOM correctly and write UTF-8/LF. Extract the final real user response and final assistant target, not the earlier few-shot pair. Add the required `response_text`/`completion` aliases, archive-derived hash IDs and source-family lineage. Explicitly mark historical query IDs, generator provenance and missing maps as unknown; a generated archive ID is not recovered provenance. Normalize the instruction scaffold to the pinned production prompt while retaining the old scaffold fingerprint.
3. **Copy KEEP candidates without semantic edits.** For REPAIR candidates, rerun exactly the tested reversible normalization, save its actual map and before/after hashes, and revalidate. Do not use global character substitutions, guessed token aliases, joined clipped words, invented facts, generic action replacements or truncated targets.
4. **Apply exclusions and protect splits.** Reserve accepted and excluded Golden source families. Keep known training families on the training side and exclude the 29 overlapping validation rows. Deduplicate admitted original source/semantic-target pairs and identical post-conversion source/target text pairs. Preserve distinct layout alternatives together and consider source-balanced sampling so repeated sources do not dominate.
5. **Review retained content before release.** Inspect a stratified training sample across documents, tables, comparisons, schedules, images, actions and long inputs. Review the 251 validation candidates independently, especially complete record counts, visible state bindings, dates/amounts and action destinations. A valid program with facts hidden in unused state must fail this review. Reject newly confirmed failures and publish revised counts rather than claiming this screen proves correctness.
6. **Run the actual model-specific preparation gate.** Use the exact E2B or 270M tokenizer and authoritative chat template, shared production prompt, strict validator and input/output bindings. Fail overflow explicitly; never slice labels to fit. Keep inference generation at 2,048 new tokens. Existing symbolic examples must have an explicitly supported symbolic-input contract; if the importer requires historical maps, resolve that contract before export rather than bypassing the guard.
7. **Keep Golden32 and Golden35 frozen and evaluation-only.** No copying, augmentation or source-family leakage from either benchmark. The current 251 validation candidates are the isolated remainder of the existing split, not a newly sampled representative validation set. Rebuilding a larger development split is a separate operation and changes these train/validation counts. Models previously trained on the contaminated archive cannot be presented as clean unseen Golden35 baselines.
8. **Revisit quarantine later using evidence.** Recover authoritative source/target maps or uncorrupted source records where available, and test each reversible conversion. Where paragraphs, rows or actions are genuinely missing and no verified existing alternative exists, leave that label excluded until regeneration is available. No augmentation is required or assumed for this first offline pass.

## Limits that affect these totals

- The 4,096/2,048 screen uses the pinned local Gemma3 vocabulary and **reconstructed** current prompt/chat framing. E2B was not tokenized here, and the archived 270M tokenizer has no authoritative chat template. Final training-machine counts can change.
- Original length screening precedes URL normalization. Some excluded long examples could shrink enough after a separately tested conversion or fit a complete-example longer-context profile. The 1,949 exclusions are not intrinsically defective data.
- Family isolation covers transitive exact/whitespace-normalized/production-URL-masked source hashes plus the inspected near-duplicate family. It does not prove that all paraphrases have been found.
- No model training, generation, device rendering or Golden score improvement was measured. Dataset quality and benchmark improvement must not be inferred from these counts alone.

## Evidence and reproduction

- [Machine-readable final totals and input/evidence hashes](summary.json)
- [Versioned policy](policy.json)
- [URL proof and post-conversion lengths](url_probe/README.md)
- [Existing-answer alternatives](siblings/README.md)
- [Prior complete archive audit](../full_data_audit_20260913/REPORT.md)
- Local full-row classification: `training/outputs/audits/offline_triage_20260913/classified_rows.csv` (metadata only, ignored; original split/line and hashes identify every decision).

Verification completed: an independent recount matched all 151,202 row decisions, category totals and first-reason totals; accepted original semantic pairs and effective text pairs were unique; known accepted train/validation/Golden family intersections were empty. Original source hashes, row bindings and normalized hashes matched. All **29 focused audit tests** passed, critical Ruff checks passed, and every local report link resolved. These checks validate the audit and its stated rules, not model behavior or complete semantic correctness.

New scripts are under `training/scripts/audits/`. With the completed prior audit indices, source files and audit dependencies present, the sequence is:

```powershell
python training/scripts/audits/full_data_sibling_repair_20260913.py
python training/scripts/audits/offline_url_probe_20260913.py --workers 4
python training/scripts/audits/offline_url_length_probe_20260913.py --workers 4
python training/scripts/audits/full_data_offline_triage_20260913.py
python -m pytest training/tests/test_full_data_offline_triage_audit.py training/tests/test_offline_url_probe_audit.py -q
```

The URL and aggregation scripts refuse to overwrite existing CSV/completion outputs; use new `--output` and `--report` destinations for another run and keep all subreports under the same new report root. The sibling script uses `--report <new-root>/siblings`. Parallel probe CSV order can differ between runs; compare keyed content and counts, not CSV byte order. This is an audit workflow, **not a command that prepares data or starts training**.
