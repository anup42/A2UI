# Full supplied-data inventory and leakage audit

This is a complete, streaming audit of `C:/Users/anupk/Downloads/training_data/train.jsonl` and `val.jsonl`, not a sample. The originally named `Downloads/training/_data` directory did not exist. Source files were not changed. DSL validity and real-tokenizer lengths are reported by the companion audits, not by this inventory.

## Main conclusions

- **The files cannot be handed directly to the current UTF-8 JSONL reader:** both are UTF-16 little-endian with a `FF FE` byte-order mark. Decode and re-export into a new, checksummed UTF-8 dataset; preserve the originals.
- **The source files contain benchmark contamination:** 26 of the 35 accepted Golden35 source responses occur in training, the excluded Golden32 case `q_012053` occurs in training, and Golden32 case `q_019461` occurs in validation. Reserve these sources before any train/validation split, including failed/replaced cases. This establishes contamination of the supplied files, not by itself what any particular checkpoint consumed.
- **The current train/validation split is not independent:** 28 exact source responses occur in both splits, affecting 28/910 validation rows (3.08%). They have different target strings, which is why exact whole-row deduplication misses them.
- **Provenance is absent from the exported records:** every record has only a `messages` key. No query IDs, response IDs, original run paths, teacher/model details, sample quality metrics, or preprocessing history survive. Full source matching against 211 current repository Stage 2/3 files recovers only 26 training source occurrences, all belonging to Golden35. It cannot establish original target-generation provenance for the rest.
- **There are many reference-grounding and encoding warning flags.** These are review flags, not automatic proofs of invalidity; combine them with strict DSL validation and semantic/content checks before deciding what to retain.

## Census

| Property | Training | Validation |
|---|---:|---:|
| Physical JSONL records | 150,292 | 910 |
| File bytes | 1,970,021,858 | 11,850,436 |
| Valid JSON objects and expected final task/answer envelope | 150,292 | 910 |
| Unique exact source responses | 146,065 | 909 |
| Unique exact target strings | 150,277 | 910 |
| Unique whole rows | 150,292 | 910 |
| Same-source groups with multiple target strings | 4,210 | 1 |
| Rows in those multiple-target groups | 8,437 | 2 |
| Source-response character median / p99 / maximum | 2,187 / 4,417 / 9,732 | 2,171 / 3,853 / 4,597 |
| Target character median / p99 / maximum | 3,283 / 5,071 / 8,727 | 3,268 / 5,085 / 6,557 |

Combined: 151,202 records, 146,946 unique exact sources, 151,186 unique raw target strings, and no duplicate whole rows. A repeated source can legitimately have several layouts; it is not inherently an incorrect target. Keep all variants in the same split and retain only semantically verified alternatives. A source-group weighting cap avoids amplifying heavily represented responses.

All records share the same five-turn `system,user,assistant,user,assistant` scaffold. The first user/assistant pair is the fixed travel-checklist in-context example; it is **not** the row's actual source/target. The audited source is the response after `Create A2UI Express v1 GenUI IR for this response:` in the final user turn. All source and target hashes cover their original, decoded, untrimmed content. Whitespace-folded and production URL-masked source hashes are also indexed.

## Golden and cross-split evidence

| Match | Split | Occurrences | Evidence |
|---|---|---:|---|
| Golden35 accepted sources | train | 26 | The corresponding records are clustered within lines 73,347–73,379; see `golden_overlap.json` for the exact case-to-line mapping. |
| Golden32 excluded `q_012053` | train | 1 | Line 131,726 |
| Golden32 accepted `q_019461` | val | 1 | Line 85 |
| Train/validation exact source overlap | both | 28 in each | Every source and its line positions are in `cross_split_sources.json`. |
| Train/validation exact target overlap | both | 1 in each | Whole-row contents differ. |

No additional accepted/excluded matches were found by the implemented exact, whitespace-folded, and production URL-masked hash comparisons. This does **not** establish that all paraphrases or source variants are independent; semantic near-duplicate checks are a separate audit. Query IDs recovered from repository files must remain qualified by run path because IDs are reused across runs.

## Grounding and encoding surface flags

| Flag | Training | Validation |
|---|---:|---:|
| Target uses placeholder tokens absent from its source | 42,808 (28.48%) | 288 (31.65%) |
| Source placeholder tokens are absent from target | 47,435 (31.56%) | 323 (35.49%) |
| Both directions of placeholder-set mismatch | 42,102 | 283 |
| Source contains a tested mojibake marker | 20,196 (13.44%) | 118 (12.97%) |
| Target contains a tested mojibake marker | 39,036 (25.97%) | 236 (25.93%) |

Placeholder patterns include image, icon, action, source, media and generic URL/asset tokens. Missing placeholders can reflect deliberate omission or remapping; extras can reflect unresolved invented references or a legitimate renderer asset. The comparison alone cannot distinguish these cases. Review source-target binding and the original URL map before remapping any token. The exported rows contain no URL map.

Mojibake patterns tested were `ΓÇ`, `â€`, `Ã`, and U+FFFD. `Ã` can occur legitimately in some languages. Decode/export repair must distinguish the UTF-16 container encoding from already-corrupted characters inside valid JSON strings. Do not run a blind global replacement over targets; recover authoritative source text and regenerate affected Stage 3 targets when semantics or glyphs are ambiguous.

Four otherwise valid training message records make the **production URL preprocessing function raise an exception**: lines 107,449; 110,047; 127,855; 138,115. Two have an invalid bracketed `Your-Public-IP` URL; two raise `Invalid IPv6 URL`. The audit retains their raw and normalized hashes, marks `url_preprocessing_error`, and leaves the masked hash NULL. The training importer needs an explicit quarantine path for such errors rather than aborting or silently dropping the record.

## Recommended order

1. Preserve these two original files and their SHA-256 hashes. Produce a new UTF-8 candidate export plus an explicit manifest and line-level lineage; do not overwrite these inputs.
2. Recover authoritative IDs and generator history where possible. For externally supplied sources without recoverable IDs, use an explicitly documented, content-addressed source identity and source-file/line provenance contract; do not claim it is an original Stage 2 ID.
3. Exclude all accepted and excluded Golden source groups, then rebuild train/validation by transitive source identity and normalized/raw/URL-masked response groups. Perform this before target filtering or selecting among variants.
4. Strict-validate complete targets, full graph reachability, wire-schema compatibility, and source/placeholder binding. Quarantine exceptions with file, physical line, source/target hashes and reasons.
5. Review multiple-target source groups using semantic equivalence and factual coverage, not string equality. Keep correct alternate layouts only within one split; choose a canonical high-quality target when alternatives lose or invent content.
6. Regenerate semantically damaged or encoding-ambiguous targets from authoritative responses through Stage 3. Avoid hand-editing or applying broad DSL string repairs.
7. Recompute actual tokenizer lengths under the shared production prompt, including few-shot supervision rules. Never truncate a target to satisfy the budget.
8. Only after quality and leakage cleanup, measure component/intent deficits against both benchmark suites and collect independent examples for deficient categories. Do not generate paraphrases of Golden responses for training.

## Reproduction and evidence

Run `python training/scripts/audits/full_data_inventory_20260913.py` from the repository root, choosing a new `--index` path if an audit index already exists. `--summarize-existing` recalculates summaries from a completed row index; it is not a replacement for reindexing modified source files.

- `files.json`: SHA-256, byte size, encoding, complete record/schema/role census, per-marker counts.
- `duplicates.json`: within/across split source, target, row, and scaffold uniqueness counts.
- `structure.json`: character distributions and full-corpus surface-quality flag counts.
- `golden_overlap.json`: accepted/excluded benchmark matches with exact case IDs and source line positions.
- `cross_split_sources.json`: all 28 exact-source split collisions and their line positions.
- `audit_manifest.json`: script hash, extraction rules, limitations, and SQLite index location.

The large SQLite index is under `training/outputs/audits/full_data_20260913/inventory.sqlite`; it contains hashes, lengths and byte offsets, not full source/target texts. Source files and actual training artifacts were not modified; no model was trained.
