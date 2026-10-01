# v10 provenance, duplicates, split leakage, and benchmark overlap

## Scope and integrity

I scanned **every current row** of `training/outputs/datasets/full_data_archive_recovered_v10/train.jsonl` (91,115) and `val.jsonl` (1,862), not a sample. Their bytes, SHA-256 hashes, and row counts exactly match `training/data/train/v10/bundle.json` (`707244a5…addf5` and `b5024919…0fa7`). These are the restored inputs under audit. This audit did not train a model or change a dataset row.

The following outputs make the checks reproducible: [leakage_audit.py](leakage_audit.py), [leakage_counts.json](leakage_counts.json), [duplicate_source_examples.csv](duplicate_source_examples.csv), [reference_census.py](reference_census.py), [reference_counts.json](reference_counts.json), [benchmark_near.py](benchmark_near.py), and [benchmark_near_counts.json](benchmark_near_counts.json). The two overlap CSVs are header-only because the defined checks found no match. Run each Python script from the repository root with Python 3.11+; no model, network access, or GPU is used.

## Source and target duplicates

| Complete-file measure | Train | Validation | Combined |
|---|---:|---:|---:|
| Rows | 91,115 | 1,862 | 92,977 |
| Unique exact source texts | 88,671 | 1,814 | 90,485 |
| Repeated exact-source groups | 2,431 | 47 | 2,478 |
| Rows in those groups | 4,875 | 95 | 4,970 |
| Repeated groups with >1 raw target hash | 2,431 | 47 | 2,478 |
| Unique raw target texts | 91,115 | 1,862 | 92,977 |

Whitespace-folded and reference-insensitive source signatures have the **same 90,485-group census** as exact source text in the current rows. No duplicate raw target text or whole source-target pair was found. There are 90,483 synthetic `source_id` groups, two fewer than exact-source groups, reflecting two additional source-variant groupings recorded in `near_source_groups.json`; those groups remain inside a single split.

All 2,478 repeated exact-source groups have multiple `effective_semantic_sha256` values as well as multiple raw target strings. **That is target-label variability, not proof of a factual contradiction.** The hash changes for a state-path rename or an alternate layout. For example, train lines **11,162 / 45,960 / 47,130** contain the same London kitchen roadmap source: the three targets hold the same four roadmap rows but choose `roadmap_data` versus `roadmap_rows` and different card/button grouping. Validation lines **9 / 244** similarly present the same Mexican fiesta source with different layout and text grouping. These inspected examples are plausible alternate layouts; they do not certify the other 2,476 groups. Review those groups for omissions, fabricated references, or contradictory values before applying one-source-one-label training or sample weighting. The example CSV preserves the top 25 groups by row multiplicity with coordinates and hashes.

## Split isolation

| Train/validation comparison | Cross-split groups | Validation rows affected |
|---|---:|---:|
| Exact source text | 0 | 0 |
| Whitespace-folded source text | 0 | 0 |
| Reference-insensitive source signature | 0 | 0 |
| Synthetic source family ID | 0 | 0 |
| Archived original-source SHA-256 | 0 | 0 |
| Exact target text | 0 | 0 |

For a bounded lexical near-source check, I indexed all 1,862 validation sources by up to 24 rare word trigrams, scanned all 91,115 training sources, and scored 134,009 candidate pairs sharing at least two anchors with word-length ratio ≥0.65. **No pair reached trigram Jaccard ≥0.60.** This reproduces the v10 manifest's source-family isolation claim within the tested method. Anchor retrieval can miss semantic paraphrases, reordered passages, large entity substitutions, or short shared content, so it cannot certify semantic independence.

## Held-out source overlap

| Local cohort | Distinct tested source cases | Direct matches in train or validation | Retained lexical near matches |
|---|---:|---:|---:|
| Golden35 accepted | 35 | 0 | 0 |
| Golden35 excluded original sources | 15 | 0 | 0 |
| Golden32 accepted | 31 (32 evaluation slots repeat one donor) | 0 | 0 |
| Golden32 excluded failed case | 1 hash-only | 0 | Not testable without source text |
| Bixby50 source-only holdout | 50 | 0 | 0 |

Direct comparisons used raw source SHA-256, whitespace-folded source SHA-256, and the repository's reference-insensitive v10 source signature. The excluded Golden35 texts came from the 50-response source run named in its benchmark manifest. For the failed Golden32 case `q_012053`, the local manifest supplies a source hash but no source text, so only direct hash matching was possible. The Bixby50 set has sources but no gold target labels.

For the 131 distinct benchmark texts available, a second complete train/validation scan used reference-insensitive words, up to 24 rare trigram anchors per benchmark, at least two common anchors, token-length ratio ≥0.40, and then retained Jaccard ≥0.60 or ≥0.90 containment of the shorter source when it had at least 50 words. It scored **3,536 candidate pairs** after the length filter and retained **zero**. This is bounded lexical evidence, not exhaustive semantic/paraphrase matching. The empty [benchmark_overlap.csv](benchmark_overlap.csv) and [benchmark_near_candidates.csv](benchmark_near_candidates.csv) record that no row met those exact match rules.

These results apply to the current v10 files. They do not establish whether an **older checkpoint** trained on the prior supplied archive consumed any formerly overlapping Golden sources. Retraining on the cleanly separated input and binding the checkpoint to its prepared-data hash is required for an unseen-source benchmark claim.

## Provenance and reference binding

The archive records a source-file coordinate, original source/target hashes, effective hashes, repair history, and assigned split for **all 92,977** rows. Recomputed effective source and target hashes match row metadata in every case; no assigned-split mismatch was found. The bundle binds the restored files to their tracked checksums. This is useful archive-level traceability.

Original generator lineage remains incomplete. In every row, `metadata.query_id` and `metadata.intent` are null, `source_id` begins with `archive-source-family-sha256:`, `archive_recovery.identity_kind` explicitly says it is **not an original generator ID**, and `missing_historical_metadata` lists original query/response IDs, generator lineage, original URL map, asset manifest, and intent. No row declares a new Stage 3 generation run or a `source_model_family`. Thus a content hash and archive coordinate should not be presented as recovered Stage 1/2/3 identity or teacher provenance.

The current `response_text` is overwhelmingly placeholder-bearing:

| Source/reference measure | Train | Validation |
|---|---:|---:|
| Rows containing placeholders | 91,052 | 1,859 |
| Rows with ≥3 distinct placeholder tokens | 87,846 | 1,778 |
| Rows with literal HTTP(S) URL still in current source | 17 | 0 |
| Rows with a nonempty **current** `url_preprocessing.url_map` | 47,180 | 928 |
| Placeholder-bearing rows with **no** current map binding | 43,872 | 931 |
| Distinct source-token occurrences by row without current map binding | 299,341 | 6,308 |

Current URL maps are all-or-none for a row's source placeholders in this scan: 47,180 training and 928 validation rows have every source token mapped; 43,872 and 931 have none mapped; zero have partial coverage. A map can hold real URL bindings created during this recovery's preprocessing (train line 2 has icon and action URL entries) even though **every row marks its historical original URL map missing**. Conversely, an empty current map does not prove a placeholder was originally an invalid URL; many sources arrived already masked. Mapless actions/media require special review or original-source recovery before claiming usable live reference behavior. The regex counts are for current source text; URL values held *inside metadata maps* are counted separately in `reference_counts.json`.

## Practical disposition

1. Keep Golden35 accepted **and excluded**, Golden32 accepted **and excluded**, and Bixby50 sources outside any future training/ordinary-validation family split. Check a prepared manifest and checkpoint data hash after filtering; do not infer an older checkpoint's exposure from these current files.
2. Preserve archive coordinates and original/effective hashes, but mark original generator/query/teacher/URL-map provenance unresolved. Where live reference restoration matters, either recover authoritative URL maps or quarantine mapless cases from that evaluation.
3. Treat 2,478 repeated exact-source groups as label-review candidates, with all variants kept in the same split and source-level weighting or selection after semantic review. Different target hashes alone are insufficient to choose or reject a layout.
4. Recheck paraphrase and scenario-family overlap with a stronger semantic review if the validation score will be described as independent of training scenarios. The present lexical checks give defensible lower-bound evidence, not a complete semantic guarantee.
