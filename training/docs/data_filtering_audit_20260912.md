# Training and evaluation data audit — 12 September 2026

This is a CPU source-data audit, not a model quality or GPU validation result. Source samples came from `Downloads/review/genUI-review/extracted_20260905_172922/genui_review_COMPLETE_zip/data`. The original files were not edited. No new target supervision was synthesized, no tokenizer was downloaded, and no full training corpus was available locally.

## Exact cohorts and the requested replacement

The archive's `data/golden32/golden32.jsonl` is a different cohort from the original `dataset/data/runs/genui_demo_golden40_gemini38_20260903` Golden32. They have zero source-ID overlap and zero normalized-response overlap. The original September 3 reference targets pass strict validation 32/32; its prepared prompt revision is a separate reproducibility concern.

The archive file SHA-256 is `69344d162f6cd944c61d7c76cede6475d769b5ebe9e4591b77e1fd86aedc0824`. It has 32 rows, 32 unique sources, one row per intent, and all rows have historical `metadata.assigned_split=val`. It passes current strict Express plus production wire-schema/root-reachability checks on **31/32** rows.

The failed archive case is slot 26, `q_012053` / `r_012053_01` / `u_012053_01`, intent `real_estate`. Component `l` has scalar `Table.highlightColumns="priority"`, which fails the production array contract.

The requested replacement is stored as source data under `training/data/eval/golden32_archive_repeat_v1`. It copies the exact valid response/target/messages/assets from slot 1, `q_017270` / `r_017270_01`, intent `automotive`, into slot 26. There is no valid same-intent donor in this one-per-intent set; first-valid fallback is explicitly recorded. The result is:

- 32 strictly valid evaluation occurrences, **31 unique cases**.
- Automotive occurs twice; real-estate has no case in this revision.
- `golden32.jsonl` SHA-256: `8c7357103e6ea52d99d66430dd4b93242e1c4a4f0bf02f2fcfdf2a2e9c48de4c`.
- Unique occurrence IDs, preserved donor source/query/response IDs, and per-row benchmark metadata.
- `benchmark_manifest.json` records the donor, failed source, both hashes, true counts and scoring policy.
- The failed source's ID and normalized-response hash remain reserved from training despite its removal from evaluation.

Use the 31-unique-source macro score as the primary summary and the 32-occurrence macro score as a separate secondary summary. Neither is a 32-independent-case score. Historical archive Golden was drawn from validation, so it is not evidence of independent final-test generalization. Regenerating the real-estate reference later would require a new version and rebaselining all comparisons.

## Sample training/validation measurements

| Supplied sample | Input rows | Strict accepted | Strict failures |
|---|---:|---:|---|
| `train_original_first300.jsonl` | 300 | 252 | 47 wire-schema; 1 wire-lowering |
| `train_bottomup_first300.jsonl` | 300 | 265 | 34 wire-schema; 1 wire-lowering |
| `val_first100.jsonl` | 100 | 88 | 11 wire-schema; 1 unreachable graph |

After reserving the replacement Golden32, its failed original, and **all 50 sources from the historical reference run**, the accepted counts are 252, 265 and **87**, respectively. The current Golden35 artifact plus its 15 excluded-source reservations preserve that complete exclusion boundary. Validation line 97 duplicates archive Golden line 28 exactly in response and completion: `q_019461` / `r_019461_01`. It is quarantined as `reserved_evaluation_source`. No additional source-content overlap with either reserved cohort was observed in these training samples.

The two training samples each contain one source response occurring twice, at lines 1 and 21, with different valid target layouts. The audit reports this rather than treating different layouts as fabricated duplicates. Exact normalized-source/semantic-target duplicates are removed by the filter. There are 281 unique shared source responses between the two training samples; all 281 have matching semantic target hashes. The different acceptance counts reflect different sample memberships, not demonstrated semantic corruption in those paired rows.

All 700 sample train/validation records contain only `messages` and `completion`: original query/source IDs and intent metadata are absent. The audit can identify normalized-response leakage but cannot certify ID-level disjointness. `--require-source-identities` therefore quarantines these slim samples. Full training preparation must recover metadata from authoritative Stage 2/3 records, not invent IDs or train directly from the review samples.

Accepted original-train component occurrences show the observed mixture: Text 2,813; Stack 1,918; Button 837; Card 636; Table 421; Formula 43; Icon 8; Image 6; Tabs 5; Divider 2. These are component occurrences in 252 accepted rows, not independent examples. The accepted validation sample after source exclusion has only two Tabs occurrences and one CheckBox occurrence. Its scarce complex/interacting components and missing intent metadata prevent a credible per-intent sufficiency claim.

The archive's `DATA_STATS.md` reports 169,898 original train and 1,024 validation rows, and 164,225 bottom-up train and 986 validation rows. Those full files are absent from this audit package. The inspected prefixes are not a random sample and their rejection rates must not be extrapolated to the full corpus.

## Golden35: current passing cohort and historical exclusions

The historical source remains at `dataset/data/runs/golden50_g25pro_20260309_204033_stitch_compare_20260429_hybrid_r4/genui.jsonl`. Its initial source-materialization audit found **35 passing references and 15 failures**. The user selected those 35 passing references as a new, explicitly named **Golden35** cohort. The active artifact is `training/data/eval/golden35_v1/golden35.jsonl`, with configuration `training/configs/datasets/golden35_stage3_eval.yaml` and prepared output `training/outputs/datasets/golden35_stage3_eval/all.jsonl`.

Golden35 is a complete 35-case evaluation set, not a partial score reported under the historical 50-case name. Its manifest records the retained source membership and all 15 exclusions. Both the 35 retained sources and the 15 excluded sources remain reserved from training and ordinary validation. The original run folder and rejected references are retained as provenance; no raw data is deleted.

| Response ID suffix (`r_…_01`) | Exclusion reason from the original audit |
|---|---|
| `000001` | Wire schema rejects Text text `{"$item":"siteName"}` |
| `000004` | Express materialization rejects Table props `genre`, `mood`, `subtitle` |
| `000006` | Wire schema rejects object-valued Table `entityMedia` |
| `000010` | Wire schema rejects Image URL `{"$item":"imageUrl"}` |
| `000011` | Express materialization rejects Table `sourceFormat` |
| `000012` | Express materialization rejects Table `sourceFormat` |
| `000017` | Wire schema rejects Text text `{"$item":"stepNumber"}` |
| `000018` | Wire schema rejects Text text `{"$template":"Step ${step}"}` |
| `000019` | Wire schema rejects Icon name `{"$item":"icon"}` |
| `000028` | Express materialization rejects unsupported `justify` distribution |
| `000035` | Express materialization rejects Table `sourceFormat` |
| `000044` | Wire schema rejects conditional `visible` object with `$state`/`eq` |
| `000046` | Express materialization rejects Table `sourceFormat` |
| `000048` | Wire schema rejects Text text `{"$item":"time"}` |
| `000050` | Express materialization rejects unsupported `justify` distribution |

These historical exclusions include legacy dynamic-binding representations. No target was repaired or rewritten to create Golden35; it retains only the already passing references. Any later producer/parser repair or Stage 3 regeneration would create a separately versioned cohort and require fresh comparisons. The excluded 15 do not block current Golden35 evaluation, and they must not enter training merely because they are omitted from scoring.

## Reusable filter and next data preparation

`training/scripts/audit_filter_training_data.py` audits prepared supervision. It requires explicit `--reserve-golden32` and `--reserve-golden35` inputs. Both raw GenUI records with response text and prepared reference rows can provide exclusions. It automatically reads adjacent manifests for failed/excluded original sources, and accepts additional `--reserve-manifest` files.

The library API is `ir_training.data.audit_filter.load_reserved_cohorts(paths, manifests)` plus `audit_and_filter_rows(rows, reserved=..., tokenizer=..., max_seq_length=..., max_input_tokens=...)`. It checks identities and normalized source text (both raw and the training URL-placeholder form), strict targets, exact source/semantic-target duplicates, optional full-example token lengths, and reports per-intent/component counts and quarantine reasons. Accepted records are retained unchanged. Specify a new `--output-dir` to write `accepted.jsonl`, `quarantine.jsonl` and `audit.json`; omission is read-only. The output is a filtering artifact, not a tokenized training-manifest substitute.

```bash
python training/scripts/audit_filter_training_data.py \
  --input /path/to/authoritative-prepared/train.jsonl \
  --reserve-golden32 training/data/eval/golden32_archive_repeat_v1/golden32.jsonl \
  --reserve-golden35 training/data/eval/golden35_v1/golden35.jsonl \
  --require-source-identities \
  --tokenizer /path/to/e2b-tokenizer --tokenizer-loader pretrained_tokenizer_fast \
  --max-seq-length 4096 --max-input-tokens 4096 \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --output-dir /path/to/new-filtered-train
```

Run the same exclusion audit on validation. Recover original source identities and intent labels first, then inspect the complete quarantine and component/intent distribution. Regenerate defective targets through Stage 3 while preserving source provenance. Add genuinely separate training responses for weakly represented capabilities such as Tabs, conditional state, repeats, interactive controls, media, and mixed card/table layouts. Do not copy, paraphrase, or regenerate held-out source responses into training. Keep alternative valid layouts together in one source split and distinguish them from exact duplicates.

The actual E2B tokenizer was unavailable, so this audit makes no claims about token truncation rates. The archive uses a 421-character system prompt, whereas the current shared prompt is 4,766 characters. Re-run full prompt-plus-target token measurements with the exact model tokenizer and chat-template fingerprint after selecting the deployment scaffold. Long rows must be quarantined or handled by an explicit context-policy change, not silently truncated. Record counts and token distributions per intent before deciding how much additional data to generate.
