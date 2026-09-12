# Archived Golden32 repeat revision

This user-requested revision has **32 evaluation occurrences and 31 unique cases**. It replaces the invalid archived case at slot 26 (`q_012053` / `r_012053_01`) with an exact copy of the first strictly valid case (`q_017270` / `r_017270_01`, automotive). No valid same-intent donor exists in this one-case-per-intent archive. The missing case's real-estate intent is therefore not covered by this revision.

The donor response, target, messages, assets, and original source/query identity are retained. Only the occurrence ID and explicit benchmark provenance distinguish the repeat. The original archive is unchanged. This is a repeated-case evaluation artifact; it is not a repaired Stage 3 reference or 32 independent test cases. Report the primary macro score over 31 unique sources and the secondary score over all 32 occurrences separately. The archive was selected from validation, so neither score is an independent final holdout estimate.

`benchmark_manifest.json` binds the original source SHA-256, output bytes, failed and donor identities, selection rule, counts, and scoring policy. The failed original's identity and normalized response hash remain reserved from training. This artifact is **different from** the original `genui_demo_golden40_gemini38_20260903` Golden32.

Reproduce from the original archive on a new host:

```bash
python training/scripts/create_golden_replacement.py \
  --source /path/to/genui_review_COMPLETE_zip/data/golden32/golden32.jsonl \
  --source-sha256 69344d162f6cd944c61d7c76cede6475d769b5ebe9e4591b77e1fd86aedc0824 \
  --failed-identity q_012053 --benchmark-id golden32_archive_repeat_v1 \
  --output-dir /path/to/new-golden32-repeat
```

Run strict training data checks while excluding both Golden cohorts before tokenized preparation. The filter retains accepted source records and emits no synthetic targets:

```bash
python training/scripts/audit_filter_training_data.py \
  --input /path/to/full-source/train.jsonl \
  --reserve-golden32 training/data/eval/golden32_archive_repeat_v1/golden32.jsonl \
  --reserve-golden35 training/data/eval/golden35_v1/golden35.jsonl \
  --require-source-identities \
  --tokenizer /path/to/local-e2b-tokenizer --tokenizer-loader pretrained_tokenizer_fast \
  --max-seq-length 4096 --max-input-tokens 4096 \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --output-dir /path/to/new-filtered-train
```

Omit `--output-dir` for a read-only audit. The adjacent replacement manifest is automatically included in source exclusions. Golden35's manifest likewise keeps its 15 excluded original sources reserved, in addition to its 35 scored cases. Run the same checks for validation, keep source IDs, and inspect quarantine and per-intent/component coverage before regenerating defective Stage 3 targets. Strict filtering is not evidence of semantic adequacy or sufficient difficult-case coverage; it provides a reproducible eligibility gate. Token coverage requires the actual saved tokenizer and cannot be inferred from character counts.
