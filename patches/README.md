# Checked Express preparation

The unsafe regex and nearest-definition experiments supplied with the September
review have been replaced. `express_bottom_up.py` wraps
`ir_training.data.express_preparation`: strict Express and wire-schema validation,
shared renderer-reference traversal, 100% root reachability, stable IDs, and exact
decoded-graph plus semantic-hash round-trip checks. Tabs, Modal, repeat/templates,
shared children, state, actions, inline components and root leaves remain part of
the contract. Invalid targets are quarantined; no content is removed or guessed.

`express_id_repair.repair()` always returns its input unchanged. Duplicate IDs
return `ambiguous_duplicate_ids_repair_disabled`; repeated/trailing output also
remains invalid. Fix decoder stopping and regenerate faulty predictions.
Historical repaired scores are diagnostic evidence only.

From the repository root on the other PC (paths are examples):

```bash
python patches/build_bottom_up_dataset.py \
  --input train=/data/normalized/train.jsonl \
  --input val=/data/normalized/val.jsonl \
  --input golden32=/data/normalized/golden32.jsonl \
  --output-dir /data/checked-root-first --ordering root-first \
  --tokenizer /models/selected-base --local-files-only --max-seq-length 4096

# Optional representation A/B: use the same sources and tokenizer settings.
python patches/build_bottom_up_dataset.py \
  --input train=/data/normalized/train.jsonl \
  --input val=/data/normalized/val.jsonl \
  --input golden32=/data/normalized/golden32.jsonl \
  --output-dir /data/checked-bottom-up --ordering bottom-up \
  --tokenizer /models/selected-base --local-files-only --max-seq-length 4096
```

This command performs CPU data preparation, with no weights or training. Without
`--tokenizer`, it does not load tokenizers either. It requires `dataset/` schemas
and `jsonschema` from the training environment. Token length checks render the full
authoritative chat template and encode with `add_special_tokens=False`; they never
truncate targets or prompts. Prepare data separately for E2B and 270M tokenizers.
`--max-input-tokens` can additionally gate the generation prompt budget.
Match `--tokenizer-loader` to the training YAML (`pretrained_tokenizer_fast` for
the reviewed E2B recipe; `auto_tokenizer` for 270M), and copy any configured
`model.chat_template_kwargs` via `--chat-template-kwargs '{"enable_thinking":false}'`.
Do not silently switch tokenizer implementations if a loader fails.

Every assistant demonstration, final target, target alias and plain prompt is
updated consistently. Each source row must bind its completion to the final
assistant turn and response to the final user turn. Split membership and source
identifiers are preserved. System/scaffold text is not replaced with a new prompt.

Each new output directory is published by one same-volume directory rename only
after all splits complete. Existing destinations are refused. Invalid rows are
saved verbatim in `quarantine.jsonl` with reason and source line; malformed JSON,
I/O errors or an empty accepted split abort the build. Inspect quarantine reasons
before training. Do not report a filtered Golden set as Golden 32: fix invalid
reference sources upstream and rebuild all 32 expected cases.

`manifest.json` records source/output/code/schema hashes, accepted source-row
hashes, rejection counts, component/reference coverage, lengths and prompt
scaffold counts. For an ordering A/B, each split's `accepted_source_rows_sha256`
must match; ordering can alter tokenizer lengths, so compare counts after gating.
`prompt_scaffolds.json` pins system and demonstration messages. Use the prepared
messages for HF evaluation and verify deployed tokenizer IDs against that scaffold
before export. Android's current root-first example does not automatically acquire
the bottom-up example. Text fingerprints alone do not prove deployment parity.
Treat multiple scaffold hashes as an explicit prompt-mixture experiment. Training,
checkpoint generation, export and device validation run separately on the GPU PC.
