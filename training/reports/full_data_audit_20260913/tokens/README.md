# Full-data token-length audit — 2026-09-13

## Scope and confidence

Every supplied training/validation row was measured: **150,292 train + 910 validation**, plus all **32 Golden32 occurrences and 35 Golden35 cases**. There were no rejected message layouts or tokenization failures. The streaming run completed in 1,388.49 seconds; it did not load model weights, download models, train, generate predictions, or change the input files.

The actual files inspected are `C:/Users/anupk/Downloads/training_data/train.jsonl` and `val.jsonl`. Both are **UTF-16 with a little-endian BOM**, not UTF-8. Import must account for this before using the repository's UTF-8 JSONL readers.

Two confidence levels must not be conflated:

- **Exact local-tokenizer measurements:** source and target text are encoded with the locally supplied Gemma3 tokenizer, without truncation, padding, or automatically added special tokens.
- **Reconstructed chat measurements:** the same tokenizer measures a stated Gemma3-style text frame with BOS, user/model turns, end-of-turn delimiters, and system instructions prepended to the first example user. This is exact for that stated frame, but is **not proof of the original/deployed chat template**. The available exported tokenizer has no chat template. The framing follows Google's documented [Gemma formatting convention](https://ai.google.dev/gemma/docs/core/prompt-structure).

No E2B tokenizer was found in the task-relevant local `Downloads/training` tree. **None of these numbers are claimed to be E2B token counts.** Run the production preparation gate with the exact E2B tokenizer and deployed chat template on the GPU machine.

## Local tokenizer evidence

| Item | Observed value |
| --- | --- |
| Tokenizer | `Downloads/training/runs/gemma270m_ir_lora/merged_hf/tokenizer.json` |
| Tokenizer SHA-256 | `a306777fab20c8efd76229cea1566e9ec976059a4f2fb73565bc944ff3c2df4f` |
| Base vocabulary / including added tokens | 262,144 / 262,145 |
| Added out-of-base-vocabulary marker | ID 262,144: `<image_soft_token>` |
| BOS / EOS / start-turn / end-turn IDs | 2 / 1 / 105 / 106 |
| Exported tokenizer class | `TokenizersBackend` |
| Local chat template | Absent from config and files |
| Local model configuration | `Gemma3ForCausalLM`, `gemma3_text`, maximum positions 32,768, vocabulary 262,144 |

Calling `PreTrainedTokenizerFast.from_pretrained(..., local_files_only=True)` and then `apply_chat_template` reproduced: `Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed`. Resolve and pin the actual training/inference template before treating this bundle as a production-ready training input.

A BOM-aware literal-string scan found no `<image_soft_token>`, `<start_of_turn>`, `<end_of_turn>`, `<bos>`, or `<eos>` strings in either data file. The added image marker therefore was not detected in these source files; this is not an assertion that the text-only model supports that extra token.

## Source and target lengths

These are actual token counts of supplied text, before source-quality/leakage filtering. The completed canonical second pass verified that **all 150,292 training and 910 validation targets are strict-valid and already equal the production canonical serializer's output**, with no binding errors recorded by that audit. Consequently canonicalization does not change any of the train/validation lengths below. Strict validity is not proof of source fidelity or benchmark independence. The target column excludes the chat end-of-turn suffix.

| Split | Source mean / p99 / max | Target mean / p99 / max | Targets over 2,048 |
| --- | ---: | ---: | ---: |
| Train, 150,292 rows | 604.316 / 1,285 / 2,616 | 998.492 / 1,542 / 2,474 | 47 |
| Validation, 910 rows | 594.295 / 1,065 / 1,344 | 996.976 / 1,507 / 2,014 | 0 |
| Golden32, 32 occurrences | 600.688 / 1,035 / 1,035 | 1,008.594 / 1,430 / 1,430 | 0 |
| Golden35, 35 cases | 667.200 / 1,353 / 1,353 | 916.714 / 1,539 / 1,539 | 0 |

Including the reconstructed end-of-turn suffix, **49 training completions** exceed 2,048; no validation or Golden completion does. The Golden maxima including that suffix are **1,432 and 1,541**, respectively. A 2,048-token generation budget is therefore sufficient for every provided Golden reference under this tokenizer/frame; this does not guarantee that a model will generate the right answer or avoid verbosity.

## Shared-prompt impact

All supplied train/validation rows use the short source scaffold. Golden32 has the same logical short scaffold; Golden35 already uses the full production scaffold. Serialized key ordering creates distinct raw scaffold hashes for some logically identical messages, so those hashes must not be interpreted as separate prompt designs.

The production shared-prompt contract measured here is `00a5af656e875f8bbe7b15e000efb4ef751f8e6223da3608648d20e94b296bf5`.

| Fixed portion | Short scaffold | Shared production scaffold |
| --- | ---: | ---: |
| System instruction tokens | 109 | 1,388 |
| Sum of system + few-shot message-body tokens | 190 | 1,469 |
| Reconstructed fixed chat prefix | 203 | 1,482 |

The new scaffold adds **1,279 tokens per example**. Mean training sequence length consequently increases from **1,829.809 to 3,108.809 tokens** (about 69.9%). This is an input-format change, not new source information or additional target supervision. It matters for compute planning and for keeping training and evaluation comparable; do not retain the short training prompt while evaluating with the full one merely to make lengths smaller.

## Shared-prompt sequence fit

Sequence counts follow the repository's SFT convention: separately tokenized generation prefix plus separately tokenized assistant suffix. They include the complete shared production prompt and all supplied target text under the reconstructed Gemma3 frame.

| Split | Mean | Median | p95 | p99 | Maximum | Over 4,096 | Over 6,144 / 8,192 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 3,108.809 | 3,120 | 3,782 | 4,160 | 5,467 | 1,945 (1.294%) | 0 / 0 |
| Validation | 3,097.270 | 3,117 | 3,743 | 4,001 | 4,693 | 6 | 0 / 0 |
| Golden32 | 3,115.281 | 3,038 | 3,847 | 3,857 | 3,857 | 0 | 0 / 0 |
| Golden35 | 3,089.914 | 3,010 | 4,039 | 4,398 | 4,398 | 1 | 0 / 0 |

Before other filtering, **148,347 train and 904 validation rows fit a 4,096-token training sequence**. All those fitting rows also fit the 2,048-token completion budget. The 47 over-budget raw targets and 49 over-budget chat completions are already within the sequence-overflow cohort.

There are three different budgets:

1. **Training sequence budget:** prompt plus supervised target, currently 4,096 by default.
2. **Inference prompt budget:** input only, currently 4,096. Exactly one training prompt exceeds it (train line 104,316: 4,120 tokens); no validation or Golden prompt exceeds it.
3. **Generation budget:** newly generated tokens only, currently 2,048. This must also fit the deployed model's total context when added to the prompt.

Golden35 line 24 has a 2,857-token prompt and a 1,541-token reference suffix: 4,398 combined. **It must not be removed from evaluation for exceeding the training sequence limit.** The model receives only its prompt during generation; the repository correctly gates Golden prompt length separately. All Golden32 prompts are 1,800–2,539 tokens and all Golden35 prompts are 1,760–2,857.

## Opinion and recommended action

1. Keep the production prompt consistent across training and both Golden sets, and measure with each actual model's tokenizer/template before finalizing capacity settings.
2. A **4,096-token main training bucket is reasonable for this corpus**: it covers 98.706% of supplied training rows under the measured frame. Do not truncate target DSL to make the remaining rows fit. If their content/quality warrants retention, use a separately audited 6,144-token bucket or regenerate a compact, fact-preserving target through Stage 3. All measured supplied sequences fit 6,144, but E2B must be checked with its own vocabulary and context contract.
3. Preserve the 2,048 default generation ceiling for the current Goldens. For a broader production deployment, inspect the 49 longer supervised completions and decide whether they need compact regeneration or a longer dedicated inference mode; silently teaching labels that cannot be completed at serving time is undesirable.
4. Use length-aware batching to reduce padding, after correctness filtering and source-group splitting. More GPUs or longer sequence settings do not cure incomplete/hallucinated targets.
5. **Length is not the dominant proof of data quality.** A row that fits every budget may still omit facts, fabricate assets, have broken Unicode, duplicate a held-out source, or lack provenance. Apply the separate IR, source-fidelity, leakage, and provenance audit decisions before calling these counts usable training rows.

## Evidence and reproduction

- `summary.json`: full histograms, all threshold counts, source hashes, exact tokenizer fingerprint, production scaffold evidence, and rejected-row lists.
- `training/outputs/audits/full_data_20260913/token_lengths.csv`: one row per measured source coordinate; keyed by split, one-based physical line, and exact raw UTF-8 target hash. Contains 151,269 data rows. This large local evidence stays outside committed reports.
- `canonical_summary.json` and `training/outputs/audits/full_data_20260913/canonical_token_lengths.csv`: completed second-pass evidence joining all 151,202 train/validation coordinates to the strict IR audit. Every target hash matched, every target was strict-valid, and every target already equaled canonical serialization. Its counts remain distinct from leakage/semantic-quality-cleaned training eligibility.

Run from the repository root:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python training/scripts/audits/full_data_tokens_20260913.py
python training/scripts/audits/full_data_tokens_20260913.py --canonical-only
python -m pytest training/tests/test_full_data_token_audit.py -q
```

The canonical-only command intentionally refuses an incomplete IR database. Audit helper tests passed (**2 tests**); full-stream measurements passed 167 direct prefix-boundary comparisons across the first 100 training rows and all 67 Golden occurrences. Those checks validate the counting implementation, not a missing model template or model quality.

Input hashes (recomputed after the full raw and canonical passes; both source files were unchanged):

- Train: `5c47fb3f07dc6f2711ece12a1a93b4ecd6977fa8060119f329db3189c2d6b910`.
- Validation: `04e88d07003803279cea7de093bd9c69bcd35ed9a5fab057d577544b8c90ba19`.
- Per-row token CSV: `d0ae09d1881b5351a72fd809436bfbcb1204539b08720b9806f21923756dd12f`.
