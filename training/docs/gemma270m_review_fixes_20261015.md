# Gemma 3 270M training review — 2026-10-15

This review covers the shared full-SFT/QAT Trainer, QAT LoRA GRPO, and the
270M LiteRT-LM export/evaluation launchers. It does not certify GPU memory,
convergence, actual LiteRT conversion, or Android delegate performance.

## Correctness fixes

- **Train/export weight parity:** effective LoRA QAT now computes the adapter
  delta outside autocast and matches PEFT's add-then-round merge arithmetic.
  Frozen base weights are not modified; gradients still reach the adapters.
  Tests compare BF16 weights and W8 quantized values against real PEFT merging.
- **Full-model serialization:** final and Golden-best full-SFT/QAT saves omit
  the incompatible generation cache implementation only while serializing.
  Training-time cache behavior and the separate ZeRO-3 save branch are unchanged.
- **Completion masking:** labels align with the tokenizer's padding side.
  Partial or length-mismatched labels are rejected instead of training on the
  wrong token positions.
- **Golden resume:** saved Trainer states refer to immutable, byte-verified
  selected-model snapshots rather than only the mutable latest-best directory.
  Missing, modified, or additional model/index files fail validation. The legacy
  standalone full-QAT runner also forwards explicit resume to its Golden callback.
- **Local base identity:** new ordinary local-base SFT resume contracts bind
  model/config bytes. GRPO records the loaded local base identity too. Merge
  verifies these recorded identities and uses the bound local source, including
  when the caller does not repeat it as a CLI argument. Tokenizer fallback no
  longer unexpectedly selects the Hub base instead of the configured source.
- **Source grouping:** inferred run-local IDs are scoped to source provenance;
  explicit global source/family IDs stay intact. Normalized duplicate sources
  link their complete families before splitting. Reserved-content checks remain
  independent of IDs. Legacy origins are recovered from saved provenance when
  available; originless legacy ID collisions are rejected conservatively.
  Pinned benchmark artifacts are not rewritten.
- **Official Q8 follow-up:** public W8 and official Q8 evaluations have separate
  reports/logs. The official artifact selection is persisted and bound to its
  bytes and training config, so later evaluation/scorecards use the same model.

## Compatibility and operation

- Old ordinary local-base checkpoints without a base digest cannot prove the
  new exact-resume contract and are rejected. Do not edit their metadata to
  claim a digest retroactively. A weight-only initialization is a new run, not
  an optimizer resume. Retained-mobile seed-manifest contracts retain their
  existing separately verified identity path.
- Hub-only base IDs remain subject to the existing remote-model contract; this
  change does not download or hash a remote model automatically. Prefer a
  frozen local model directory for reproducible 270M experiments.
- Keep `best_golden_checkpoint_snapshots/` alongside numbered Trainer
  checkpoints when relocating resumable runs. It retains one model copy per
  improved best referenced by a save and is not pruned by `save_total_limit`.
  Use a fresh output directory for continuation; replaying a step into a
  conflicting existing snapshot fails closed.
- Re-prepare ordinary training data to obtain scoped fallback IDs. Existing
  prepared data and pinned Golden/Bixby fixtures are not edited in place.
- The older `train_full_finetune_qat.py` Space runner is a distinct experimental
  entry point. Its documented DeepSpeed configuration is valid for that runner;
  it still does not implement the shared pipeline's complete optimizer/data/base
  resume-provenance contract. The callback fix does not claim otherwise.

## Validation boundary

Completed regression runs: 591 passed / 2 skipped for the main training/export
suite, and 292 passed / 1 skipped for the final data/parity suite (these suites
overlap; their counts are not an independent-test total). The final run also
loaded the staged `fake_quant.py` version to verify the LoRA fix independently
of unrelated working-tree embedding edits. Compilation, scoped fatal-error Ruff,
and staged whitespace checks passed.

Regression coverage includes real tiny CPU HF Trainer full-SFT/full-QAT runs,
save/reload checks, real PEFT merge parity with CPU autocast on/off, and offline
GRPO/runtime/reward/export, dataset-overlap, resume, and provenance tests.
Use the target training environment for H100/DeepSpeed training and the target
LiteRT runtime/device for end-to-end conversion and GPU inference validation.
