# Validation record — 16 September 2026

## Final integrated check

**215 passed, 31 deprecation warnings, 17 subtests passed in 52.11 seconds.**

```text
python -m pytest dataset/tests/test_h100_launcher.py dataset/tests/test_list_items_contract.py dataset/tests/test_ir_formats.py dataset/tests/test_source_quality.py dataset/tests/test_stage2_asset_validation.py dataset/tests/test_genui_metric_v5_4_run_audit_regressions.py dataset/tests/test_metric_diagnostic_metadata.py dataset/tests/test_generation_quality_regressions.py dataset/tests/test_stage3_a2ui_express.py dataset/tests/test_reasoning_capture.py dataset/tests/test_a2ui_express_prompt_generation.py training/tests/test_reference_binding.py training/tests/test_repairs.py training/tests/test_ir_targets.py training/tests/test_source_group_splits.py training/tests/test_training_scaffold.py -q
```

The executed command also used a fresh temporary directory and wrote [JUnit XML](C:/Users/anupk/Downloads/gemma_quality_cpu_tests/final_integrated.xml). Tests that invoke pipeline functions use mocked providers and tiny synthetic fixtures. They are not a real generation or training run.

## Other evidence

- Three standalone Kotlin/JUnit list-parser tests passed using cached Kotlin 2.1.20. The source parser and tests were compiled together; this does not establish whole-app compatibility with the project's configured Kotlin/Compose toolchain.
- `gradlew --offline :app:testDebugUnitTest --tests com.samsung.genuicraft.renderer.FlatListItemsTest` failed before app compilation because Android Gradle plugin 8.7.3 was unavailable offline. The wrapper distribution was downloaded; no APK was built or installed.
- Four static prompt copies verified with `python dataset/scripts/generate_a2ui_express_prompt.py --check`.
- Scoped `git diff --check` passed after eliminating a generated empty-property trailing space.
- The broader evaluator test selection reported 214 passes, 277 deselected. This overlaps the integrated check.
- Archive SHA-256: `361c02d642aa4847812a3bfb8e41b609666ae7cda440009330c4f5eaf4018dc0`; all 2,283 extracted file hashes still match.
- Read-only historical reference/wire probe: 321 compatible, 666 quarantined, 12 pre-existing format rejects; 887 preserved actions in compatible rows. No new training dataset was prepared.

## Not executed

No real LLM requests, GPU server startup, data regeneration, training, Android rendering/interaction evaluation, or performance benchmark. Linux process-group behavior and installed vLLM flag compatibility require the server pilot documented in the H100 runbook.

## H100 task completion recheck

The initial follow-up review verified launcher-to-client configuration propagation, aligned the prompt cap with context/output budgets, rejected mismatched endpoint counts, and made retry budgets explicit. That revision's focused generation, reasoning and launcher suite passed **42 tests** with mocked HTTP and file-only replica children. [Recheck JUnit XML](C:/Users/anupk/Downloads/gemma_quality_cpu_tests/h100_final_recheck.xml). Its reasoning-off speed candidate was subsequently removed at the user's direction; the correction below supersedes that setting.

The pipeline review, launch configuration and run instructions are complete. A claim of measured optimal speed requires executing the comparison on the actual H100 server; none was executed on this PC.

## Reasoning enabled correction

The H100 profile now pins thinking on for server setup and client requests, including when older activation scripts export disabled settings. It restores 8,192 completion tokens per stage, derives a 7,680-token Stage 3 prompt budget at 16,384 context, and prevents context-error retries from shrinking the completion budget. The generic adapter leaves template defaults intact when no thinking setting is supplied.

**46 focused tests passed in 10.75 seconds** (19 existing deprecation warnings): launcher plans and shell syntax, inherited disabled-setting protection, mocked HTTP reasoning controls, generation regressions, and reasoning capture. [JUnit XML](C:/Users/anupk/Downloads/gemma_quality_cpu_tests/h100_reasoning_on.xml). No model serving, real generation, training or GPU benchmark was run.
