# Evidence index

This directory contains portable summaries, per-case results and artifact/source hashes for the GenUICraft implementation. Read the repository's [VALIDATION.md](../../VALIDATION.md) for the interpretation and remaining Bixby integration limits.

- `v10_installed_sha256.txt`: historical installed APK identity.
- `v10_bixby_publication.json`: all 25 files copied into Bixby's local Maven repository, with matching hashes.
- `gemma50_v7_gpu_mtp`: full live Gemma generation, 50 cases; not a v10 generation run.
- `gemma50_v9_revalidated`: exact captured model outputs revalidated/rendered by the updated converter and renderer, with zero live model calls.
- `gemma_evidence_audit.md`: independent completeness, source-provenance, heading-only output-difference and renderer-capture audit.
- `gemma_v10_gpu_mtp_smoke`: two live checks on the delivery binary, separate from the full v7 run.
- `sdk_tests_publish.log`, `v9_actions_literals_tables.log`: SDK unit/build results and four focused renderer device tests. The v9 renderer bytecode is identical to v10.
- `gemma50_v7_runtime_evidence.txt`, `gemma_v10_runtime_evidence.txt`: native runtime/backend excerpts.

The full model attempts, source requests, generated Express/JSON, PNG/XML captures, Gradle logs and Bixby helper harnesses remain in `A2UI/tmp/genuicraft_20260918/`. The copied Bixby `RESULTS.md` files are summaries; their relative run commands and artifact paths refer to those original harness directories, not to runnable projects in this evidence folder. Model weights and APK/AAR binaries are not duplicated here.

The corpus is a frozen set of historical Perplexity answers used to test faithful conversion. Its content is not independently verified current factual advice. Screenshot/column checks are bounded and do not establish complete semantic or pixel-level correctness.
