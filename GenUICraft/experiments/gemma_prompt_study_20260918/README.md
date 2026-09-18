# Gemma E2B prompt study — 18 September 2026

This is an on-device development study requested specifically for Bixby50. It tunes on that corpus; results are not a held-out generalization estimate.

**Complete: retain the accepted v10 delivery (50/50).** Ten new strategies were investigated. The strongest screen candidate, scaffold, failed final confirmation at 48/50 and is not shipped. Production source and rebuilt AAR hashes match v10 exactly. See [REPORT.md](REPORT.md) for the complete decision, native GPU+MTP throughput, and limitations.

## Fixed execution and acceptance

Screening and broader confirmation used the published v10 AAR in the SDK-only reference consumer. Prompt overrides called the public `GenUiConverter.withPrompt` API with `useSourceBindings=true`. Final scaffold confirmation used the v11 candidate AAR. Model weights, GPU backend, MTP, enabled thinking with a 1,024-token budget, temperature 0, content validators, and renderer remained fixed. Failed outputs are retained and never replaced by a fallback layout. The model chooses a source-bound Express layout; the SDK inserts exact original source values and validates all bindings and content.

Initial app SHA-256: `19eee5045f55553205e57b5e50e4c46332d4c6dfc9c4ed04a35598e1ef5cb5a9`. Baseline prompt SHA-256: `154ff4d27d8e37b81279e7979c18b4fd90bbdd5135a4e124917712a85438535b`.

## Staged comparison

1. Fresh baseline, nine materially different prompt candidates, and a repeated baseline on BXP-003/030/032/037, with zero repairs. These cases include all three historical first-attempt failures. They intentionally form a difficult development screen.
2. Broader format coverage for the strongest candidates. Preserve first-attempt and repaired counts separately.
3. Full 50-case live confirmation of the selected prompt with the production repair budget. Promote only after inspecting failures and renderer evidence.

Each run stores the exact prompt, checksum, command, case IDs, app identity, temperatures, raw attempts, validation failures, generated Express/JSON, screenshots, and completion status. The corpus test retains source order regardless of the requested case ordering, so BXP-003 is the first initialization-inclusive screen case. Whole-process PSS is sampled; it is not model-only memory.

`tools/run_prompt_study.py` executes a JSON plan serially against the device. Raw artifacts live under `../../../tmp/genuicraft_prompt_study_20260918/`. Portable summaries are in `results/`; `evidence/` reconciles every completed attempt and preserves final generated documents. Failed and intentionally unfinished cases remain explicitly identified.

Latency comparisons must use the same cases and repair budgets. The old full-run median includes different cases and repairs and is only historical context. Baseline bookends and temperature readings expose drift; they do not create a thermally controlled benchmark. GPU+MTP runtime identity proves configuration, not accepted speculative-token rates.

Two additional variants were frozen after inspecting the first compact-prompt rejection: `mapping` uses a shorter concrete mapping with decoded-JSON punctuation guidance; `baseline_targeted` retains the original examples and adds explicit delimiter, decoded-root, and element/binding independence rules. Their screening results remain development data.

## Adaptive rejection rule

After compact failed 0/4, example-first passed 2/4, and grammar failed its first two cases, the screen was amended to stop a new candidate after two completed validation/render failures. Grammar was stopped while its third case was still generating; its result is **0/2 completed, with two unfinished**, not 0/4. In-flight work is excluded from completed-case latency and result counts. Every surviving candidate still runs all four cases. The remaining screen and baseline bookend use a five-minute per-case cap; the initial baseline and earlier runs used ten minutes, and all their completed cases took under five minutes. Full production confirmation retains its original ten-minute cap and one-repair budget. Early rejections are recorded separately from completed runs.

The targeted-baseline candidate completed the hard screen at 3/4, improving on the fresh baseline at 1/4. It is provisionally eligible for broader testing despite the original ideal 4/4 gate: both historical missing-component/opening defects were fixed, while a distinct binding-token error remains. This adaptive advancement is not production acceptance; the full corpus with the production repair budget and first-pass counts must determine promotion.

During broader confirmation, targeted-baseline completed 5/6 successfully and failed BXP-025 on shortened bindings. Together with its known BXP-030 screen failure, its best possible score across the planned 16 distinct cases was 14/16, below scaffold's completed 15/16. It was deliberately stopped at that point under an additional adaptive upper-bound rule; six broader cases remain unfinished. The decision was timestamped before the force-stop in `adaptive_rejection_decision.json`. Android reports that intentional process stop as an instrumentation crash; it was not a spontaneous native crash. Report 5/6 completed, not 5/12. This is a development-selection rule, not an unbiased generalization estimate.

## Additional input-scaffold experiment

After the system-prompt variants exposed repeated root, binding, and element-name copying failures, a tenth candidate was prepared. The test provider appends a complete Express scaffold after the original source JSON. It supplies the same fixed block-to-component mapping already required by the original prompt, leaving `SELECT_DOMAIN` and `SELECT_PRESENTATION` placeholders for each table. Gemma must return the entire program with valid choices. The AAR still performs its unchanged binding/content validation; no generated output is repaired or substituted by the scaffold helper.

This prototype is explicitly marked `inputScaffold=true`, saves each effective input prompt, and uses the same actual AAR Gemma provider/GPU/MTP/thinking settings. It requires the newly compiled instrumentation APK. A separate no-inference contract test checks all 50 source records plus code/divider/literal cases, including the bounded-repair user-prompt shape. That fixture test is not model-success evidence.

The scaffold was subsequently ported into the locally built v11 SDK candidate. Only that experimental implementation adds `useLayoutScaffold`; the accepted production API has no such argument. The v11 full run used `packagedPrompt=true`, `recordInputs=true`, and no test-provider `inputScaffold` wrapper. Every recorded input contained exactly one SDK scaffold.

The candidate passed 289 JVM tests, including byte-for-byte prototype parity across all 50 source inputs, code/divider/literal handling, initial/repair prompt placement, custom-prompt compatibility, and failure without a fallback. These are source-contract tests with fake providers, not extra live-model successes. The final live run nevertheless failed two cases, so production was restored and its original 285 tests passed again. Candidate hashes remain in `results/v11_build_manifest.json` and `v11_sdk_source_hashes.json`.

`candidate_source/` preserves the candidate's changed converter, added scaffold helper, prompt and four-test class at their original relative paths. They are outside Gradle's production/test source sets. The frozen `final_plan.json` describes the historical candidate run; rerunning it requires first restoring those candidate files and rebuilding/publishing/installing that candidate. Running it against the retained v10 binary would not reproduce the scaffold experiment. No production IR was manually rewritten.
