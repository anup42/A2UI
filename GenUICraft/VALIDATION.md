# GenUICraft validation — 18 September 2026

The separate **19 September trained E2B v10 W4 pilot** is documented in
[its report](validation/20260919_e2b_v10_w4/REPORT.md). The user stopped it after
12 completed cases: one compiled/rendered with poor content/layout, ten returned
invalid Express, and one timed out. Native GPU decode median was **6.38 tokens/sec**
across 11 returned outputs. This package has no MTP drafter and is an additional
app option, not a replacement for the accepted official E2B prompt. SDK 0.2.0
updates LiteRT-LM to 0.16.1 and adds the training-compatible prompt adapter.
The historical acceptance results below remain attached to their original builds.

The library and reference consumer are built and tested. Both providers completed the 50-case corpus, with the build-specific evidence below. Bixby source integration is prepared; Bixby is not built or installed because its private dependencies are unavailable.

The subsequent **v14 renderer visual update** is recorded in [the visual review report](validation/20260918_visual/REPORT.md). The v10 evidence below remains the conversion baseline. V14 changes eight renderer/theme/wrapper files, retains the accepted conversion prompts and native runtime, and replaces the copied AAR in Bixby. Its AAR SHA-256 is `bdd8b06710e4118aa130fb506e181527b31fa6d204b858b47237fa3bff5160fd`; 294 JVM tests and four focused device checks pass. Current build and publication hashes are in [the v14 evidence](validation/20260918_visual/host_build_evidence.json).

V14 also passes **50/50 saved Gemma JSON renderer replays** and **5/5 selected dark-mode replays at 130% font scale**, with zero model calls. The full audit verifies all source hashes, 222 capture pairs, 23 tables/100 columns and every vertical end. See the [before/after gallery](validation/20260918_visual/gallery/gallery.html) and [independent replay audit](validation/20260918_visual/full_replay_v14_audit.json). These are renderer checks, not another model-generation run.

## Accepted delivery v10

- **285 JVM tests pass** in 31 suites, with zero failures, errors or skipped tests. Release publication and the actual AAR consumer build/install succeed.
- AAR: `genuicraft/build/outputs/aar/genuicraft-release.aar`, **9,353,001 bytes**, SHA-256 `7b33e20601a12f8090cec1f774ab2141bba35c3fea470353bdbf5995ec40c108`.
- Reference APK SHA-256: `19eee5045f55553205e57b5e50e4c46332d4c6dfc9c4ed04a35598e1ef5cb5a9`; test APK: `74a1b626d46aacc4db5ef0582e80a3226a8743150e52211eb57ca11b3a793388`. Installed-file hashes match.
- **4/4 focused device tests pass on v9**: renderer-only callbacks, exact source URLs/literal labels, literal text through JSON replay, and horizontal table access. V10 has an identical `classes.jar`, renderer sources, Gemma prompt and native libraries; only the Gauss prompt differs.
- **Gemma live v7: 50/50 successes**, 47 first attempt and 3 repaired (BXP-030/032/037), zero fallbacks. All 53 attempts report GPU+MTP; thinking remained enabled.
- **Gemma captured-output revalidation/rendering on v9: 50/50**, zero live model calls and zero further repairs. This is not another live inference run.
- **Gauss live v10: 50/50 successes**, 49 first attempt and one repaired (BXP-011), zero fallbacks. Its **saved-JSON renderer replay also passes 50/50**, with zero live model calls.
- **Gemma live v10: 2/2 first-attempt successes**, BXP-003 and BXP-005, at 28.504 and 55.847 seconds, with zero repairs/fallbacks. This confirms the delivery binary separately from the full v7 generation run and v9 captured-output replay.
- All **25 files** in Bixby's copied `aars/genuicraft-maven` publication match v10 by SHA-256, including AAR, POM, module metadata and checksum files.
- **34 distinct exact Bixby helper tests pass** in isolated public-dependency harnesses: provider cancellation/cleanup (2), action policy (4), attachment state (1), request lifecycle (6), source parsing (9), and streaming/provenance (12). The 32-test harness ran twice with fresh outputs. Original/copy hashes match. Its three Android/logging/view shims are compile-only and their behavior is not exercised; the provider harness uses JUnit 5 rather than Bixby's JUnit 6. These are not full Bixby build or Android lifecycle results.

## Test setup

- Device: connected Samsung SM-F776U, serial R3GL203AKSF, Android 17.
- Reference consumer: A2UI Android app, package `com.samsung.genuicraft`, consuming Maven coordinate `com.samsung.genuicraft:genuicraft:0.1.0`.
- Corpus: the 50 captured Perplexity responses in `tmp/bixby_perplexity_check/run_50_exact/responses.jsonl`; copied source wording into the test fixture without re-querying historical facts. IDs BXP-001 through BXP-050; 49 Markdown responses and one plain response.
- Gemma model: official `litert-community/gemma-4-E2B-it-litert-lm` generic package, 2,588,147,712 bytes, SHA-256 `181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c`. Both transfer and on-device file hashes were verified.
- The local fine-tuned v6 package was not used for acceptance: its own README states that its 128-token embedder export must be regenerated for the required prompt length.
- Gauss: configured endpoint and model used by the supplied PAC checkout; server reasoning is excluded from UI content.
- Acceptance uses the reference app's `genUiSdkOnlyNative=true` build, excluding all parent-owned JNI libraries. APK-entry hashes match the SDK/Maven closure. The final runtime log at 05:55:23 IST reports `Gemma4 backend=GPU; MTP=true; modelSupportsMtp=true; thinking=true`, alongside successful OpenCL initialization. This proves selected backend/capability/flags, not speculative-token acceptance rate. The final diagnostics window contained no fatal-exception or target-app ANR marker. See [ON_DEVICE_RUNTIME.md](ON_DEVICE_RUNTIME.md).

## Evidence and interpretation

The full live Gemma run used **v7**. The complete 106-file source manifests show exactly six changes between v7 and v9: heading semantics in `SourceBindings.kt` and five renderer files. Providers, prompts and JNI files are unchanged. V10 changes only the Gauss prompt from v9, with identical compiled classes. The current renderer/converter revalidated every exact successful v7 raw output. All 50 raw-input and saved-JSON provenance hashes match. Nine output documents changed, exclusively by adding 17 heading variants; no source content, value or action changed. See [the independent Gemma evidence audit](validation/20260918/gemma_evidence_audit.md).

The Gemma replay saved 221 screenshot/hierarchy pairs and checked 23 tables. None had unobserved columns under the header-or-representative-cell check; 20 used horizontal scrolling. Every case reached an observed vertical end, with at most 3 of 12 allowed swipes. Selected screenshots were reviewed for train cards, restored headings, wide tables and comparisons. V7's accessibility cache sometimes retained a prior case; the v9 replay explicitly refreshes it. Bounded text/column visibility is not a pixel-perfect or complete semantic-rendering proof.

The Gauss v10 JSON replay saved 231 screenshot/hierarchy pairs and checked 23 tables with 100 columns, with no unobserved columns or reached vertical limits. All 23 tables used horizontal scrolling. Each saved JSON is replayed without changing its bytes or calling a model.

| Full corpus | Median conversion | p95 conversion | First case, including initialization | Maximum sampled process PSS |
| --- | ---: | ---: | ---: | ---: |
| Gemma v7 GPU+MTP | 93.944 s | 208.825 s | 40.974 s | 4,599,314 KB (~4.39 GiB) |
| Gauss v10 | 11.851 s | 17.284 s | 7.407 s | 354,323 KB (~346 MiB) |

Gemma's longest conversion took 342.430 seconds. Its observed latency and memory use are material constraints for interactive on-device use. A 256-token thinking-budget experiment gave no consistent benefit, so the default remains 1,024 with thinking enabled. First and subsequent cases differ in content; their timings are not a controlled cold/warm comparison.

Host Gauss probes validate prompt syntax only. Android instrumentation invokes the actual AAR provider/converter, compiles output, performs content checks, replays the saved JSON through the AAR renderer, captures the visible viewport and a scrolled viewport, and saves every model attempt. No fallback is counted as success.

Mechanical content checks cover wording counts, citation markers, supplied links, numeric counts/signs/units and allowed actions. They cannot prove semantic equivalence or that every rendered row is visually correct. Renderer smoke checks verify a non-error screen with visible text; selected screenshots also receive manual review. Whole-process PSS includes the app and model and is sampled during conversion; it is not a model-only allocation measurement. Render wait time includes test synchronization and is not a frame-time benchmark.

The Gemma corpus is a sustained sequential run. At 04:40 IST the device reported thermal status 2 (skin sensor about 42 °C); thermal controls were not changed. Latencies are observed on this device under that workload, not a thermally controlled performance comparison.

## Development findings

- Initial Gauss smoke: 5/5 first-attempt conversions and renders; median 9,647 ms; maximum sampled app PSS 341,289 KB.
- Initial Gemma smoke: native GPU runtime initialized, but 0/2 valid conversions due to malformed Express. No silent CPU or text fallback.
- Revised direct-copy Gemma probes still changed numeric content, both with and without speculative decoding. Typed source bindings now preserve the original values while the model generates the layout. The SDK rejects missing, reordered, repeated or misused bindings; failed inference is not replaced by a deterministic layout. Reasoning and MTP remain enabled.
- Weather renderer corrected: preserves supplied row order, no longer infers today from a weekday without a year, and uses neutral text for undated forecasts. Before/after device screenshots confirm the dated BXP-001 forecast changed from an incorrect Today/Fri-Wed-Thu display to Wed-Sep-9/Wed-Thu-Fri.
- Action test diagnostics found that the reference policy intentionally blocks example.com. Device fixtures were changed to an allowed public URL, with a host callback that prevents external navigation.
- Independent review led to bounded transport reads, redirect rejection, lifecycle fixes, stricter content checks and deterministic source attribution.
- V9 renderer corrections preserve explicit table presentation, wrap cells, align column widths and remove the viewport-sized fade layer. Card routes are no longer overridden by the global table path; schedule/entity cards preserve all labels and values without field-count or text-truncation caps. Action labels are paired with the correct safe links.
- The v9 Gauss full run passed 49/50 (48 first attempt, one repaired) and correctly rejected BXP-029 after both attempts repeated a numeric heading as a table title. V10 adds an explicit single-occurrence heading rule to the Gauss prompt. The validator was not relaxed. The failed development run remains in the evidence.

## Bixby boundary

Bixby source integration is prepared in `C:/Users/anupk/Downloads/Bixby_18Sep`. It is not built or installed because the supplied checkout has unavailable dependencies. A2UI test-app results do not establish full Bixby runtime compatibility.

Live Perplexity streaming and final-only paths use structured provider provenance. Reopened history arrives as opaque legacy WebView state without the original answer/provider metadata, so that path retains the legacy renderer; the native overlay is cleared during restore. See the Bixby checkout's `docs/genuicraft_bixby_integration.md` for integration and configuration details.

Provider provenance is request-scoped: the existing `RendererEventData` does not carry immutable capsule identity for each source event. The parser filters sources by request ID. Mixed-capsule attribution within one request was not observed in the supplied fixture and is not established by these tests; adding a filter based on renderer type or the capsule-context boolean would discard legitimate events. This boundary needs a matching Bixby runtime trace before further narrowing.

## Later prompt study — not promoted

The later GPU+MTP prompt study evaluated **10 new prompt/input strategies**, excluding the baseline controls. The v11 scaffold candidate completed all 50 cases with **48 successes: 45 first attempt and 3 repaired** (BXP-013/029/036), **2 final failures**, and **zero fallbacks**. BXP-046 failed source-block-order validation; BXP-050 returned an incomplete A2UI envelope. The candidate is **rejected for delivery**. Production was restored to accepted v10 before the separate v14 renderer visual update described above. The accepted Gemma prompt and conversion path remain unchanged. Candidate research is retained under `experiments/gemma_prompt_study_20260918/candidate_source`; see [the prompt-study report](experiments/gemma_prompt_study_20260918/REPORT.md). No new 50/50 acceptance or full renderer replay is claimed for this candidate.

Restoration verified all 106 production source hashes and rebuilt an AAR byte-identical to accepted v10. The 285 JVM tests pass again, and all 25 Maven publication files match the Bixby copy. The freshly rebuilt/installed reference APK has SHA-256 `c69438ed88ab69189dbf7db8468c12889114bd565f747fb829ca0d13d8dfbc7a`; this differs from the historical APK above. Its installed-file hash, complete native-library closure, packaged baseline prompt and absence of the experimental scaffold class were checked. See [restoration evidence](experiments/gemma_prompt_study_20260918/results/accepted_v10_restoration.json).

A separate three-case native throughput replay used captured v11 candidate prompts with identical GPU+MTP/thinking runtime settings and benchmark counters enabled: **57.624, 30.775 and 31.992 decode tokens/sec**, median **31.992**. This measures LiteRT-reported generation throughput with reasoning enabled, not visible Express text speed or the retained baseline's end-to-end latency. Raw native counters, prompt hashes and timing distinctions are preserved in the prompt-study report. It is not a new conversion acceptance run or an MTP speedup comparison.

## Artifact evidence

Portable counts, per-case results, source/artifact hashes and runtime excerpts are collected under [validation/20260918](validation/20260918). Complete attempts, Express/JSON documents, screenshots, hierarchies and logs remain under `A2UI/tmp/genuicraft_20260918/`. V7's AAR is retained and rehashed; its older APK/test-APK hashes are recorded, but those binaries were not retained after subsequent builds. The older v7 run configuration lacks a corpus hash; every v9 replay request exactly matches the currently hashed 50-case fixture (`fc46aa381957bed206f9e0f53e28edbb21b5096ca093985ad184fda73faaea0a`).

The restored reference app also passed all four focused device checks (host action callback, exact source URL, literal text, and horizontal table access), with zero model calls. The test log is preserved in the prompt-study results.
