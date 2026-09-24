# Fold7 comparison: newer trained E2B versus the previous checkpoint

**Decision:** the newer checkpoint is available in the GenUICraft SDK test screen, but it is not an improvement to promote as the default. In this five-case screen, neither checkpoint produced valid raw A2UI Express or a source-faithful answer. Generated-output repair made all ten outputs renderable, but it did not repair factual content.

## Models and test setup

The user-supplied newer file was `/sdcard/gemma4_e2b_a2ui_mobile.litertlm` on Galaxy Z Fold7 (`R3CY30QFWLP`, `SM-F966B`). It was copied into the app's external model directory as `gemma4_e2b_a2ui_mobile_20260924.litertlm`; source and installed copy both have SHA-256 `cf65377a9d6c9e6160b8657bd5f8849065e4c9d3b00b9388ab17f58301fab189`. The previous app model remained in place with SHA-256 `4675f37353e41c786a3e94f03ba4f64ad366a2e4e5d188bb92f5bef3922a75fe`. Both packages are 2,588,147,712 bytes. The file hashes prove these are different binaries; no training manifest was supplied to independently verify the epoch count.

The current Android test app was rebuilt and installed with GenUICraft AAR **0.5.0**. Its Android test APK was rebuilt and installed too. The SDK Bixby50 screen now selects the newer file as its trained E2B path and reports **Ready · 2.59 GB · GPU**; see [the selected-model screen](new_model_selected.png). The previous package is preserved. The app's separate GenUI/IR Demo provider settings were not changed.

Both matched runs used the same five captured Bixby50 responses, pinned trained-model prompt and source corpus (identical SHA-256 values in both [new](new/run_config.json) and [old](old/run_config.json) run configurations), temperature 0, 8,192 context tokens, 2,048 output-token limit, GPU, MTP enabled, and one model call per case. Source-text fallback was **disabled**. Generated-DSL repair was **enabled** and source-integrity enforcement was relaxed only to inspect and screenshot the model's output. Thus a renderable result is not counted as a faithful conversion. The benchmark now records `rawStrictValid`, repair kind, fallback use and source-fidelity diagnostics separately from rendering.

Device logs confirm `backend=GPU; MTP=true; MTPRequested=true; modelSupportsMtp=true` for both runs. Native MTP drafter success was **74.01%** for the newer run and **76.09%** for the previous run. These are aggregate end-of-run observations, not per-case acceptance rates; see [the four filtered runtime lines](runtime_log_evidence.txt).

## Result

| Gate or metric | Previous | Newer |
|---|---:|---:|
| Raw output strictly compiles | 0/5 | 0/5 |
| Generated-output repair + native render | 5/5 | 5/5 |
| Passes mechanical source-integrity checks | 0/5 | 0/5 |
| Source-text fallback | 0 | 0 |
| Total conversion time | 153.28 s | 164.99 s |
| Output tokens, all cases | 6,875 | 7,790 |
| Mean native decode rate | 49.72 token/s | 52.72 token/s |

The newer model decoded about **6% more tokens per second**, but generated **13% more tokens** and took **7.6% longer overall** for these five requests. It reached the 2,048-token cap in BXP-001, BXP-030 and BXP-032; the previous model reached the cap in BXP-030, BXP-032 and BXP-037. The first case in each run includes cold model initialization, so its latency is not a pure decode comparison. Case-level measurements are in [comparison.csv](comparison.csv), [new results](new/results.json) and [previous results](old/results.json).

| Case | Previous: time / output / decode | Newer: time / output / decode | Visual and content review |
|---|---:|---:|---|
| BXP-001 weather | 13.84 s / 288 / 48.15 t/s | 54.03 s / 2,049 / 46.09 t/s | The newer raw output loops on `forecast_...` until the cap and its repaired view drops the forecast prose and planning tips. Both show wrong `211` low temperature and `Fri, Sep 1`. [New](new/BXP-001/screen_scrolled.png) · [previous](old/BXP-001/screen_scrolled.png) |
| BXP-003 trains | 11.30 s / 438 / 46.87 t/s | 18.68 s / 812 / 51.06 t/s | Both render three useful-looking train cards but corrupt train numbers and times. Newer shows `120007`, `11:300`, `15:100`; previous shows `120007`, `11:000`, `15:5`. These cannot be trusted as timetables. [New](new/BXP-003/screen.png) · [previous](old/BXP-003/screen.png) |
| BXP-030 EPF/PPF/NPS | 42.59 s / 2,050 / 50.98 t/s | 35.19 s / 2,050 / 62.15 t/s | Neither preserves the comparison matrix. The newer repair gives loose bullets and changes the source's employee contribution from `12%` to `2%`; the previous repair leaves oversized disconnected headings. [New](new/BXP-030/screen.png) · [previous](old/BXP-030/screen.png) |
| BXP-032 insurance terms | 39.25 s / 2,049 / 55.73 t/s | 34.46 s / 2,050 / 63.44 t/s | Both retain a partial two-column table while losing the source's third effect column and corrupting text. The newer output loses an additional citation (`[3]`) and repeats a long numeric string in generated content. [New](new/BXP-032/screen.png) · [previous](old/BXP-032/screen.png) |
| BXP-037 school boards | 46.30 s / 2,050 / 46.86 t/s | 22.64 s / 829 / 40.85 t/s | The newer output retains more citation markers but collapses the CBSE/CISCE/IB mapping into a flat bullet list. The previous repair keeps entity cards but omits more citations and also changes text. [New](new/BXP-037/screen.png) · [previous](old/BXP-037/screen.png) |

The `strictValid` field in the existing per-case result schema means that the converter returned a compiled document **after the configured repair**. For direct model quality, use the new `rawStrictValid` field: it is false for all ten cases. `GENERATED_DSL_REPAIR` is recorded for every render; `usedFallback` is false. Each case records source-fidelity warnings, including missing citations and altered or missing wording or numbers. Representative raw outputs are preserved as `output.express` under each case directory, alongside `a2ui.json`, metrics, screenshots and source text.

This is a deliberately small, single-run comparison on historical captured Markdown, not an estimate for all Bixby50 cases or factual verification of those historical answers. Model output length and run order can affect timing. Only the Fold7 cover display was inspected. The newer model should remain an experimental option until raw syntax and exact source preservation improve; additional epochs alone did not resolve either problem here.
