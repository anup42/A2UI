# Rank-64 E2B evaluation — 27 September 2026

**Decision:** keep the supplied rank-64 LiteRT-LM package experimental. Its supplied rank-64 checkpoint predictions improve raw syntax on Golden35, but do not improve Bixby50 overall. On the connected Galaxy Z Flip8, the rank-64 package produced **0/5 raw strict outputs**, rendered **4/5 only after generated-output repair**, and passed **0/5 source-fidelity checks**. The previous LiteRT-LM package also produced **0/5 raw strict outputs** and **0/5 source-faithful results**, but rendered **5/5** and finished the matched five cases about **77 seconds sooner**. A renderable screen is not a faithful conversion.

**Metadata follow-up, 27 September:** new training/merge/export receipts bind the
tested rank-64 LiteRT file to the selected step-9,000 adapter and reveal a
reproducible QAT-versus-merge rounding mismatch. See the
[metadata audit](../../../training/reports/20260927_r64_metadata_audit/REPORT.md).
References below to missing manifests describe the inputs at the time of the
device comparison. No new device run is included in that follow-up.

## What was tested

The connected device was `R3GL203AKSF` (`SM-F776U`, Galaxy Z Flip8, Android SDK 37). The supplied file `C:\Users\anupk\Downloads\trained_model_data\gemma4_e2b_a2ui_mobile.litertlm` is 2,588,147,712 bytes, SHA-256 `c1794ee273ac13f9ddc3c24e1037e17367aabc6a810a3de3c836e1fa79efbf44`. It was copied without replacing the previous model to `sdk_models/gemma4_e2b_a2ui_mobile_r64_20260927.litertlm`; the device copy had the same hash. The previous `sdk_models/e2b_v10_w4.litertlm` has SHA-256 `4675f37353e41c786a3e94f03ba4f64ad366a2e4e5d188bb92f5bef3922a75fe`.

The app and Android test APK were assembled from the current checkout (Gradle found their build tasks up-to-date) and replacement-installed before the matched comparison. An earlier installed test APK ignored the requested MTP/source-fallback flags; its five-case run was **discarded**. The current runs record the intended flags and the shared prompt/corpus hashes. The comparison script verifies the **same five source records, per-case rendered prompt hashes, corpus, runtime settings, and device build**: [per-case comparison](device_comparison.csv), [verification script](summarize_device.py), [previous run configuration](device/old_mtp_on/run_config.json), [rank-64 run configuration](device/r64_mtp_on/run_config.json).

Both matched runs used BXP-001, 003, 030, 032, and 037; GPU; MTP requested and actually enabled; temperature 0; 8,192 context; 2,048 output-token cap; one provider call per case; generated-DSL repair enabled; source fallback **disabled**. Source-integrity enforcement was relaxed solely so the generated result could be inspected and screenshot. The native log confirms MTP drafter success rates of **76.09%** for the previous model and **75.36%** for rank 64; these are aggregate session rates, not case-level values: [old runtime evidence](device/old_runtime_evidence.txt), [rank-64 runtime evidence](device/r64_runtime_evidence.txt).

## Matched on-device result

| Measure | Previous LiteRT-LM | Supplied rank-64 LiteRT-LM |
|---|---:|---:|
| Raw output strictly compiles | 0/5 | 0/5 |
| Rendered after generated-output repair | 5/5 | 4/5 |
| Passes SDK mechanical source-fidelity checks | 0/5 | 0/5 |
| Source-text fallback | 0 | 0 |
| Output tokens | 6,875 | 8,742 |
| Sum of conversion times | 301.75 s | 378.63 s |
| Token-weighted native decode | 24.47 token/s | 24.76 token/s |
| Outputs at/near the 2,048-token cap | 3/5 | 4/5 |

The rank-64 package generated **27% more tokens**, taking **25% longer** overall despite almost the same token-weighted decode speed. These were single sequential runs, rank 64 then previous; device heat and case order make small throughput differences inconclusive. The extra generation and repair failures, rather than decode speed, explain the practical latency and quality regression. Android's `strictValid` in the per-case JSON is **postrepair**; `rawStrictValid` and the independent Python scorer both report 0/5 raw validity. The instrumentation test exits with one assertion failure for the rank-64 run because BXP-030 did not render; all five cases completed and the artifacts were saved.

| Case | Previous: time / output | Rank 64: time / output | Review |
|---|---:|---:|---|
| BXP-001 weather | 10.35 s / 288 | 66.35 s / 2,048 | Previous screen shows a forecast but corrupts a low temperature to `211` and loses citations. Rank 64 loops through repeated nested tabs and empty strings; repair shows only “High” and “Low”, with no forecast. [Previous](device/old_mtp_on/BXP-001/screen.png) · [rank 64](device/r64_mtp_on/BXP-001/screen.png) |
| BXP-003 trains | 19.31 s / 438 | 32.20 s / 543 | Both produce attractive train cards but change train numbers, station abbreviations, and times. The source's `12007`/`10:35–10:45` becomes `120007`/`10:35–1004` in the previous screen and `10007`/`1:355–0:50:4` with rank 64. Neither is safe as a timetable. [Previous](device/old_mtp_on/BXP-003/screen.png) · [rank 64](device/r64_mtp_on/BXP-003/screen.png) |
| BXP-030 EPF/PPF/NPS | 94.26 s / 2,050 | 82.94 s / 2,051 | Both reach the output cap. Previous repair renders an oversized isolated statement instead of the comparison. Rank-64 repair cannot produce a document, so nothing renders. [Previous](device/old_mtp_on/BXP-030/screen.png) · [rank-64 result](device/r64_mtp_on/BXP-030/result.json) |
| BXP-032 insurance terms | 85.79 s / 2,049 | 91.87 s / 2,050 | Rank-64 repair shows prose but loses the source's key-term table and six citation markers. Both are flagged for changed or missing numbers. [Rank 64](device/r64_mtp_on/BXP-032/screen.png) |
| BXP-037 school boards | 92.05 s / 2,050 | 105.28 s / 2,050 | Both require repair and omit/change cited details; neither is source-faithful. [Previous](device/old_mtp_on/BXP-037/screen.png) · [rank 64](device/r64_mtp_on/BXP-037/screen.png) |

The raw outputs, repaired A2UI, source records, screenshots, and independent scored reports are in [previous run](device/old_mtp_on/) and [rank-64 run](device/r64_mtp_on/). The independent reports verify 5/5 prompt hashes against the pinned scaffold and 0/5 raw strict outputs: [previous scored report](device/old_mtp_on/REPORT.md), [rank-64 scored report](device/r64_mtp_on/REPORT.md).

## MTP isolation

The same rank-64 LiteRT-LM package was then run on GPU **without MTP** for BXP-001 and BXP-003, with the same prompt, corpus, and no source fallback. The runtime log confirms `MTP=false`: [evidence](device/r64_mtp_off_runtime_evidence.txt). Both outputs still failed raw strict compilation and source-fidelity checks, then rendered only after generated-output repair. BXP-001 again hit 2,048 tokens (136.41 s, 15.57 token/s) and repaired into a list of field/variable names rather than forecast values; [see its screen](device/r64_mtp_off/BXP-001/screen.png). BXP-003 still corrupted train numbers/times (45.24 s, 12.87 token/s). [MTP-off run](device/r64_mtp_off/). MTP improves speed here but is **not sufficient to explain the rank-64 package's broken content**.

## Supplied rank-32 versus rank-64 checkpoint predictions

The six `e2b_runD_predictions/*_scored.jsonl` files were analyzed as supplied, with **no repair or fallback**. Rank-32 and rank-64 rows pair exactly by ID, source hash, expected hash, rendered-prompt hash, and generation-policy hash; [reproducible analysis](analyze_predictions.py), [aggregate JSON with file hashes](prediction_summary.json), [117 paired case rows](paired_predictions.csv). These predictions ran on `cuda:0`, not the phone, and are not a mobile benchmark. Golden32 has 32 occurrences but only 31 unique sources.

| Test set | Rank 32 raw valid | Rank 64 raw valid | Rank 32 fully root-reachable | Rank 64 fully root-reachable | Rank 32 / rank 64 at output cap | Mean v5.4 reward, rank 32 / rank 64 |
|---|---:|---:|---:|---:|---:|---:|
| Bixby50 (source-only) | 30/50 | **29/50** | 11/50 | **6/50** | 13 / **7** | 37.80 / **32.37** |
| Golden32 (31 unique sources) | 26/32 | **28/32** | 4/32 | **4/32** | 2 / **1** | 44.29 / **42.67** |
| Golden35 | 10/35 | **18/35** | 1/35 | **4/35** | 16 / **13** | 12.43 / **28.09** |

Rank 64 makes a substantial syntax gain on Golden35 but **not** on Bixby50, the relevant response style. Fewer output-cap events do not mean reliable completion: on Bixby50, 14/21 invalid rank-64 outputs stopped before the cap but still had an empty/missing envelope, dangling reference, duplicate ID, or malformed expression. Among its 29 syntactically valid Bixby50 predictions, **23 have unreachable components**, so even a successful parser can leave source content invisible. Its mean Bixby50 reward falls by 5.43 points despite slightly fewer capped outputs.

Representative evidence from the supplied predictions:

- **BXP-003:** the rank-64 checkpoint prediction preserves source train values in its state and reaches the full graph, but invents a `View Arrival Times` button without a supplied action target. The mobile rank-64 export instead corrupts the values. This is a useful syntax improvement in the checkpoint prediction, not evidence of a trustworthy shipped result.
- **BXP-008:** rank 64 emits only `</a2ui>` in five tokens. Rank 32 also fails, beginning with stray text before its opening tag.
- **BXP-030:** rank 32 repeats component IDs until the 2,048-token cap; rank 64 stops earlier but reuses component ID `h`, so both are invalid.
- **BXP-032:** rank 64 closes a parseable document, but only 75% of its components are reachable and its scored content-unit fidelity is 0.39. On-device repair loses the table.
- **Golden35 q_000005:** rank 64 changes a capped rank-32 output into a parseable laptop UI, but 14% of components remain unreachable; number/date/unit fidelity is 0.22 and action/source-link fidelity 0.17 in the supplied scorer.
- **Golden35 q_000020:** rank 64 regresses from a parseable Kyoto itinerary to repeated “The response is ready for implementation.” text and an EOS before `</a2ui>`.
- **Golden35 q_000023:** rank 64 fixes an unsupported `EmailPreview` property and parses, but repeats the sender name and drops the source's send action. Strict validity alone misses that loss.

The supplied v5.4 score is useful for structural triage, but a high `content_coverage` can coexist with disconnected components, missing actions, or corrupt facts. Golden exact/semantic reference matches are 0 for both ranks; because alternate layouts are possible, this is not by itself a rejection criterion. The source-only Bixby set has no reference IR. Neither the prediction files nor the LiteRT-LM binary came with a checkpoint/export manifest, so the file names and user description identify rank 64, but these artifacts cannot independently prove the binary was exported from the exact scored checkpoint.

## Training and export issues indicated by the evidence

1. **Graph construction remains unstable.** Missing child IDs, duplicate IDs, unreachable components, and premature envelope closure persist across all sets. The rank-64 Golden35 syntax gain has not translated into a reliable complete graph. Add a graph-reachability and exact-source gate to checkpoint selection, and evaluate on the raw output before general repair.
2. **Long complex responses are still hard.** Rank 64 hits the 2,048-token cap on 13/35 Golden35 cases; every capped prediction is invalid. In Golden35, invalid rank-64 sources have median 3,727 characters versus 2,181 for valid ones. The prior [complete v10 dataset audit](../../../training/reports/v10_full_audit_20260926/REPORT.md) found that the deployed prompt alone raises full-sequence length enough to exceed a 4,096-token budget for many raw rows. The actual prepared-row/sequence-length receipt for this checkpoint is missing, so verify it before assigning causality. Train and serve against a consistent, adequately sized context/output policy; do not truncate target IR.
3. **Response distribution and supervision fidelity need work.** The v10 audit found nearly all training sources have synthetic media/action/URL scaffolding, while the frozen Bixby50 responses largely have numeric citations and no such scaffolding. It also confirmed target omissions and URL-binding gaps. This can teach unsupported controls and missing details. Build a separately held-out, production-style development cohort and regenerate defective Stage 3 targets through the prompt/pipeline; keep Bixby50 out of training.
4. **Checkpoint-to-LiteRT parity is the immediate blocker.** The rank-64 offline predictions and mobile runs have the **same per-case rendered prompt SHA-256 and input-token count in all five matched cases**, yet BXP-001 changes from a 312-token parseable prediction to a 2,048-token malformed mobile output. Disabling MTP does not fix it. Test the exact checkpoint, merged weights, quantized LiteRT export, and Android runtime on the same tokens with recorded hashes and generation options. Until that chain is verified, do not attribute the gap to LoRA rank, quantization, or a specific runtime bug.

This is a five-case single-run **device sample**, not a 50-case device pass or an independent factual audit of the original Bixby answers. The supplied offline files cover all 50 Bixby examples plus Golden32/35, but they are separate CUDA predictions. No user model or app default was overwritten; the rank-64 package remains installed under its distinct filename for follow-up testing.
