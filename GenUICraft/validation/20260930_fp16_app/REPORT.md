# Corrected FP16 in the GenUICraft test app

30 September 2026 · Flip8 SM-F776U · GenUICraft SDK 0.5.8

**Corrected FP16 is implemented, installed, and verified on GPU. For this trained model, use it with MTP enabled.** The three-case MTP-on pilot produced valid raw Express and rendered all three repaired outputs. FP32 remains the default; FP16 without MTP still has a serious output-quality failure.

## Observed speed

| MTP | FP32 native tok/s | FP16 native tok/s | Observed change | Median generation FP32 → FP16 |
|---|---|---|---|---|
| On | 10.84 | 16.41 | +51.4% | 52.87 s → 31.66 s |
| Off | 12.55 | 12.87 | +2.5% | 40.98 s → 50.49 s |

Native decode throughput is total native output tokens divided by their summed native decode durations. Generation time includes prefill and decode, excludes the separate warmup and UI rendering. Engine creation and model verification are excluded from these warm measurements.

**Thermal limitation:** the phone reported moderate throttling (status 2). GPU frequency ceilings changed between 160 and 342 MHz during the MTP measurements. This is one paired pilot, not a counterbalanced benchmark or a guarantee that FP16 alone causes the entire observed gain. Do not compare MTP-on directly against MTP-off as an MTP speedup measurement; their thermal conditions differ.

## MTP-on case details

| Case | Precision | Input / output tokens | Native tok/s | Generation s | First token s | Observed GPU ceiling MHz |
|---|---|---|---|---|---|---|
| BXP-001 | FP32 | 3174 / 301 | 8.57 | 52.87 | 17.80 | 160, 191 |
| BXP-003 | FP32 | 3394 / 561 | 10.93 | 61.78 | 10.50 | 160, 191, 222 |
| BXP-004 | FP32 | 3327 / 430 | 13.13 | 42.73 | 10.02 | 191 |
| BXP-001 | FP16 corrected | 3174 / 302 | 16.57 | 23.79 | 5.59 | 191, 222, 282 |
| BXP-003 | FP16 corrected | 3394 / 521 | 15.02 | 41.43 | 6.77 | 160, 222 |
| BXP-004 | FP16 corrected | 3327 / 432 | 18.34 | 31.66 | 8.12 | 191, 222 |

Drafter acceptance: **69.57% FP32**, **72.37% corrected FP16**. These native engine-close counters include each arm's warmup.

## Quality and rendering

| Precision | MTP | Raw SDK compile | After repair compile | v5.4 raw mean | v5.4 repaired mean |
|---|---|---|---|---|---|
| FP32 | Off | 3/3 | 3/3 | 86.72 | 93.60 |
| FP16_CORRECTED | Off | 1/3 | 3/3 | 29.96 | 84.42 |
| FP32 | On | 2/3 | 3/3 | 43.34 | 93.22 |
| FP16_CORRECTED | On | 3/3 | 3/3 | 67.05 | 93.18 |

The MTP-on repaired scores are **93.18 FP16 versus 93.22 FP32** on these three samples. These are source-to-UI representation scores, not a percentage of model accuracy or proof of checkpoint parity. SDK compilation and the Python v5.4 contract have different graph/reachability checks; their raw results are reported separately.

- **Weather (BXP-001):** the FP16/MTP screenshot shows the forecast values and weather cards.
- **Trains (BXP-003):** FP16/MTP shows all three train names, stations, times, durations and seating classes in compact cards. SDK recovery still resolves graph issues.
- **Baggage (BXP-004):** FP16/MTP shows the cabin/checked-baggage limits and charge sections. With MTP off, the same input loops through short identifiers until 2,048 tokens; repair returns a list of those identifiers, **not a useful answer**. Its 62.81 repaired v5.4 score demonstrates why a structural score alone is insufficient.
- All 6 FP16 repaired documents compile and their screenshots were captured. The MTP-off baggage case is a semantic failure despite rendering. One FP32/MTP-off UI-visibility probe timed out, although its saved screenshot visibly contains the correct baggage result.

No source-text fallback was used. Raw output, repaired output and screenshots are preserved separately. The native API labeled the token-limited baggage output COMPLETED; its actual 2,048-token length and incomplete envelope are retained.

## Use the installed option

1. Open **GenUICraft SDK → Trained E2B → Model setup → GPU precision → FP16 (corrected)**.
2. Keep **Settings → MTP acceleration** on. The device's saved preference has been set to FP16 corrected with MTP on.
3. GenUI pipeline and IR demo use the same shared precision preference and SDK inference path. The global app Settings also exposes the precision choice.

The prepared file `model-fp16-corrected.litertlm` and its `.fp16.json` manifest are installed beside the original model in the app's `files/sdk_models` directory. The original model remains available for FP32. The correction is pinned to this rank-64 export and an 8,192-token context; it is not automatic conversion of arbitrary models.

## What changed and validation

- Shared AAR: corrected target and MTP-drafter RoPE lookups; selective float intermediates for GPU Q/DQ; FP16 storage and matrix path retained. Original weight bytes are unchanged.
- Native adapter: guarded against exact SDK library Build IDs, including LiteRT JNI's embedded GPU compiler. Device logs confirm **197 corrected Q/DQ blocks without MTP**, **206 with MTP**, and zero rejected kernels. This is a scoped compatibility adapter, not an upstream LiteRT source rebuild.
- Test app: shared precision selection, prepared-model readiness, independent MTP control, truthful active-runtime labels and an experimental-mode explanation.
- Validation: **441 SDK unit tests + 17 focused app tests passed**; model graph/weight integrity independently verified; final app build and installation completed. Final APK changes after the measurements only clarify UI text; the benchmarked AAR/model/runtime are unchanged.

A final run through the normal **Generate UI** button confirmed `GPU+FP16_CORRECTED+MTP`: 3,174 input tokens, 302 output tokens, 20.19 native decode tokens/s, and 42.666 s total including initialization. This separate UI smoke run is excluded from the paired speed table. See [the final app screen](final_ui_render.png) and [installed APK verification](final_delivery.json).

The initial ABBA plan was curtailed after its first completed FP32/FP16 pair to keep the evaluation bounded. A second repeat was stopped before any measured case completed. The MTP-on pair was then run under a separate explicit single-pair protocol. This report uses **12 completed measured outputs**, three cases in each precision/MTP combination; warmups, smoke runs and the interrupted repeat are excluded.

## Evidence

[MTP-on measurements](mtp_comparison/REPORT.md) · [MTP-off measurements and curtailed protocol](comparison/REPORT.md) · [MTP-on quality](mtp_comparison/QUALITY.md) · [MTP-off quality](comparison/QUALITY.md) · [Model verification](model_verification.txt) · [Software hashes](software_manifest.json) · [Pilot data](pilot_summary.json) · [Implementation and preparation instructions](../../docs/fp16-corrected.md)
