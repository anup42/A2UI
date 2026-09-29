"""Rebuild the bounded app pilot report from preserved first-pair measurements."""
import html
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "|" + "---|" * len(headers)] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


data = []
summaries = []
for name, mtp in (("comparison", False), ("mtp_comparison", True)):
    folder = ROOT / name
    quality = read(folder / "quality_scores.json")
    telemetry = [json.loads(line) for line in (folder / "telemetry.jsonl").read_text().splitlines()]
    for precision in ("FP32", "FP16_CORRECTED"):
        label = f"r01_{precision.lower()}_mtp_{'on' if mtp else 'off'}"
        run = folder / "runs" / label
        rows = read(run / "results.json")
        assert len(rows) == 3 and all(row["runtime"] is not None for row in rows)
        rate = sum(row["outputTokens"] for row in rows) / sum(
            row["outputTokens"] / row["decodeTokensPerSecond"] for row in rows)
        q = next(s for s in quality["summary"] if s["mtp"] == mtp and s["precision"] == precision)
        summaries.append({"mtp": mtp, "precision": precision, "n": 3, "nativeDecodeTokensPerSecond": rate,
                          "medianProviderSeconds": statistics.median(r["providerElapsedMs"] for r in rows) / 1000,
                          "totalProviderSeconds": sum(r["providerElapsedMs"] for r in rows) / 1000,
                          "medianTtftSeconds": statistics.median(r["timeToFirstTokenSeconds"] for r in rows),
                          "rawSdkStrictValid": sum(r["rawStrictValid"] for r in rows),
                          "repairedSdkStrictValid": sum(r["repairedStrictValid"] for r in rows),
                          "renderSmokeValid": sum(r["sdkRenderSmokeValid"] for r in rows),
                          "rawV54Mean": q["rawMean"], "repairedV54Mean": q["repairedMean"],
                          "sessionMetrics": read(run / "session_metrics.json")})
        for r in rows:
            samples = [t for t in telemetry if t["label"] == label and
                       r["deviceBefore"]["epochMs"] <= t["epochMs"] <= r["deviceAfter"]["epochMs"]]
            clocks = [t["gpuMaxClockHz"] / 1e6 for t in samples if t.get("gpuMaxClockHz")]
            data.append({**r, "mtp": mtp, "precision": precision,
                         "path": (run / r["id"]).relative_to(ROOT).as_posix(),
                         "gpuCeilingMhz": sorted(set(clocks))})

improvements = []
for mtp in (False, True):
    a = next(s for s in summaries if s["mtp"] == mtp and s["precision"] == "FP32")
    b = next(s for s in summaries if s["mtp"] == mtp and s["precision"] == "FP16_CORRECTED")
    improvements.append({"mtp": mtp, "decodePercent": (b["nativeDecodeTokensPerSecond"] / a["nativeDecodeTokensPerSecond"] - 1) * 100,
                         "medianProviderTimeReductionPercent": (1 - b["medianProviderSeconds"] / a["medianProviderSeconds"]) * 100})
(ROOT / "pilot_summary.json").write_text(json.dumps({"summaries": summaries, "improvements": improvements,
                                                     "observations": data}, indent=2) + "\n", encoding="utf-8")

lines = ["# Corrected FP16 in the GenUICraft test app", "", "30 September 2026 · Flip8 SM-F776U · GenUICraft SDK 0.5.8", "",
         "**Corrected FP16 is implemented, installed, and verified on GPU. For this trained model, use it with MTP enabled.** "
         "The three-case MTP-on pilot produced valid raw Express and rendered all three repaired outputs. "
         "FP32 remains the default; FP16 without MTP still has a serious output-quality failure.", "",
         "## Observed speed", "",
         table(["MTP", "FP32 native tok/s", "FP16 native tok/s", "Observed change", "Median generation FP32 → FP16"],
               [["On" if mtp else "Off",
                 f'{next(s for s in summaries if s["mtp"] == mtp and s["precision"] == "FP32")["nativeDecodeTokensPerSecond"]:.2f}',
                 f'{next(s for s in summaries if s["mtp"] == mtp and s["precision"] == "FP16_CORRECTED")["nativeDecodeTokensPerSecond"]:.2f}',
                 f'{next(i for i in improvements if i["mtp"] == mtp)["decodePercent"]:+.1f}%',
                 f'{next(s for s in summaries if s["mtp"] == mtp and s["precision"] == "FP32")["medianProviderSeconds"]:.2f} s → '
                 f'{next(s for s in summaries if s["mtp"] == mtp and s["precision"] == "FP16_CORRECTED")["medianProviderSeconds"]:.2f} s']
                for mtp in (True, False)]), "",
         "Native decode throughput is total native output tokens divided by their summed native decode durations. "
         "Generation time includes prefill and decode, excludes the separate warmup and UI rendering. "
         "Engine creation and model verification are excluded from these warm measurements.", "",
         "**Thermal limitation:** the phone reported moderate throttling (status 2). GPU frequency ceilings changed between "
         "160 and 342 MHz during the MTP measurements. This is one paired pilot, not a counterbalanced benchmark or a "
         "guarantee that FP16 alone causes the entire observed gain. Do not compare MTP-on directly against MTP-off as an "
         "MTP speedup measurement; their thermal conditions differ.", "",
         "## MTP-on case details", "",
         table(["Case", "Precision", "Input / output tokens", "Native tok/s", "Generation s", "First token s", "Observed GPU ceiling MHz"],
               [[r["id"], "FP32" if r["precision"] == "FP32" else "FP16 corrected", f'{r["inputTokens"]} / {r["outputTokens"]}',
                 f'{r["decodeTokensPerSecond"]:.2f}', f'{r["providerElapsedMs"] / 1000:.2f}',
                 f'{r["timeToFirstTokenSeconds"]:.2f}', ", ".join(f"{v:g}" for v in r["gpuCeilingMhz"])] for r in data if r["mtp"]]), "",
         "Drafter acceptance: **69.57% FP32**, **72.37% corrected FP16**. These native engine-close counters include each arm's warmup.", "",
         "## Quality and rendering", "",
         table(["Precision", "MTP", "Raw SDK compile", "After repair compile", "v5.4 raw mean", "v5.4 repaired mean"],
               [[s["precision"], "On" if s["mtp"] else "Off", f'{s["rawSdkStrictValid"]}/3', f'{s["repairedSdkStrictValid"]}/3',
                 f'{s["rawV54Mean"]:.2f}', f'{s["repairedV54Mean"]:.2f}'] for s in summaries]), "",
         "The MTP-on repaired scores are **93.18 FP16 versus 93.22 FP32** on these three samples. "
         "These are source-to-UI representation scores, not a percentage of model accuracy or proof of checkpoint parity. "
         "SDK compilation and the Python v5.4 contract have different graph/reachability checks; their raw results are reported separately.", "",
         "- **Weather (BXP-001):** the FP16/MTP screenshot shows the forecast values and weather cards.",
         "- **Trains (BXP-003):** FP16/MTP shows all three train names, stations, times, durations and seating classes in compact cards. SDK recovery still resolves graph issues.",
         "- **Baggage (BXP-004):** FP16/MTP shows the cabin/checked-baggage limits and charge sections. With MTP off, the same input loops through short identifiers until 2,048 tokens; repair returns a list of those identifiers, **not a useful answer**. Its 62.81 repaired v5.4 score demonstrates why a structural score alone is insufficient.",
         "- All 6 FP16 repaired documents compile and their screenshots were captured. The MTP-off baggage case is a semantic failure despite rendering. One FP32/MTP-off UI-visibility probe timed out, although its saved screenshot visibly contains the correct baggage result.", "",
         "No source-text fallback was used. Raw output, repaired output and screenshots are preserved separately. "
         "The native API labeled the token-limited baggage output COMPLETED; its actual 2,048-token length and incomplete envelope are retained.", "",
         "## Use the installed option", "",
         "1. Open **GenUICraft SDK → Trained E2B → Model setup → GPU precision → FP16 (corrected)**.",
         "2. Keep **Settings → MTP acceleration** on. The device's saved preference has been set to FP16 corrected with MTP on.",
         "3. GenUI pipeline and IR demo use the same shared precision preference and SDK inference path. The global app Settings also exposes the precision choice.", "",
         "The prepared file `model-fp16-corrected.litertlm` and its `.fp16.json` manifest are installed beside the original "
         "model in the app's `files/sdk_models` directory. The original model remains available for FP32. "
         "The correction is pinned to this rank-64 export and an 8,192-token context; it is not automatic conversion of arbitrary models.", "",
         "## What changed and validation", "",
         "- Shared AAR: corrected target and MTP-drafter RoPE lookups; selective float intermediates for GPU Q/DQ; FP16 storage and matrix path retained. Original weight bytes are unchanged.",
         "- Native adapter: guarded against exact SDK library Build IDs, including LiteRT JNI's embedded GPU compiler. Device logs confirm **197 corrected Q/DQ blocks without MTP**, **206 with MTP**, and zero rejected kernels. This is a scoped compatibility adapter, not an upstream LiteRT source rebuild.",
         "- Test app: shared precision selection, prepared-model readiness, independent MTP control, truthful active-runtime labels and an experimental-mode explanation.",
         "- Validation: **441 SDK unit tests + 17 focused app tests passed**; model graph/weight integrity independently verified; final app build and installation completed. Final APK changes after the measurements only clarify UI text; the benchmarked AAR/model/runtime are unchanged.", "",
         "A final run through the normal **Generate UI** button confirmed `GPU+FP16_CORRECTED+MTP`: "
         "3,174 input tokens, 302 output tokens, 20.19 native decode tokens/s, and 42.666 s total including initialization. "
         "This separate UI smoke run is excluded from the paired speed table. See [the final app screen](final_ui_render.png) "
         "and [installed APK verification](final_delivery.json).", "",
         "The initial ABBA plan was curtailed after its first completed FP32/FP16 pair to keep the evaluation bounded. "
         "A second repeat was stopped before any measured case completed. The MTP-on pair was then run under a separate explicit single-pair protocol. "
         "This report uses **12 completed measured outputs**, three cases in each precision/MTP combination; warmups, smoke runs and the interrupted repeat are excluded.", "",
         "## Evidence", "",
         "[MTP-on measurements](mtp_comparison/REPORT.md) · [MTP-off measurements and curtailed protocol](comparison/REPORT.md) · "
         "[MTP-on quality](mtp_comparison/QUALITY.md) · [MTP-off quality](comparison/QUALITY.md) · "
         "[Model verification](model_verification.txt) · [Software hashes](software_manifest.json) · [Pilot data](pilot_summary.json) · "
         "[Implementation and preparation instructions](../../docs/fp16-corrected.md)", ""]
(ROOT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

cards = []
for case in ("BXP-001", "BXP-003", "BXP-004"):
    items = []
    for r in data:
        if not r["mtp"] or r["id"] != case:
            continue
        p = r["path"]
        items.append(f'<article><h3>{html.escape(r["precision"])}</h3><p>{r["decodeTokensPerSecond"]:.2f} tok/s · '
                     f'{r["providerElapsedMs"] / 1000:.2f} s · {r["outputTokens"]} output tokens</p>'
                     f'<a href="{p}/attempt_1_raw.express">Raw output</a> · <a href="{p}/repaired.express">Repaired output</a>'
                     f'<img loading="lazy" src="{p}/sdk_render.png" alt="{case} {r["precision"]} on-device rendering"></article>')
    cards.append(f'<section><h2>{case} · MTP on</h2><div class="grid">{"".join(items)}</div></section>')
page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GenUICraft corrected FP16 app pilot</title><style>
body{margin:0;background:#f4f7fb;color:#16253d;font:16px/1.55 system-ui}main{max-width:1160px;margin:auto;padding:28px}
header,section{background:white;border:1px solid #dce3ed;border-radius:16px;padding:24px;margin-bottom:20px}
h1{font-size:30px;margin:0 0 12px}h2{font-size:22px}h3{font-size:16px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}
article{min-width:0}img{width:100%;max-width:360px;display:block;margin-top:15px;border:1px solid #dce3ed;border-radius:12px}
.note{background:#fff3d5;padding:16px;border-radius:12px}.number{font-size:32px;font-weight:700}a{color:#175bc4}
@media(max-width:650px){.grid{grid-template-columns:1fr}main{padding:12px}}</style><main>
<header><h1>Corrected FP16 in the test app</h1><p>Flip8 · trained rank-64 E2B · GenUICraft 0.5.8 · three-case pilot</p>
<div class="grid"><div><div class="number">10.84 → 16.41 tok/s</div><p>MTP on: observed +51% native decode throughput</p></div>
<div><div class="number">52.87 → 31.66 s</div><p>MTP on: median warm generation time</p></div></div>
<p class="note">The phone was thermally throttled and GPU clocks differed. These are observed pilot results, not a guaranteed speedup.
Without MTP, FP16 gained only about 3% in decode speed and failed the baggage case semantically. Keep MTP on for this experimental option.</p>
<p>MTP-on repaired v5.4: FP32 <b>93.22</b>, FP16 <b>93.18</b>. A score is not a percentage of factual accuracy.</p>
<a href="REPORT.md">Complete report and methodology</a> · <a href="pilot_summary.json">All measurements</a></header>
'''
(ROOT / "index.html").write_text(page + "".join(cards) + "</main></html>\n", encoding="utf-8")
print(json.dumps(improvements, indent=2))
