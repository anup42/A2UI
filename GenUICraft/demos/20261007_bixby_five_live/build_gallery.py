"""Package the five visually reviewed live Bixby exports and their evidence."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent
CASES = [
    ("01_flights", "Flights", "Bixby_GenUICraft_Flights", "Airline logos, fare, duration and departure/arrival times together.",
     ["Air India Express", "Akasa Air", "IndiGo", "6,613", "7,064", "7,473", "06:05", "08:40", "07:25", "10:00", "10:10", "12:40"]),
    ("02_trains", "Trains", "Bixby_GenUICraft_Trains", "Four compact rail services with fares and journey times visible.",
     ["20172", "12002", "12280", "14212", "95", "340", "181", "90", "14:40", "06:00", "06:55", "17:40"]),
    ("03_phones_cards", "Phone comparison", "Bixby_GenUICraft_PhoneComparison", "One card per phone with labeled price, battery, camera and update support.",
     ["Samsung Galaxy S25", "iPhone 16", "Google Pixel 9", "62,999", "69,900", "79,999", "4,000", "3,561", "4,700"]),
    ("04_climate", "Seasonal weather", "Bixby_GenUICraft_WeatherComparison", "Month, high/low temperatures, rainfall and conditions within the phone width.",
     ["January", "April", "July", "October", "29", "16", "33", "35", "20", "23", "127", "168"]),
    ("05_restaurants", "Restaurant comparison", "Bixby_GenUICraft_RestaurantComparison", "Cost, opening hours and address grouped with each restaurant.",
     ["MTR", "Vidyarthi Bhavan", "Brahmins Coffee Bar", "200", "400", "150", "300", "7:30am", "9:00pm", "6:30am", "11:30am", "Gandhi Bazaar", "Not available"]),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ui_text(path: Path) -> str:
    return " ".join(n.get("text", "") + " " + n.get("content-desc", "")
                    for n in ET.parse(path).iter("node"))


def main() -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    (ROOT / "review").mkdir(exist_ok=True)
    records = []
    proof = []
    for case, title, stem, benefit, values in CASES:
        source = ROOT / "evidence" / case
        metadata = json.loads((ROOT / f"{stem}.metadata.json").read_text(encoding="utf-8"))
        video = ROOT / f"{stem}.mp4"
        assert metadata["output"]["strict_decode"] == "passed"
        assert metadata["output"]["sha256"] == sha256(video)
        assert (metadata["output"]["width"], metadata["output"]["height"]) == (1080, 1080)
        runtime = (source / "runtime_evidence.txt").read_text(encoding="utf-8")
        match = re.search(r"conversion succeeded: attempts=(\d+), repair=(\w+), warnings=(\d+)", runtime)
        assert match, f"No successful native conversion evidence for {case}"
        assert match[2] in {"NONE", "GENERATED_DSL_REPAIR"}, "Source fallback is not a model demo"
        original, generated = ui_text(source / "bixby_original.xml"), ui_text(source / "genuicraft_final.xml")
        checks = [{"value": value, "in_original_answer_tree": value in original,
                   "in_generated_answer_tree": value in generated} for value in values]
        assert all(check["in_original_answer_tree"] and check["in_generated_answer_tree"] for check in checks), case
        green = next(a for a in metadata["annotations"] if a["accent"] == "#198E44")
        # Annotation times already reflect validated raw-to-export edits.
        moment = (green["output_start"] + green["output_end"]) / 2
        poster = ROOT / "review" / f"{case}.jpg"
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", str(moment), "-i", str(video),
                        "-frames:v", "1", "-q:v", "2", str(poster)], check=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        proof.extend([f"\n[{case}]", *[line for line in runtime.splitlines()
                      if "conversion succeeded" in line or "policy=genuicraft" in line or "Gemma4 backend=" in line]])
        records.append({"id": case, "title": title, "video": video.name,
                        "poster": str(poster.relative_to(ROOT)).replace("\\", "/"), "benefit": benefit,
                        "query": (source / "query.txt").read_text(encoding="utf-8"),
                        "duration_seconds": metadata["output"]["duration_seconds"], "bytes": video.stat().st_size,
                        "sha256": metadata["output"]["sha256"], "source_sha256": metadata["source"]["sha256"],
                        "conversion": {"attempts": int(match[1]), "repair": match[2], "warnings": int(match[3])},
                        "key_value_presence_checks": checks,
                        "visible_limitations": (["The renderer uses decorative placeholders; these are not restaurant photos.",
                          "Signature dish is retained in the accessible row data but is not displayed as a visible field.",
                          "The large placeholder and truncated long address still need renderer refinement."]
                          if case == "05_restaurants" else
                          ["The generated section title says Weather Forecast although the request asks for typical seasonal climate."]
                          if case == "04_climate" else [])})
    (ROOT / "runtime_evidence.txt").write_text("\n".join(proof), encoding="utf-8")
    manifest = {"date": "2026-10-07", "app": "Bixby 5.0.10.50", "device_model": "SM-F776U",
                "inference": "trained E2B, GPU, FP16_CORRECTED, MTP enabled",
                "model_sha256": "7f01bdf1c6ba9bdf658e57c75001fc35a42edad88238bc5dab0a5f7dcf3de373",
                "pipeline": "Bixby answer -> on-device A2UI Express -> shared SDK repair/compiler -> renderer",
                "evaluation_scope": "Qualitative presentation comparison; selected key values checked against the same Bixby answer. Not an accuracy benchmark or independent fact verification.",
                "records": records}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    cards = []
    for record in records:
        e = html.escape
        cards.append(f'''<article id="{record['id']}"><div class="heading"><h2>{e(record['title'])}</h2><span>{record['duration_seconds']:.1f} s</span></div>
<p>{e(record['benefit'])}</p><video controls playsinline preload="metadata" poster="{record['poster']}" aria-label="{e(record['title'])} live Bixby demo"><source src="{record['video']}" type="video/mp4"></video>
<div class="links"><a href="{record['video']}" download>Download MP4</a><a href="{Path(record['video']).stem}.metadata.json">Capture and editing evidence</a></div>
<details><summary>Query and observed result</summary><p>{e(record['query'])}</p><p>Native conversion: {e(record['conversion']['repair'])}; one model attempt. Key-value checks: {len(record['key_value_presence_checks'])}/{len(record['key_value_presence_checks'])} present in both answer trees.</p>
{''.join('<p class="note">'+e(x)+'</p>' for x in record['visible_limitations'])}</details></article>''')
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bixby × GenUICraft — Five live demos</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f7f8fa;color:#182333;font:16px/1.6 system-ui,Segoe UI,sans-serif}main{max-width:1260px;margin:auto;padding:36px 24px}header{margin-bottom:28px}h1{font-size:clamp(26px,4vw,40px);line-height:1.2;letter-spacing:-1px;margin:10px 0 14px}header p{max-width:870px;color:#556270}.eyebrow{color:#1967c8;font-weight:700;letter-spacing:1px;font-size:13px}.badges{display:flex;flex-wrap:wrap;gap:8px}.badges span{background:#e9effa;color:#174e96;border-radius:30px;padding:5px 13px;font-size:13px}nav{display:flex;flex-wrap:wrap;gap:14px;margin:24px 0}a{color:#1967c8;text-underline-offset:3px}section{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}article{background:white;border:1px solid #dde3ec;border-radius:20px;padding:20px;scroll-margin-top:18px}.heading{display:flex;align-items:center;justify-content:space-between;gap:12px}h2{margin:0;font-size:22px}.heading span{color:#556270;font-size:14px}article>p{min-height:52px;color:#556270;margin:12px 0}video{width:100%;display:block;aspect-ratio:1;border-radius:12px;background:#f7f8fa}.links{display:flex;justify-content:space-between;flex-wrap:wrap;gap:12px;font-size:14px;margin-top:14px}details{font-size:14px;margin-top:16px;border-top:1px solid #e9edf3;padding-top:12px}summary{cursor:pointer;color:#556270}.note{color:#765826}footer{margin:30px 0;color:#556270;font-size:14px}@media(max-width:760px){section{grid-template-columns:1fr}main{padding:24px 14px}article>p{min-height:0}}</style></head><body><main><header><div class="eyebrow">BIXBY × GENUICRAFT</div><h1>Five live UI comparisons</h1><p>Actual Bixby requests on the connected Samsung device. Watch A2UI Express stream, see the generated UI, and switch to the original answer in the same conversation.</p><div class="badges"><span>Trained E2B on GPU</span><span>Corrected FP16 + MTP</span><span>Live device footage</span><span>1080p MP4 · silent</span></div><nav>'''
    page += "".join(f'<a href="#{r["id"]}">{html.escape(r["title"])}</a>' for r in records)
    page += '</nav></header><section>' + "".join(cards) + '</section><footer>Videos have captions and arrows. Verified empty view-switch waits are cut; only the longer flight Express stream is accelerated, with its spinner preserved at normal speed. These are presentation demos, not latency measurements. <a href="REPORT.md">Read the review and test report</a>.</footer></main><script>document.querySelectorAll("video").forEach(v=>v.addEventListener("play",()=>document.querySelectorAll("video").forEach(o=>{if(o!==v)o.pause()})));</script></body></html>'
    (ROOT / "index.html").write_text(page, encoding="utf-8")

    table = "\n".join(f"| {r['title']} | [{r['video']}]({r['video']}) | {r['benefit']} | {r['conversion']['repair']} | {r['conversion']['warnings']} |" for r in records)
    queries = "\n\n".join(f"**{r['title']}**\n\n{r['query']}" for r in records)
    report = f'''# Five live Bixby GenUICraft demos — 7 October 2026

The clearest benefit is turning wide, multi-field answer tables into cards that fit the phone width. Flights and trains are the strongest demonstrations: fare and timing fields stay visible together. Product and seasonal-weather rows also become labeled groups. Restaurant cards bring cost, hours and address into view, with the visual limitations below.

All five selected videos contain actual device footage: a fresh Bixby request, on-device A2UI Express streaming, generated output, and a same-answer switch to the original Bixby view. No model output was manually rewritten and no screenshot slideshow was substituted for the recording.

| Scenario | Video | Observed benefit | SDK repair | Warnings |
|---|---|---|---|---|
{table}

## Code review before testing

- `GenUICraft/genuicraft/src/main/java/com/samsung/genuicraft/sdk/GenUiTrainedConverter.kt:62` runs the model then calls the shared SDK compiler/recovery. The frozen `e2b_v10_shared_prompt.json` remains the trained-model prompt; no prompt changes were made for these recordings.
- `sdk/internal/renderer/flat/compose/FlatDirectTableRender.kt:250` honors explicit table presentation; specialized train, itinerary, flight and adaptive routes follow. This is why a strict table output does not automatically become cards.
- `sdk/internal/renderer/native/intents/train/NativeTrainSemantics.kt:26` recognizes rail identity and travel fields. The visible selected rail layout groups services compactly.
- `sdk/internal/renderer/FlatSpecRenderer.kt:5632` selects adaptive feature/entity/metric/timeline presentations based on shape and screen width. Entity comparisons are more promising than unstructured prose.
- `sdk/internal/renderer/FlatRestaurantBookingCards.kt:174` recognizes restaurant rows with location/rating information and provides a dedicated card route.
- Native weather, itinerary, recipe and checklist routes were candidates. A rich existing Bixby places response is a weaker conversion candidate than a concise Markdown answer because of context size and existing native presentation.

## Device and native execution evidence

Installed Bixby: **5.0.10.50**, package `com.samsung.android.bixby.agent`, connected **SM-F776U**, portrait 1080 × 2520. GenUICraft was enabled, precision `fp16_corrected`, MTP enabled. The first run's native engine log explicitly confirms `backend=GPU; precision=FP16_CORRECTED; MTP=true; MTPRequested=true; modelSupportsMtp=true`; subsequent conversions reuse the same engine and emit the scoped FP16 policy evidence.

Corrected model SHA-256: `7f01bdf1c6ba9bdf658e57c75001fc35a42edad88238bc5dab0a5f7dcf3de373`. The Q/DQ adapter reports 206 blocks, zero rejections. All selected conversions completed in **one model attempt**, with `NONE` or `GENERATED_DSL_REPAIR`, rather than a source fallback. Per-video success evidence is in [runtime_evidence.txt](runtime_evidence.txt). Tokens/s is not claimed: Bixby's integration disables detailed native metrics.

Selected key names, prices, times and units were checked against the same original-answer UI tree. This is a bounded preservation check, not a model accuracy score or independent verification of Bixby's travel, weather, product or restaurant facts. Compiler warning counts are retained above; a successful conversion does not mean perfect fidelity.

## Candidates not selected

There were **11 live capture/test runs across 10 distinct queries**. Five were selected for a visible presentation benefit. Other runs remain in local `evidence/`:

- Phone feature matrix: conversion logged success but the GenUICraft view was empty. Replaced with an entity-row phone comparison; the successful phone run was recorded again after fixing capture automation's UI-idle timeout.
- Two-day Mysuru itinerary: Bixby returned a much larger native places response. The converter rejected the context budget: estimated input 7,337 + output reserve 2,048 + template reserve 256 exceeds 8,192. It was not a successful model-generation demo.
- Energy comparison with explanation: appliance cost chips were useful, but a prose paragraph was rendered as a clipped `Formula` element. Excluded.
- Energy table-only comparison and apartment document table: rendered successfully as horizontally scrolling tables. They did not show enough additional presentation value to replace one of the five selected card scenarios.

## Visible gaps in selected cases

- Restaurant card cost, hours and address are easier to find. Its large gradient is a **decorative placeholder**, not a fetched restaurant photo. Signature dish survives in accessible row data but is not shown as a visible field; a long address is ellipsized. These need renderer refinement and are not presented as solved in this demo.
- The seasonal-weather output adds the title “Weather Forecast” to a typical-climate comparison. The month values were checked; the title should be made more appropriate by the generation/recovery pipeline.
- View switching caused several brief empty intervals. Those verified post-generation waits are removed in the edited videos, with the exact source intervals recorded in each metadata JSON. Native generation/progress is retained. No elapsed-time claim should be inferred from the edited playback.

## Video validation and editing

Five silent H.264 videos, 1080 × 1080 at 30 fps. Each raw recording and export passed strict FFmpeg decoding. Captions, arrows and transitions were reviewed against actual source/export frames. Only the longer flight Express stream is accelerated; its actual recorded spinner patch plays at normal speed. Other streams remain at normal speed. No timer, source-time text or speed label appears in the videos.

Each `.metadata.json` records source/export SHA-256, dimensions, frame count, crop, timing edits and annotations. Contact sheets and gallery posters contain actual exported frames. Original recordings and full device traces remain locally in ignored `evidence/`; concise runtime proof, selected videos, metadata and review artifacts are retained with the demo package.

Open [index.html](index.html) for all five players. Recreate exports with `render_demo.py --video evidence/<case>/raw.mp4 --plan <scenario>.plan.json --out <video>.mp4`, then run `build_gallery.py`.

## Queries

{queries}
'''
    (ROOT / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"videos": len(records), "total_bytes": sum(r["bytes"] for r in records),
                      "index": str(ROOT / "index.html"), "report": str(ROOT / "REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
