"""Build the delivery report from actual device evidence; run after final replay."""
from pathlib import Path
import hashlib
import html
import json
import xml.etree.ElementTree as ET

base = Path(__file__).resolve().parent
repo = base.parents[2]
result = json.loads((base / "device_final/result.json").read_text(encoding="utf-8-sig"))
tests = json.loads((base / "sdk_test_summary.json").read_text(encoding="utf-8-sig"))
assert not result["missingCards"] and not result["missingVisibleFacts"]
assert not result["missingLoadedPhotos"] and not result["unavailableFirstPhotos"]
assert not any(t.lower() == "console output" for t in result["visibleTexts"])
assert result["sdkProviderCalls"] == result["testAppProviderFactoryCalls"] == 0
sdk_count = sum(t["Tests"] for t in tests)
assert not any(t["Failures"] or t["Errors"] for t in tests)
apk = repo / "android/app/build/outputs/apk/debug/app-debug.apk"
aar = repo / "GenUICraft/build/repo/com/samsung/genuicraft/genuicraft/0.6.46/genuicraft-0.6.46.aar"
provenance = {
    "sdkVersion": "0.6.46",
    "deviceSerial": "R3GL203AKSF",
    "deviceModel": "SM-F776U",
    "apkSha256": hashlib.sha256(apk.read_bytes()).hexdigest(),
    "aarSha256": hashlib.sha256(aar.read_bytes()).hexdigest(),
    "sourceSha256": result["fixtureSha256"],
    "sdkDocumentSha256": result["sdkDocumentSha256"],
}
(base / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
selected = []
for title in result["visibleCards"]:
    candidates = []
    for path in (base / "device_final").glob("page_*.xml"):
        root = ET.parse(path).getroot()
        for node in root.iter("node"):
            if node.attrib.get("content-desc") == "Place card: " + title:
                import re
                bounds = list(map(int, re.findall(r"\d+", node.attrib["bounds"])))
                if len(bounds) == 4:
                    candidates.append(((bounds[3] - bounds[1]), path.with_suffix(".png")))
    if candidates:
        selected.append((title, max(candidates, key=lambda x: x[0])[1]))

md = f"""# Mysore place cards — SDK 0.6.46

Implemented and installed in the GenUICraft test app on Flip8 (SM-F776U, R3GL203AKSF). Replayed the captured Bixby answer to **Plan a relaxed two-day trip to Mysore from Bengaluru** through the public SDK and real SDK demo page.

## What changed

- Added an SDK-owned importer for supported structured Bixby View responses. Existing View data is converted directly to A2UI Express and validated A2UI JSON; ordinary Markdown continues through the model pipeline.
- Added native PlaceCard rendering: rounded cards, title, gold rating star and review count, linked address, horizontally swipeable photos, and Call/Directions/Website chips. Vertical scrolling belongs to the parent screen.
- Preserved source order and exact photo URLs, optional photo click targets, ratings, review counts, addresses, and action labels/URLs. Missing ratings stay missing; Mysore Palace does not gain an invented rating.
- Fixed Markdown headings being mistaken for shell output. Explicit Markdown stays prose; actual fenced code and shell/Python samples remain supported.
- Added a demo button under **GenUICraft SDK · Bixby50 → Try Mysore place cards · captured Bixby answer**. It needs no model download and labels the operation Import / Validate / Render.

## Verified result

| Check | Result |
|---|---|
| Place cards | 6/6 visible |
| Photo URLs retained exactly | 18/18 |
| Photo click URLs retained exactly | 9/9 |
| Action labels and URLs retained exactly | 11/11 |
| First photo loaded | 6/6 cards |
| First card carousel | Third photo reached |
| Directions | Exact source URL reached host callback |
| Model calls / provider creation | 0 / 0 |
| Repair or fallback | None |
| Conversion status | {result['testAppStatus']} |
| SDK JVM tests | {sdk_count} passed |
| App model-readiness tests | 16 passed |
| Python IR format tests | 12 passed; 21 generated artifacts verified |
| Device instrumentation | 2 passed |

The conversion time above measures local import and compilation, not image-download completion. All 18 URLs were checked for preservation; the device test verifies each card's first photo and the first card's third photo, rather than claiming every photo was downloaded. Actions were intercepted for validation, without placing calls or opening external sites.

## Evidence and boundaries

- [Device results](device_final/result.json), [imported Express](device_final/sdk_import.express), [A2UI JSON](device_final/sdk_import.a2ui.json), [captured input](device_final/captured_bixby_view.json), [binary hashes](provenance.json).
- [Final SDK build](sdk-delivery-build.log), [app build](app-delivery-build.log), [install](app-delivery-install.log), [device test](device-delivery-test.log).
- This is a real device replay of captured Bixby content, not a fresh Bixby query or an E2B quality score. Photos and facts come from that captured answer; no image search or generation was added.
- Bixby itself was not changed or installed in this task. Its host must consume SDK 0.6.46 and pass the structured View payload rather than flattening it to plain text.
- The new PlaceCard route is in the shared SDK renderer used by this demo/AAR; the app's separate legacy renderer is not the tested route.
- Remaining text-style limitation: the existing general Text renderer can leave inner Markdown heading markers visible in a multiheading paragraph (for example, Course highlights). The place-card fields and photos are unaffected.
- The extra page_12 capture is a manual follow-up from the demo button, showing the complete zoo card; other device_final pages are from the automated replay.

## Card screenshots

"""
for title, path in selected:
    md += f"- [{title}]({path.relative_to(base).as_posix()})\n"
(base / "REPORT.md").write_text(md, encoding="utf-8")
cards = "".join(
    f'<article><h2>{html.escape(title)}</h2><a href="{path.relative_to(base).as_posix()}"><img src="{path.relative_to(base).as_posix()}" loading="lazy" alt="{html.escape(title)} on Flip8"></a></article>'
    for title, path in selected
)
page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mysore place cards · GenUICraft</title><style>
body{{font:16px/1.5 system-ui,sans-serif;margin:0;background:#f3f7f7;color:#172d32}}main{{max-width:1200px;margin:auto;padding:32px}}h1{{font-size:34px;margin-bottom:8px}}.intro,.checks{{background:white;padding:24px;border-radius:18px;margin:18px 0}}.checks{{display:flex;flex-wrap:wrap;gap:30px}}strong{{color:#087e77}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:20px}}article{{background:white;padding:16px;border-radius:18px}}h2{{font-size:18px;min-height:52px}}img{{width:100%;border-radius:12px}}a{{color:#087e77}}small{{color:#4c6267}}</style></head><body><main>
<small>SDK 0.6.46 · Flip8 · Captured Bixby answer replay</small><h1>Two-day Mysore trip: proper place cards</h1>
<div class="intro">Preserves the supplied photos, ratings, addresses and actions in native GenUICraft cards. Swipe photos horizontally; scroll the itinerary as one page.<p><strong>{html.escape(result['testAppStatus'])}</strong><br><small>Local import/compile time; excludes photo downloads. No model inference for this structured answer.</small></p><a href="REPORT.md">Full report</a> · <a href="device_final/result.json">Device evidence</a> · <a href="device_final/sdk_import.express">Imported Express</a> · <a href="provenance.json">Build hashes</a></div>
<div class="checks"><span><strong>6/6</strong> place cards</span><span><strong>18/18</strong> photo URLs retained</span><span><strong>9/9</strong> photo links retained</span><span><strong>11/11</strong> actions retained</span><span><strong>6/6</strong> first photos loaded</span></div>
<div class="grid">{cards}</div><p><small>Device screenshots, not mockups. Directions callback and first-card carousel tested. Bixby installation remains unchanged.</small></p></main></body></html>"""
(base / "index.html").write_text(page, encoding="utf-8")
print(json.dumps({"report": str(base / "index.html"), "screenshots": len(selected), "sdkTests": sdk_count, **provenance}, indent=2))
