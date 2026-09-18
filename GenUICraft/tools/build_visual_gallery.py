"""Build a local, byte-preserving before/after renderer screenshot gallery.

The inputs must be complete 50-case saved replay runs. The historical side must
be a captured raw-model revalidation and the new side a JSON-only renderer
replay of the exact same accepted documents. No image processing is performed.

Usage:
  python tools/build_visual_gallery.py --before BEFORE_RUN --after AFTER_RUN \
      --output validation/20260918_visual/gallery
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
CORPUS = PROJECT / "genuicraft/src/test/resources/genuicraft_bixby50.jsonl"
IDS = [f"BXP-{index:03}" for index in range(1, 51)]
CAPTURE_NAME = re.compile(r"[a-z][a-z0-9_]*")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique_object)


def exact_ids(values, label):
    require(isinstance(values, list) and all(isinstance(value, str) for value in values),
            f"Invalid {label} IDs")
    counts = Counter(values)
    missing = sorted(set(IDS) - counts.keys())
    extra = sorted(counts.keys() - set(IDS))
    duplicate = sorted(key for key, count in counts.items() if count != 1)
    require(not missing and not extra and not duplicate and len(values) == len(IDS),
            f"{label}: missing={missing}, unexpected={extra}, duplicate={duplicate}")


def exact_integer(value, expected, label):
    require(type(value) is int and value == expected, f"{label} must be integer {expected}")


def indexed_rows(value, label):
    require(isinstance(value, list) and all(isinstance(row, dict) for row in value),
            f"Invalid {label}")
    exact_ids([row.get("id") for row in value], label)
    return {row["id"]: row for row in value}


def png_dimensions(data, label):
    require(len(data) >= 24 and data.startswith(PNG_SIGNATURE), f"Invalid PNG: {label}")
    require(data[12:16] == b"IHDR", f"PNG lacks IHDR: {label}")
    width, height = struct.unpack(">II", data[16:24])
    require(100 < width <= 20_000 and 100 < height <= 20_000,
            f"Invalid PNG dimensions for {label}: {width}x{height}")
    return width, height


def relative_to_workspace(path):
    try:
        return path.resolve().relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError:
        return Path(os.path.relpath(path.resolve(), start=WORKSPACE.resolve())).as_posix()


def load_run(run, expected_kind, expected_mode, expected_provider_calls, label):
    run = run.resolve()
    require(run.is_dir(), f"Missing {label} run: {run}")
    config = read_json(run / "replay_config.json")
    summary = read_json(run / "replay_summary.json")
    reports = indexed_rows(read_json(run / "replay_results.json"), f"{label} replay results")
    require(config.get("runId") == run.name and summary.get("runId") == run.name,
            f"{label} runId does not match directory")
    exact_ids(config.get("cases"), f"{label} config")
    for record_name, record in (("config", config), ("summary", summary)):
        require(record.get("kind") == expected_kind and record.get("replayMode") == expected_mode,
                f"Unexpected {label} {record_name} kind or replay mode")
        exact_integer(record.get("modelCalls"), 0, f"{label} {record_name}.modelCalls")
        require(record.get("inferenceEvaluated") is False,
                f"{label} {record_name} must exclude inference evaluation")
    for field, expected in (("total", 50), ("rendered", 50), ("failed", 0)):
        exact_integer(summary.get(field), expected, f"{label} summary.{field}")
    exact_ids(sorted(path.name for path in run.iterdir()
                     if path.is_dir() and path.name.startswith("BXP-")), f"{label} directories")
    for case_id, report in reports.items():
        require(report == read_json(run / case_id / "replay_result.json"),
                f"{label} {case_id} per-case report differs from replay_results.json")
        require(report.get("kind") == expected_kind and report.get("replayMode") == expected_mode,
                f"Unexpected {label} {case_id} kind or replay mode")
        require(report.get("status") == "rendered" and report.get("issues") == [],
                f"{label} {case_id} did not render cleanly")
        exact_integer(report.get("modelCalls"), 0, f"{label} {case_id}.modelCalls")
        exact_integer(report.get("capturedProviderCalls"), expected_provider_calls,
                      f"{label} {case_id}.capturedProviderCalls")
        require(report.get("inferenceEvaluated") is False,
                f"{label} {case_id} must exclude inference evaluation")
    return run, config, reports


def capture_names(report, case_dir, label):
    captures = report.get("captures")
    require(isinstance(captures, list) and captures and all(isinstance(item, dict) for item in captures),
            f"Missing {label} capture records")
    names = []
    for capture in captures:
        name = capture.get("name")
        require(isinstance(name, str) and CAPTURE_NAME.fullmatch(name),
                f"Invalid {label} capture name: {name!r}")
        require(capture.get("screenshot") is True, f"Unsuccessful {label} screenshot: {name}")
        require(name not in names, f"Duplicate {label} capture: {name}")
        require((case_dir / f"{name}.png").is_file(), f"Missing {label} PNG: {name}")
        names.append(name)
    require(names[0] == "initial", f"First {label} capture must be initial")
    return names


def gallery_html(cases, before_name, after_name):
    encoded = (json.dumps(cases, ensure_ascii=False, separators=(",", ":"))
               .replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))
    before_label = html.escape(before_name)
    after_label = html.escape(after_name)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GenUICraft renderer before/after gallery</title>
<style>
:root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; background:#eef2f6; color:#172033; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; min-width:320px; }}
header {{ position:sticky; top:0; z-index:5; padding:16px 20px 14px; background:rgba(255,255,255,.97); border-bottom:1px solid #d8dee9; box-shadow:0 4px 18px rgba(31,43,66,.07); }}
h1 {{ margin:0 0 5px; font-size:clamp(20px,2.3vw,30px); letter-spacing:-.02em; }}
.note {{ margin:0; max-width:1100px; color:#526076; font-size:14px; line-height:1.45; }}
.controls {{ display:flex; flex-wrap:wrap; align-items:end; gap:10px; margin-top:13px; }}
label {{ display:grid; gap:4px; color:#526076; font-size:12px; font-weight:700; letter-spacing:.02em; }}
select, button {{ min-height:40px; border:1px solid #bcc6d6; border-radius:9px; background:#fff; color:#172033; font:inherit; }}
select {{ padding:0 34px 0 11px; max-width:min(460px,88vw); }}
button {{ padding:0 13px; cursor:pointer; font-weight:700; }}
button:hover {{ border-color:#61708a; background:#f6f8fb; }}
button[aria-pressed="true"] {{ color:#fff; border-color:#334155; background:#334155; }}
.spacer {{ flex:1; }}
main {{ padding:18px 20px 28px; }}
.case-head {{ max-width:1200px; margin:0 auto 14px; }}
.case-head h2 {{ margin:0; font-size:20px; }}
.query {{ margin:7px 0 0; color:#46546b; line-height:1.5; }}
.capture-note {{ margin:8px 0 0; color:#6b7280; font-size:13px; }}
.compare {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:16px; max-width:1800px; margin:0 auto; }}
.panel {{ min-width:0; overflow:hidden; border:1px solid #ccd4e0; border-radius:14px; background:#fff; box-shadow:0 8px 24px rgba(31,43,66,.08); }}
.panel-head {{ display:flex; align-items:center; justify-content:space-between; gap:8px; padding:11px 13px; border-bottom:1px solid #dde3ec; }}
.panel-title {{ font-weight:800; }}
.meta {{ color:#657187; font-size:12px; font-variant-numeric:tabular-nums; }}
.open {{ color:#315ea8; font-size:13px; font-weight:700; text-decoration:none; }}
.open:hover {{ text-decoration:underline; }}
.shot {{ height:calc(100vh - 285px); min-height:440px; overflow:auto; text-align:center; background:repeating-conic-gradient(#eef1f5 0 25%,#f8fafc 0 50%) 50%/18px 18px; }}
.shot img {{ display:block; width:100%; height:auto; margin:0 auto; }}
body.native .shot img {{ width:auto; max-width:none; }}
.empty {{ padding:30px; color:#8a94a6; }}
@media (max-width:900px) {{
  header {{ position:relative; }}
  main {{ padding:14px 10px 22px; }}
  .compare {{ grid-template-columns:1fr; }}
  .shot {{ height:auto; min-height:300px; max-height:75vh; }}
  .spacer {{ display:none; }}
}}
@media (prefers-reduced-motion:no-preference) {{ button,select {{ transition:.15s ease; }} }}
</style>
</head>
<body>
<header>
  <h1>GenUICraft renderer before / after</h1>
  <p class="note">The two columns render the same accepted JSON bytes, and both saved replay runs record zero model calls. PNGs are copied without image editing. Historical and current screenshots can stop at different scroll extents because layout changed; matching capture names still represent each run’s own viewport.</p>
  <div class="controls">
    <button id="previous" type="button" aria-label="Previous case">← Previous</button>
    <label>Case<select id="case"></select></label>
    <button id="next" type="button" aria-label="Next case">Next →</button>
    <label>Matching capture<select id="capture"></select></label>
    <span class="spacer"></span>
    <button id="fit" type="button" aria-pressed="true">Fit width</button>
    <button id="native" type="button" aria-pressed="false">1:1 pixels</button>
  </div>
</header>
<main>
  <section class="case-head" aria-live="polite">
    <h2 id="title"></h2>
    <p id="query" class="query"></p>
    <p id="capture-note" class="capture-note"></p>
  </section>
  <section class="compare">
    <article class="panel">
      <div class="panel-head"><span class="panel-title">Before · {before_label}</span><span><span id="before-meta" class="meta"></span> · <a id="before-open" class="open" target="_blank" rel="noopener">Open PNG</a></span></div>
      <div id="before-shot" class="shot"><img id="before-image" alt=""></div>
    </article>
    <article class="panel">
      <div class="panel-head"><span class="panel-title">After · {after_label}</span><span><span id="after-meta" class="meta"></span> · <a id="after-open" class="open" target="_blank" rel="noopener">Open PNG</a></span></div>
      <div id="after-shot" class="shot"><img id="after-image" alt=""></div>
    </article>
  </section>
</main>
<script>
const cases={encoded};
const byId=Object.fromEntries(cases.map(item=>[item.id,item]));
const caseSelect=document.getElementById('case');
const captureSelect=document.getElementById('capture');
for(const item of cases){{const option=document.createElement('option');option.value=item.id;option.textContent=`${{item.id}} · ${{item.domain}}`;caseSelect.append(option);}}
function stateFromHash(){{const [caseId,capture]=decodeURIComponent(location.hash.slice(1)).split('/');return {{caseId:byId[caseId]?caseId:cases[0].id,capture}};}}
function render(preferredCapture){{
  const item=byId[caseSelect.value]||cases[0];
  const allowed=item.captures.map(c=>c.name);
  const selected=allowed.includes(preferredCapture)?preferredCapture:(allowed.includes(captureSelect.value)?captureSelect.value:'initial');
  captureSelect.replaceChildren(...item.captures.map(c=>{{const o=document.createElement('option');o.value=c.name;o.textContent=`${{c.name}}.png`;return o;}}));
  captureSelect.value=selected;
  const capture=item.captures.find(c=>c.name===selected);
  document.getElementById('title').textContent=`${{item.id}} · ${{item.domain}}`;
  document.getElementById('query').textContent=item.query;
  document.getElementById('capture-note').textContent=`Matching saved filename: ${{selected}}.png. Before ${{capture.before.width}}×${{capture.before.height}}; after ${{capture.after.width}}×${{capture.after.height}}. Panels scroll independently because the two layouts can reach different positions.`;
  for(const side of ['before','after']){{
    const image=document.getElementById(`${{side}}-image`);const data=capture[side];
    image.src=data.path;image.alt=`${{side==='before'?'Before':'After'}} ${{item.id}} ${{selected}} screenshot`;
    document.getElementById(`${{side}}-meta`).textContent=`${{data.width}}×${{data.height}}`;
    document.getElementById(`${{side}}-open`).href=data.path;
    document.getElementById(`${{side}}-shot`).scrollTo(0,0);
  }}
  const hash=`${{item.id}}/${{selected}}`;if(decodeURIComponent(location.hash.slice(1))!==hash)history.replaceState(null,'',`#${{encodeURIComponent(hash)}}`);
}}
function move(delta){{const index=cases.findIndex(item=>item.id===caseSelect.value);caseSelect.value=cases[(index+delta+cases.length)%cases.length].id;render('initial');}}
caseSelect.addEventListener('change',()=>render('initial'));
captureSelect.addEventListener('change',()=>render(captureSelect.value));
document.getElementById('previous').addEventListener('click',()=>move(-1));
document.getElementById('next').addEventListener('click',()=>move(1));
for(const mode of ['fit','native'])document.getElementById(mode).addEventListener('click',()=>{{document.body.classList.toggle('native',mode==='native');document.getElementById('fit').setAttribute('aria-pressed',String(mode==='fit'));document.getElementById('native').setAttribute('aria-pressed',String(mode==='native'));}});
addEventListener('hashchange',()=>{{const state=stateFromHash();caseSelect.value=state.caseId;render(state.capture);}});
const initial=stateFromHash();caseSelect.value=initial.caseId;render(initial.capture);
</script>
</body>
</html>
"""


def build(before, after, output):
    output = output.resolve()
    require(not output.exists(), f"Refusing to overwrite existing gallery: {output}")
    before, before_config, before_reports = load_run(
        before, "captured_model_revalidation", "raw_model", 1, "before"
    )
    after, after_config, after_reports = load_run(after, "renderer_replay", "json", 0, "after")
    require(before != after, "Before and after runs must differ")

    corpus_bytes = CORPUS.read_bytes()
    corpus = indexed_rows([
        json.loads(line, object_pairs_hook=unique_object)
        for line in corpus_bytes.decode("utf-8-sig").splitlines() if line.strip()
    ], "corpus")
    cases, copy_plan = [], []
    for case_id in IDS:
        before_case, after_case = before / case_id, after / case_id
        before_document = (before_case / "revalidated.output.a2ui.json").read_bytes()
        after_document = (after_case / "source.output.a2ui.json").read_bytes()
        require(before_document == after_document,
                f"{case_id} before and after accepted JSON bytes differ")
        document_hash = sha(before_document)
        require(after_reports[case_id].get("sourceJsonSha256") == document_hash,
                f"{case_id} after report source JSON hash mismatch")
        before_names = capture_names(before_reports[case_id], before_case, f"before {case_id}")
        after_names = capture_names(after_reports[case_id], after_case, f"after {case_id}")
        common_names = [name for name in after_names if name in set(before_names)]
        require(common_names and common_names[0] == "initial", f"{case_id} has no matching initial capture")
        captures = []
        for name in common_names:
            capture = {"name": name}
            for side, case_dir in (("before", before_case), ("after", after_case)):
                source_png = case_dir / f"{name}.png"
                data = source_png.read_bytes()
                width, height = png_dimensions(data, f"{side} {case_id}/{name}.png")
                relative = Path("assets") / case_id / side / f"{name}.png"
                capture[side] = {
                    "path": relative.as_posix(),
                    "width": width,
                    "height": height,
                    "sha256": sha(data),
                }
                copy_plan.append((source_png, relative, capture[side]["sha256"]))
            captures.append(capture)
        row = corpus[case_id]
        require(isinstance(row.get("query"), str) and row["query"].strip(), f"Missing query for {case_id}")
        require(isinstance(row.get("domain"), str) and row["domain"].strip(), f"Missing domain for {case_id}")
        cases.append({
            "id": case_id,
            "domain": row["domain"],
            "query": row["query"],
            "documentSha256": document_hash,
            "captures": captures,
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{output.name}.staging-", dir=output.parent))
    try:
        for source_png, relative, expected_hash in copy_plan:
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_png, destination)
            require(sha(destination.read_bytes()) == expected_hash,
                    f"Copied PNG differs from source: {relative.as_posix()}")
        (temporary / "gallery.html").write_text(
            gallery_html(cases, before.name, after.name), encoding="utf-8", newline="\n"
        )
        manifest = {
            "schemaVersion": 1,
            "kind": "byte_preserving_renderer_visual_gallery",
            "beforeRun": relative_to_workspace(before),
            "afterRun": relative_to_workspace(after),
            "beforeRunId": before_config["runId"],
            "afterRunId": after_config["runId"],
            "caseCount": len(cases),
            "modelCalls": {"before": 0, "after": 0},
            "sameAcceptedJson": True,
            "imageTransformations": False,
            "corpusSha256": sha(corpus_bytes),
            "assetCount": len(copy_plan),
            "cases": cases,
            "createdUtc": datetime.now(timezone.utc).isoformat(),
        }
        (temporary / "gallery_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = build(args.before, args.after, args.output)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "gallery": str((args.output.resolve() / "gallery.html")),
        "cases": manifest["caseCount"],
        "assets": manifest["assetCount"],
        "sameAcceptedJson": manifest["sameAcceptedJson"],
        "modelCalls": manifest["modelCalls"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
