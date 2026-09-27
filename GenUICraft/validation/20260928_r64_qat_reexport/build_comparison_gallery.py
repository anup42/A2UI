#!/usr/bin/env python3
"""Build a local, side-by-side view of raw text and rendered screenshots."""

from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OLD = ROOT / "baseline_v10"
NEW = ROOT / "device" / "flip8_r64_qat_mtp_on_20260928_r1"
CASES = (
    ("BXP-001", "Weather"),
    ("BXP-003", "Trains"),
    ("BXP-030", "EPF / PPF / NPS"),
    ("BXP-032", "Insurance"),
    ("BXP-037", "School boards"),
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def panel(root: Path, case_id: str, label: str) -> str:
    result = load_json(root / case_id / "result.json")
    raw = (root / case_id / "output.express").read_text(encoding="utf-8")
    strict_error = next(
        (
            item.removeprefix("Strict compile rejected output: ")
            for item in result.get("warnings", [])
            if item.startswith("Strict compile rejected output: ")
        ),
        "No strict compiler error recorded",
    )
    metrics = result["metrics"]
    relative = (root / case_id).relative_to(ROOT).as_posix()
    return f"""
    <article class="panel">
      <h3>{html.escape(label)}</h3>
      <p class="metrics">{result['elapsedMs'] / 1000:.2f} s ·
      {metrics['outputTokens']} output tokens ·
      {metrics['decodeTokensPerSecond']:.2f} native tokens/s</p>
      <p class="flag">Raw strict: {str(result.get('rawStrictValid', False)).lower()} ·
      Repair: {html.escape(str(result.get('repairKind', 'none')))} ·
      Source warnings: {result.get('sourceFidelityWarnings', 0)}</p>
      <p class="error">{html.escape(strict_error)}</p>
      <a href="{relative}/screen.png"><img src="{relative}/screen.png"
        alt="{html.escape(label)} screenshot for {case_id}" loading="lazy"></a>
      <details><summary>Scrolled screenshot</summary>
        <a href="{relative}/screen_scrolled.png"><img src="{relative}/screen_scrolled.png"
          alt="{html.escape(label)} scrolled screenshot for {case_id}" loading="lazy"></a>
      </details>
      <details><summary>Exact raw Express output before repair</summary>
        <p><a href="{relative}/output.express">Open raw file</a></p>
        <pre>{html.escape(raw)}</pre>
      </details>
    </article>"""


def main() -> None:
    old_config = load_json(OLD / "run_config.json")
    new_config = load_json(NEW / "run_config.json")
    assert old_config["cases"] == new_config["cases"] == [item[0] for item in CASES]
    assert old_config["corpus"] == new_config["corpus"]
    assert old_config["prompt"] == new_config["prompt"]
    assert old_config["runtime"] == new_config["runtime"]
    assert old_config["device"]["fingerprint"] == new_config["device"]["fingerprint"]
    sections = []
    for case_id, title in CASES:
        old_result = load_json(OLD / case_id / "result.json")
        new_result = load_json(NEW / case_id / "result.json")
        assert old_result["renderedPromptSha256"] == new_result["renderedPromptSha256"]
        assert load_json(OLD / case_id / "source.json") == load_json(NEW / case_id / "source.json")
        sections.append(
            f'<section><h2>{case_id} · {html.escape(title)}</h2><div class="pair">'
            + panel(OLD, case_id, "Current test-app model: trained E2B V10")
            + panel(NEW, case_id, "New model: rank-64 QAT-compatible")
            + "</div></section>"
        )
    content = """<!doctype html><html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>V10 vs rank-64 QAT LiteRT-LM comparison</title>
    <style>
    :root{font-family:system-ui,sans-serif;color:#172039;background:#f5f7fb}
    body{margin:auto;max-width:1500px;padding:24px}
    h1{font-size:1.8rem}h2{margin:0 0 16px}h3{margin:0}
    .intro{max-width:900px;line-height:1.5}
    section{margin:28px 0 48px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}
    .panel{background:white;border:1px solid #d7dfed;border-radius:18px;padding:18px;min-width:0}
    .metrics{font-weight:650}.flag,.error{font-size:.9rem;line-height:1.4}
    .error{color:#8a2637}.panel img{display:block;width:100%;height:auto;border:1px solid #e3e8f2;border-radius:8px}
    details{margin-top:12px}summary{cursor:pointer;color:#1f4bb5;font-weight:600}
    pre{max-height:420px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f3fa;padding:16px;border-radius:10px}
    @media(max-width:800px){.pair{grid-template-columns:1fr}body{padding:12px}}
    </style></head><body><h1>Trained E2B V10 vs rank-64 QAT-compatible</h1>
    <p class="intro">Five matched Bixby50 cases on the same Flip8 build, corpus,
    prompt and GPU+MTP settings. Screenshots show output after generated-DSL repair;
    expand each raw section to inspect the exact model text before repair.
    Both models failed strict validation on all five raw outputs.</p>
    """ + "\n".join(sections) + "</body></html>\n"
    output = ROOT / "V10_vs_QAT_side_by_side.html"
    output.write_text(content, encoding="utf-8")
    print(f"Wrote {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
