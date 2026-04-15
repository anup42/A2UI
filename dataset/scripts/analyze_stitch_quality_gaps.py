#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_MARKDOWN_LINE_RE = re.compile(r"^\s*(?:[-*]\s+|#+\s+|\|).*")

_CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("Weather", ("weather", "forecast", "climate", "temperature", "rain", "humidity")),
    ("Travel", ("travel", "itinerary", "trip", "tour", "visa", "route")),
    ("Booking", ("booking", "book", "hotel", "stay", "reservation", "option")),
    ("Comparison", ("compare", "comparison", "vs", "versus", "pros", "cons")),
    ("Calculation", ("calculate", "calculation", "estimate", "cost breakdown", "math")),
    ("Data_visualisation", ("dashboard", "chart", "graph", "trend", "kpi", "visual")),
    ("Event_schedule", ("schedule", "agenda", "timeline", "session", "event", "calendar")),
    ("Technical_Support", ("troubleshoot", "support", "error", "issue", "debug", "fix")),
    ("Status_check", ("status", "health", "uptime", "incident", "tracking", "progress")),
    ("Recipe", ("recipe", "cook", "ingredients", "meal", "dish")),
    ("Education", ("study", "learn", "course", "lesson", "exam", "education")),
    ("Documentation", ("documentation", "docs", "guide", "manual", "api reference")),
    ("Productivity", ("productivity", "plan", "todo", "task", "workflow", "habit")),
    ("Research_Analysis", ("research", "analysis", "insight", "findings", "report")),
    ("Navigation", ("navigate", "directions", "map", "route guidance")),
    ("Creative_writing", ("story", "poem", "creative", "writing", "script")),
    ("Entertainment", ("movie", "music", "game", "entertainment", "playlist")),
    ("Localisation", ("localization", "localisation", "translate", "language", "locale")),
]


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except Exception:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def infer_category(intent: str, query_text: str, response_text: str) -> str:
    haystack = " | ".join([intent or "", query_text or "", response_text or ""])
    source = normalize(haystack)
    for category, keywords in _CATEGORY_KEYWORDS:
        if any(token in source for token in keywords):
            return category
    return "General"


def flatten_text_nodes(genui_json: Any) -> list[str]:
    if not isinstance(genui_json, dict):
        return []
    elements = genui_json.get("elements")
    if not isinstance(elements, dict):
        return []
    texts: list[str] = []
    for element in elements.values():
        if not isinstance(element, dict):
            continue
        if str(element.get("type") or "").strip().lower() != "text":
            continue
        props = element.get("props")
        if not isinstance(props, dict):
            continue
        for key in ("text", "title", "label", "content", "value"):
            value = props.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value)
                break
    return texts


def collect_components(genui_json: Any) -> list[tuple[str, dict[str, Any]]]:
    if not isinstance(genui_json, dict):
        return []
    elements = genui_json.get("elements")
    if not isinstance(elements, dict):
        return []
    out: list[tuple[str, dict[str, Any]]] = []
    for element_id, element in elements.items():
        if isinstance(element_id, str) and isinstance(element, dict):
            out.append((element_id, element))
    return out


def has_markdown_leak(text_nodes: list[str]) -> tuple[bool, int]:
    leak_lines = 0
    for text in text_nodes:
        for line in text.splitlines():
            if _MARKDOWN_LINE_RE.match(line):
                leak_lines += 1
    return leak_lines > 0, leak_lines


def table_domains(components: list[tuple[str, dict[str, Any]]]) -> list[str]:
    domains: list[str] = []
    for _, element in components:
        if str(element.get("type") or "").strip().lower() != "table":
            continue
        props = element.get("props")
        if not isinstance(props, dict):
            continue
        domain = str(props.get("domain") or "").strip().lower()
        if domain:
            domains.append(domain)
    return sorted(set(domains))


def screenshot_exists(run_dir: Path, ui_id: str) -> tuple[bool, str]:
    candidate_dirs = [
        run_dir / "android_device_rendered",
        run_dir / "android_device_rendered_stitchfix",
        run_dir / "rendered",
    ]
    for base_dir in candidate_dirs:
        if not base_dir.exists():
            continue
        direct = base_dir / f"{ui_id}.png"
        if direct.exists():
            return True, str(direct.relative_to(run_dir))
        prefixed = sorted(base_dir.glob(f"*_{ui_id}.png"))
        if prefixed:
            return True, str(prefixed[0].relative_to(run_dir))
    return False, ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Stitch-style quality gaps for a run.")
    parser.add_argument("--run_id", required=True, help="Run id under dataset/data/runs")
    parser.add_argument(
        "--root",
        default=None,
        help="Repo root path (default: auto from script path).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[2]
    src = root / "dataset" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from pipeline.metrics import compute_ui_metrics  # imported lazily after path setup

    run_dir = root / "dataset" / "data" / "runs" / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    queries = {row.get("query_id"): row for row in iter_jsonl(run_dir / "queries.jsonl")}
    responses = {row.get("response_id"): row for row in iter_jsonl(run_dir / "responses.jsonl")}
    rows = iter_jsonl(run_dir / "genui.jsonl")
    if not rows:
        raise SystemExit(f"No records found in {run_dir / 'genui.jsonl'}")

    csv_rows: list[dict[str, Any]] = []
    json_records: list[dict[str, Any]] = []
    gap_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    category_gap_counter: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        ui_id = str(row.get("ui_id") or "").strip()
        if not ui_id:
            continue
        query_id = str(row.get("query_id") or "").strip()
        response_id = str(row.get("response_id") or "").strip()
        query_text = str((queries.get(query_id) or {}).get("query_text") or "").strip()
        response_text = str(row.get("response_text") or "").strip()
        if not response_text:
            response_row = responses.get(response_id) or {}
            response_text = str(response_row.get("response_text") or "").strip()
        intent = str(row.get("intent") or "").strip()
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        validation = row.get("validation") if isinstance(row.get("validation"), dict) else {}
        errors = validation.get("errors") if isinstance(validation.get("errors"), list) else []
        error_text = " | ".join(str(item) for item in errors)

        genui_json = row.get("genui_json")
        fresh_metrics = compute_ui_metrics(response_text, genui_json)
        metrics = {**metrics, **fresh_metrics}
        components = collect_components(genui_json)
        text_nodes = flatten_text_nodes(genui_json)
        markdown_leak, markdown_leak_lines = has_markdown_leak(text_nodes)
        domains = table_domains(components)
        has_table = bool(domains) or float(metrics.get("table_pattern_detected", 0.0)) >= 1.0
        image_presence = float(metrics.get("image_presence", 0.0))
        icon_presence = float(metrics.get("icon_presence", 0.0))
        chunking = float(metrics.get("information_chunking_score", 0.0))
        table_cov = float(metrics.get("table_cell_coverage", 0.0))
        fallback_generated = "fallback_generated" in error_text
        screenshot_ok, screenshot_rel = screenshot_exists(run_dir, ui_id)

        category = infer_category(intent=intent, query_text=query_text, response_text=response_text)
        category_counter[category] += 1

        gaps: list[str] = []
        if fallback_generated:
            gaps.append("fallback_generated")
        if markdown_leak:
            gaps.append("markdown_leakage")
        if has_table and table_cov < 0.55:
            gaps.append("weak_table_coverage")
        if chunking < 0.45:
            gaps.append("dense_text_layout")
        if (category in {"Weather", "Travel", "Booking", "Comparison", "Recipe"}) and (image_presence < 1.0 and icon_presence < 1.0):
            gaps.append("missing_media_signal")
        if not screenshot_ok:
            gaps.append("missing_screenshot")

        for gap in gaps:
            gap_counter[gap] += 1
            category_gap_counter[category][gap] += 1

        element_count = len(components)
        ir_byte_size = len(json.dumps(genui_json, ensure_ascii=False).encode("utf-8")) if genui_json is not None else 0

        record = {
            "ui_id": ui_id,
            "query_id": query_id,
            "response_id": response_id,
            "intent": intent,
            "category": category,
            "ir_element_count": element_count,
            "ir_byte_size": ir_byte_size,
            "table_domains": domains,
            "has_table": has_table,
            "table_cell_coverage": table_cov,
            "information_chunking_score": chunking,
            "image_presence": image_presence,
            "icon_presence": icon_presence,
            "markdown_leak_lines": markdown_leak_lines,
            "fallback_generated": fallback_generated,
            "screenshot_exists": screenshot_ok,
            "screenshot_path": screenshot_rel,
            "gaps": gaps,
        }
        json_records.append(record)
        csv_rows.append(
            {
                "ui_id": ui_id,
                "query_id": query_id,
                "response_id": response_id,
                "intent": intent,
                "category": category,
                "ir_element_count": element_count,
                "ir_byte_size": ir_byte_size,
                "table_domains": ",".join(domains),
                "table_cell_coverage": f"{table_cov:.4f}",
                "information_chunking_score": f"{chunking:.4f}",
                "image_presence": f"{image_presence:.4f}",
                "icon_presence": f"{icon_presence:.4f}",
                "markdown_leak_lines": markdown_leak_lines,
                "fallback_generated": int(fallback_generated),
                "screenshot_exists": int(screenshot_ok),
                "screenshot_path": screenshot_rel,
                "gaps": ";".join(gaps),
            }
        )

    summary = {
        "run_id": args.run_id,
        "records": len(json_records),
        "gap_counts": dict(sorted(gap_counter.items())),
        "category_counts": dict(sorted(category_counter.items())),
        "category_gap_counts": {
            category: dict(sorted(counter.items()))
            for category, counter in sorted(category_gap_counter.items())
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    out_json = run_dir / "quality_gaps.json"
    out_csv = run_dir / "quality_gaps.csv"
    out_json.write_text(
        json.dumps(
            {
                "summary": summary,
                "records": json_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    fieldnames = list(csv_rows[0].keys()) if csv_rows else []
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"[ok] quality gaps written: {out_json}")
    print(f"[ok] quality gaps csv written: {out_csv}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
