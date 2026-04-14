#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build before-vs-after markdown report for run quality.")
    parser.add_argument("--baseline_run_id", required=True)
    parser.add_argument("--candidate_run_id", required=True)
    parser.add_argument("--root", default=None, help="Repo root path (default: auto from script path)")
    return parser.parse_args()


def by_ui(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in records:
        ui_id = str(item.get("ui_id") or "").strip()
        if ui_id:
            out[ui_id] = item
    return out


def gap_counter(records: list[dict[str, Any]]) -> Counter[str]:
    c: Counter[str] = Counter()
    for item in records:
        for gap in item.get("gaps", []):
            c[str(gap)] += 1
    return c


def category_gap_counter(records: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    out: dict[str, Counter[str]] = defaultdict(Counter)
    for item in records:
        category = str(item.get("category") or "General")
        for gap in item.get("gaps", []):
            out[category][str(gap)] += 1
    return out


def metric_delta_table(base_agg: dict[str, Any], cand_agg: dict[str, Any]) -> list[tuple[str, float, float, float]]:
    keys = [
        "overall_score",
        "markdown_leakage_rate_avg",
        "table_cell_coverage_avg",
        "table_pattern_detected_rate",
        "information_chunking_score_avg",
        "image_presence_rate",
        "icon_presence_rate",
        "intent_expectation_pass_rate",
        "intent_score_avg",
    ]
    rows: list[tuple[str, float, float, float]] = []
    for key in keys:
        b = float(base_agg.get(key, 0.0))
        c = float(cand_agg.get(key, 0.0))
        rows.append((key, b, c, c - b))
    return rows


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[2]
    runs_root = root / "dataset" / "data" / "runs"
    base_dir = runs_root / args.baseline_run_id
    cand_dir = runs_root / args.candidate_run_id
    if not base_dir.exists():
        raise SystemExit(f"Baseline run dir not found: {base_dir}")
    if not cand_dir.exists():
        raise SystemExit(f"Candidate run dir not found: {cand_dir}")

    base_gaps_payload = load_json(base_dir / "quality_gaps.json")
    cand_gaps_payload = load_json(cand_dir / "quality_gaps.json")
    base_records = list(base_gaps_payload.get("records") or [])
    cand_records = list(cand_gaps_payload.get("records") or [])

    base_by_ui = by_ui(base_records)
    cand_by_ui = by_ui(cand_records)
    shared_ui_ids = sorted(set(base_by_ui.keys()) & set(cand_by_ui.keys()))

    base_agg = load_json(base_dir / "aggregates.json")
    cand_agg = load_json(cand_dir / "aggregates.json")

    base_gap_counts = gap_counter(base_records)
    cand_gap_counts = gap_counter(cand_records)
    category_base = category_gap_counter(base_records)
    category_cand = category_gap_counter(cand_records)

    improved_examples: list[tuple[str, str, list[str], list[str], str]] = []
    for ui_id in shared_ui_ids:
        b = base_by_ui[ui_id]
        c = cand_by_ui[ui_id]
        b_gaps = set(str(item) for item in (b.get("gaps") or []))
        c_gaps = set(str(item) for item in (c.get("gaps") or []))
        if len(c_gaps) < len(b_gaps):
            category = str(c.get("category") or b.get("category") or "General")
            screenshot = str(c.get("screenshot_path") or "")
            improved_examples.append(
                (
                    ui_id,
                    category,
                    sorted(b_gaps),
                    sorted(c_gaps),
                    screenshot,
                )
            )

    metric_rows = metric_delta_table(base_agg, cand_agg)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []
    lines.append("# Before vs After: Stitch-Guided Golden50 Recovery")
    lines.append("")
    lines.append(f"- Generated: {timestamp}")
    lines.append(f"- Baseline: `{args.baseline_run_id}`")
    lines.append(f"- Candidate: `{args.candidate_run_id}`")
    lines.append("")

    lines.append("## Metric Delta")
    lines.append("")
    lines.append("| metric | baseline | candidate | delta |")
    lines.append("|---|---:|---:|---:|")
    for metric, b, c, d in metric_rows:
        lines.append(f"| {metric} | {b:.4f} | {c:.4f} | {d:+.4f} |")
    lines.append("")

    lines.append("## Gap Delta (Count)")
    lines.append("")
    lines.append("| gap | baseline | candidate | delta |")
    lines.append("|---|---:|---:|---:|")
    all_gaps = sorted(set(base_gap_counts.keys()) | set(cand_gap_counts.keys()))
    for gap in all_gaps:
        b = int(base_gap_counts.get(gap, 0))
        c = int(cand_gap_counts.get(gap, 0))
        lines.append(f"| {gap} | {b} | {c} | {c - b:+d} |")
    lines.append("")

    lines.append("## Category Gap Delta")
    lines.append("")
    lines.append("| category | baseline_gap_count | candidate_gap_count | delta |")
    lines.append("|---|---:|---:|---:|")
    categories = sorted(set(category_base.keys()) | set(category_cand.keys()))
    for category in categories:
        b = sum(category_base.get(category, Counter()).values())
        c = sum(category_cand.get(category, Counter()).values())
        lines.append(f"| {category} | {b} | {c} | {c - b:+d} |")
    lines.append("")

    lines.append("## Improved Examples")
    lines.append("")
    if not improved_examples:
        lines.append("- No per-UI gap-count reduction detected in shared UI ids.")
    else:
        lines.append("| ui_id | category | baseline_gaps | candidate_gaps | screenshot |")
        lines.append("|---|---|---|---|---|")
        for ui_id, category, b_gaps, c_gaps, screenshot in improved_examples[:15]:
            screenshot_cell = screenshot or "-"
            lines.append(
                f"| `{ui_id}` | {category} | {', '.join(b_gaps) or '-'} | {', '.join(c_gaps) or '-'} | `{screenshot_cell}` |"
            )
    lines.append("")

    out_path = cand_dir / "before_vs_after.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] report written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

