import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.metrics import aggregate_metrics, compute_overall_score  # noqa: E402
from pipeline.storage import iter_jsonl  # noqa: E402
from utils.config import load_yaml  # noqa: E402


def _load_weights(weights_path: Path) -> dict:
    if not weights_path.exists():
        return {}
    cfg = load_yaml(weights_path)
    return cfg.get("evaluation", {}).get("weights", {}) or {}


def _load_aggregates(run_dir: Path, weights: dict, recompute: bool) -> dict | None:
    agg_path = run_dir / "aggregates.json"
    aggregates = None
    if agg_path.exists() and not recompute:
        try:
            aggregates = json.loads(agg_path.read_text(encoding="utf-8"))
        except Exception:
            aggregates = None
    if aggregates is None:
        genui_path = run_dir / "genui.jsonl"
        rows = list(iter_jsonl(genui_path))
        if not rows:
            return None
        responses_path = run_dir / "responses.jsonl"
        response_map: dict[str, str] = {}
        if responses_path.exists():
            for response_row in iter_jsonl(responses_path):
                response_id = response_row.get("response_id")
                response_text = response_row.get("response_text")
                if isinstance(response_id, str) and isinstance(response_text, str):
                    response_map[response_id] = response_text
        if response_map:
            for row in rows:
                if row.get("response_text"):
                    continue
                response_id = row.get("response_id")
                if isinstance(response_id, str):
                    backfill = response_map.get(response_id)
                    if isinstance(backfill, str):
                        row["response_text"] = backfill
        aggregates = aggregate_metrics(rows)
    aggregates["overall_score"] = compute_overall_score(aggregates, weights)
    return aggregates


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (int, float, str, bool))


def _format_scalar(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "-"
    return str(value)


def _group_for_column(column: str) -> str:
    if column == "run_id":
        return "Run"
    if column == "overall_score":
        return "Overall"
    if column in {"counts", "schema_valid_strict_rate", "render_ok_rate"}:
        return "Validity"
    if column in {
        "content_coverage_avg",
        "lint_score_avg",
        "dup_rate_avg",
        "markdown_leakage_rate_avg",
    }:
        return "Content"
    if column in {
        "component_count_avg",
        "unique_component_types_avg",
        "max_tree_depth_avg",
        "avg_tree_depth_avg",
        "container_to_text_ratio_avg",
        "information_chunking_score_avg",
        "ui_modularity_score_avg",
        "ui_decomposition_score_avg",
        "section_heading_coverage_avg",
    }:
        return "Structure"
    if column in {"actionable_elements_avg", "action_coverage_avg", "url_as_text_rate_avg"}:
        return "Actions"
    if column in {
        "table_pattern_detected_rate",
        "table_cell_coverage_avg",
        "table_required_rate",
        "table_ok_rate",
    }:
        return "Table"
    if column in {"missing_ids_avg", "missing_ids_rate_avg", "dangling_components_avg", "dangling_components_rate_avg"}:
        return "Reference"
    if column in {"intent_expectation_pass_rate", "intent_score_avg"}:
        return "Intent"
    if column in {"latency_ms_avg", "latency_ms_p95"}:
        return "Latency"
    return "Other"


def _build_column_order(rows: list[dict]) -> list[str]:
    preferred = [
        "run_id",
        "overall_score",
        "counts",
        "schema_valid_strict_rate",
        "render_ok_rate",
        "content_coverage_avg",
        "lint_score_avg",
        "dup_rate_avg",
        "component_count_avg",
        "unique_component_types_avg",
        "max_tree_depth_avg",
        "avg_tree_depth_avg",
        "container_to_text_ratio_avg",
        "information_chunking_score_avg",
        "ui_modularity_score_avg",
        "ui_decomposition_score_avg",
        "section_heading_coverage_avg",
        "actionable_elements_avg",
        "action_coverage_avg",
        "url_as_text_rate_avg",
        "table_pattern_detected_rate",
        "table_cell_coverage_avg",
        "table_required_rate",
        "table_ok_rate",
        "markdown_leakage_rate_avg",
        "missing_ids_avg",
        "missing_ids_rate_avg",
        "dangling_components_avg",
        "dangling_components_rate_avg",
        "intent_expectation_pass_rate",
        "intent_score_avg",
        "latency_ms_avg",
        "latency_ms_p95",
    ]
    seen: set[str] = set()
    for row in rows:
        seen.update(row.keys())
    seen.discard("intent_stats")
    columns: list[str] = [c for c in preferred if c in seen]
    remaining = sorted(c for c in seen if c not in columns)
    columns.extend(remaining)
    return columns


def _format_table(rows: list[dict], headers: list[str]) -> str:
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(_format_scalar(row.get(h, "-"))))

    def fmt_row(row: dict) -> str:
        parts = []
        for h in headers:
            value = _format_scalar(row.get(h, "-"))
            parts.append(value.ljust(widths[h]))
        return " | ".join(parts)

    # Top grouped header row.
    grouped_parts: list[str] = []
    idx = 0
    while idx < len(headers):
        group = _group_for_column(headers[idx])
        end = idx
        while end + 1 < len(headers) and _group_for_column(headers[end + 1]) == group:
            end += 1
        span_width = sum(widths[headers[i]] for i in range(idx, end + 1)) + 3 * (end - idx)
        grouped_parts.append(group.center(span_width))
        idx = end + 1

    lines = [" | ".join(grouped_parts)]
    lines.append(fmt_row({h: h for h in headers}))
    lines.append("-+-".join("-" * widths[h] for h in headers))
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _write_csv(rows: list[dict], headers: list[str], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h) for h in headers})


def _plot_overall_scores(rows: list[dict], out_path: Path) -> None:
    names = [str(r.get("run_id", "")) for r in rows]
    scores = [_to_float(r.get("overall_score")) for r in rows]
    order = np.argsort(np.array(scores))[::-1]
    names = [names[i] for i in order]
    scores = [scores[i] for i in order]

    fig, ax = plt.subplots(figsize=(13, max(4, 0.45 * len(names))))
    y = np.arange(len(names))
    bars = ax.barh(y, scores, color="#0072DE")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Overall Score")
    ax.set_title("Model Ranking by Overall Score")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    for idx, bar in enumerate(bars):
        ax.text(
            bar.get_width() + 0.15,
            bar.get_y() + bar.get_height() / 2.0,
            f"{scores[idx]:.2f}",
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_key_heatmap(rows: list[dict], out_path: Path) -> None:
    metric_cols = [
        "overall_score",
        "schema_valid_strict_rate",
        "content_coverage_avg",
        "action_coverage_avg",
        "table_pattern_detected_rate",
        "intent_score_avg",
        "ui_decomposition_score_avg",
        "markdown_leakage_rate_avg",
    ]
    metric_labels = [
        "overall",
        "schema",
        "coverage",
        "action",
        "table_det",
        "intent",
        "ui_decomp",
        "md_leak",
    ]
    names = [str(r.get("run_id", "")) for r in rows]
    matrix = np.array([[_to_float(r.get(c)) for c in metric_cols] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(14, max(4, 0.45 * len(names))))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_xticklabels(metric_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title("Key Metric Heatmap (raw values)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.set_ylabel("value", rotation=270, labelpad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_coverage_vs_actions(rows: list[dict], out_path: Path) -> None:
    x = np.array([_to_float(r.get("content_coverage_avg")) for r in rows], dtype=float)
    y = np.array([_to_float(r.get("action_coverage_avg")) for r in rows], dtype=float)
    c = np.array([_to_float(r.get("table_pattern_detected_rate")) for r in rows], dtype=float)
    s = np.array([max(_to_float(r.get("overall_score")), 0.0) * 10 + 30 for r in rows], dtype=float)
    labels = [str(r.get("run_id", "")) for r in rows]

    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    sc = ax.scatter(x, y, c=c, s=s, cmap="plasma", alpha=0.85, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("content_coverage_avg")
    ax.set_ylabel("action_coverage_avg")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_title("Coverage vs Actionability (color=table_pattern_detected_rate, size=overall_score)")
    for i, label in enumerate(labels):
        ax.annotate(label, (x[i], y[i]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.set_ylabel("table_pattern_detected_rate", rotation=270, labelpad=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _export_artifacts(rows: list[dict], headers: list[str], export_dir: Path) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = export_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    (export_dir / "rankings.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(rows, headers, export_dir / "rankings.csv")
    _plot_overall_scores(rows, plots_dir / "01_overall_score.png")
    _plot_key_heatmap(rows, plots_dir / "02_key_metrics_heatmap.png")
    _plot_coverage_vs_actions(rows, plots_dir / "03_coverage_vs_actions.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default=str(ROOT / "data" / "runs"))
    parser.add_argument("--weights", type=str, default=str(ROOT / "configs" / "run.yaml"))
    parser.add_argument("--format", type=str, choices=["table", "json"], default="table")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--recompute", action="store_true", help="Recompute aggregates from genui.jsonl")
    parser.add_argument("--include-glob", type=str, default="*", help="Include run directories matching this glob")
    parser.add_argument(
        "--exclude-glob",
        action="append",
        default=[],
        help="Exclude run directories matching this glob (repeatable)",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default=None,
        help="If set, exports rankings.json, rankings.csv and comparison plots to this folder",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    weights = _load_weights(Path(args.weights))

    rows = []
    if runs_dir.exists():
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if not run_dir.match(args.include_glob):
                continue
            if any(run_dir.match(pattern) for pattern in args.exclude_glob):
                continue
            aggregates = _load_aggregates(run_dir, weights, args.recompute)
            if not aggregates:
                continue
            row: dict[str, Any] = {"run_id": run_dir.name}
            for key, value in aggregates.items():
                # Keep only flat overall values in rankings. Exclude nested intent breakdowns.
                if key == "intent_stats":
                    continue
                if _is_scalar(value):
                    row[key] = value
            rows.append(row)

    rows.sort(key=lambda item: float(item.get("overall_score", 0.0)), reverse=True)
    headers = _build_column_order(rows)

    if args.format == "json":
        output = json.dumps(rows, ensure_ascii=False, indent=2)
    else:
        output = _format_table(rows, headers)

    if args.export_dir:
        _export_artifacts(rows, headers, Path(args.export_dir).resolve())

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
