import argparse
import json
import sys
from pathlib import Path

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


def _format_table(rows: list[dict]) -> str:
    headers = [
        "run_id",
        "overall_score",
        "schema_valid_strict_rate",
        "content_coverage_avg",
        "action_coverage_avg",
        "table_pattern_detected_rate",
        "markdown_leakage_rate_avg",
        "render_ok_rate",
    ]
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, "-"))))

    def fmt_row(row: dict) -> str:
        parts = []
        for h in headers:
            value = row.get(h, "-")
            parts.append(str(value).ljust(widths[h]))
        return " | ".join(parts)

    lines = [fmt_row({h: h for h in headers})]
    lines.append("-+-".join("-" * widths[h] for h in headers))
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default=str(ROOT / "data" / "runs"))
    parser.add_argument("--weights", type=str, default=str(ROOT / "configs" / "run.yaml"))
    parser.add_argument("--format", type=str, choices=["table", "json"], default="table")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--recompute", action="store_true", help="Recompute aggregates from genui.jsonl")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    weights = _load_weights(Path(args.weights))

    rows = []
    if runs_dir.exists():
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            aggregates = _load_aggregates(run_dir, weights, args.recompute)
            if not aggregates:
                continue
            row = {
                "run_id": run_dir.name,
                "overall_score": round(float(aggregates.get("overall_score", 0.0)), 4),
                "schema_valid_strict_rate": round(float(aggregates.get("schema_valid_strict_rate", 0.0)), 4),
                "content_coverage_avg": round(float(aggregates.get("content_coverage_avg", 0.0)), 4),
                "action_coverage_avg": round(float(aggregates.get("action_coverage_avg", 0.0)), 4),
                "table_pattern_detected_rate": round(float(aggregates.get("table_pattern_detected_rate", 0.0)), 4),
                "markdown_leakage_rate_avg": round(float(aggregates.get("markdown_leakage_rate_avg", 0.0)), 4),
                "render_ok_rate": aggregates.get("render_ok_rate"),
            }
            rows.append(row)

    rows.sort(key=lambda item: float(item.get("overall_score", 0.0)), reverse=True)

    if args.format == "json":
        output = json.dumps(rows, ensure_ascii=False, indent=2)
    else:
        output = _format_table(rows)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
