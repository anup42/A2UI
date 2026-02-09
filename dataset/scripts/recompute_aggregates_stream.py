import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.metrics import (  # noqa: E402
    compute_ui_metrics,
    compute_intent_metrics,
    content_coverage,
    dup_rate,
    lint_score,
    count_tokens,
    compute_overall_score,
)
from pipeline.storage import iter_jsonl  # noqa: E402
from utils.config import load_yaml  # noqa: E402

try:  # optional, speeds up large jsonl parsing
    import orjson  # type: ignore

    def _loads(payload: str) -> dict:
        return orjson.loads(payload)

except Exception:  # pragma: no cover

    def _loads(payload: str) -> dict:
        return json.loads(payload)


def _normalize_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _load_map(path: Path, key: str) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    if not path.exists():
        return mapping
    for row in iter_jsonl(path):
        value = row.get(key)
        if value:
            mapping[value] = row
    return mapping


def _resolve_run_dir(runs_dir: Path, run_id: str) -> Path:
    candidate = Path(run_id)
    if candidate.exists():
        return candidate
    return runs_dir / run_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default=str(ROOT / "data" / "runs"))
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--genui-path", type=str, default="")
    parser.add_argument("--weights", type=str, default=str(ROOT / "configs" / "run.yaml"))
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    run_dir = _resolve_run_dir(runs_dir, args.run_id).resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    weights_cfg = load_yaml(Path(args.weights)).get("evaluation", {}).get("weights", {}) or {}

    responses_map = _load_map(run_dir / "responses.jsonl", "response_id")
    queries_map = _load_map(run_dir / "queries.jsonl", "query_id")
    genui_path = Path(args.genui_path).resolve() if args.genui_path else (run_dir / "genui.jsonl")
    if not genui_path.exists():
        raise SystemExit(f"genui.jsonl not found in {run_dir}")

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    latency_values: list[float] = []
    render_values: list[float] = []
    intent_stats: dict[str, dict[str, float]] = {}
    table_ok_values: list[float] = []

    total_rows = 0
    with genui_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = _loads(line)
            except Exception:
                continue
            genui_json = row.get("genui_json") or row.get("a2ui_json")
            response_id = row.get("response_id")
            query_id = row.get("query_id")
            response = responses_map.get(response_id, {})
            response_text = response.get("response_text", "")
            query_info = queries_map.get(query_id, {})
            intent_value = row.get("intent") or query_info.get("intent")
            tags_value = row.get("tags") or query_info.get("tags")
            if not isinstance(tags_value, list):
                tags_value = []

            if not genui_json:
                continue
            total_rows += 1

            row_metrics = row.get("metrics") or {}
            metrics = {
                "content_coverage": row_metrics.get("content_coverage"),
                "dup_rate": row_metrics.get("dup_rate"),
                "lint_score": row_metrics.get("lint_score"),
                "output_tokens_toon": row_metrics.get("output_tokens_toon"),
                "output_tokens_json": row_metrics.get("output_tokens_json"),
            }
            if metrics["content_coverage"] is None:
                metrics["content_coverage"] = (
                    content_coverage(response_text, genui_json) if response_text else 0.0
                )
            if metrics["dup_rate"] is None:
                metrics["dup_rate"] = dup_rate(genui_json)
            if metrics["lint_score"] is None:
                metrics["lint_score"] = lint_score(genui_json)
            if metrics["output_tokens_toon"] is None:
                metrics["output_tokens_toon"] = count_tokens(str(row.get("toon", "")))
            if metrics["output_tokens_json"] is None:
                metrics["output_tokens_json"] = count_tokens(json.dumps(genui_json, ensure_ascii=False))
            metrics.update(compute_ui_metrics(response_text, genui_json))
            intent_metrics = compute_intent_metrics(intent_value, tags_value, response_text, metrics)
            intent_bucket = intent_metrics.pop("intent_bucket", "unknown") or "unknown"
            metrics.update(intent_metrics)
            if metrics.get("intent_require_table", 0.0) >= 1.0:
                table_ok_values.append(_normalize_float(metrics.get("intent_table_ok", 0.0)))

            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + _normalize_float(value)
                counts[key] = counts.get(key, 0) + 1

            validation = row.get("validation", {})
            if isinstance(validation, dict):
                strict_ok = 1.0 if validation.get("schema_valid_strict") else 0.0
                sums["schema_valid_strict"] = sums.get("schema_valid_strict", 0.0) + strict_ok
                counts["schema_valid_strict"] = counts.get("schema_valid_strict", 0) + 1

            render = row.get("render")
            if isinstance(render, dict):
                render_values.append(1.0 if render.get("image_ok") else 0.0)

            gen = row.get("gen", {})
            if isinstance(gen, dict):
                latency_values.append(_normalize_float(gen.get("latency_ms", 0.0)))

            stats = intent_stats.setdefault(
                intent_bucket,
                {
                    "count": 0.0,
                    "expectation_pass": 0.0,
                    "intent_score": 0.0,
                    "require_table": 0.0,
                    "require_actions": 0.0,
                    "require_sections": 0.0,
                    "table_ok": 0.0,
                    "actions_ok": 0.0,
                    "sections_ok": 0.0,
                },
            )
            stats["count"] += 1.0
            stats["expectation_pass"] += metrics.get("intent_expectation_pass", 0.0)
            stats["intent_score"] += metrics.get("intent_score", 0.0)
            stats["require_table"] += metrics.get("intent_require_table", 0.0)
            stats["require_actions"] += metrics.get("intent_require_actions", 0.0)
            stats["require_sections"] += metrics.get("intent_require_sections", 0.0)
            stats["table_ok"] += metrics.get("intent_table_ok", 0.0)
            stats["actions_ok"] += metrics.get("intent_actions_ok", 0.0)
            stats["sections_ok"] += metrics.get("intent_sections_ok", 0.0)

    def mean(key: str) -> float:
        if counts.get(key, 0) == 0:
            return 0.0
        return sums.get(key, 0.0) / counts[key]

    def pct(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        values = sorted(values)
        k = int(math.ceil((p / 100.0) * len(values))) - 1
        k = max(0, min(k, len(values) - 1))
        return values[k]

    intent_stats_out = {}
    for intent_bucket, stats in intent_stats.items():
        count = stats.get("count", 0.0) or 0.0
        if count <= 0:
            continue
        intent_stats_out[intent_bucket] = {
            "count": int(count),
            "expectation_pass_rate": stats["expectation_pass"] / count,
            "intent_score_avg": stats["intent_score"] / count,
            "table_expected_rate": stats["require_table"] / count,
            "table_ok_rate": stats["table_ok"] / count,
            "action_expected_rate": stats["require_actions"] / count,
            "action_ok_rate": stats["actions_ok"] / count,
            "section_expected_rate": stats["require_sections"] / count,
            "section_ok_rate": stats["sections_ok"] / count,
        }

    aggregates = {
        "counts": total_rows,
        "schema_valid_strict_rate": mean("schema_valid_strict"),
        "content_coverage_avg": mean("content_coverage"),
        "lint_score_avg": mean("lint_score"),
        "dup_rate_avg": mean("dup_rate"),
        "component_count_avg": mean("component_count"),
        "unique_component_types_avg": mean("unique_component_types"),
        "max_tree_depth_avg": mean("max_tree_depth"),
        "avg_tree_depth_avg": mean("avg_tree_depth"),
        "container_to_text_ratio_avg": mean("container_to_text_ratio"),
        "information_chunking_score_avg": mean("information_chunking_score"),
        "ui_modularity_score_avg": mean("ui_modularity_score"),
        "ui_decomposition_score_avg": mean("ui_decomposition_score"),
        "actionable_elements_avg": mean("actionable_elements"),
        "action_coverage_avg": mean("action_coverage"),
        "url_as_text_rate_avg": mean("url_as_text_rate"),
        "table_pattern_detected_rate": mean("table_pattern_detected"),
        "table_cell_coverage_avg": mean("table_cell_coverage"),
        "section_heading_coverage_avg": mean("section_heading_coverage"),
        "markdown_leakage_rate_avg": mean("markdown_leakage_rate"),
        "missing_ids_avg": mean("missing_ids"),
        "missing_ids_rate_avg": mean("missing_ids_rate"),
        "dangling_components_avg": mean("dangling_components"),
        "dangling_components_rate_avg": mean("dangling_components_rate"),
        "intent_expectation_pass_rate": mean("intent_expectation_pass"),
        "intent_score_avg": mean("intent_score"),
        "table_required_rate": mean("intent_require_table"),
        "table_ok_rate": (sum(table_ok_values) / len(table_ok_values)) if table_ok_values else 0.0,
        "intent_stats": intent_stats_out or None,
        "render_ok_rate": sum(render_values) / len(render_values) if render_values else None,
        "latency_ms_avg": sum(latency_values) / len(latency_values) if latency_values else 0.0,
        "latency_ms_p95": pct(latency_values, 95) if latency_values else 0.0,
    }

    aggregates["overall_score"] = compute_overall_score(aggregates, weights_cfg)

    out_path = run_dir / "aggregates.json"
    out_path.write_text(json.dumps(aggregates, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
