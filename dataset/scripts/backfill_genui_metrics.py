import argparse
import json
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
)
from pipeline.storage import iter_jsonl  # noqa: E402


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
    parser.add_argument("--run-id", type=str, required=True, help="Run id or path to run directory")
    parser.add_argument("--output", type=str, default=None, help="Output jsonl path (default overwrites)")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    run_dir = _resolve_run_dir(runs_dir, args.run_id).resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    responses_map = _load_map(run_dir / "responses.jsonl", "response_id")
    queries_map = _load_map(run_dir / "queries.jsonl", "query_id")

    genui_path = run_dir / "genui.jsonl"
    if not genui_path.exists():
        raise SystemExit(f"genui.jsonl not found in {run_dir}")

    backup_path = genui_path.with_suffix(".jsonl.bak")
    if not backup_path.exists():
        backup_path.write_text(genui_path.read_text(encoding="utf-8"), encoding="utf-8")

    output_path = Path(args.output).resolve() if args.output else genui_path.with_suffix(".jsonl.tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle = output_path.open("w", encoding="utf-8")

    for row in iter_jsonl(genui_path):
        response_id = row.get("response_id")
        query_id = row.get("query_id")
        response = responses_map.get(response_id, {})
        response_text = response.get("response_text", "")
        query_info = queries_map.get(query_id, {})

        genui_json = row.get("genui_json") or row.get("a2ui_json")
        if not genui_json or not response_text:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            continue

        metrics = row.get("metrics", {}) or {}
        metrics["content_coverage"] = content_coverage(response_text, genui_json)
        metrics["dup_rate"] = dup_rate(genui_json)
        metrics["lint_score"] = lint_score(genui_json)
        metrics["output_tokens_toon"] = metrics.get(
            "output_tokens_toon", count_tokens(str(row.get("toon", "")))
        )
        metrics["output_tokens_json"] = count_tokens(
            json.dumps(genui_json, ensure_ascii=False)
        )
        metrics.update(compute_ui_metrics(response_text, genui_json))

        intent_value = row.get("intent") or query_info.get("intent")
        tags_value = row.get("tags") or query_info.get("tags")
        if not isinstance(tags_value, list):
            tags_value = []
        intent_metrics = compute_intent_metrics(intent_value, tags_value, response_text, metrics)
        intent_bucket = intent_metrics.pop("intent_bucket", row.get("intent_bucket"))
        metrics.update(intent_metrics)

        row["metrics"] = metrics
        row["intent"] = intent_value
        row["tags"] = tags_value
        row["intent_bucket"] = intent_bucket
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    handle.close()
    if output_path != genui_path:
        output_path.replace(genui_path)
    print(f"Backfilled metrics for {run_dir}")


if __name__ == "__main__":
    main()
