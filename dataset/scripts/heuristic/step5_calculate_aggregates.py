#!/usr/bin/env python3
r"""
Standalone script to generate aggregates.json from genui.jsonl, render.jsonl, queries.jsonl, and responses.jsonl.
Uses the EXACT same logic as stage3_genui.py's _write_aggregates function.

Usage:
    python -m pipeline.generate_aggregates --run-dir <path> --output <path> --config <path>

Example (Windows):
    python -m pipeline.generate_aggregates --run-dir "C:\Users\adarsh.ag\Desktop\GenUI-LM\dataset\data\runs\RUN_ID" --output "C:\Users\adarsh.ag\Desktop\GenUI-LM\dataset\data\runs\RUN_ID\aggregates_test.json" --config "C:\Users\adarsh.ag\Desktop\GenUI-LM\dataset\configs\run.yaml"
"""

import json
import sys
from pathlib import Path
from typing import Any

# Add src directory to path to import pipeline modules
SCRIPT_DIR = Path(__file__).parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from pipeline.metrics import (
    aggregate_metrics,
    compute_overall_score,
    compute_media_score,
)
from pipeline.storage import iter_jsonl


def load_weights_from_yaml(config_path: Path) -> dict[str, float]:
    """Load evaluation weights from run.yaml config file."""
    import re

    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}, using default weights")
        return {}

    content = config_path.read_text(encoding="utf-8")
    weights: dict[str, float] = {}

    # Parse YAML manually (simple key: value extraction from evaluation.weights section)
    in_weights_section = False
    for line in content.splitlines():
        if "evaluation:" in line:
            in_weights_section = False
        if "weights:" in line:
            in_weights_section = True
            continue
        if in_weights_section:
            # Match lines like "    schema_valid_strict: 5.0"
            match = re.match(r"^\s+([a-z_]+):\s*([-\d.]+)", line)
            if match:
                key = match.group(1)
                value = float(match.group(2))
                weights[key] = value
            elif line.strip() and not line.strip().startswith("#"):
                # End of weights section if we hit another top-level key
                if not line.startswith(" ") and not line.startswith("\t"):
                    in_weights_section = False

    return weights


def generate_aggregates(
    run_dir: Path,
    output_path: Path,
    weights: dict[str, float],
) -> None:
    """
    Generate aggregates.json using the EXACT same logic as stage3_genui.py's _write_aggregates.

    This mimics every detail:
    1. Load response_text_by_id from responses.jsonl
    2. Load genui.jsonl rows
    3. Backfill response_text from responses.jsonl if not in genui row
    4. Load render.jsonl into render_rows_by_ui_id
    5. Call aggregate_metrics(rows, render_rows_by_ui_id=...)
    6. Compute overall_score and media_score
    7. Write JSON output
    """

    # Input file paths (same as stage3_genui.py)
    genui_path = run_dir / "genui.jsonl"
    responses_path = run_dir / "responses.jsonl"
    queries_path = run_dir / "queries.jsonl"
    render_log_path = run_dir / "render.jsonl"

    # Validate inputs
    if not genui_path.exists():
        raise FileNotFoundError(f"genui.jsonl not found at {genui_path}")

    # Step 1: Load response_text_by_id from responses.jsonl (EXACT as stage3_genui.py lines 587-592)
    response_text_by_id: dict[str, str] = {}
    if responses_path.exists():
        print(f"Loading responses.jsonl from {responses_path}...")
        for row in iter_jsonl(responses_path):
            response_id = row.get("response_id")
            response_text = row.get("response_text")
            if isinstance(response_id, str) and isinstance(response_text, str):
                response_text_by_id[response_id] = response_text
        print(f"Loaded {len(response_text_by_id)} response texts")

    # Step 2: Load genui.jsonl rows (EXACT as stage3_genui.py line 603)
    print(f"Loading genui.jsonl from {genui_path}...")
    rows = list(iter_jsonl(genui_path))
    print(f"Loaded {len(rows)} rows from genui.jsonl")

    # Step 3: Backfill response_text from responses.jsonl (EXACT as stage3_genui.py lines 604-612)
    if response_text_by_id:
        for row in rows:
            if row.get("response_text"):
                continue
            response_id = row.get("response_id")
            if isinstance(response_id, str):
                backfill = response_text_by_id.get(response_id)
                if isinstance(backfill, str):
                    row["response_text"] = backfill

    # Step 4: Load render.jsonl into render_rows_by_ui_id (EXACT as stage3_genui.py lines 613-619)
    render_rows_by_ui_id: dict[str, dict[str, Any]] = {}
    if render_log_path.exists():
        print(f"Loading render.jsonl from {render_log_path}...")
        for render_row in iter_jsonl(render_log_path):
            ui_id = render_row.get("ui_id")
            if isinstance(ui_id, str) and ui_id:
                render_rows_by_ui_id[ui_id] = render_row
        print(f"Loaded {len(render_rows_by_ui_id)} render entries")

    # Step 5: Load queries.jsonl for intent/tags (same pattern as stage3_genui.py lines 572-581)
    intent_lookup: dict[str, dict[str, Any]] = {}
    if queries_path.exists():
        print(f"Loading queries.jsonl from {queries_path}...")
        for row in iter_jsonl(queries_path):
            query_id = row.get("query_id")
            if not query_id:
                continue
            intent_lookup[query_id] = {
                "intent": row.get("intent"),
                "tags": row.get("tags") if isinstance(row.get("tags"), list) else [],
            }
        print(f"Loaded {len(intent_lookup)} query intents")

        # Enrich genui rows with intent data
        for row in rows:
            query_id = row.get("query_id")
            if query_id and query_id in intent_lookup:
                row["intent"] = intent_lookup[query_id]["intent"]
                row["tags"] = intent_lookup[query_id]["tags"]

    # Step 6: Call aggregate_metrics (EXACT as stage3_genui.py line 620)
    print("Computing aggregate metrics...")
    aggregates = aggregate_metrics(rows, render_rows_by_ui_id=render_rows_by_ui_id)

    # Step 7: Compute overall_score (EXACT as stage3_genui.py lines 621-624)
    print("Computing overall score...")
    aggregates["overall_score"] = compute_overall_score(
        aggregates,
        weights,
    )

    # Step 8: Compute media_score (EXACT as stage3_genui.py line 625)
    print("Computing media score...")
    aggregates["media_score"] = compute_media_score(aggregates)

    # Step 9: Add per-row details to output
    print("Adding per-row details...")
    row_details = []
    for row in rows:
        row_detail = {
            "ui_id": row.get("ui_id"),
            "response_id": row.get("response_id"),
            "query_id": row.get("query_id"),
            "intent": row.get("intent"),
            "intent_bucket": row.get("intent_bucket"),
            "tags": row.get("tags", []),
            "validation": row.get("validation", {}),
            "metrics": row.get("metrics", {}),
            "gen": row.get("gen", {}),
            "created_at": row.get("created_at"),
        }
        row_details.append(row_detail)
    aggregates["rows"] = row_details

    # Step 10: Write JSON output (EXACT as stage3_genui.py line 626)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(aggregates, indent=2), encoding="utf-8")
    print(f"Aggregates written to {output_path}")

    return aggregates


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate aggregates.json using the exact same logic as stage3_genui.py's _write_aggregates"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to run directory containing genui.jsonl, responses.jsonl, queries.jsonl, render.jsonl"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for aggregates.json"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to run.yaml config file for weights (optional)"
    )

    args = parser.parse_args()

    # Load weights from config if provided
    weights: dict[str, float] = {}
    if args.config:
        print(f"Loading weights from {args.config}...")
        weights = load_weights_from_yaml(args.config)
        print(f"Loaded {len(weights)} weight entries")
    else:
        print("No config file provided, using default weights")

    # Generate aggregates
    aggregates = generate_aggregates(args.run_dir, args.output, weights)

    print("\n=== Summary ===")
    print(f"Total entries: {aggregates.get('counts', 'N/A')}")
    print(f"Overall score: {(aggregates.get('overall_score', 'N/A')):.2f}")
    print(f"Media score: {aggregates.get('media_score', 'N/A')}")
    print(f"Schema valid rate: {aggregates.get('schema_valid_strict_rate', 'N/A'):.4f}")
    print(f"Content coverage avg: {aggregates.get('content_coverage_avg', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
