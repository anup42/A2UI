#!/usr/bin/env python3
"""Filter genui.jsonl to keep only ui_id < ui_000100 and run heuristic pipeline.

This script:
1. Filters input file to keep only rows where ui_id < ui_000100
2. Saves filtered output as updated_genui.jsonl
3. Runs heuristic pipeline automatically on the filtered file

Usage:
    python3 filter_genui_and_eval.py \
        --input /path/to/genui.jsonl \
        --output_dir /path/to/output \
        --heuristic-util-folder /path/to/golden_100_test \
        --heuristic-config /path/to/run.yaml \
        --heuristic-script-dir /path/to/dataset/scripts
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def parse_ui_id(ui_id):
    """Extract numeric value from ui_id string (e.g., 'ui_000061' -> 61)."""
    if not ui_id:
        return None
    match = re.search(r'ui[_-]?(\d+)', ui_id, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def filter_genui_file(input_path, output_path, max_ui_id=100):
    """Filter genui.jsonl: remove rows where ui_id >= max_ui_id. That's it."""
    print(f"\nReading input file: {input_path}")

    kept = 0
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as f_in:
        with open(output_path, "w", encoding="utf-8") as f_out:
            for line_num, line in enumerate(f_in, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ui_id = entry.get("ui_id")
                ui_num = parse_ui_id(ui_id)

                if ui_num is not None and ui_num >= max_ui_id:
                    skipped += 1
                    continue

                f_out.write(line + "\n")
                kept += 1

    print(f"\n{'=' * 60}")
    print("FILTER SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Input file:       {input_path}")
    print(f"  Output file:      {output_path}")
    print(f"  Max ui_id:        < ui_{max_ui_id:06d}")
    print(f"  Kept:             {kept}")
    print(f"  Skipped:          {skipped}")
    print(f"{'=' * 60}")

    return kept


def run_heuristic_pipeline(input_path, output_dir, util_folder, config_path, script_dir):
    """Run the heuristic pipeline on filtered predictions."""
    print(f"\n{'=' * 60}")
    print("RUNNING HEURISTIC PIPELINE")
    print(f"{'=' * 60}")

    # Auto-detect script dir if not provided
    if not script_dir:
        script_dir = str(Path(__file__).resolve().parents[4] / "dataset" / "scripts")
        print(f"  Auto-detected script dir: {script_dir}")

    pipeline_script = os.path.join(script_dir, "run_heuristic_pipeline.py")
    if not os.path.exists(pipeline_script):
        print(f"  ERROR: run_heuristic_pipeline.py not found at {pipeline_script}")
        return None

    cmd = [
        sys.executable,
        pipeline_script,
        "--input", str(input_path),
        "--util-folder", str(util_folder),
    ]
    if config_path:
        cmd.extend(["--config", str(config_path)])

    print(f"  Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"  ERROR: Heuristic pipeline failed!")
        return None

    # Read aggregates.json
    aggregates_path = os.path.join(output_dir, "aggregates.json")
    if os.path.exists(aggregates_path):
        with open(aggregates_path, "r", encoding="utf-8") as f:
            aggregates = json.load(f)
        print(f"\n  Aggregates loaded from {aggregates_path}")
        return aggregates
    else:
        print(f"  WARNING: aggregates.json not found at {aggregates_path}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Filter genui.jsonl (ui_id < 100) and run heuristic pipeline"
    )

    # Input/Output
    parser.add_argument("--input", type=str, required=True,
                        help="Input genui.jsonl file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for filtered file and aggregates")
    parser.add_argument("--max-ui-id", type=int, default=100,
                        help="Maximum ui_id to keep (exclusive, default: 100)")

    # Heuristic pipeline
    parser.add_argument("--heuristic-util-folder", type=str, required=True,
                        help="Path to golden util folder (e.g., golden_100_test)")
    parser.add_argument("--heuristic-config", type=str, default=None,
                        help="Path to run.yaml config file")
    parser.add_argument("--heuristic-script-dir", type=str, default="",
                        help="Directory containing run_heuristic_pipeline.py")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("FILTER GENUI + HEURISTIC EVAL")
    print("=" * 60)
    print(f"  Input:              {args.input}")
    print(f"  Output dir:         {args.output_dir}")
    print(f"  Max ui_id:          < ui_{args.max_ui_id:06d}")
    print(f"  Util folder:        {args.heuristic_util_folder}")
    print(f"  Config:             {args.heuristic_config or 'default'}")
    print(f"  Script dir:         {args.heuristic_script_dir or 'auto-detect'}")
    print("=" * 60)

    # Step 1: Filter genui file
    output_path = os.path.join(args.output_dir, "updated_genui.jsonl")
    filter_genui_file(args.input, output_path, max_ui_id=args.max_ui_id)

    # Step 2: Run heuristic pipeline
    aggregates = run_heuristic_pipeline(
        input_path=output_path,
        output_dir=args.output_dir,
        util_folder=args.heuristic_util_folder,
        config_path=args.heuristic_config,
        script_dir=args.heuristic_script_dir,
    )

    if aggregates:
        print("\n" + "=" * 60)
        print("HEURISTIC METRICS (from filtered data)")
        print("=" * 60)
        print(f"  overall_score:           {aggregates.get('overall_score', 'N/A')}")
        print(f"  media_score:              {aggregates.get('media_score', 'N/A')}")
        print(f"  schema_valid_strict_rate: {aggregates.get('schema_valid_strict_rate', 'N/A')}")
        print(f"  content_coverage_avg:     {aggregates.get('content_coverage_avg', 'N/A')}")
        print(f"  intent_score_avg:         {aggregates.get('intent_score_avg', 'N/A')}")
        print(f"  action_coverage_avg:      {aggregates.get('action_coverage_avg', 'N/A')}")
        print(f"  lint_score_avg:            {aggregates.get('lint_score_avg', 'N/A')}")
        print(f"  dup_rate_avg:             {aggregates.get('dup_rate_avg', 'N/A')}")
        print("=" * 60)

        # Save summary
        summary_path = os.path.join(args.output_dir, "filter_eval_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "input_file": args.input,
                "filtered_output": output_path,
                "max_ui_id": f"< ui_{args.max_ui_id:06d}",
                "num_samples": aggregates.get("num_samples"),
                "overall_score": aggregates.get("overall_score"),
                "media_score": aggregates.get("media_score"),
                "schema_valid_strict_rate": aggregates.get("schema_valid_strict_rate"),
                "content_coverage_avg": aggregates.get("content_coverage_avg"),
                "intent_score_avg": aggregates.get("intent_score_avg"),
                "action_coverage_avg": aggregates.get("action_coverage_avg"),
                "lint_score_avg": aggregates.get("lint_score_avg"),
                "dup_rate_avg": aggregates.get("dup_rate_avg"),
            }, f, indent=2)
        print(f"\n  Summary saved to: {summary_path}")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
