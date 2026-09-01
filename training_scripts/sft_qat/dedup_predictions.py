#!/usr/bin/env python3
"""Deduplicate predictions.jsonl and run heuristic pipeline.

This script:
1. Filters out metadata rows and invalid entries
2. Deduplicates by (response_id, query_id) pair - keeps first occurrence
3. Saves deduplicated output as updated_genui.jsonl in pipeline-compatible format
4. Runs heuristic pipeline automatically on the deduplicated file

Output format matches what step1_genui_raw.py expects:
- response_id: unique response identifier
- query_id: unique query identifier
- genui_json: the generated JSON response (must be valid JSON string with {"root"...})
- prediction: the raw model output (for reference)

Usage:
    python dedup_predictions.py \
        --input predictions.jsonl \
        --output_dir ./output \
        --util-folder /path/to/golden_100_test \
        --config /path/to/run.yaml \
        --heuristic-script-dir /path/to/dataset/scripts
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def extract_response_id(entry):
    """Extract response_id from entry (can be in multiple fields)."""
    # Check top-level fields first
    for key in ("response_id", "responseId", "id", "sample_id"):
        if key in entry and entry[key]:
            return str(entry[key])

    # Try extracting from reference JSON
    reference = entry.get("reference", "")
    if reference:
        try:
            if isinstance(reference, str):
                ref_data = json.loads(reference)
                if isinstance(ref_data, dict):
                    for key in ("response_id", "responseId", "id"):
                        if key in ref_data and ref_data[key]:
                            return str(ref_data[key])
        except (json.JSONDecodeError, TypeError):
            pass

    return None


def extract_query_id(entry):
    """Extract query_id from entry (can be in multiple fields)."""
    # Check top-level fields first
    for key in ("query_id", "queryId", "query", "input_id"):
        if key in entry and entry[key]:
            return str(entry[key])

    # Try extracting from reference JSON
    reference = entry.get("reference", "")
    if reference:
        try:
            if isinstance(reference, str):
                ref_data = json.loads(reference)
                if isinstance(ref_data, dict):
                    for key in ("query_id", "queryId", "query"):
                        if key in ref_data and ref_data[key]:
                            return str(ref_data[key])
        except (json.JSONDecodeError, TypeError):
            pass

    return None


def extract_genui_json(entry):
    """Extract genui_json from entry (prediction or reference)."""
    # Check if prediction contains valid JSON
    prediction = entry.get("prediction", "")
    if prediction:
        # Check if it looks like JSON
        if isinstance(prediction, str) and prediction.strip().startswith("{"):
            try:
                # Validate it's proper JSON
                json.loads(prediction)
                return prediction
            except (json.JSONDecodeError, TypeError):
                pass
        # If prediction is already a dict, serialize it
        if isinstance(prediction, dict):
            return json.dumps(prediction, ensure_ascii=False)

    # Try reference field
    reference = entry.get("reference", "")
    if reference:
        if isinstance(reference, str) and reference.strip().startswith("{"):
            try:
                json.loads(reference)
                return reference
            except (json.JSONDecodeError, TypeError):
                pass
        if isinstance(reference, dict):
            return json.dumps(reference, ensure_ascii=False)

    return None


def deduplicate_predictions(input_path, output_path):
    """Deduplicate predictions by response_id and query_id, keeping first occurrence."""
    print(f"\nReading predictions from: {input_path}")

    valid_entries = []
    seen_pairs = set()  # (response_id, query_id) pairs
    skipped_metadata = 0
    skipped_duplicate = 0
    skipped_invalid = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Line {line_num}: Invalid JSON - {e}")
                continue

            # Extract IDs for deduplication
            response_id = extract_response_id(entry)
            query_id = extract_query_id(entry)

            # For predictions from eval_quantized.py, use index as fallback
            if response_id is None and query_id is None:
                if "index" in entry:
                    unique_key = f"idx_{entry.get('index', line_num):05d}"
                else:
                    skipped_metadata += 1
                    continue
            else:
                unique_key = f"{response_id or ''}_{query_id or ''}"
                if not unique_key.strip('_'):
                    skipped_metadata += 1
                    continue

            # Check for duplicates
            if unique_key in seen_pairs:
                skipped_duplicate += 1
                print(f"  Skipping duplicate: {unique_key}")
                continue

            # Extract genui_json (required for pipeline)
            genui_json = extract_genui_json(entry)
            if genui_json is None:
                skipped_invalid += 1
                print(f"  Skipping invalid (no genui_json): {unique_key}")
                continue

            seen_pairs.add(unique_key)

            # Convert to pipeline-compatible format
            output_entry = {
                "response_id": response_id or f"idx_{entry.get('index', line_num):05d}",
                "query_id": query_id or "",
                "genui_json": genui_json,
                "prediction": entry.get("prediction", ""),
                "reference": entry.get("reference", ""),
            }

            # Preserve other fields
            for k, v in entry.items():
                if k not in output_entry:
                    output_entry[k] = v

            valid_entries.append(output_entry)

    # Print summary
    print(f"\n{'=' * 60}")
    print("DEDUPLICATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Input entries:        {line_num}")
    print(f"  Skipped (metadata):   {skipped_metadata}")
    print(f"  Skipped (duplicate):  {skipped_duplicate}")
    print(f"  Skipped (invalid):    {skipped_invalid}")
    print(f"  Output entries:       {len(valid_entries)}")
    print(f"  Unique (response_id, query_id) pairs: {len(seen_pairs)}")
    print(f"{'=' * 60}")

    # Write output file
    print(f"\nWriting deduplicated predictions to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in valid_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return len(valid_entries)


def run_heuristic_pipeline(input_path, output_dir, util_folder, config_path, script_dir):
    """Run the heuristic pipeline on deduplicated predictions."""
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
        description="Deduplicate predictions and run heuristic pipeline"
    )

    # Input/Output
    parser.add_argument("--input", type=str, required=True,
                        help="Input predictions.jsonl file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for deduplicated file and aggregates")

    # Heuristic pipeline
    parser.add_argument("--util-folder", type=str, required=True,
                        help="Path to golden util folder (e.g., golden_100_test)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to run.yaml config file")
    parser.add_argument("--heuristic-script-dir", type=str, default="",
                        help="Directory containing run_heuristic_pipeline.py")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("DEDUP PREDICTIONS + HEURISTIC EVAL")
    print("=" * 60)
    print(f"  Input:          {args.input}")
    print(f"  Output dir:     {args.output_dir}")
    print(f"  Util folder:    {args.util_folder}")
    print(f"  Config:         {args.config or 'default'}")
    print("=" * 60)

    # Step 1: Deduplicate predictions
    output_path = os.path.join(args.output_dir, "updated_genui.jsonl")
    deduplicate_predictions(args.input, output_path)

    # Step 2: Run heuristic pipeline
    aggregates = run_heuristic_pipeline(
        input_path=output_path,
        output_dir=args.output_dir,
        util_folder=args.util_folder,
        config_path=args.config,
        script_dir=args.heuristic_script_dir,
    )

    if aggregates:
        print("\n" + "=" * 60)
        print("HEURISTIC METRICS (from deduplicated data)")
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
        summary_path = os.path.join(args.output_dir, "dedup_eval_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "input_file": args.input,
                "dedup_output": output_path,
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
