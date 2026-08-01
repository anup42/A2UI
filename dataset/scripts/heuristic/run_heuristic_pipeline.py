#!/usr/bin/env python3
r"""
Main pipeline script that orchestrates steps 0-5 for processing predictions.jsonl.

Usage:
    python run_pipeline.py --input <path_to_predictions.jsonl> --util-folder <path_to_util_folder> [--config <path_to_run.yaml>]

Example:
    python run_pipeline.py --input "C:\Users\adarsh.ag\Desktop\GenUI-LM\dataset\data\runs\golden50_qwen3_1.7b\predictions.jsonl" --util-folder "C:\Users\adarsh.ag\Desktop\GenUI-LM\dataset\data\runs\golden50_qwen3_0.6" --config "C:\Users\adarsh.ag\Desktop\GenUI-LM\dataset\configs\run.yaml"
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(step_name: str, cmd: list[str]) -> bool:
    """Run a pipeline step and return True if successful."""
    print(f"\n{'='*60}")
    print(f"Running Step: {step_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n[ERROR] Step {step_name} failed!")
        return False

    print(f"\n[SUCCESS] Step {step_name} completed!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete GenUI pipeline (steps 0-5)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to predictions.jsonl file"
    )
    parser.add_argument(
        "--util-folder",
        type=Path,
        required=True,
        help="Path to util folder containing queries.jsonl, responses.jsonl, golden50_url_mapping.jsonl"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to run.yaml config file (for step 5)"
    )
    parser.add_argument(
        "--skip-steps",
        type=str,
        default="",
        help="Comma-separated list of steps to skip (e.g., '0,2,4')"
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Start from this step number (0-5)"
    )

    args = parser.parse_args()

    # Validate input file
    if not args.input.exists():
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    # Get the run directory (parent of predictions.jsonl)
    run_dir = args.input.parent
    script_dir = Path(__file__).parent

    # Determine which steps to skip
    skip_steps = set()
    if args.skip_steps:
        skip_steps = set(int(s.strip()) for s in args.skip_steps.split(","))

    # Determine starting step
    start_from = args.start_from

    print(f"\n{'#'*60}")
    print(f"# GenUI Pipeline - Complete Processing")
    print(f"{'#'*60}")
    print(f"Input file: {args.input}")
    print(f"Run directory: {run_dir}")
    print(f"Config file: {args.config if args.config else 'Not provided'}")
    print(f"Steps to skip: {skip_steps if skip_steps else 'None'}")
    print(f"Starting from step: {start_from}")

    # Track overall success
    all_success = True

    # ========== STEP 0: Rename prediction keys to genui_json ==========
    if 0 in skip_steps or start_from > 0:
        print(f"\n[SKIP] Step 0: Skipping as requested")
    else:
        step0_cmd = [
            sys.executable,
            str(script_dir / "step0_rename.py"),
            str(args.input)
        ]
        if not run_step("0 - Rename prediction keys", step0_cmd):
            all_success = False

    # After step 0, the file is renamed to genui.jsonl
    genui_path = run_dir / "genui.jsonl"

    # ========== STEP 1: Copy utility files ==========
    if 1 in skip_steps or start_from > 1:
        print(f"\n[SKIP] Step 1: Skipping as requested")
    else:
        step1_cmd = [
            sys.executable,
            str(script_dir / "step2_copyutils.py"),
            str(args.util_folder),
            str(run_dir)
        ]
        if not run_step("1 - Copy utility files", step1_cmd):
            all_success = False

    # ========== STEP 2: Fix genui_json raw values ==========
    if 2 in skip_steps or start_from > 2:
        print(f"\n[SKIP] Step 2: Skipping as requested")
    else:
        step2_cmd = [
            sys.executable,
            str(script_dir / "step1_genui_raw.py"),
            str(genui_path)
        ]
        if not run_step("2 - Fix genui_json raw values", step2_cmd):
            all_success = False

    # ========== STEP 3: Fix genui_json strings (convert to objects) ==========
    if 3 in skip_steps or start_from > 3:
        print(f"\n[SKIP] Step 3: Skipping as requested")
    else:
        step3_cmd = [
            sys.executable,
            str(script_dir / "step3_fix_genui_json_strings.py"),
            str(genui_path)
        ]
        if not run_step("3 - Fix genui_json strings", step3_cmd):
            all_success = False

    # ========== STEP 4: Fix predictions (add validation, metrics, etc.) ==========
    if 4 in skip_steps or start_from > 4:
        print(f"\n[SKIP] Step 4: Skipping as requested")
    else:
        step4_cmd = [
            sys.executable,
            str(script_dir / "step4_fix_genui_predictions_v2.py"),
            str(run_dir),
            "--url_unmask"
        ]
        if not run_step("4 - Fix predictions (add validation/metrics)", step4_cmd):
            all_success = False

    # ========== STEP 5: Calculate aggregates ==========
    if 5 in skip_steps or start_from > 5:
        print(f"\n[SKIP] Step 5: Skipping as requested")
    else:
        aggregates_output = run_dir / "aggregates.json"
        step5_cmd = [
            sys.executable,
            str(script_dir / "step5_calculate_aggregates.py"),
            "--run-dir", str(run_dir),
            "--output", str(aggregates_output)
        ]
        if args.config:
            step5_cmd.extend(["--config", str(args.config)])

        if not run_step("5 - Calculate aggregates", step5_cmd):
            all_success = False

    # ========== FINAL SUMMARY ==========
    print(f"\n{'#'*60}")
    if all_success:
        print(f"# Pipeline completed successfully!")
        print(f"{'#'*60}")
        print(f"\nOutput files:")
        print(f"  - {genui_path}")
        print(f"  - {run_dir / 'aggregates.json'}")
    else:
        print(f"# Pipeline completed with errors!")
        print(f"{'#'*60}")
        print("\nSome steps failed. Please review the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
