#!/usr/bin/env python3
"""Run the verified GPT30K Gemma 3 270M SFT, W8 export, and native evaluations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ir_training.pipeline.golden_training import sha256
from run_golden_deployment import main as deploy

SPLIT_HASHES = {
    "train.jsonl": "d2e93cbdb58b6b1b4937829e9cfabd38cf17c3c8fe3817aa7283620b5921782a",
    "val.jsonl": "4bf7512d2aaf46f8b2454f4b0c23cac9a84f36125ab5fd54a74fd9ea5739e585",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True, help="Extracted verified GPT30K dataset")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exporter-python", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Execute on the GPU host; defaults to a read-only plan")
    parser.add_argument("--resume-run", action="store_true", help="Reuse verified stages after training")
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser()
    for filename, expected in SPLIT_HASHES.items():
        path = input_dir / filename
        if not path.is_file() or sha256(path) != expected:
            parser.error(f"Missing or changed GPT30K split: {path}")

    argv = [
        "--model-dir", str(args.model_dir), "--input-dir", str(input_dir), "--output-dir", str(args.output_dir),
        "--exporter-python", str(args.exporter_python), "--runtime-python", str(args.runtime_python),
        "--epochs", "1", "--max-seq-length", "8192", "--max-input-tokens", "8192", "--max-new-tokens", "2048",
        "--cache-length", "10240", "--variants", "w8",
    ]
    if args.execute:
        argv.append("--execute")
    if args.resume_run:
        argv.append("--resume-run")
    return deploy(argv, required_profile="270m")


if __name__ == "__main__":
    raise SystemExit(main())
