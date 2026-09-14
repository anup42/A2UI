"""Built-in target-blind LiteRT-LM GPU runtime worker and prerequisite probe."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "dataset" / "src"))

from ir_training.eval.litert_gpu import run_gpu_worker, runtime_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help="Internal bounded subprocess mode")
    parser.add_argument("--preflight", action="store_true", help="Check API/version/NVIDIA prerequisites only")
    parser.add_argument("--model")
    parser.add_argument("--requests", help="Hash-bound a2ui_external_generation_v1 JSONL")
    parser.add_argument("--outputs")
    parser.add_argument("--cache-dir")
    parser.add_argument("--report", help="Write a machine-readable prerequisite/worker report JSON")
    args = parser.parse_args()
    if args.preflight == args.worker:
        parser.error("Choose exactly one of --preflight or --worker")
    if args.worker and not all((args.model, args.requests, args.outputs)):
        parser.error("--worker requires --model, --requests and --outputs")
    try:
        if args.preflight:
            result = runtime_preflight()
        else:
            result = run_gpu_worker(model_path=args.model, requests_path=args.requests,
                                    outputs_path=args.outputs, cache_dir=args.cache_dir)
    except Exception as exc:
        if args.report:
            report = Path(args.report).expanduser().resolve()
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps({"status": "failed", "error": str(exc)}, indent=2), encoding="utf-8")
        raise
    if args.report:
        report = Path(args.report).expanduser().resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
