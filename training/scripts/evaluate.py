from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATASET_SRC = ROOT.parent / "dataset" / "src"
if DATASET_SRC.exists():
    sys.path.insert(0, str(DATASET_SRC))

from ir_training.eval.compare_to_baseline import evaluate_predictions
from ir_training.eval.generate import generate_predictions
from ir_training.common.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and/or evaluate IR model predictions.")
    parser.add_argument("--config", help="Optional model config for generation.")
    parser.add_argument("--split", help="Split JSONL to generate from.")
    parser.add_argument("--predictions", help="Existing predictions JSONL to score.")
    parser.add_argument("--output-dir", default="training/outputs/eval", help="Directory for evaluation artifacts.")
    parser.add_argument("--weights-config", default="dataset/configs/run.yaml", help="Dataset run YAML containing evaluation.weights.")
    parser.add_argument("--baseline-aggregate", help="Optional baseline aggregates.json for overall score delta.")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    predictions_path = Path(args.predictions) if args.predictions else output_dir / "predictions.jsonl"
    if args.config and args.split:
        config = load_yaml(Path(args.config).resolve())
        generated = generate_predictions(config, args.split, predictions_path, max_rows=args.max_rows)
        print(f"generated={generated} predictions_path={predictions_path}")
    if not predictions_path.exists():
        raise SystemExit(f"Missing predictions file: {predictions_path}")
    aggregate = evaluate_predictions(
        predictions_path,
        output_dir=output_dir,
        weights_config_path=args.weights_config,
        baseline_aggregate_path=args.baseline_aggregate,
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
