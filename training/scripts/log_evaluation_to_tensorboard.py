from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.eval.tensorboard_logging import log_evaluation_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log an existing evaluation aggregate to TensorBoard and an audit JSON sidecar."
    )
    parser.add_argument("--aggregate", required=True, help="aggregate_metrics.json to log.")
    parser.add_argument("--tensorboard-root", default="tensorboard")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evaluation-name", required=True)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Attach a named artifact identity; repeat as needed.",
    )
    args = parser.parse_args()

    aggregate_path = Path(args.aggregate).expanduser().resolve()
    metrics = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict):
        raise ValueError("Evaluation aggregate must be a JSON object.")
    artifacts = _parse_artifacts(args.artifact)
    record = log_evaluation_result(
        args.tensorboard_root,
        run_id=args.run_id,
        evaluation_name=args.evaluation_name,
        metrics=metrics,
        step=args.step,
        artifacts=artifacts,
        source_aggregate_path=aggregate_path,
    )
    print(json.dumps(record, indent=2, ensure_ascii=False))


def _parse_artifacts(values: list[str]) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name.strip() or not path.strip():
            raise ValueError(f"Invalid --artifact {value!r}; expected NAME=PATH.")
        if name.strip() in artifacts:
            raise ValueError(f"Duplicate --artifact name: {name.strip()}")
        artifacts[name.strip()] = path.strip()
    return artifacts


if __name__ == "__main__":
    main()
