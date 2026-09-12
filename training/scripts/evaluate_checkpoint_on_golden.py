from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "dataset" / "src"))

from ir_training.common.config import load_yaml
from ir_training.eval.compare_to_baseline import evaluate_predictions
from ir_training.eval.generate import generate_predictions
from ir_training.eval.golden_set import load_fixed_golden_rows
from ir_training.eval.tensorboard_logging import log_evaluation_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a merged HF model or PEFT adapter on a fixed A2UI Express "
            "Golden set, then log the complete result to TensorBoard."
        )
    )
    parser.add_argument("--config", required=True, help="Base training/model YAML.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--checkpoint-kind",
        choices=("auto", "adapter", "merged"),
        default="auto",
    )
    parser.add_argument(
        "--qat-mode",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "auto reapplies configured fake QAT for adapter checkpoints and "
            "leaves merged checkpoints dense."
        ),
    )
    parser.add_argument(
        "--split",
        default=str(
            ROOT
            / "outputs"
            / "datasets"
            / "golden32_20260903_eval_prompt_v2"
            / "all.jsonl"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tensorboard-root", default="tensorboard")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evaluation-name", default="checkpoint")
    parser.add_argument("--step", type=int)
    parser.add_argument("--max-rows", type=int, default=32)
    parser.add_argument("--required-rows", type=int, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--weights-config",
        default=str(REPO_ROOT / "dataset" / "configs" / "run.yaml"),
    )
    parser.add_argument("--baseline-aggregate")
    parser.add_argument(
        "--metric-version", choices=("legacy", "v5_4", "dual"), default="dual"
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory is missing: {checkpoint}")
    split = Path(args.split).expanduser().resolve()
    load_fixed_golden_rows(
        split,
        max_rows=args.max_rows,
        required_rows=args.required_rows,
        require_exact_rows=True,
        require_unique_rows=True,
    )
    kind = _checkpoint_kind(checkpoint, args.checkpoint_kind)
    config_path = Path(args.config).expanduser().resolve()
    config = copy.deepcopy(load_yaml(config_path))
    model_cfg = config.setdefault("model", {})
    if not isinstance(model_cfg, dict):
        raise ValueError("config.model must be a YAML object.")
    adapter_checkpoint: Path | None = None
    if kind == "adapter":
        adapter_checkpoint = checkpoint
    else:
        if not (checkpoint / "config.json").is_file():
            raise FileNotFoundError(
                f"Merged checkpoint is missing config.json: {checkpoint}"
            )
        model_cfg["model_source"] = str(checkpoint)
        if _contains_tokenizer(checkpoint):
            model_cfg["tokenizer_source"] = str(checkpoint)
    qat_cfg = config.get("qat") if isinstance(config.get("qat"), dict) else {}
    apply_qat = (
        bool(qat_cfg.get("enabled", False)) and kind == "adapter"
        if args.qat_mode == "auto"
        else args.qat_mode == "on"
    )
    if args.qat_mode == "on" and not bool(qat_cfg.get("enabled", False)):
        raise ValueError("--qat-mode on requires qat.enabled=true in --config.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    generated = generate_predictions(
        config,
        split,
        predictions_path,
        max_rows=args.max_rows,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        adapter_checkpoint=adapter_checkpoint,
        apply_qat=apply_qat,
    )
    if generated != args.required_rows:
        raise RuntimeError(
            f"Checkpoint evaluation generated {generated} rows; "
            f"expected {args.required_rows}."
        )
    aggregate = evaluate_predictions(
        predictions_path,
        output_dir=output_dir,
        weights_config_path=args.weights_config,
        baseline_aggregate_path=args.baseline_aggregate,
        metric_version=args.metric_version,
    )
    aggregate_path = output_dir / "aggregate_metrics.json"
    step = args.step if args.step is not None else _checkpoint_step(checkpoint)
    record = log_evaluation_result(
        args.tensorboard_root,
        run_id=args.run_id,
        evaluation_name=args.evaluation_name,
        metrics=aggregate,
        step=step,
        artifacts={
            "checkpoint": checkpoint,
            "training_config": config_path,
            "golden_split": split,
            "predictions": predictions_path,
            "scored_predictions": output_dir / "scored_predictions.jsonl",
            "aggregate_metrics": aggregate_path,
        },
        metadata={
            "checkpoint_kind": kind,
            "config": str(config_path),
            "split": str(split),
            "rows": generated,
            "metric_version": args.metric_version,
            "qat_mode": args.qat_mode,
            "qat_applied": apply_qat,
        },
        source_aggregate_path=aggregate_path,
    )
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_kind": kind,
        "row_count": generated,
        "qat_applied": apply_qat,
        "aggregate": aggregate,
        "tensorboard_record": record["record_path"],
    }
    (output_dir / "evaluation_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _checkpoint_kind(checkpoint: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    return "adapter" if (checkpoint / "adapter_config.json").is_file() else "merged"


def _contains_tokenizer(checkpoint: Path) -> bool:
    return any(
        (checkpoint / name).is_file()
        for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
    )


def _checkpoint_step(checkpoint: Path) -> int:
    metadata_path = checkpoint / "training_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        role = metadata.get("checkpoint_role")
        best = metadata.get("best_golden_eval") or {}
        if role == "best_golden" or checkpoint.name == "best_golden_checkpoint":
            step = best.get("step", metadata.get("checkpoint_step"))
        else:
            step = metadata.get("checkpoint_step")
        if type(step) is int and step >= 0:
            return step
    for name in ("best_metric_info.json", "trainer_state.json"):
        path = checkpoint / name
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            step = payload.get("step") if name == "best_metric_info.json" else payload.get("global_step")
            if type(step) is int and step >= 0:
                return step
    match = re.search(
        r"(?:checkpoint|step)[-_]?(\d+)", checkpoint.name, re.IGNORECASE
    )
    if match:
        return int(match.group(1))
    raise ValueError("Checkpoint step is not recorded; provide --step explicitly instead of logging a trained model at step zero.")


if __name__ == "__main__":
    main()
