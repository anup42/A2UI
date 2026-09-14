from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "dataset" / "src"))

from ir_training.common.config import load_yaml
from ir_training.eval.compare_to_baseline import evaluate_predictions
from ir_training.eval.external_runner import (
    PROTOCOL_VERSION,
    run_external_generation,
    validate_external_runner_config,
)
from ir_training.eval.tensorboard_logging import log_evaluation_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate any LiteRT-LM package through the versioned external JSONL "
            "runner protocol, then score it and log it to TensorBoard."
        )
    )
    parser.add_argument("--model", required=True, help="LiteRT-LM package.")
    runner_choice = parser.add_mutually_exclusive_group(required=True)
    runner_choice.add_argument("--runner-config")
    runner_choice.add_argument("--builtin-gpu", action="store_true", help="Use the pinned, verified NVIDIA GPU runtime")
    parser.add_argument("--model-config", help="Resolved training model config; required by --builtin-gpu for exact prompt/tokenizer parity")
    parser.add_argument("--require-prepared-contract", action="store_true", help="Verify source/run/Golden/tokenizer bindings before inference")
    parser.add_argument("--runtime-python", help="Separate Python with requirements-litertlm-runtime.txt installed")
    parser.add_argument("--runtime-cache-dir")
    parser.add_argument("--runtime-timeout-seconds", type=float, default=7200)
    parser.add_argument("--case-timeout-seconds", type=float, default=600)
    parser.add_argument("--load-timeout-seconds", type=float, default=1800)
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
    parser.add_argument("--evaluation-name", required=True)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=32)
    parser.add_argument("--required-rows", type=int, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--mtp-enabled", action="store_true")
    parser.add_argument(
        "--weights-config",
        default=str(REPO_ROOT / "dataset" / "configs" / "run.yaml"),
    )
    parser.add_argument("--baseline-aggregate")
    parser.add_argument(
        "--metric-version", choices=("legacy", "v5_4", "dual"), default="dual"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    runner_config_path = Path(args.runner_config).expanduser().resolve() if args.runner_config else None
    if args.builtin_gpu:
        if not args.model_config:
            parser.error("--builtin-gpu requires --model-config")
        if args.max_rows != args.required_rows:
            parser.error("Built-in Golden evaluation requires --max-rows equal --required-rows (no partial subsets)")
        from ir_training.eval.litert_gpu import run_litert_gpu_generation

        config_path = Path(args.model_config).expanduser().resolve()
        model_config = load_yaml(config_path)
        if args.require_prepared_contract:
            from ir_training.eval.prepared_contract import (
                verify_evaluation_prepared_contract,
            )

            model_config["prepared_evaluation_contract"] = verify_evaluation_prepared_contract(
                config_path, Path(args.split).expanduser().resolve(),
                required_rows=args.required_rows, max_input_tokens=args.max_input_tokens,
            )
        manifest = run_litert_gpu_generation(
            model_path=args.model, split_path=args.split, output_dir=output_dir,
            model_config=model_config,
            max_input_tokens=args.max_input_tokens, max_new_tokens=args.max_new_tokens,
            required_rows=args.required_rows, mtp_enabled=args.mtp_enabled,
            runtime_python=args.runtime_python, cache_dir=args.runtime_cache_dir,
            timeout_seconds=args.runtime_timeout_seconds, case_timeout_seconds=args.case_timeout_seconds,
            load_timeout_seconds=args.load_timeout_seconds,
        )
    else:
        if args.require_prepared_contract:
            parser.error("--require-prepared-contract requires --builtin-gpu (external runner tokenizer parity is not verifiable)")
        runner = load_yaml(runner_config_path)
        validation = validate_external_runner_config(runner)
        if validation["placeholder"]:
            raise ValueError("Runner config still contains a placeholder executable path.")
        command = runner.get("command")
        assert isinstance(command, list)
        manifest = run_external_generation(
            command_template=command,
            model_path=args.model,
            split_path=args.split,
            output_dir=output_dir,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            max_rows=args.max_rows,
            required_rows=args.required_rows,
            mtp_enabled=args.mtp_enabled,
            timeout_seconds=(
                int(runner["timeout_seconds"])
                if runner.get("timeout_seconds") is not None
                else None
            ),
            working_dir=runner.get("working_dir"),
            environment=(
                runner.get("environment")
                if isinstance(runner.get("environment"), dict)
                else None
            ),
        )
    predictions_path = Path(manifest["predictions_path"])
    aggregate = evaluate_predictions(
        predictions_path,
        output_dir=output_dir,
        weights_config_path=args.weights_config,
        baseline_aggregate_path=args.baseline_aggregate,
        metric_version=args.metric_version,
    )
    aggregate.update(manifest.get("runtime_metrics") or {})
    aggregate_path = output_dir / "aggregate_metrics.json"
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    model = Path(args.model).expanduser().resolve()
    record = log_evaluation_result(
        args.tensorboard_root,
        run_id=args.run_id,
        evaluation_name=args.evaluation_name,
        metrics=aggregate,
        step=args.step,
        artifacts={
            "litertlm": model,
            **({"runner_config": runner_config_path} if runner_config_path else {"model_config": Path(args.model_config).resolve()}),
            "golden_split": Path(args.split).expanduser().resolve(),
            "predictions": predictions_path,
            "scored_predictions": output_dir / "scored_predictions.jsonl",
            "aggregate_metrics": aggregate_path,
            "external_runner_manifest": manifest["manifest_path"],
            "runner_log": manifest["runner_log_path"],
        },
        metadata={
            "protocol": PROTOCOL_VERSION,
            "runner_config": str(runner_config_path) if runner_config_path else "builtin_gpu",
            "gpu_execution_evidence": manifest.get("evidence_scope"),
            "split": str(Path(args.split).expanduser().resolve()),
            "rows": manifest["row_count"],
            "metric_version": args.metric_version,
            "mtp_enabled": args.mtp_enabled,
        },
        source_aggregate_path=aggregate_path,
    )
    result = {
        "model": str(model),
        "row_count": manifest["row_count"],
        "mtp_enabled": args.mtp_enabled,
        "aggregate": aggregate,
        "external_runner_manifest": manifest["manifest_path"],
        "tensorboard_record": record["record_path"],
    }
    (output_dir / "evaluation_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
