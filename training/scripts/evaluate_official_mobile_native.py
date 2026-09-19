#!/usr/bin/env python3
"""Post-export Android GPU quality evaluation for official-mobile artifacts.

The command is plan-only unless ``--execute`` is supplied. Execution requires
an explicit Android serial and preinstalled app/instrumentation APKs; it never
builds, installs, downloads, trains, selects a checkpoint, or probes desktop
Vulkan.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "dataset" / "src"))

from ir_training.common.jsonl import write_jsonl
from ir_training.eval.android_native_quality import (
    APP_PACKAGE,
    COHORTS,
    RUNNER,
    TEST_CLASS,
    AdbNativeBatchRunner,
    build_bound_requests,
    build_native_predictions,
    comparison_rows,
    device_requests,
    file_sha256,
    inspect_completed_run,
    render_results,
    require_nonoverlapping_paths,
    resolve_adb,
    validate_native_outputs,
)
from ir_training.eval.compare_to_baseline import evaluate_predictions
from ir_training.eval.tensorboard_logging import (
    log_evaluation_result,
    resolve_tensorboard_run_dir,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or execute a hash-bound, all-cohort Android LiteRT GPU quality "
            "evaluation of a completed official-mobile retained-scale export."
        )
    )
    parser.add_argument(
        "--run-dir", required=True, help="Completed official-mobile output directory."
    )
    parser.add_argument(
        "--output-dir", required=True, help="Fresh, disjoint report directory."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform ADB staging and native inference.",
    )
    parser.add_argument(
        "--adb", help="adb executable; resolved from PATH when omitted."
    )
    parser.add_argument(
        "--serial", help="Required explicit Android serial for --execute."
    )
    parser.add_argument("--app-package", default=APP_PACKAGE)
    parser.add_argument("--runner", default=RUNNER)
    parser.add_argument("--test-class", default=TEST_CLASS)
    parser.add_argument("--case-timeout-seconds", type=float, default=1800)
    parser.add_argument("--total-timeout-seconds", type=float, default=172800)
    parser.add_argument("--progress-seconds", type=float, default=30)
    parser.add_argument(
        "--weights-config",
        default=str(REPO_ROOT / "dataset" / "configs" / "run.yaml"),
        help="Existing v5.4 scorer weights configuration.",
    )
    parser.add_argument(
        "--tensorboard-root",
        help="Override the completed plan's TensorBoard root for this host.",
    )
    return parser


def _binding_summary(binding: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "workflow": binding["plan"]["workflow"],
        "run_dir": str(binding["run_dir"]),
        "plan_path": str(binding["plan_path"]),
        "plan_sha256": binding["plan_sha256"],
        "manifest_path": str(binding["manifest_path"]),
        "best_checkpoint": str(binding["paths"]["best_checkpoint"]),
        "exported_litertlm": str(binding["paths"]["litertlm"]),
        "exported_litertlm_sha256": binding["model_sha256"],
        "export_report": str(binding["paths"]["export_report"]),
        "mtp_enabled": False,
        "max_input_tokens": binding["max_input_tokens"],
        "max_new_tokens": binding["max_new_tokens"],
        "prepared_manifest_sha256": binding["prepared_manifest_sha256"],
        "cohorts": {
            name: {
                "required_rows": details["count"],
                "split": str(details["split"]),
                "split_sha256": file_sha256(details["split"]),
                "benchmark_kind": details["prepared_contract"]["benchmark_kind"],
                "reference_available": details["prepared_contract"].get(
                    "reference_available", True
                ),
                "hf_evaluation_result": str(
                    details["hf_dir"] / "evaluation_result.json"
                ),
                "hf_evaluation_result_sha256": file_sha256(
                    details["hf_dir"] / "evaluation_result.json"
                ),
                "hf_predictions_sha256": file_sha256(
                    details["hf_dir"] / "predictions.jsonl"
                ),
            }
            for name, details in binding["cohorts"].items()
        },
        "receipt_bindings": binding["receipt_bindings"],
        "native_contract": {
            "backend": "GPU",
            "cpu_fallback_allowed": False,
            "one_engine_for_all_cases": True,
            "fresh_conversation_per_case": True,
            "conversation_template_applied": True,
            "exact_runtime_rendered_prompt_required": True,
            "full_gpu_delegation_log_gate_required": True,
            "native_token_ids_available": False,
            "native_token_parity_claimed": False,
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _artifact_bindings(output: Path) -> dict[str, str]:
    names = [
        "binding_report.json",
        "scorer_binding.json",
        "requests.jsonl",
        "device_requests.jsonl",
        "device_manifest.json",
        "device_outputs.jsonl",
        "device_cleanup.json",
        "gpu_delegation_evidence.json",
        "native_runtime_rows.jsonl",
        "comparison.json",
        "results.md",
        "instrumentation.log",
        "device_logcat.txt",
    ]
    paths = [output / name for name in names]
    paths.extend(
        output / "cohorts" / cohort / name
        for cohort in COHORTS
        for name in (
            "predictions.jsonl",
            "scored_predictions.jsonl",
            "aggregate_metrics.json",
            "checkpoint_rescored/predictions.jsonl",
            "checkpoint_rescored/scored_predictions.jsonl",
            "checkpoint_rescored/aggregate_metrics.json",
        )
    )
    return {str(path): file_sha256(path) for path in paths if path.is_file()}


def _plan(
    binding: dict[str, Any], *, output: Path, args: argparse.Namespace
) -> dict[str, Any]:
    weights = Path(args.weights_config).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Scorer weights config is missing: {weights}")
    return {
        "schema_version": 1,
        "mode": "execute" if args.execute else "plan_only",
        "run_dir": str(binding["run_dir"]),
        "output_dir": str(output),
        "artifact": str(binding["paths"]["litertlm"]),
        "artifact_sha256": binding["model_sha256"],
        "best_checkpoint": str(binding["paths"]["best_checkpoint"]),
        "cohorts": COHORTS,
        "total_rows": sum(COHORTS.values()),
        "max_input_tokens": binding["max_input_tokens"],
        "max_new_tokens": binding["max_new_tokens"],
        "backend": "GPU",
        "mtp_enabled": False,
        "requires_explicit_serial": True,
        "serial": args.serial
        if args.execute
        else (args.serial or "<required only with --execute>"),
        "build_install_download_training": False,
        "desktop_vulkan_required": False,
        "device_staging": "/data/local/tmp/official_mobile_native_quality/<uuid>",
        "cleanup_verification_required": True,
        "native_token_ids_available": False,
        "scorer": {
            "metric_version": "v5_4",
            "weights_config": str(weights),
            "weights_config_sha256": file_sha256(weights),
            "checkpoint_scores_recomputed": True,
        },
        "tensorboard_root": args.tensorboard_root
        or binding["plan"]["options"].get("tensorboard_root", "/tensorboard"),
        "failure_policy": "no CPU fallback; no partial scores; missing GPU delegation fails closed",
    }


def execute(
    args: argparse.Namespace, binding: dict[str, Any], output: Path
) -> dict[str, Any]:
    if not args.serial or args.serial.strip().startswith("<"):
        raise ValueError("--execute requires an explicit non-placeholder --serial")
    tensorboard_root = args.tensorboard_root or binding["plan"]["options"].get(
        "tensorboard_root", "/tensorboard"
    )
    tensorboard_run_dir = resolve_tensorboard_run_dir(
        tensorboard_root, run_id=output.name
    )
    try:
        tensorboard_run_dir.relative_to(binding["run_dir"])
    except ValueError:
        pass
    else:
        raise ValueError(
            "Native TensorBoard output must not modify the completed official-mobile run"
        )
    if tensorboard_run_dir.exists():
        raise FileExistsError(
            f"Choose a fresh native TensorBoard run ID/output name: {tensorboard_run_dir}"
        )
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "manifest.json"
    started = datetime.now(timezone.utc).isoformat()
    state: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": started,
        "plan": _plan(binding, output=output, args=args),
        "active_stage": "bind_inputs",
        "completed_stages": [],
        "error": None,
    }
    _write_json(manifest_path, state)
    try:
        _write_json(output / "binding_report.json", _binding_summary(binding))
        state["completed_stages"].append("bind_inputs")
        state["active_stage"] = "format_prompts"
        _write_json(manifest_path, state)

        requests = build_bound_requests(binding)
        write_jsonl(output / "requests.jsonl", requests)
        write_jsonl(output / "device_requests.jsonl", device_requests(requests))
        state["completed_stages"].append("format_prompts")
        state["active_stage"] = "android_gpu_inference"
        _write_json(manifest_path, state)

        runner = AdbNativeBatchRunner(
            adb=resolve_adb(args.adb),
            serial=args.serial,
            app_package=args.app_package,
            runner=args.runner,
            test_class=args.test_class,
        )
        device = runner.run(
            model=binding["paths"]["litertlm"],
            model_sha256=binding["model_sha256"],
            device_requests_path=output / "device_requests.jsonl",
            output_dir=output,
            total_timeout=args.total_timeout_seconds,
            case_timeout=args.case_timeout_seconds,
            progress_seconds=args.progress_seconds,
        )
        _write_json(
            output / "gpu_delegation_evidence.json",
            {
                "schema_version": 1,
                "scope": "one GPU engine initialization reused by all fresh per-case conversations",
                "artifact_sha256": binding["model_sha256"],
                "device_model_sha256_before": device["device_model_sha256_before"],
                "device_model_sha256_after": device["device_model_sha256_after"],
                "evidence": device["delegation"],
            },
        )
        runtime_rows = validate_native_outputs(
            requests,
            device["device_outputs"],
            device["device_manifest"],
            device["delegation"],
        )
        write_jsonl(output / "native_runtime_rows.jsonl", runtime_rows)
        state["completed_stages"].append("android_gpu_inference")
        state["active_stage"] = "strict_v5_4_scoring"
        _write_json(manifest_path, state)

        weights_path = Path(args.weights_config).expanduser().resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(f"Scorer weights config is missing: {weights_path}")
        _write_json(
            output / "scorer_binding.json",
            {
                "metric_version": "v5_4",
                "weights_config": str(weights_path),
                "weights_config_sha256": file_sha256(weights_path),
                "checkpoint_scores_recomputed": True,
                "reason": "checkpoint and native aggregates must use the identical scorer binding",
            },
        )
        predictions = build_native_predictions(binding, runtime_rows)
        native_aggregates: dict[str, dict[str, Any]] = {}
        checkpoint_aggregates: dict[str, dict[str, Any]] = {}
        tensorboard_records: dict[str, Any] = {}
        for cohort, cohort_predictions in predictions.items():
            cohort_dir = output / "cohorts" / cohort
            cohort_dir.mkdir(parents=True, exist_ok=False)
            prediction_path = cohort_dir / "predictions.jsonl"
            write_jsonl(prediction_path, cohort_predictions)
            native_aggregates[cohort] = evaluate_predictions(
                prediction_path,
                output_dir=cohort_dir,
                weights_config_path=weights_path,
                metric_version="v5_4",
            )
            checkpoint_dir = cohort_dir / "checkpoint_rescored"
            checkpoint_aggregates[cohort] = evaluate_predictions(
                binding["cohorts"][cohort]["hf_dir"] / "predictions.jsonl",
                output_dir=checkpoint_dir,
                weights_config_path=weights_path,
                metric_version="v5_4",
            )
            if (
                len(cohort_predictions) != COHORTS[cohort]
                or native_aggregates[cohort].get("count") != COHORTS[cohort]
                or checkpoint_aggregates[cohort].get("count") != COHORTS[cohort]
            ):
                raise RuntimeError(f"Scoring coverage changed for {cohort}")
            tensorboard_records[cohort] = log_evaluation_result(
                tensorboard_root,
                run_id=output.name,
                evaluation_name=f"native_{cohort}",
                metrics=native_aggregates[cohort],
                artifacts={
                    "exported_litertlm": binding["paths"]["litertlm"],
                    "native_predictions": prediction_path,
                    "native_scored_predictions": cohort_dir
                    / "scored_predictions.jsonl",
                    "native_aggregate": cohort_dir / "aggregate_metrics.json",
                    "scorer_weights": weights_path,
                },
                metadata={
                    "backend": "GPU",
                    "mtp_enabled": False,
                    "model_sha256": binding["model_sha256"],
                    "native_token_ids_available": False,
                    "full_gpu_delegation_verified": True,
                    "rows": COHORTS[cohort],
                    "metric_version": "v5_4",
                },
                source_aggregate_path=cohort_dir / "aggregate_metrics.json",
                detail="minimal",
            )
        comparisons = comparison_rows(binding, native_aggregates, checkpoint_aggregates)
        _write_json(
            output / "comparison.json",
            {
                "schema_version": 1,
                "checkpoint_role": "selected_before_export",
                "native_role": "post_export_quality_only_never_selection",
                "rows": comparisons,
            },
        )
        (output / "results.md").write_text(
            render_results(
                comparisons,
                model_sha256=binding["model_sha256"],
                report_status="passed",
            ),
            encoding="utf-8",
            newline="\n",
        )
        cleanup = json.loads(
            (output / "device_cleanup.json").read_text(encoding="utf-8")
        )
        if cleanup.get("verified") is not True:
            raise RuntimeError("Device cleanup is not verified")
        state["completed_stages"].append("strict_v5_4_scoring")
        state.update(
            status="complete",
            active_stage=None,
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            row_count=sum(COHORTS.values()),
            exact_cohort_coverage={name: count for name, count in COHORTS.items()},
            full_gpu_delegation_verified=True,
            native_token_parity_claimed=False,
            scorer_binding={
                "metric_version": "v5_4",
                "weights_config": str(weights_path),
                "weights_config_sha256": file_sha256(weights_path),
                "checkpoint_scores_recomputed": True,
            },
            tensorboard_root=str(tensorboard_root),
            tensorboard_run_dir=str(tensorboard_run_dir),
            tensorboard_records={
                name: record["record_path"]
                for name, record in tensorboard_records.items()
            },
            artifacts=_artifact_bindings(output),
        )
        _write_json(manifest_path, state)
        return state
    except BaseException as exc:
        state.update(
            status="failed",
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            error=f"{type(exc).__name__}: {exc}",
            artifacts=_artifact_bindings(output),
        )
        _write_json(manifest_path, state)
        if not (output / "results.md").exists():
            (output / "results.md").write_text(
                "# Official-mobile native Android quality\n\n"
                f"Status: failed\n\nFailure: {state['error']}\n\n"
                "No partial native score is published as a pass.\n",
                encoding="utf-8",
                newline="\n",
            )
        raise


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run, output = require_nonoverlapping_paths(args.run_dir, args.output_dir)
    binding = inspect_completed_run(run)
    plan = _plan(binding, output=output, args=args)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        print(
            "Plan only. Nothing was written, staged, built, installed, downloaded, trained, or executed."
        )
        return 0
    state = execute(args, binding, output)
    print((output / "results.md").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": state["status"],
                "report": str(output / "manifest.json"),
                "summary": str(output / "results.md"),
                "rows": state["row_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
