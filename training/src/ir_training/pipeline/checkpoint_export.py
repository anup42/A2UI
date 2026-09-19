"""Export an existing dense SFT checkpoint; never train or run inference."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ir_training.common.bounded_command import run_bounded_command
from ir_training.common.progress import Progress, log
from ir_training.pipeline.deployment_export import (
    PROFILES,
    build_deployment_export_plan,
    validate_deployment_export_output,
)
from ir_training.train.resume_contract import resolve_export_training_lineage


@dataclass(frozen=True)
class CheckpointExportOptions:
    profile: str
    fit_dir: Path
    output_dir: Path
    exporter_python: Path
    training_python: Path = Path(sys.executable)
    checkpoint: Path | None = None
    cache_length: int = 8192
    allow_experimental_formats: bool = False
    stage_timeout_seconds: float = 172800
    progress_seconds: float = 10
    variants: tuple[str, ...] | None = None


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _required_file(path: Path) -> Path:
    if not path.is_file():
        raise ValueError(f"Required training/export input is missing: {path}")
    return path


def _record(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_checkpoint_export_plan(options: CheckpointExportOptions) -> dict[str, Any]:
    if options.profile not in PROFILES:
        raise ValueError("Export profile must be e2b or 270m")
    if not all(
        math.isfinite(value) and value > 0
        for value in (options.stage_timeout_seconds, options.progress_seconds)
    ):
        raise ValueError(
            "Stage timeout and progress interval must be positive and finite"
        )
    fit = options.fit_dir.expanduser().resolve()
    config_path = _required_file(fit / "training_config.yaml")
    checkpoint = (
        (options.checkpoint or fit / "training/best_golden_checkpoint")
        .expanduser()
        .resolve()
    )
    _required_file(checkpoint / "training_metadata.json")
    _required_file(fit / "preparation_report.json")
    lineage = resolve_export_training_lineage(config_path, checkpoint)
    config, metadata = lineage["config"], lineage["metadata"]
    model = config.get("model") or {}
    if (config.get("qat") or {}).get("enabled") or model.get(
        "mobile_training_seed_manifest"
    ):
        raise ValueError(
            "This script exports dense SFT checkpoints, not retained-scale/QAT checkpoints"
        )
    if (
        type(metadata.get("checkpoint_step")) is not int
        or metadata["checkpoint_step"] <= 0
    ):
        raise ValueError("Checkpoint has no positive optimizer-step provenance")
    if metadata.get("checkpoint_kind") not in {"lora_adapter", "full_model"}:
        raise ValueError("Checkpoint must be a saved dense LoRA adapter or full model")
    inputs = []
    for section, name in (
        (model, "model_source"),
        (config.get("run") or {}, "dataset_dir"),
    ):
        value = section.get(name)
        if (
            not isinstance(value, str)
            or not Path(value).is_absolute()
            or not Path(value).is_dir()
        ):
            raise ValueError(
                f"Original {name} must remain accessible at its saved absolute path: {value!r}"
            )
        inputs.append(Path(value).resolve())
    base_model, dataset = inputs
    model_type = _json(_required_file(base_model / "config.json")).get("model_type")
    if model_type not in PROFILES[options.profile]:
        raise ValueError(
            f"Profile {options.profile} cannot export model_type={model_type!r}"
        )
    for name in ("manifest.json", "train.jsonl"):
        _required_file(dataset / name)
    requested_output = options.output_dir.expanduser()
    output = requested_output.resolve()
    for protected in (fit, checkpoint, base_model, dataset):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError(
                f"Export output must not overlap training/model/data inputs: {protected}"
            )
    if requested_output.is_symlink() or output.exists():
        raise ValueError(
            f"Export output must be a NEW directory; existing results are never overwritten: {output}"
        )
    golden = config.get("golden_eval") or {}
    plan = build_deployment_export_plan(
        profile=options.profile,
        training_config_path=Path(lineage["training_config"]),
        preparation_config_path=config_path
        if lineage["resume_lineage"]["resumed"]
        else None,
        checkpoint_dir=checkpoint,
        output_dir=output,
        training_python=options.training_python,
        exporter_python=options.exporter_python,
        model_dir=base_model,
        cache_length=options.cache_length,
        max_input_tokens=golden.get("max_input_tokens", 4096),
        max_new_tokens=golden.get("max_new_tokens", 2048),
        selected_variants=options.variants,
    )
    plan.update(
        schema_version=1,
        workflow="checkpoint_export_only_v1",
        output_dir=str(output),
        training_config_sha256=lineage["training_config_sha256"],
        preparation_config=lineage["preparation_config"],
        preparation_config_sha256=lineage["preparation_config_sha256"],
        resume_lineage=lineage["resume_lineage"],
        checkpoint_step=metadata["checkpoint_step"],
        requires_merged_hf_and_real_litertlm_evaluation=False,
        runtime_evaluation_deferred=True,
        training_executed=False,
        quality_evaluation_performed=False,
        runtime_gpu_tested=False,
        allow_experimental_formats=options.allow_experimental_formats,
        stage_timeout_seconds=options.stage_timeout_seconds,
        progress_seconds=options.progress_seconds,
    )
    plan["stages"] = [
        {"name": "exporter_preflight", "command": plan["probe_command"]},
        {"name": "merge", "command": plan["prepare_command"]},
        *[
            {"name": f"export_{name}", "command": spec["command"]}
            for name, spec in plan["variants"].items()
        ],
    ]
    return plan


def _validate_stage(plan: dict[str, Any], name: str) -> dict[str, Any]:
    if name == "exporter_preflight":
        report = _json(Path(plan["output_dir"]) / "exporter_preflight.json")
        if report.get("status") != "passed" or report.get("profile") != plan["profile"]:
            raise ValueError(
                "Exporter preflight did not produce a matching successful report"
            )
        if "w248" in plan["variants"] and report.get("variants") != list(plan["variants"]):
            raise ValueError("W248 exporter preflight did not screen the requested variants")
    elif name == "merge":
        report = _json(Path(plan["merged_model_dir"]) / "deployment_source.json")
        expected = {
            key: plan[key]
            for key in (
                "profile",
                "source_checkpoint",
                "merged_model_dir",
                "training_config_sha256",
            )
        }
        if any(
            report.get(key) != value for key, value in expected.items()
        ) or not report.get("merged_files"):
            raise ValueError("Merged checkpoint lacks matching source provenance")
        if report.get("official_retained_scale_export") is not False:
            raise ValueError("Merged checkpoint is not a dense deployment source")
        if plan["resume_lineage"]["resumed"]:
            for key in (
                "preparation_config",
                "preparation_config_sha256",
                "resume_lineage",
            ):
                if report.get(key) != plan[key]:
                    raise ValueError(
                        f"Merged checkpoint resume provenance changed: {key}"
                    )
    else:
        return validate_deployment_export_output(plan, name.removeprefix("export_"))
    return {}


def print_export_summary(state: dict[str, Any]) -> None:
    print(
        f"\nCheckpoint export: {state['status']} (no training or inference)", flush=True
    )
    print("Variant | Weight format | MiB | Status | Artifact", flush=True)
    for name, spec in state["plan"]["variants"].items():
        artifact = state["artifacts"].get(name)
        status = (
            "exported; not evaluated"
            if artifact
            else (
                state["status"]
                if state["active_stage"] == f"export_{name}"
                else "not run"
            )
        )
        size = f"{artifact['size_bytes'] / (1024 * 1024):.1f}" if artifact else "-"
        print(
            f"{name.upper()} | {spec['kind']} | {size} | {status} | {spec['artifact']}",
            flush=True,
        )


def run_checkpoint_export(
    options: CheckpointExportOptions,
    *,
    execute: bool = False,
    command_runner: Callable = run_bounded_command,
) -> dict[str, Any]:
    plan = build_checkpoint_export_plan(options)
    if not execute:
        return {**plan, "status": "plan_only"}
    if not options.allow_experimental_formats and any(spec["experimental"] for spec in plan["variants"].values()):
        raise ValueError(
            "Selected exports include experimental W16/W4/W248; explicitly pass --allow-experimental-formats"
        )
    output = Path(plan["output_dir"])
    # Exclusive creation also prevents two launchers from sharing the same output.
    output.mkdir(parents=True, exist_ok=False)
    record = output / "checkpoint_export_manifest.json"
    state = {
        "plan": plan,
        "status": "running",
        "active_stage": None,
        "completed": {},
        "artifacts": {},
        "training_executed": False,
        "quality_evaluation_performed": False,
        "runtime_gpu_tested": False,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _record(record, state)
    environment = {
        **os.environ,
        **plan["environment"],
        "TOKENIZERS_PARALLELISM": "false",
    }
    log(
        "Export only: CPU merge/conversion; no training, Golden tests, Vulkan probe or MTP export"
    )
    try:
        for index, stage in enumerate(plan["stages"], 1):
            name = stage["name"]
            state["active_stage"] = name
            _record(record, state)
            logfile = output / "logs" / f"{name}.log"
            log(f"Export stage {index}/{len(plan['stages'])}: {name}")
            started = time.monotonic()
            command_runner(
                stage["command"],
                logfile,
                environment,
                timeout_seconds=options.stage_timeout_seconds,
                progress_seconds=options.progress_seconds,
            )
            with Progress(
                f"Verify {name} evidence",
                unit="stage",
                interval=options.progress_seconds,
            ):
                artifact = _validate_stage(plan, name)
            if artifact:
                artifact["size_bytes"] = Path(artifact["artifact"]).stat().st_size
                state["artifacts"][name.removeprefix("export_")] = artifact
            state["completed"][name] = {
                "elapsed_seconds": time.monotonic() - started,
                "log": str(logfile),
            }
            state["active_stage"] = None
            _record(record, state)
        state["status"] = "exported_not_evaluated"
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        _record(record, state)
    except BaseException as exc:
        state.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=str(exc),
        )
        _record(record, state)
        log(
            f"Export stopped at {state['active_stage']}; completed files and logs retained in {output}"
        )
        raise
    finally:
        print_export_summary(state)
    return state
