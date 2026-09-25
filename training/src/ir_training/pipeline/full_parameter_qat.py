"""Experimental Gemma 4 E2B all-parameter QAT training and W248 export.

This lane intentionally reuses the checked data preparation, GPU launcher,
Golden evaluation, and dense deployment exporter without sharing the official
retained-scale LoRA recipe.  Planning is offline.  Every executable stage is a
bounded subprocess and a fresh output directory is mandatory.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ir_training.common.bounded_command import run_bounded_command
from ir_training.common.config import load_yaml, training_root
from ir_training.common.progress import Progress, log
from ir_training.pipeline.golden_training import (
    GOLDENS,
    GoldenTrainingOptions,
    _write,
    prepare_data,
    sha256,
)
from ir_training.pipeline.golden_training import build_plan as build_preparation_plan

SELECTOR = "unique_source_generation_reward_v5_4_avg"
WORKFLOW = "e2b_all_parameter_qat_v1"
ALLOWED_H100_COUNTS = frozenset({2, 4, 8})


@dataclass(frozen=True)
class FullParameterQATOptions:
    model_dir: Path
    input_dir: Path
    output_dir: Path
    exporter_python: Path
    devices: str = "auto"
    epochs: float = 2.0
    steps: int | None = None
    resume_from_checkpoint: Path | None = None
    learning_rate: float = 1e-5
    eval_steps: int = 500
    golden_every_steps: int = 1000
    max_seq_length: int = 4096
    max_new_tokens: int = 2048
    microbatch: int | None = None
    effective_batch: int | None = None
    dataloader_workers: int | None = None
    prepare_workers: int = 0
    preparation_cache: bool = True
    preparation_cache_dir: Path | None = None
    tensorboard_root: str = "/tensorboard"
    progress_seconds: float = 10.0
    stage_timeout_seconds: float = 172800.0
    generation_timeout_seconds: float = 7200.0
    allow_experimental_export: bool = False
    distributed_backend: str = "ddp"
    max_input_tokens: int = 5120
    zero_stage: int = 2


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _yaml(path: Path, value: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(value, stream, sort_keys=False)


def _scripts() -> None:
    value = str(training_root() / "scripts")
    if value not in sys.path:
        sys.path.insert(0, value)


def _bindings(paths: list[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        resolved = Path(path).resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"Required stage artifact is not a file: {resolved}")
        result[str(resolved)] = sha256(resolved)
    if not result:
        raise ValueError("Stage produced no bound evidence")
    return result


def _verify_bindings(records: dict[str, str]) -> None:
    if not records:
        raise ValueError("Stage has no bound evidence")
    for filename, digest in records.items():
        path = Path(filename)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Bound stage artifact changed or disappeared: {path}")


def build_plan(options: FullParameterQATOptions) -> dict[str, Any]:
    """Create a read-only plan without importing torch or probing CUDA."""
    values = asdict(options)
    if values["distributed_backend"] not in {"ddp", "sharded"}:
        raise ValueError("distributed_backend must be one of: ddp, sharded")
    from ir_training.train.sharded_contract import resolve_zero_stage

    zero_stage = resolve_zero_stage({
        "distributed_backend": values["distributed_backend"],
        "zero_stage": values["zero_stage"],
    })
    if zero_stage == 2:
        # Keep legacy/default plan serialization and hashes unchanged.
        values.pop("zero_stage")
    for key in ("model_dir", "input_dir", "output_dir", "preparation_cache_dir", "resume_from_checkpoint"):
        if values[key] is not None:
            values[key] = str(Path(values[key]).expanduser().resolve())
    if values["resume_from_checkpoint"] is None:
        # Preserve the historical fresh-run plan when resume is not requested.
        values.pop("resume_from_checkpoint")
    # Do not resolve a venv Python symlink into the system interpreter.
    values["exporter_python"] = os.path.abspath(
        os.path.expanduser(str(values["exporter_python"]))
    )
    for name in (
        "learning_rate",
        "progress_seconds",
        "stage_timeout_seconds",
        "generation_timeout_seconds",
    ):
        if not math.isfinite(values[name]) or values[name] <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if options.microbatch not in (None, 1):
        raise ValueError("All-parameter E2B QAT permits only --microbatch 1")
    for name in ("max_seq_length", "max_input_tokens", "max_new_tokens"):
        if type(values[name]) is not int or values[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    output = Path(values["output_dir"])
    seed = Path(values["model_dir"])
    inputs = Path(values["input_dir"])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", output.name):
        raise ValueError("Output directory name must be a safe run ID (1-96 characters)")
    for protected in (seed, inputs):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("Run output must be separate from model and input directories")
    required_seed = (
        seed / "config.json",
        seed / "tokenizer_config.json",
        seed / "mobile_training_seed_manifest.json",
        seed / "mobile_qparams.json",
        seed / "mobile_qparams.safetensors",
        Path(values["exporter_python"]),
    )
    for path in required_seed:
        if not path.is_file():
            raise FileNotFoundError(f"Required reconstructed mobile-seed artifact missing: {path}")
    if not any(
        path.name != "mobile_qparams.safetensors"
        for path in seed.glob("*.safetensors")
    ):
        raise FileNotFoundError(f"Reconstructed mobile seed has no safetensor weights: {seed}")

    resume = None
    if values.get("resume_from_checkpoint"):
        import copy

        from ir_training.train.full_qat_resume import horizon_record, source_config

        checkpoint = Path(values["resume_from_checkpoint"])
        source_path, source, _ = source_config(checkpoint)
        source_run = Path(source["run"]["output_dir"]).resolve().parents[1]
        if (output.is_relative_to(source_run) or source_run.is_relative_to(output)
                or checkpoint.is_relative_to(output)):
            raise ValueError("Full-QAT resume output must be fresh and separate from the source run")
        if Path(source["model"]["model_source"]).resolve() != seed:
            raise ValueError("Full-QAT resume requires the original reconstructed model seed")
        requested = copy.deepcopy(source)
        requested["training"]["epochs"] = values["epochs"]
        if values["steps"] is None:
            requested["training"].pop("max_steps", None)
        else:
            requested["training"]["max_steps"] = values["steps"]
        resume = {
            "checkpoint": str(checkpoint), "source_config": str(source_path),
            "source_config_sha256": sha256(source_path),
            "metadata_sha256": sha256(checkpoint / "training_metadata.json"),
            "horizon": horizon_record(checkpoint, requested),
        }

    preparation = build_preparation_plan(
        GoldenTrainingOptions(
            model_dir=seed,
            input_dir=inputs,
            output_dir=output,
            devices=options.devices,
            epochs=options.epochs,
            steps=options.steps,
            eval_steps=options.eval_steps,
            golden_every_steps=options.golden_every_steps,
            max_seq_length=options.max_seq_length,
            max_input_tokens=options.max_input_tokens,
            max_new_tokens=options.max_new_tokens,
            tensorboard_root=options.tensorboard_root,
            microbatch=options.microbatch,
            effective_batch=options.effective_batch,
            dataloader_workers=options.dataloader_workers,
            prepare_workers=options.prepare_workers,
            progress_seconds=options.progress_seconds,
            preparation_cache=options.preparation_cache,
            preparation_cache_dir=options.preparation_cache_dir,
            learning_rate=options.learning_rate,
        ),
        preparation_only=True,
    )
    fit = output / "fit"
    paths = {
        "config": fit / "training_config.yaml",
        "preparation_report": fit / "preparation_report.json",
        "training": fit / "training",
        "best_checkpoint": fit / "training/best_golden_checkpoint",
        "final_checkpoint": fit / "training/final_model",
        "export": output / "experimental_w248_export",
        "scorecard": output / "evaluation_scorecard.json",
        "results": output / "results.md",
    }
    training_contract = {
        "parameter_scope": "all_unique_text_model_parameters",
        "master_parameter_dtype": "float32",
        "autocast": "bfloat16",
        "optimizer": "adafactor",
        "quantizer": "ste_ai_edge",
        "quantized_modules": "all fully-connected weights and both token embeddings",
        "activation_quantization": False,
        "retained_mobile_scales": False,
        "lora": False,
    }
    if values["distributed_backend"] == "sharded":
        training_contract["optimizer"] = "adamw_torch"
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "options": values,
        "preparation": preparation,
        "paths": {key: str(value) for key, value in paths.items()},
        **({"resume": resume} if resume is not None else {}),
        "stages": (["environment"] if values["distributed_backend"] == "sharded" else []) + [
            "assets",
            "prepare",
            "configure",
            "preflight",
            "training",
            "best_golden32",
            "final_golden35",
            "final_bixby50",
            "export",
            "scorecard",
        ],
        "distributed_training": {
            "backend": values["distributed_backend"],
            "optimizer": training_contract["optimizer"],
            **({"zero_stage": zero_stage} if zero_stage == 3 else {}),
        },
        "training_contract": training_contract,
        "gpu_contract": {
            "allowed_world_sizes": sorted(ALLOWED_H100_COUNTS),
            "accelerator": "NVIDIA H100 with at least 79 GiB per selected rank",
            "microbatch": 1,
            "effective_batch_default": 32,
            "memory_certification": "live maximum-shape backward preflight; planning alone does not prove fit",
        },
        "selection": {
            "cohort": "golden32",
            "metric": SELECTOR,
            "golden35_used": False,
            "bixby50_used": False,
        },
        "export": {
            "profile": "e2b",
            "variant": "w248",
            "kind": "experimental dynamic PTQ from a fresh dense graph",
            "official_static_mobile_layout": False,
            "allow_experimental_export": options.allow_experimental_export,
        },
        "mtp": {"training": False, "inference": False, "exported": False},
        "native_runtime_evaluated": False,
        "training_executed": False,
    }


def validate_h100_profile(profile: dict[str, Any]) -> None:
    """Fail closed before allocating a replicated FP32 full model."""
    world_size = profile.get("world_size")
    if world_size not in ALLOWED_H100_COUNTS:
        raise ValueError("All-parameter E2B QAT requires exactly 2, 4, or 8 selected H100 GPUs")
    selected = profile.get("selected_devices") or []
    if len(selected) != world_size:
        raise ValueError("GPU profile selected-device count disagrees with world size")
    if any(
        "H100" not in str(device.get("name", "")).upper()
        or int(device.get("total_memory_bytes", 0)) < 79 * 1024**3
        for device in selected
    ):
        raise ValueError("Every selected rank must be an H100 with at least 79 GiB")
    if profile.get("dtype") != "bfloat16":
        raise ValueError("All-parameter E2B QAT requires native BF16 H100 execution")
    if profile.get("microbatch") != 1:
        raise ValueError("All-parameter E2B QAT fixes per-rank microbatch to 1 for memory safety")
    memory = profile.get("memory_safety") or {}
    if profile.get("requires_backward_preflight") is not True or memory.get("benchmark_verified") is not False:
        raise ValueError("GPU profile must preserve the live backward-preflight memory gate")


def training_config(plan: dict[str, Any], profile: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """Derive the independent full-QAT recipe from the validated mobile base."""
    validate_h100_profile(profile)
    from ir_training.pipeline.official_mobile import (
        training_config as mobile_base_config,
    )
    from ir_training.qat.full_model_contract import (
        configure_full_qat,
        validate_full_qat_config,
    )

    values, paths = plan["options"], plan["paths"]
    # The mobile base helper has its own LoRA-specific resume branch. This
    # lane always derives its independent dense recipe before applying its
    # own full-QAT continuation contract below.
    base_plan = {key: value for key, value in plan.items() if key != "resume"}
    if plan.get("resume"):
        # Preserve the original bounded-run save/eval cadence when only its
        # step horizon grows (for example, 20 -> 30 with default eval_steps).
        from ir_training.train.full_qat_resume import source_config

        _, source_config_value, _ = source_config(Path(plan["resume"]["checkpoint"]))
        base_plan = {**base_plan, "options": {
            **values, "steps": source_config_value["training"].get("max_steps"),
        }}
    base = mobile_base_config(base_plan, profile, report)
    base["run"].update(
        id=Path(values["output_dir"]).name,
        output_dir=paths["training"],
    )
    # The official-mobile launcher normally rebinds these inherited defaults
    # in its own reservation step.  This independent lane does not call that
    # launcher, so bind every writable training path before applying/validating
    # the full-QAT recipe.
    base["golden_eval"]["output_dir"] = str(
        Path(paths["training"]) / "periodic_golden32"
    )
    # Older saved plans used one shared limit; keep those runs consistent.
    base["golden_eval"]["max_input_tokens"] = values.get("max_input_tokens", values["max_seq_length"])
    base["training"]["logging_dir"] = str(
        Path(paths["training"]) / "tensorboard"
    )
    config = configure_full_qat(
        base,
        distributed_backend=values.get("distributed_backend", "ddp"),
        zero_stage=values.get("zero_stage", 2),
    )
    if plan.get("resume"):
        from ir_training.train.full_qat_resume import POLICY, horizon_record

        checkpoint = Path(values["resume_from_checkpoint"])
        if values["steps"] is None:
            config["training"].pop("max_steps", None)
        else:
            config["training"]["max_steps"] = values["steps"]
        horizon = horizon_record(checkpoint, config)
        if horizon != plan["resume"]["horizon"]:
            raise ValueError("Full-QAT resume horizon changed after planning")
        config["training"].update(
            refuse_resume=False, resume_from_checkpoint=str(checkpoint),
            resume_policy=POLICY, resume_horizon=horizon,
            warmup_steps=horizon["scheduler_warmup_steps"],
        )
    validate_full_qat_config(config)
    return config


def _assets(plan: dict[str, Any]) -> list[Path]:
    from ir_training.qat.mobile_qparams import verify_mobile_qparams_contract
    from ir_training.qat.mobile_training_seed import (
        OFFICIAL_MOBILE_MODEL_ID,
        verify_configured_mobile_training_seed,
    )

    seed = Path(plan["options"]["model_dir"])
    model = {
        "model_id": OFFICIAL_MOBILE_MODEL_ID,
        "model_source": str(seed),
        "mobile_training_seed_manifest": str(seed / "mobile_training_seed_manifest.json"),
        "mobile_qparams_contract": str(seed / "mobile_qparams.json"),
    }
    seed_report = verify_configured_mobile_training_seed(
        model, base=training_root(), require_materialized=True
    )
    qparams_report = verify_mobile_qparams_contract(
        seed / "mobile_qparams.json", base=training_root()
    )
    if seed_report.get("verified") is not True or qparams_report.get("verified") is not True:
        raise ValueError("Reconstructed official-mobile seed provenance did not verify")
    destination = Path(plan["options"]["output_dir"]) / "mobile_seed_verified.json"
    _write(
        destination,
        {
            "seed": seed_report,
            "qparams": qparams_report,
            "qparams_used_as_static_training_scales": False,
            "dynamic_all_parameter_qat": True,
        },
    )
    # Screen the exact isolated exporter and complete Gemma 4 routing before
    # any GPU training.  This is API/recipe screening, not a conversion claim.
    from ir_training.pipeline.deployment_export import build_deployment_export_plan

    values = plan["options"]
    probe_root = Path(values["output_dir"]) / "exporter_preflight"
    export_plan = build_deployment_export_plan(
        profile="e2b",
        training_config_path=Path(plan["paths"]["config"]),
        checkpoint_dir=Path(plan["paths"]["best_checkpoint"]),
        output_dir=probe_root,
        training_python=sys.executable,
        exporter_python=values["exporter_python"],
        model_dir=seed,
        cache_length=8192,
        max_input_tokens=values.get("max_input_tokens", values["max_seq_length"]),
        max_new_tokens=values["max_new_tokens"],
        selected_variants=("w248",),
        full_parameter_export=True,
    )
    run_bounded_command(
        export_plan["probe_command"],
        Path(values["output_dir"]) / "logs/exporter_preflight.log",
        {**os.environ, **export_plan["environment"]},
        timeout_seconds=values["stage_timeout_seconds"],
        progress_seconds=values["progress_seconds"],
    )
    exporter_report = probe_root / "exporter_preflight.json"
    probe = _json(exporter_report)
    if (
        probe.get("status") != "passed"
        or probe.get("profile") != "e2b"
        or probe.get("variants") != ["w248"]
        or probe.get("model_loaded") is not False
        or probe.get("conversion_tested") is not False
        or probe.get("full_parameter_export") is not True
        or probe.get("all_parameter_serialization_required") is not True
    ):
        raise ValueError("Experimental W248 exporter preflight did not pass exactly")
    return [
        destination,
        exporter_report,
        seed / "mobile_training_seed_manifest.json",
        seed / "mobile_qparams.json",
    ]


def _configure(plan: dict[str, Any]) -> list[Path]:
    _scripts()
    from prepare_review_training import verify_prepared

    from ir_training.train.gpu_profile import build_gpu_profile, detect_cuda_devices

    values = plan["options"]
    output = Path(values["output_dir"])
    if plan.get("resume"):
        source = plan["resume"]
        if (sha256(Path(source["source_config"])) != source["source_config_sha256"]
                or sha256(Path(source["checkpoint"]) / "training_metadata.json") != source["metadata_sha256"]):
            raise ValueError("Full-QAT resume source changed after planning")
    profile = build_gpu_profile(
        detect_cuda_devices(),
        model="e2b",
        devices=values["devices"],
        microbatch=values["microbatch"],
        effective_batch=values["effective_batch"],
        dataloader_workers=values["dataloader_workers"],
    )
    validate_h100_profile(profile)
    report = verify_prepared(
        output / "prepared",
        output / "prepared/golden32.jsonl",
        golden35=output / "prepared/golden35.jsonl",
        bixby50=output / "prepared/bixby50.jsonl",
        max_sequence=values["max_seq_length"],
        max_prompt=values.get("max_input_tokens", values["max_seq_length"]),
    )
    config_path = Path(plan["paths"]["config"])
    config = training_config(plan, profile, report)
    if plan.get("resume"):
        from ir_training.train.full_qat_resume import verify_continuation

        verify_continuation(Path(plan["resume"]["checkpoint"]), config)
    _yaml(config_path, config)
    seed = Path(values["model_dir"])
    report.update(
        training_executed=False,
        model_loaded=False,
        profile="all_parameter_qat",
        workflow=WORKFLOW,
        training_config_sha256=sha256(config_path),
        gpu_profile=profile,
        model_files={
            path.name: sha256(path)
            for path in sorted(seed.iterdir())
            if path.is_file()
            and path.suffix in {".json", ".safetensors", ".model", ".jinja"}
        },
    )
    report_path = Path(plan["paths"]["preparation_report"])
    _write(report_path, report)
    _scripts()
    from launch_review_training import verify_launch_binding

    verify_launch_binding(config_path)
    log(
        f"All-parameter GPU profile: {profile['world_size']} H100 ranks; "
        f"microbatch 1; accumulation {profile['gradient_accumulation_steps']}; "
        f"effective batch {profile['effective_batch_size']}; live backward preflight remains mandatory"
    )
    return [config_path, report_path]


def launch_command(plan: dict[str, Any], *, preflight: bool = False) -> list[str]:
    command = [
        sys.executable,
        str(training_root() / "scripts/launch_review_training.py"),
        "--config",
        plan["paths"]["config"],
    ]
    if preflight:
        command.append("--preflight-only")
    return [*command, "--execute"]


def _gpu_environment(plan: dict[str, Any]) -> dict[str, str]:
    """Use only the device allocation frozen into the reviewed fit config."""
    from ir_training.train.gpu_profile import training_environment

    config = load_yaml(Path(plan["paths"]["config"]))
    profile = (config.get("runtime") or {}).get("gpu_profile")
    validate_h100_profile(profile)
    environment = training_environment(
        profile, tensorboard_root=plan["options"]["tensorboard_root"]
    )
    environment["A2UI_TENSORBOARD_DETAIL"] = "minimal"
    return environment


def evaluation_command(plan: dict[str, Any], cohort: str, stage: str) -> list[str]:
    values = plan["options"]
    output = Path(values["output_dir"])
    rows = plan["preparation"]["goldens"][cohort]["rows"]
    return [
        sys.executable,
        str(training_root() / "scripts/evaluate_checkpoint_on_golden.py"),
        "--config",
        plan["paths"]["config"],
        "--checkpoint",
        plan["paths"]["best_checkpoint"],
        "--checkpoint-kind",
        "merged",
        "--qat-mode",
        "on",
        "--require-prepared-contract",
        "--require-gpu",
        "--devices",
        "auto",
        "--split",
        str(output / f"prepared/{cohort}.jsonl"),
        "--max-rows",
        str(rows),
        "--required-rows",
        str(rows),
        "--max-input-tokens",
        str(values.get("max_input_tokens", values["max_seq_length"])),
        "--max-new-tokens",
        str(values["max_new_tokens"]),
        "--output-dir",
        str(output / f"evaluations/{stage}"),
        "--run-id",
        output.name,
        "--evaluation-name",
        stage,
        "--tensorboard-root",
        values["tensorboard_root"],
        "--generation-timeout-seconds",
        str(values["generation_timeout_seconds"]),
        "--metric-version",
        "v5_4",
    ]


def checkpoint_export_options(plan: dict[str, Any]):
    from ir_training.pipeline.checkpoint_export import CheckpointExportOptions

    values = plan["options"]
    return CheckpointExportOptions(
        profile="e2b",
        fit_dir=Path(plan["paths"]["config"]).parent,
        checkpoint=Path(plan["paths"]["best_checkpoint"]),
        output_dir=Path(plan["paths"]["export"]),
        exporter_python=Path(values["exporter_python"]),
        training_python=Path(sys.executable),
        variants=("w248",),
        allow_experimental_formats=True,
        stage_timeout_seconds=values["stage_timeout_seconds"],
        progress_seconds=values["progress_seconds"],
    )


def _validate_best_checkpoint(plan: dict[str, Any]) -> dict[str, Any]:
    from ir_training.pipeline.deployment_export import verify_checkpoint_source

    checkpoint = Path(plan["paths"]["best_checkpoint"])
    metadata = _json(checkpoint / "training_metadata.json")
    binding = verify_checkpoint_source(
        Path(plan["paths"]["config"]), checkpoint, "e2b"
    )
    result = binding.get("full_qat_contract") or {}
    if result.get("verified") is not True:
        raise ValueError("Selected checkpoint did not satisfy the all-parameter QAT contract")
    if metadata.get("checkpoint_role") != "best_golden":
        raise ValueError("Training did not publish the Golden32-selected full-model checkpoint")
    selected = metadata.get("best_golden_eval") or {}
    if selected.get("metric") != SELECTOR:
        raise ValueError("Best checkpoint was not selected by the required Golden32 metric")
    return result


def _evaluation_files(plan: dict[str, Any], stage: str) -> list[Path]:
    from ir_training.pipeline.golden_deployment import _evaluation

    directory = Path(plan["options"]["output_dir"]) / "evaluations" / stage
    cohort = "golden32" if stage == "best_golden32" else stage.removeprefix("final_")
    result, evidence = _evaluation(
        directory,
        GOLDENS[cohort][1],
        artifact=Path(plan["paths"]["best_checkpoint"]),
    )
    if result.get("qat_applied") is not True:
        raise ValueError(f"Incomplete or non-QAT evaluation: {stage}")
    return evidence


def _optimizer_preflight_files(plan: dict[str, Any]) -> list[Path]:
    from ir_training.qat.full_model_contract import validate_optimizer_probe
    from ir_training.train.sharded_contract import resolve_zero_stage

    config_path = Path(plan["paths"]["config"])
    config = load_yaml(config_path)
    profile = (config.get("runtime") or {}).get("gpu_profile") or {}
    validate_h100_profile(profile)
    world_size = profile["world_size"]
    distributed_backend = (config.get("training") or {}).get("distributed_backend", "ddp")
    if distributed_backend != plan["options"].get("distributed_backend", "ddp"):
        raise ValueError("Optimizer preflight backend does not match the planned backend")
    zero_stage = resolve_zero_stage(config.get("training") or {})
    if zero_stage != plan["options"].get("zero_stage", 2):
        raise ValueError("Optimizer preflight ZeRO stage does not match the planned stage")
    log(
        "Validate optimizer preflight: "
        f"distributed_backend={distributed_backend}, zero_stage={zero_stage}"
    )
    accumulation_steps = (config.get("training") or {}).get("gradient_accumulation_steps")
    result = []
    for rank in range(world_size):
        path = Path(plan["paths"]["training"]) / f"full_optimizer_preflight_rank{rank}.json"
        report = _json(path)
        probe = report.get("probe") or {}
        if (
            report.get("training_config_sha256") != sha256(config_path)
            or type(report.get("rank")) is not int
            or report.get("rank") != rank
            or type(report.get("world_size")) is not int
            or report.get("world_size") != world_size
        ):
            raise ValueError(f"Rank {rank} optimizer preflight evidence is incomplete")
        try:
            validate_optimizer_probe(
                probe,
                accumulation_steps=accumulation_steps,
                world_size=world_size,
                local_rank=rank,
                distributed_backend=distributed_backend,
                zero_stage=zero_stage,
            )
            from ir_training.qat.full_model_contract import _validate_sharded_binding
            _validate_sharded_binding(config, probe, world_size)
        except ValueError as exc:
            raise ValueError(f"Rank {rank} optimizer preflight evidence is incomplete") from exc
        result.append(path)
    return result


def _available_evaluations(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    from ir_training.pipeline.golden_deployment import _evaluation

    output = Path(plan["options"]["output_dir"])
    evaluations = {}
    for stage in ("best_golden32", "final_golden35", "final_bixby50"):
        directory = output / f"evaluations/{stage}"
        cohort = "golden32" if stage == "best_golden32" else stage.removeprefix("final_")
        try:
            result, _ = _evaluation(
                directory,
                GOLDENS[cohort][1],
                artifact=Path(plan["paths"]["best_checkpoint"]),
            )
            if result.get("qat_applied") is not True:
                continue
            evaluations[stage] = result
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            # Failure reporting must remain available even when a partial or
            # corrupt cohort directory was the original failure.
            continue
    return evaluations


def _render_results(
    plan: dict[str, Any], *, status: str, error: str | None = None
) -> str:
    evaluations = _available_evaluations(plan)
    rows = []
    for stage, cohort in (
        ("best_golden32", "Golden32 (selection)"),
        ("final_golden35", "Golden35 (final only)"),
        ("final_bixby50", "Bixby50 (final only)"),
    ):
        result = evaluations.get(stage) or {}
        aggregate = result.get("aggregate") or {}
        metric = SELECTOR if stage == "best_golden32" else "generation_reward_v5_4_avg"
        rows.append(
            f"| {cohort} | {'evaluated' if result else 'not available'} | "
            f"{result.get('row_count', '-')} | {aggregate.get(metric, 'n/a')} | "
            f"{aggregate.get('schema_valid_strict_rate', 'n/a')} |"
        )
    export_manifest = Path(plan["paths"]["export"]) / "checkpoint_export_manifest.json"
    export_status = "not completed"
    if export_manifest.is_file():
        try:
            export_status = str(_json(export_manifest).get("status") or "unknown")
        except (OSError, ValueError, json.JSONDecodeError):
            export_status = "invalid partial evidence"
    return "\n".join(
        [
            "# All-parameter Gemma 4 E2B QAT results",
            "",
            f"Run status: {status}",
            "",
            "| Cohort | Status | Rows | Selection/final reward | Strict-valid rate |",
            "|---|---|---:|---:|---:|",
            *rows,
            "",
            "Golden32 alone selects the checkpoint. Golden35 and Bixby50 are final-only holdouts.",
            f"Export: {export_status}; experimental fresh-graph dynamic W2/W4/W8, not Google's official static-A8/mobile topology.",
            "MTP: disabled. Native LiteRT-LM/Android quality and speed: not evaluated by this training run.",
            *([f"Failure: {error}"] if error else []),
            "",
        ]
    )


def _scorecard(plan: dict[str, Any]) -> list[Path]:
    evaluations = _available_evaluations(plan)
    required = {"best_golden32", "final_golden35", "final_bixby50"}
    if set(evaluations) != required:
        raise ValueError("Scorecard is missing one or more required cohort evaluations")
    export_manifest = _json(
        Path(plan["paths"]["export"]) / "checkpoint_export_manifest.json"
    )
    if (
        export_manifest.get("status") != "exported_not_evaluated"
        or set(export_manifest.get("artifacts") or {}) != {"w248"}
    ):
        raise ValueError("Scorecard requires a completed validated W248 export")
    scorecard = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": "complete",
        "selected_checkpoint": plan["paths"]["best_checkpoint"],
        "selection_metric": SELECTOR,
        "golden35_used_for_selection": False,
        "bixby50_used_for_selection": False,
        "evaluations": evaluations,
        "export": {
            "path": plan["paths"]["export"],
            "variant": "w248",
            "experimental": True,
            "fresh_dense_graph": True,
            "official_static_mobile_layout": False,
        },
        "mtp": False,
        "native_runtime_evaluated": False,
    }
    scorecard_path = Path(plan["paths"]["scorecard"])
    _write(scorecard_path, scorecard)
    text = _render_results(plan, status="complete")
    results = Path(plan["paths"]["results"])
    results.write_text(text, encoding="utf-8")
    return [scorecard_path, results]


def _environment(plan: dict[str, Any]) -> dict[str, str]:
    environment = dict(os.environ)
    # Set before the disposable probe/training subprocesses import torch. Scope
    # this fragmentation mitigation to the full-parameter lane, and preserve
    # both current and legacy explicit user/scheduler allocator settings.
    if not any(name in environment for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")):
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    environment.update(
        A2UI_TENSORBOARD_ROOT=plan["options"]["tensorboard_root"],
        A2UI_TENSORBOARD_DETAIL="minimal",
        PYTHONUNBUFFERED="1",
        PYTHONIOENCODING="utf-8",
        TOKENIZERS_PARALLELISM="false",
    )
    return environment


def run_stage(plan: dict[str, Any], stage: str) -> list[Path]:
    values = plan["options"]
    output = Path(values["output_dir"])
    if stage == "environment":
        if values.get("distributed_backend") != "sharded":
            raise ValueError("The sharded environment stage must not run in the DDP lane")
        from ir_training.train.sharded_contract import validate_sharded_runtime

        # Before seed inspection, preparation, or any weight allocation. SFT
        # repeats this in each actual torchrun worker before loading its model.
        report = validate_sharded_runtime()
        path = output / "sharded_environment.json"
        _write(path, report)
        return [path]
    if stage == "assets":
        return _assets(plan)
    if stage == "prepare":
        prepare_data(plan["preparation"])
        return [
            output / "data_audit.json",
            *sorted((output / "prepared").glob("*.json*")),
            *map(Path, plan["preparation"]["source_files"]),
        ]
    if stage == "configure":
        return _configure(plan)
    if stage in {"preflight", "training"}:
        _scripts()
        from launch_review_training import verify_launch_binding

        verify_launch_binding(Path(plan["paths"]["config"]))
        log_path = output / f"logs/{stage}_launch.log"
        run_bounded_command(
            launch_command(plan, preflight=stage == "preflight"),
            log_path,
            _environment(plan),
            timeout_seconds=values["stage_timeout_seconds"],
            progress_seconds=values["progress_seconds"],
        )
        if stage == "preflight":
            return [log_path, *_optimizer_preflight_files(plan)]
        contract = _validate_best_checkpoint(plan)
        contract_path = output / "best_checkpoint_full_qat_contract.json"
        _write(contract_path, contract)
        checkpoint = Path(plan["paths"]["best_checkpoint"])
        final = Path(plan["paths"]["final_checkpoint"])
        for directory in (checkpoint, final):
            if not directory.is_dir() or not any(directory.glob("*.safetensors")):
                raise ValueError(f"Training did not publish the required full checkpoint: {directory}")
        return [
            log_path,
            contract_path,
            *[path for directory in (checkpoint, final) for path in sorted(directory.iterdir()) if path.is_file()],
        ]
    if stage in {"best_golden32", "final_golden35", "final_bixby50"}:
        _validate_best_checkpoint(plan)
        cohort = "golden32" if stage == "best_golden32" else stage.removeprefix("final_")
        run_bounded_command(
            evaluation_command(plan, cohort, stage),
            output / f"logs/{stage}_evaluation.log",
            _gpu_environment(plan),
            timeout_seconds=values["stage_timeout_seconds"],
            progress_seconds=values["progress_seconds"],
        )
        return _evaluation_files(plan, stage)
    if stage == "export":
        if values.get("allow_experimental_export") is not True:
            raise ValueError("Execution requires explicit --allow-experimental-export")
        _validate_best_checkpoint(plan)
        from ir_training.pipeline.checkpoint_export import run_checkpoint_export

        state = run_checkpoint_export(checkpoint_export_options(plan), execute=True)
        manifest = Path(plan["paths"]["export"]) / "checkpoint_export_manifest.json"
        artifacts = [Path(item["artifact"]) for item in state.get("artifacts", {}).values()]
        return [manifest, *artifacts]
    if stage == "scorecard":
        return _scorecard(plan)
    raise ValueError(f"Unsupported all-parameter QAT stage: {stage}")


def run_pipeline(
    options: FullParameterQATOptions,
    *,
    execute: bool = False,
    command_runner: Callable = run_bounded_command,
) -> dict[str, Any]:
    plan = build_plan(options)
    if not execute:
        return {**plan, "status": "plan_only"}
    if not options.allow_experimental_export:
        raise ValueError("--execute requires explicit --allow-experimental-export")
    output = Path(plan["options"]["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    plan_path = output / "full_parameter_qat_plan.json"
    manifest = output / "full_parameter_qat_manifest.json"
    _write(plan_path, plan)
    plan_digest = sha256(plan_path)
    state: dict[str, Any] = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "plan": plan,
        "status": "running",
        "active_stage": None,
        "completed": {},
    }
    _write(manifest, state)
    try:
        for index, stage in enumerate(plan["stages"], 1):
            state["active_stage"] = stage
            _write(manifest, state)
            log(f"All-parameter QAT stage {index}/{len(plan['stages'])}: {stage}")
            command = [
                sys.executable,
                str(training_root() / "scripts/run_full_parameter_qat_pipeline.py"),
                "--worker-stage",
                stage,
                "--plan-file",
                str(plan_path),
            ]
            command_runner(
                command,
                output / f"logs/{stage}.log",
                _environment(plan),
                timeout_seconds=(min(options.stage_timeout_seconds, 120.0)
                                 if stage == "environment" else options.stage_timeout_seconds),
                progress_seconds=options.progress_seconds,
            )
            receipt = _json(output / f"stage_receipts/{stage}.json")
            if (
                sha256(plan_path) != plan_digest
                or receipt.get("stage") != stage
                or receipt.get("plan_sha256") != plan_digest
            ):
                raise ValueError(f"Stage receipt is not bound to this run: {stage}")
            _verify_bindings(receipt.get("files") or {})
            state["completed"][stage] = receipt
            _write(manifest, state)
        state.update(status="complete", active_stage=None)
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _write(manifest, state)
        summary = _render_results(
            plan, status=state["status"], error=state.get("error")
        )
        Path(plan["paths"]["results"]).write_text(summary, encoding="utf-8")
        print(summary, flush=True)
    return state


def worker(plan_path: Path, stage: str) -> None:
    plan_digest = sha256(plan_path)
    plan = _json(plan_path)
    if plan.get("workflow") != WORKFLOW or stage not in plan.get("stages", []):
        raise ValueError("Worker requires a planned all-parameter QAT stage")
    output = Path(plan["options"]["output_dir"])
    if plan_path.resolve() != (output / "full_parameter_qat_plan.json").resolve():
        raise ValueError("Worker plan must be inside its own run output")
    manifest = _json(output / "full_parameter_qat_manifest.json")
    if (
        manifest.get("plan") != plan
        or manifest.get("status") != "running"
        or manifest.get("active_stage") != stage
    ):
        raise ValueError("Worker is not bound to the active pipeline stage")
    prior = plan["stages"][: plan["stages"].index(stage)]
    if list(manifest.get("completed", {})) != prior:
        raise ValueError("Pipeline stage prerequisites are incomplete or out of order")
    for name in prior:
        receipt = manifest["completed"][name]
        if receipt.get("plan_sha256") != plan_digest:
            raise ValueError(f"Prior stage belongs to a different plan: {name}")
        _verify_bindings(receipt.get("files") or {})
    with Progress(
        f"All-parameter QAT {stage}",
        unit="stage",
        interval=plan["options"]["progress_seconds"],
    ):
        files = run_stage(plan, stage)
        if sha256(plan_path) != plan_digest:
            raise ValueError("Pipeline plan changed while the worker ran")
        receipt = {"stage": stage, "plan_sha256": plan_digest, "files": _bindings(files)}
    _write(output / f"stage_receipts/{stage}.json", receipt)
