"""Mobile-seed retained-scale QAT, checked holdouts, and official-layout export.

This is deliberately not the dense SFT/PTQ deployment workflow. The released
mobile package is the layout authority; only trained projection codes change.
All expensive stages run in bounded subprocesses. No implicit training resume,
CPU evaluation fallback, or assertion of unmeasured device throughput is made.
"""
from __future__ import annotations

import copy
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
from ir_training.common.progress import log
from ir_training.pipeline.golden_training import (
    GOLDENS,
    GoldenTrainingOptions,
    _write,
    prepare_data,
    sha256,
)
from ir_training.pipeline.golden_training import (
    build_plan as build_preparation_plan,
)
from ir_training.qat.mobile_training_seed import (
    OFFICIAL_LITERTLM_SHA256,
    OFFICIAL_MOBILE_MODEL_ID,
    OFFICIAL_MOBILE_SAFETENSORS_SHA256,
)
from ir_training.qat.numeric_preflight import (
    OFFICIAL_MOBILE_WORKFLOW,
    RETAINED_MOBILE_POLICY,
    resolve_numeric_policy,
)

SELECTOR = "unique_source_generation_reward_v5_4_avg"
WORKFLOW = OFFICIAL_MOBILE_WORKFLOW
NO_OP_CHECKS = frozenset({
    "official_artifact_sha256_pinned", "retained_training_config_verified",
    "materialized_seed_provenance_verified", "retained_qparams_verified",
    "exact_205_key_buffer_bijection", "materialized_seed_mapping_205",
    "materialized_seed_projections_processed_205", "materialized_seed_quantization_verified",
    "unique_materialized_code_buffers_205", "every_materialized_code_buffer_matches_official",
    "zero_adapter_target_byte_exact", "frozen_72_byte_exact", "no_target_buffer_changed",
    "official_retained_weight_scales_exact", "official_retained_a8_scales_exact",
    "weight_qparams_byte_exact", "activation_a8_qparams_byte_exact",
    "all_tensor_qparams_byte_exact", "graph_layout_execution_identity",
    "virtual_package_identity", "official_source_untouched",
})


@dataclass(frozen=True)
class OfficialMobileOptions:
    model_dir: Path
    input_dir: Path | None
    source_safetensors: Path
    official_litertlm: Path
    output_dir: Path
    prepared_input_dir: Path | None = None
    exporter_python: Path | None = None
    devices: str = "auto"
    epochs: float | None = None
    steps: int | None = None
    resume_from_checkpoint: Path | None = None
    learning_rate: float | None = None
    lora_rank: int | None = None
    lora_alpha: int | None = None
    seed: int | None = None
    eval_steps: int = 500
    golden_every_steps: int = 1000
    max_seq_length: int = 4096
    max_input_tokens: int = 5120
    max_new_tokens: int = 2048
    microbatch: int | None = None
    effective_batch: int | None = None
    dataloader_workers: int | None = None
    prepare_workers: int = 0
    preparation_cache: bool = True
    preparation_cache_dir: Path | None = None
    tensorboard_root: str = "/tensorboard"
    progress_seconds: float = 10
    stage_timeout_seconds: float = 172800
    generation_timeout_seconds: float = 7200
    augmentation: str = "none"
    augmentation_max_extra_fraction: float = 0.10
    augmentation_max_family_repeats: int = 2
    augmentation_teacher_model: str = "muse_glimmer_30b_sglang_reasoning_dflash"
    augmentation_python: Path | None = None
    augmentation_max_samples: int = 500
    augmentation_timeout_seconds: float = 7200
    benchmark_android: bool = False
    adb: str = "adb"
    serial: str | None = None


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")  # noqa: TRY004 -- invalid file content
    return value


def _scripts() -> None:
    path = str(training_root() / "scripts")
    if path not in sys.path:
        sys.path.insert(0, path)


def build_plan(options: OfficialMobileOptions) -> dict[str, Any]:
    """Offline plan: no CUDA probe, downloads, model load, or filesystem writes."""
    values = asdict(options)
    for key in ("model_dir", "input_dir", "source_safetensors", "official_litertlm",
                "output_dir", "preparation_cache_dir", "resume_from_checkpoint"):
        if values[key] is not None:
            values[key] = str(Path(values[key]).expanduser().resolve())
    if values["prepared_input_dir"] is not None:
        values["prepared_input_dir"] = os.path.abspath(os.path.expanduser(str(values["prepared_input_dir"])))
    # venv/bin/python is often a symlink. Resolving it would silently use the
    # system interpreter and lose the isolated exporter dependencies on Linux.
    values["exporter_python"] = os.path.abspath(os.path.expanduser(
        str(values["exporter_python"] or sys.executable)))
    values["augmentation_python"] = os.path.abspath(os.path.expanduser(
        str(values["augmentation_python"] or sys.executable)))
    if options.input_dir and options.prepared_input_dir:
        raise ValueError("Choose --input-dir or --prepared-input-dir, not both")
    if not options.input_dir and not options.prepared_input_dir and not values["resume_from_checkpoint"]:
        raise ValueError("Choose --input-dir or --prepared-input-dir")
    if options.prepared_input_dir and options.augmentation != "none":
        raise ValueError("--prepared-input-dir cannot be combined with --augmentation")
    if values["resume_from_checkpoint"] and options.prepared_input_dir:
        raise ValueError("--prepared-input-dir cannot override the saved dataset on resume")
    if values["resume_from_checkpoint"] and options.augmentation != "none":
        raise ValueError("--augmentation cannot change data on resume; reuse the saved dataset without the flag or start a fresh experiment")
    resume = None
    if values["resume_from_checkpoint"]:
        from ir_training.train.mobile_resume import horizon_record, source_config
        checkpoint = Path(values["resume_from_checkpoint"])
        source_path, saved, _ = source_config(checkpoint)
        if values["input_dir"] is None:
            # Only the read-only preparation plan needs an input path here.
            # Resume itself uses the saved dataset and verifies its launch binding.
            values["input_dir"] = str(Path(saved["run"]["dataset_dir"]).resolve())
        resume = {"checkpoint": str(checkpoint), "source_config": str(source_path),
                  "source_config_sha256": sha256(source_path),
                  "metadata_sha256": sha256(checkpoint / "training_metadata.json"),
                  "prepared": saved["run"]["dataset_dir"]}
        if values["epochs"] is None:
            values["epochs"] = saved["training"].get("epochs", 2)
        if values["steps"] is None:
            values["steps"] = saved["training"].get("max_steps")
        requested = copy.deepcopy(saved)
        requested["training"]["epochs"] = values["epochs"]
        if values["steps"] is not None:
            requested["training"]["max_steps"] = values["steps"]
        resume["horizon"] = horizon_record(checkpoint, requested)
    if values["epochs"] is None:
        values["epochs"] = 2.0
    # Omitted knobs keep historical fresh defaults, but inherit the saved
    # recipe on continuation. An experiment is a fresh run, never a resume
    # that quietly changes adapter capacity, optimizer settings, or RNG.
    recipe_options = {
        "lora_rank": ("lora", "r", 16),
        "lora_alpha": ("lora", "alpha", 16),
        "seed": ("training", "seed", 42),
        "learning_rate": ("training", "learning_rate", 1e-5),
    }
    for name, (section, key, default) in recipe_options.items():
        inherited = saved[section][key] if resume else default
        if values[name] is None:
            values[name] = inherited
        elif resume and values[name] != inherited:
            raise ValueError(f"--{name.replace('_', '-')} cannot change on resume; start a fresh experiment")
    for name in ("lora_rank", "lora_alpha"):
        if type(values[name]) is not int or values[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("max_seq_length", "max_input_tokens", "max_new_tokens"):
        if type(values[name]) is not int or values[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(values["seed"]) is not int or not 0 <= values["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    for name in ("stage_timeout_seconds", "generation_timeout_seconds", "progress_seconds", "learning_rate"):
        if isinstance(values[name], bool) or not isinstance(values[name], (int, float)) or not math.isfinite(values[name]) or values[name] <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if options.benchmark_android and not options.serial:
        raise ValueError("--benchmark-android requires an explicit --serial")
    output, seed = Path(values["output_dir"]), Path(values["model_dir"])
    if resume:
        source_run = Path(saved["run"]["output_dir"]).resolve().parent
        if output.is_relative_to(source_run) or source_run.is_relative_to(output):
            raise ValueError("Continuation output must be fresh and separate from its source run")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", output.name):
        raise ValueError("Output directory name must be a safe run ID (1-96 characters)")
    for protected in (seed, Path(values["prepared_input_dir"] or values["input_dir"])):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("Run output must be separate from model and input directories (no nested paths)")
    for path in (seed / "mobile_training_seed_manifest.json", seed / "mobile_qparams.json",
                 seed / "mobile_qparams.safetensors", Path(values["source_safetensors"]),
                 Path(values["official_litertlm"]), Path(values["exporter_python"])):
        if not path.is_file():
            raise FileNotFoundError(f"Required mobile artifact missing: {path}. See OFFICIAL_MOBILE_QAT_PIPELINE.md")
        if path.is_relative_to(output):
            raise ValueError("Run output must not contain any input artifact")
    preparation = build_preparation_plan(
        GoldenTrainingOptions(
            model_dir=seed, input_dir=Path(values["input_dir"]) if values["input_dir"] else None, output_dir=output,
            prepared_input_dir=Path(values["prepared_input_dir"]) if values["prepared_input_dir"] else None,
            # Only this shared CPU preparation is reused; its dense configure/train
            # commands are NEVER executed by the mobile workflow.
            epochs=values["epochs"], steps=values["steps"], eval_steps=options.eval_steps,
            golden_every_steps=options.golden_every_steps, max_seq_length=options.max_seq_length,
            max_input_tokens=options.max_input_tokens, max_new_tokens=options.max_new_tokens,
            devices=options.devices, microbatch=options.microbatch, effective_batch=options.effective_batch,
            prepare_workers=options.prepare_workers, preparation_cache=options.preparation_cache,
            preparation_cache_dir=options.preparation_cache_dir, progress_seconds=options.progress_seconds,
            tensorboard_root=options.tensorboard_root, learning_rate=values["learning_rate"], seed=values["seed"],
            augmentation=options.augmentation,
            augmentation_max_extra_fraction=options.augmentation_max_extra_fraction,
            augmentation_max_family_repeats=options.augmentation_max_family_repeats,
            augmentation_teacher_model=options.augmentation_teacher_model,
            augmentation_python=options.augmentation_python,
            augmentation_max_samples=options.augmentation_max_samples,
            augmentation_timeout_seconds=options.augmentation_timeout_seconds,
        ),
        preparation_only=True,
    )
    preparation = {key: preparation[key] for key in ("options", "source_files", "shared_prompt", "goldens")}
    train_root = output / "training" / output.name
    paths = {
        "prepared": Path(resume["prepared"]) if resume else output / ("augmented" if options.augmentation != "none" else "prepared"),
        "source_config": output / "configs/mobile_training.yaml",
        "config": train_root / "launch/resolved_training_config.yaml",
        "launch_plan": train_root / "launch/launch_plan.json",
        "preflight_report": train_root / "launch/preflight_report.json",
        "no_op_export_report": output / "pretraining_noop_export.json",
        "best_checkpoint": train_root / "best_golden_checkpoint",
        "merged": output / "merged_best_hf",
        "export_dir": output / "retained_scale_export",
        "export_report": output / "retained_scale_export/gemma4_retained_scale_code_only_report.json",
        "litertlm": output / "retained_scale_export/gemma4_e2b_a2ui_mobile.litertlm",
        "mobile_config": output / "configs/mobile_deployment.yaml",
        "android_report": output / "android_gpu/target_only/android_litertlm_gpu_parity_report.json",
    }
    return {
        "schema_version": 1, "workflow": WORKFLOW, "options": values,
        "preparation": preparation, "resume": resume, "paths": {key: str(path) for key, path in paths.items()},
        "stages": ["assets", "prepare", *(["augment"] if not resume and options.augmentation != "none" else []), "configure", "no_op_export", "preflight", "training",
                   *[f"best_{name}" for name in GOLDENS], "merge", "export",
                   *(["android_benchmark"] if options.benchmark_android else [])],
        "selection": {"cohort": "golden32", "metric": SELECTOR, "golden35_used": False, "bixby50_used": False},
        "format": {"name": "official-mobile wNa8o8", "target_weight_bits": [2, 4, 8],
                   "activation_bits": 8, "official_sha256": OFFICIAL_LITERTLM_SHA256,
                   "layout_policy": "change only 205 projection code buffers; retain graph, scales and 72 frozen constants"},
        "mtp": {"training": False, "inference": False, "official_section_preserved_but_unused": True},
        "speed_parity": "unverified until matched Android GPU target-only benchmark passes",
        "native_litert_golden_tests": False,
        "native_quality_validation": "separate Android command; required before native quality promotion",
        "training_numeric_contract": {"activation_quantizer": "gemma_mobile_srq",
            "frozen_w8_activation_modules": 70, "native_kv_cache_simulated": False,
            "native_numeric_parity_verified": False},
    }


def training_config(plan: dict, profile: dict, preparation_report: dict) -> dict:
    """Resolve the existing strict mobile recipe without editing any source YAML."""
    values, paths = plan["options"], plan["paths"]
    seed, output = Path(values["model_dir"]), Path(values["output_dir"])
    if plan.get("resume"):
        config = load_yaml(Path(plan["resume"]["source_config"]))
        if sha256(Path(plan["resume"]["source_config"])) != plan["resume"]["source_config_sha256"]:
            raise ValueError("Source resume config changed after planning")
    else:
        config = copy.deepcopy(load_yaml(training_root() / "configs/models/gemma4_e2b_mobile_seed_ir_qat_sft.yaml"))
    prepared = Path(paths.get("prepared", str(output / "prepared")))
    if profile["dtype"] != "bfloat16":
        raise ValueError("Official mobile retained-scale QAT requires native BF16 GPUs")
    from ir_training.train.gpu_profile import apply_gpu_profile
    from ir_training.train.recipe import validate_effective_batch, validate_sft_recipe
    apply_gpu_profile(config, profile)
    config["run"].update(dataset_dir=str(prepared), prepared_manifest_required=True,
                         dataset_format="a2ui_express_v1", purpose=WORKFLOW)
    # Deliberate v2 opt-in, not an implicit model-name exception. Standalone
    # historical YAMLs keep their BF16 parity thresholds unchanged.
    config["preflight"]["numeric_policy"] = RETAINED_MOBILE_POLICY
    resolve_numeric_policy(config)
    config["model"].update(model_source=str(seed), tokenizer_source=str(seed),
        mobile_training_seed_manifest=str(seed / "mobile_training_seed_manifest.json"),
        mobile_qparams_contract=str(seed / "mobile_qparams.json"),
        chat_template_kwargs=preparation_report["tokenizer"].get("chat_template_kwargs") or {},
        max_output_tokens=values["max_new_tokens"])
    training = config["training"]
    # Extending a short step-capped run must not also extend its originally
    # capped evaluation/save interval, including across multiple continuations.
    cadence_cap = (plan["resume"]["horizon"]["original"]["max_steps"]
                   if plan.get("resume") else values["steps"])
    cadence = min(values["eval_steps"], cadence_cap) if cadence_cap else values["eval_steps"]
    golden_interval = max(1, math.ceil(values["golden_every_steps"] / cadence))
    # The independent full-parameter lane also reuses this base builder. Only
    # official LoRA plans opt into experiment knobs; old plans keep defaults.
    if plan.get("workflow") == WORKFLOW:
        config["lora"].update(r=values.get("lora_rank", config["lora"]["r"]),
                              alpha=values.get("lora_alpha", config["lora"]["alpha"]))
        if "seed" in values:
            training["seed"] = values["seed"]
            # Do not add a field to historical checkpoint continuations.
            # Fresh experiments decouple sampler RNG from adapter shapes.
            if not plan.get("resume"):
                training["data_seed"] = values["seed"]
    training.update(epochs=values["epochs"], learning_rate=values["learning_rate"],
        eval_steps=cadence, save_steps=cadence, eval_strategy="steps", max_seq_length=values["max_seq_length"],
        tensorboard_root=values["tensorboard_root"], tensorboard_subdir="training", tensorboard_detail="minimal",
        token_cache=True, token_cache_dir=(training["token_cache_dir"] if plan.get("resume") else plan["preparation"]["options"]["token_cache_dir"]),
        disable_cudnn_sdpa=True, backward_preflight=True, overflow_policy="error", trainer_backend="hf")
    if values["steps"] is not None:
        training["max_steps"] = values["steps"]
    training["refuse_resume"] = not bool(plan.get("resume"))
    if plan.get("resume"):
        from ir_training.train.mobile_resume import POLICY, horizon_record
        training.update(resume_from_checkpoint=values["resume_from_checkpoint"], resume_policy=POLICY)
        training["resume_horizon"] = horizon_record(Path(values["resume_from_checkpoint"]), config)
    config["golden_eval"].update(dataset_dir=str(prepared), split="golden32",
        split_path=str(prepared / "golden32.jsonl"), max_rows=32, required_rows=32,
        max_input_tokens=values.get("max_input_tokens", values["max_seq_length"]),
        max_new_tokens=values["max_new_tokens"],
        metric_version="v5_4", metric_for_best_model=SELECTOR,
        interval=golden_interval, evaluate_at_end=True,
        requested_every_optimizer_steps=values["golden_every_steps"], use_cache=True,
        resolved_every_optimizer_steps=golden_interval * cadence,
        stop_strings=["</a2ui>"], metric_log_prefix="golden32", tensorboard_evaluation_name="periodic_golden32",
        best_checkpoint_dir=paths["best_checkpoint"])
    config["final_evaluation_datasets"] = copy.deepcopy(preparation_report["final_evaluation_datasets"])
    # There is no assistant/drafter training branch in this workflow.
    config.pop("qat_mtp", None)
    validate_sft_recipe(config)
    validate_effective_batch(training, profile["world_size"])
    return config


def mobile_export_config(plan: dict) -> dict:
    paths, values = plan["paths"], plan["options"]
    config = copy.deepcopy(load_yaml(training_root() / "configs/pipelines/gemma4_e2b_mobile_mtp.yaml"))
    pipeline = config["pipeline"]
    pipeline.update(id=Path(values["output_dir"]).name, output_dir=values["output_dir"], training_config=paths["config"])
    pipeline["source"].update(base_litertlm=values["official_litertlm"], merged_model_dir=paths["merged"],
                              output_litertlm=paths["litertlm"])
    pipeline["mtp"].update(enabled=False, train_assistant=False, weight_source="official", preserve_official_section=True)
    pipeline["public_export"]["enabled"] = False
    pipeline["retained_scale_export"].update(output_dir=paths["export_dir"], report=paths["export_report"],
                                             official_artifact_sha256=OFFICIAL_LITERTLM_SHA256)
    pipeline["android"].update(output_dir=str(Path(values["output_dir"]) / "android_gpu"))
    return config


def _yaml(path: Path, payload: dict) -> None:
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)


def _file_bindings(paths) -> dict[str, str]:
    result = {}
    for path in paths:
        path = Path(path).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"Required stage artifact is not a file: {path}")
        result[str(path)] = sha256(path)
    return result


def _verify_bindings(records: dict[str, str]) -> None:
    if not records:
        raise ValueError("Stage has no bound evidence")
    for filename, digest in records.items():
        path = Path(filename)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Bound artifact changed or disappeared: {path}")


def _launch(plan: dict):
    _scripts()
    import run_gemma4_mobile_qat as mobile
    launch = _json(Path(plan["paths"]["launch_plan"]))
    paths = {key: Path(value) for key, value in launch["paths"].items()}
    return mobile, launch, paths


def _configure(plan: dict) -> list[Path]:
    _scripts()
    import run_gemma4_mobile_qat as mobile
    from prepare_review_training import verify_prepared

    from ir_training.train.gpu_profile import build_gpu_profile, detect_cuda_devices
    values, output = plan["options"], Path(plan["options"]["output_dir"])
    profile = build_gpu_profile(detect_cuda_devices(), model="e2b", devices=values["devices"],
        microbatch=values["microbatch"], effective_batch=values["effective_batch"],
        dataloader_workers=values["dataloader_workers"])
    prepared = Path(plan["paths"]["prepared"])
    report = verify_prepared(prepared, prepared / "golden32.jsonl",
        golden35=prepared / "golden35.jsonl", bixby50=prepared / "bixby50.jsonl",
        max_sequence=values["max_seq_length"],
        max_prompt=values.get("max_input_tokens", values["max_seq_length"]))
    source = Path(plan["paths"]["source_config"])
    _yaml(source, training_config(plan, profile, report))
    launch, resolved, paths = mobile.build_launch_plan(source, run_id=output.name, runs_root=output / "training",
        num_gpus=profile["world_size"], source_safetensors=values["source_safetensors"], host_gpu_profile=profile)
    if not launch["checks"]["contract_ok"]:
        raise ValueError(f"Mobile launch contract failed: {launch['checks']['issues']}")
    if plan.get("resume"):
        from ir_training.train.mobile_resume import (
            verify_continuation,
            verify_export_lineage,
        )
        checkpoint = Path(values["resume_from_checkpoint"])
        if sha256(checkpoint / "training_metadata.json") != plan["resume"]["metadata_sha256"]:
            raise ValueError("Resume checkpoint changed after planning")
        verify_export_lineage(Path(plan["resume"]["source_config"]), checkpoint)
        launch["resume_state"] = verify_continuation(checkpoint, resolved)
    mobile._reserve_run(launch, resolved, paths)
    if paths["resolved_config"].resolve() != Path(plan["paths"]["config"]).resolve():
        raise ValueError("Mobile resolved config path differs from pipeline plan")
    seed = Path(values["model_dir"])
    report.update(training_executed=False, model_loaded=False, profile="official_mobile",
        training_config_sha256=sha256(paths["resolved_config"]), gpu_profile=profile,
        model_files={path.name: sha256(path) for path in sorted(seed.iterdir())
                     if path.is_file() and path.suffix in {".json", ".safetensors", ".model", ".jinja"}})
    prepared_report = paths["launch_dir"] / "preparation_report.json"
    _write(prepared_report, report)
    # Bind this additional contract before preflight/training, not retrospectively.
    launch["bound_artifacts"]["prepared_evaluation_report"] = mobile._file_identity(prepared_report)
    _write(paths["launch_plan"], launch)
    deployment_config = Path(plan["paths"]["mobile_config"])
    _yaml(deployment_config, mobile_export_config(plan))
    log(f"GPU profile: {profile['world_size']} ranks; microbatch {profile['microbatch']}; "
        f"accumulation {profile['gradient_accumulation_steps']}; global batch {profile['effective_batch_size']}; "
        f"CPU data workers {profile['total_dataloader_workers']}")
    return [source, paths["resolved_config"], paths["launch_plan"], prepared_report, deployment_config]


def evaluation_command(plan: dict, cohort: str) -> list[str]:
    values, paths = plan["options"], plan["paths"]
    output = Path(values["output_dir"])
    count = plan["preparation"]["goldens"][cohort]["rows"]
    return [sys.executable, str(training_root() / "scripts/evaluate_checkpoint_on_golden.py"),
        "--config", paths["config"], "--checkpoint", paths["best_checkpoint"], "--checkpoint-kind", "adapter",
        "--qat-mode", "on", "--require-prepared-contract", "--require-gpu", "--devices", "auto",
        "--split", str(Path(paths["prepared"]) / f"{cohort}.jsonl"), "--max-rows", str(count), "--required-rows", str(count),
        "--max-input-tokens", str(values.get("max_input_tokens", values["max_seq_length"])),
        "--max-new-tokens", str(values["max_new_tokens"]),
        "--output-dir", str(output / f"evaluations/best_{cohort}"), "--run-id", output.name,
        "--evaluation-name", f"best_{cohort}", "--tensorboard-root", values["tensorboard_root"],
        "--generation-timeout-seconds", str(values["generation_timeout_seconds"]), "--metric-version", "v5_4"]


def _mobile_command(plan: dict, stage: str) -> list[str]:
    values, paths = plan["options"], plan["paths"]
    flags = {"merge": "--execute-merge", "export": "--execute-retained-scale-export",
             "android_benchmark": "--validate-android-gpu"}
    python = values["exporter_python"] if stage == "export" else sys.executable
    command = [python, str(training_root() / "scripts/run_gemma4_e2b_mobile_mtp.py"),
               "--config", paths["mobile_config"], flags[stage]]
    if stage == "android_benchmark":
        command += ["--adb", values["adb"], "--serial", values["serial"]]
    return command


def benchmark_command(plan: dict) -> list[str]:
    """Portable command also emitted when the training host has no Android device."""
    values, paths = plan["options"], plan["paths"]
    return [sys.executable, str(training_root() / "scripts/benchmark_android_litertlm_gpu_parity.py"),
        "--official", values["official_litertlm"], "--candidate", paths["litertlm"],
        "--output-dir", str(Path(values["output_dir"]) / "android_gpu/target_only"),
        "--max-num-tokens", "2048", "--output-tokens", "64", "--warm-runs", "3",
        "--max-throughput-regression-percent", "10", "--top-k", "1", "--top-p", "1.0",
        "--temperature", "0", "--seed", "42", "--prompt", "Write exactly one hundred numbered words.",
        "--adb", values["adb"], "--serial", values["serial"] or "<ANDROID_SERIAL>"]


def native_quality_command(plan: dict) -> list[str]:
    """Separate post-export gate; never needs Vulkan on the training host."""
    values = plan["options"]
    output = Path(values["output_dir"])
    return [sys.executable, str(training_root() / "scripts/evaluate_official_mobile_native.py"),
            "--run-dir", str(output), "--output-dir", str(output.parent / f"{output.name}_native_quality"),
            "--adb", values["adb"], "--serial", values["serial"] or "<ANDROID_SERIAL>"]


def no_op_export_command(plan: dict) -> list[str]:
    values, paths = plan["options"], plan["paths"]
    seed = Path(values["model_dir"])
    return [values["exporter_python"], str(training_root() / "scripts/verify_gemma4_retained_scale_pretraining.py"),
            "--official-litertlm", values["official_litertlm"],
            "--official-artifact-sha256", OFFICIAL_LITERTLM_SHA256,
            "--training-config", paths["config"],
            "--mobile-training-seed-manifest", str(seed / "mobile_training_seed_manifest.json"),
            "--mobile-qparams-contract", str(seed / "mobile_qparams.json"),
            "--zero-adapter-checkpoint", str(seed), "--report", paths["no_op_export_report"]]


def _require_no_op_export(plan: dict) -> dict:
    report = _json(Path(plan["paths"]["no_op_export_report"]))
    checks = report.get("checks") or {}
    if (report.get("passed") is not True or report.get("gate_status") != "PASSED"
            or report.get("mode") != "retained_scale_pretraining_noop_v1"
            or report.get("official_source_after_sha256") != OFFICIAL_LITERTLM_SHA256
            or not isinstance(checks, dict) or not NO_OP_CHECKS.issubset(checks)
            or not all(value is True for value in checks.values())):
        raise ValueError("Training requires a passing real retained-scale no-op export gate")
    values, paths = plan["options"], plan["paths"]
    config_identity = report.get("resolved_training_config_identity") or {}
    seed_identity = report.get("mobile_training_seed") or {}
    qparams_identity = report.get("mobile_qparams") or {}
    seed = Path(values["model_dir"])
    bindings = (
        (config_identity, "path", "sha256", Path(paths["config"])),
        (seed_identity, "path", "manifest_sha256", seed / "mobile_training_seed_manifest.json"),
        (qparams_identity, "contract_path", "contract_sha256", seed / "mobile_qparams.json"),
    )
    for identity, path_key, hash_key, expected in bindings:
        if (identity.get("verified") is not True
                or Path(str(identity.get(path_key) or "")).resolve() != expected.resolve()
                or identity.get(hash_key) != sha256(expected)):
            raise ValueError(f"Pretraining no-op report is stale or bound to different inputs: {expected}")
    if Path(str(report.get("official_litertlm") or "")).resolve() != Path(values["official_litertlm"]).resolve():
        raise ValueError("Pretraining no-op report belongs to a different official package path")
    return report


def _environment(plan: dict, *, gpu: bool = False) -> dict[str, str]:
    values = plan["options"]
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "A2UI_TENSORBOARD_ROOT": values["tensorboard_root"],
           "A2UI_TENSORBOARD_DETAIL": "minimal", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    if gpu:
        from ir_training.train.gpu_profile import (
            training_environment,
            verify_gpu_profile,
        )
        _, launch, _ = _launch(plan)
        verify_gpu_profile(launch["host_gpu_profile"])
        env.update(training_environment(launch["host_gpu_profile"], tensorboard_root=values["tensorboard_root"]))
        env.update(A2UI_TENSORBOARD_ROOT=values["tensorboard_root"], A2UI_TENSORBOARD_DETAIL="minimal",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return env


def probe_export_environment(official_litertlm: Path | None = None) -> dict:
    """Cheap schema roundtrip in the actual exporter interpreter, without a model."""
    _scripts()
    import importlib.metadata

    import build_gemma4_retained_scale_litertlm as exporter
    import flatbuffers
    import run_gemma4_e2b_mobile_mtp  # noqa: F401 -- entry point used by this lane
    from tflite_schema_compat import schema_module
    schema = schema_module("Model")
    builder = flatbuffers.Builder(128)
    schema.ModelStart(builder)
    schema.ModelAddVersion(builder, 3)
    offset = schema.ModelEnd(builder)
    builder.Finish(offset, file_identifier=b"TFL3")
    encoded = bytes(builder.Output())
    if exporter._schema_model(encoded).Version() != 3:
        raise ValueError("Exporter FlatBuffer roundtrip failed")
    fingerprint = exporter._graph_report(encoded)
    for name in ("graph", "graph_without_buffer_indices"):
        if not fingerprint[name].get("available") or not fingerprint[name].get("execution_contract_complete"):
            raise ValueError("Exporter schema cannot decode execution contracts")
    versions = {}
    for package in ("numpy", "safetensors", "flatbuffers", "ai-edge-litert", "litert-torch", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    official_graphs = []
    if official_litertlm is not None:
        # Memory-mapped static decoding, not GPU inference or weight loading.
        # Detect unsupported official operator-option schemas before training.
        package = exporter.inspect_litertlm(official_litertlm, inspect_tflite=True)
        exporter._section_by_model_type(package, exporter.MTP_MODEL_TYPE)
        for model_type in (exporter.TARGET_MODEL_TYPE,):
            section = exporter._section_by_model_type(package, model_type)
            graph = section.get("graph") or {}
            if not graph.get("available") or not graph.get("execution_contract_complete"):
                raise ValueError(f"Exporter cannot completely decode official {model_type} graph")
            official_graphs.append({"model_type": model_type,
                **{key: graph[key] for key in ("structural_sha256", "quantization_layout_sha256", "execution_contract_sha256")}})
    return {"passed": True, "python": sys.executable, "versions": versions,
            "schema_roundtrip": True, "model_loaded": False,
            "official_graphs": official_graphs,
            "full_official_model_compatibility_tested": False}


def _assets(plan: dict) -> list[Path]:
    from ir_training.qat.mobile_qparams import verify_mobile_qparams_contract
    from ir_training.qat.mobile_training_seed import (
        verify_configured_mobile_training_seed,
    )
    from ir_training.qat.published_qparams import verify_published_activation_scales
    values = plan["options"]
    seed, output = Path(values["model_dir"]), Path(values["output_dir"])
    model = {"model_id": OFFICIAL_MOBILE_MODEL_ID, "model_source": str(seed),
             "mobile_training_seed_manifest": str(seed / "mobile_training_seed_manifest.json"),
             "mobile_qparams_contract": str(seed / "mobile_qparams.json")}
    report = verify_configured_mobile_training_seed(model, base=training_root(), require_materialized=True)
    qparams = verify_mobile_qparams_contract(seed / "mobile_qparams.json", base=training_root())
    if not report.get("verified") or not qparams.get("verified"):
        raise ValueError("Mobile seed/retained qparams verification failed; dense SFT seeds cannot use this pipeline")
    inputs = {values["source_safetensors"]: OFFICIAL_MOBILE_SAFETENSORS_SHA256,
              values["official_litertlm"]: OFFICIAL_LITERTLM_SHA256}
    _verify_bindings(inputs)
    published_a8 = verify_published_activation_scales(
        Path(values["source_safetensors"]),
        seed / "mobile_training_seed_manifest.json",
        seed / "mobile_qparams.json",
        verified_source_sha256=OFFICIAL_MOBILE_SAFETENSORS_SHA256,
    )
    if published_a8.get("verified") is not True:
        raise ValueError("Retained activation scales do not match the pinned public mobile checkpoint")
    result = output / "mobile_assets_verified.json"
    _write(result, {"seed": report, "qparams": qparams, "official_inputs": inputs,
                    "published_activation_scales": published_a8,
                    "mtp_training": False, "mtp_inference": False})
    # Verify imports, schema APIs and static official graph decoding before
    # expensive training. This does not prove kernel/runtime compatibility.
    environment_report = output / "logs/export_environment.json"
    command = [values["exporter_python"], "-c",
        ("import sys,json; from pathlib import Path; sys.path.insert(0, 'training/src'); "
        "from ir_training.pipeline.official_mobile import probe_export_environment,_write; "
        "report=probe_export_environment(Path(sys.argv[1])); _write(Path(sys.argv[2]),report); "
        "print(json.dumps(report, indent=2))"), values["official_litertlm"], str(environment_report)]
    run_bounded_command(command, output / "logs/export_environment.log", _environment(plan),
                        timeout_seconds=300, progress_seconds=values["progress_seconds"],
                        emit_heartbeat=False)
    return [result, environment_report, *map(Path, inputs), seed / "mobile_training_seed_manifest.json", seed / "mobile_qparams.json"]


def run_stage(plan: dict, stage: str) -> list[Path]:
    """One isolated worker, also used by unit tests with tiny/mocked models."""
    values, paths = plan["options"], plan["paths"]
    output = Path(values["output_dir"])
    if stage == "assets":
        return _assets(plan)
    if stage == "prepare":
        if plan.get("resume"):
            from launch_review_training import verify_launch_binding
            source = Path(plan["resume"]["source_config"])
            verify_launch_binding(source)
            prepared = Path(paths["prepared"])
            _write(output / "data_audit.json", {"mode": "reuse_verified_prepared_data", "source_config": str(source),
                   "source_config_sha256": sha256(source), "prepared": str(prepared)})
            return [output / "data_audit.json", source, source.parent / "preparation_report.json",
                    *sorted(prepared.glob("*.json*"))]
        prepare_data(plan["preparation"])
        prepared_files = (sorted(path for path in (output / "prepared").rglob("*") if path.is_file())
                          if values.get("prepared_input_dir") else sorted((output / "prepared").glob("*.json*")))
        return [output / "data_audit.json", *prepared_files,
                *map(Path, plan["preparation"]["source_files"])]
    if stage == "augment":
        if plan.get("resume") or values.get("augmentation", "none") == "none":
            raise ValueError("Augmentation requires an explicitly enabled fresh run")
        report = {}
        if values["augmentation"] == "semantic":
            from ir_training.data.semantic_augmentation import augment_training_at_startup
            report = augment_training_at_startup(plan["preparation"])
        else:
            from ir_training.data.augmentation import augment_prepared_training
            augment_prepared_training(output / "prepared", output / "augmented", seed=values["seed"],
                max_extra_fraction=values["augmentation_max_extra_fraction"],
                max_family_copies=values["augmentation_max_family_repeats"], progress_seconds=values["progress_seconds"])
        _scripts()
        from prepare_review_training import verify_prepared
        verify_prepared(output / "augmented", output / "prepared/golden32.jsonl",
                        golden35=output / "prepared/golden35.jsonl", bixby50=output / "prepared/bixby50.jsonl",
                        max_sequence=values["max_seq_length"], max_prompt=values["max_input_tokens"])
        return sorted({*[path for path in (output / "augmented").rglob("*") if path.is_file()],
                       *map(Path, report.get("artifact_paths", []))})
    if stage == "configure":
        return _configure(plan)
    if stage == "no_op_export":
        env = _environment(plan)
        env["CUDA_VISIBLE_DEVICES"] = ""
        run_bounded_command(no_op_export_command(plan), output / "logs/no_op_export_worker.log", env,
                            timeout_seconds=values["stage_timeout_seconds"],
                            progress_seconds=values["progress_seconds"], emit_heartbeat=False)
        _require_no_op_export(plan)
        return [Path(paths["no_op_export_report"])]
    if stage in {"preflight", "training"}:
        _require_no_op_export(plan)
        _scripts()
        from launch_review_training import verify_launch_binding
        mobile, launch, launch_paths = _launch(plan)
        verify_launch_binding(launch_paths["resolved_config"])
        changed = mobile._verify_bound_artifacts(launch)
        if changed:
            raise ValueError(f"Mobile launch inputs changed: {changed}")
        if stage == "preflight":
            report = mobile._run_preflights(launch, paths=launch_paths)
            if not report["all_passed"]:
                raise ValueError(f"Mobile preflight failed: {launch_paths['preflight_report']}")
            return [launch_paths["preflight_report"], *sorted((launch_paths["launch_dir"] / "preflight").rglob("*.json"))]
        if not _json(launch_paths["preflight_report"]).get("all_passed"):
            raise ValueError("Training requires completed mobile preflights")
        run_bounded_command(launch["training_command"], launch_paths["training_log"], _environment(plan, gpu=True),
                            timeout_seconds=values["stage_timeout_seconds"],
                            progress_seconds=values["progress_seconds"], emit_heartbeat=False)
        best = Path(paths["best_checkpoint"])
        metadata = _json(best / "training_metadata.json")
        if metadata.get("checkpoint_role") != "best_golden" or (metadata.get("best_golden_eval") or {}).get("metric") != SELECTOR:
            raise ValueError("Training did not publish the Golden32-selected best checkpoint")
        if metadata.get("training_config_sha256") != sha256(Path(paths["config"])):
            raise ValueError("Best checkpoint belongs to a different training config")
        if not (best / "adapter_model.safetensors").is_file():
            raise ValueError("Best checkpoint adapter weights are missing")
        return [path for path in sorted(best.iterdir()) if path.is_file()]
    if stage.startswith("best_"):
        from ir_training.pipeline.golden_deployment import _evaluation
        cohort = stage.removeprefix("best_")
        run_bounded_command(evaluation_command(plan, cohort), output / f"logs/{stage}_worker.log", _environment(plan, gpu=True),
                            timeout_seconds=values["stage_timeout_seconds"],
                            progress_seconds=values["progress_seconds"], emit_heartbeat=False)
        result, evidence = _evaluation(output / f"evaluations/{stage}", GOLDENS[cohort][1], artifact=Path(paths["best_checkpoint"]))
        if result.get("qat_applied") is not True:
            raise ValueError("Best checkpoint was not evaluated with retained-scale QAT active")
        return evidence
    if stage in {"merge", "export", "android_benchmark"}:
        env = _environment(plan)
        # CPU merge/export avoids a full dense base landing on GPU0. No Vulkan
        # or desktop LiteRT GPU dependency is needed for this export-only route.
        env["CUDA_VISIBLE_DEVICES"] = ""
        env.pop("A2UI_CUDA_VISIBLE_DEVICES", None)
        env.pop("A2UI_EXCLUDE_CUDA_DEVICES", None)
        run_bounded_command(_mobile_command(plan, stage), output / f"logs/{stage}_worker.log", env,
                            timeout_seconds=values["stage_timeout_seconds"],
                            progress_seconds=values["progress_seconds"], emit_heartbeat=False)
        if stage == "merge":
            return [path for path in sorted(Path(paths["merged"]).iterdir()) if path.is_file()]
        if stage == "export":
            # The existing orchestrator validates every required export gate
            # and live input/output identity before returning exit code zero.
            report = _json(Path(paths["export_report"]))
            if report.get("passed") is not True or report.get("executed") is not True:
                raise ValueError("Retained-scale exporter did not pass")
            runtime = output / "deployment_runtime.json"
            _write(runtime, {"model": paths["litertlm"], "model_sha256": sha256(Path(paths["litertlm"])),
                "required_mtp_enabled": False, "official_drafter_present_but_not_validated_for_this_target": True,
                "native_quality_evaluated": False, "speed_parity_measured_at_export": False,
                "subsequent_speed_report": paths["android_report"],
                "native_quality_command": native_quality_command(plan),
                "benchmark_command": benchmark_command(plan)})
            return [Path(paths["export_report"]), Path(paths["litertlm"]), runtime]
        from ir_training.eval.android_gpu_report import load_android_gpu_parity_report
        load_android_gpu_parity_report(paths["android_report"], expected_mtp=False,
            expected_official_artifact=values["official_litertlm"], expected_candidate_artifact=paths["litertlm"])
        return [Path(paths["android_report"])]
    raise ValueError(f"Unsupported pipeline stage: {stage}")


def _summary(state: dict) -> str:
    from ir_training.pipeline.result_tables import _metric, _table
    plan = state["plan"]
    rows = []
    for cohort, (_, count, _) in GOLDENS.items():
        result = state.get("results", {}).get(cohort, {})
        rows.append([cohort, "evaluated" if result else "not evaluated", f"{result.get('row_count', '?')}/{count}",
                     _metric(result, "generation_reward_v5_4_avg"), _metric(result, "schema_valid_strict_rate", percent=True),
                     _metric(result, SELECTOR) if cohort == "golden32" else "not used for selection"])
    return "\n".join([
        "# Official-layout mobile QAT results", "", f"Run status: {state['status']}", "",
        "Best checkpoint evaluated with retained-scale fake QAT (not native LiteRT quality scores).", "",
        _table(("Cohort", "Status", "Rows", "Reward v5.4 (0-100)", "Strict-valid %", "Unique-source selection reward"), rows), "",
        "Golden32: 32 occurrences / 31 unique sources. Golden35 and Bixby50 are final-only holdouts.",
        "Bixby50 has source responses but no reference IR: source-grounded reward/validity, not reference-match accuracy.", "",
        f"LiteRT-LM export: {'verified official layout' if 'export' in state['completed'] else 'not completed'}.",
        f"MTP: disabled; unused official drafter bytes preserved. Artifact: {plan['paths']['litertlm']}",
        f"Device speed parity: {'passed target-only benchmark (within 10%)' if 'android_benchmark' in state['completed'] else 'not measured'}.",
        "Native LiteRT golden/Bixby scores require the separate native_quality_command.json handoff (plan-only until --execute).",
        *(["", f"Failure: {state['error']}"] if state.get("error") else []), "",
    ])


def run_pipeline(options: OfficialMobileOptions, *, execute: bool = False,
                 command_runner: Callable = run_bounded_command) -> dict:
    plan = build_plan(options)
    if not execute:
        return plan
    from ir_training.pipeline.deployment_recovery import deployment_lock
    output = Path(plan["options"]["output_dir"])
    with deployment_lock(output):
        output.mkdir(parents=True, exist_ok=False)
        plan_path = output / "official_mobile_plan.json"
        _write(plan_path, plan)
        plan_digest = sha256(plan_path)
        state = {"workflow": WORKFLOW, "plan": plan, "status": "running", "completed": {}, "results": {}}
        manifest = output / "official_mobile_manifest.json"
        _write(output / "benchmark_command.json", {"command": benchmark_command(plan), "mtp": False,
               "note": "Run on a host with the app + instrumentation installed. Speed is unverified until this passes."})
        _write(output / "native_quality_command.json", {"command": native_quality_command(plan), "mtp": False,
               "execute": False, "note": "Separate Android quality gate. Review plan, set serial, then append --execute. No desktop Vulkan required."})
        try:
            for index, stage in enumerate(plan["stages"], 1):
                state["active_stage"] = stage
                _write(manifest, state)
                log(f"Official mobile stage {index}/{len(plan['stages'])}: {stage}")
                command = [sys.executable, str(training_root() / "scripts/run_official_mobile_pipeline.py"),
                           "--worker-stage", stage, "--plan-file", str(plan_path)]
                command_runner(command, output / f"logs/{stage}.log", _environment(plan),
                               timeout_seconds=options.stage_timeout_seconds,
                               progress_seconds=options.progress_seconds, emit_heartbeat=False)
                receipt = _json(output / f"stage_receipts/{stage}.json")
                if sha256(plan_path) != plan_digest or receipt.get("stage") != stage or receipt.get("plan_sha256") != plan_digest:
                    raise ValueError(f"Stage receipt is not bound to this run: {stage}")
                _verify_bindings(receipt.get("files") or {})
                state["completed"][stage] = receipt
                if stage.startswith("best_"):
                    state["results"][stage.removeprefix("best_")] = _json(output / f"evaluations/{stage}/evaluation_result.json")
                _write(manifest, state)
            state.update(status="complete", active_stage=None)
        except BaseException as exc:
            state.update(status="failed", error=str(exc))
            log("Pipeline stopped. Data/checkpoints/evidence retained; no automatic training restart or CPU inference fallback.")
            raise
        finally:
            _write(manifest, state)
            summary = _summary(state)
            (output / "results.md").write_text(summary, encoding="utf-8")
            print(summary, flush=True)
        return state


def worker(plan_path: Path, stage: str) -> None:
    _scripts()
    plan_digest = sha256(plan_path)
    plan = _json(plan_path)
    if plan.get("workflow") != WORKFLOW or stage not in plan.get("stages", []):
        raise ValueError("Worker requires an official mobile plan and a planned stage")
    output = Path(plan["options"]["output_dir"])
    if plan_path.resolve() != (output / "official_mobile_plan.json").resolve():
        raise ValueError("Worker plan must be inside its own run output")
    # A worker cannot jump over failed training/holdout stages. Recheck their
    # content before export, so the model that was scored is the model exported.
    manifest = _json(output / "official_mobile_manifest.json")
    if manifest.get("plan") != plan or manifest.get("active_stage") != stage or manifest.get("status") != "running":
        raise ValueError("Worker is not bound to the active pipeline stage")
    prior = plan["stages"][:plan["stages"].index(stage)]
    if list(manifest.get("completed", {})) != prior:
        raise ValueError("Pipeline stage prerequisites are incomplete or out of order")
    # Large source files are revalidated by the launch/export contracts. Keep
    # the inexpensive stage-receipt check on every transition, then rehash
    # checkpoint/test evidence immediately before merge and export.
    for name in prior:
        receipt = manifest["completed"][name]
        if receipt.get("plan_sha256") != plan_digest:
            raise ValueError(f"Prior stage belongs to a different plan: {name}")
        if name == "no_op_export":
            _verify_bindings(receipt.get("files") or {})
        if stage in {"merge", "export"} and (name == "training" or name.startswith("best_")):
            _verify_bindings(receipt.get("files") or {})
    files = run_stage(plan, stage)
    if sha256(plan_path) != plan_digest:
        raise ValueError("Pipeline plan changed while the worker ran")
    receipt = {"stage": stage, "plan_sha256": plan_digest, "files": _file_bindings(files)}
    _write(output / f"stage_receipts/{stage}.json", receipt)
