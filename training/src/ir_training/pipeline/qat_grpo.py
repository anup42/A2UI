"""Opt-in QAT LoRA GRPO; reuse preparation, QAT, merge, exporters and native QA.

No SFT launcher is replaced. Each expensive stage has a bounded subprocess and
hash-bound receipt. Native testing is mandatory for completion, or explicitly
deferred to the Android host (never reported as a successful device test).
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ir_training.common.bounded_command import run_bounded_command
from ir_training.common.config import load_yaml, resolve_path, training_root
from ir_training.common.progress import log
from ir_training.pipeline import official_mobile
from ir_training.pipeline.golden_training import (
    GOLDENS,
    GoldenTrainingOptions,
    _write,
    prepare_data,
    sha256,
)
from ir_training.pipeline.golden_training import (
    build_plan as preparation_plan,
)

WORKFLOW = "qat_lora_grpo_v1"
PLAN_NAME = "qat_grpo_plan.json"
MANIFEST_NAME = "qat_grpo_manifest.json"


@dataclass(frozen=True)
class QATGRPOOptions:
    family: str
    model_dir: Path
    sft_config: Path
    sft_checkpoint: Path
    output_dir: Path
    official_litertlm: Path
    exporter_python: Path
    input_dir: Path | None = None
    prepared_input_dir: Path | None = None
    source_safetensors: Path | None = None
    devices: str = "auto"
    max_steps: int = 200
    learning_rate: float = 1e-6
    num_generations: int = 4
    microbatch: int = 1
    effective_batch: int = 32
    max_seq_length: int = 4096
    max_input_tokens: int = 5120
    max_new_tokens: int = 2048
    golden_every_steps: int = 50
    seed: int = 42
    prepare_workers: int = 0
    tensorboard_root: str = "/tensorboard"
    stage_timeout_seconds: float = 172800
    generation_timeout_seconds: float = 7200
    progress_seconds: float = 30
    adb: str = "adb"
    serial: str | None = None
    defer_native_eval: bool = False


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file())


def build_plan(options: QATGRPOOptions) -> dict:
    """No downloads, model loading, GPU probing or output writes."""
    if options.family not in {"e2b", "270m"}:
        raise ValueError("--family must be e2b or 270m")
    if (options.input_dir is None) == (options.prepared_input_dir is None):
        raise ValueError("Choose exactly one input-dir or prepared-input-dir")
    if options.max_steps < 20:
        raise ValueError("GRPO requires at least one complete 20-update health window")
    if options.num_generations < 2 or options.effective_batch % options.num_generations:
        raise ValueError("Effective completion batch must be divisible by num-generations >= 2")
    if min(options.microbatch, options.effective_batch, options.golden_every_steps) < 1:
        raise ValueError("Batch and evaluation cadence must be positive")
    if options.golden_every_steps < 20:
        raise ValueError("Golden checkpoint selection must start after the 20-update GRPO health window")
    if not math.isfinite(options.learning_rate) or options.learning_rate <= 0:
        raise ValueError("Learning rate must be positive and finite")
    if any(not math.isfinite(value) or value <= 0 for value in (
        options.stage_timeout_seconds, options.generation_timeout_seconds, options.progress_seconds,
    )):
        raise ValueError("Timeouts and progress intervals must be positive and finite")
    context = 8192 if options.family == "e2b" else 32768
    if max(options.max_seq_length, options.max_input_tokens) + options.max_new_tokens + 1 > context:
        raise ValueError("Prompt + completion + GRPO stop sentinel exceeds model context")
    if options.max_new_tokens > 4096:
        raise ValueError("Native quality probe supports at most 4096 output tokens")
    values = asdict(options)
    for key, value in list(values.items()):
        if isinstance(value, Path):
            # Preserve venv interpreter symlinks.
            values[key] = os.path.abspath(str(value.expanduser())) if key == "exporter_python" else str(value.expanduser().resolve())
    output = Path(values["output_dir"])
    for key in ("model_dir", "sft_checkpoint", "input_dir", "prepared_input_dir"):
        if values[key] is not None:
            protected = Path(values[key])
            if output.is_relative_to(protected) or protected.is_relative_to(output):
                raise ValueError(f"Output must be disjoint from {key}")
    required = [Path(values[key]) for key in ("sft_config", "official_litertlm", "exporter_python")]
    required += [Path(values["sft_checkpoint"]) / name for name in (
        "adapter_config.json", "adapter_model.safetensors", "training_metadata.json",
    )]
    if options.family == "e2b":
        if options.source_safetensors is None:
            raise ValueError("E2B requires the pinned --source-safetensors")
        required += [Path(values["source_safetensors"])]
        required += [Path(values["model_dir"]) / name for name in (
            "mobile_training_seed_manifest.json", "mobile_qparams.json", "mobile_qparams.safetensors",
        )]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.is_relative_to(output):
            raise ValueError("Output cannot contain an input artifact")
    source_config = load_yaml(values["sft_config"])
    if source_config.get("training", {}).get("method") != "qat_lora_sft":
        raise ValueError("Start from a verified QAT LoRA SFT adapter, not a full/dense checkpoint")
    source_model = source_config.get("model", {})
    if not Path(source_model.get("model_source") or "").is_absolute():
        raise ValueError("The original resolved SFT config must bind an absolute local model_source")
    original_source = resolve_path(source_model.get("model_source") or source_model.get("model_id", ""), training_root())
    if original_source.resolve() != Path(values["model_dir"]):
        raise ValueError("model-dir must be the original local base bound by the SFT config")
    metadata = _json(Path(values["sft_checkpoint"]) / "training_metadata.json")
    if metadata.get("training_config_sha256") != sha256(Path(values["sft_config"])):
        raise ValueError("SFT config is not the original checkpoint-bound config")
    expected_scale = "retained_mobile" if options.family == "e2b" else "dynamic"
    if source_config.get("qat", {}).get("scale_mode", "dynamic") != expected_scale:
        raise ValueError("SFT quantization contract does not match the requested family")
    prep = preparation_plan(GoldenTrainingOptions(
        model_dir=options.model_dir, input_dir=options.input_dir, prepared_input_dir=options.prepared_input_dir,
        output_dir=options.output_dir, profile=options.family, max_seq_length=options.max_seq_length,
        max_input_tokens=options.max_input_tokens, max_new_tokens=options.max_new_tokens,
        seed=options.seed, prepare_workers=options.prepare_workers, progress_seconds=options.progress_seconds,
    ), preparation_only=True)
    export_dir = output / "export"
    paths = {
        "prepared": output / "prepared", "config": output / "configs/grpo_training.yaml",
        "best_checkpoint": output / "training/best_golden_checkpoint",
        "merged": output / "merged_best_hf", "export_dir": export_dir,
        "export_report": export_dir / ("gemma4_retained_scale_code_only_report.json" if options.family == "e2b" else "checkpoint_official_topology_report.json"),
        "litertlm": export_dir / f"{options.family}_qat_grpo.litertlm",
        "no_op_export_report": output / "pretraining_noop_export.json",
        "deployment_config": output / "configs/deployment.yaml",
        "gpu_profile": output / "configs/gpu_profile.json",
    }
    template270 = load_yaml(training_root() / "configs/pipelines/gemma3_270m_qat_litertlm.yaml")
    official_hash = (official_mobile.OFFICIAL_LITERTLM_SHA256 if options.family == "e2b"
                     else template270["pipeline"]["exact_topology"]["official_artifact_sha256"])
    return {
        "schema_version": 1, "workflow": WORKFLOW, "options": values,
        "preparation": prep, "paths": {key: str(value) for key, value in paths.items()},
        "stages": ["assets", "prepare", "configure", *(["no_op_export"] if options.family == "e2b" else []),
                   "training", *[f"best_{name}" for name in GOLDENS], "merge", "export", "native_quality"],
        "source_config_sha256": sha256(Path(values["sft_config"])),
        "source_adapter_files": official_mobile._file_bindings(required[3:6]),
        "selection": {"cohort": "golden32", "metric": official_mobile.SELECTOR,
                      "golden35_used": False, "bixby50_used": False},
        "official_artifact_sha256": official_hash,
        "format": "official retained W2/W4/W8 A8" if options.family == "e2b" else "official Gemma 3 270M INT8 topology",
        "mtp": {"training": False, "inference": False, "official_section_preserved": options.family == "e2b"},
        "native_quality_status": "required; explicitly deferred" if options.defer_native_eval else "required after export",
    }


def training_config(plan: dict, profile: dict, prepared: dict) -> dict:
    values, paths = plan["options"], plan["paths"]
    if sha256(Path(values["sft_config"])) != plan["source_config_sha256"]:
        raise ValueError("Source SFT config changed since planning")
    config = copy.deepcopy(load_yaml(values["sft_config"]))
    from ir_training.train.gpu_profile import apply_gpu_profile
    if profile["dtype"] != "bfloat16":
        raise ValueError("QAT GRPO requires native BF16 GPUs")
    if profile["effective_batch_size"] % values["num_generations"]:
        raise ValueError("Effective completion batch must divide into complete GRPO groups")
    apply_gpu_profile(config, profile)
    output = Path(values["output_dir"])
    config["run"].update(id=output.name, output_dir=str(output / "training"), dataset_dir=paths["prepared"],
                         prepared_manifest_required=True, dataset_format="a2ui_express_v1", purpose=WORKFLOW)
    training = config["training"]
    for key in ("resume_from_checkpoint", "resume_policy", "resume_horizon", "deepspeed", "fsdp"):
        training.pop(key, None)
    training.update(method="qat_lora_grpo", trainer_backend="hf", refuse_resume=True,
                    max_steps=values["max_steps"], learning_rate=values["learning_rate"], seed=values["seed"],
                    max_seq_length=values["max_seq_length"], logging_steps=1,
                    eval_steps=values["golden_every_steps"], save_steps=values["golden_every_steps"],
                    save_total_limit=2, report_to="tensorboard", logging_dir=str(output / "tensorboard/training"))
    # Validation also samples complete GRPO groups across all ranks. This is
    # independent of the unchanged effective training-completion batch.
    training["per_device_eval_batch_size"] = values["num_generations"] // math.gcd(
        values["num_generations"], profile["world_size"]
    )
    config["grpo"] = {
        "family": values["family"], "sft_checkpoint": values["sft_checkpoint"],
        "sft_training_config": values["sft_config"], "num_generations": values["num_generations"],
        "max_prompt_length": values["max_seq_length"], "max_completion_length": values["max_new_tokens"],
        "beta": 0.0, "loss_type": "dr_grpo", "scale_rewards": "batch",
        "temperature": 0.8, "top_p": 1.0, "top_k": 0, "num_iterations": 1,
        "reward_policy": "qat_source_fidelity_v1", "health": {"window_steps": 20},
    }
    config["model"].update(tokenizer_source=values["model_dir"],
                           chat_template_kwargs=prepared["tokenizer"].get("chat_template_kwargs") or {},
                           max_output_tokens=values["max_new_tokens"])
    config["golden_eval"].update(enabled=True, dataset_dir=paths["prepared"], split="golden32",
        split_path=str(Path(paths["prepared"]) / "golden32.jsonl"), max_rows=32, required_rows=32,
        require_exact_rows=True, require_unique_rows=True, max_input_tokens=values["max_input_tokens"],
        max_new_tokens=values["max_new_tokens"], trigger="evaluate", interval=1, evaluate_at_end=True,
        metric_version="v5_4", metric_for_best_model=official_mobile.SELECTOR, greater_is_better=True,
        save_best_checkpoint=True, best_checkpoint_dir=paths["best_checkpoint"],
        output_dir=str(output / "evaluations/periodic_golden32"), stop_strings=["</a2ui>"],
        weights_config=str(training_root().parent / "dataset/configs/run.yaml"))
    config["final_evaluation_datasets"] = copy.deepcopy(prepared["final_evaluation_datasets"])
    if values["family"] == "e2b":
        config.setdefault("preflight", {})["numeric_policy"] = "retained_mobile_safety_v1"
    else:
        preflight = config.setdefault("preflight", {})
        for key, value in {
            "rows": 4, "logit_probe_tokens": 32, "max_initial_completion_loss": 15.0,
            "require_initial_loss_gate": True, "require_zero_adapter_parity": True,
            "require_greedy_determinism": True, "greedy_probe_rows": 1,
            "greedy_probe_new_tokens": 16, "min_greedy_tokens": 8,
        }.items():
            preflight.setdefault(key, value)
    config.pop("qat_mtp", None)
    return config


def export_command(plan: dict) -> list[str]:
    values, paths = plan["options"], plan["paths"]
    if values["family"] == "e2b":
        seed = Path(values["model_dir"])
        return [values["exporter_python"], str(training_root() / "scripts/build_gemma4_retained_scale_litertlm.py"),
            "--official-litertlm", values["official_litertlm"], "--official-artifact-sha256", plan["official_artifact_sha256"],
            "--checkpoint", paths["merged"], "--adapter-checkpoint", paths["best_checkpoint"],
            "--training-config", paths["config"], "--mobile-training-seed-manifest", str(seed / "mobile_training_seed_manifest.json"),
            "--mobile-qparams-contract", str(seed / "mobile_qparams.json"), "--zero-adapter-checkpoint", str(seed),
            "--output-dir", paths["export_dir"], "--output-litertlm", paths["litertlm"], "--report", paths["export_report"],
            "--qat-compatible-weights", "--execute"]
    from ir_training.pipeline.gemma270m_litertlm import build_pipeline_plan
    deployment = build_pipeline_plan(load_yaml(paths["deployment_config"]), best_checkpoint_override=paths["best_checkpoint"])
    command = list(deployment["exact_topology"]["command"])
    command[0] = values["exporter_python"]
    return command


def native_command(plan: dict, *, execute: bool = False) -> list[str]:
    values = plan["options"]
    output = Path(values["output_dir"])
    command = [sys.executable, str(training_root() / "scripts/evaluate_official_mobile_native.py"),
               "--run-dir", str(output), "--output-dir", str(output.parent / f"{output.name}_native_quality"),
               "--adb", values["adb"], "--serial", values["serial"] or "<ANDROID_SERIAL>"]
    if execute:
        command.append("--execute")
    return command


def _environment(plan: dict, *, gpu: bool = False) -> dict[str, str]:
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    if gpu:
        from ir_training.train.gpu_profile import (
            training_environment,
            verify_gpu_profile,
        )
        profile = _json(Path(plan["paths"]["gpu_profile"]))
        verify_gpu_profile(profile)
        env.update(training_environment(profile, tensorboard_root=plan["options"]["tensorboard_root"]))
    return env


def _run(plan: dict, command: list[str], stage: str, *, gpu: bool = False, cpu: bool = False) -> None:
    env = _environment(plan, gpu=gpu)
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env.pop("A2UI_CUDA_VISIBLE_DEVICES", None)
        env.pop("A2UI_EXCLUDE_CUDA_DEVICES", None)
    values = plan["options"]
    run_bounded_command(command, Path(values["output_dir"]) / f"logs/{stage}_worker.log", env,
                        timeout_seconds=values["stage_timeout_seconds"], progress_seconds=values["progress_seconds"],
                        emit_heartbeat=False)


def run_stage(plan: dict, stage: str) -> list[Path]:
    values, paths = plan["options"], plan["paths"]
    output = Path(values["output_dir"])
    official_mobile._verify_bindings(plan["source_adapter_files"])
    if sha256(Path(values["sft_config"])) != plan["source_config_sha256"]:
        raise ValueError("Input SFT config changed")
    if stage == "assets":
        if sha256(Path(values["official_litertlm"])) != plan["official_artifact_sha256"]:
            raise ValueError("Official package differs from the pinned family release")
        if values["family"] == "e2b":
            return official_mobile._assets(plan)
        _write(output / "assets_verified.json", {"official_sha256": plan["official_artifact_sha256"], "family": "270m"})
        return [output / "assets_verified.json"]
    if stage == "prepare":
        prepare_data(plan["preparation"])
        return [output / "data_audit.json", *_files(Path(paths["prepared"]))]
    if stage == "configure":
        from ir_training.train.gpu_profile import build_gpu_profile, detect_cuda_devices
        official_mobile._scripts()
        from prepare_review_training import verify_prepared
        prepared_dir = Path(paths["prepared"])
        prepared = verify_prepared(prepared_dir, prepared_dir / "golden32.jsonl",
            golden35=prepared_dir / "golden35.jsonl", bixby50=prepared_dir / "bixby50.jsonl",
            max_sequence=values["max_seq_length"], max_prompt=values["max_input_tokens"])
        profile = build_gpu_profile(detect_cuda_devices(), model=values["family"], devices=values["devices"],
                                    microbatch=values["microbatch"], effective_batch=values["effective_batch"])
        config = training_config(plan, profile, prepared)
        from ir_training.train.qat_grpo_contract import verify_sft_adapter_lineage
        lineage = verify_sft_adapter_lineage(config)
        if lineage.get("verified") is not True:
            raise ValueError(f"Starting QAT SFT adapter provenance failed: {lineage}")
        official_mobile._yaml(Path(paths["config"]), config)
        _write(Path(paths["gpu_profile"]), profile)
        prepared.update(training_config_sha256=sha256(Path(paths["config"])),
                        model_files={p.name: sha256(p) for p in Path(values["model_dir"]).iterdir()
                                     if p.is_file() and p.suffix in {".json", ".safetensors", ".model", ".jinja"}})
        _write(Path(paths["config"]).parent / "preparation_report.json", prepared)
        if values["family"] == "270m":
            deployment = copy.deepcopy(load_yaml(training_root() / "configs/pipelines/gemma3_270m_qat_litertlm.yaml"))
            pipeline = deployment["pipeline"]
            pipeline.update(training_config=paths["config"], output_dir=str(output))
            pipeline["source"].update(merged_model_dir=paths["merged"], official_litertlm=values["official_litertlm"])
            pipeline["export"].update(artifact=paths["litertlm"], output_dir=str(output / "unused_public_export"))
            pipeline["exact_topology"].update(output_dir=paths["export_dir"])
            official_mobile._yaml(Path(paths["deployment_config"]), deployment)
        # Check the reviewed TRL source/API before the costly E2B no-op model
        # gate. This command imports dependencies but never loads the model.
        _run(plan, [sys.executable, str(training_root() / "scripts/train_qat_grpo.py"),
                    "--config", paths["config"], "--dependency-preflight-only"],
             "dependency_preflight", gpu=True)
        return _files(output / "configs")
    if stage == "no_op_export":
        _run(plan, official_mobile.no_op_export_command(plan), stage, cpu=True)
        official_mobile._require_no_op_export(plan)
        return [Path(paths["no_op_export_report"])]
    if stage == "training":
        if values["family"] == "e2b":
            official_mobile._require_no_op_export(plan)
        profile = _json(Path(paths["gpu_profile"]))
        command = [sys.executable]
        if profile["world_size"] > 1:
            command += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(profile["world_size"])]
        command += [str(training_root() / "scripts/train_qat_grpo.py"), "--config", paths["config"]]
        _run(plan, command, stage, gpu=True)
        metadata = _json(Path(paths["best_checkpoint"]) / "training_metadata.json")
        if (metadata.get("checkpoint_role") != "best_golden"
                or metadata.get("training_config_sha256") != sha256(Path(paths["config"]))
                or (metadata.get("best_golden_eval") or {}).get("metric") != official_mobile.SELECTOR
                or (metadata.get("training") or {}).get("method") != "qat_lora_grpo"):
            raise ValueError("GRPO did not publish a bound Golden32-selected QAT adapter")
        return _files(Path(paths["best_checkpoint"]))
    if stage.startswith("best_"):
        from ir_training.pipeline.golden_deployment import _evaluation
        cohort = stage.removeprefix("best_")
        _run(plan, official_mobile.evaluation_command(plan, cohort), stage, gpu=True)
        result, evidence = _evaluation(output / f"evaluations/{stage}", GOLDENS[cohort][1], artifact=Path(paths["best_checkpoint"]))
        if result.get("qat_applied") is not True:
            raise ValueError("Selected checkpoint was evaluated without QAT")
        return evidence
    if stage == "merge":
        from ir_training.export.merge_lora import merge_lora_adapter
        config = load_yaml(paths["config"])
        merge_lora_adapter(base_model_id=config["model"]["model_id"], adapter_dir=paths["best_checkpoint"],
                           base_model_source=values["model_dir"],
                           mobile_training_seed_manifest=config["model"].get("mobile_training_seed_manifest"),
                           output_dir=paths["merged"], model_loader="auto_causal_lm", dtype="bfloat16",
                           trust_remote_code=False, training_config_path=paths["config"])
        return _files(Path(paths["merged"]))
    if stage == "export":
        _run(plan, export_command(plan), stage, cpu=True)
        verify_export_report(plan)
        return [Path(paths["export_report"]), Path(paths["litertlm"])]
    if stage == "native_quality":
        _run(plan, native_command(plan, execute=True), stage)
        native_dir = output.parent / f"{output.name}_native_quality"
        report = _json(native_dir / "manifest.json")
        if report.get("status") != "complete":
            raise ValueError("Native quality evaluation did not complete")
        return _files(native_dir)
    raise ValueError(f"Unknown QAT GRPO stage: {stage}")


def verify_export_report(plan: dict) -> dict:
    paths = plan["paths"]
    report = _json(Path(paths["export_report"]))
    digest = sha256(Path(paths["litertlm"]))
    if plan["options"]["family"] == "e2b":
        valid = (report.get("passed") is True and report.get("executed") is True
                 and report.get("output_sha256") == digest
                 and report.get("mode") == "retained_scale_qat_compatible_weights_v1")
    else:
        valid = (report.get("final_artifact_gate_pass") is True
                 and bool(report.get("gates")) and all(v is True for v in report["gates"].values())
                 and report.get("package_boundary", {}).get("output_sha256") == digest)
    if not valid:
        raise ValueError("Canonical exporter did not certify the actual package bytes")
    return report


def run_pipeline(options: QATGRPOOptions, *, execute: bool = False, command_runner=run_bounded_command) -> dict:
    plan = build_plan(options)
    if not execute:
        return {"status": "plan_only", "plan": plan, "native_quality_command": native_command(plan)}
    if not options.defer_native_eval and (not options.serial or options.serial.startswith("<")):
        raise ValueError("Execution requires Android --serial, or explicitly --defer-native-eval")
    output = Path(plan["options"]["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    plan_path = output / PLAN_NAME
    _write(plan_path, plan)
    digest = sha256(plan_path)
    state: dict[str, Any] = {"workflow": WORKFLOW, "plan": plan, "status": "running", "completed": {}}
    manifest = output / MANIFEST_NAME
    _write(output / "native_quality_command.json", {"command": native_command(plan, execute=True), "native_test_passed": False})
    try:
        for stage in plan["stages"]:
            if stage == "native_quality" and options.defer_native_eval:
                state.update(status="awaiting_native_evaluation", active_stage=None)
                break
            state["active_stage"] = stage
            _write(manifest, state)
            log(f"QAT GRPO stage: {stage}")
            command = [sys.executable, str(training_root() / "scripts/run_qat_grpo_pipeline.py"),
                       "--worker-stage", stage, "--plan-file", str(plan_path)]
            env = _environment(plan)
            if stage == "merge":
                env["CUDA_VISIBLE_DEVICES"] = ""
                env.pop("A2UI_CUDA_VISIBLE_DEVICES", None)
                env.pop("A2UI_EXCLUDE_CUDA_DEVICES", None)
            command_runner(command, output / f"logs/{stage}.log", env,
                           timeout_seconds=options.stage_timeout_seconds, progress_seconds=options.progress_seconds,
                           emit_heartbeat=False)
            receipt = _json(output / f"stage_receipts/{stage}.json")
            if sha256(plan_path) != digest or receipt.get("plan_sha256") != digest or receipt.get("stage") != stage:
                raise ValueError("Worker receipt is not bound to the immutable GRPO plan")
            if not isinstance(receipt.get("files"), dict) or not receipt["files"]:
                raise ValueError("Worker stage supplied no bound evidence")
            official_mobile._verify_bindings(receipt.get("files") or {})
            state["completed"][stage] = receipt
            _write(manifest, state)
        else:
            state.update(status="complete", active_stage=None)
    except BaseException as exc:
        state.update(status="failed", error=str(exc))
        raise
    finally:
        _write(manifest, state)
        _write(output / "results.json", {"status": state["status"], "artifact": plan["paths"]["litertlm"],
               "export_passed": "export" in state["completed"], "native_test_passed": "native_quality" in state["completed"],
               "selection": plan["selection"], "mtp": plan["mtp"]})
    return state


def worker(plan_path: Path, stage: str) -> None:
    plan = _json(plan_path)
    output = Path(plan["options"]["output_dir"])
    if plan.get("workflow") != WORKFLOW or plan_path.resolve() != (output / PLAN_NAME).resolve() or stage not in plan["stages"]:
        raise ValueError("Worker requires its own QAT GRPO plan and a planned stage")
    state = _json(output / MANIFEST_NAME)
    prior = plan["stages"][:plan["stages"].index(stage)]
    if (state.get("plan") != plan or state.get("status") != "running"
            or state.get("active_stage") != stage or list(state.get("completed", {})) != prior):
        raise ValueError("Worker prerequisites are incomplete or out of order")
    digest = sha256(plan_path)
    for name in prior:
        receipt = state["completed"][name]
        if receipt.get("plan_sha256") != digest:
            raise ValueError("Prior stage belongs to another plan")
        # Verify immutable configuration on every transition; selected model,
        # scores and package again before either export or native evaluation.
        if name == "configure" or stage in {"merge", "export", "native_quality"}:
            official_mobile._verify_bindings(receipt["files"])
    files = run_stage(plan, stage)
    if not files:
        raise ValueError("Worker stage supplied no evidence files")
    if sha256(plan_path) != digest:
        raise ValueError("GRPO plan changed while worker ran")
    _write(output / f"stage_receipts/{stage}.json", {
        "stage": stage, "plan_sha256": digest, "files": official_mobile._file_bindings(files),
    })
