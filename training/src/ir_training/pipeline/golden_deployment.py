"""Current dual-Golden training, optional screening, and real GPU deployment tests.

No automatic retries or CPU-inference fallback.  Completion requires all eight
LiteRT evaluations, not just successful subprocess exits or export filenames.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ir_training.common.bounded_command import run_bounded_command
from ir_training.common.config import load_yaml, repo_root
from ir_training.common.progress import Progress, log
from ir_training.eval.tensorboard_logging import (
    _summary_writer_factory,
    flatten_scalar_metrics,
)
from ir_training.pipeline.deployment_export import (
    build_deployment_export_plan,
    validate_deployment_export_output,
)
from ir_training.pipeline.experiments import (
    ExperimentOptions,
    build_experiment_plan,
    run_experiments,
)
from ir_training.pipeline.golden_training import (
    GoldenTrainingOptions,
    _write,
    build_plan,
    evaluation_command,
    run_pipeline,
    sha256,
)


@dataclass(frozen=True)
class GoldenDeploymentOptions:
    base: GoldenTrainingOptions
    exporter_python: Path
    runtime_python: Path
    tune: bool = False
    trial_steps: int = 1000
    trials_file: Path | None = None
    include_augmentation: bool = False
    allow_experimental_formats: bool = False
    cache_length: int = 8192
    stage_timeout_seconds: float = 172800
    generation_timeout_seconds: float = 7200
    case_timeout_seconds: float = 600
    load_timeout_seconds: float = 1800


def _base(options: GoldenDeploymentOptions) -> GoldenTrainingOptions:
    output = options.base.output_dir.expanduser().resolve()
    cache = options.base.preparation_cache_dir or output.parent / ".golden-preparation-cache"
    return replace(options.base, preparation_cache_dir=cache,
                   token_cache_dir=options.base.token_cache_dir or cache / "tokens")


def build_deployment_plan(options: GoldenDeploymentOptions) -> dict[str, Any]:
    base = _base(options)
    outer = build_plan(base)  # Validate root against inputs, budgets and cache paths too.
    output = Path(outer["options"]["output_dir"])
    for name in ("model_dir", "input_dir", "source_run_dir"):
        if outer["options"][name] and output.is_relative_to(Path(outer["options"][name])):
            raise ValueError("Deployment output must be outside model/source inputs")
    if base.qat:
        raise ValueError("This full deployment workflow exports dense SFT checkpoints; retained-scale/QAT export remains in its separate pipeline")
    if not base.evaluate_golden35:
        raise ValueError("Full deployment requires Golden35 after final training; tuning defers it internally")
    if not options.tune and (options.trials_file is not None or options.include_augmentation):
        raise ValueError("--trials-file and --include-augmentation require --tune")
    for value in (options.stage_timeout_seconds, options.generation_timeout_seconds,
                  options.case_timeout_seconds, options.load_timeout_seconds):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Timeouts must be positive and finite")
    runtime = options.runtime_python.expanduser()
    if not runtime.is_absolute() or not runtime.is_file():
        raise ValueError("--runtime-python must be an existing absolute Python executable")
    run_id = (re.sub(r"[^A-Za-z0-9_-]", "_", output.name) or "deployment") + "_" + hashlib.sha256(str(output).encode()).hexdigest()[:8]
    training_output = output / (run_id + "_training")
    training = build_plan(replace(base, output_dir=training_output))
    tuning = None
    if options.tune:
        tuning = build_experiment_plan(ExperimentOptions(
            base=replace(base, output_dir=output / (run_id + "_tuning"), steps=None),
            trial_steps=options.trial_steps, trials_file=options.trials_file,
            include_augmentation=options.include_augmentation, evaluate_selected_holdout=False))
    export = build_deployment_export_plan(
        profile=base.profile, training_config_path=training_output / "fit/training_config.yaml",
        checkpoint_dir=training_output / "fit/training/best_golden_checkpoint",
        output_dir=output / "deployment", model_dir=Path(outer["options"]["model_dir"]),
        training_python=sys.executable, exporter_python=options.exporter_python,
        cache_length=options.cache_length, max_input_tokens=base.max_input_tokens,
        max_new_tokens=base.max_new_tokens)
    return {
        "schema_version": 1, "workflow": "dual_golden_gpu_deployment_v1", "output_dir": str(output),
        "run_id": run_id, "training": training, "tuning": tuning, "export": export,
        # Do not resolve a venv Python symlink: its invocation path selects the venv.
        "runtime_python": str(runtime.absolute()),
        "runtime_probe_command": [str(runtime.absolute()), "-u", str(repo_root() / "training/scripts/run_litertlm_gpu.py"),
                                  "--preflight", "--report", str(output / "runtime_preflight.json")],
        "allow_experimental_formats": options.allow_experimental_formats,
        "tensorboard_dir": str(Path(base.tensorboard_root).expanduser().resolve() / "deployments" / run_id),
        "timeouts": {name: getattr(options, name) for name in ("stage_timeout_seconds", "generation_timeout_seconds", "case_timeout_seconds", "load_timeout_seconds")},
        "stages": ["host_preflight", "exporter_preflight", "runtime_preflight", *(["tuning", "lock_hyperparameters"] if tuning else []),
                   "full_training_and_checkpoint_evaluation", "merge", "merged_golden32", "merged_golden35",
                   *[stage for variant in export["variants"] for stage in (f"export_{variant}", f"{variant}_golden32", f"{variant}_golden35")], "scorecard"],
        "gpu_policy": {"training": "all selected CUDA GPUs; conservative H100 profile with backward preflight",
                       "hf_final_evaluation": "independent case shards over all selected GPUs",
                       "litert_evaluation": "one verified NVIDIA GPU; pinned native API has no device selector",
                       "conversion": "CPU in isolated exporter environment", "cpu_inference_fallback": False},
        "selection_policy": "Golden32 (31 unique sources) only. Tuning locks settings, then starts a fresh full run. Golden35 never selects a trial/checkpoint/precision.",
        "experimental": "W16 and W4 require explicit acknowledgement and real host export/GPU evaluation; unsupported variants fail, never skip.",
        "runtime_preflight_scope": "Dependency/device screening only; actual model kernels are verified after export, not guaranteed before training.",
        "official_retained_scale_export": False, "mtp_exported": False,
    }


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")  # noqa: TRY004 - malformed file, not a caller argument type
    return value


def _bindings(paths: list[Path]) -> dict[str, str]:
    with Progress("Bind completed stage evidence", unit="stage"):
        values = {}
        for path in paths:
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing/empty stage evidence: {path}")
            values[str(path.resolve())] = sha256(path)
    return values


def _checkpoint_step(checkpoint: Path) -> int:
    metadata = _json(checkpoint / "training_metadata.json")
    step = metadata.get("checkpoint_step")
    if metadata.get("checkpoint_role") == "best_golden" or checkpoint.name == "best_golden_checkpoint":
        step = (metadata.get("best_golden_eval") or {}).get("step", step)
    if type(step) is not int or step <= 0:
        raise ValueError("Selected checkpoint has no positive bound optimizer step")
    return step


def _evaluation(path: Path, count: int, *, artifact: Path, litert: bool = False) -> tuple[dict, list[Path]]:
    result = _json(path / "evaluation_result.json")
    if result.get("row_count") != count:
        raise ValueError(f"Incomplete {count}-row evaluation: {path}")
    key = "model" if litert else "checkpoint"
    if Path(result.get(key, "")).resolve() != artifact.resolve():
        raise ValueError(f"Evaluation used a different {key}: {path}")
    aggregate = _json(path / "aggregate_metrics.json")
    if aggregate != result.get("aggregate"):
        raise ValueError(f"Evaluation aggregate is not bound to its result: {path}")
    metric = "unique_source_generation_reward_v5_4_avg" if count == 32 else "generation_reward_v5_4_avg"
    score = aggregate.get(metric)
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError(f"Missing finite {metric}: {path}")
    evidence = [path / name for name in ("evaluation_result.json", "aggregate_metrics.json", "predictions.jsonl", "scored_predictions.jsonl")]
    for name in ("predictions.jsonl", "scored_predictions.jsonl"):
        with (path / name).open(encoding="utf-8") as stream:
            if sum(1 for line in stream if line.strip()) != count:
                raise ValueError(f"Incomplete prediction/scoring evidence: {path / name}")
    record = Path(result.get("tensorboard_record", ""))
    if not record.is_file():
        raise ValueError(f"TensorBoard evaluation record missing: {path}")
    evidence.append(record)
    if litert:
        manifest_path = Path(result.get("external_runner_manifest", ""))
        manifest = _json(manifest_path)
        if manifest.get("row_count") != count or manifest.get("model_sha256") != sha256(artifact):
            raise ValueError("LiteRT runner manifest has incomplete coverage")
        gpu = manifest.get("gpu_execution") or {}
        if (gpu.get("requested_backend") != "gpu" or gpu.get("engine_backend") != "gpu"
                or gpu.get("allocation_observed") is not True or gpu.get("tokenizer_parity_passed") is not True
                or not gpu.get("gpu_uuids")):
            raise ValueError("LiteRT evaluation lacks verified GPU allocation/tokenizer evidence")
        evidence.extend([manifest_path, Path(manifest["runner_outputs_path"]), Path(manifest["requests_path"]),
                         Path(manifest["runner_log_path"])])
    return result, evidence


def run_deployment(options: GoldenDeploymentOptions, *, execute: bool = False,
                   command_runner: Callable = run_bounded_command,
                   pipeline_runner: Callable = run_pipeline, experiment_runner: Callable = run_experiments,
                   writer_factory: Callable | None = None, gpu_probe: Callable | None = None) -> dict:
    plan = build_deployment_plan(options)
    if not execute:
        return {**plan, "status": "plan_only", "training_executed": False}
    if not options.allow_experimental_formats:
        raise ValueError("All four formats include experimental W16/W4. Read the deployment runbook, then explicitly pass --allow-experimental-formats")
    output = Path(plan["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    record = output / "deployment_manifest.json"
    state: dict = {"plan": plan, "status": "running", "completed": {}, "active_stage": None,
                   "started_at": datetime.now(timezone.utc).isoformat(), "results": {}}
    _write(record, state)
    environment = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
                   "A2UI_TENSORBOARD_ROOT": options.base.tensorboard_root}
    writer = None

    def command(argv, logfile, env):
        argv = list(argv)
        if any(Path(part).name == "evaluate_checkpoint_on_golden.py" for part in argv):
            argv.extend(["--require-gpu", "--generation-timeout-seconds", str(options.generation_timeout_seconds)])
        command_runner(argv, logfile, env, timeout_seconds=options.stage_timeout_seconds,
                       progress_seconds=options.base.progress_seconds)

    def stage(name, work):
        state.update(active_stage=name, status="running")
        _write(record, state)
        start = time.monotonic()
        log(f"Full deployment stage {len(state['completed']) + 1}/{len(plan['stages'])}: {name}")
        with Progress(name, unit="stage", interval=options.base.progress_seconds):
            paths = work()
        elapsed = time.monotonic() - start
        state["completed"][name] = {"files": _bindings(paths), "elapsed_seconds": elapsed}
        state["active_stage"] = None
        _write(record, state)
        writer.add_scalar(f"stages/{name}/elapsed_seconds", elapsed, 0)
        writer.flush()

    def log_result(label, cohort, result, step):
        state["results"][f"{label}_{cohort}"] = result
        for key, value in flatten_scalar_metrics(result["aggregate"]).items():
            writer.add_scalar(f"evaluation/{label}/{cohort}/{key}", value, step)
        writer.flush()
        metric = "unique_source_generation_reward_v5_4_avg" if cohort == "golden32" else "generation_reward_v5_4_avg"
        log(f"{label} {cohort}: {metric}={result['aggregate'][metric]:.6f}; rows={result['row_count']}")

    try:
        factory = writer_factory or _summary_writer_factory()
        writer = factory(log_dir=plan["tensorboard_dir"])
        writer.add_text("deployment/plan", json.dumps(plan, indent=2), 0)
        writer.flush()  # Fail on an unwritable /tensorboard before training.

        def host_preflight():
            from ir_training.train.gpu_profile import (
                build_gpu_profile,
                detect_cuda_devices,
            )
            inventory = (gpu_probe or detect_cuda_devices)()
            profile = build_gpu_profile(inventory, model=options.base.profile, devices=options.base.devices,
                                        microbatch=options.base.microbatch, effective_batch=options.base.effective_batch,
                                        dataloader_workers=options.base.dataloader_workers)
            _write(output / "gpu_preflight.json", profile)
            allowed = [item.get("uuid") for item in profile["selected_devices"]]
            if any(not value for value in allowed):
                raise ValueError("LiteRT GPU allocation checks require the selected NVIDIA GPU UUIDs")
            environment["A2UI_LITERT_ALLOWED_GPU_UUIDS"] = json.dumps(allowed)
            log(f"Training/HF evaluation: {profile['world_size']} GPUs; microbatch={profile['microbatch']}; effective batch={profile['effective_batch_size']}")
            return [output / "gpu_preflight.json"]

        stage("host_preflight", host_preflight)
        export = plan["export"]
        export_env = {**environment, **export["environment"]}
        def probe_exporter():
            command(export["probe_command"], output / "logs/exporter_preflight.log", export_env)
            path = output / "deployment/exporter_preflight.json"
            if _json(path).get("status") != "passed":
                raise ValueError("Exporter preflight did not pass")
            return [path]
        stage("exporter_preflight", probe_exporter)

        def probe_runtime():
            command(plan["runtime_probe_command"], output / "logs/runtime_preflight.log", environment)
            path = output / "runtime_preflight.json"
            # The subprocess exits nonzero for unsupported package/API/host.
            if _json(path).get("status") != "prerequisites_passed":
                raise ValueError("LiteRT runtime prerequisite checks did not pass")
            return [path]
        stage("runtime_preflight", probe_runtime)
        base = _base(options)
        training_options = replace(base, output_dir=Path(plan["training"]["options"]["output_dir"]))
        if options.tune:
            tuned: dict = {}
            def tune():
                result = experiment_runner(ExperimentOptions(
                    base=replace(base, output_dir=Path(plan["tuning"]["output_dir"]), steps=None),
                    trial_steps=options.trial_steps, trials_file=options.trials_file,
                    include_augmentation=options.include_augmentation, evaluate_selected_holdout=False), execute=True,
                    pipeline_runner=lambda child, **kwargs: pipeline_runner(child, command_runner=command, **kwargs),
                    command_runner=command)
                if result.get("status") != "complete" or result.get("selected_golden35") is not None:
                    raise ValueError("Screening must complete without evaluating Golden35")
                selected = [trial for trial in result["plan"]["trials"] if trial["name"] == result["selected_trial"]]
                if len(selected) != 1:
                    raise ValueError("Screening did not lock exactly one winner")
                tuned.update(selected[0]["parameters"])
                directory = Path(plan["tuning"]["output_dir"])
                return [directory / "experiments_manifest.json", directory / "selection_locked.json", directory / "comparison.json"]
            stage("tuning", tune)
            training_options = replace(training_options, **tuned)
            def lock_parameters():
                _write(output / "locked_hyperparameters.json", {"parameters": tuned, "golden35_seen": False,
                       "training_plan": build_plan(training_options), "fresh_full_training": True})
                return [output / "locked_hyperparameters.json"]
            stage("lock_hyperparameters", lock_parameters)
        training_plan = build_plan(training_options)
        state["actual_training_plan"] = training_plan
        training_output = training_options.output_dir
        def train():
            result = pipeline_runner(training_options, execute=True, command_runner=command)
            if result.get("status") != "complete":
                raise ValueError("Full training did not complete")
            evidence = [training_output / "pipeline_manifest.json", training_output / "evaluation_scorecard.json"]
            for role in ("best", "final"):
                checkpoint = training_output / "fit/training" / ("best_golden_checkpoint" if role == "best" else
                              ("final_adapter" if base.profile == "e2b" else "final_model"))
                for cohort, count in (("golden32", 32), ("golden35", 35)):
                    entry = result["completed"][f"{role}_{cohort}"]
                    results = [Path(p) for p in entry["files"] if Path(p).name == "evaluation_result.json"]
                    if len(results) != 1 or sha256(results[0]) != entry["files"][str(results[0])]:
                        raise ValueError("Training evaluation evidence is not hash-bound")
                    evaluated, files = _evaluation(results[0].parent, count, artifact=checkpoint)
                    log_result(f"checkpoint_{role}", cohort, evaluated, _checkpoint_step(checkpoint))
                    evidence.extend(files)
            return evidence
        stage("full_training_and_checkpoint_evaluation", train)
        checkpoint = Path(export["source_checkpoint"])
        step = _checkpoint_step(checkpoint)
        runtime = load_yaml(training_output / "fit/training_config.yaml")["runtime"]
        eval_env = {**environment, "CUDA_VISIBLE_DEVICES": runtime["cuda_visible_devices"], "A2UI_SKIP_CUDA_DEVICE_NORMALIZE": "1"}
        for name in ("A2UI_CUDA_VISIBLE_DEVICES", "A2UI_EXCLUDE_CUDA_DEVICES"):
            eval_env.pop(name, None)
        gpu_profile = _json(output / "gpu_preflight.json")
        allowed_uuids = [item.get("uuid") for item in gpu_profile["selected_devices"]]
        if any(not value for value in allowed_uuids):
            raise ValueError("LiteRT GPU allocation checks require the selected NVIDIA GPU UUIDs")
        eval_env["A2UI_LITERT_ALLOWED_GPU_UUIDS"] = json.dumps(allowed_uuids)

        def merge():
            command(export["prepare_command"], output / "logs/merge.log", export_env)
            directory = Path(export["merged_model_dir"])
            return [directory / "deployment_source.json"]
        stage("merge", merge)
        for cohort, count in (("golden32", 32), ("golden35", 35)):
            def evaluate_merged(cohort=cohort, count=count):
                destination = output / "evaluations" / f"merged_{cohort}"
                argv = evaluation_command(training_plan, "best", cohort, destination)
                argv[argv.index("--checkpoint") + 1] = export["merged_model_dir"]
                argv[argv.index("--checkpoint-kind") + 1] = "merged"
                argv[argv.index("--evaluation-name") + 1] = f"merged_{cohort}"
                argv.extend(["--step", str(step)])
                command(argv, output / f"logs/merged_{cohort}.log", eval_env)
                evaluated, paths = _evaluation(destination, count, artifact=Path(export["merged_model_dir"]))
                log_result("merged", cohort, evaluated, step)
                return paths
            stage(f"merged_{cohort}", evaluate_merged)
        for variant, specification in export["variants"].items():
            def convert(variant=variant, specification=specification):
                command(specification["command"], output / f"logs/export_{variant}.log", export_env)
                result = validate_deployment_export_output(export, variant)
                state.setdefault("exports", {})[variant] = result
                return [Path(path) for path in result["files"]]
            stage(f"export_{variant}", convert)
            for cohort, count in (("golden32", 32), ("golden35", 35)):
                def evaluate_variant(variant=variant, specification=specification, cohort=cohort, count=count):
                    destination = output / "evaluations" / f"{variant}_{cohort}"
                    argv = [sys.executable, "-u", str(repo_root() / "training/scripts/evaluate_litertlm_on_golden.py"),
                            "--builtin-gpu", "--require-prepared-contract", "--model-config", str(training_output / "fit/training_config.yaml"),
                            "--runtime-python", plan["runtime_python"], "--model", specification["artifact"],
                            "--split", str(training_output / f"prepared/{cohort}.jsonl"), "--output-dir", str(destination),
                            "--max-rows", str(count), "--required-rows", str(count), "--max-input-tokens", str(base.max_input_tokens),
                            "--max-new-tokens", str(base.max_new_tokens), "--run-id", plan["run_id"], "--evaluation-name", f"{variant}_{cohort}",
                            "--step", str(step), "--tensorboard-root", base.tensorboard_root, "--metric-version", "v5_4",
                            "--case-timeout-seconds", str(options.case_timeout_seconds), "--load-timeout-seconds", str(options.load_timeout_seconds),
                            "--runtime-timeout-seconds", str(options.generation_timeout_seconds),
                            "--runtime-cache-dir", str(output / "runtime_cache")]
                    command(argv, output / f"logs/{variant}_{cohort}.log", eval_env)
                    evaluated, paths = _evaluation(destination, count, artifact=Path(specification["artifact"]), litert=True)
                    log_result(variant, cohort, evaluated, step)
                    return paths
                stage(f"{variant}_{cohort}", evaluate_variant)
        def scorecard():
            # Verify all published evidence again: a partial/corrupt run never
            # receives a successful scorecard, including changed model bytes.
            with Progress("Verify final deployment evidence", unit="stage"):
                for item in state["completed"].values():
                    for name, digest in item["files"].items():
                        if sha256(Path(name)) != digest:
                            raise ValueError(f"Completed evidence changed: {name}")
            expected = {f"{label}_{cohort}" for label in ("checkpoint_best", "checkpoint_final", "merged", "w32", "w16", "w8", "w4") for cohort in ("golden32", "golden35")}
            if set(state["results"]) != expected:
                raise ValueError("Deployment scorecard is missing required evaluations")
            value = {"schema_version": 1, "status": "complete", "results": state["results"], "profile": base.profile,
                     "source_checkpoint": str(checkpoint), "checkpoint_step": step, "variants": state["exports"],
                     "gpu_policy": plan["gpu_policy"], "golden35_used_for_selection": False, "golden32_unique_sources": 31,
                     "golden35_unique_sources": 35, "tensorboard_dir": plan["tensorboard_dir"],
                     "evidence": {name: item["files"] for name, item in state["completed"].items()},
                     "official_retained_scale_export": False, "mtp_exported": False}
            parameters = {"profile": base.profile, "max_new_tokens": base.max_new_tokens,
                          "max_seq_length": base.max_seq_length, "seed": base.seed,
                          "epochs": base.epochs, "tuning": options.tune, "selection": "golden32_only",
                          "augmentation": training_options.augmentation, "training_gpus": gpu_profile["world_size"],
                          "effective_batch_size": gpu_profile["effective_batch_size"]}
            config = load_yaml(training_output / "fit/training_config.yaml")
            for name in ("learning_rate", "weight_decay", "warmup_ratio"):
                parameters[name] = config["training"][name]
            hmetrics = {f"hparam/{label}/{key}": val for label, result in state["results"].items()
                        for key, val in flatten_scalar_metrics(result["aggregate"]).items()}
            writer.add_hparams(parameters, hmetrics, run_name="final_comparison", global_step=step)
            writer.flush()
            lines = ["# Deployment results", "", "Golden32 has 32 occurrences / 31 unique sources; Golden35 is the held-out cohort.", "",
                     "| Model | Golden32 unique-source reward | Golden35 reward |", "|---|---:|---:|"]
            for label in ("checkpoint_best", "checkpoint_final", "merged", "w32", "w16", "w8", "w4"):
                g32 = state["results"][f"{label}_golden32"]["aggregate"]["unique_source_generation_reward_v5_4_avg"]
                g35 = state["results"][f"{label}_golden35"]["aggregate"]["generation_reward_v5_4_avg"]
                lines.append(f"| {label} | {g32:.6f} | {g35:.6f} |")
            (output / "deployment_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
            _write(output / "deployment_scorecard.json", value)
            return [output / "deployment_scorecard.json", output / "deployment_results.md"]
        stage("scorecard", scorecard)
        state.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat())
        _write(record, state)
        return state
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _write(record, state)
        log(f"Deployment stopped at {state['active_stage']}. Completed data/checkpoints/logs retained; no automatic training restart or CPU fallback.")
        raise
    finally:
        if writer is not None:
            writer.close()
