"""Sequential, bounded Golden32 development experiments with a locked Golden35 holdout.

This is a small reproducible screening runner, not Bayesian search and not a
claim that short-run winners remain best after a full training schedule.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable

from ir_training.common.config import load_yaml, repo_root
from ir_training.common.progress import Progress, fingerprint_file, log
from ir_training.eval.tensorboard_logging import _summary_writer_factory, flatten_scalar_metrics
from ir_training.pipeline.golden_training import (
    GoldenTrainingOptions, _run_command, _write, build_plan, evaluation_command,
    run_pipeline, sha256,
)

SELECTION_METRIC = "unique_source_generation_reward_v5_4_avg"
MAX_TRIALS = 12
TRIAL_FIELDS = {"learning_rate", "weight_decay", "warmup_ratio", "augmentation"}


@dataclass(frozen=True)
class ExperimentOptions:
    base: GoldenTrainingOptions
    trial_steps: int
    include_augmentation: bool = False
    trials_file: Path | None = None
    evaluate_selected_holdout: bool = True


def _defaults(base: GoldenTrainingOptions) -> dict[str, Any]:
    name = "gemma4_e2b" if base.profile == "e2b" else "gemma3_270m"
    recipe = load_yaml(repo_root() / "training/configs/models" / f"{name}_a2ui_express_review_sft.yaml")
    training = recipe["training"]
    return {
        "learning_rate": base.learning_rate if base.learning_rate is not None else (5e-6 if base.qat else training["learning_rate"]),
        "weight_decay": base.weight_decay if base.weight_decay is not None else training["weight_decay"],
        "warmup_ratio": base.warmup_ratio if base.warmup_ratio is not None else training["warmup_ratio"],
        "augmentation": "none",
    }


def _trial_specs(options: ExperimentOptions) -> list[dict[str, Any]]:
    baseline = _defaults(options.base)
    specs = [{"name": "baseline", **baseline}]
    if options.trials_file is not None:
        values = json.loads(options.trials_file.read_text(encoding="utf-8-sig"))
        if not isinstance(values, list) or not 1 <= len(values) < MAX_TRIALS:
            raise ValueError(f"Trials JSON must contain 1 to {MAX_TRIALS - 1} objects; baseline is always added")
        for value in values:
            if not isinstance(value, dict) or set(value) - TRIAL_FIELDS - {"name"}:
                raise ValueError("Trials may vary only name, learning_rate, weight_decay, warmup_ratio, augmentation")
            specs.append({**baseline, **value})
    else:
        specs.extend([
            {**baseline, "name": "lr_half", "learning_rate": baseline["learning_rate"] / 2},
            {**baseline, "name": "lr_double", "learning_rate": baseline["learning_rate"] * 2},
            {**baseline, "name": "regularization", "weight_decay": 0.05, "warmup_ratio": 0.05},
        ])
    if options.include_augmentation:
        specs.append({**baseline, "name": "rare_components", "augmentation": "rare_components"})
    if len(specs) > MAX_TRIALS:
        raise ValueError(f"At most {MAX_TRIALS} sequential trials including baseline/augmentation are allowed")
    names: set[str] = set()
    seen: set[str] = set()
    for spec in specs:
        name = spec.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,47}", name) or name in names:
            raise ValueError("Every trial needs a unique safe name (letters, digits, underscore or hyphen; max 48)")
        names.add(name)
        for key in ("learning_rate", "weight_decay", "warmup_ratio"):
            value = spec[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Trial {name}: {key} must be finite numeric")
        if not 0 < spec["learning_rate"] <= 0.001 or not 0 <= spec["weight_decay"] <= 1 or not 0 <= spec["warmup_ratio"] < 1:
            raise ValueError(f"Trial {name}: unsafe learning rate, weight decay or warmup range")
        if spec["augmentation"] not in {"none", "rare_components"}:
            raise ValueError(f"Trial {name}: unsupported augmentation")
        identity = json.dumps({key: spec[key] for key in sorted(TRIAL_FIELDS)}, sort_keys=True)
        if identity in seen:
            raise ValueError(f"Duplicate hyperparameter configuration: {name}")
        seen.add(identity)
    return specs


def build_experiment_plan(options: ExperimentOptions) -> dict[str, Any]:
    """Validate without outputs, model loading, TensorBoard imports or GPU probes."""
    if isinstance(options.trial_steps, bool) or not isinstance(options.trial_steps, int) or options.trial_steps <= 0:
        raise ValueError("An explicit positive --trial-steps optimizer budget is required")
    if options.base.steps is not None and options.base.steps != options.trial_steps:
        raise ValueError("Use --trial-steps for the common budget; conflicting base.steps is not allowed")
    if options.base.augmentation != "none":
        raise ValueError("Experiments always include an unaugmented baseline; use --include-augmentation or trials JSON")
    output = options.base.output_dir.expanduser().resolve()
    # Include a stable path suffix so two independent suites with equal folder
    # names do not merge their TensorBoard streams.
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", output.name).strip("_") or "experiment"
    suite_id = safe + "_" + hashlib.sha256(str(output).encode()).hexdigest()[:8]
    cache_root = (options.base.preparation_cache_dir or output.parent / ".golden-preparation-cache").expanduser().resolve()
    token_root = (options.base.token_cache_dir or cache_root / "tokens").expanduser().resolve()
    if any(cache.is_relative_to(output) or output.is_relative_to(cache) for cache in (cache_root, token_root)):
        raise ValueError("Experiment caches must be outside and must not contain the experiment output directory")
    trials = []
    for index, spec in enumerate(_trial_specs(options)):
        trial_output = output / "trials" / f"{suite_id}_{index:02d}_{spec['name']}"
        base = replace(options.base, output_dir=trial_output, steps=options.trial_steps,
                       evaluate_golden35=False, preparation_cache_dir=cache_root, token_cache_dir=token_root,
                       **{key: spec[key] for key in TRIAL_FIELDS})
        trial_plan = build_plan(base)
        # Validate the suite root as well as child run paths. An empty parent
        # containing source/model must never become the experiment workspace.
        for protected in (Path(trial_plan["options"]["model_dir"]), Path(trial_plan["options"]["input_dir"] or trial_plan["options"]["source_run_dir"])):
            if output == protected or output in protected.parents or output.is_relative_to(protected):
                raise ValueError("Experiment output must be outside and must not contain model/source inputs")
        trials.append({"index": index, "name": spec["name"], "parameters": {key: spec[key] for key in TRIAL_FIELDS}, "plan": trial_plan})
    return {
        "schema_version": 1, "workflow": "sequential_golden32_screening_v1", "suite_id": suite_id,
        "output_dir": str(output), "tensorboard_dir": str(Path(options.base.tensorboard_root).expanduser().resolve() / "experiments" / suite_id),
        "trial_steps": options.trial_steps, "total_optimizer_step_budget": options.trial_steps * len(trials),
        "max_concurrent_training_runs": 1, "trials": trials,
        "selection": {"cohort": "golden32", "metric": SELECTION_METRIC, "checkpoint": "best", "direction": "maximize", "ties": "earliest trial (baseline first)"},
        "evaluate_selected_holdout": options.evaluate_selected_holdout,
        "golden35_policy": ("No per-trial Golden35 inference. Lock winner using Golden32, then evaluate its existing best checkpoint once on Golden35. Never select using Golden35."
                            if options.evaluate_selected_holdout else "No Golden35 inference during screening. Lock hyperparameters using Golden32; the deployment pipeline trains a fresh full run before holdout evaluation."),
        "warning": "Screening uses equal optimizer steps, seed and requested effective batch. Short-run ranking is not proof of full-run quality; Golden32 is development data, not an unbiased final benchmark.",
    }


def _options_from_plan(plan: dict[str, Any]) -> GoldenTrainingOptions:
    values = dict(plan["options"])
    for name in ("model_dir", "output_dir", "source_run_dir", "input_dir", "preparation_cache_dir", "token_cache_dir"):
        if values[name] is not None:
            values[name] = Path(values[name])
    return GoldenTrainingOptions(**values)


def _numeric_metric(result: dict[str, Any], name: str) -> float:
    value = result.get("aggregate", {}).get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Missing or nonfinite required development metric: {name}; refusing to choose a winner")
    return float(value)


def _input_bindings(plan: dict[str, Any], interval: float) -> dict[str, str]:
    first = plan["trials"][0]["plan"]
    model = Path(first["options"]["model_dir"])
    files = [*map(Path, first["source_files"]), *sorted(path for path in model.rglob("*") if path.is_file())]
    for item in first["goldens"].values():
        files.extend(Path(item[key]) for key in ("path", "benchmark_manifest_path"))
    # Preserve logical file entries in a resolved model directory: HF snapshot
    # files may be symlinks to blobs outside that directory. Hashing follows the
    # link, but inventory membership must refer to the snapshot entry.
    bindings = {str(path.absolute()): fingerprint_file(path, interval=interval)["sha256"] for path in files}
    from ir_training.pipeline.preparation_cache import identity
    with Progress("Bind experiment code, production prompt, recipes and schemas", unit="stage", interval=interval):
        prepared_identity = identity(first, {path: bindings[str(Path(path).absolute())] for path in first["source_files"]})
        # Preparation uses semantic hashes with relative.py:function keys;
        # experiment execution deliberately pins each complete file's bytes.
        extras = {repo_root() / key.partition(":")[0] for key in prepared_identity["implementation"]}
        extras.update((repo_root() / "training/configs").rglob("*.yaml"))
        # Cache reuse ignores unrelated training changes; a running experiment
        # still requires one immutable implementation across every trial.
        extras.update((repo_root() / "training/src/ir_training").rglob("*.py"))
        extras.add(repo_root() / first["shared_prompt"]["source_path"])
        for name in ("run_golden_experiments.py", "run_golden_training.py", "prepare_review_training.py", "launch_review_training.py", "train_sft.py", "evaluate_checkpoint_on_golden.py"):
            extras.add(repo_root() / "training/scripts" / name)
        bindings.update({str(path.absolute()): sha256(path) for path in extras})
    return bindings


def _execution_context(plan: dict[str, Any], bindings: dict[str, str]) -> dict[str, Any]:
    from ir_training.pipeline.preparation_cache import identity
    first = plan["trials"][0]["plan"]
    result = identity(first, {path: bindings[str(Path(path).absolute())] for path in first["source_files"]})
    # Include training runtime dependencies, not just tokenizer/validation ones.
    packages = {}
    for name in ("torch", "accelerate", "peft", "trl", "datasets", "tensorboard", "bitsandbytes"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"preparation_identity": result, "training_packages": packages}


def _verify_execution_context(plan: dict[str, Any], bindings: dict[str, str], expected: dict[str, Any]) -> None:
    if _execution_context(plan, bindings) != expected:
        raise ValueError("Experiment implementation, tokenizer assets or Python package versions changed between trials")


def _verify_bindings(bindings: dict[str, str], *, label: str, interval: float) -> None:
    with Progress(label, unit="stage", interval=interval):
        for path, digest in bindings.items():
            if not Path(path).is_file() or sha256(Path(path)) != digest:
                raise ValueError(f"Bound experiment artifact changed: {path}")


def _verify_directory_inventory(directory: Path, bindings: dict[str, str]) -> None:
    directory = directory.resolve()
    prefix = os.path.normcase(str(directory)).rstrip(os.sep) + os.sep
    normalized = {os.path.normcase(os.path.abspath(path)) for path in bindings}
    expected = {path for path in normalized if path.startswith(prefix)}
    actual = {os.path.normcase(os.path.abspath(path)) for path in directory.rglob("*") if path.is_file()}
    if not expected or actual != expected:
        raise ValueError(f"Bound model/checkpoint file inventory changed: {directory}")


def _completed_bindings(output: Path, state: dict[str, Any]) -> dict[str, str]:
    if state.get("status") != "complete":
        raise ValueError("Trial did not complete; experiment stops without advancing or silently restarting training")
    if any("golden35" in name for name in state.get("completed", {})):
        raise ValueError("A development trial evaluated the locked Golden35 holdout")
    required = {"prepare", "configure", "preflight", "training", "best_golden32", "final_golden32", "scorecard"}
    if not required.issubset(state.get("completed", {})):
        raise ValueError("Trial is missing required completed stages")
    bindings = {path: digest for stage in state["completed"].values() for path, digest in stage["files"].items()}
    bindings[str(output / "pipeline_manifest.json")] = sha256(output / "pipeline_manifest.json")
    return bindings


def _trial_result(trial: dict[str, Any], state: dict[str, Any], elapsed: float) -> dict[str, Any]:
    output = Path(trial["plan"]["options"]["output_dir"])
    scorecard = json.loads((output / "evaluation_scorecard.json").read_text(encoding="utf-8"))
    values = scorecard.get("evaluations", {})
    if set(values) != {"best_golden32", "final_golden32"}:
        raise ValueError("Trial scorecard must contain only complete best/final Golden32 evaluations")
    for value in values.values():
        if value.get("row_count") != 32:
            raise ValueError("Incomplete Golden32 evaluation; refusing to rank partial results")
    best = _numeric_metric(values["best_golden32"], SELECTION_METRIC)
    final = _numeric_metric(values["final_golden32"], SELECTION_METRIC)
    config = load_yaml(output / "fit/training_config.yaml")
    expected = trial["plan"]["options"]
    for name in ("learning_rate", "weight_decay", "warmup_ratio", "seed"):
        if config["training"].get(name) != expected[name]:
            raise ValueError(f"Realized training config disagrees with trial parameter: {name}")
    if config["training"].get("max_steps") != expected["steps"]:
        raise ValueError("Realized optimizer-step budget differs from the common trial budget")
    gpu = config.get("runtime", {}).get("gpu_profile", {})
    return {"name": trial["name"], "index": trial["index"], "output_dir": str(output),
            "parameters": trial["parameters"], "status": "complete", "elapsed_seconds": elapsed,
            "selection_score": best, "final_golden32_score": final,
            "gpu_profile": gpu, "effective_batch_size": config["training"].get("expected_effective_batch_size"),
            "evaluations": values, "artifact_bindings": _completed_bindings(output, state)}


def _record_comparison(writer: Any, trial: dict[str, Any], *, budget: int) -> None:
    metrics = {"hparam/best_golden32": trial["selection_score"], "hparam/final_golden32": trial["final_golden32_score"],
               "hparam/pipeline_elapsed_seconds": trial["elapsed_seconds"]}
    writer.add_hparams({**trial["parameters"], "optimizer_steps": budget}, metrics, run_name=trial["name"], global_step=budget)
    for role, result in trial["evaluations"].items():
        for key, value in flatten_scalar_metrics(result.get("aggregate", {})).items():
            writer.add_scalar(f"comparison/{trial['name']}/{role}/{key}", value, budget)
    writer.add_text(f"comparison/{trial['name']}/summary", "```json\n" + json.dumps({key: value for key, value in trial.items() if key != "artifact_bindings"}, indent=2) + "\n```", budget)
    writer.flush()


def _full_training_handoff(plan: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    values = dict(plan["trials"][selected["index"]]["plan"]["options"])
    values.update(steps=None, evaluate_golden35=True,
                  output_dir=str(Path(plan["output_dir"]).parent / f"{plan['suite_id']}_selected_full"))
    command = [sys.executable, str(repo_root() / "training/scripts/run_golden_training.py")]
    for key, value in values.items():
        if value is None or key == "evaluate_golden35":
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if key in {"preparation_cache", "token_cache", "gradient_checkpointing"}:
                command.append(flag if value else "--no-" + key.replace("_", "-"))
            elif value:
                command.append(flag)
        else:
            command.extend((flag, str(value)))
    return {"options": values, "plan_only_command_argv": command,
            "execute_instruction": "Review the plan, then append --execute. This command has not been run.",
            "warning": "Use an adequate full epoch budget. The hyperparameters were locked on short-run Golden32 development scores; do not revise them using the Golden35 score."}


def _evaluate_holdout(plan: dict[str, Any], selected: dict[str, Any], bindings: dict[str, str], *, command_runner: Callable, interval: float, execution_context: dict[str, Any]) -> dict[str, Any]:
    trial = plan["trials"][selected["index"]]
    destination = Path(plan["output_dir"]) / "selected_golden35"
    if destination.exists():
        raise FileExistsError("Golden35 evaluation already exists; this experiment never automatically retries the holdout")
    _verify_bindings(bindings, label="Verify locked winner, model and source before Golden35", interval=interval)
    _verify_execution_context(plan, bindings, execution_context)
    options = trial["plan"]["options"]
    model_dir = Path(options["model_dir"])
    checkpoint = Path(options["output_dir"]) / "fit/training/best_golden_checkpoint"
    _verify_directory_inventory(model_dir, bindings)
    _verify_directory_inventory(checkpoint, bindings)
    runtime = load_yaml(Path(options["output_dir"]) / "fit/training_config.yaml")["runtime"]
    environment = {**os.environ, "A2UI_TENSORBOARD_ROOT": options["tensorboard_root"], "PYTHONUNBUFFERED": "1",
                   "CUDA_VISIBLE_DEVICES": runtime["cuda_visible_devices"], "A2UI_SKIP_CUDA_DEVICE_NORMALIZE": "1", "TOKENIZERS_PARALLELISM": "false"}
    environment.pop("A2UI_CUDA_VISIBLE_DEVICES", None)
    environment.pop("A2UI_EXCLUDE_CUDA_DEVICES", None)
    command = evaluation_command(trial["plan"], "best", "golden35", destination)
    with Progress("Locked winner: Golden35 holdout evaluation (one checkpoint, one pass)", unit="stage", interval=interval):
        command_runner(command, Path(plan["output_dir"]) / "logs/selected_golden35.log", environment)
    result = json.loads((destination / "evaluation_result.json").read_text(encoding="utf-8"))
    if result.get("row_count") != 35:
        raise ValueError("Selected checkpoint did not complete all 35 holdout rows")
    _numeric_metric(result, "generation_reward_v5_4_avg")
    for name in ("aggregate_metrics.json", "predictions.jsonl", "scored_predictions.jsonl"):
        if not (destination / name).is_file():
            raise ValueError(f"Missing holdout evidence: {name}")
    _verify_bindings(bindings, label="Verify winner and inputs remained unchanged during holdout", interval=interval)
    _verify_execution_context(plan, bindings, execution_context)
    _verify_directory_inventory(model_dir, bindings)
    _verify_directory_inventory(checkpoint, bindings)
    return result


def run_experiments(options: ExperimentOptions, *, execute: bool = False,
                    pipeline_runner: Callable = run_pipeline, command_runner: Callable = _run_command,
                    writer_factory: Callable | None = None) -> dict[str, Any]:
    plan = build_experiment_plan(options)
    if not execute:
        return {**plan, "status": "plan_only", "training_executed": False}
    output = Path(plan["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    record = output / "experiments_manifest.json"
    state: dict[str, Any] = {"schema_version": 1, "plan": plan, "status": "running", "active_trial": None,
                             "trials": [], "started_at": datetime.now(timezone.utc).isoformat(), "golden35_used_for_selection": False}
    _write(record, state)
    writer = None
    try:
        # Fail before any costly preparation/training if comparison logging is
        # unavailable or the TensorBoard mount is not writable.
        factory = writer_factory or _summary_writer_factory()
        writer = factory(log_dir=plan["tensorboard_dir"])
        writer.add_text("experiment/plan", "```json\n" + json.dumps(plan, indent=2) + "\n```", 0)
        writer.flush()
        log(f"Sequential experiment: {len(plan['trials'])} trials x {options.trial_steps} optimizer steps; TensorBoard: {plan['tensorboard_dir']}")
        bindings = _input_bindings(plan, options.base.progress_seconds)
        state["input_bindings"] = bindings
        execution_context = _execution_context(plan, bindings)
        state["execution_context"] = execution_context
        _write(record, state)
        effective_batch = None
        for trial in plan["trials"]:
            state["active_trial"] = trial["name"]
            _write(record, state)
            log(f"Trial {trial['index'] + 1}/{len(plan['trials'])}: {trial['name']}; {json.dumps(trial['parameters'], sort_keys=True)}; Golden35 disabled")
            _verify_bindings(bindings, label="Verify common experiment inputs", interval=options.base.progress_seconds)
            _verify_execution_context(plan, bindings, execution_context)
            _verify_directory_inventory(options.base.model_dir, bindings)
            started = time.monotonic()
            result = pipeline_runner(_options_from_plan(trial["plan"]), execute=True)
            summary = _trial_result(trial, result, time.monotonic() - started)
            _verify_bindings(bindings, label="Verify common inputs after trial", interval=options.base.progress_seconds)
            _verify_execution_context(plan, bindings, execution_context)
            _verify_directory_inventory(options.base.model_dir, bindings)
            _verify_bindings(summary["artifact_bindings"], label="Verify trial evidence before comparison", interval=options.base.progress_seconds)
            batch = summary["effective_batch_size"]
            if isinstance(batch, bool) or not isinstance(batch, int) or batch <= 0:
                raise ValueError("Trial did not report a positive realized effective batch size")
            if effective_batch is not None and effective_batch != batch:
                raise ValueError("Realized effective batch changed between trials; refusing unfair comparison")
            effective_batch = batch
            state["trials"].append(summary)
            state["active_trial"] = None
            _write(record, state)
            _record_comparison(writer, summary, budget=options.trial_steps)
            log(f"Trial {trial['name']} complete in {summary['elapsed_seconds']:.1f}s; best Golden32={summary['selection_score']:.6f}; final Golden32={summary['final_golden32_score']:.6f}")
        selected = max(state["trials"], key=lambda item: item["selection_score"])
        selection_path = output / "selection_locked.json"
        selection = {"schema_version": 1, "locked_at": datetime.now(timezone.utc).isoformat(), "trial": selected["name"],
                     "selection_metric": SELECTION_METRIC, "selection_score": selected["selection_score"],
                     "golden35_seen": False, "checkpoint": str(Path(selected["output_dir"]) / "fit/training/best_golden_checkpoint"),
                     "artifact_bindings": {**bindings, **selected["artifact_bindings"]}}
        _write(selection_path, selection)
        _write(output / "selected_full_training_options.json", _full_training_handoff(plan, selected))
        holdout_bindings = {**selection["artifact_bindings"], str(selection_path): sha256(selection_path)}
        state.update(status="holdout_evaluation" if options.evaluate_selected_holdout else "selection_locked", selected_trial=selected["name"])
        _write(record, state)
        log(f"Winner locked before holdout: {selected['name']}. Golden35 cannot change this selection.")
        holdout = (_evaluate_holdout(plan, selected, holdout_bindings, command_runner=command_runner, interval=options.base.progress_seconds,
                                    execution_context=execution_context) if options.evaluate_selected_holdout else None)
        state.update(status="complete", selected_golden35=holdout, finished_at=datetime.now(timezone.utc).isoformat())
        _write(record, state)
        comparison = {"status": "complete", "selected_trial": selected["name"], "selection": plan["selection"],
                      "golden35_used_for_selection": False, "selected_golden35": holdout,
                      "trials": [{key: value for key, value in item.items() if key != "artifact_bindings"} for item in state["trials"]],
                      "warning": plan["warning"], "tensorboard_dir": plan["tensorboard_dir"]}
        _write(output / "comparison.json", comparison)
        for key, value in flatten_scalar_metrics((holdout or {}).get("aggregate", {})).items():
            writer.add_scalar(f"selected_holdout/golden35/{key}", value, options.trial_steps)
        writer.add_text("experiment/selection", json.dumps({"selected_trial": selected["name"], "golden35_used_for_selection": False}), options.trial_steps)
        writer.flush()
        return state
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _write(record, state)
        log("Experiment stopped. Completed trials are retained; no failed training or holdout is automatically resumed. Inspect experiments_manifest.json and logs.")
        raise
    finally:
        if writer is not None:
            writer.close()
