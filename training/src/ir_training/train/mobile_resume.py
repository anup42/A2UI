"""Explicit, horizon-only continuation of the official retained-mobile LoRA run.

The source run is immutable. A continuation owns new output/launch/log paths,
but reuses the original prepared data and every model/training/QAT setting.
This policy is not available to dense, full-QAT or ordinary LoRA workflows.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

from ir_training.common.config import load_yaml, resolve_path, training_root
from ir_training.qat.numeric_preflight import OFFICIAL_MOBILE_WORKFLOW
from ir_training.train import resume_contract as rc
from ir_training.train.recipe import optimizer_steps

POLICY = "retained_mobile_horizon_extension_v1"
LAUNCH_MODE = "explicit_retained_mobile_continuation"


def enabled(config: dict) -> bool:
    policy = (config.get("training") or {}).get("resume_policy")
    if policy is None:
        return False
    if policy != POLICY:
        raise ValueError(f"Unknown training.resume_policy: {policy!r}")
    training, qat = config.get("training") or {}, config.get("qat") or {}
    if (
        (config.get("run") or {}).get("purpose") != OFFICIAL_MOBILE_WORKFLOW
        or training.get("method") != "qat_lora_sft"
        or qat.get("enabled") is not True
        or qat.get("scale_mode") != "retained_mobile"
        or training.get("refuse_resume") is not False
        or rc._resume_source(config) is None
    ):
        raise ValueError(
            "Horizon continuation requires explicit official retained-mobile LoRA resume"
        )
    return True


def source_config(checkpoint: Path) -> tuple[Path, dict, dict]:
    checkpoint = checkpoint.resolve(strict=True)
    metadata = rc._read_object(checkpoint / "training_metadata.json")
    snapshot = checkpoint / "training_config.yaml"
    recorded = metadata.get("config_path")
    path = resolve_path(recorded, training_root()) if recorded else snapshot
    if not path.is_file():
        path = snapshot
    path = rc._bound_config(checkpoint, metadata, path)
    config = load_yaml(path)
    if (
        (config.get("run") or {}).get("purpose") != OFFICIAL_MOBILE_WORKFLOW
        or (config.get("training") or {}).get("method") != "qat_lora_sft"
        or (config.get("qat") or {}).get("scale_mode") != "retained_mobile"
    ):
        raise ValueError(
            "Resume source is not an official retained-mobile QAT LoRA checkpoint"
        )
    return path, config, metadata


def _horizon(config: dict) -> dict:
    training = config["training"]
    epochs, steps = training.get("epochs", 2), training.get("max_steps")
    if (
        isinstance(epochs, bool)
        or not isinstance(epochs, (int, float))
        or not math.isfinite(epochs)
        or epochs <= 0
    ):
        raise ValueError("Resume epochs must be positive and finite")
    if steps is not None and (type(steps) is not int or steps <= 0):
        raise ValueError("Resume max_steps must be a positive integer")
    dataset = resolve_path(config["run"]["dataset_dir"], training_root())
    with (dataset / "train.jsonl").open(encoding="utf-8") as stream:
        rows = sum(bool(line.strip()) for line in stream)
    batch = rc._lineage_batch(config)
    micro, accumulation = (
        training["per_device_train_batch_size"],
        training["gradient_accumulation_steps"],
    )
    total = (
        steps
        if steps is not None
        else optimizer_steps(
            rows=rows,
            world_size=batch // (micro * accumulation),
            microbatch=micro,
            accumulation=accumulation,
            epochs=epochs,
        )
    )
    return {"epochs": epochs, "max_steps": steps, "total_optimizer_steps": total}


def horizon_record(checkpoint: Path, config: dict) -> dict:
    path, original, metadata = source_config(checkpoint)
    previous = _horizon(original)
    requested = _horizon(config)
    state = rc._read_object(checkpoint / "trainer_state.json")
    step = state.get("global_step")
    if type(step) is not int or step <= 0 or metadata.get("checkpoint_step") != step:
        raise ValueError(
            "Resume requires matching positive Trainer and checkpoint optimizer steps"
        )
    if (
        checkpoint.name != f"checkpoint-{step}"
        or metadata.get("checkpoint_role") != "trainer_intermediate"
    ):
        raise ValueError(
            "Resume requires a numbered Trainer checkpoint, not a weight-only best/final adapter"
        )
    if (previous["max_steps"] is None) != (requested["max_steps"] is None):
        raise ValueError("Resume cannot switch between epochs and max_steps horizons")
    if requested["epochs"] < previous["epochs"] or (
        previous["max_steps"] is not None
        and (
            requested["max_steps"] < previous["max_steps"]
            or not rc._same(requested["epochs"], previous["epochs"])
        )
    ):
        raise ValueError("Resume may only increase the active training horizon")
    if (
        requested["total_optimizer_steps"] < previous["total_optimizer_steps"]
        or requested["total_optimizer_steps"] <= step
    ):
        raise ValueError(
            "Resume horizon must not decrease and must extend beyond the completed global step"
        )
    if (
        state.get("max_steps") is not None
        and state["max_steps"] != previous["total_optimizer_steps"]
    ):
        raise ValueError("Saved Trainer horizon differs from its bound resume contract")
    inherited = (original.get("training") or {}).get("resume_horizon")
    if enabled(original):
        if not isinstance(inherited, dict):
            raise ValueError("Resume source lacks its original horizon lineage")
        origin_path = Path(inherited["original_config"])
        origin_sha = inherited["original_config_sha256"]
        if not origin_path.is_file() or rc.file_sha256(origin_path) != origin_sha:
            raise ValueError("Original resume config changed or disappeared")
        origin = _horizon(load_yaml(origin_path))
        if origin != inherited.get("original"):
            raise ValueError("Original resume horizon changed")
        warmup = inherited["scheduler_warmup_steps"]
    else:
        if (
            rc._resume_source(original) is not None
            or original["training"].get("refuse_resume") is not True
        ):
            raise ValueError(
                "Resume chain must originate in a strict fresh official run"
            )
        origin_path, origin_sha, origin = path, rc.file_sha256(path), previous
        training = original["training"]
        warmup = training.get("warmup_steps")
        if warmup is None:
            warmup = math.ceil(
                previous["total_optimizer_steps"] * training.get("warmup_ratio", 0.03)
            )
    return {
        "policy": POLICY,
        "source_checkpoint": str(checkpoint.resolve()),
        "source_config": str(path),
        "source_config_sha256": rc.file_sha256(path),
        "original_config": str(origin_path),
        "original_config_sha256": origin_sha,
        "original": origin,
        "previous": previous,
        "requested": requested,
        "completed_global_step": step,
        "scheduler_warmup_steps": warmup,
    }


def _check_config_delta(source: dict, current: dict) -> None:
    # These are output locations, not recipe exceptions. Data/model/token-cache
    # paths and all benchmark/generation/cadence settings remain identical.
    operational = {
        ("run", key)
        for key in ("id", "output_dir", "launch_plan_path", "preflight_report_path")
    }
    operational |= {
        ("training", "logging_dir"),
        ("golden_eval", "output_dir"),
        ("golden_eval", "best_checkpoint_dir"),
        ("runtime", "gpu_profile"),
        ("runtime", "cuda_visible_devices"),
    }
    allowed = operational | {
        ("training", key)
        for key in (
            "epochs",
            "max_steps",
            "resume_from_checkpoint",
            "refuse_resume",
            "resume_policy",
            "resume_horizon",
        )
    }
    left, right = copy.deepcopy(source), copy.deepcopy(current)
    for path in allowed:
        left.get(path[0], {}).pop(path[1], None)
        right.get(path[0], {}).pop(path[1], None)
    changes = rc._config_changes(left, right)
    if changes:
        raise ValueError(
            "Unrelated config changes in official resume: "
            + ", ".join(sorted(".".join(p) for p in changes))
        )
    run_root = Path(current["run"]["output_dir"]).resolve().parent
    if run_root == Path(source["run"]["output_dir"]).resolve().parent:
        raise ValueError(
            "Official continuation requires a fresh output run; never overwrite the source run"
        )
    expected = {
        ("run", "id"): run_root.name,
        ("run", "output_dir"): str(run_root / "trainer"),
        ("run", "launch_plan_path"): str(run_root / "launch/launch_plan.json"),
        ("run", "preflight_report_path"): str(
            run_root / "launch/preflight_report.json"
        ),
        ("golden_eval", "best_checkpoint_dir"): str(
            run_root / "best_golden_checkpoint"
        ),
    }
    for (section, key), value in expected.items():
        if current[section].get(key) != value:
            raise ValueError(f"Resume output relocation is invalid: {section}.{key}")
    if (
        not Path(current["golden_eval"]["output_dir"])
        .resolve()
        .is_relative_to(run_root)
    ):
        raise ValueError("Resume Golden output must stay in the new run")


def verify_continuation(
    checkpoint: Path, config: dict, expected: dict | None = None
) -> dict:
    if not enabled(config) or rc._resume_source(config) != checkpoint.resolve():
        raise ValueError("Resume must explicitly identify this source checkpoint")
    _, source, metadata = source_config(checkpoint)
    dataset = resolve_path(source["run"]["dataset_dir"], training_root())
    contract = rc.build_resume_contract(
        source, dataset, effective_batch=rc._lineage_batch(source)
    )
    if set(contract["splits"]) != {"train", "val"}:
        raise ValueError("Resume requires both original prepared train and val hashes")
    rc._verify_lineage_metadata(metadata, source, contract, dataset)
    rc._verify_source_inventory(checkpoint, metadata)
    verified = rc.verify_resume_contract(checkpoint, contract)
    training = source["training"]
    world = contract["effective_batch_size"] // (
        training["per_device_train_batch_size"]
        * training["gradient_accumulation_steps"]
    )
    if world == 1 and not (checkpoint / "rng_state.pth").is_file():
        raise ValueError("Single-rank resume requires rng_state.pth")
    _check_config_delta(source, config)
    requested = rc.build_resume_contract(
        config, dataset, effective_batch=rc._lineage_batch(config)
    )
    if expected is not None and not rc._same(expected, requested):
        raise ValueError("Requested resume contract differs from the active config")
    left, right = copy.deepcopy(contract), copy.deepcopy(requested)
    for value in (left, right):
        for field in ("epochs", "max_steps"):
            value["recipe"].pop(field, None)
    if not rc._same(left, right):
        raise ValueError("Only the training horizon may change in the resume contract")
    horizon = horizon_record(checkpoint, config)
    if not rc._same(config["training"].get("resume_horizon"), horizon):
        raise ValueError("Resume horizon/config lineage is missing or changed")
    golden_state = checkpoint / "golden_callback_state.json"
    if not golden_state.is_file():
        raise ValueError(
            "Golden-enabled resume requires saved golden_callback_state.json"
        )
    # Bind optimizer/scheduler/RNG/selector state at validation, then recheck it
    # before training/export. Never load pickle files in this read-only check.
    files = [
        checkpoint / name
        for name in (
            "optimizer.pt",
            "scheduler.pt",
            "trainer_state.json",
            "golden_callback_state.json",
        )
    ]
    files.extend(sorted(checkpoint.glob("rng_state*.pth")))
    return {
        **verified,
        "checkpoint": str(checkpoint.resolve()),
        "continuation": horizon,
        "state_files_sha256": {p.name: rc.file_sha256(p) for p in files},
    }


def verify_export_lineage(config_path: Path, checkpoint: Path) -> dict:
    """Verify every physical resume edge; original source runs stay immutable."""
    config_path, checkpoint = config_path.resolve(), checkpoint.resolve()
    supplied_config_sha256 = rc.file_sha256(config_path)
    current_config, current = load_yaml(config_path), checkpoint
    hops, seen = [], set()
    initial_config = current_config
    for _ in range(64):
        if current in seen:
            raise ValueError("Cycle in official resume lineage")
        seen.add(current)
        bound_path, bound, metadata = source_config(current)
        if current == checkpoint and supplied_config_sha256 != metadata.get(
            "training_config_sha256"
        ):
            raise ValueError(
                "Supplied export config does not match the checkpoint config SHA256"
            )
        if not rc._same(bound, current_config):
            raise ValueError("Checkpoint config differs from export resume config")
        dataset = resolve_path(bound["run"]["dataset_dir"], training_root())
        contract = rc.build_resume_contract(
            bound, dataset, effective_batch=rc._lineage_batch(bound)
        )
        rc._verify_lineage_metadata(metadata, bound, contract, dataset)
        rc._verify_source_inventory(current, metadata)
        if not enabled(bound):
            if (
                rc._resume_source(bound) is not None
                or metadata.get("resume_state")
                or metadata.get("resume_from_checkpoint")
            ):
                raise ValueError(
                    "Official continuation must terminate at a fresh checkpoint"
                )
            preparation = rc._read_object(bound_path.parent / "preparation_report.json")
            if preparation.get("training_config_sha256") != rc.file_sha256(
                bound_path
            ) or not preparation.get("model_files"):
                raise ValueError("Original prepared model/config identity changed")
            return {
                "verified": True,
                "resumed": bool(hops),
                "hops": hops,
                "original_config": str(bound_path),
                "original_config_sha256": rc.file_sha256(bound_path),
                "original_horizon": _horizon(bound),
                "extended_horizon": _horizon(initial_config),
            }
        source = rc._resume_source(bound)
        report = verify_continuation(source, bound)
        if metadata.get("resume_from_checkpoint") != str(source) or not rc._same(
            metadata.get("resume_state"), report
        ):
            raise ValueError(
                "Saved continuation state/source differs from verified physical lineage"
            )
        hops.append(
            {
                "checkpoint": str(current),
                "training_config": str(bound_path),
                "training_config_sha256": rc.file_sha256(bound_path),
                "resume_state": report,
                "verification_mode": "physical_source",
            }
        )
        _, current_config, _ = source_config(source)
        current = source
    raise ValueError("Official resume lineage exceeds the 64-hop safety limit")
