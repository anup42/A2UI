"""Fail-closed continuation of a numbered, full-parameter QAT Trainer checkpoint.

The source run is read-only. A new run regenerates the prepared data, verifies
its bytes against the source, and restores Trainer's model, optimizer,
scheduler, RNG and Golden selector state. Weight-only best/final exports are
never resume sources.
"""
from __future__ import annotations

import copy
import math
import os
import re
from pathlib import Path
from typing import Any

from ir_training.common.config import load_yaml, resolve_path, training_root
from ir_training.train import resume_contract as rc
from ir_training.train.recipe import optimizer_steps

POLICY = "full_parameter_qat_continuation_v1"
WORKFLOW = "e2b_all_parameter_qat_v1"


def enabled(config: dict[str, Any]) -> bool:
    training = config.get("training") or {}
    policy = training.get("resume_policy")
    if policy is None:
        return False
    if policy != POLICY:
        raise ValueError(f"Unknown training.resume_policy: {policy!r}")
    if (config.get("run") or {}).get("purpose") != WORKFLOW or training.get("method") != "full_finetune_qat":
        raise ValueError("Full-QAT continuation policy cannot be used for another training workflow")
    if training.get("refuse_resume") is not False or rc._resume_source(config) is None:
        raise ValueError("Full-QAT continuation requires an explicit checkpoint and refuse_resume=false")
    return True


def source_config(checkpoint: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    checkpoint = checkpoint.resolve(strict=True)
    if not checkpoint.is_dir():
        raise ValueError("Full-QAT resume source must be a checkpoint directory")
    metadata = rc._read_object(checkpoint / "training_metadata.json")
    snapshot = checkpoint / "training_config.yaml"
    recorded = metadata.get("config_path")
    path = resolve_path(recorded, training_root()) if recorded else snapshot
    if not path.is_file():
        path = snapshot
    path = rc._bound_config(checkpoint, metadata, path)
    config = load_yaml(path)
    if ((config.get("run") or {}).get("purpose") != WORKFLOW
            or (config.get("training") or {}).get("method") != "full_finetune_qat"
            or metadata.get("checkpoint_kind") != "full_model"):
        raise ValueError("Resume source is not a full-parameter QAT model checkpoint")
    from ir_training.qat.full_model_contract import validate_full_qat_config

    validate_full_qat_config(config)
    return path, config, metadata


def _horizon(config: dict[str, Any]) -> dict[str, Any]:
    training = config["training"]
    epochs, steps = training.get("epochs", 2), training.get("max_steps")
    if isinstance(epochs, bool) or not isinstance(epochs, (int, float)) or not math.isfinite(epochs) or epochs <= 0:
        raise ValueError("Resume epochs must be positive and finite")
    if steps is not None and (type(steps) is not int or steps <= 0):
        raise ValueError("Resume max_steps must be a positive integer")
    dataset = resolve_path(config["run"]["dataset_dir"], training_root())
    with (dataset / "train.jsonl").open(encoding="utf-8") as stream:
        rows = sum(bool(line.strip()) for line in stream)
    if rows == 0:
        raise ValueError("Resume train split is empty")
    batch = rc._lineage_batch(config)
    micro = training["per_device_train_batch_size"]
    accumulation = training["gradient_accumulation_steps"]
    total = steps if steps is not None else optimizer_steps(
        rows=rows, world_size=batch // (micro * accumulation),
        microbatch=micro, accumulation=accumulation, epochs=epochs,
    )
    return {"epochs": epochs, "max_steps": steps, "total_optimizer_steps": total}


def horizon_record(checkpoint: Path, requested: dict[str, Any]) -> dict[str, Any]:
    path, source, metadata = source_config(checkpoint)
    previous, next_horizon = _horizon(source), _horizon(requested)
    state = rc._read_object(checkpoint / "trainer_state.json")
    step = state.get("global_step")
    if (type(step) is not int or step <= 0 or metadata.get("checkpoint_step") != step
            or checkpoint.name != f"checkpoint-{step}"
            or metadata.get("checkpoint_role") != "trainer_intermediate"):
        raise ValueError("Resume requires a numbered Trainer checkpoint with matching optimizer step")
    if (previous["max_steps"] is None) != (next_horizon["max_steps"] is None):
        raise ValueError("Resume cannot switch between epoch and max_steps horizons")
    if (next_horizon["epochs"] < previous["epochs"]
            or next_horizon["total_optimizer_steps"] < previous["total_optimizer_steps"]
            or next_horizon["total_optimizer_steps"] <= step
            or (previous["max_steps"] is not None and next_horizon["epochs"] != previous["epochs"])):
        raise ValueError("Resume may only keep or increase the training horizon beyond the checkpoint step")
    if state.get("max_steps") is not None and state["max_steps"] != previous["total_optimizer_steps"]:
        raise ValueError("Saved Trainer horizon differs from its bound training config")
    inherited = source["training"].get("resume_horizon")
    if enabled(source):
        if not isinstance(inherited, dict):
            raise ValueError("Resume source lacks its prior horizon lineage")
        original_path = Path(inherited["original_config"])
        original_sha = inherited["original_config_sha256"]
        if not original_path.is_file() or rc.file_sha256(original_path) != original_sha:
            raise ValueError("Original full-QAT config changed or disappeared")
        original = inherited["original"]
        warmup = inherited["scheduler_warmup_steps"]
    else:
        if rc._resume_source(source) is not None or source["training"].get("refuse_resume") is not True:
            raise ValueError("Resume chain must originate in a strict fresh full-QAT run")
        original_path, original, warmup = path, previous, source["training"].get("warmup_steps")
        original_sha = rc.file_sha256(path)
        if warmup is None:
            warmup = math.ceil(previous["total_optimizer_steps"] * source["training"].get("warmup_ratio", 0.03))
    return {
        "policy": POLICY, "source_checkpoint": str(checkpoint.resolve()),
        "source_config": str(path), "source_config_sha256": rc.file_sha256(path),
        "original_config": str(original_path), "original_config_sha256": original_sha,
        "original": original, "previous": previous, "requested": next_horizon,
        "completed_global_step": step, "scheduler_warmup_steps": warmup,
    }


def _state_files(checkpoint: Path, source: dict[str, Any], step: int) -> list[Path]:
    files = [checkpoint / name for name in ("trainer_state.json", "golden_callback_state.json")]
    training = source["training"]
    world = rc._lineage_batch(source) // (
        training["per_device_train_batch_size"] * training["gradient_accumulation_steps"]
    )
    files.extend(checkpoint / f"rng_state_{rank}.pth" for rank in range(world))
    if training.get("distributed_backend", "ddp") == "sharded":
        latest = checkpoint / "latest"
        if not latest.is_file() or latest.is_symlink():
            raise ValueError("Sharded resume requires a regular DeepSpeed latest tag")
        tag = latest.read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"global_step\d+", tag) or int(tag.removeprefix("global_step")) != step:
            raise ValueError("DeepSpeed checkpoint tag does not match Trainer global_step")
        directory = checkpoint / tag
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError("Sharded resume requires the matching DeepSpeed state directory")
        optimizer = sorted(directory.glob("*_optim_states.pt"))
        model = sorted(directory.glob("*_model_states.pt"))
        if len(optimizer) < world or not model:
            raise ValueError("Sharded resume lacks complete optimizer/model state shards")
        files.extend([latest, *optimizer, *model])
    else:
        files.extend(checkpoint / name for name in ("optimizer.pt", "scheduler.pt"))
    if any(not path.is_file() or path.is_symlink() for path in files):
        raise ValueError("Resume checkpoint lacks optimizer/scheduler/RNG/Golden state files")
    return files


def _check_config_delta(source: dict[str, Any], current: dict[str, Any]) -> None:
    source_root = Path(source["run"]["output_dir"]).resolve().parents[1]
    current_root = Path(current["run"]["output_dir"]).resolve().parents[1]
    if source_root == current_root or current_root.is_relative_to(source_root) or source_root.is_relative_to(current_root):
        raise ValueError("Resume requires a fresh, separate output run")

    def relocate(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, str) and (value == str(source_root) or value.startswith(str(source_root) + os.sep)):
            return str(current_root) + value[len(str(source_root)):]
        return value

    left, right = relocate(copy.deepcopy(source)), copy.deepcopy(current)
    left["run"]["id"] = right["run"]["id"]
    # Hardware inventory includes host-local identifiers/CPU availability.
    # Recipe, world size, effective batch and per-rank accumulation remain
    # strictly compared elsewhere; a different H100 allocation is permissible.
    for config in (left, right):
        config.get("runtime", {}).pop("gpu_profile", None)
        config.get("runtime", {}).pop("cuda_visible_devices", None)
    for config in (left, right):
        training = config["training"]
        for name in ("epochs", "max_steps", "warmup_steps", "resume_from_checkpoint", "resume_policy", "resume_horizon", "refuse_resume"):
            training.pop(name, None)
    changes = rc._config_changes(left, right)
    if changes:
        raise ValueError("Unrelated full-QAT resume config changes: " + ", ".join(sorted(".".join(path) for path in changes)))


def verify_continuation(checkpoint: Path, config: dict[str, Any], expected: dict | None = None) -> dict[str, Any]:
    checkpoint = checkpoint.resolve(strict=True)
    if not enabled(config) or rc._resume_source(config) != checkpoint:
        raise ValueError("Full-QAT resume must explicitly identify its checkpoint")
    _, source, metadata = source_config(checkpoint)
    if (config.get("training") or {}).get("distributed_backend", "ddp") != source["training"].get("distributed_backend", "ddp"):
        raise ValueError("Full-QAT resume cannot switch distributed backend")
    if (config["training"].get("zero_stage", 2) != source["training"].get("zero_stage", 2)):
        raise ValueError("Full-QAT resume cannot switch ZeRO stage")
    old_data = resolve_path(source["run"]["dataset_dir"], training_root())
    new_data = resolve_path(config["run"]["dataset_dir"], training_root())
    old_contract = rc.build_resume_contract(source, old_data, effective_batch=rc._lineage_batch(source))
    new_contract = rc.build_resume_contract(config, new_data, effective_batch=rc._lineage_batch(config))
    if set(old_contract["splits"]) != {"train", "val"} or old_contract["splits"] != new_contract["splits"]:
        raise ValueError("Full-QAT resume requires identical prepared train and val bytes")
    for name in ("golden32.jsonl", "golden35.jsonl", "bixby50.jsonl"):
        old_path, new_path = old_data / name, new_data / name
        if not old_path.is_file() or not new_path.is_file() or rc.file_sha256(old_path) != rc.file_sha256(new_path):
            raise ValueError(f"Full-QAT resume requires identical prepared holdout bytes: {name}")
    rc._verify_lineage_metadata(metadata, source, old_contract, old_data)
    rc._verify_source_inventory(checkpoint, metadata)
    if expected is not None and not rc._same(expected, new_contract):
        raise ValueError("Requested full-QAT resume contract differs from active config")
    left, right = copy.deepcopy(old_contract), copy.deepcopy(new_contract)
    for contract in (left, right):
        for name in ("epochs", "max_steps", "warmup_steps"):
            contract["recipe"].pop(name, None)
    if not rc._same(left, right):
        raise ValueError("Only the full-QAT training horizon may change")
    _check_config_delta(source, config)
    horizon = horizon_record(checkpoint, config)
    if not rc._same(config["training"].get("resume_horizon"), horizon):
        raise ValueError("Full-QAT resume horizon lineage is missing or changed")
    if config["training"].get("warmup_steps") != horizon["scheduler_warmup_steps"]:
        raise ValueError("Full-QAT resume changed the original scheduler warmup")
    state_files = _state_files(checkpoint, source, horizon["completed_global_step"])
    return {
        "verified": True, "global_step": horizon["completed_global_step"],
        "metadata_sha256": rc.file_sha256(checkpoint / "training_metadata.json"),
        "checkpoint": str(checkpoint), "continuation": horizon,
        "state_files_sha256": {str(path.relative_to(checkpoint)): rc.file_sha256(path) for path in state_files},
    }


def verify_export_lineage(config_path: Path, checkpoint: Path) -> dict[str, Any]:
    """Recheck the physical resume chain before dense export; no deleted-source fallback."""
    config_path, checkpoint = config_path.resolve(strict=True), checkpoint.resolve(strict=True)
    config = load_yaml(config_path)
    current_meta = rc._read_object(checkpoint / "training_metadata.json")
    if current_meta.get("training_config_sha256") != rc.file_sha256(config_path):
        raise ValueError("Selected full-QAT checkpoint is not bound to its resumed config")
    hops: list[dict[str, Any]] = []
    seen: set[Path] = set()
    while enabled(config):
        source = rc._resume_source(config)
        if source in seen or source == checkpoint or len(hops) >= 64:
            raise ValueError("Cycle or excessive depth in full-QAT resume lineage")
        seen.add(source)
        verified = verify_continuation(source, config)
        if not rc._same(current_meta.get("resume_state"), verified):
            raise ValueError("Selected full-QAT checkpoint records different resume source evidence")
        hops.append({"source_checkpoint": str(source), "resume_state": verified})
        source_path, config, current_meta = source_config(source)
        checkpoint = source
    return {"verified": True, "resumed": True, "physical_chain_complete": True,
            "hops": hops, "original_config": str(source_path),
            "original_config_sha256": rc.file_sha256(source_path)}
