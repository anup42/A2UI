"""Resume means restoring the same optimizer, data and recipe, not warm starting."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ir_training.common.config import load_yaml, resolve_path, training_root
from ir_training.common.progress import Progress, log


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def build_resume_contract(config: dict[str, Any], dataset_dir: Path, *, effective_batch: int) -> dict[str, Any]:
    training = config.get("training") or {}
    fields = {name: training.get(name, default) for name, default in {
        "method": "lora_sft", "learning_rate": 2e-4, "weight_decay": 0.0,
        "seed": 42, "max_seq_length": 8192, "epochs": 2, "max_steps": None,
        "warmup_ratio": 0.03, "warmup_steps": None, "max_grad_norm": 1.0,
        "per_device_train_batch_size": 1, "gradient_accumulation_steps": 16,
    }.items()}
    return {
        "version": 1, "recipe": fields, "effective_batch_size": effective_batch,
        "model_id": (config.get("model") or {}).get("model_id"),
        "lora": config.get("lora") or {}, "qat": config.get("qat") or {},
        "splits": {name: file_sha256(dataset_dir / f"{name}.jsonl") for name in ("train", "val") if (dataset_dir / f"{name}.jsonl").is_file()},
        "loss_normalization": "mean_supervised_tokens_per_microbatch_then_mean_microbatches",
    }


def verify_resume_contract(checkpoint: Path, expected: dict[str, Any]) -> dict[str, Any]:
    required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "training_metadata.json")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if not list(checkpoint.glob("rng_state*.pth")):
        missing.append("rng_state*.pth")
    recipe = expected.get("recipe") or {}
    world_size = expected["effective_batch_size"] // (recipe["per_device_train_batch_size"] * recipe["gradient_accumulation_steps"])
    if world_size > 1:
        missing.extend(f"rng_state_{rank}.pth" for rank in range(world_size) if not (checkpoint / f"rng_state_{rank}.pth").is_file())
    if missing:
        raise ValueError(f"Cannot resume without saved optimizer/scheduler/RNG/provenance: {missing}. Use a new run for an intentional weight-only initialization.")
    metadata = json.loads((checkpoint / "training_metadata.json").read_text(encoding="utf-8"))
    actual = metadata.get("resume_contract")
    if actual != expected:
        changed = sorted(key for key in expected if not isinstance(actual, dict) or actual.get(key) != expected[key])
        raise ValueError(f"Resume contract is absent or changed: {changed}. Keep the original data, batch and schedule; use a new run for a changed recipe.")
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if int(state.get("global_step", 0)) <= 0:
        raise ValueError("Resume checkpoint has no completed optimizer step.")
    manifests = metadata.get("adapter_checkpoints") or []
    bound = False
    for record in manifests:
        files = record.get("files") or []
        if files and all((checkpoint / item["path"]).is_file() and file_sha256(checkpoint / item["path"]) == item.get("sha256") for item in files):
            bound = True
            break
    if not bound:
        raise ValueError("Resume model/adapter files do not match a recorded checkpoint manifest.")
    return {"verified": True, "global_step": int(state["global_step"]), "metadata_sha256": file_sha256(checkpoint / "training_metadata.json")}


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a provenance object: {path}")
    return value


def _same(left: Any, right: Any) -> bool:
    # Unlike Python equality, distinguish bool/int and int/float in provenance.
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _config_changes(original: Any, actual: Any, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    if isinstance(original, dict) and isinstance(actual, dict):
        changes = []
        for key in original.keys() | actual.keys():
            if key not in original or key not in actual:
                changes.append((*prefix, key))
            else:
                changes.extend(_config_changes(original[key], actual[key], (*prefix, key)))
        return changes
    return [] if _same(original, actual) else [prefix]


def _resume_source(config: dict[str, Any]) -> Path | None:
    value = (config.get("training") or {}).get("resume_from_checkpoint")
    if value is None:
        value = config.get("resume_from_checkpoint")
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Resume source must be an explicit checkpoint path")
    return resolve_path(value, training_root()).resolve()


def _bound_config(checkpoint: Path, metadata: dict[str, Any], original: Path,
                  explicit: Path | None = None) -> Path:
    digest = metadata.get("training_config_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Selected checkpoint does not match a bound training config SHA256")
    candidates = []
    if explicit is not None:
        if not explicit.is_file() or file_sha256(explicit) != digest:
            raise ValueError("Explicit training config does not match the checkpoint config SHA256")
        candidates.append(explicit)
    recorded = metadata.get("config_path")
    if recorded:
        path = resolve_path(recorded, training_root())
        if path.is_file():
            candidates.append(path)
    snapshot = checkpoint / "training_config.yaml"
    if snapshot.is_file():
        candidates.append(snapshot)
    # Every available recorded copy must agree. Never hide a mutated config by
    # silently falling back to a second, still-matching copy.
    for path in candidates:
        if file_sha256(path) != digest:
            raise ValueError(f"Recorded checkpoint training config changed: {path}")
    if file_sha256(original) == digest:
        return original
    if candidates:
        return candidates[0].resolve()
    raise ValueError("Checkpoint does not match the original training_config.yaml and its bound resume config is missing")


def _lineage_batch(config: dict[str, Any]) -> int:
    training, runtime = config.get("training") or {}, config.get("runtime") or {}
    values = [training.get("per_device_train_batch_size", 1), training.get("gradient_accumulation_steps", 16)]
    world = runtime.get("world_size", (runtime.get("gpu_profile") or {}).get("world_size"))
    expected = training.get("expected_effective_batch_size")
    for value in (*values, world, expected):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError("Resume effective batch and GPU count must be positive integers")
    microbatch_total = values[0] * values[1]
    effective = expected if expected is not None else microbatch_total * (world or 1)
    if effective % microbatch_total or (world is not None and effective != microbatch_total * world):
        raise ValueError("Resume effective batch differs from the original GPU/microbatch configuration")
    return effective


def _verify_lineage_metadata(metadata: dict[str, Any], config: dict[str, Any], expected: dict[str, Any], dataset: Path) -> None:
    if not _same(metadata.get("resume_contract"), expected):
        raise ValueError("Resume contract differs from original model/data/LoRA/training recipe/effective batch")
    if not _same(metadata.get("effective_batch_size"), expected["effective_batch_size"]):
        raise ValueError("Resume metadata effective batch differs from the original configuration")
    if not _same(metadata.get("training"), config.get("training") or {}):
        raise ValueError("Resume metadata training recipe differs from its bound config")
    if not _same(metadata.get("lora"), config.get("lora") or {}):
        raise ValueError("Resume metadata LoRA settings differ from its bound config")
    model = dict(config.get("model") or {})
    recorded_model = dict(metadata.get("model") or {})
    aliases = {"bf16": "bfloat16", "fp16": "float16", "half": "float16", "fp32": "float32", "full": "float32"}
    for entry in (model, recorded_model):
        dtype = str(entry.get("dtype", "bfloat16")).lower()
        entry["dtype"] = aliases.get(dtype, dtype)
    if not _same(recorded_model, model):
        raise ValueError("Resume metadata model differs from its bound config")
    if not isinstance(metadata.get("dataset_dir"), str) or resolve_path(metadata["dataset_dir"], training_root()) != dataset:
        raise ValueError("Resume metadata dataset source differs from its bound config")


def _verify_source_inventory(checkpoint: Path, metadata: dict[str, Any]) -> None:
    """Keep full weight/tokenizer binding, not only the old resume manifest subset."""
    from ir_training.train.callbacks import _checkpoint_adapter_manifest

    saved = metadata.get("checkpoint_adapter_files")
    if not isinstance(saved, list) or not saved:
        raise ValueError("Resume source lacks its full checkpoint/tokenizer file inventory")
    expected = {}
    for item in saved:
        name = item.get("path") if isinstance(item, dict) else None
        if not isinstance(name, str) or Path(name).name != name or name in expected:
            raise ValueError("Unsafe or duplicate file in resume source inventory")
        path = (checkpoint / name).resolve()
        if not path.is_relative_to(checkpoint) or not path.is_file():
            raise ValueError("Missing or escaping file in resume source inventory")
        expected[name] = (item.get("sha256"), item.get("size_bytes"))
    actual = _checkpoint_adapter_manifest(checkpoint, role="resume_export")
    if {item["path"]: (item["sha256"], item["size"]) for item in actual["files"]} != expected:
        raise ValueError("Resume source checkpoint weights/tokenizer inventory changed")


def _resume_source_directory_absent(recorded_source: str, source: Path) -> bool:
    # Inspect the saved, unresolved path: a dangling symlink or a damaged source
    # directory is not retention. Never turn permission/I/O errors into absence.
    path = Path(recorded_source)
    if not path.is_absolute():
        path = training_root() / path
    try:
        path.lstat()
    except FileNotFoundError:
        try:
            source.lstat()
        except FileNotFoundError:
            return True
    return False


def _verify_retained_resume_evidence(*, checkpoint: Path, metadata: dict[str, Any],
                                     config: dict[str, Any], config_path: Path,
                                     source: Path, expected: dict[str, Any]) -> dict[str, Any]:
    """Export-only compatibility with Trainer retention; never resume training.

    The caller has already checked the full config delta, resume contract and
    current train/val bytes. Trust the surviving training record's historical
    verification, not a fabricated reconstruction of deleted metadata.
    """
    state = metadata["resume_state"]
    step, digest = state.get("global_step"), state.get("metadata_sha256")
    if type(step) is not int or step <= 0:
        raise ValueError("Missing positive integer resume_state.global_step for retained-source export")
    if (not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or len(set(digest)) == 1):
        raise ValueError("Missing or malformed resume_state.metadata_sha256 for retained-source export")
    # Trainer retention deletes numbered checkpoints in this run's output, not
    # arbitrary paths, unavailable volumes, or weight-only best/final folders.
    output = (config.get("run") or {}).get("output_dir")
    if (not isinstance(output, str) or not output.strip()
            or source.parent != resolve_path(output, training_root()).resolve()
            or source.name != f"checkpoint-{step}" or not source.parent.is_dir()):
        raise ValueError("Missing resume source path/step does not identify a retained Trainer checkpoint in this run")
    current_step = metadata.get("checkpoint_step")
    if type(current_step) is not int or current_step < step:
        raise ValueError("Retained-source export needs a surviving checkpoint at or after the recorded resume step")
    if set(expected.get("splits", {})) != {"train", "val"}:
        raise ValueError("Retained-source export requires both prepared train and val hashes")
    _verify_source_inventory(checkpoint, metadata)
    names = {item["path"] for item in metadata["checkpoint_adapter_files"]}
    if not any(name.startswith("tokenizer") for name in names):
        raise ValueError("Retained-source export requires a hash-bound surviving tokenizer inventory")
    # Preserve optional stronger evidence if present; the legacy writer stored
    # the path separately in metadata, so it cannot be mandatory in resume_state.
    if "checkpoint" in state and state["checkpoint"] != str(source):
        raise ValueError("Recorded resume_state checkpoint path differs from the missing source")
    return {"checkpoint": str(checkpoint), "training_config": str(config_path),
            "training_config_sha256": file_sha256(config_path), "resume_source": str(source),
            "resume_state": dict(state), "verification_mode": "saved_resume_evidence",
            "source_files_rechecked": False, "source_metadata_rehashed": False,
            "surviving_metadata_sha256": file_sha256(checkpoint / "training_metadata.json"),
            "surviving_checkpoint_files_rechecked": True,
            "prepared_split_sha256": dict(expected["splits"])}


def resolve_export_training_lineage(preparation_config: Path, checkpoint: Path, *,
                                    training_config: Path | None = None) -> dict[str, Any]:
    """Resolve the actual config without rewriting any saved provenance.

    Preparation stays bound to the original config. Each resume edge must keep
    every setting except the resume pointer identical and prove the saved source
    contract, metadata digest and optimizer step. If retention removed the whole
    source, explicitly report reliance on surviving verified resume evidence.
    No model is instantiated and no provenance files are modified.
    """
    original_path, checkpoint = preparation_config.resolve(), checkpoint.resolve()
    original = load_yaml(original_path)
    original_sha = file_sha256(original_path)
    metadata = _read_object(checkpoint / "training_metadata.json")
    actual_path = _bound_config(checkpoint, metadata, original_path, training_config)
    preparation = _read_object(original_path.parent / "preparation_report.json")
    if preparation.get("training_config_sha256") != original_sha or not preparation.get("model_files"):
        raise ValueError("Missing or mismatched original model preparation report")
    if _resume_source(original) is not None:
        raise ValueError("Preparation must remain bound to the original, non-resumed training config")
    actual = load_yaml(actual_path)
    hops = []
    result = {"config": actual, "metadata": metadata, "preparation": preparation,
              "training_config": str(actual_path), "training_config_sha256": file_sha256(actual_path),
              "preparation_config": str(original_path), "preparation_config_sha256": original_sha,
              "resume_lineage": {"verified": True, "resumed": False, "hops": hops}}
    has_resume = _resume_source(actual) is not None or bool(metadata.get("resume_from_checkpoint") or metadata.get("resume_state"))
    if result["training_config_sha256"] == original_sha and not has_resume:
        return result
    result["resume_lineage"]["resumed"] = True
    dataset = resolve_path(original["run"]["dataset_dir"], training_root())
    if not (dataset / "train.jsonl").is_file():
        raise ValueError("Original training data is missing for resume-lineage verification")
    with Progress("Verify export resume lineage (data and source checkpoint hashes)", unit="stage"):
        expected = build_resume_contract(original, dataset, effective_batch=_lineage_batch(original))
        visited = set()
        current, current_meta, current_config, current_path = checkpoint, metadata, actual, actual_path
        for _ in range(64):
            if current in visited:
                raise ValueError("Cycle in checkpoint resume lineage")
            visited.add(current)
            changes = _config_changes(original, current_config)
            allowed = {("training", "resume_from_checkpoint"), ("resume_from_checkpoint",)}
            unrelated = sorted(".".join(path) for path in changes if path not in allowed)
            if unrelated:
                raise ValueError(f"Unrelated config changes in resume lineage: {unrelated}")
            _verify_lineage_metadata(current_meta, current_config, expected, dataset)
            source = _resume_source(current_config)
            if source is None:
                if file_sha256(current_path) != original_sha or current_meta.get("resume_from_checkpoint") or current_meta.get("resume_state"):
                    raise ValueError("Resume lineage does not terminate at the original prepared config")
                return result
            recorded_source = current_meta.get("resume_from_checkpoint")
            if not isinstance(recorded_source, str) or resolve_path(recorded_source, training_root()).resolve() != source:
                raise ValueError("Resume source differs between metadata and bound config")
            if source in visited:
                raise ValueError("Cycle in checkpoint resume lineage")
            recorded_state = current_meta.get("resume_state")
            if not isinstance(recorded_state, dict) or recorded_state.get("verified") is not True:
                raise ValueError("Missing verified resume_state provenance")
            if _resume_source_directory_absent(recorded_source, source):
                hop = _verify_retained_resume_evidence(checkpoint=current, metadata=current_meta,
                    config=current_config, config_path=current_path, source=source, expected=expected)
                if not _resume_source_directory_absent(recorded_source, source):
                    raise ValueError("Resume source reappeared during export verification; retry for full physical-source checks")
                hops.append(hop)
                result["resume_lineage"].update({"verification_mode": "saved_resume_evidence",
                    "physical_chain_complete": False, "missing_sources": [str(source)]})
                log(f"WARNING: Resume source is absent: {source}; export relies on saved verified resume evidence. "
                    "Surviving config/data/weight/tokenizer hashes checked; deleted source bytes cannot be rechecked.")
                return result
            source_meta = _read_object(source / "training_metadata.json")
            if recorded_state.get("metadata_sha256") != file_sha256(source / "training_metadata.json"):
                raise ValueError("Resume source metadata changed since training resumed")
            _verify_source_inventory(source, source_meta)
            verified = verify_resume_contract(source, expected)
            if any(not _same(recorded_state.get(key), value) for key, value in verified.items()):
                raise ValueError("Recorded resume state differs from the verified source checkpoint")
            if not _same(source_meta.get("checkpoint_step"), verified["global_step"]):
                raise ValueError("Resume source optimizer step differs from checkpoint metadata")
            source_config_path = _bound_config(source, source_meta, original_path)
            hops.append({"checkpoint": str(current), "training_config": str(current_path),
                         "training_config_sha256": file_sha256(current_path), "resume_source": str(source),
                         "source_config": str(source_config_path), "source_config_sha256": file_sha256(source_config_path),
                         "resume_state": verified, "verification_mode": "physical_source",
                         "source_files_rechecked": True, "source_metadata_rehashed": True})
            current, current_meta, current_path = source, source_meta, source_config_path
            current_config = load_yaml(current_path)
        raise ValueError("Resume lineage exceeds the 64-hop safety limit")
