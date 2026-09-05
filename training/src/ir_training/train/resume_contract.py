"""Resume means restoring the same optimizer, data and recipe, not warm starting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


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
