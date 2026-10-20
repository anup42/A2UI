"""Explicit opt-in objectives for the shared QAT post-training/export workflow."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any


def objective(config: dict[str, Any]) -> str:
    value = (config.get("grpo") or {}).get("objective", "grpo")
    if value not in ("grpo", "grpo_sft", "dpo"):
        raise ValueError("QAT objective must be grpo, grpo_sft, or dpo")
    return value


def _positive(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be positive and finite")


def validate_objective(config: dict[str, Any]) -> str:
    mode = objective(config)
    grpo = config.get("grpo") or {}
    if mode != "grpo_sft" and grpo.get("rehearsal") is not None:
        raise ValueError("rehearsal settings require objective=grpo_sft")
    if mode != "dpo" and grpo.get("dpo") is not None:
        raise ValueError("DPO settings require objective=dpo")
    if mode == "grpo_sft":
        settings = grpo.get("rehearsal")
        if not isinstance(settings, dict) or set(settings) != {"weight"}:
            raise ValueError("GRPO SFT rehearsal requires explicit weight only")
        _positive(settings["weight"], "rehearsal weight")
        if grpo.get("loss_type", "dr_grpo") != "dr_grpo":
            raise ValueError("Rehearsal normalization is reviewed for dr_grpo only")
    elif mode == "dpo":
        settings = grpo.get("dpo")
        if not isinstance(settings, dict) or set(settings) != {"beta", "preferences", "preferences_sha256"}:
            raise ValueError("DPO requires beta and a hash-bound preference file")
        _positive(settings["beta"], "DPO beta")
        digest = settings["preferences_sha256"]
        if (not isinstance(settings["preferences"], str) or not settings["preferences"]
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("DPO preferences must have a path and SHA256 identity")
        if grpo.get("rollout_qat_weight_cache", False):
            raise ValueError("DPO has no online rollout weight cache")
    return mode


def implementation_identity(mode: str) -> dict[str, str]:
    names = ["qat_objectives.py", "qat_posttrain_data.py"]
    names.append("qat_dpo.py")  # Also supplies the bounded-memory rehearsal log-probability primitive.
    if mode == "grpo_sft":
        names.append("qat_rehearsal.py")
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_text(encoding="utf-8").encode()).hexdigest()
            for name in names}
