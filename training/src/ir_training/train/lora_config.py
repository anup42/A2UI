from __future__ import annotations

from typing import Any

from ir_training.models.base import ModelAdapter


def build_lora_config(adapter: ModelAdapter, cfg: dict[str, Any]):
    try:
        from peft import LoraConfig  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency failure path
        raise RuntimeError("Install training/requirements-training.txt before building LoRA configs.") from exc

    target_modules = cfg.get("target_modules", "auto")
    if target_modules == "auto" or not target_modules:
        target_modules = adapter.default_lora_targets()
    return LoraConfig(
        r=int(cfg.get("r", 16)),
        lora_alpha=int(cfg.get("alpha", 32)),
        lora_dropout=float(cfg.get("dropout", 0.05)),
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
