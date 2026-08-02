from __future__ import annotations

from typing import Any

from ir_training.models.base import ModelAdapter


def build_lora_config(adapter: ModelAdapter, cfg: dict[str, Any]):
    try:
        from peft import LoraConfig  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency failure path
        raise RuntimeError("Install training/requirements-training.txt before building LoRA configs.") from exc

    target_modules = cfg.get("target_modules", "auto")
    use_peft_default = str(target_modules).strip().lower() in {
        "peft-default",
        "peft_default",
        "peft",
    }
    if target_modules == "auto" or not target_modules:
        target_modules = adapter.default_lora_targets()
    kwargs: dict[str, Any] = {
        "r": int(cfg.get("r", 16)),
        "lora_alpha": int(cfg.get("alpha", 32)),
        "lora_dropout": float(cfg.get("dropout", 0.05)),
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    if not use_peft_default:
        kwargs["target_modules"] = target_modules
    modules_to_save = cfg.get("modules_to_save")
    if isinstance(modules_to_save, list) and modules_to_save:
        kwargs["modules_to_save"] = [str(item) for item in modules_to_save]
    if "ensure_weight_tying" in cfg:
        kwargs["ensure_weight_tying"] = bool(cfg.get("ensure_weight_tying"))
    return LoraConfig(**kwargs)
