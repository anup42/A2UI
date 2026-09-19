from __future__ import annotations

from typing import Any

from ir_training.models.base import ModelAdapter
from ir_training.train.lora_targets import resolve_lora_config_targets

__all__ = ["build_lora_config", "load_retained_mobile_lora_qparams", "resolve_lora_config_targets"]


def load_retained_mobile_lora_qparams(
    model_cfg: dict[str, Any], qat_cfg: dict[str, Any], *, verified_seed: dict[str, Any],
):
    """Enable the narrow target resolver only for the verified official seed."""
    from ir_training.qat.mobile_training_seed import OFFICIAL_MOBILE_MODEL_ID

    if (model_cfg.get("model_id") != OFFICIAL_MOBILE_MODEL_ID
            or not qat_cfg.get("enabled") or qat_cfg.get("scale_mode") != "retained_mobile"):
        return None
    if verified_seed.get("required") is not True or verified_seed.get("verified") is not True:
        raise ValueError("Retained mobile LoRA requires a verified official mobile seed")
    contract = qat_cfg.get("mobile_qparams_contract") or model_cfg.get("mobile_qparams_contract")
    if not contract:
        raise ValueError("Retained mobile LoRA requires the seed-bound mobile_qparams_contract")
    from ir_training.qat.mobile_qparams import MobileQParams

    qparams = MobileQParams(contract)
    seed_qparams = verified_seed.get("mobile_qparams") or {}
    hashes = ("contract_sha256", "scale_storage_sha256", "inventory_sha256")
    if (seed_qparams.get("verified") is not True
            or any(not seed_qparams.get(key) or seed_qparams[key] != qparams.report.get(key) for key in hashes)):
        raise ValueError("LoRA qparams differ from the verified mobile seed contract")
    return qparams


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
