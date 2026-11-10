"""Verified weight-only initialization for a new official-mobile SFT run."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ir_training.common.config import load_yaml, training_root
from ir_training.qat.fake_quant import QATSpec
from ir_training.qat.numeric_preflight import OFFICIAL_MOBILE_WORKFLOW
from ir_training.train.qat_grpo_contract import file_identity

KEY = "sft_adapter_initialization"
MODE = "sft_adapter"
ZERO_EVIDENCE = {
    "required": False,
    "reason": "continued_from_trained_sft_adapter",
    "verified_zero_delta": False,
}
LORA_FIELDS = ("r", "alpha", "dropout", "target_modules", "modules_to_save")


def initialization(config: dict[str, Any]) -> dict[str, Any] | None:
    training = config.get("training") or {}
    if KEY not in training:
        return None
    record = training[KEY]
    if (
        (config.get("run") or {}).get("purpose") != OFFICIAL_MOBILE_WORKFLOW
        or training.get("method") != "qat_lora_sft"
        or training.get("refuse_resume") is not True
        or training.get("resume_from_checkpoint")
        or config.get("resume_from_checkpoint")
        or training.get("resume_policy") is not None
    ):
        raise ValueError("Weight-only SFT initialization requires a fresh official-mobile QAT LoRA run")
    if (
        not isinstance(record, dict)
        or set(record) != {"checkpoint", "source_config", "files"}
        or any(not isinstance(record.get(key), str) or not Path(record[key]).is_absolute()
               for key in ("checkpoint", "source_config"))
        or not isinstance(record.get("files"), dict)
    ):
        raise ValueError("Invalid official-mobile SFT initialization binding")
    return record


def bind_source(checkpoint: Path, source_config: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = checkpoint.resolve(strict=True)
    source_config = source_config.resolve(strict=True)
    source = load_yaml(source_config)
    if (
        (source.get("run") or {}).get("purpose") != OFFICIAL_MOBILE_WORKFLOW
        or (source.get("training") or {}).get("method") != "qat_lora_sft"
        or KEY in (source.get("training") or {})
        or (source.get("qat") or {}).get("scale_mode") != "retained_mobile"
    ):
        raise ValueError("Source must be an original official-mobile retained-scale QAT LoRA SFT run")
    if (checkpoint / "adapter_model.bin").exists():
        raise ValueError("Weight-only SFT initialization requires one safetensors adapter weight file")
    paths = {
        "training_config": source_config,
        "training_metadata": checkpoint / "training_metadata.json",
        "adapter_config": checkpoint / "adapter_config.json",
        "adapter_weights": checkpoint / "adapter_model.safetensors",
    }
    return {"checkpoint": str(checkpoint), "source_config": str(source_config),
            "files": {name: file_identity(path) for name, path in paths.items()}}, source


def require_source(config: dict[str, Any]) -> dict[str, Any]:
    record = initialization(config)
    if record is None:
        raise ValueError("Missing official-mobile SFT initialization binding")
    checkpoint, source_config = Path(record["checkpoint"]), Path(record["source_config"])
    rebound, source = bind_source(checkpoint, source_config)
    if rebound != record:
        raise ValueError("Official-mobile SFT initialization files changed after binding")
    model, source_model = config.get("model") or {}, source.get("model") or {}
    model_fields = (
        "model_id", "model_source", "tokenizer_source", "mobile_training_seed_manifest", "mobile_qparams_contract",
    )
    if any(model.get(key) != source_model.get(key) for key in model_fields):
        raise ValueError("Official-mobile SFT initialization requires the original seed, tokenizer and qparams")
    if QATSpec.from_config(config).to_dict() != QATSpec.from_config(source).to_dict():
        raise ValueError("Official-mobile SFT initialization requires the original QAT contract")
    if any((config.get("lora") or {}).get(key) != (source.get("lora") or {}).get(key) for key in LORA_FIELDS):
        raise ValueError("Official-mobile SFT initialization requires the original LoRA contract")

    from ir_training.export.merge_lora import _training_provenance

    provenance = _training_provenance(checkpoint, source_config, base=training_root())
    report = provenance["training_run_metadata"]
    if report.get("verified") is not True:
        failed = [name for name, passed in report.get("checks", {}).items() if not passed]
        raise ValueError(f"Official-mobile SFT initialization provenance failed: {failed}")
    return record


def require_loaded_adapter(record: dict[str, Any], loaded: dict[str, Any]) -> None:
    nonzero = loaded.get("nonzero_adapter_pairs")
    if not (
        loaded.get("checkpoint") == record["checkpoint"]
        and loaded.get("adapter_sha256") == record["files"]["adapter_weights"]["sha256"]
        and loaded.get("adapter_pair_count") == 205
        and loaded.get("finite_adapter_pairs") == 205
        and type(nonzero) is int
        and 0 < nonzero <= 205
    ):
        raise ValueError("Loaded SFT adapter does not match the bound finite 205-projection adapter")


def require_loaded_weights(model: Any, record: dict[str, Any], loaded: dict[str, Any]) -> None:
    from peft import get_peft_model_state_dict
    from safetensors import safe_open

    require_loaded_adapter(record, loaded)
    tensors = get_peft_model_state_dict(model, adapter_name=loaded["adapter_name"], save_embedding_layers=False)
    expected = record["files"]["adapter_weights"]
    with safe_open(expected["path"], framework="pt", device="cpu") as saved:
        if len(tensors) != 2 * 205 or tensors.keys() != set(saved.keys()):
            raise ValueError("Loaded SFT adapter tensor keys do not match the exact 205 saved LoRA pairs")
        for name, weight in tensors.items():
            original = saved.get_tensor(name)
            live = weight.detach().cpu()
            if not live.equal(original):
                raise ValueError(f"Loaded SFT adapter tensor differs from the checkpoint: {name}")
    if any(file_identity(identity["path"]) != identity for identity in record["files"].values()):
        raise ValueError("SFT initialization files changed during loading")


def verify_numeric_initialization(config: dict[str, Any], numeric: dict[str, Any]) -> bool:
    record = require_source(config)
    require_loaded_adapter(record, numeric.get("source_adapter") or {})
    return (
        numeric.get("adapter_initialization_mode") == MODE
        and numeric.get("zero_adapter_initialization") == ZERO_EVIDENCE
        and numeric.get("resume_state") is None
        and numeric.get("resume_adapter") is None
    )
