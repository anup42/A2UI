from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ir_training.common.config import resolve_path, training_root
from ir_training.models.hf_loading import load_hf_model


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_lora_adapter(
    base_model_id: str,
    adapter_dir: str | Path,
    output_dir: str | Path,
    *,
    model_loader: str = "auto_causal_lm",
    dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    processor_model_id: str | None = None,
    training_config_path: str | Path | None = None,
) -> Path:
    """Merge an adapter while recording what this operation does not prove.

    A merge from a QAT-derived base remains a floating-point Hugging Face model.
    It is not packed INT4 and this function does not perform continued QAT or
    modify an MTP assistant.
    """

    try:
        from peft import PeftModel  # type: ignore
        from transformers import AutoProcessor, AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency failure path
        raise RuntimeError("Install the training requirements before merging LoRA adapters.") from exc

    if not str(base_model_id).strip():
        raise ValueError("base_model_id is required")
    if not str(adapter_dir).strip():
        raise ValueError("adapter_dir is required")
    if not str(output_dir).strip():
        raise ValueError("output_dir is required")

    base = training_root()
    adapter_path = resolve_path(adapter_dir, base)
    out_dir = resolve_path(output_dir, base)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Missing LoRA adapter directory: {adapter_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_config: dict[str, Any] = {
        "model_loader": model_loader,
        "dtype": dtype,
        "device_map": "auto",
        "trust_remote_code": trust_remote_code,
        "load_in_4bit": False,
    }
    model = load_hf_model(base_model_id, model_config)
    model = PeftModel.from_pretrained(model, str(adapter_path))
    merged = model.merge_and_unload()
    merged.save_pretrained(str(out_dir), safe_serialization=True)

    processor_saved = False
    try:
        processor = AutoProcessor.from_pretrained(str(adapter_path), trust_remote_code=trust_remote_code)
        processor.save_pretrained(str(out_dir))
        processor_saved = True
    except Exception:  # noqa: BLE001 - adapter processor metadata is optional
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=trust_remote_code)
            tokenizer.save_pretrained(str(out_dir))
            processor_saved = True
        except Exception:  # noqa: BLE001 - fall back to the declared base below
            processor_saved = False
    if not processor_saved:
        processor_source = processor_model_id or base_model_id
        try:
            processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=trust_remote_code)
            processor.save_pretrained(str(out_dir))
        except Exception:  # noqa: BLE001 - some model families expose only a tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=trust_remote_code)
            tokenizer.save_pretrained(str(out_dir))

    training_provenance: dict[str, Any] = {
        "training_config": None,
        "training_config_sha256": None,
        "training_method": None,
        "qat_enabled": None,
        "qat_profile": None,
    }
    if training_config_path is not None:
        from ir_training.common.config import load_yaml

        resolved_training_config = resolve_path(training_config_path, base)
        if not resolved_training_config.is_file():
            raise FileNotFoundError(
                f"Missing training config for merge provenance: {resolved_training_config}"
            )
        training_bytes = resolved_training_config.read_bytes()
        training_config = load_yaml(resolved_training_config)
        training_section = (
            training_config.get("training")
            if isinstance(training_config.get("training"), dict)
            else {}
        )
        qat_section = (
            training_config.get("qat")
            if isinstance(training_config.get("qat"), dict)
            else {}
        )
        training_provenance = {
            "training_config": str(resolved_training_config),
            "training_config_sha256": hashlib.sha256(training_bytes).hexdigest(),
            "training_method": training_section.get("method"),
            "qat_enabled": bool(qat_section.get("enabled", False)),
            "qat_profile": qat_section.get("profile"),
        }

    adapter_files = []
    for candidate in sorted(adapter_path.glob("adapter*")):
        if not candidate.is_file():
            continue
        adapter_files.append(
            {
                "path": str(candidate),
                "size": int(candidate.stat().st_size),
                "sha256": _sha256_file(candidate),
            }
        )

    merged_model_files = []
    merged_candidates = list(out_dir.glob("model*.safetensors"))
    index_path = out_dir / "model.safetensors.index.json"
    if index_path.is_file():
        merged_candidates.append(index_path)
    for candidate in sorted(merged_candidates):
        if not candidate.is_file():
            continue
        merged_model_files.append(
            {
                "path": candidate.name,
                "size": int(candidate.stat().st_size),
                "sha256": _sha256_file(candidate),
            }
        )
    if not merged_model_files:
        raise RuntimeError(
            "Merged model did not produce model*.safetensors despite safe serialization."
        )

    metadata = {
        "manifest_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model_id": base_model_id,
        "adapter_dir": str(adapter_path),
        "merged_model_dir": str(out_dir),
        "model_loader": model_loader,
        "merge_dtype": dtype,
        "base_is_qat_derived": "-qat-" in base_model_id.lower(),
        "continued_qat_performed": False,
        "packed_int4_output": False,
        "requires_post_merge_quantization": True,
        "mtp_assistant_trained_or_modified": False,
        "adapter_files": adapter_files,
        "merged_model_files": merged_model_files,
        **training_provenance,
    }
    (out_dir / "qat_mtp_merge_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_dir
