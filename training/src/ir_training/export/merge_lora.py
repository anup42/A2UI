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


def _adapter_file_records(adapter_path: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": candidate.name,
            "size": int(candidate.stat().st_size),
            "sha256": _sha256_file(candidate),
        }
        for candidate in sorted(adapter_path.glob("adapter*"))
        if candidate.is_file()
    ]


def _normalized_file_records(records: Any) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        return []
    normalized = []
    for item in records:
        if not isinstance(item, dict):
            continue
        try:
            size = int(item.get("size", -1) or -1)
        except (TypeError, ValueError):
            size = -1
        normalized.append(
            {
                "path": str(item.get("path") or ""),
                "size": size,
                "sha256": str(item.get("sha256") or "").lower(),
            }
        )
    return sorted(normalized, key=lambda item: item["path"])


def _verify_qat_training_metadata(
    adapter_path: Path,
    *,
    training_config_sha256: str,
    training_method: str,
) -> dict[str, Any]:
    metadata_path = next(
        (
            candidate
            for candidate in (
                adapter_path / "training_metadata.json",
                adapter_path.parent / "training_metadata.json",
            )
            if candidate.is_file()
        ),
        None,
    )
    checks = {
        "metadata_present": metadata_path is not None,
        "metadata_v2_or_newer": False,
        "training_config_hash_matches": False,
        "training_method_matches": False,
        "qat_enabled": False,
        "effective_merged_weight_qat": False,
        "effective_lora_wrappers_recorded": False,
        "all_lora_adapter_linears_covered": False,
        "zero_lora_dropout": False,
        "git_commit_recorded": False,
        "adapter_checkpoint_hashes_match": False,
    }
    report: dict[str, Any] = {
        "required": True,
        "path": str(metadata_path) if metadata_path else None,
        "sha256": _sha256_file(metadata_path) if metadata_path else None,
        "checks": checks,
        "verified": False,
    }
    if metadata_path is None:
        return report
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report["error"] = f"Could not read training metadata: {exc}"
        return report
    if not isinstance(metadata, dict):
        report["error"] = "Training metadata root is not an object."
        return report
    qat = metadata.get("qat") if isinstance(metadata.get("qat"), dict) else {}
    qat_spec = qat.get("spec") if isinstance(qat.get("spec"), dict) else {}
    lora = metadata.get("lora") if isinstance(metadata.get("lora"), dict) else {}
    training = (
        metadata.get("training")
        if isinstance(metadata.get("training"), dict)
        else {}
    )
    checks["metadata_v2_or_newer"] = (
        int(metadata.get("training_metadata_version", 0) or 0) >= 2
    )
    checks["training_config_hash_matches"] = (
        str(metadata.get("training_config_sha256") or "").lower()
        == training_config_sha256.lower()
    )
    checks["training_method_matches"] = (
        str(training.get("method") or "") == training_method
    )
    checks["qat_enabled"] = qat.get("enabled") is True
    checks["effective_merged_weight_qat"] = bool(
        qat.get("effective_merged_weight_qat_enabled") is True
        and qat_spec.get("effective_merged_weight") is True
    )
    checks["effective_lora_wrappers_recorded"] = (
        int(qat.get("wrapped_effective_lora_count", 0) or 0) > 0
    )
    checks["all_lora_adapter_linears_covered"] = not bool(
        qat.get("uncovered_lora_adapter_linear_names")
    )
    try:
        checks["zero_lora_dropout"] = float(lora.get("dropout", -1.0)) == 0.0
    except (TypeError, ValueError):
        checks["zero_lora_dropout"] = False
    git_commit = str(metadata.get("git_commit") or "").strip().lower()
    checks["git_commit_recorded"] = bool(git_commit and git_commit != "unknown")
    actual_files = _normalized_file_records(_adapter_file_records(adapter_path))
    checkpoints = metadata.get("adapter_checkpoints")
    if isinstance(checkpoints, list):
        checks["adapter_checkpoint_hashes_match"] = any(
            isinstance(checkpoint, dict)
            and _normalized_file_records(checkpoint.get("files")) == actual_files
            for checkpoint in checkpoints
        )
    report["training_git_commit"] = git_commit or None
    report["adapter_files"] = actual_files
    report["verified"] = bool(all(checks.values()))
    return report


def _training_provenance(
    adapter_path: Path,
    training_config_path: str | Path | None,
    *,
    base: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "training_config": None,
        "training_config_sha256": None,
        "training_method": None,
        "qat_enabled": None,
        "qat_profile": None,
        "qat_effective_merged_weight": None,
        "lora_dropout": None,
        "training_run_metadata": {
            "required": False,
            "verified": False,
        },
    }
    if training_config_path is None:
        return result
    from ir_training.common.config import load_yaml

    resolved_config = resolve_path(training_config_path, base)
    if not resolved_config.is_file():
        raise FileNotFoundError(
            f"Missing training config for merge provenance: {resolved_config}"
        )
    config_bytes = resolved_config.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    config = load_yaml(resolved_config)
    training = config.get("training") if isinstance(config.get("training"), dict) else {}
    qat = config.get("qat") if isinstance(config.get("qat"), dict) else {}
    lora = config.get("lora") if isinstance(config.get("lora"), dict) else {}
    training_method = str(training.get("method") or "")
    qat_enabled = bool(qat.get("enabled", False))
    result.update(
        {
            "training_config": str(resolved_config),
            "training_config_sha256": config_sha256,
            "training_method": training_method,
            "qat_enabled": qat_enabled,
            "qat_profile": qat.get("profile"),
            "qat_effective_merged_weight": bool(
                qat.get("effective_merged_weight", False)
            ),
            "lora_dropout": lora.get("dropout"),
        }
    )
    if qat_enabled:
        run_report = _verify_qat_training_metadata(
            adapter_path,
            training_config_sha256=config_sha256,
            training_method=training_method,
        )
        result["training_run_metadata"] = run_report
        if not run_report["verified"]:
            failed = [
                name
                for name, passed in run_report["checks"].items()
                if not passed
            ]
            raise ValueError(
                "QAT adapter training provenance is incomplete or mismatched: "
                + ", ".join(failed)
            )
    return result


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

    A merge from a QAT-trained adapter remains a floating-point Hugging Face
    model. It is not packed INT4; this function verifies the training-time QAT
    provenance but does not itself run QAT or modify an MTP assistant.
    """

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
    training_provenance = _training_provenance(
        adapter_path,
        training_config_path,
        base=base,
    )
    try:
        from peft import PeftModel  # type: ignore
        from transformers import AutoProcessor, AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency failure path
        raise RuntimeError(
            "Install the training requirements before merging LoRA adapters."
        ) from exc
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

    adapter_files = _adapter_file_records(adapter_path)

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
        "manifest_version": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model_id": base_model_id,
        "adapter_dir": str(adapter_path),
        "merged_model_dir": str(out_dir),
        "model_loader": model_loader,
        "merge_dtype": dtype,
        "base_is_qat_derived": "-qat-" in base_model_id.lower(),
        "continued_qat_performed": bool(training_provenance["qat_enabled"]),
        "merge_performed_qat": False,
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
