"""Bind a prepared dataset to the tokenizer actually loaded by SFT."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def value_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def verify_tokenizer_binding(dataset_dir: Path, tokenizer: Any, model_config: dict[str, Any], *, required: bool = False) -> dict[str, Any]:
    path = dataset_dir / "manifest.json"
    if not path.is_file():
        if required:
            raise ValueError("Review training requires the checked dataset preparation manifest.")
        return {"required": False, "verified": False, "reason": "legacy dataset without preparation manifest"}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = manifest.get("tokenizer")
    if not isinstance(expected, dict) or "vocabulary_sha256" not in expected:
        if required:
            raise ValueError("Prepare the dataset with its actual tokenizer before review training.")
        return {"required": False, "verified": False, "reason": "legacy manifest without tokenizer binding"}
    actual = {
        "vocabulary_sha256": value_sha256(tokenizer.get_vocab()),
        "chat_template_sha256": value_sha256(getattr(tokenizer, "chat_template", None)),
        "chat_template_kwargs": model_config.get("chat_template_kwargs") or {},
    }
    for key, value in actual.items():
        if expected.get(key) != value:
            raise ValueError(f"Prepared dataset tokenizer binding mismatch: {key}. Reprepare using this model's tokenizer and template.")
    for name in ("train", "val"):
        split = dataset_dir / f"{name}.jsonl"
        identity = (manifest.get("splits") or {}).get(name) or {}
        digest = hashlib.sha256()
        with split.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if identity.get("output_sha256") != digest.hexdigest():
            raise ValueError(f"Prepared {name} split changed after manifest creation.")
    return {"required": required, "verified": True, "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **actual}
