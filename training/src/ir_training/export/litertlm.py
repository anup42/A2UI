from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ir_training.common.config import resolve_path, training_root
from ir_training.export.manifest import build_manifest, write_manifest


def export_litertlm_package(config: dict[str, Any]) -> dict[str, Any]:
    run_cfg = config.get("run") if isinstance(config.get("run"), dict) else {}
    source_cfg = config.get("source") if isinstance(config.get("source"), dict) else {}
    export_cfg = config.get("export") if isinstance(config.get("export"), dict) else {}
    android_cfg = config.get("android") if isinstance(config.get("android"), dict) else {}

    base = training_root()
    output_dir = resolve_path(run_cfg.get("output_dir", "outputs/export/gemma_e2b_ir_litertlm"), base)
    merged_model_dir = resolve_path(source_cfg.get("merged_model_dir", "runs/gemma_e2b_ir_lora/merged_hf"), base)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_files: list[Path] = []
    if merged_model_dir.exists():
        model_out = output_dir / "hf_model"
        if model_out.exists():
            shutil.rmtree(model_out)
        shutil.copytree(merged_model_dir, model_out)
        copied_files.extend([p for p in model_out.rglob("*") if p.is_file()])

    runtime_note = output_dir / "LITERTLM_EXPORT_NOTE.txt"
    runtime_note.write_text(
        "This package contains metadata and, when available, the merged HF model. "
        "Run the LiteRT-LM conversion toolchain for the target Gemma variant to create the final .litertlm file.\n",
        encoding="utf-8",
    )
    copied_files.append(runtime_note)

    manifest = build_manifest(
        output_dir=output_dir,
        base_model=str(source_cfg.get("base_model_id") or ""),
        adapter_type="lora",
        training_data_run=str(android_cfg.get("stage") or "stage3_ir"),
        prompt_version=None,
        schema_version=None,
        runtime=str(export_cfg.get("runtime", "litertlm")),
        min_app_version=str(export_cfg.get("min_app_version", "1.1.0")),
        max_input_tokens=int(export_cfg.get("max_input_tokens", 8192)),
        max_output_tokens=int(export_cfg.get("max_output_tokens", 8192)),
        files=copied_files,
    )
    manifest.update(
        {
            "package_name": android_cfg.get("package_name"),
            "display_name": android_cfg.get("display_name"),
            "format": export_cfg.get("format", "litertlm"),
        }
    )
    write_manifest(output_dir, manifest)
    return manifest
