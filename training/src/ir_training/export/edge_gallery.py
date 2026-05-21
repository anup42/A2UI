from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ir_training.common.config import resolve_path, training_root
from ir_training.export.manifest import build_manifest, write_manifest


class EdgeGalleryExportError(RuntimeError):
    """Raised when a Google AI Edge Gallery export cannot be completed."""


def export_edge_gallery_model(config: dict[str, Any], *, dry_run: bool = False, model_source_override: str | None = None) -> dict[str, Any]:
    """Export a merged Hugging Face model to Google AI Edge Gallery's .litertlm format.

    The actual conversion is delegated to Google's LiteRT Torch Hugging Face export tool:
    `litert-torch export_hf`. This wrapper keeps our repo-specific source paths,
    manifest, and adb import notes consistent while failing loudly if the converter is
    missing or the conversion does not produce a .litertlm file.
    """

    run_cfg = _section(config, "run")
    source_cfg = _section(config, "source")
    export_cfg = _section(config, "export")
    android_cfg = _section(config, "android")

    base = training_root()
    output_dir = resolve_path(run_cfg.get("output_dir", "outputs/export/gemma4_e2b_ir_edge_gallery"), base)
    litert_output_dir = resolve_path(export_cfg.get("litert_output_dir", output_dir / "litertlm"), base)
    output_dir.mkdir(parents=True, exist_ok=True)
    litert_output_dir.mkdir(parents=True, exist_ok=True)

    model_source = _select_model_source(
        source_cfg=source_cfg,
        base=base,
        dry_run=dry_run,
        override=model_source_override,
    )
    command = build_litert_export_command(
        model_source=model_source,
        output_dir=litert_output_dir,
        export_cfg=export_cfg,
    )

    plan_path = _write_export_plan(
        output_dir=output_dir,
        command=command,
        model_source=model_source,
        litert_output_dir=litert_output_dir,
        dry_run=dry_run,
    )
    script_path = _write_powershell_runner(output_dir=output_dir, command=command)

    log_path = output_dir / "litert_torch_export.log"
    litert_files: list[Path] = []
    if not dry_run:
        _run_converter(command=command, log_path=log_path)
        litert_files = sorted(litert_output_dir.rglob("*.litertlm"))
        if not litert_files:
            raise EdgeGalleryExportError(
                f"LiteRT export completed but no .litertlm file was produced under {litert_output_dir}. "
                "Check litert_torch_export.log and the installed litert-torch version."
            )

    import_notes_path = _write_import_notes(
        output_dir=output_dir,
        litert_files=litert_files,
        display_name=str(android_cfg.get("display_name") or run_cfg.get("id") or "A2UI IR model"),
    )

    manifest_files = [plan_path, script_path, import_notes_path]
    if log_path.exists():
        manifest_files.append(log_path)
    manifest_files.extend(litert_files)

    manifest = build_manifest(
        output_dir=output_dir,
        base_model=str(source_cfg.get("base_model_id") or ""),
        adapter_type="lora_merged" if source_cfg.get("adapter_dir") else "base",
        training_data_run=str(android_cfg.get("stage") or "stage3_ir"),
        prompt_version=None,
        schema_version="flat-spec",
        runtime="litertlm",
        min_app_version=str(export_cfg.get("min_app_version", "1.1.0")),
        max_input_tokens=int(export_cfg.get("max_input_tokens", 8192)),
        max_output_tokens=int(export_cfg.get("max_output_tokens", 8192)),
        files=manifest_files,
    )
    manifest.update(
        {
            "package_name": android_cfg.get("package_name"),
            "display_name": android_cfg.get("display_name"),
            "format": "litertlm",
            "edge_gallery": {
                "import_format": ".litertlm",
                "dry_run": dry_run,
                "model_source": str(model_source),
                "litert_output_dir": str(litert_output_dir),
                "model_files": [str(path.relative_to(output_dir)) for path in litert_files],
                "adb_push": [f"adb push {path} /sdcard/Download/" for path in litert_files],
            },
        }
    )
    write_manifest(output_dir, manifest)
    return manifest


def build_litert_export_command(*, model_source: str | Path, output_dir: str | Path, export_cfg: dict[str, Any]) -> list[str]:
    command_name = str(export_cfg.get("command") or "litert-torch")
    command = [
        command_name,
        "export_hf",
        f"--model={model_source}",
        f"--output_dir={output_dir}",
    ]

    task = str(export_cfg.get("task") or "").strip()
    if task:
        command.append(f"--task={task}")

    quantization_recipe = str(export_cfg.get("quantization_recipe") or "").strip()
    if quantization_recipe:
        command.append(f"--quantization_recipe={quantization_recipe}")

    jinja_override = str(export_cfg.get("jinja_chat_template_override") or "").strip()
    if jinja_override:
        command.append(f"--jinja_chat_template_override={jinja_override}")

    if bool(export_cfg.get("externalize_embedder", False)):
        command.append("--externalize_embedder")

    for flag in export_cfg.get("extra_flags") or []:
        flag_text = str(flag).strip()
        if flag_text:
            command.append(flag_text)
    return command


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    return value if isinstance(value, dict) else {}


def _select_model_source(*, source_cfg: dict[str, Any], base: Path, dry_run: bool, override: str | None) -> str:
    if override:
        return override

    adapter_dir_value = source_cfg.get("adapter_dir")
    merged_model_value = source_cfg.get("merged_model_dir")
    base_model_id = str(source_cfg.get("base_model_id") or "").strip()
    if not base_model_id:
        raise EdgeGalleryExportError("source.base_model_id is required")

    merged_model_dir = resolve_path(merged_model_value, base) if merged_model_value else None
    if merged_model_dir is not None and (merged_model_dir.exists() or dry_run):
        return str(merged_model_dir)

    if adapter_dir_value:
        raise EdgeGalleryExportError(
            "A LoRA adapter is configured, but source.merged_model_dir does not exist. "
            "Run with --merge-lora first, or point --model-source to an already merged Hugging Face model directory."
        )

    return base_model_id


def _run_converter(*, command: list[str], log_path: Path) -> None:
    executable = shutil.which(command[0])
    if executable is None:
        raise EdgeGalleryExportError(
            "Missing LiteRT Torch CLI. Install the Edge export environment first, for example: "
            "python -m pip install -r training/requirements-edge-export.txt"
        )
    command = [executable, *command[1:]]
    with log_path.open("w", encoding="utf-8") as log:
        log.write("Command:\n")
        log.write(_format_command(command) + "\n\n")
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log.write(process.stdout or "")
    if process.returncode != 0:
        raise EdgeGalleryExportError(
            f"LiteRT Torch export failed with exit code {process.returncode}. See {log_path}."
        )


def _write_export_plan(*, output_dir: Path, command: list[str], model_source: str, litert_output_dir: Path, dry_run: bool) -> Path:
    path = output_dir / "edge_gallery_export_plan.json"
    payload = {
        "dry_run": dry_run,
        "model_source": model_source,
        "litert_output_dir": str(litert_output_dir),
        "command": command,
        "command_text": _format_command(command),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_powershell_runner(*, output_dir: Path, command: list[str]) -> Path:
    path = output_dir / "run_litert_export.ps1"
    path.write_text(
        "$ErrorActionPreference = 'Stop'\n"
        + _format_command_for_powershell(command)
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_import_notes(*, output_dir: Path, litert_files: list[Path], display_name: str) -> Path:
    path = output_dir / "EDGE_GALLERY_IMPORT.md"
    if litert_files:
        push_lines = "\n".join(f"adb push {item} /sdcard/Download/" for item in litert_files)
    else:
        push_lines = "adb push <generated-model>.litertlm /sdcard/Download/"
    path.write_text(
        "# Google AI Edge Gallery Import\n\n"
        f"Display name: {display_name}\n\n"
        "1. Push the generated `.litertlm` file to the device Download folder.\n\n"
        "```powershell\n"
        f"{push_lines}\n"
        "```\n\n"
        "2. Open Google AI Edge Gallery, tap `+`, choose the `.litertlm` file, configure runtime parameters, and import.\n",
        encoding="utf-8",
    )
    return path


def _format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _format_command_for_powershell(command: list[str]) -> str:
    escaped = []
    for part in command:
        escaped.append("'" + part.replace("'", "''") + "'")
    return "& " + " ".join(escaped)
