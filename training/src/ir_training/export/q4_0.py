from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ir_training.common.config import resolve_path, training_root
from ir_training.export.manifest import sha256_file
from ir_training.qat_mtp.workflow import (
    OFFICIAL_QAT_ASSISTANT,
    checkpoint_precision_family,
)


class Q40ConversionError(RuntimeError):
    """Raised when the external GGUF/Q4_0 conversion cannot be completed."""


def build_q4_0_conversion_plan(
    config: dict[str, Any],
    *,
    llama_cpp_dir_override: str | Path | None = None,
) -> dict[str, Any]:
    base = training_root()
    run_cfg = _section(config, "run")
    source_cfg = _section(config, "source")
    conversion_cfg = _section(config, "conversion")
    compatibility_cfg = _section(config, "compatibility")
    target_base_model_id = str(source_cfg.get("target_base_model_id") or "")
    assistant_model_id = str(source_cfg.get("assistant_model_id") or "")
    if checkpoint_precision_family(target_base_model_id) != "q4_0":
        raise ValueError("source.target_base_model_id must identify the Q4_0 QAT-derived target family.")
    if assistant_model_id != OFFICIAL_QAT_ASSISTANT:
        raise ValueError(f"source.assistant_model_id must be the exact matching assistant {OFFICIAL_QAT_ASSISTANT}.")
    if checkpoint_precision_family(assistant_model_id) != checkpoint_precision_family(target_base_model_id):
        raise ValueError("Target and assistant QAT precision families must match before conversion.")
    if not bool(compatibility_cfg.get("final_runtime_validation_required", False)):
        raise ValueError("compatibility.final_runtime_validation_required must remain true.")
    output_dir = resolve_path(run_cfg.get("output_dir", "outputs/export/gemma4_e2b_qat_q4_0"), base)
    llama_cpp_dir = resolve_path(
        llama_cpp_dir_override or conversion_cfg.get("llama_cpp_dir", "external/llama.cpp"),
        base,
    )
    target_dir = resolve_path(source_cfg.get("target_model_dir", "runs/gemma4_e2b_ir_qat_lora/merged_hf"), base)
    assistant_dir = resolve_path(
        source_cfg.get("assistant_model_dir", "outputs/model_cache/gemma4_e2b_qat_assistant"),
        base,
    )
    converter = llama_cpp_dir / str(conversion_cfg.get("converter", "convert_hf_to_gguf.py"))
    quantizer = _configured_quantizer(llama_cpp_dir, conversion_cfg)
    target_f16 = output_dir / str(conversion_cfg.get("target_f16_name", "gemma4-e2b-ir-qat-f16.gguf"))
    target_q4 = output_dir / str(conversion_cfg.get("target_q4_name", "gemma4-e2b-ir-qat-q4_0.gguf"))
    assistant_f16 = output_dir / str(conversion_cfg.get("assistant_f16_name", "gemma4-e2b-qat-assistant-f16.gguf"))
    assistant_q4 = output_dir / str(conversion_cfg.get("assistant_q4_name", "gemma4-e2b-qat-assistant-q4_0.gguf"))
    include_assistant = bool(conversion_cfg.get("include_assistant", True))

    steps = [
        {
            "name": "convert_target_to_f16_gguf",
            "command": [sys.executable, str(converter), str(target_dir), "--outfile", str(target_f16), "--outtype", "f16"],
        },
        {
            "name": "quantize_target_q4_0",
            "command": [str(quantizer), str(target_f16), str(target_q4), "Q4_0"],
        },
    ]
    if include_assistant:
        steps.extend(
            [
                {
                    "name": "convert_assistant_to_f16_gguf",
                    "command": [
                        sys.executable,
                        str(converter),
                        str(assistant_dir),
                        "--outfile",
                        str(assistant_f16),
                        "--outtype",
                        "f16",
                    ],
                },
                {
                    "name": "quantize_assistant_q4_0",
                    "command": [str(quantizer), str(assistant_f16), str(assistant_q4), "Q4_0"],
                },
            ]
        )
    return {
        "run_id": run_cfg.get("id"),
        "output_dir": str(output_dir),
        "llama_cpp_dir": str(llama_cpp_dir),
        "converter": str(converter),
        "quantizer": str(quantizer),
        "target_model_dir": str(target_dir),
        "target_base_model_id": target_base_model_id,
        "assistant_model_id": assistant_model_id,
        "assistant_model_dir": str(assistant_dir),
        "quantization": "Q4_0",
        "include_assistant": include_assistant,
        "steps": steps,
        "outputs": {
            "target_q4_0": str(target_q4),
            "assistant_q4_0": str(assistant_q4) if include_assistant else None,
        },
        "intermediate_outputs": [
            str(target_f16),
            *([str(assistant_f16)] if include_assistant else []),
        ],
        "continued_qat_performed": False,
        "assistant_training_performed": False,
        "assistant_conversion_support": "requires_pinned_converter_and_runtime_verification",
        "mobile_wna8o8_output": False,
        "final_runtime_validation_required": True,
    }


def execute_q4_0_conversion(
    config: dict[str, Any],
    *,
    llama_cpp_dir_override: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    plan = build_q4_0_conversion_plan(config, llama_cpp_dir_override=llama_cpp_dir_override)
    target_dir = Path(plan["target_model_dir"])
    converter = Path(plan["converter"])
    quantizer = Path(plan["quantizer"])
    output_dir = Path(plan["output_dir"])
    if not target_dir.exists():
        raise Q40ConversionError(f"Missing merged target: {target_dir}")
    if not converter.exists():
        raise Q40ConversionError(f"Missing llama.cpp converter: {converter}")
    if not quantizer.exists():
        raise Q40ConversionError(f"Missing llama.cpp quantizer: {quantizer}")

    if plan["include_assistant"]:
        _materialize_assistant(config, Path(plan["assistant_model_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [Path(value) for value in plan["outputs"].values() if value]
    step_output_paths = [*output_paths, *(Path(value) for value in plan["intermediate_outputs"])]
    existing = [path for path in step_output_paths if path.exists()]
    if existing and not force:
        raise Q40ConversionError(
            "Refusing to overwrite existing packed outputs without --force: " + ", ".join(str(path) for path in existing)
        )

    logs: list[str] = []
    for step in plan["steps"]:
        log_path = output_dir / f"{step['name']}.log"
        _run_step(step["command"], log_path)
        logs.append(str(log_path))
    missing = [path for path in step_output_paths if not path.exists()]
    if missing:
        raise Q40ConversionError("Conversion completed without expected output(s): " + ", ".join(str(path) for path in missing))

    manifest = {
        **plan,
        "executed": True,
        "logs": logs,
        "artifacts": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in output_paths
        ],
        "promotion_blocked_until_runtime_validation": True,
    }
    (output_dir / "q4_0_conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def _materialize_assistant(config: dict[str, Any], assistant_dir: Path) -> None:
    if assistant_dir.exists() and any(assistant_dir.iterdir()):
        return
    source_cfg = _section(config, "source")
    assistant_model_id = str(source_cfg.get("assistant_model_id") or "").strip()
    if not assistant_model_id:
        raise Q40ConversionError("source.assistant_model_id is required when conversion.include_assistant=true")
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency failure path
        raise Q40ConversionError("Install training/requirements-gemma4-qat.txt to download the assistant checkpoint.") from exc
    assistant_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=assistant_model_id, local_dir=str(assistant_dir))


def _configured_quantizer(llama_cpp_dir: Path, conversion_cfg: dict[str, Any]) -> Path:
    configured = str(conversion_cfg.get("quantizer") or "").strip()
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else llama_cpp_dir / path
    candidates = [
        llama_cpp_dir / "llama-quantize.exe",
        llama_cpp_dir / "build" / "bin" / "Release" / "llama-quantize.exe",
        llama_cpp_dir / "build" / "bin" / "llama-quantize.exe",
        llama_cpp_dir / "llama-quantize",
        llama_cpp_dir / "build" / "bin" / "llama-quantize",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def _run_step(command: list[str], log_path: Path) -> None:
    executable = Path(command[0])
    resolved_executable = str(executable) if executable.exists() else shutil.which(command[0])
    if resolved_executable is None:
        raise Q40ConversionError(f"Missing conversion executable: {command[0]}")
    resolved_command = [resolved_executable, *command[1:]]
    with log_path.open("w", encoding="utf-8") as log:
        log.write("Command:\n" + subprocess.list2cmdline(resolved_command) + "\n\n")
        process = subprocess.run(resolved_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log.write(process.stdout or "")
    if process.returncode != 0:
        raise Q40ConversionError(f"Conversion step failed with exit code {process.returncode}: {log_path}")


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    return value if isinstance(value, dict) else {}
