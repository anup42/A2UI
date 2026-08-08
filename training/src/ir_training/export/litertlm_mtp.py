"""Compose a fine-tuned LiteRT-LM target with a released MTP section.

Google's public Python stack does not currently expose the private Gemma 4
mobile ``wNa8o8`` exporter.  This module therefore does not invent a model
conversion step.  It implements the safe part of the hand-off: once a target
TFLite section has been produced by a compatible exporter, replace only the
matching target section in an official package and preserve the official MTP
drafter section byte-for-byte.

The package header stores absolute section ranges.  Replacing a section is
safe without reserializing the header only when the replacement has the same
byte length and the same observable graph/quantization layout.  The composer
enforces both conditions and refuses to produce a package otherwise.  This
keeps a generic ``litert-torch`` export from being mislabeled as a production
Gemma mobile package.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from ir_training.export.litertlm_inspector import (
    LiteRTLMInspectionError,
    _tflite_graph_fingerprint,
    inspect_litertlm,
)


class LiteRTLMMTPPackagingError(RuntimeError):
    """Raised when a target cannot be safely composed with the MTP package."""


def _model_type(section: dict[str, Any]) -> str:
    for item in section.get("items") or []:
        if item.get("key") == "model_type":
            return str(item.get("value") or "")
    return ""


def find_model_section(report: dict[str, Any], model_type: str) -> dict[str, Any]:
    """Find one TFLite section by its LiteRT-LM ``model_type`` metadata."""

    wanted = str(model_type).strip().lower()
    matches = [
        section
        for section in report.get("sections") or []
        if section.get("data_type_name") == "TFLiteModel"
        and _model_type(section).strip().lower() == wanted
    ]
    if not matches:
        available = sorted(
            _model_type(section)
            for section in report.get("sections") or []
            if section.get("data_type_name") == "TFLiteModel"
        )
        raise LiteRTLMMTPPackagingError(
            f"No TFLite model_type={model_type!r} section found. Available: {available}"
        )
    if len(matches) != 1:
        raise LiteRTLMMTPPackagingError(
            f"Expected one TFLite model_type={model_type!r} section, found {len(matches)}."
        )
    return matches[0]


def read_section(path: str | Path, section: dict[str, Any]) -> bytes:
    """Read one inspected package section with bounds checks."""

    artifact = Path(path).expanduser().resolve()
    begin = int(section.get("begin_offset", -1))
    end = int(section.get("end_offset", -1))
    if begin < 0 or end < begin or end > artifact.stat().st_size:
        raise LiteRTLMMTPPackagingError(
            f"Invalid section range [{begin}, {end}) for {artifact}."
        )
    with artifact.open("rb") as handle:
        handle.seek(begin)
        data = handle.read(end - begin)
    if len(data) != end - begin:
        raise LiteRTLMMTPPackagingError(
            f"Could not read complete section [{begin}, {end}) from {artifact}."
        )
    return data


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _graph_report(data: bytes) -> dict[str, Any]:
    """Return graph fingerprints without buffer-index-sensitive serialization."""

    return _tflite_graph_fingerprint(
        memoryview(data), include_details=False, include_buffer_indices=False
    )


def compare_target_layout(base_bytes: bytes, candidate_bytes: bytes) -> dict[str, Any]:
    """Compare observable target graph and quantization layout.

    The comparison deliberately excludes buffer indices because a valid
    FlatBuffer reserialization may renumber buffers.  It still compares
    operator/tensor topology, all decoded builtin/custom option values,
    signatures, metadata, tensor semantics, buffer storage layout, and
    quantization dimensions/scale counts. Quantized scales and buffer payloads
    are expected to differ after fine-tuning and are masked by the execution
    contract.
    """

    base_graph = _graph_report(base_bytes)
    candidate_graph = _graph_report(candidate_bytes)
    if not base_graph.get("available") or not candidate_graph.get("available"):
        return {
            "available": False,
            "ok": False,
            "reason": (
                "TFLite graph inspection is unavailable. Install the generated "
                "tflite bindings in the conversion environment before packaging."
            ),
            "base": base_graph,
            "candidate": candidate_graph,
        }
    structural_match = base_graph.get("structural_sha256") == candidate_graph.get(
        "structural_sha256"
    )
    layout_match = base_graph.get("quantization_layout_sha256") == candidate_graph.get(
        "quantization_layout_sha256"
    )
    storage_match = base_graph.get("buffer_storage_sha256") == candidate_graph.get(
        "buffer_storage_sha256"
    )
    execution_contract_complete = bool(
        base_graph.get("execution_contract_complete")
        and candidate_graph.get("execution_contract_complete")
    )
    execution_contract_match = bool(
        execution_contract_complete
        and base_graph.get("execution_contract_sha256")
        and base_graph.get("execution_contract_sha256")
        == candidate_graph.get("execution_contract_sha256")
    )
    return {
        "available": True,
        "ok": bool(
            structural_match
            and layout_match
            and storage_match
            and execution_contract_match
        ),
        "structural_match": bool(structural_match),
        "quantization_layout_match": bool(layout_match),
        "buffer_storage_match": bool(storage_match),
        "execution_contract_complete": execution_contract_complete,
        "execution_contract_match": execution_contract_match,
        "base": base_graph,
        "candidate": candidate_graph,
        "interpretation": (
            "Complete execution-contract/topology/layout parity is required; "
            "learned packed constants, per-weight scales, and private calibration "
            "values are intentionally allowed to differ."
        ),
    }


def _replace_same_size_section(
    source: Path,
    section: dict[str, Any],
    replacement: bytes,
    output: Path,
    *,
    force: bool = False,
) -> None:
    """Stream a package while replacing exactly one same-sized section."""

    begin = int(section["begin_offset"])
    size = int(section["size"])
    if len(replacement) != size:
        raise LiteRTLMMTPPackagingError(
            "A LiteRT-LM header rewrite is required when target section sizes "
            f"differ (base={size}, candidate={len(replacement)})."
        )
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise LiteRTLMMTPPackagingError("Output package must differ from the base package.")
    if output.exists() and not force:
        raise LiteRTLMMTPPackagingError(
            f"Refusing to overwrite existing package {output}; pass force=True to replace it."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    with source.open("rb") as src, output.open("wb") as dst:
        remaining = begin
        while remaining:
            block = src.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise LiteRTLMMTPPackagingError("Unexpected EOF before target section.")
            dst.write(block)
            remaining -= len(block)
        dst.write(replacement)
        src.seek(begin + size)
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)


def _section_hash(report: dict[str, Any], model_type: str) -> str:
    section = find_model_section(report, model_type)
    value = section.get("sha256")
    if not value:
        raise LiteRTLMMTPPackagingError(
            "Section hashes were not requested; inspect packages with include_hashes=True."
        )
    return str(value)


def compose_with_default_mtp(
    *,
    base_litertlm: str | Path,
    output_litertlm: str | Path,
    target_litertlm: str | Path | None = None,
    target_section: str | Path | None = None,
    target_model_type: str = "tf_lite_prefill_decode",
    mtp_model_type: str = "tf_lite_mtp_drafter",
    require_graph_compatibility: bool = True,
    force: bool = False,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create a fine-tuned target package while preserving default MTP bytes.

    Exactly one of ``target_litertlm`` and ``target_section`` is required.
    ``target_litertlm`` must contain a section with ``target_model_type``;
    ``target_section`` is a raw TFLite FlatBuffer (``TFL3``).  The base package
    must already contain the official ``mtp_model_type`` section.
    """

    if bool(target_litertlm) == bool(target_section):
        raise LiteRTLMMTPPackagingError(
            "Provide exactly one of target_litertlm or target_section."
        )
    base_path = Path(base_litertlm).expanduser().resolve()
    output_path = Path(output_litertlm).expanduser().resolve()
    if not base_path.is_file():
        raise LiteRTLMMTPPackagingError(f"Base LiteRT-LM package does not exist: {base_path}")

    try:
        base_report = inspect_litertlm(
            base_path, include_hashes=True, inspect_tflite=False, include_graph_details=False
        )
    except (OSError, LiteRTLMInspectionError) as exc:
        raise LiteRTLMMTPPackagingError(f"Could not inspect base LiteRT-LM: {exc}") from exc
    target_base_section = find_model_section(base_report, target_model_type)
    # This is intentionally a hard requirement: the output must carry the
    # released/default assistant, not a newly converted public candidate.
    mtp_base_section = find_model_section(base_report, mtp_model_type)
    base_target_bytes = read_section(base_path, target_base_section)

    candidate_package: dict[str, Any] | None = None
    candidate_path: Path | None = None
    if target_litertlm:
        candidate_path = Path(target_litertlm).expanduser().resolve()
        if not candidate_path.is_file():
            raise LiteRTLMMTPPackagingError(
                f"Candidate target LiteRT-LM package does not exist: {candidate_path}"
            )
        try:
            candidate_package = inspect_litertlm(
                candidate_path,
                include_hashes=True,
                inspect_tflite=False,
                include_graph_details=False,
            )
        except (OSError, LiteRTLMInspectionError) as exc:
            raise LiteRTLMMTPPackagingError(
                f"Could not inspect candidate LiteRT-LM: {exc}"
            ) from exc
        candidate_section = find_model_section(candidate_package, target_model_type)
        candidate_bytes = read_section(candidate_path, candidate_section)
        candidate_source = str(candidate_path)
    else:
        candidate_path = Path(target_section).expanduser().resolve()  # type: ignore[arg-type]
        if not candidate_path.is_file():
            raise LiteRTLMMTPPackagingError(
                f"Candidate target TFLite section does not exist: {candidate_path}"
            )
        candidate_bytes = candidate_path.read_bytes()
        candidate_source = str(candidate_path)

    if candidate_bytes[:4] != b"TFL3":
        raise LiteRTLMMTPPackagingError(
            f"Candidate target section is not a TFLite FlatBuffer (missing TFL3): {candidate_source}"
        )
    if len(candidate_bytes) != int(target_base_section["size"]):
        raise LiteRTLMMTPPackagingError(
            "Target section size differs from the official package. A header-aware "
            f"composer is required (base={target_base_section['size']}, candidate={len(candidate_bytes)})."
        )

    layout = compare_target_layout(base_target_bytes, candidate_bytes)
    if require_graph_compatibility and not layout.get("ok", False):
        raise LiteRTLMMTPPackagingError(
            "Candidate target does not match the official target topology/quantization layout. "
            f"Details: {json.dumps(layout, ensure_ascii=False)[:4000]}"
        )

    _replace_same_size_section(
        base_path, target_base_section, candidate_bytes, output_path, force=force
    )
    try:
        output_report = inspect_litertlm(
            output_path, include_hashes=True, inspect_tflite=False, include_graph_details=False
        )
    except (OSError, LiteRTLMInspectionError) as exc:
        if output_path.exists():
            output_path.unlink()
        raise LiteRTLMMTPPackagingError(
            f"Composed package failed LiteRT-LM inspection: {exc}"
        ) from exc

    output_mtp_hash = _section_hash(output_report, mtp_model_type)
    base_mtp_hash = str(mtp_base_section.get("sha256") or "")
    output_target_hash = _section_hash(output_report, target_model_type)
    target_hash = _sha256(candidate_bytes)
    mtp_preserved = bool(base_mtp_hash and output_mtp_hash == base_mtp_hash)
    target_replaced = output_target_hash == target_hash
    non_target_sections_unchanged = _non_target_sections_unchanged(
        base_report, output_report, target_model_type
    )
    if not mtp_preserved or not target_replaced or not non_target_sections_unchanged:
        if output_path.exists():
            output_path.unlink()
        raise LiteRTLMMTPPackagingError(
            "Composed package validation failed: "
            f"mtp_preserved={mtp_preserved}, target_replaced={target_replaced}, "
            f"non_target_sections_unchanged={non_target_sections_unchanged}."
        )

    manifest = {
        "format": "litertlm",
        "package_path": str(output_path),
        "base_package": str(base_path),
        "target_source": candidate_source,
        "target_model_type": target_model_type,
        "target_section_sha256": output_target_hash,
        "target_layout": layout,
        "mtp": {
            "enabled": True,
            "mode": "default_official_section_preserved",
            "model_type": mtp_model_type,
            "section_sha256": output_mtp_hash,
            "source_package": str(base_path),
            "trained_or_modified": False,
        },
        "android_gpu": {
            "mtp_requested": True,
            "runtime": "litertlm",
            "delegate": "gpu",
            "device_validation": "not_run",
            "structural_package_validation": "passed",
            "note": (
                "A connected Android GPU run with the LiteRT-LM runtime is still "
                "required before claiming device compatibility."
            ),
        },
        "validation": {
            "mtp_section_preserved_byte_exact": mtp_preserved,
            "target_section_replaced": target_replaced,
            "non_target_sections_unchanged": non_target_sections_unchanged,
            "header_preserved": base_report.get("header", {}).get("header_sha256")
            == output_report.get("header", {}).get("header_sha256"),
        },
    }
    if manifest_path:
        manifest_file = Path(manifest_path).expanduser().resolve()
    else:
        manifest_file = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest["manifest_path"] = str(manifest_file)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def _non_target_sections_unchanged(
    base_report: dict[str, Any], output_report: dict[str, Any], target_model_type: str
) -> bool:
    base_sections = base_report.get("sections") or []
    output_sections = output_report.get("sections") or []
    if len(base_sections) != len(output_sections):
        return False
    wanted = str(target_model_type).strip().lower()
    for left, right in zip(base_sections, output_sections):
        if left.get("data_type_name") != right.get("data_type_name"):
            return False
        if _model_type(left).strip().lower() == wanted:
            continue
        if left.get("sha256") != right.get("sha256"):
            return False
    return True
