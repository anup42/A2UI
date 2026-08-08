"""Validate a Gemma 3 270M LiteRT-LM package for Android GPU use.

This package gate expects a single Gemma 3 270M prefill/decode model and does
not expect a Gemma 4 MTP drafter.  It cannot prove device delegate execution;
an optional device JSON report is required for that claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.eval.android_gpu_report import (
    AndroidGpuParityReportError,
    load_android_gpu_parity_report,
)
from ir_training.export.litertlm_inspector import (
    LiteRTLMInspectionError,
    compare_litertlm_reports,
    inspect_litertlm,
)
from ir_training.export.litertlm_mtp import (
    LiteRTLMMTPPackagingError,
    find_model_section,
)


def validate_package(
    artifact: str | Path,
    *,
    inspect_graphs: bool = False,
    device_report: str | Path | None = None,
    official_artifact: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(artifact).expanduser().resolve()
    try:
        report = inspect_litertlm(
            path,
            include_hashes=True,
            inspect_tflite=inspect_graphs,
            include_graph_details=False,
        )
    except (OSError, LiteRTLMInspectionError) as exc:
        return {"ok": False, "artifact": str(path), "errors": [str(exc)], "checks": {}}

    errors: list[str] = []
    warnings: list[str] = []
    try:
        target = find_model_section(report, "tf_lite_prefill_decode")
    except LiteRTLMMTPPackagingError as exc:
        target = None
        errors.append(str(exc))
    if target is not None and not target.get("alignment_ok"):
        errors.append("The Gemma 270M prefill/decode section is not aligned.")

    sections = report.get("sections") or []
    ordered = all(
        bool(section.get("alignment_ok")) and bool(section.get("ordered_after_previous"))
        for section in sections
    )
    if not ordered:
        errors.append("One or more LiteRT-LM sections are misaligned or out of order.")

    mtp_sections = [
        section
        for section in sections
        if any(
            item.get("key") == "model_type"
            and str(item.get("value") or "").lower() == "tf_lite_mtp_drafter"
            for item in section.get("items") or []
        )
    ]
    if mtp_sections:
        warnings.append(
            "The package contains an MTP drafter section; Gemma 3 270M pipeline does not use it."
        )

    graph = None
    execution_contract_complete = False
    official_graph_comparison: dict[str, Any] = {"provided": False}
    if target is not None and inspect_graphs:
        graph = next(
            (item for item in report.get("graphs") or [] if item.get("section_index") == target.get("index")),
            None,
        )
        if not graph or not graph.get("available"):
            errors.append("The target TFLite graph could not be inspected.")
        else:
            execution_contract_complete = bool(
                graph.get("execution_contract_complete")
            )
            if not execution_contract_complete:
                errors.append(
                    "The target TFLite execution contract was not completely decoded."
                )
        if official_artifact:
            try:
                official_report = inspect_litertlm(
                    Path(official_artifact).expanduser().resolve(),
                    include_hashes=False,
                    inspect_tflite=True,
                    include_graph_details=False,
                )
                comparison = compare_litertlm_reports(
                    official_report, report
                )
                required = {
                    "header_metadata_match": comparison.get(
                        "header_metadata_match"
                    )
                    is True,
                    "section_layout_match": comparison.get(
                        "section_layout_match"
                    )
                    is True,
                    "graph_structure_match": comparison.get(
                        "graph_structure_match"
                    )
                    is True,
                    "quantization_layout_match": comparison.get(
                        "quantization_layout_match"
                    )
                    is True,
                    "execution_contract_complete": comparison.get(
                        "execution_contract_complete"
                    )
                    is True,
                    "execution_contract_match": comparison.get(
                        "execution_contract_match"
                    )
                    is True,
                }
                official_graph_comparison = {
                    "provided": True,
                    "checks": required,
                    "comparison": comparison,
                    "validated": all(required.values()),
                }
                if not official_graph_comparison["validated"]:
                    errors.append(
                        "Candidate graph does not match the official complete execution contract."
                    )
            except (OSError, LiteRTLMInspectionError) as exc:
                official_graph_comparison = {
                    "provided": True,
                    "validated": False,
                    "error": str(exc),
                }
                errors.append(
                    f"Official graph comparison failed: {exc}"
                )

    device = {"provided": False, "validated": False}
    if device_report:
        try:
            device = load_android_gpu_parity_report(
                device_report,
                expected_mtp=False,
                require_mtp_acceptance=False,
                expected_official_artifact=official_artifact,
                expected_candidate_artifact=path,
            )
            device["provided"] = True
            device["validated"] = True
        except AndroidGpuParityReportError as exc:
            errors.append(f"Device report did not prove 270M GPU parity: {exc}")
            device = {"provided": True, "validated": False}

    return {
        "ok": not errors,
        "artifact": str(path),
        "model_family": "gemma3_270m",
        "quantization": "WI8_AFP32",
        "sections": len(sections),
        "checks": {
            "package_inspection": True,
            "prefill_decode_present": target is not None,
            "all_sections_aligned_and_ordered": ordered,
            "mtp_not_required": not mtp_sections,
            "graph_inspection_requested": inspect_graphs,
            "execution_contract_complete": execution_contract_complete,
            "official_execution_contract_match": bool(
                official_graph_comparison.get("validated")
            ),
            "gpu_device_validation": bool(device.get("validated")),
        },
        "warnings": warnings,
        "device": device,
        "official_graph_comparison": official_graph_comparison,
        "errors": errors,
        "limitations": [
            "Desktop inspection cannot prove Android GPU delegate execution.",
            "This pipeline does not provide MTP acceleration for Gemma 3 270M.",
            "The export recipe is weight-only INT8 with FP32 activations, not INT4/Q4_0.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Gemma 3 270M LiteRT-LM readiness for Android GPU.")
    parser.add_argument("artifact")
    parser.add_argument("--inspect-graphs", action="store_true")
    parser.add_argument("--device-report")
    parser.add_argument(
        "--official-artifact",
        help="Optional official Q8 package to re-hash against the device report.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_package(
        args.artifact,
        inspect_graphs=args.inspect_graphs,
        device_report=args.device_report,
        official_artifact=args.official_artifact,
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
