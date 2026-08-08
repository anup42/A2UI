"""Validate LiteRT-LM package structure for an MTP=true Android GPU run.

This is a desktop/package gate.  It does not emulate the Android LiteRT-LM
runtime or prove GPU delegate execution.  Pass a JSON device report from an
Android harness with ``--device-report`` when that runtime test has been run.
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
    inspect_litertlm,
)
from ir_training.export.litertlm_mtp import (
    LiteRTLMMTPPackagingError,
    find_model_section,
)


def validate_package(
    artifact: str | Path,
    *,
    mtp_model_type: str = "tf_lite_mtp_drafter",
    device_report: str | Path | None = None,
    target_only_device_report: str | Path | None = None,
    official_artifact: str | Path | None = None,
    inspect_graphs: bool = False,
) -> dict[str, Any]:
    path = Path(artifact).expanduser().resolve()
    try:
        report = inspect_litertlm(
            path, include_hashes=True, inspect_tflite=inspect_graphs, include_graph_details=False
        )
    except (OSError, LiteRTLMInspectionError) as exc:
        return {"ok": False, "artifact": str(path), "errors": [str(exc)], "checks": {}}

    errors: list[str] = []
    sections = report.get("sections") or []
    try:
        mtp = find_model_section(report, mtp_model_type)
    except LiteRTLMMTPPackagingError as exc:
        mtp = None
        errors.append(str(exc))
    if mtp is not None:
        if not mtp.get("alignment_ok"):
            errors.append("MTP section is not aligned to the LiteRT-LM section boundary.")
        graph = next(
            (item for item in report.get("graphs") or [] if item.get("section_index") == mtp.get("index")),
            None,
        )
        if inspect_graphs and graph is not None and not graph.get("available"):
            errors.append("MTP TFLite graph could not be inspected in this environment.")

    ordered = all(
        bool(section.get("alignment_ok")) and bool(section.get("ordered_after_previous"))
        for section in sections
    )
    if not ordered:
        errors.append("One or more LiteRT-LM sections are misaligned or out of order.")

    device: dict[str, Any] = {
        "target_only": {"provided": False, "validated": False},
        "mtp_on": {"provided": False, "validated": False},
    }
    if bool(target_only_device_report) != bool(device_report):
        errors.append(
            "Provide both target-only and MTP-on device reports for runtime validation."
        )
    if target_only_device_report:
        try:
            target_device = load_android_gpu_parity_report(
                target_only_device_report,
                expected_mtp=False,
                require_mtp_acceptance=False,
                expected_official_artifact=official_artifact,
                expected_candidate_artifact=path,
            )
            target_device["provided"] = True
            target_device["validated"] = True
            device["target_only"] = target_device
        except AndroidGpuParityReportError as exc:
            errors.append(f"Target-only device report did not prove GPU parity: {exc}")
            device["target_only"] = {"provided": True, "validated": False}
    if device_report:
        try:
            mtp_device = load_android_gpu_parity_report(
                device_report,
                expected_mtp=True,
                require_mtp_acceptance=True,
                expected_official_artifact=official_artifact,
                expected_candidate_artifact=path,
            )
            mtp_device["provided"] = True
            mtp_device["validated"] = True
            device["mtp_on"] = mtp_device
        except AndroidGpuParityReportError as exc:
            errors.append(f"MTP-on device report did not prove GPU parity: {exc}")
            device["mtp_on"] = {"provided": True, "validated": False}

    target_validated = bool(device["target_only"].get("validated"))
    mtp_validated = bool(device["mtp_on"].get("validated"))

    return {
        "ok": not errors,
        "artifact": str(path),
        "mtp_model_type": mtp_model_type,
        "sections": len(sections),
        "checks": {
            "package_inspection": True,
            "mtp_section_present": mtp is not None,
            "all_sections_aligned_and_ordered": ordered,
            "graph_inspection_requested": inspect_graphs,
            "mtp_true_requested": True,
            "target_only_gpu_device_validation": target_validated,
            "mtp_on_gpu_device_validation": mtp_validated,
            "dual_mode_gpu_device_validation": target_validated and mtp_validated,
        },
        "device": device,
        "errors": errors,
        "limitations": [
            "Desktop inspection cannot prove Android GPU delegate execution.",
            "MTP acceptance/rate after target fine-tuning requires an on-device runtime test.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate .litertlm package readiness for mtp=true + Android GPU.")
    parser.add_argument("artifact", help="Path to the composed .litertlm package.")
    parser.add_argument("--mtp-model-type", default="tf_lite_mtp_drafter")
    parser.add_argument(
        "--inspect-graphs",
        action="store_true",
        help="Parse every embedded TFLite graph (slower; package-only validation is the default).",
    )
    parser.add_argument("--device-report", help="Optional JSON report from a real Android GPU runtime test.")
    parser.add_argument(
        "--official-artifact",
        help="Optional official package to re-hash against both device reports.",
    )
    parser.add_argument(
        "--target-only-device-report",
        help="Optional schema-v5 target-only parity report; required with --device-report for a shipping-speed claim.",
    )
    parser.add_argument("--output", type=Path, help="Optional output JSON path.")
    args = parser.parse_args()
    result = validate_package(
        args.artifact,
        mtp_model_type=args.mtp_model_type,
        device_report=args.device_report,
        target_only_device_report=args.target_only_device_report,
        official_artifact=args.official_artifact,
        inspect_graphs=args.inspect_graphs,
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
