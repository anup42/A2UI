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

from ir_training.export.litertlm_inspector import LiteRTLMInspectionError, inspect_litertlm
from ir_training.export.litertlm_mtp import find_model_section


def validate_package(
    artifact: str | Path,
    *,
    mtp_model_type: str = "tf_lite_mtp_drafter",
    device_report: str | Path | None = None,
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
    except Exception as exc:
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

    device = {"provided": False, "validated": False}
    if device_report:
        device_path = Path(device_report).expanduser().resolve()
        try:
            device = json.loads(device_path.read_text(encoding="utf-8"))
            device["provided"] = True
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"Could not read device report: {exc}")
            device = {"provided": True, "validated": False}
        if device.get("mtp_enabled") is not True:
            errors.append("Device report does not prove mtp_enabled=true.")
        if str(device.get("delegate", "")).lower() != "gpu":
            errors.append("Device report does not prove GPU delegate execution.")
        if str(device.get("status", "")).lower() not in {"passed", "pass", "ok"}:
            errors.append("Device report status is not passed/ok.")
        device["validated"] = not errors

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
            "gpu_device_validation": bool(device.get("validated")),
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
    parser.add_argument("--output", type=Path, help="Optional output JSON path.")
    args = parser.parse_args()
    result = validate_package(
        args.artifact,
        mtp_model_type=args.mtp_model_type,
        device_report=args.device_report,
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
