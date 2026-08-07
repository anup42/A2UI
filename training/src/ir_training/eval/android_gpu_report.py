"""Fail-closed validation for Android LiteRT-LM GPU parity reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class AndroidGpuParityReportError(ValueError):
    """Raised when a device report does not prove the requested runtime mode."""


def validate_android_gpu_parity_report(
    report: dict[str, Any],
    *,
    expected_mtp: bool,
    require_mtp_acceptance: bool | None = None,
) -> dict[str, bool]:
    """Return named gates or raise when a schema-v2 parity report is incomplete."""

    comparison = report.get("comparison")
    if not isinstance(comparison, dict):
        raise AndroidGpuParityReportError("report has no comparison object")
    if require_mtp_acceptance is None:
        require_mtp_acceptance = expected_mtp
    checks = {
        "schema_v2_or_newer": int(report.get("schema_version", 0) or 0) >= 2,
        "mode_matches": report.get("mtp_enabled") is expected_mtp,
        "structural_gpu_parity": comparison.get("structural_gpu_parity_pass") is True,
        "fixed_length_throughput_sample": (
            comparison.get("throughput_sample_comparable") is True
        ),
        "throughput_gate": comparison.get("throughput_gate_pass") is True,
        "overall_pass": comparison.get("overall_pass") is True,
    }
    if require_mtp_acceptance:
        checks["mtp_acceptance"] = comparison.get("mtp_acceptance_gate_pass") is True
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AndroidGpuParityReportError(f"failed required gates {failed}")
    return checks


def load_android_gpu_parity_report(
    report_path: str | Path,
    *,
    expected_mtp: bool,
    require_mtp_acceptance: bool | None = None,
) -> dict[str, Any]:
    """Read and validate one report written by the Android parity runner."""

    path = Path(report_path).expanduser().resolve()
    if not path.is_file():
        raise AndroidGpuParityReportError(f"report was not written: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AndroidGpuParityReportError(f"could not read report {path}: {exc}") from exc
    if not isinstance(report, dict):
        raise AndroidGpuParityReportError(f"report root is not an object: {path}")
    report["pipeline_gate_checks"] = validate_android_gpu_parity_report(
        report,
        expected_mtp=expected_mtp,
        require_mtp_acceptance=require_mtp_acceptance,
    )
    return report
