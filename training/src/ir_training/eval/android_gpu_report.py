"""Fail-closed validation for Android LiteRT-LM GPU parity reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MIN_PERFORMANCE_WARM_RUNS = 3
PERFORMANCE_SELECTION_POLICY = "median_of_all_warm_runs_all_must_be_valid"


class AndroidGpuParityReportError(ValueError):
    """Raised when a device report does not prove the requested runtime mode."""


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_android_gpu_parity_report(
    report: dict[str, Any],
    *,
    expected_mtp: bool,
    require_mtp_acceptance: bool | None = None,
    expected_official_sha256: str | None = None,
    expected_candidate_sha256: str | None = None,
) -> dict[str, bool]:
    """Return named gates or raise when a schema-v5 parity report is incomplete."""

    comparison = report.get("comparison")
    if not isinstance(comparison, dict):
        raise AndroidGpuParityReportError("report has no comparison object")
    if require_mtp_acceptance is None:
        require_mtp_acceptance = expected_mtp
    official_identity = comparison.get("official_artifact_identity")
    if not isinstance(official_identity, dict):
        official_identity = {}
    candidate_identity = comparison.get("candidate_artifact_identity")
    if not isinstance(candidate_identity, dict):
        candidate_identity = {}
    reported_official_sha256 = str(
        official_identity.get("host_sha256") or ""
    ).lower()
    reported_candidate_sha256 = str(
        candidate_identity.get("host_sha256") or ""
    ).lower()
    checks = {
        "schema_v5_or_newer": int(report.get("schema_version", 0) or 0) >= 5,
        "mode_matches": report.get("mtp_enabled") is expected_mtp,
        "minimum_three_warm_runs": int(report.get("warm_run_count", 0) or 0)
        >= MIN_PERFORMANCE_WARM_RUNS,
        "median_warm_selection_policy": report.get(
            "performance_selection_policy"
        )
        == PERFORMANCE_SELECTION_POLICY,
        "official_artifact_identity": comparison.get(
            "official_artifact_identity_verified"
        )
        is True,
        "candidate_artifact_identity": comparison.get(
            "candidate_artifact_identity_verified"
        )
        is True,
        "structural_gpu_parity": comparison.get("structural_gpu_parity_pass") is True,
        "fixed_length_throughput_sample": (
            comparison.get("throughput_sample_comparable") is True
        ),
        "all_warm_samples_valid": comparison.get("warm_run_gate_pass") is True,
        "all_warm_structural_samples_valid": comparison.get(
            "warm_structural_gate_pass"
        )
        is True,
        "throughput_gate": comparison.get("throughput_gate_pass") is True,
        "overall_pass": comparison.get("overall_pass") is True,
    }
    if expected_candidate_sha256 is not None:
        checks["candidate_artifact_sha256_matches_pipeline_output"] = (
            reported_candidate_sha256 == expected_candidate_sha256.lower()
        )
    if expected_official_sha256 is not None:
        checks["official_artifact_sha256_matches_pipeline_reference"] = (
            reported_official_sha256 == expected_official_sha256.lower()
        )
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
    expected_official_artifact: str | Path | None = None,
    expected_candidate_artifact: str | Path | None = None,
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
    def expected_sha256(
        artifact: str | Path | None, *, role: str
    ) -> str | None:
        if artifact is None:
            return None
        artifact_path = Path(artifact).expanduser().resolve()
        if not artifact_path.is_file():
            raise AndroidGpuParityReportError(
                f"{role} artifact was not written: {artifact_path}"
            )
        try:
            return _sha256_file(artifact_path)
        except OSError as exc:
            raise AndroidGpuParityReportError(
                f"could not hash {role} artifact {artifact_path}: {exc}"
            ) from exc
    expected_official_sha256 = expected_sha256(
        expected_official_artifact, role="official"
    )
    expected_candidate_sha256 = expected_sha256(
        expected_candidate_artifact, role="candidate"
    )
    report["pipeline_gate_checks"] = validate_android_gpu_parity_report(
        report,
        expected_mtp=expected_mtp,
        require_mtp_acceptance=require_mtp_acceptance,
        expected_official_sha256=expected_official_sha256,
        expected_candidate_sha256=expected_candidate_sha256,
    )
    return report
