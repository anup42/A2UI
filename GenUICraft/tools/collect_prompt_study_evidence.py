"""Collect a portable, hash-reconciled prompt-study evidence bundle.

Only completed runs and runs explicitly marked ``earlyRejected`` are included.
The raw study directory is read-only: all copied and generated files are written
to a new output directory through a sibling staging directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FINAL_CASE_IDS = tuple(f"BXP-{index:03d}" for index in range(1, 51))
REQUIRED_RUN_FILES = (
    "experiment.json",
    "run_config.json",
    "results.json",
    "experiment_prompt.txt",
    "runtime_log.txt",
)
OPTIONAL_RUN_FILES = (
    "summary.json",
    "host_summary.json",
    "attempt_evidence.json",
    "adaptive_rejection_decision.json",
)
CASE_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,60}\Z")
ATTEMPT_PATTERN = re.compile(
    r"attempt_([1-9][0-9]*)(?:(\.txt)|(_metrics\.json)|(_input\.txt))\Z"
)
BANNED_SUFFIXES = {
    ".apk",
    ".aar",
    ".png",
    ".litertlm",
    ".tflite",
    ".onnx",
    ".safetensors",
    ".gguf",
}


class EvidenceError(RuntimeError):
    """Raised when raw evidence is inconsistent or incomplete."""


def _load_json(path: Path, expected_type: type[Any]) -> Any:
    if not path.is_file():
        raise EvidenceError(f"Missing required JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"Cannot read JSON evidence {path}: {error}") from error
    if not isinstance(value, expected_type):
        raise EvidenceError(
            f"Expected {expected_type.__name__} in {path}, got {type(value).__name__}"
        )
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise EvidenceError(f"Cannot hash evidence file {path}: {error}") from error
    return digest.hexdigest()


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EvidenceError(f"Evidence path must be a regular, non-symlink file: {path}")
    try:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
    except (OSError, ValueError) as error:
        raise EvidenceError(f"Evidence file is outside the study root: {path}") from error
    return {"sourcePath": relative, "bytes": size, "sha256": _sha256(path)}


def _case_ids(value: Any, label: str, run: Path) -> list[str]:
    if not isinstance(value, list) or not value:
        raise EvidenceError(f"{label} must be a non-empty list in {run}")
    if not all(isinstance(case_id, str) and CASE_ID_PATTERN.fullmatch(case_id) for case_id in value):
        raise EvidenceError(f"{label} contains an invalid case ID in {run}")
    if len(value) != len(set(value)):
        raise EvidenceError(f"{label} contains duplicate case IDs in {run}")
    return value


def _attempt_number(row: dict[str, Any], run: Path) -> int:
    attempts = row.get("attempts")
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise EvidenceError(f"Invalid attempt count for {row.get('id')!r} in {run}")
    return attempts


def _discover_attempts(run: Path, root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    discovered: dict[tuple[str, int], dict[str, Any]] = {}
    for case_dir in sorted((path for path in run.iterdir() if path.is_dir()), key=lambda path: path.name):
        for path in sorted(case_dir.iterdir(), key=lambda item: item.name):
            if not path.is_file():
                continue
            match = ATTEMPT_PATTERN.fullmatch(path.name)
            if not match:
                continue
            attempt = int(match.group(1))
            kind = "rawOutput" if match.group(2) else "metrics" if match.group(3) else "input"
            entry = discovered.setdefault((case_dir.name, attempt), {})
            if kind in entry:
                raise EvidenceError(f"Duplicate {kind} artifact for {case_dir.name} attempt {attempt} in {run}")
            record = _file_record(path, root)
            if kind == "metrics":
                metric = _load_json(path, dict)
                if "runtime" in metric:
                    record["runtime"] = metric["runtime"]
            entry[kind] = record
    return discovered


def _validate_completion(
    run: Path,
    summary: dict[str, Any] | None,
    rows: list[dict[str, Any]],
    planned: list[str],
    disposition: str,
) -> None:
    if disposition == "completed" and summary is None:
        raise EvidenceError(f"Completed run has no summary.json: {run}")
    if summary is None:
        return
    successes = sum(row.get("status") == "success" for row in rows)
    expected = {
        "runId": run.name,
        "total": len(rows),
        "success": successes,
        "failure": len(rows) - successes,
    }
    for field, expected_value in expected.items():
        if summary.get(field) != expected_value:
            raise EvidenceError(
                f"summary.json {field!r} mismatch in {run}: "
                f"expected {expected_value!r}, got {summary.get(field)!r}"
            )
    if disposition == "completed" and len(rows) != len(planned):
        raise EvidenceError(f"Completed run has {len(rows)} results for {len(planned)} cases: {run}")


def _inspect_run(
    run: Path, root: Path
) -> tuple[dict[str, Any], list[tuple[Path, Path, str]], dict[str, Any] | None]:
    meta = _load_json(run / "experiment.json", dict)
    early_rejected = meta.get("earlyRejected") is True
    summary_path = run / "summary.json"
    disposition = "early_rejected" if early_rejected else "completed"

    for name in REQUIRED_RUN_FILES:
        path = run / name
        if path.is_symlink() or not path.is_file():
            raise EvidenceError(f"Eligible run is missing required evidence {name}: {run}")

    config = _load_json(run / "run_config.json", dict)
    rows = _load_json(run / "results.json", list)
    if not all(isinstance(row, dict) for row in rows):
        raise EvidenceError(f"results.json must contain objects: {run}")
    summary = _load_json(summary_path, dict) if summary_path.is_file() else None
    for optional_json in (
        "host_summary.json",
        "attempt_evidence.json",
        "adaptive_rejection_decision.json",
    ):
        path = run / optional_json
        if path.exists():
            _load_json(path, dict)

    planned = _case_ids(meta.get("cases"), "experiment cases", run)
    configured = _case_ids(config.get("cases"), "configured cases", run)
    if planned != configured:
        raise EvidenceError(f"Experiment/config case order or membership differs: {run}")

    observed: list[str] = []
    rows_by_id: dict[str, dict[str, Any]] = {}
    max_attempts = meta.get("repairs")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 0:
        raise EvidenceError(f"Invalid repair count in {run}")
    for row in rows:
        case_id = row.get("id")
        if not isinstance(case_id, str) or not CASE_ID_PATTERN.fullmatch(case_id):
            raise EvidenceError(f"Invalid result case ID in {run}: {case_id!r}")
        if case_id in rows_by_id:
            raise EvidenceError(f"Duplicate result case ID {case_id!r} in {run}")
        attempts = _attempt_number(row, run)
        if attempts > max_attempts + 1:
            raise EvidenceError(f"Attempt count exceeds repair policy for {case_id} in {run}")
        observed.append(case_id)
        rows_by_id[case_id] = row

    unexpected_cases = [case_id for case_id in observed if case_id not in set(planned)]
    missing_cases = [case_id for case_id in planned if case_id not in rows_by_id]
    if unexpected_cases:
        raise EvidenceError(f"Unexpected result case IDs in {run}: {unexpected_cases}")
    if missing_cases and disposition == "completed":
        raise EvidenceError(f"Completed run is missing case results in {run}: {missing_cases}")
    _validate_completion(run, summary, rows, planned, disposition)

    prompt = _file_record(run / "experiment_prompt.txt", root)
    expected_prompt_hash = meta.get("promptSha256")
    if not isinstance(expected_prompt_hash, str) or prompt["sha256"] != expected_prompt_hash:
        raise EvidenceError(f"experiment_prompt.txt hash does not match experiment.json: {run}")
    if "promptSha256" in config and config["promptSha256"] != expected_prompt_hash:
        raise EvidenceError(f"Configured prompt hash does not match experiment.json: {run}")
    if "corpusSha256" in config and config["corpusSha256"] != meta.get("corpusSha256"):
        raise EvidenceError(f"Configured corpus hash does not match experiment.json: {run}")

    input_required = bool(meta.get("inputScaffold") or meta.get("recordInputs"))
    discovered = _discover_attempts(run, root)
    expected_attempts = {
        (case_id, attempt)
        for case_id, row in rows_by_id.items()
        for attempt in range(1, _attempt_number(row, run) + 1)
    }
    discovered_cases = {case_id for case_id, _ in discovered}
    unexpected_artifact_cases = sorted(discovered_cases - set(planned))
    if unexpected_artifact_cases:
        raise EvidenceError(f"Attempt artifacts exist for unplanned cases in {run}: {unexpected_artifact_cases}")

    required_parts = ("rawOutput", "metrics", "input") if input_required else ("rawOutput", "metrics")
    missing_declared: list[str] = []
    attempts_manifest: list[dict[str, Any]] = []
    all_attempts = sorted(expected_attempts | set(discovered), key=lambda item: (item[0], item[1]))
    for case_id, attempt in all_attempts:
        artifacts = discovered.get((case_id, attempt), {})
        declared = (case_id, attempt) in expected_attempts
        missing_parts = [part for part in required_parts if part not in artifacts]
        if declared:
            missing_declared.extend(
                f"{case_id}/attempt_{attempt}:{part}" for part in missing_parts
            )
        attempts_manifest.append(
            {
                "caseId": case_id,
                "attempt": attempt,
                "classification": "completed_result_attempt" if declared else "inflight_or_orphaned",
                "artifacts": {part: artifacts.get(part) for part in ("rawOutput", "metrics", "input")},
                "missingRequiredParts": missing_parts,
            }
        )
    if missing_declared:
        raise EvidenceError(f"Declared model attempts have missing evidence in {run}: {missing_declared}")

    inflight_keys = sorted(set(discovered) - expected_attempts, key=lambda item: (item[0], item[1]))
    if inflight_keys and disposition != "early_rejected":
        raise EvidenceError(f"Completed run has unexplained attempt artifacts in {run}: {inflight_keys}")

    copied_files: list[dict[str, Any]] = []
    copy_plan: list[tuple[Path, Path, str]] = []
    for name in (*REQUIRED_RUN_FILES, *OPTIONAL_RUN_FILES):
        source = run / name
        if not source.exists():
            continue
        record = _file_record(source, root)
        destination = Path("runs") / run.name / name
        copied_files.append({**record, "portablePath": destination.as_posix()})
        copy_plan.append((source, destination, record["sha256"]))

    metric_count = sum("metrics" in artifacts for artifacts in discovered.values())
    raw_count = sum("rawOutput" in artifacts for artifacts in discovered.values())
    input_count = sum("input" in artifacts for artifacts in discovered.values())
    run_manifest: dict[str, Any] = {
        "run": run.name,
        "disposition": disposition,
        "portablePath": (Path("runs") / run.name).as_posix(),
        "plannedCaseIds": planned,
        "observedCaseIds": observed,
        "missingCaseIds": missing_cases,
        "unexpectedCaseIds": unexpected_cases,
        "inputEvidenceRequired": input_required,
        "resultCounts": {
            "planned": len(planned),
            "observed": len(rows),
            "success": sum(row.get("status") == "success" for row in rows),
            "nonSuccess": sum(row.get("status") != "success" for row in rows),
        },
        "modelCounts": {
            "declaredCompletedCalls": len(expected_attempts),
            "capturedMetricFiles": metric_count,
            "capturedRawOutputs": raw_count,
            "capturedInputs": input_count,
            "inflightOrOrphanedAttempts": len(inflight_keys),
        },
        "missingDeclaredAttemptArtifacts": missing_declared,
        "inflightOrOrphanedAttempts": [
            {"caseId": case_id, "attempt": attempt} for case_id, attempt in inflight_keys
        ],
        "attempts": attempts_manifest,
        "copiedFiles": copied_files,
    }

    final_spec = None
    if (
        disposition == "completed"
        and meta.get("packagedPrompt") is True
        and set(planned) == set(FINAL_CASE_IDS)
    ):
        final_spec = _inspect_final50(run, root, meta, config, rows_by_id, discovered)
    return run_manifest, copy_plan, final_spec


def _inspect_final50(
    run: Path,
    root: Path,
    meta: dict[str, Any],
    config: dict[str, Any],
    rows_by_id: dict[str, dict[str, Any]],
    discovered: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    if meta.get("packagedPrompt") is not True or config.get("promptPath") != "aar_asset":
        raise EvidenceError(f"Final-50 run {run.name} is not a recorded packaged-prompt run")
    if meta.get("recordInputs") is not True or config.get("recordInputs") is not True:
        raise EvidenceError(f"Final-50 run {run.name} must record effective per-case inputs")

    cases: list[dict[str, Any]] = []
    copies: list[tuple[Path, Path, str]] = []
    failed_case_ids: list[str] = []
    for case_id, row in rows_by_id.items():
        if row.get("status") != "success":
            failed_case_ids.append(case_id)
            continue
        successful_attempt = _attempt_number(row, run)
        attempt = discovered.get((case_id, successful_attempt), {})
        for part in ("rawOutput", "metrics", "input"):
            if part not in attempt:
                raise EvidenceError(
                    f"Successful final case {case_id} is missing {part} for attempt {successful_attempt}"
                )

        case_dir = run / case_id
        document_records: dict[str, dict[str, Any]] = {}
        for name in ("output.a2ui.json", "output.express"):
            path = case_dir / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
                raise EvidenceError(f"Successful final case {case_id} is missing final document {name}")
            if name.endswith(".json"):
                document = _load_json(path, list)
                if not document:
                    raise EvidenceError(f"Successful final case {case_id} has an empty {name}")
            else:
                try:
                    express = path.read_text(encoding="utf-8")
                except (OSError, UnicodeError) as error:
                    raise EvidenceError(f"Cannot read final document {path}: {error}") from error
                if "<a2ui>" not in express or "</a2ui>" not in express:
                    raise EvidenceError(f"Successful final case {case_id} has an incomplete {name}")
            document_records[name] = _file_record(path, root)

        destination = Path("final50") / run.name / case_id
        raw_name = f"attempt_{successful_attempt}.txt"
        metrics_name = f"attempt_{successful_attempt}_metrics.json"
        copies.extend(
            [
                (
                    case_dir / "output.a2ui.json",
                    destination / "output.a2ui.json",
                    document_records["output.a2ui.json"]["sha256"],
                ),
                (
                    case_dir / "output.express",
                    destination / "output.express",
                    document_records["output.express"]["sha256"],
                ),
                (case_dir / raw_name, destination / raw_name, attempt["rawOutput"]["sha256"]),
                (
                    case_dir / metrics_name,
                    destination / metrics_name,
                    attempt["metrics"]["sha256"],
                ),
            ]
        )
        cases.append(
            {
                "caseId": case_id,
                "status": "success",
                "successfulAttempt": successful_attempt,
                "inputSha256": attempt["input"]["sha256"],
                "inputBytes": attempt["input"]["bytes"],
                "copiedArtifacts": {
                    "output.a2ui.json": document_records["output.a2ui.json"],
                    "output.express": document_records["output.express"],
                    raw_name: attempt["rawOutput"],
                    metrics_name: attempt["metrics"],
                },
            }
        )
    return {
        "status": "complete_collected",
        "run": run.name,
        "successfulCases": len(cases),
        "failedCaseIds": failed_case_ids,
        "cases": cases,
        "copies": copies,
    }


def _copy_verified(source: Path, destination: Path, expected_sha256: str | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(source, destination)
    except OSError as error:
        raise EvidenceError(f"Cannot copy {source} to {destination}: {error}") from error
    source_hash = expected_sha256 or _sha256(source)
    destination_hash = _sha256(destination)
    if destination_hash != source_hash:
        raise EvidenceError(f"Copied evidence hash mismatch: {destination}")


def _write_final_case_evidence(stage: Path, final_spec: dict[str, Any]) -> None:
    for case in final_spec["cases"]:
        destination = stage / "final50" / final_spec["run"] / case["caseId"] / "evidence.json"
        payload = {key: value for key, value in case.items() if key != "copiedArtifacts"}
        payload["copiedArtifacts"] = {
            name: {"bytes": record["bytes"], "sha256": record["sha256"]}
            for name, record in case["copiedArtifacts"].items()
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def collect(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    if not root.is_dir():
        raise EvidenceError(f"Raw study root is not a directory: {root}")
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise EvidenceError("Portable output must not be inside the raw study directory")
    if output.exists():
        raise EvidenceError(f"Portable output already exists; refusing stale or mixed evidence: {output}")

    run_manifests: list[dict[str, Any]] = []
    copy_plan: list[tuple[Path, Path, str]] = []
    skipped_runs: list[dict[str, str]] = []
    final_specs: list[dict[str, Any]] = []

    for run in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name):
        experiment_path = run / "experiment.json"
        if not experiment_path.is_file():
            continue
        meta = _load_json(experiment_path, dict)
        eligible = (run / "summary.json").is_file() or meta.get("earlyRejected") is True
        if not eligible:
            skipped_runs.append(
                {"run": run.name, "reason": "not completed and not explicitly earlyRejected"}
            )
            continue
        run_manifest, run_copies, candidate_final = _inspect_run(run, root)
        run_manifests.append(run_manifest)
        copy_plan.extend(run_copies)
        if candidate_final is not None:
            final_specs.append(candidate_final)
            copy_plan.extend(candidate_final["copies"])

    if not run_manifests:
        raise EvidenceError(f"No completed or explicitly early-rejected runs found under {root}")

    totals = {
        "includedRuns": len(run_manifests),
        "completedRuns": sum(run["disposition"] == "completed" for run in run_manifests),
        "earlyRejectedRuns": sum(run["disposition"] == "early_rejected" for run in run_manifests),
        "plannedCases": sum(run["resultCounts"]["planned"] for run in run_manifests),
        "observedCases": sum(run["resultCounts"]["observed"] for run in run_manifests),
        "declaredCompletedModelCalls": sum(
            run["modelCounts"]["declaredCompletedCalls"] for run in run_manifests
        ),
        "capturedMetricFiles": sum(
            run["modelCounts"]["capturedMetricFiles"] for run in run_manifests
        ),
        "inflightOrOrphanedAttempts": sum(
            run["modelCounts"]["inflightOrOrphanedAttempts"] for run in run_manifests
        ),
    }
    final_status: dict[str, Any] = {
        "status": "complete_collected" if final_specs else "not_present",
        "runs": [
            {key: value for key, value in final_spec.items() if key != "copies"}
            for final_spec in final_specs
        ],
    }
    manifest = {
        "schemaVersion": 1,
        "createdUtc": datetime.now(timezone.utc).isoformat(),
        "sourceRoot": str(root),
        "totals": totals,
        "skippedRuns": skipped_runs,
        "runs": run_manifests,
        "final50": final_status,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    if stage.resolve().parent != output.parent or stage.is_symlink():
        raise EvidenceError(f"Staging directory escaped the intended output parent: {stage}")
    try:
        copied_destinations: set[Path] = set()
        for source, relative_destination, expected_sha256 in copy_plan:
            if relative_destination in copied_destinations:
                raise EvidenceError(f"Duplicate portable destination: {relative_destination}")
            copied_destinations.add(relative_destination)
            _copy_verified(source, stage / relative_destination, expected_sha256)
        for final_spec in final_specs:
            _write_final_case_evidence(stage, final_spec)
        manifest_path = stage / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        prohibited = [
            path.relative_to(stage).as_posix()
            for path in stage.rglob("*")
            if path.is_file() and path.suffix.lower() in BANNED_SUFFIXES
        ]
        if prohibited:
            raise EvidenceError(f"Prohibited binary/media artifacts entered portable output: {prohibited}")
        if stage.resolve().parent != output.parent or output.resolve().parent != output.parent:
            raise EvidenceError("Staging/output paths changed before publication")
        stage.rename(output)
    except Exception:
        if not stage.is_symlink() and stage.resolve().parent == output.parent:
            shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Raw prompt-study run directory")
    parser.add_argument("--output", required=True, type=Path, help="New portable evidence directory")
    args = parser.parse_args()
    try:
        manifest = collect(args.root, args.output)
    except (EvidenceError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "totals": manifest["totals"],
                "final50": manifest["final50"]["status"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
