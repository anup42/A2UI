"""Stage byte-identical, previously revalidated GenUI documents for JSON replay.

This tool does not run an LLM and does not turn renderer revalidation records into
generation-success records. It validates the saved raw-model replay evidence, copies
only the accepted documents and their original reports, and writes an explicit
provenance manifest.

Usage:
  python tools/stage_revalidated_documents.py --source-run SOURCE --output OUTPUT
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile


IDS = [f"BXP-{index:03}" for index in range(1, 51)]
SHA256 = re.compile(r"[0-9a-f]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique_object)


def exact_ids(values, label):
    require(isinstance(values, list) and all(isinstance(value, str) for value in values),
            f"Invalid {label} IDs")
    counts = Counter(values)
    missing = sorted(set(IDS) - counts.keys())
    extra = sorted(counts.keys() - set(IDS))
    duplicate = sorted(key for key, count in counts.items() if count != 1)
    require(not missing and not extra and not duplicate and len(values) == len(IDS),
            f"{label}: missing={missing}, unexpected={extra}, duplicate={duplicate}")


def integer(value, expected, label):
    require(type(value) is int and value == expected, f"{label} must be integer {expected}")


def validate_prior_report(case_id, report):
    require(isinstance(report, dict), f"Invalid source report for {case_id}")
    require(report.get("id") == case_id, f"Source report ID mismatch for {case_id}")
    require(report.get("kind") == "captured_model_revalidation",
            f"{case_id} is not a captured-model revalidation")
    require(report.get("replayMode") == "raw_model", f"{case_id} is not a raw-model revalidation")
    require(report.get("status") == "rendered", f"{case_id} prior revalidation was not rendered")
    require(report.get("issues") == [], f"{case_id} prior revalidation has issues")
    integer(report.get("capturedProviderCalls"), 1, f"{case_id}.capturedProviderCalls")
    integer(report.get("modelCalls"), 0, f"{case_id}.modelCalls")
    require(report.get("inferenceEvaluated") is False,
            f"{case_id} must explicitly exclude inference evaluation")
    source_hash = report.get("sourceJsonSha256")
    require(isinstance(source_hash, str) and SHA256.fullmatch(source_hash),
            f"{case_id}.sourceJsonSha256 is invalid")


def collect(source: Path, output: Path):
    source, output = source.resolve(), output.resolve()
    require(source.is_dir(), f"Missing source run: {source}")
    require(source != output, "Source and output directories must differ")
    require(not output.exists(), f"Refusing to overwrite existing output: {output}")

    replay_results_path = source / "replay_results.json"
    replay_results_bytes = replay_results_path.read_bytes()
    replay_results = read_json(replay_results_path)
    require(isinstance(replay_results, list) and all(isinstance(row, dict) for row in replay_results),
            "Invalid source replay_results.json")
    exact_ids([row.get("id") for row in replay_results], "source replay results")
    reports = {row["id"]: row for row in replay_results}
    exact_ids(sorted(path.name for path in source.iterdir()
                     if path.is_dir() and path.name.startswith("BXP-")), "source directories")

    cases = []
    payloads = {}
    for case_id in IDS:
        case_dir = source / case_id
        document_path = case_dir / "revalidated.output.a2ui.json"
        report_path = case_dir / "replay_result.json"
        document_bytes = document_path.read_bytes()
        report_bytes = report_path.read_bytes()
        report = read_json(report_path)
        require(report == reports[case_id],
                f"{case_id} replay_result.json differs from source replay_results.json")
        validate_prior_report(case_id, report)
        document_hash = sha(document_bytes)
        recorded_revalidated_hash = report.get("revalidatedJsonSha256")
        if recorded_revalidated_hash is not None:
            require(recorded_revalidated_hash == document_hash,
                    f"{case_id}.revalidatedJsonSha256 does not match revalidated output")
        # Parsing establishes that staging cannot silently copy an empty/non-JSON artifact.
        parsed_document = json.loads(document_bytes.decode("utf-8-sig"), object_pairs_hook=unique_object)
        require(isinstance(parsed_document, (dict, list)), f"{case_id} revalidated output is not JSON")
        payloads[case_id] = (document_bytes, report_bytes)
        cases.append({
            "id": case_id,
            "originalDocumentRelativePath": f"{case_id}/revalidated.output.a2ui.json",
            "stagedDocumentRelativePath": f"{case_id}/output.a2ui.json",
            "documentSha256": document_hash,
            "originalReportRelativePath": f"{case_id}/replay_result.json",
            "stagedReportRelativePath": f"{case_id}/replay_result.json",
            "sourceReportSha256": sha(report_bytes),
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        original_relative = Path(os.path.relpath(source, start=output)).as_posix()
    except ValueError as error:
        raise ValueError("Source and output must share a filesystem drive for relative provenance") from error
    manifest = {
        "schemaVersion": 1,
        "kind": "staged_revalidated_documents",
        "stagedRunId": output.name,
        "sourceRunId": source.name,
        "originalRunRelativePath": original_relative,
        "caseCount": len(IDS),
        "modelCalls": 0,
        "inferenceEvaluated": False,
        "llmInvoked": False,
        "originalReplayResultsRelativePath": "replay_results.json",
        "stagedReplayResultsRelativePath": "replay_results.json",
        "replayResultsSha256": sha(replay_results_bytes),
        "cases": cases,
        "createdUtc": datetime.now(timezone.utc).isoformat(),
    }

    temporary = Path(tempfile.mkdtemp(prefix=f"{output.name}.staging-", dir=output.parent))
    try:
        for case_id in IDS:
            case_output = temporary / case_id
            case_output.mkdir()
            document_bytes, report_bytes = payloads[case_id]
            (case_output / "output.a2ui.json").write_bytes(document_bytes)
            (case_output / "replay_result.json").write_bytes(report_bytes)
        (temporary / "replay_results.json").write_bytes(replay_results_bytes)
        (temporary / "source_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = collect(args.source_run, args.output)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "kind": manifest["kind"],
        "sourceRunId": manifest["sourceRunId"],
        "stagedRunId": manifest["stagedRunId"],
        "cases": manifest["caseCount"],
        "modelCalls": manifest["modelCalls"],
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
