"""Explicit repeated-case benchmark revisions; never repair a reference target."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ir_training.data.express_preparation import PreparationError, TASK_PREFIX, prepare_row


def read_rows_strict(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: malformed JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def serialize_rows(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for row in rows).encode("utf-8")


def response_text(row: Mapping[str, Any]) -> str:
    text = row.get("response_text")
    if not isinstance(text, str) or not text.strip():
        messages = row.get("messages") or []
        text = next((message["content"][len(TASK_PREFIX):] for message in reversed(messages)
                     if isinstance(message, dict) and message.get("role") == "user"
                     and isinstance(message.get("content"), str)
                     and message["content"].startswith(TASK_PREFIX)), "")
    if not text.strip():
        raise ValueError("Row has no source response")
    return text


def source_identity_record(row: Mapping[str, Any]) -> dict[str, str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    source = str(row.get("source_id") or metadata.get("source_id") or row.get("query_id") or metadata.get("query_id") or "").strip()
    return {
        "source_id": source,
        "query_id": str(row.get("query_id") or metadata.get("query_id") or source),
        "response_id": str(row.get("response_id") or metadata.get("response_id") or ""),
        "row_id": str(row.get("id") or ""),
        "response_sha256": hashlib.sha256(" ".join(response_text(row).split()).encode("utf-8")).hexdigest(),
    }


def _matches(row: Mapping[str, Any], identity: str) -> bool:
    record = source_identity_record(row)
    return identity in {record[key] for key in ("source_id", "query_id", "response_id", "row_id")}


def build_replacement(
    rows: list[dict[str, Any]], *, original_source_sha256: str,
    failed_identity: str, benchmark_id: str, donor_identity: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replace exactly one invalid slot with an unchanged valid donor case.

    If no donor is named, prefer a valid case in the failed case's intent;
    otherwise choose the first strictly valid source row. Source identities
    stay unchanged, so evaluators can score unique cases separately.
    """
    if len(rows) != 32:
        raise ValueError("The replacement revision requires exactly 32 input rows")
    if not benchmark_id or not failed_identity:
        raise ValueError("benchmark_id and failed_identity are required")
    if len(original_source_sha256) != 64 or any(c not in "0123456789abcdef" for c in original_source_sha256):
        raise ValueError("Original source SHA-256 is required")
    identities = [source_identity_record(row) for row in rows]
    if any(not item["source_id"] for item in identities) or len({item["source_id"] for item in identities}) != 32:
        raise ValueError("Original benchmark must contain 32 unique source IDs")
    if len({item["response_sha256"] for item in identities}) != 32:
        raise ValueError("Original benchmark must contain 32 unique source responses")
    failed_positions = [i for i, row in enumerate(rows) if _matches(row, failed_identity)]
    if len(failed_positions) != 1:
        raise ValueError("failed_identity must identify exactly one source row")
    failed_position = failed_positions[0]
    valid = []
    failures = {}
    for index, row in enumerate(rows):
        try:
            prepare_row(row, "root-first")
            valid.append(index)
        except PreparationError as exc:
            failures[index] = {"reason": exc.reason, "detail": str(exc)}
    if set(failures) != {failed_position}:
        raise ValueError("Exactly the named failed reference must be invalid; do not hide other failures")
    failed = rows[failed_position]
    if donor_identity:
        candidates = [i for i in valid if _matches(rows[i], donor_identity)]
        if len(candidates) != 1:
            raise ValueError("donor_identity must identify one strictly valid source row")
        donor_position = candidates[0]
        selection = "explicit_donor"
    else:
        intent = failed.get("intent_bucket") or (failed.get("metadata") or {}).get("intent_bucket")
        candidates = [i for i in valid if intent and (rows[i].get("intent_bucket") or (rows[i].get("metadata") or {}).get("intent_bucket")) == intent]
        donor_position = (candidates or valid)[0]
        selection = "same_intent_first_valid" if candidates else "first_strictly_valid"
    output = deepcopy(rows)
    donor = rows[donor_position]
    occurrence_id = f"{benchmark_id}:repeat:{identities[donor_position]['source_id']}:slot{failed_position + 1}"
    output[failed_position] = deepcopy(donor)
    output[failed_position]["id"] = occurrence_id
    for index, row in enumerate(output):
        row.setdefault("metadata", {})["benchmark"] = {
            "id": benchmark_id, "benchmark_id": benchmark_id, "kind": "explicit_repeated_case",
            "row_count": 32, "unique_source_count": 31,
            "occurrence_id": row.get("id"), "is_repeated_occurrence": index == failed_position,
        }
    output[failed_position].setdefault("metadata", {}).update(
        repeated_from=deepcopy(identities[donor_position]), replaces=deepcopy(identities[failed_position]),
    )
    manifest = {
        "schema_version": 1, "benchmark_id": benchmark_id,
        "benchmark_kind": "explicit_repeated_case", "row_count": 32,
        "unique_source_count": 31, "duplicate_occurrence_count": 1,
        "original_source_sha256": original_source_sha256,
        "output_sha256": hashlib.sha256(serialize_rows(output)).hexdigest(),
        "output_file": "golden32.jsonl", "donor_selection": selection,
        "replacements": [{"slot_1based": failed_position + 1, "donor_slot_1based": donor_position + 1,
                          "failed": identities[failed_position], "donor": identities[donor_position],
                          "occurrence_id": occurrence_id, "original_failure": failures[failed_position]}],
        "excluded_sources": [identities[failed_position]],
        "strict_validation": {"express": True, "wire_schema": True, "root_reachability": 1.0, "semantic_roundtrip": True},
        "scoring_policy": {"primary": "unique_source_macro", "secondary": "all_occurrences_macro",
                           "independent_case_count": 31, "independent_test_set": False,
                           "note": "32 evaluation occurrences contain 31 unique cases. The repeated donor is not a new reference or independent observation."},
    }
    validate_replacement_rows(output, manifest)
    return output, manifest


def validate_replacement_rows(rows: list[dict[str, Any]], manifest: Mapping[str, Any]) -> None:
    """Validate raw or canonically prepared cases against the repeat contract.

    The caller binds raw bytes to ``output_sha256`` or binds prepared bytes
    through its preparation manifest. Re-serialization may legitimately
    change those bytes, but must preserve the exact donor/repeat relationship.
    """
    if manifest.get("schema_version") != 1 or manifest.get("benchmark_kind") != "explicit_repeated_case":
        raise ValueError("Unsupported repeated-case benchmark manifest")
    if (manifest.get("row_count"), manifest.get("unique_source_count"), manifest.get("duplicate_occurrence_count")) != (32, 31, 1):
        raise ValueError("Expected 32 occurrences and 31 unique source cases")
    if len(rows) != 32:
        raise ValueError("Repeated-case benchmark row count differs from manifest")
    for row in rows:
        prepare_row(row, "root-first")
    identities = [source_identity_record(row) for row in rows]
    if len({item["source_id"] for item in identities}) != 31 or len({item["response_sha256"] for item in identities}) != 31:
        raise ValueError("Repeated-case unique identity count differs")
    if len({row.get("id") for row in rows}) != 32:
        raise ValueError("Occurrence IDs must be unique")
    replacements = manifest.get("replacements") or []
    if len(replacements) != 1:
        raise ValueError("Exactly one declared replacement is supported")
    replacement = replacements[0]
    slot, donor_slot = replacement["slot_1based"] - 1, replacement["donor_slot_1based"] - 1
    if slot == donor_slot or min(slot, donor_slot) < 0 or max(slot, donor_slot) >= 32:
        raise ValueError("Invalid repeated-case slots")
    repeat, donor = rows[slot], rows[donor_slot]
    for index, row in enumerate(rows):
        expected_benchmark = {
            "id": manifest["benchmark_id"], "benchmark_id": manifest["benchmark_id"],
            "kind": "explicit_repeated_case", "row_count": 32, "unique_source_count": 31,
            "occurrence_id": row.get("id"), "is_repeated_occurrence": index == slot,
        }
        if (row.get("metadata") or {}).get("benchmark") != expected_benchmark:
            raise ValueError("Row benchmark occurrence metadata differs")
    if source_identity_record(donor) != replacement["donor"]:
        raise ValueError("Declared donor identity differs from source row")
    if repeat.get("id") != replacement["occurrence_id"]:
        raise ValueError("Repeated occurrence ID differs")
    expected = deepcopy(donor)
    expected["id"] = replacement["occurrence_id"]
    expected.setdefault("metadata", {}).update(
        repeated_from=deepcopy(replacement["donor"]), replaces=deepcopy(replacement["failed"]),
        benchmark={"id": manifest["benchmark_id"], "benchmark_id": manifest["benchmark_id"], "kind": "explicit_repeated_case",
                   "row_count": 32, "unique_source_count": 31,
                   "occurrence_id": replacement["occurrence_id"], "is_repeated_occurrence": True},
    )
    if repeat != expected:
        raise ValueError("Repeated case changed the donor payload or provenance")
    if manifest.get("excluded_sources") != [replacement["failed"]]:
        raise ValueError("Failed source must remain reserved from training")
    if replacement["failed"]["source_id"] in {item["source_id"] for item in identities}:
        raise ValueError("Failed source is still present in the benchmark")
