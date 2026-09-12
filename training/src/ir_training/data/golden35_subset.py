"""Frozen, explicitly approved Golden35 membership from the historical 50-case source."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
from typing import Any, Mapping

from ir_training.data.express_preparation import prepare_row
from ir_training.data.golden_replacement import response_text, serialize_rows, source_identity_record
from ir_training.data.url_preprocess import preprocess_training_urls


BENCHMARK_ID = "golden35_v1"
SOURCE_RUN = "dataset/data/runs/golden50_g25pro_20260309_204033_stitch_compare_20260429_hybrid_r4"
SOURCE_GENUI_SHA256 = "64946a64aab0745cf5137f4831c4ca1902bff312a01169c2bd67e27cfde6c5fe"
SOURCE_RESPONSES_SHA256 = "5a375c0526446218ba47321c7a77ffe7979c33bcdc7ba8a59397ad14f2453fc3"
APPROVED_SOURCE_IDS = tuple(f"q_{index:06d}" for index in (
    2, 3, 5, 7, 8, 9, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27,
    29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 45, 47, 49,
))
FROZEN_EXCLUSION_REASONS = {
    "q_000001": "wire_schema_invalid: Text.text uses unsupported $item binding (siteName)",
    "q_000004": "express_materialization_error: Table props genre, mood, subtitle are unsupported",
    "q_000006": "wire_schema_invalid: Table.entityMedia object does not satisfy the wire contract",
    "q_000010": "wire_schema_invalid: Image.url uses unsupported $item binding (imageUrl)",
    "q_000011": "express_materialization_error: Table.sourceFormat is unsupported",
    "q_000012": "express_materialization_error: Table.sourceFormat is unsupported",
    "q_000017": "wire_schema_invalid: Text.text uses unsupported $item binding (stepNumber)",
    "q_000018": "wire_schema_invalid: Text.text uses unsupported $template binding",
    "q_000019": "wire_schema_invalid: Icon.name uses unsupported $item binding (icon)",
    "q_000028": "express_materialization_error: unsupported justify distribution",
    "q_000035": "express_materialization_error: Table.sourceFormat is unsupported",
    "q_000044": "wire_schema_invalid: conditional visible object does not satisfy the wire contract",
    "q_000046": "express_materialization_error: Table.sourceFormat is unsupported",
    "q_000048": "wire_schema_invalid: Text.text uses unsupported $item binding (time)",
    "q_000050": "express_materialization_error: unsupported justify distribution",
}


def _membership_hash(source_ids: list[str] | tuple[str, ...]) -> str:
    return hashlib.sha256(("\n".join(source_ids) + "\n").encode("utf-8")).hexdigest()


def build_golden35_subset(
    prepared_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]], *,
    source_genui_sha256: str, source_responses_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the approved identities, never the next 35 rows that happen to pass."""
    if source_genui_sha256 != SOURCE_GENUI_SHA256 or source_responses_sha256 != SOURCE_RESPONSES_SHA256:
        raise ValueError("Golden35 original source hashes differ from the frozen revision")
    originals = {source_identity_record(row)["source_id"]: row for row in source_rows}
    original_membership = set(APPROVED_SOURCE_IDS) | set(FROZEN_EXCLUSION_REASONS)
    if len(source_rows) != 50 or set(originals) != original_membership:
        raise ValueError("Golden35 original source must retain all 50 original identities")
    candidates = {}
    for row in prepared_rows:
        identity = source_identity_record(row)["source_id"]
        if identity in candidates:
            raise ValueError("Duplicate prepared source identity")
        candidates[identity] = row
    missing = set(APPROVED_SOURCE_IDS) - set(candidates)
    if missing:
        raise ValueError(f"Approved Golden35 sources no longer materialize strictly: {sorted(missing)}")
    rows = [deepcopy(candidates[identity]) for identity in APPROVED_SOURCE_IDS]
    accepted = []
    for row in rows:
        _, target, _ = prepare_row(row, "root-first")
        identity = source_identity_record(row)
        row.setdefault("metadata", {})["benchmark"] = {
            "id": BENCHMARK_ID, "benchmark_id": BENCHMARK_ID, "kind": "fixed_strict_subset",
            "row_count": 35, "unique_source_count": 35, "occurrence_id": row["id"],
            "is_repeated_occurrence": False,
        }
        accepted.append({**identity, "semantic_sha256": target.semantic_sha256,
                         "intent_bucket": row.get("intent_bucket") or "unknown"})
    excluded = []
    for source_id, reason in FROZEN_EXCLUSION_REASONS.items():
        row = originals[source_id]
        identity = source_identity_record(row)
        masked = preprocess_training_urls(response_text(row), {}, enabled=True).response_text
        masked_hash = hashlib.sha256(" ".join(masked.split()).encode("utf-8")).hexdigest()
        excluded.append({**identity, "raw_response_sha256": identity["response_sha256"],
                         "response_sha256": masked_hash, "masked_response_sha256": masked_hash,
                         "response_sha256s": sorted({identity["response_sha256"], masked_hash}),
                         "intent_bucket": row.get("intent_bucket") or "unknown", "reason": reason})
    manifest = {
        "schema_version": 1, "benchmark_id": BENCHMARK_ID, "benchmark_kind": "fixed_strict_subset",
        "row_count": 35, "unique_source_count": 35, "duplicate_occurrence_count": 0,
        "original_source_row_count": 50, "excluded_source_count": 15,
        "source_run": SOURCE_RUN, "source_genui_sha256": SOURCE_GENUI_SHA256,
        "source_responses_sha256": SOURCE_RESPONSES_SHA256,
        "output_file": "golden35.jsonl", "output_sha256": hashlib.sha256(serialize_rows(rows)).hexdigest(),
        "accepted_source_ids": list(APPROVED_SOURCE_IDS),
        "accepted_membership_sha256": _membership_hash(APPROVED_SOURCE_IDS),
        "accepted_sources": accepted, "excluded_sources": excluded,
        "intent_counts": dict(sorted(Counter(str(row.get("intent_bucket") or "unknown") for row in rows).items())),
        "strict_validation": {"express": True, "wire_schema": True, "root_reachability": 1.0, "semantic_roundtrip": True},
        "scoring_policy": {"primary": "all_35_unique_cases_macro", "independent_case_count": 35,
                           "note": "A fixed 35-case strict-valid subset; do not label it Golden50 or compare its aggregate directly with the original 50-case cohort."},
    }
    validate_golden35_rows(rows, manifest)
    return rows, manifest


def validate_golden35_rows(rows: list[dict[str, Any]], manifest: Mapping[str, Any]) -> None:
    """Validate raw or token-prepared rows; the caller separately binds file bytes."""
    if manifest.get("schema_version") != 1 or manifest.get("benchmark_kind") != "fixed_strict_subset" or manifest.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError("Unsupported Golden35 benchmark revision")
    if (manifest.get("row_count"), manifest.get("unique_source_count"), manifest.get("duplicate_occurrence_count")) != (35, 35, 0):
        raise ValueError("Golden35 requires exactly 35 unique cases")
    if manifest.get("source_genui_sha256") != SOURCE_GENUI_SHA256 or manifest.get("source_responses_sha256") != SOURCE_RESPONSES_SHA256:
        raise ValueError("Golden35 source pins differ")
    if manifest.get("accepted_source_ids") != list(APPROVED_SOURCE_IDS) or manifest.get("accepted_membership_sha256") != _membership_hash(APPROVED_SOURCE_IDS):
        raise ValueError("Golden35 approved membership differs")
    accepted = manifest.get("accepted_sources") or []
    excluded = manifest.get("excluded_sources") or []
    if len(rows) != 35 or len(accepted) != 35 or len(excluded) != 15 or {item["source_id"] for item in excluded} != set(FROZEN_EXCLUSION_REASONS):
        raise ValueError("Golden35 row or exclusion count differs")
    observed = [source_identity_record(row) for row in rows]
    if {item["source_id"] for item in observed} != set(APPROVED_SOURCE_IDS) or len({item["response_sha256"] for item in observed}) != 35:
        raise ValueError("Golden35 source membership or response uniqueness differs")
    expected_by_id = {item["source_id"]: item for item in accepted}
    if set(expected_by_id) != set(APPROVED_SOURCE_IDS):
        raise ValueError("Golden35 accepted identity evidence differs")
    for row, identity in zip(rows, observed):
        expected = expected_by_id[identity["source_id"]]
        _, target, _ = prepare_row(row, "root-first")
        if any(expected.get(key) != value for key, value in identity.items()) or target.semantic_sha256 != expected.get("semantic_sha256"):
            raise ValueError("Golden35 source/target identity changed")
        benchmark = (row.get("metadata") or {}).get("benchmark") or {}
        if benchmark != {"id": BENCHMARK_ID, "benchmark_id": BENCHMARK_ID, "kind": "fixed_strict_subset",
                         "row_count": 35, "unique_source_count": 35, "occurrence_id": row.get("id"), "is_repeated_occurrence": False}:
            raise ValueError("Golden35 row benchmark metadata differs")


validate_subset_rows = validate_golden35_rows
