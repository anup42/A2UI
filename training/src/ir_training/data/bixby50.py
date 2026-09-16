"""Frozen Bixby/Perplexity response-only holdout; never a training target set."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Mapping, Sequence


BENCHMARK_ID = "bixby50_v1"
SOURCE_RESPONSES_SHA256 = "5c8d92953589b346591c62f3dae240cefe22498ced23c5d50e5153b5ba9e9f71"
SOURCE_RESPONSES_PATH = "tmp/bixby_perplexity_check/run_50_delivery/responses.jsonl"
APPROVED_SOURCE_IDS = tuple(f"BXP-{index:03d}" for index in range(1, 51))
SELECTION_ROLE = "final_only_holdout"
ARTIFACT_SHA256 = "27bdc9ad4c7d3735d3c5f7c26fa37be2472e7aafd7c99afa31910a22b5bfdb4d"
MANIFEST_SHA256 = "576ebd54f90c514ab1508626de5f4848558d04fec99b2008d392848ee6dfbdde"
TARGET_FIELDS = ("completion", "completion_targets", "a2ui_express", "canonical_graph", "genui_json", "target", "expected", "expected_json",
                 "expected_ui_contract", "expected_ui_contract_v5_3", "expected_ui_contract_v5_4")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def serialize_rows(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return ("\n".join(_json(row) for row in rows) + "\n").encode("utf-8")


def build_bixby50(
    source_rows: Sequence[Mapping[str, Any]], *, source_responses_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Copy only approved public scenario content, never logs or generated IR."""
    if source_responses_sha256 != SOURCE_RESPONSES_SHA256:
        raise ValueError("Bixby50 source hash differs from the frozen delivery")
    if len(source_rows) != 50 or {row.get("scenario_id") for row in source_rows} != set(APPROVED_SOURCE_IDS):
        raise ValueError("Bixby50 requires exactly the 50 approved scenario IDs")
    indexed = {row["scenario_id"]: row for row in source_rows}
    rows = []
    for source_id in APPROVED_SOURCE_IDS:
        source = indexed[source_id]
        if (source.get("provider") != "Perplexity" or source.get("typed_query_exact") is not True
                or source.get("status") not in {"complete", "complete_non_markdown"}):
            raise ValueError(f"Bixby50 collection evidence is incomplete: {source_id}")
        for name in ("markdown_response", "user_query", "domain"):
            if not isinstance(source.get(name), str) or not source[name].strip():
                raise ValueError(f"Bixby50 missing {name}: {source_id}")
        rows.append({
            "id": source_id, "source_id": source_id, "response_id": source_id,
            "response_text": source["markdown_response"],
            "reference_available": False, "evaluation_only": True,
            "selection_role": SELECTION_ROLE,
            "metadata": {
                "benchmark": {"benchmark_id": BENCHMARK_ID, "kind": "source_only_holdout",
                              "row_count": 50, "unique_source_count": 50,
                              "reference_available": False, "evaluation_only": True,
                              "selection_role": SELECTION_ROLE},
                "original_query": source["user_query"],
                "domain": source["domain"], "provider": "Perplexity",
                "collection_date": "2026-09-09", "collection_status": source["status"],
                "markdown_status": source.get("markdown_status"),
                "reference_available": False, "evaluation_only": True,
                "selection_role": SELECTION_ROLE,
            },
        })
    manifest = {
        "schema_version": 1, "benchmark_id": BENCHMARK_ID,
        "benchmark_kind": "source_only_holdout", "row_count": 50,
        "unique_source_count": 50, "duplicate_occurrence_count": 0,
        "reference_available": False, "evaluation_only": True,
        "selection_role": SELECTION_ROLE,
        "source_responses_path": SOURCE_RESPONSES_PATH,
        "source_responses_sha256": SOURCE_RESPONSES_SHA256,
        "output_file": "bixby50.jsonl", "output_sha256": hashlib.sha256(serialize_rows(rows)).hexdigest(),
        "accepted_source_ids": list(APPROVED_SOURCE_IDS),
        "accepted_membership_sha256": _sha(_json(list(APPROVED_SOURCE_IDS))),
        "accepted_sources": [
            {"source_id": row["source_id"], "row_id": row["id"], "response_id": row["response_id"],
             "response_sha256": _sha(row["response_text"]),
             "query_sha256": _sha(row["metadata"]["original_query"]),
             "domain": row["metadata"]["domain"]}
            for row in rows
        ],
        "limitations": [
            "No reference IR exists; reference-match metrics are not applicable.",
            "Frozen Bixby responses are inputs, not verified factual ground truth.",
            "BXP-038 is a captured refusal and is retained unchanged.",
            "Numeric citation markers have no captured URLs; do not invent citation links.",
            "Never use for training, validation loss, tuning, or checkpoint selection.",
        ],
    }
    manifest["contract_sha256"] = _sha(_json(manifest))
    validate_source_only_rows(rows, manifest)
    return rows, manifest


def validate_source_only_rows(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate immutable source identity in either raw or prepared holdout rows.

    Prepared rows may include the shared prompt's few-shot assistant examples,
    but must end in the task user turn, never an invented reference answer.
    """
    def invalid(reason: str) -> None:
        raise ValueError(f"Bixby50 source-only benchmark invalid: {reason}")

    if not isinstance(contract, Mapping):
        invalid("manifest must be an object")
    if (contract.get("schema_version") != 1 or contract.get("benchmark_id") != BENCHMARK_ID
            or contract.get("benchmark_kind") != "source_only_holdout"):
        invalid("unsupported benchmark contract")
    if (contract.get("row_count") != 50 or contract.get("unique_source_count") != 50
            or contract.get("duplicate_occurrence_count") != 0):
        invalid("exact cohort size is required")
    if (contract.get("reference_available") is not False or contract.get("evaluation_only") is not True
            or contract.get("selection_role") != SELECTION_ROLE):
        invalid("evaluation-only flags cannot be dropped")
    if contract.get("source_responses_sha256") != SOURCE_RESPONSES_SHA256:
        invalid("source delivery hash changed")
    if (contract.get("accepted_source_ids") != list(APPROVED_SOURCE_IDS)
            or contract.get("accepted_membership_sha256") != _sha(_json(list(APPROVED_SOURCE_IDS)))):
        invalid("approved membership changed")
    expected_contract_hash = _sha(_json({key: value for key, value in contract.items() if key != "contract_sha256"}))
    if contract.get("contract_sha256") != expected_contract_hash:
        invalid("manifest checksum changed")
    accepted = contract.get("accepted_sources")
    if (not isinstance(accepted, list) or len(accepted) != 50
            or any(not isinstance(item, Mapping) for item in accepted)
            or [item.get("source_id") for item in accepted] != list(APPROVED_SOURCE_IDS)):
        invalid("frozen source evidence changed")
    evidence = {item["source_id"]: item for item in accepted}
    if len(rows) != 50 or {row.get("source_id") for row in rows} != set(APPROVED_SOURCE_IDS):
        invalid("exactly 50 unique approved sources are required")
    response_hashes = set()
    for row in rows:
        source_id = row["source_id"]
        item = evidence[source_id]
        if any(row.get(key) != source_id for key in ("id", "response_id")):
            invalid(f"identity changed: {source_id}")
        if item.get("row_id") != source_id or item.get("response_id") != source_id:
            invalid(f"source evidence identity changed: {source_id}")
        if (row.get("reference_available") is not False or row.get("evaluation_only") is not True
                or row.get("selection_role") != SELECTION_ROLE):
            invalid(f"evaluation-only row flags changed: {source_id}")
        metadata = row.get("metadata")
        if (not isinstance(metadata, Mapping)
                or metadata.get("reference_available") is not False or metadata.get("evaluation_only") is not True
                or metadata.get("selection_role") != SELECTION_ROLE):
            invalid(f"evaluation-only metadata changed: {source_id}")
        benchmark = metadata.get("benchmark")
        if (not isinstance(benchmark, Mapping) or benchmark.get("benchmark_id") != BENCHMARK_ID
                or benchmark.get("kind") != "source_only_holdout" or benchmark.get("row_count") != 50
                or benchmark.get("unique_source_count") != 50 or benchmark.get("reference_available") is not False
                or benchmark.get("evaluation_only") is not True or benchmark.get("selection_role") != SELECTION_ROLE):
            invalid(f"source-only benchmark metadata changed: {source_id}")
        response = row.get("raw_response_text", row.get("response_text"))
        if not isinstance(response, str) or not response.strip() or _sha(response) != item.get("response_sha256"):
            invalid(f"frozen source response changed: {source_id}")
        # This frozen delivery has no URLs to mask. Changing response_text would
        # therefore change what is evaluated even if raw_response_text survives.
        if row.get("response_text") != response:
            invalid(f"prepared source response changed: {source_id}")
        response_hashes.add(_sha(response))
        query = metadata.get("original_query")
        if (not isinstance(query, str) or _sha(query) != item.get("query_sha256")
                or metadata.get("domain") != item.get("domain")):
            invalid(f"source provenance changed: {source_id}")
        if any(container.get(key) not in (None, "", {}, []) for container in (row, metadata) for key in TARGET_FIELDS):
            invalid(f"fabricated reference target: {source_id}")
        if row.get("target_validation") not in (None, "not_applicable_source_only"):
            invalid(f"target validation misrepresented: {source_id}")
        if "messages" in row:
            messages = row["messages"]
            if (not isinstance(messages, list) or not messages or not isinstance(messages[-1], Mapping)
                    or messages[-1].get("role") != "user"):
                invalid(f"prepared messages must end in a user turn without reference target: {source_id}")
    if len(response_hashes) != 50:
        invalid("all 50 source responses must be unique")
    return deepcopy(dict(contract))


def validate_source_only_predictions(rows: Sequence[Mapping[str, Any]]) -> None:
    """Bind standalone scoring to the bundled sources, not arbitrary claimed IDs."""
    from ir_training.common.config import repo_root

    directory = repo_root() / "training/data/eval/bixby50_v1"
    raw = (directory / "bixby50.jsonl").read_bytes()
    manifest_bytes = (directory / "benchmark_manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != ARTIFACT_SHA256 or hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Frozen Bixby50 scoring artifacts changed")
    references = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    validate_source_only_rows(references, json.loads(manifest_bytes))
    sources = {row["source_id"]: row for row in references}
    if len(rows) != 50 or {row.get("source_id") for row in rows} != set(APPROVED_SOURCE_IDS):
        raise ValueError("Bixby50 scoring requires exactly the 50 approved source IDs")
    for row in rows:
        source = sources[row["source_id"]]
        if (any(row.get(key) != source[key] for key in ("id", "source_id", "response_id", "response_text"))
                or row.get("benchmark") != source["metadata"]["benchmark"]
                or row.get("reference_available") is not False or row.get("evaluation_only") is not True
                or row.get("selection_role") != SELECTION_ROLE):
            raise ValueError(f"Bixby50 prediction source or holdout flags changed: {row.get('id')}")
        metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
        if any(container.get(key) not in (None, "", {}, []) for container in (row, metadata) for key in TARGET_FIELDS):
            raise ValueError("Bixby50 source-only scoring cannot use fabricated reference targets or UI contracts")
