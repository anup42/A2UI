"""Strict, opt-in admission for regenerated v11 supervision; never repairs data.

Existing Stage 3/SFT/QAT admission defaults do not import this module. Passing
this additional gate is not a source factuality or device-rendering certificate.
Known source defects are content-hash-bound holds, not instructions to fix prose.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from pipeline.genui_quality import generation_reward_a2ui_express_v1
from pipeline.genui_quality.config_v5_4 import coerce_reward_config_v5_4
from pipeline.genui_quality.evidence_v5_4 import collect_output_evidence_v5_4
from pipeline.genui_quality.graph import audit_renderer_graph
from pipeline.training_fidelity import (
    _tokens,
    audit_training_fidelity,
    bounded_fidelity_config,
)

POLICY_VERSION = "v11-regeneration-guard-20261023-v1"
_ROOT = Path(__file__).resolve().parents[3]
SOURCE_HOLDS_PATH = _ROOT / "dataset/config/v11_source_quality_holds_20261023.json"
_TASK_PREFIX = "Create A2UI Express v1 GenUI IR for this response:\n\n"
_SHA = re.compile(r"[0-9a-f]{64}")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_text(row: Mapping[str, Any]) -> str:
    """Return the exact final-task source; conflicting aliases fail closed."""
    if not isinstance(row, Mapping):
        raise ValueError("invalid_source_row")  # noqa: TRY004 - invalid external record
    candidates = []
    for key in ("source", "response_text"):
        if key in row:
            if not isinstance(row[key], str):
                raise ValueError("invalid_source_text:" + key)
            candidates.append(row[key])
    messages = row.get("messages")
    if isinstance(messages, list):
        users = [item for item in messages if isinstance(item, Mapping) and item.get("role") == "user"]
        if users:
            content = users[-1].get("content")
            if not isinstance(content, str) or not content.startswith(_TASK_PREFIX):
                raise ValueError("source_task_prefix_mismatch")
            candidates.append(content[len(_TASK_PREFIX):])
    if not candidates or not candidates[0].strip():
        raise ValueError("missing_source_text")
    if any(value != candidates[0] for value in candidates[1:]):
        raise ValueError("source_alias_mismatch")
    return candidates[0]


def _catalog() -> list[dict[str, Any]]:
    value = json.loads(SOURCE_HOLDS_PATH.read_text(encoding="utf-8"))
    if value.get("policy_version") != POLICY_VERSION or not isinstance(value.get("holds"), list):
        raise ValueError("invalid_source_hold_catalog")
    seen = set()
    for entry in value["holds"]:
        if (not isinstance(entry, dict) or not isinstance(entry.get("row_id"), str)
                or not entry["row_id"] or not _SHA.fullmatch(str(entry.get("source_sha256", "")))
                or not isinstance(entry.get("reason"), str) or not entry["reason"].strip()
                or not isinstance(entry.get("source_id"), str) or not entry["source_id"]):
            raise ValueError("invalid_source_hold_entry")
        key = (entry["row_id"], entry["source_sha256"])
        if key in seen:
            raise ValueError("duplicate_source_hold_entry")
        seen.add(key)
    return value["holds"]


def guard_identity() -> dict[str, str]:
    """Include both policy bytes in workflow resume/provenance contracts."""
    _catalog()  # A malformed or missing policy is a setup failure, never bypassed.
    return {"policy_version": POLICY_VERSION,
            "guard_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_hold_catalog_sha256": hashlib.sha256(SOURCE_HOLDS_PATH.read_bytes()).hexdigest()}


def assess_source(row: Mapping[str, Any]) -> dict[str, Any]:
    """Quarantine known defects and mismatched source identities without editing."""
    catalog = _catalog()
    try:
        source = source_text(row)
    except ValueError as exc:
        return {"policy_version": POLICY_VERSION, "eligible": False,
                "status": "quarantined", "reasons": [str(exc)]}
    digest = _sha(source)
    reasons = []
    for key in ("source_sha256", "reference_source_sha256"):
        if key in row and row[key] != digest:
            reasons.append("source_hash_mismatch:" + key)
    metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
    recovery = metadata.get("archive_recovery")
    if (isinstance(recovery, Mapping) and "effective_source_sha256" in recovery
            and recovery["effective_source_sha256"] != digest):
        reasons.append("source_hash_mismatch:archive_recovery.effective_source_sha256")
    identities = {str(value) for value in (row.get("id"), row.get("source_id"),
                  metadata.get("source_id"), metadata.get("response_id"), metadata.get("ui_id")) if value}
    matches = [entry for entry in catalog if entry["source_sha256"] == digest]
    if matches:
        reasons.append("audited_source_quality_hold")
    # A changed source under an audited identity needs an explicit new review,
    # not an accidental release caused by the old content hash no longer matching.
    if any(entry["row_id"] in identities or entry["source_id"] in identities
           for entry in catalog if entry["source_sha256"] != digest):
        reasons.append("audited_source_identity_hash_mismatch")
    return {"policy_version": POLICY_VERSION, "eligible": not reasons,
            "status": "quarantined" if reasons else "not_held",
            "source_sha256": digest, "reasons": list(dict.fromkeys(reasons)),
            "known_holds": [dict(entry) for entry in matches],
            "source_factuality_verified": False,
            "limitations": "Only reviewed content hashes are source-quality holds; unknown sources are not factuality-certified."}


def _training_imports():
    # The opt-in dataset command also consumes the proven training import gate.
    training_src = str(_ROOT / "training/src")
    if training_src not in sys.path:
        sys.path.insert(0, training_src)
    from ir_training.data.express_preparation import PreparationError, serialize_checked
    from ir_training.data.reference_binding import bind_source_target_references
    from ir_training.data.release_review import reviewed_hold
    return PreparationError, serialize_checked, bind_source_target_references, reviewed_hold


def _reference_context(row: Mapping[str, Any]) -> dict[str, Any]:
    """Carry source-side maps, never an old target's restoration certificate."""
    result = {key: row[key] for key in ("reference_map", "asset_placeholder_map", "url_map",
              "reference_source_sha256") if key in row}
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping) and isinstance(metadata.get("url_preprocessing"), Mapping):
        result["metadata"] = {"url_preprocessing": dict(metadata["url_preprocessing"])}
    return result


def _unsupported_prose(source: str, graph: Mapping[str, Any]) -> dict[str, Any]:
    """Reject added visible prose using ordered source spans, never token bags.

This supplements, rather than replaces, canonical role/table/action validation
and the independent entity ownership certificate. Hidden/unused state is not
support. Multi-line source prose may be joined, but its token order cannot be
changed to manufacture a claim that did not occur in the source.
"""
    config = bounded_fidelity_config(source, graph)
    evidence = collect_output_evidence_v5_4(graph, audit_renderer_graph(graph), config)
    source_sequence = " " + _tokens(source) + " "
    unsupported = []
    for item in evidence.evidence_ownership:
        if item.get("owner") != "generic_visible_content":
            continue
        raw = str(item.get("text") or "")
        # Each literal line/bullet is an independently visible claim. Splitting
        # on a semicolon is intentionally avoided: it can hide qualifier loss.
        for line in raw.splitlines():
            normalized = _tokens(line)
            if normalized and " " + normalized + " " not in source_sequence:
                unsupported.append({"component_id": item.get("component_id"), "text": line})
    return {"complete": evidence.dynamic_evidence_complete,
            "unsupported_count": len(unsupported), "unsupported_examples": unsupported[:16]}


def _fidelity_reasons(fidelity: Mapping[str, Any], ownership: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Resolve only literal-audit ordering false positives with exact ownership.

The legacy diagnostic appends Card.title after its children. It can therefore
miss a whole literal source record whose title and fields were correctly split.
Only a complete exact structured certificate for the *same whole source unit*
can resolve that diagnostic. No token union, subset, fuzzy match or canonical
training-acceptance override is permitted, including when examples truncate.
"""
    reasons = list(fidelity["reasons"])
    if ("source_literal_not_exactly_represented" not in reasons
            or fidelity.get("missing_unit_examples_truncated")
            or ownership.get("binding_complete") is not True
            or ownership.get("issues")
            or ownership.get("structured_record_count") != ownership.get("eligible_record_count")):
        return reasons, []
    certified_units = set()
    for record in ownership.get("certified_records", ()):
        if record.get("structured") is not True or record.get("actions_exact") is not True:
            continue
        values = [record.get("source_text", ""), *record.get("original_source_units", ())]
        certified_units.update(_tokens(value) for value in values if isinstance(value, str) and value.strip())
    missing = fidelity.get("missing_units", ())
    covered = [str(unit["unit_id"]) for unit in missing if _tokens(unit["text"]) in certified_units]
    if len(covered) == fidelity.get("missing_unit_count") == len(missing):
        reasons.remove("source_literal_not_exactly_represented")
    return reasons, covered


def assess_candidate(row: Mapping[str, Any], target: str | Mapping[str, Any]) -> dict[str, Any]:
    """Require *all* existing gates plus strict regeneration-only safeguards."""
    source_report = assess_source(row)
    report: dict[str, Any] = {"policy_version": POLICY_VERSION, "eligible": False,
                              "status": "rejected", "reasons": list(source_report["reasons"]),
                              "source": source_report}
    if not source_report["eligible"]:
        return report
    source = source_text(row)
    PreparationError, serialize_checked, bind_references, reviewed_hold = _training_imports()
    from pipeline.genui_quality.grpo_entity_content import (
        summarize_grpo_entity_detail,
    )
    from pipeline.ir_formats import express

    try:
        if isinstance(target, Mapping):
            completion = express.encode(dict(target), shorten_ids=False)
        elif isinstance(target, str):
            completion = target
        else:
            raise ValueError("candidate_requires_express_or_graph")  # noqa: TRY004 - invalid external candidate
        checked = serialize_checked(completion)
        hold = reviewed_hold(row, checked.graph, checked.semantic_sha256)
        if hold is not None:
            report["reasons"].append(hold[0])
            report["reviewed_hold"] = {"reason": hold[0], "detail": hold[1]}
        bound = bind_references(source, checked.graph, _reference_context(row))
    except (PreparationError, ValueError) as exc:
        report["reasons"].append(getattr(exc, "reason", "candidate_validation_failed"))
        report["validation_error"] = str(exc)
        return report

    # Recompute authoritative acceptance; never trust a serialized eligible flag.
    # Explicitly select the source-bound entity policy only in this opt-in
    # workflow. This is the same canonical scorer and unchanged thresholds,
    # not a post-hoc override of a failed default acceptance result.
    config = bounded_fidelity_config(bound.source, bound.graph, replace(
        coerce_reward_config_v5_4(None), grpo_entity_extraction=True))
    scoring_text = express.encode(bound.graph, shorten_ids=False)
    reward = generation_reward_a2ui_express_v1(scoring_text, bound.source, config=config)
    admission = reward.evidence.get("training_acceptance")
    report["canonical_acceptance"] = admission
    report["canonical_policy"] = {"scorer": "generation_reward_a2ui_express_v1",
                                  "grpo_entity_extraction": True,
                                  "thresholds": "unchanged"}
    if (not isinstance(admission, Mapping) or admission.get("eligible") is not True
            or admission.get("blocking_reasons") != [] or admission.get("review_reasons") != []):
        report["reasons"].append("canonical_training_ineligible")
    # Reuse only this freshly scored certificate, never row-supplied evidence.
    ownership = summarize_grpo_entity_detail(reward.evidence.get("grpo_entity_extraction"))
    report["entity_ownership"] = ownership
    fidelity = audit_training_fidelity(bound.source, bound.graph)
    report["fidelity"] = fidelity
    fidelity_reasons, covered = _fidelity_reasons(fidelity, ownership)
    report["literal_missing_units_certified_by_entity_ownership"] = covered
    report["reasons"].extend("fidelity:" + reason for reason in fidelity_reasons)
    prose = _unsupported_prose(bound.source, bound.graph)
    report["visible_prose"] = prose
    if not prose["complete"]:
        report["reasons"].append("renderer_evidence_incomplete")
    if prose["unsupported_count"]:
        report["reasons"].append("unsupported_visible_prose")
    if ownership["applicable"]:
        if ownership.get("binding_complete") is not True:
            report["reasons"].append("entity_binding_incomplete")
        if ownership.get("structured_record_count", 0) != ownership["eligible_record_count"]:
            report["reasons"].append("entity_fields_not_structured")
    report["reference_binding"] = bound.evidence
    report["candidate_sha256"] = _sha(completion)
    report["semantic_sha256"] = checked.semantic_sha256
    report["reasons"] = list(dict.fromkeys(report["reasons"]))
    report["eligible"] = not report["reasons"]
    report["status"] = "accepted" if report["eligible"] else "rejected"
    return report
