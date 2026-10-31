"""Audit returned reasoning summaries separately from Stage 3 training targets.

These are provider summaries, not the model's private reasoning trace. Raw Batch
responses remain the authoritative evidence, including unknown output fields.
"""
from __future__ import annotations

from pathlib import Path

from pipeline.openai_stage3_batch_runner import read_rows, write_rows
from pipeline.v11_regeneration_io import digest


def _summary_parts(body: dict) -> tuple[list[dict], list[str], bool]:
    parts, warnings = [], []
    has_reasoning = False
    output = body.get("output", [])
    if not isinstance(output, list):
        return [], ["malformed_output"], False
    for output_index, item in enumerate(output):
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            continue
        has_reasoning = True
        summary = item.get("summary", [])
        if not isinstance(summary, list):
            warnings.append(f"malformed_summary:{output_index}")
            continue
        for summary_index, part in enumerate(summary):
            if (not isinstance(part, dict) or part.get("type") != "summary_text"
                    or not isinstance(part.get("text"), str)):
                warnings.append(f"unsupported_summary_part:{output_index}:{summary_index}")
                continue
            # Preserve exact returned text, ordering and item IDs. Never extract
            # encrypted_content, assistant output_text, or a guessed rationale.
            parts.append({"item_id": item.get("id"), "output_index": output_index,
                          "summary_index": summary_index, "text": part["text"]})
    return parts, warnings, has_reasoning


def write_reasoning_summaries(output: Path, contract: dict, indexed: dict) -> dict:
    bindings = contract["bindings"]
    selected = read_rows(output / "selected.jsonl")
    sources = {row["response_id"]: row["source_id"] for row in selected}
    assessments = read_rows(output / "assessments.jsonl")
    by_id = {row["custom_id"]: row for row in assessments}
    if (len(sources) != len(selected) or len(sources) != len(bindings)
            or set(sources) != set(bindings.values())
            or len(by_id) != len(assessments) or set(by_id) != set(bindings)
            or not set(indexed) <= set(bindings)):
        raise ValueError("Reasoning summary source/assessment bindings changed")
    rows = []
    contract_hash = digest(contract)
    for custom_id, prepared_response_id in bindings.items():
        record = indexed.get(custom_id)
        response = (record or {}).get("response") or {}
        body = response.get("body") or {}
        parts, warnings, has_reasoning = _summary_parts(body)
        has_text = any(part["text"].strip() for part in parts)
        summary_status = ("missing_result" if record is None else "available" if has_text
                          else "empty" if has_reasoning else "not_returned")
        assessment = by_id[custom_id]
        rows.append({
            "schema_version": 1, "text_kind": "reasoning_summary",
            "custom_id": custom_id, "source_id": sources[prepared_response_id],
            "prepared_response_id": prepared_response_id, "api_response_id": body.get("id"),
            "response_status": body.get("status"), "http_status": response.get("status_code"),
            "api_error": (record or {}).get("error") or body.get("error"),
            "usage": body.get("usage"), "summary_status": summary_status,
            "summary_text": "\n\n".join(part["text"] for part in parts),
            "summary_parts": parts, "summary_warnings": warnings,
            "eligible": assessment["eligible"], "reasons": assessment.get("reasons", []),
            "contract_sha256": contract_hash,
        })
    name = "reasoning_summaries.jsonl"
    write_rows(output / name, rows)
    available = sum(row["summary_status"] == "available" for row in rows)
    return {"file": name, "requested": len(rows), "available": available,
            "unavailable": len(rows) - available, "text_kind": "reasoning_summary",
            "raw_reasoning_available": False}
