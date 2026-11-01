"""Read-only reassessment of a sealed Batch archive under the opt-in v2 policy.

This deliberately does not call pilot.load: historical code fingerprints may
differ. Archive seals and source/candidate bindings remain mandatory.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from statistics import mean

from pipeline import openai_stage3_batch as pilot
from pipeline import openai_stage3_batch_runner as runner
from pipeline import v11_regeneration_guard as guard
from pipeline.training_augmentation import admission_errors
from pipeline.v11_regeneration_io import (
    atomic_bytes,
    canonical,
    checked,
    digest,
    file_hash,
    safe_path,
    save,
)

VERSION = "openai-stage3-offline-rescore-v1"


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(text):
    value = json.loads(text, object_pairs_hook=_pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON")))
    if not isinstance(value, dict):
        raise ValueError("Archive records must be JSON objects")  # noqa: TRY004 - invalid external JSON
    return value


def _rows(path):
    return [_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _contained(root, name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("Unsafe archive path")
    parts = name.split("/")
    if any(part in {"", ".", ".."} for part in parts) or PurePosixPath(name).is_absolute():
        raise ValueError("Unsafe archive path")
    path = safe_path(root / name)
    if root not in path.parents:
        raise ValueError("Archive path escapes its directory")
    return path


def _hashes(root, manifest):
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("Missing archive file manifest")
    for name, expected in manifest.items():
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError("Malformed archive file hash")
        if file_hash(_contained(root, name)) != expected:
            raise ValueError(f"Archive file changed: {name}")


def _snapshot(root):
    result = {}
    for path in sorted(root.rglob("*")):
        safe_path(path)
        if path.is_file():
            result[path.relative_to(root).as_posix()] = file_hash(path)
    return result


def _indexed(rows, key, expected=None):
    result = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value.strip() or value in result:
            raise ValueError(f"Malformed or duplicate {key}")
        if expected is not None and value not in expected:
            raise ValueError(f"Unknown {key}")
        result[value] = row
    return result


def _load_archive(root):
    envelopes = {}
    for name in ("contract", "submission", "collection"):
        path = _contained(root, name + ".json")
        _json(path.read_text(encoding="utf-8"))  # Reject duplicate keys before checked().
        envelopes[name] = checked(path)
        if not isinstance(envelopes[name], dict):
            raise ValueError("Malformed archive seal payload")  # noqa: TRY004 - invalid external JSON
    contract, submission, collection = (envelopes[k] for k in ("contract", "submission", "collection"))
    if contract.get("version") != pilot.VERSION:
        raise ValueError("Unknown archived pilot version")
    config = contract.get("config")
    if (not isinstance(config, dict) or type(config.get("samples")) is not int or config["samples"] < 1
            or not isinstance(config.get("model"), str) or not config["model"]
            or not isinstance(contract.get("code_hashes"), dict) or not contract["code_hashes"]
            or not isinstance(contract.get("guard"), dict)):
        raise ValueError("Malformed archived pilot configuration or fingerprint")
    pilot.fidelity_v2.profile(config)
    identity = digest(contract)
    if submission.get("contract_sha256") != identity or collection.get("contract_sha256") != identity:
        raise ValueError("Archive contract binding mismatch")
    batch_id = submission.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id or collection.get("batch_id") != batch_id:
        raise ValueError("Archive batch binding mismatch")
    for item in (contract, collection):
        _hashes(root, item.get("files"))
    if not {"selected.jsonl", "requests.jsonl", "responses.jsonl", "queries.jsonl"} <= contract["files"].keys():
        raise ValueError("Unsealed selection or request files")
    if not {"report.json", "assessments.jsonl"} <= collection["files"].keys():
        raise ValueError("Unsealed collection report or assessments")
    report = _json((root / "report.json").read_text(encoding="utf-8"))
    if report.get("contract_sha256") != identity:
        raise ValueError("Report contract binding mismatch")
    stage_name = report.get("stage3_artifacts")
    stage = _contained(root, stage_name)
    if not stage.is_dir() or not any(name.startswith(stage_name + "/") for name in collection["files"]):
        raise ValueError("Unsealed Stage 3 artifact directory")
    candidate_name = stage_name + "/genui.jsonl"
    candidate_path = _contained(root, candidate_name)
    if candidate_path.exists() and candidate_name not in collection["files"]:
        raise ValueError("Unsealed canonical Stage 3 candidates")
    bindings = contract.get("bindings")
    if not isinstance(bindings, dict) or not bindings or any(
            not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in bindings.items()):
        raise ValueError("Malformed request bindings")
    if len(set(bindings.values())) != len(bindings):
        raise ValueError("Duplicate response bindings")
    selected = _rows(root / "selected.jsonl")
    sources = _indexed(selected, "source_id")
    responses = _indexed(selected, "response_id", set(bindings.values()))
    if set(responses) != set(bindings.values()) or contract.get("config", {}).get("samples") != len(selected):
        raise ValueError("Selection does not match bound sample count")
    for entry in selected:
        row = entry.get("row")
        if (not isinstance(row, dict) or not isinstance(row.get("metadata", {}), dict)
                or pilot.existing._row_id(row) != entry["source_id"]
                or entry.get("original_split") != "train"
                or (row.get("metadata") or {}).get("assigned_split", "train") != "train"
                or type(entry.get("line")) is not int or entry["line"] < 1
                or not isinstance(entry.get("original_row_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", entry["original_row_sha256"]) is None
                or not isinstance(entry.get("selection_bucket", "random"), str)
                or entry["response_id"] != "p-" + digest(row)[:24]):
            raise ValueError("Selected source identity mismatch")
    requests = _indexed(_rows(root / "requests.jsonl"), "custom_id", set(bindings))
    if set(requests) != set(bindings):
        raise ValueError("Request identities do not match bindings")
    for ordinal, (custom_id, request) in enumerate(requests.items()):
        body = request.get("body")
        if (request.get("method") != "POST" or request.get("url") != "/v1/responses"
                or not isinstance(body, dict) or body.get("model") != config["model"]
                or custom_id != runner.request_id(body) + f"-{ordinal:04d}"):
            raise ValueError("Archived request transport or identity mismatch")
    prepared = _indexed(_rows(root / "responses.jsonl"), "response_id", set(responses))
    queries = _indexed(_rows(root / "queries.jsonl"), "query_id", set(responses))
    if set(prepared) != set(responses) or set(queries) != set(responses):
        raise ValueError("Prepared source selection mismatch")
    for response_id, entry in responses.items():
        if (prepared[response_id].get("query_id") != response_id or
                prepared[response_id].get("response_text") != pilot.existing._generation_source(entry["row"])):
            raise ValueError("Prepared source binding mismatch")
    official = _indexed(_rows(root / "assessments.jsonl"), "source_id", set(sources))
    reverse = {v: k for k, v in bindings.items()}
    if set(official) != set(sources):
        raise ValueError("Archived assessment coverage mismatch")
    for entry in selected:
        assessment = official[entry["source_id"]]
        if assessment.get("custom_id") != reverse[entry["response_id"]] or type(assessment.get("eligible")) is not bool:
            raise ValueError("Archived assessment binding mismatch")
        if not isinstance(assessment.get("reasons", []), list) or not all(
                isinstance(reason, str) for reason in assessment.get("reasons", [])):
            raise ValueError("Malformed archived assessment reasons")
    records = []
    for name in ("output_file_id.jsonl", "error_file_id.jsonl"):
        if (root / name).exists():
            if name not in collection["files"]:
                raise ValueError("Unsealed Batch result file")
            records.extend(_rows(root / name))
    indexed = pilot.index_results(records, set(bindings))
    candidates = _indexed(_rows(candidate_path) if candidate_path.exists() else [], "response_id", set(responses))
    for response_id, candidate in candidates.items():
        if candidate.get("query_id") != response_id:
            raise ValueError("Canonical candidate query binding mismatch")
    return contract, collection, selected, official, indexed, candidates


def _reject(reason):
    return {"eligible": False, "status": "rejected", "reasons": [reason]}


def _summary(rows):
    result = {"requests": len(rows), "canonical_candidates": sum(r["canonical_candidate"] for r in rows),
              "canonical_valid_candidates": sum(r["canonical_valid"] for r in rows)}
    for field in ("official", "legacy", "v2", "baseline_v2"):
        result[field + "_accepted"] = sum(r[field].get("eligible") is True for r in rows)
        result[field + "_reasons"] = dict(Counter(reason for r in rows for reason in r[field].get("reasons", [])))
    result["official_to_v2"] = dict(Counter(
        ("accepted" if r["official"]["eligible"] else "rejected") + "->" +
        ("accepted" if r["v2"]["eligible"] else "rejected") for r in rows))
    result["legacy_to_v2"] = dict(Counter(
        ("accepted" if r["legacy"]["eligible"] else "rejected") + "->" +
        ("accepted" if r["v2"]["eligible"] else "rejected") for r in rows))
    for field in ("v2", "baseline_v2"):
        scores = [r[field]["score_v5_4"] for r in rows if isinstance(r[field].get("score_v5_4"), (int, float))]
        result[field + "_scores"] = {"count": len(scores), "mean": mean(scores) if scores else None,
                                       "min": min(scores) if scores else None, "max": max(scores) if scores else None}
    paired = [r["v2"]["score_v5_4"] - r["baseline_v2"]["score_v5_4"] for r in rows
              if "score_v5_4" in r["v2"] and "score_v5_4" in r["baseline_v2"]]
    result["paired_score_delta"] = {"count": len(paired), "mean": mean(paired) if paired else None,
                                     "improved": sum(x > 1e-8 for x in paired),
                                     "degraded": sum(x < -1e-8 for x in paired)}
    return result


def rescore(pilot_dir: Path, output_dir: Path) -> dict:
    """Write assessment reports to a fresh sibling; never regenerate or admit data."""
    from ir_training.data.express_preparation import serialize_checked
    from ir_training.data.reference_binding import bind_source_target_references

    from pipeline.ir_formats import express

    root, output = safe_path(pilot_dir), safe_path(output_dir)
    if not root.is_dir() or output.exists() or output.parent != root.parent or output == root:
        raise ValueError("Rescore requires a fresh sibling output directory")
    before = _snapshot(root)
    contract, collection, selected, official, indexed, candidates = _load_archive(root)
    current = pilot.fingerprint()
    reverse = {v: k for k, v in contract["bindings"].items()}
    rows = []
    for entry in selected:
        source = entry["row"]
        candidate = candidates.get(entry["response_id"])
        custom_id = reverse[entry["response_id"]]
        item = {"source_id": entry["source_id"], "response_id": entry["response_id"], "custom_id": custom_id,
                "category": entry.get("selection_bucket", "random"), "canonical_candidate": candidate is not None,
                "canonical_valid": False, "diagnostic_only": True,
                "official": official[entry["source_id"]],
                "baseline_v2": guard.assess_candidate(source, source.get("completion"), pilot_fidelity_v2=True)}
        try:
            paid_text = runner.response_text(indexed.get(custom_id, {}), contract["config"])
            if candidate is None:
                raise ValueError("canonical_stage3_rejected")
            errors = admission_errors(candidate, pilot.existing._generation_source(source))
            item["archived_admission_errors"] = errors
            item["archived_training_acceptance"] = candidate.get("training_acceptance")
            item["archived_generation_reward_v5_4"] = candidate.get("generation_reward_v5_4")
            # Historical quality certificates are evidence, not authority for
            # a current-policy comparison. Structural/provenance failures stay
            # excluded; both guards recompute the quality and fidelity gates.
            structural = [reason for reason in errors if reason in {
                "stage3_not_accepted", "schema_valid_strict", "standard_a2ui_valid",
                "generation_failed_or_incomplete", "source_binding_mismatch", "missing_express_completion"}]
            if structural:
                raise ValueError(";".join(structural))
            if paid_text != candidate.get("genui_raw_completion"):
                raise ValueError("canonical_candidate_paid_output_mismatch")
            raw_parsed = serialize_checked(paid_text, "root-first")
            parsed = serialize_checked(candidate["a2ui_express"], "root-first")
            raw_bound = bind_source_target_references(candidate["response_text"], raw_parsed.graph, candidate)
            bound = bind_source_target_references(candidate["response_text"], parsed.graph, candidate)
            raw_semantics = serialize_checked(express.encode(raw_bound.graph, shorten_ids=False), "root-first")
            candidate_semantics = serialize_checked(express.encode(bound.graph, shorten_ids=False), "root-first")
            if raw_semantics.semantic_sha256 != candidate_semantics.semantic_sha256:
                raise ValueError("canonical_candidate_paid_semantics_mismatch")
            item["paid_output_sha256"] = digest(paid_text)
            target = pilot.existing._candidate_target(source, {"a2ui_express": express.encode(bound.graph, shorten_ids=False)})
            item["canonical_valid"] = True
            item["target_sha256"] = digest(target)
            item["legacy"] = guard.assess_candidate(source, target)
            item["v2"] = guard.assess_candidate(source, target, pilot_fidelity_v2=True)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            item["legacy"] = _reject(str(exc))
            item["v2"] = _reject(str(exc))
        rows.append(item)
    after = _snapshot(root)
    if after != before:
        raise ValueError("Archive changed during offline rescore")
    if pilot.fingerprint() != current:
        raise ValueError("Current scoring code changed during rescore")
    categories = defaultdict(list)
    for row in rows:
        categories[row["category"]].append(row)
    summary = {"version": VERSION, "status": "offline_rescored_no_training_output", "training_input_ready": False,
               "diagnostic_only": True,
               "scope": "Assessment only; sealed archive unchanged; no API calls, replay or regeneration.",
               "batch_id": collection["batch_id"], "contract_sha256": digest(contract),
               "overall": _summary(rows), "categories": {k: _summary(v) for k, v in sorted(categories.items())}}
    output.mkdir(exist_ok=False)
    runner.write_rows(output / "assessments.jsonl", rows)
    atomic_bytes(output / "summary.json", canonical(summary) + b"\n")
    save(output / "rescore.json", {"version": VERSION, "pilot_dir": str(root),
         "batch_id": collection["batch_id"], "contract_sha256": digest(contract),
         "archived_code_hashes": contract.get("code_hashes"), "current_code_hashes": current,
         "archived_guard": contract.get("guard"), "current_guard": guard.guard_identity(),
         "archive_hashes_before": before, "archive_hashes_after": after,
         "files": {name: file_hash(output / name) for name in ("assessments.jsonl", "summary.json")}})
    return summary
