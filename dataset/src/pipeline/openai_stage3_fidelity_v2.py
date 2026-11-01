"""Explicit OpenAI pilot policy only; never imported by default training runners."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROFILE = "fidelity_v2"
PROFILES = ("legacy", PROFILE)
POLICY_VERSION = "openai-stage3-fidelity-v2-20261101"
HOLDS_PATH = ROOT / "dataset/configs/openai_stage3_fidelity_v2_source_holds.json"
SUPPLEMENT_PATH = ROOT / "dataset/prompts/openai_stage3_fidelity_v2.md"


def profile(config: dict) -> str:
    value = config.get("pilot_profile", "legacy")
    if value not in PROFILES:
        raise ValueError("Unknown OpenAI pilot profile")
    return value


def source_payload(source: str, asset_mapping: str = "") -> str:
    """Serialize supplied fields directly: never split or search source contents."""
    return json.dumps({"source_response_text": source, "asset_mapping": asset_mapping},
                      ensure_ascii=False, sort_keys=True)


def reference_preflight(row: dict) -> dict:
    """Require real donors for reserved reference tokens before paid selection.

    This checks identity, not remote availability or local file existence. URLs
    and paths are kept byte-for-byte; no fetching, normalizing or asset repair.
    """
    from ir_training.data.reference_binding import _reference_map, reference_tokens
    from ir_training.data.url_preprocess import _REFERENCE_RE, restore_url_placeholders

    from pipeline.v11_regeneration_guard import source_text

    reasons = []
    try:
        source = source_text(row)
        mapping = _reference_map(row)
        tokens = reference_tokens(source)
        missing = sorted(tokens - mapping.keys())
        if missing:
            reasons.append("pilot_unresolved_source_references:" + ",".join(missing))
        for token in sorted(tokens & mapping.keys()):
            donor = mapping[token]
            # The canonical reference grammar handles Windows drive paths and
            # quoted paths with spaces as well as URL/action destinations.
            if not (_REFERENCE_RE.fullmatch(donor) or _REFERENCE_RE.fullmatch('"' + donor + '"')):
                reasons.append("pilot_invalid_source_reference_donor:" + token)
        expected = row.get("reference_source_sha256")
        if expected and expected != hashlib.sha256(source.encode("utf-8")).hexdigest():
            reasons.append("reference_source_hash_mismatch")
        restored = restore_url_placeholders(source, mapping)
        unresolved = sorted(reference_tokens(restored))
        if unresolved and not missing:
            reasons.append("pilot_unresolved_source_references:" + ",".join(unresolved))
        return {"eligible": not reasons, "reasons": reasons,
                "source_reference_count": len(tokens),
                "restored_source_sha256": hashlib.sha256(restored.encode("utf-8")).hexdigest()}
    except (ValueError, TypeError, KeyError) as exc:
        return {"eligible": False, "reasons": ["pilot_source_reference_preflight:" + str(exc)]}


def _holds() -> list[dict]:
    catalog = json.loads(HOLDS_PATH.read_text(encoding="utf-8"))
    if catalog.get("policy_version") != POLICY_VERSION or not isinstance(catalog.get("holds"), list):
        raise ValueError("invalid_pilot_source_hold_catalog")
    seen = set()
    for entry in catalog["holds"]:
        if (not isinstance(entry, dict) or not isinstance(entry.get("source_id"), str)
                or not entry["source_id"] or not isinstance(entry.get("reason"), str)
                or not entry["reason"] or not isinstance(entry.get("source_sha256"), str)
                or not re.fullmatch("[0-9a-f]{64}", entry["source_sha256"])):
            raise ValueError("invalid_pilot_source_hold_entry")
        if entry["source_sha256"] in seen:
            raise ValueError("duplicate_pilot_source_hold_entry")
        seen.add(entry["source_sha256"])
    return catalog["holds"]


def assess_source(row: dict, *, catalog: list[dict] | None = None) -> dict:
    """Additional pilot holds and preflight, independent of shared hold policy."""
    from pipeline.v11_regeneration_guard import source_text

    # Selection may pass the catalog already validated for this preparation.
    # Fingerprint checks reject any on-disk policy change before publication.
    holds = _holds() if catalog is None else catalog
    report = reference_preflight(row)
    reasons = list(report["reasons"])
    try:
        source_hash = hashlib.sha256(source_text(row).encode("utf-8")).hexdigest()
    except ValueError:
        source_hash = None
    source_hashes = {source_hash, report.get("restored_source_sha256")}
    matches = [entry for entry in holds if entry["source_sha256"] in source_hashes]
    metadata = row.get("metadata") or {}
    identities = {str(value) for value in (row.get("id"), row.get("source_id"),
                  row.get("response_id"), metadata.get("source_id"),
                  metadata.get("response_id"), metadata.get("ui_id")) if value}
    if matches:
        reasons.append("pilot_audited_source_quality_hold")
    if any(entry["source_id"] in identities and entry["source_sha256"] not in source_hashes for entry in holds):
        reasons.append("pilot_audited_source_identity_hash_mismatch")
    return {**report, "eligible": not reasons, "reasons": list(dict.fromkeys(reasons)),
            "policy_version": POLICY_VERSION, "pilot_profile": PROFILE,
            "source_sha256": source_hash, "known_holds": matches,
            "source_factuality_verified": False}
