"""Bounded source-reference transport for synthetic augmentation; never fetches.

Only explicit donor bindings and pipeline-declared synthetic references are
authorized. Stage3 remains responsible for generating, validating and restoring
the UI; this module never rewrites a target or relaxes its admission gates.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from pipeline.source_quality import assess_source_quality, parse_actions
from pipeline.stage3_genui import _mask_model_references, _restore_model_references

VERSION = "augmentation-references-v1"
_KINDS = "IMAGE_URL|ICON_URL|MEDIA_URL|ACTION_URL|SOURCE_URL|URL|IMAGE_ASSET|ICON_ASSET|MEDIA_ASSET"
_TOKEN_START = re.compile(rf"\[(?:{_KINDS})_", re.IGNORECASE)
_TOKEN = re.compile(rf"\[(?:{_KINDS})_\d+\]")
_MEDIA = re.compile(r"\b(Icon|Image|Video|Media)\s*=\s*[\"']?(\[(?:" + _KINDS + r")_\d+\])", re.IGNORECASE)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _tokens(text: str) -> list[str]:
    result = []
    for start in _TOKEN_START.finditer(text):
        match = _TOKEN.match(text, start.start())
        if match is None:
            raise ValueError("malformed_reference_placeholder")
        result.append(match.group())
    return result


def _checked_map(mapping: Any) -> dict[str, str]:
    if not isinstance(mapping, dict):
        raise ValueError("invalid_reference_map")  # noqa: TRY004 - external artifact validation
    for token, raw in mapping.items():
        if not isinstance(token, str) or not _TOKEN.fullmatch(token):
            raise ValueError("invalid_reference_map_key")
        if not isinstance(raw, str) or not raw.strip() or raw != raw.strip() or _TOKEN_START.search(raw):
            raise ValueError("invalid_reference_map_value")
        # Quoting preserves whitespace and terminal punctuation in local paths.
        # The canonical matcher still refuses arbitrary prose as a reference.
        _, forward, _ = _mask_model_references('"' + raw + '"', [])
        if raw not in forward:
            raise ValueError("invalid_reference_map_destination")
    return dict(mapping)


def assert_reference_safe_source(source: str) -> None:
    """Fail closed on unresolved source tokens or deterministic action defects.

This is deliberately not source factuality or resource-existence verification.
Use bind_generated_source with the original context for a stricter allowlist
check when resuming a previously generated source.
"""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("empty_reference_source")
    if _tokens(source):
        raise ValueError("unbound_reference_placeholder")
    quality = assess_source_quality({}, source)
    errors = [finding["code"] for finding in quality["findings"] if finding["severity"] == "error"]
    if errors:
        raise ValueError("invalid_source_reference:" + ",".join(sorted(set(errors))))


def build_reference_context(
    donor_text: str, candidate_id: str, *, category: str | None = None,
    donor_reference_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Mask a donor and a minimal, category-specific synthetic reference bank.

Donor bytes outside this returned context are untouched. A donor token may be
restored only from its supplied map, never guessed from its index or role.
Synthetic HTTPS references use a reserved .invalid host and mock actions use
an explicit action: destination. They are identities, not existing resources.
"""
    if not isinstance(donor_text, str) or not donor_text.strip():
        raise ValueError("empty_donor_reference_source")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("empty_candidate_reference_id")
    donor_map = _checked_map(donor_reference_map or {})
    missing = set(_tokens(donor_text)) - set(donor_map)
    if missing:
        raise ValueError("unbound_donor_reference:" + ",".join(sorted(missing)))
    raw_donor = _restore_model_references(donor_text, donor_map)
    assert_reference_safe_source(raw_donor)

    namespace = _hash(candidate_id)[:24]
    synthetic: list[dict[str, str]] = []
    if category == "literal_media_references":
        synthetic = [
            {"role": "image", "destination": f"https://synthetic.invalid/{namespace}/image.png"},
            {"role": "icon", "destination": f"https://synthetic.invalid/{namespace}/icon.svg"},
            {"role": "source", "destination": f"https://synthetic.invalid/{namespace}/source"},
        ]
    elif category in {"action_no_action_contrasts", "forms_rare_controls"}:
        synthetic = [{"role": "action", "destination": f"action://synthetic/{namespace}/mock-submit"}]

    # A single canonical registry, donor first, prevents token-index collisions.
    combined = raw_donor + "\n" + "\n".join(item["destination"] for item in synthetic)
    _, forward, reverse = _mask_model_references(combined, [])
    masked_donor, donor_forward, _ = _mask_model_references(raw_donor, [])
    # Both traversals begin at the identical donor bytes and therefore assign
    # the same tokens, regardless of how many synthetic references follow.
    if any(forward[raw] != token for raw, token in donor_forward.items()):
        raise ValueError("inconsistent_reference_registry")
    supplied = [{"token": token, "destination": raw} for raw, token in donor_forward.items()]
    synthetic = [dict(item, token=forward[item["destination"]]) for item in synthetic]
    return {
        "version": VERSION,
        "donor_response_text": masked_donor,
        "reference_map": reverse,
        "supplied_references": supplied,
        "synthetic_references": synthetic,
        "reference_policy": (
            "Use only exact declared tokens or their exact destinations. Do not invent tokens, URLs, "
            "local assets, reference bindings or backend capabilities. Preserve donor references and "
            "their entity associations when retained. Synthetic references are optional identities: "
            "their reserved .invalid URLs do not claim that an asset or source exists, was downloaded "
            "or externally verified. Synthetic action: destinations are mocks only, never real "
            "operations. Identify synthetic/mock content as such. Use an icon only when relevant; "
            "otherwise omit its declaration. Never use image/icon references as action destinations, "
            "or source/action references as media. An informational no-action variant must omit actions. "
            "These reference rules clarify and override requests for unspecified synthetic placeholders."
        ),
    }


def bind_generated_source(source: str, context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Restore only authorized tokens; safe to repeat on an already bound source."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("empty_generated_reference_source")
    if not isinstance(context, dict) or context.get("version") != VERSION:
        raise ValueError("invalid_reference_context")
    mapping = _checked_map(context.get("reference_map"))
    tokens = set(_tokens(source))
    unknown = tokens - set(mapping)
    if unknown:
        raise ValueError("unknown_generated_reference:" + ",".join(sorted(unknown)))

    _, raw_references, _ = _mask_model_references(source, [])
    unauthorized = set(raw_references) - set(mapping.values())
    if unauthorized:
        raise ValueError("undeclared_generated_reference")
    restored = _restore_model_references(source, mapping)
    assert_reference_safe_source(restored)
    # Validate after substitution too: a known token followed by '/invented'
    # or '?redirect=...' must not create a new, unauthorized destination.
    used_raw = set(_mask_model_references(restored, [])[1])
    if used_raw - set(mapping.values()):
        raise ValueError("modified_generated_reference")

    # Classify explicit live source declarations using the same registry. This
    # catches swapped synthetic roles without interpreting arbitrary prose.
    synthetic = context.get("synthetic_references", [])
    if not isinstance(synthetic, list):
        raise ValueError("invalid_synthetic_reference_context")  # noqa: TRY004 - external artifact validation
    synthetic_roles = {}
    for item in synthetic:
        if not isinstance(item, dict) or mapping.get(item.get("token")) != item.get("destination"):
            raise ValueError("invalid_synthetic_reference_context")
        synthetic_roles[item["destination"]] = item.get("role")
    for action in parse_actions(restored):
        role = synthetic_roles.get(action["destination"])
        if role is not None and role not in {"action", "source"}:
            raise ValueError("synthetic_reference_role_mismatch:action")
    masked, _, canonical_map = _mask_model_references(restored, [])
    for match in _MEDIA.finditer(masked):
        raw = canonical_map[match[2]]
        role = synthetic_roles.get(raw)
        expected = match[1].lower()
        if role is not None and role != expected:
            raise ValueError("synthetic_reference_role_mismatch:" + expected)

    evidence = {
        "version": VERSION,
        "reference_map_sha256": _hash(json.dumps(mapping, ensure_ascii=False, sort_keys=True)),
        "source_sha256": _hash(restored),
        "used_reference_count": len(used_raw),
        "synthetic_reference_count": len(used_raw & set(synthetic_roles)),
        "asset_downloads_performed": False,
        "resource_existence_verified": False,
    }
    return restored, evidence
