"""Shared Android flat-renderer capabilities and truthful coverage metrics.

The source of truth lives in ``dataset/schema/renderer_capabilities.json``.
Android generates a Kotlin contract from the same file, so Python metrics never
need to guess which types, props, domains, actions, or chart subtypes are real.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlparse

from .flat_spec_semantics import iter_renderer_references

_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "schema" / "renderer_capabilities.json"


def _load_shared_manifest() -> dict[str, Any]:
    try:
        manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Unable to load shared renderer capability manifest at {_MANIFEST_PATH}"
        ) from error
    if not isinstance(manifest, dict) or manifest.get("version") != "2.0.0":
        raise RuntimeError("renderer_capabilities.json must use capability version 2.0.0")
    return manifest


_MANIFEST = _load_shared_manifest()
RENDERER_CAPABILITY_VERSION = str(_MANIFEST["version"])


class _Type:
    __slots__ = ("canonical", "runtime_key", "aliases", "props", "audited")

    def __init__(
        self,
        canonical: str,
        runtime_key: str,
        aliases: tuple[str, ...],
        props: tuple[str, ...],
    ) -> None:
        self.canonical = canonical
        self.runtime_key = runtime_key
        self.aliases = aliases
        self.props = frozenset(props)
        # Version 2 admits only prop-audited types into the manifest.
        self.audited = True


_TYPES: tuple[_Type, ...] = tuple(
    _Type(
        canonical=str(entry["canonical"]),
        runtime_key=str(entry["runtime_key"]),
        aliases=tuple(str(value).lower() for value in entry.get("aliases", [])),
        props=tuple(str(value) for value in entry.get("consumed_props", [])),
    )
    for entry in _MANIFEST["types"]
)
_ALIAS_TO_TYPE: dict[str, _Type] = {
    alias: entry for entry in _TYPES for alias in entry.aliases
}
_CANONICAL_TO_TYPE = {entry.canonical: entry for entry in _TYPES}
for alias, descriptor in _MANIFEST.get("compatibility_type_aliases", {}).items():
    canonical = descriptor.get("canonical")
    if canonical in _CANONICAL_TO_TYPE:
        _ALIAS_TO_TYPE[str(alias).lower()] = _CANONICAL_TO_TYPE[canonical]

_ACTION_DESCRIPTORS: dict[str, Mapping[str, Any]] = {
    str(action["runtime_key"]).lower(): action for action in _MANIFEST["actions"]
}
_IMPLEMENTED_ACTIONS = frozenset(_ACTION_DESCRIPTORS)
_STUB_ACTIONS: frozenset[str] = frozenset()

_TABLE_DOMAINS = frozenset(
    str(value).lower() for value in _MANIFEST["table_domains"]["canonical"]
)
_TABLE_DOMAIN_ALIASES = {
    str(alias).lower(): str(canonical).lower()
    for alias, canonical in _MANIFEST["table_domains"].get("aliases", {}).items()
}


def capability_manifest() -> dict[str, Any]:
    """Return a detached, machine-readable copy of the shared contract."""

    result = json.loads(json.dumps(_MANIFEST))
    result["stub_actions"] = sorted(_STUB_ACTIONS)
    result["deprecated_metrics"] = {
        "renderer_support_rate": "structural-only; use renderer_structural_type_coverage"
    }
    return result


def _resolve_type(raw: Any) -> _Type | None:
    if not isinstance(raw, str):
        return None
    return _ALIAS_TO_TYPE.get(raw.strip().lower())


def _iter_action_bindings(element: Mapping[str, Any]):
    for container_key in ("on", "watch"):
        container = element.get(container_key)
        if not isinstance(container, Mapping):
            continue
        for binding in container.values():
            candidates = binding if isinstance(binding, list) else [binding]
            for candidate in candidates:
                if isinstance(candidate, Mapping):
                    yield candidate


def _iter_actions(element: Mapping[str, Any]):
    """Compatibility iterator used by older tests and reports."""

    for binding in _iter_action_bindings(element):
        name = binding.get("action")
        if isinstance(name, str) and name.strip():
            yield name.strip().lower()


def _has_nonblank(params: Mapping[str, Any], key: str) -> bool:
    if key not in params:
        return False
    value = params[key]
    return not isinstance(value, str) or bool(value.strip())


def _safe_action_url(raw: Any) -> bool:
    if not isinstance(raw, str) or not raw.strip():
        return False
    value = raw.strip()
    if value.lower().startswith("tel:"):
        return bool(re.fullmatch(r"tel:[+0-9().\-\s]{3,}", value, flags=re.IGNORECASE))
    parsed = urlparse(value)
    return parsed.scheme.lower() == "https" and bool(parsed.netloc)


def _is_valid_action(binding: Mapping[str, Any]) -> bool:
    raw_name = binding.get("action")
    if not isinstance(raw_name, str):
        return False
    runtime_key = raw_name.strip().lower()
    descriptor = _ACTION_DESCRIPTORS.get(runtime_key)
    if descriptor is None:
        return False
    params = binding.get("params", {})
    if not isinstance(params, Mapping):
        return False
    if any(key not in params for key in descriptor.get("required", [])):
        return False
    for group in descriptor.get("required_any", []):
        if not any(_has_nonblank(params, str(key)) for key in group):
            return False
    if descriptor.get("safe_url"):
        group = descriptor.get("required_any", [[]])[0]
        raw_url = next((params[key] for key in group if _has_nonblank(params, key)), None)
        if not _safe_action_url(raw_url):
            return False
    return True


def _normalized_table_domain(raw: Any) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    token = raw.strip().lower()
    return _TABLE_DOMAIN_ALIASES.get(token, token)


def renderer_coverage(genui_json: Any) -> dict[str, float]:
    """Return independent coverage dimensions for one flat-spec record."""

    if not isinstance(genui_json, Mapping):
        return {}
    elements = genui_json.get("elements")
    if not isinstance(elements, Mapping):
        return {}

    total = 0
    supported = 0
    audited_props = 0
    consumed_props = 0
    dropped: Counter[str] = Counter()
    unsupported: Counter[str] = Counter()
    total_actions = 0
    valid_actions = 0
    total_tables = 0
    routable_tables = 0
    total_refs = 0
    resolved_refs = 0

    for element in elements.values():
        if not isinstance(element, Mapping):
            continue
        total += 1
        entry = _resolve_type(element.get("type"))
        if entry is None:
            unsupported[str(element.get("type"))] += 1
        else:
            supported += 1
            props = element.get("props")
            if isinstance(props, Mapping):
                for key in props:
                    audited_props += 1
                    if key in entry.props:
                        consumed_props += 1
                    else:
                        dropped[f"{entry.canonical}.{key}"] += 1
                if entry.runtime_key == "table":
                    total_tables += 1
                    if _normalized_table_domain(props.get("domain")) in _TABLE_DOMAINS:
                        routable_tables += 1

        for binding in _iter_action_bindings(element):
            total_actions += 1
            if _is_valid_action(binding):
                valid_actions += 1

        for reference in iter_renderer_references(element):
            total_refs += 1
            if reference.target_id in elements:
                resolved_refs += 1

    if total == 0:
        return {}

    structural = supported / total
    prop_coverage = consumed_props / audited_props if audited_props > 0 else 1.0
    action_coverage = valid_actions / total_actions if total_actions > 0 else 1.0
    domain_coverage = routable_tables / total_tables if total_tables > 0 else 1.0
    return {
        # Deprecated compatibility name: structural support only in v2.
        "renderer_support_rate": structural,
        "renderer_support_rate_structural_only": structural,
        "renderer_structural_type_coverage": structural,
        "renderer_consumed_prop_coverage": prop_coverage,
        "renderer_action_valid_coverage": action_coverage,
        "renderer_domain_route_coverage": domain_coverage,
        "renderer_element_count": float(total),
        "renderer_unsupported_count": float(total - supported),
        # Deprecated compatibility names retained for one report version.
        "prop_consumption_rate": prop_coverage,
        "action_stub_rate": 1.0 - action_coverage if total_actions > 0 else 0.0,
        "dropped_prop_count": float(sum(dropped.values())),
        "reference_resolution_rate": (
            resolved_refs / total_refs if total_refs > 0 else 1.0
        ),
    }


def renderer_program_coverage(
    records: list[Any],
    *,
    expected_intents: set[str],
    visual_fixture_intents: set[str],
    end_to_end_passed_intents: set[str],
) -> dict[str, float]:
    """Aggregate honest structural/runtime and intent-program dimensions."""

    dimensions = [renderer_coverage(record) for record in records]
    dimensions = [value for value in dimensions if value]

    def average(key: str) -> float:
        if not dimensions:
            return 0.0
        return sum(float(item[key]) for item in dimensions) / len(dimensions)

    denominator = len(expected_intents)
    visual = len(expected_intents & visual_fixture_intents) / denominator if denominator else 1.0
    end_to_end = len(expected_intents & end_to_end_passed_intents) / denominator if denominator else 1.0
    return {
        "renderer_structural_type_coverage": average("renderer_structural_type_coverage"),
        "renderer_consumed_prop_coverage": average("renderer_consumed_prop_coverage"),
        "renderer_action_valid_coverage": average("renderer_action_valid_coverage"),
        "renderer_domain_route_coverage": average("renderer_domain_route_coverage"),
        "renderer_visual_fixture_coverage": visual,
        "renderer_end_to_end_intent_coverage": end_to_end,
    }


def dropped_prop_histogram(records: list[Any]) -> dict[str, int]:
    """Return ``Type.prop -> count`` for emitted but unconsumed props."""

    histogram: Counter[str] = Counter()
    for genui_json in records:
        if not isinstance(genui_json, Mapping):
            continue
        elements = genui_json.get("elements")
        if not isinstance(elements, Mapping):
            continue
        for element in elements.values():
            if not isinstance(element, Mapping):
                continue
            entry = _resolve_type(element.get("type"))
            if entry is None:
                continue
            props = element.get("props")
            if not isinstance(props, Mapping):
                continue
            for key in props:
                if key not in entry.props:
                    histogram[f"{entry.canonical}.{key}"] += 1
    return dict(histogram.most_common())
