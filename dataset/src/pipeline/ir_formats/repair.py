"""Explicit, loss-aware repairs for historical UI graphs.

Repairs run before the strict Express materializer.  They are deliberately
small and deterministic: layout aliases may be normalized, unsafe URL actions
may be removed, and a few known presentation-only fields may be translated.
Unknown semantic properties are left untouched so strict validation still
rejects data that cannot be represented safely.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import math
import re
from typing import Any, Mapping
from urllib.parse import urlparse

from .common import load_catalog


GAP_TOKENS = ("none", "sm", "md", "lg", "xl")

_GAP_ALIASES = {
    "0": "none",
    "none": "none",
    "zero": "none",
    "xs": "sm",
    "xsmall": "sm",
    "extrasmall": "sm",
    "small": "sm",
    "sm": "sm",
    "md": "md",
    "medium": "md",
    "normal": "md",
    "default": "md",
    "lg": "lg",
    "large": "lg",
    "xl": "xl",
    "xlarge": "xl",
    "extralarge": "xl",
}

_LAYOUT_ONLY_PROPS = {
    "paddingTop",
    "paddingBottom",
    "paddingLeft",
    "paddingRight",
    "marginTop",
    "marginBottom",
    "marginLeft",
    "marginRight",
}


@dataclass(frozen=True)
class RepairChange:
    """One auditable transformation applied to a graph."""

    kind: str
    path: str
    detail: str = ""

    def as_dict(self) -> dict[str, str]:
        value = {"kind": self.kind, "path": self.path}
        if self.detail:
            value["detail"] = self.detail
        return value


@dataclass(frozen=True)
class RepairResult:
    """Repaired graph plus the exact transformations that were applied."""

    graph: dict[str, Any]
    changes: tuple[RepairChange, ...] = ()

    @property
    def applied(self) -> bool:
        return bool(self.changes)

    def merge(self, other: "RepairResult") -> "RepairResult":
        return RepairResult(
            graph=other.graph,
            changes=self.changes + other.changes,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "changes": [change.as_dict() for change in self.changes],
        }


def repair_graph(value: Mapping[str, Any]) -> RepairResult:
    """Repair a canonical-shaped graph without weakening strict validation."""

    graph = deepcopy(dict(value))
    elements = graph.get("elements")
    if not isinstance(elements, Mapping):
        return RepairResult(graph=graph)

    catalog = _catalog_components()
    changes: list[RepairChange] = []
    repaired_elements: dict[str, Any] = {}
    for raw_id, raw_element in elements.items():
        element_id = str(raw_id)
        if not isinstance(raw_element, Mapping):
            repaired_elements[element_id] = raw_element
            continue

        element = deepcopy(dict(raw_element))
        element_type = str(element.get("type") or "")
        props = element.get("props")
        if isinstance(props, Mapping):
            repaired_props = deepcopy(dict(props))
            _repair_gap(repaired_props, element_id, changes)
            _repair_known_semantic_props(element_type, repaired_props, element_id, changes)
            _drop_known_layout_aliases(
                element_type,
                repaired_props,
                element_id,
                catalog,
                changes,
            )
            element["props"] = repaired_props

        for field in ("on", "watch"):
            actions = element.get(field)
            if isinstance(actions, Mapping):
                repaired_actions = _repair_action_map(actions, f"{element_id}.{field}", changes)
                if repaired_actions:
                    element[field] = repaired_actions
                else:
                    element.pop(field, None)

        repaired_elements[element_id] = element

    graph["elements"] = repaired_elements
    return RepairResult(graph=graph, changes=tuple(changes))


@lru_cache(maxsize=1)
def _catalog_components() -> Mapping[str, Any]:
    catalog = load_catalog()
    components = catalog.get("components", {})
    return components if isinstance(components, Mapping) else {}


def _repair_gap(
    props: dict[str, Any],
    element_id: str,
    changes: list[RepairChange],
) -> None:
    if "gap" not in props:
        return
    original = props["gap"]
    normalized = normalize_gap(original)
    if normalized == original:
        return
    props["gap"] = normalized
    changes.append(
        RepairChange(
            kind="gap_normalized",
            path=f"elements.{element_id}.props.gap",
            detail=f"{original!r}->{normalized!r}",
        )
    )


def normalize_gap(value: Any) -> str:
    """Map common/custom spacing values to the nearest renderer token."""

    if isinstance(value, bool):
        return "md"
    if isinstance(value, (int, float)):
        return _numeric_gap(float(value))
    if isinstance(value, str):
        token = value.strip().lower()
        if token in GAP_TOKENS:
            return token
        compact = re.sub(r"[\s_\-]+", "", token)
        if compact in _GAP_ALIASES:
            return _GAP_ALIASES[compact]
        try:
            return _numeric_gap(float(token))
        except ValueError:
            return "md"
    return "md"


def _numeric_gap(value: float) -> str:
    if not math.isfinite(value) or value <= 0:
        return "none"
    if value <= 8:
        return "sm"
    if value <= 16:
        return "md"
    if value <= 24:
        return "lg"
    return "xl"


def _repair_known_semantic_props(
    element_type: str,
    props: dict[str, Any],
    element_id: str,
    changes: list[RepairChange],
) -> None:
    # EmailPreview has no dedicated signature field in the active catalog, but
    # moving it into the body preserves the user-visible information.
    if element_type != "EmailPreview" or "signature" not in props:
        return
    signature = props.pop("signature")
    body = props.get("body")
    if body in (None, ""):
        props["body"] = signature
    elif isinstance(body, str) and isinstance(signature, str):
        props["body"] = f"{body}\n\n{signature}"
    else:
        # Keep the property for strict validation when it cannot be represented
        # without changing its value's type.
        props["signature"] = signature
        return
    changes.append(
        RepairChange(
            kind="semantic_prop_mapped",
            path=f"elements.{element_id}.props.signature",
            detail="signature->body",
        )
    )


def _drop_known_layout_aliases(
    element_type: str,
    props: dict[str, Any],
    element_id: str,
    catalog: Any,
    changes: list[RepairChange],
) -> None:
    descriptor = catalog.get(element_type, {}) if isinstance(catalog, Mapping) else {}
    allowed = set(descriptor.get("allowedProperties", ())) if isinstance(descriptor, Mapping) else set()
    if not allowed and isinstance(descriptor, Mapping):
        allowed = set(descriptor.get("consumedProps", ())) | set(descriptor.get("positional", ()))
    for prop in list(props):
        if prop not in _LAYOUT_ONLY_PROPS or prop in allowed:
            continue
        props.pop(prop, None)
        changes.append(
            RepairChange(
                kind="layout_prop_dropped",
                path=f"elements.{element_id}.props.{prop}",
                detail="unsupported_layout_alias",
            )
        )


def _repair_action_map(
    actions: Mapping[str, Any],
    path: str,
    changes: list[RepairChange],
) -> dict[str, Any]:
    repaired: dict[str, Any] = {}
    for event_name, value in actions.items():
        action = _repair_action(value, f"{path}.{event_name}", changes)
        if action is not None:
            repaired[str(event_name)] = action
    return repaired


def _repair_action(value: Any, path: str, changes: list[RepairChange]) -> Any:
    if isinstance(value, list):
        repaired = []
        for index, item in enumerate(value):
            action = _repair_action(item, f"{path}[{index}]", changes)
            if action is not None:
                repaired.append(action)
        return repaired or None
    if not isinstance(value, Mapping):
        return value
    action = deepcopy(dict(value))
    if str(action.get("action") or "").strip().lower() != "openurl":
        return action
    params = action.get("params")
    url = params.get("url") if isinstance(params, Mapping) else None
    if _is_safe_url(url):
        return action
    changes.append(
        RepairChange(
            kind="unsafe_url_action_removed",
            path=path,
            detail="openUrl",
        )
    )
    return None


def _is_safe_url(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    token = value.strip()
    if re.fullmatch(r"\[(?:(?:ACTION|SOURCE)_URL|URL)_\d+\]", token):
        return True
    if re.fullmatch(r"tel:[+0-9().\-\s]{3,}", token, flags=re.IGNORECASE):
        return True
    parsed = urlparse(token)
    return parsed.scheme.lower() == "https" and bool(parsed.netloc)


__all__ = ["GAP_TOKENS", "RepairChange", "RepairResult", "normalize_gap", "repair_graph"]
