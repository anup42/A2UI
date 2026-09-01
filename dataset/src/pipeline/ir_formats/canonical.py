"""Strict canonical UI-graph contract used after an active Express parse.

This module deliberately has no legacy-container or migration behavior.  The
legacy FlatSpec contract remains in ``pipeline.flat_spec_contract`` and is
entered only by explicit migration/offline callers.  Active Express and wire
code use this contract so malformed model output cannot be normalized through
the legacy importer.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from ..renderer_semantics import iter_renderer_references


@dataclass(frozen=True)
class CanonicalValidation:
    spec: dict[str, Any] | None
    error: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.spec is not None and self.error is None


_REPEAT_FIELDS = {"statePath", "key", "template", "itemTemplate", "child"}
_REQUIRED_ACTION_PARAMS = {
    "openUrl": {"url"},
    "setState": {"statePath", "value"},
    "pushState": {"statePath", "value"},
    "removeState": {"statePath", "index"},
    "validateForm": set(),
    "emitEvent": {"name"},
}


def validate_canonical_graph(
    value: Any,
    *,
    require_root: bool = True,
) -> CanonicalValidation:
    """Validate a canonical typed graph without legacy coercion.

    ``require_root=False`` is used only by the encoder before deterministic ID
    rewriting; the active decoded graph always requires the reserved ``root``
    id.  The returned graph is a deep copy and is never silently rewritten.
    """

    if not isinstance(value, Mapping):
        return CanonicalValidation(None, "Canonical UI graph must be an object")
    allowed_top = {"root", "state", "elements"}
    unknown_top = sorted(set(value) - allowed_top)
    if unknown_top:
        return CanonicalValidation(None, f"Unknown canonical graph field(s): {', '.join(map(str, unknown_top))}")

    root = value.get("root")
    if not isinstance(root, str) or not root.strip():
        return CanonicalValidation(None, "Canonical graph root must be a non-empty string")
    if require_root and root != "root":
        return CanonicalValidation(None, "Canonical graph root must be the reserved 'root' id")

    state = value.get("state")
    if not isinstance(state, Mapping):
        return CanonicalValidation(None, "Canonical graph state must be an object")
    elements = value.get("elements")
    if not isinstance(elements, Mapping) or not elements:
        return CanonicalValidation(None, "Canonical graph elements must be a non-empty object")
    if root not in elements:
        return CanonicalValidation(None, f"Canonical graph root id {root!r} is missing")

    catalog, actions = _catalog_and_actions()
    element_ids = {str(key) for key in elements}
    normalized: dict[str, Any] = {
        "root": root,
        "state": deepcopy(dict(state)),
        "elements": {},
    }
    for raw_id, raw_element in elements.items():
        element_id = str(raw_id)
        if not element_id.strip():
            return CanonicalValidation(None, "Canonical graph element ids must be non-empty")
        if not isinstance(raw_element, Mapping):
            return CanonicalValidation(None, f"Element {element_id!r} must be an object")
        allowed_fields = {"type", "props", "children", "repeat", "visible", "on", "watch"}
        unknown_fields = sorted(set(raw_element) - allowed_fields)
        if unknown_fields:
            return CanonicalValidation(
                None,
                f"Element {element_id!r} has unsupported field(s): {', '.join(map(str, unknown_fields))}",
            )
        element_type = raw_element.get("type")
        if not isinstance(element_type, str) or not element_type.strip():
            return CanonicalValidation(None, f"Element {element_id!r} has no canonical type")
        descriptor = catalog.get(element_type)
        if not isinstance(descriptor, Mapping):
            return CanonicalValidation(None, f"Element {element_id!r} has unsupported type {element_type!r}")
        if descriptor.get("canonicalType") and descriptor.get("canonicalType") != element_type:
            return CanonicalValidation(None, f"Element {element_id!r} uses non-canonical type {element_type!r}")

        props = raw_element.get("props")
        if not isinstance(props, Mapping):
            return CanonicalValidation(None, f"Element {element_id!r} props must be an object")
        allowed_props = set(_allowed_properties(descriptor))
        unknown_props = sorted(set(props) - allowed_props)
        if unknown_props:
            return CanonicalValidation(
                None,
                f"Element {element_id!r} has unsupported prop(s): {', '.join(map(str, unknown_props))}",
            )
        normalized_props = deepcopy(dict(props))
        if element_type in {"Stack", "Row", "Column"}:
            # Models commonly emit CSS/UI shorthand. Lower harmless aliases
            # before strict catalog validation to avoid repair calls.
            gap_aliases = {"xs": "sm", "small": "sm", "medium": "md", "large": "lg"}
            wrap_aliases = {"yes": "wrap", "no": "nowrap"}
            if normalized_props.get("gap") in gap_aliases:
                normalized_props["gap"] = gap_aliases[normalized_props["gap"]]
            if normalized_props.get("wrap") in wrap_aliases:
                normalized_props["wrap"] = wrap_aliases[normalized_props["wrap"]]
            if normalized_props.get("justify") == "between":
                normalized_props["justify"] = "spaceBetween"
        property_error = _validate_property_values(
            element_type, normalized_props, f"Element {element_id!r}.props"
        )
        if property_error:
            return CanonicalValidation(None, property_error)

        children = raw_element.get("children")
        if not isinstance(children, list) or any(not isinstance(child, str) or not child.strip() for child in children):
            return CanonicalValidation(None, f"Element {element_id!r} children must be non-empty string references")

        normalized_element: dict[str, Any] = {
            "type": element_type,
            "props": normalized_props,
            "children": deepcopy(children),
        }
        if "repeat" in raw_element:
            repeat = raw_element.get("repeat")
            if not isinstance(repeat, Mapping):
                return CanonicalValidation(None, f"Element {element_id!r} repeat must be an object")
            state_path = repeat.get("statePath")
            if not isinstance(state_path, str) or not state_path.strip():
                return CanonicalValidation(None, f"Element {element_id!r} repeat.statePath is required")
            unknown_repeat = sorted(set(repeat) - _REPEAT_FIELDS)
            if unknown_repeat:
                return CanonicalValidation(
                    None,
                    f"Element {element_id!r} repeat has unsupported field(s): {', '.join(map(str, unknown_repeat))}",
                )
            normalized_element["repeat"] = deepcopy(dict(repeat))
        if "visible" in raw_element:
            normalized_element["visible"] = deepcopy(raw_element["visible"])
        if "on" in raw_element:
            error = _validate_action_map(raw_element.get("on"), actions, f"Element {element_id!r}.on")
            if error:
                return CanonicalValidation(None, error)
            normalized_element["on"] = deepcopy(dict(raw_element["on"]))
        if "watch" in raw_element:
            watch = raw_element.get("watch")
            if not isinstance(watch, Mapping):
                return CanonicalValidation(None, f"Element {element_id!r}.watch must be an object")
            for path, action in watch.items():
                if not isinstance(path, str) or not path.startswith("/") or not path.strip("/"):
                    return CanonicalValidation(None, f"Element {element_id!r}.watch key {path!r} must be a JSON pointer")
                error = _validate_action(action, actions, f"Element {element_id!r}.watch.{path}")
                if error:
                    return CanonicalValidation(None, error)
            normalized_element["watch"] = deepcopy(dict(watch))
        normalized["elements"][element_id] = normalized_element

    # Use the shared renderer-reference inventory for every edge, including
    # Tabs/Modal/reference properties and repeat templates.
    for element_id, element in normalized["elements"].items():
        for reference in iter_renderer_references(element):
            if reference.target_id not in element_ids:
                return CanonicalValidation(
                    None,
                    f"Element {element_id!r} references missing {reference.target_id!r} at {reference.source_path}",
                )

    cycle = _reachable_cycle(normalized["root"], normalized["elements"])
    if cycle is not None:
        return CanonicalValidation(None, f"Canonical graph contains reachable reference cycle at {cycle!r}")
    return CanonicalValidation(normalized)


def canonical_graph(value: Any, *, require_root: bool = True) -> dict[str, Any]:
    result = validate_canonical_graph(value, require_root=require_root)
    if not result.is_valid or result.spec is None:
        raise ValueError(result.error or "Invalid canonical UI graph")
    return result.spec


def rewrite_element_ids(
    value: Mapping[str, Any],
    *,
    shorten: bool = True,
    reserve_root: bool = False,
) -> dict[str, Any]:
    """Rewrite all renderer references without invoking legacy normalization."""

    source = canonical_graph(value, require_root=False)
    elements = source["elements"]
    root = source["root"]
    order: list[str] = []
    seen: set[str] = set()

    def walk(element_id: str) -> None:
        if element_id in seen or element_id not in elements:
            return
        seen.add(element_id)
        order.append(element_id)
        for reference in iter_renderer_references(elements[element_id]):
            walk(reference.target_id)

    walk(root)
    for element_id in elements:
        walk(str(element_id))

    if shorten:
        mapping = {root: "root"}
        mapping.update({old: _short_id(index) for index, old in enumerate(x for x in order if x != root)})
    else:
        mapping = {old: old for old in order}
        if reserve_root and root != "root":
            replacement = "root"
            if replacement in mapping:
                index = 1
                while f"root_{index}" in mapping.values():
                    index += 1
                mapping[root] = f"root_{index}"
            else:
                mapping[root] = replacement

    rewritten: dict[str, Any] = {}
    for old_id in order:
        element = deepcopy(elements[old_id])
        for reference in iter_renderer_references(element):
            _set_path(element, reference.source_path, mapping[reference.target_id])
        rewritten[mapping[old_id]] = element
    return {
        "root": mapping.get(root, root),
        "state": deepcopy(source["state"]),
        "elements": rewritten,
    }


def semantic_hash(value: Mapping[str, Any]) -> str:
    canonical = rewrite_element_ids(value, shorten=True, reserve_root=True)
    encoded = json.dumps(
        _normalize_semantic_bindings(canonical),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalize_semantic_bindings(value: Any, *, field_name: str | None = None) -> Any:
    """Treat Express path strings and standard binding objects as equivalent.

    The canonical graph keeps the renderer's historical path-string shape for
    compatibility, while the compiled wire uses the standard ``{"path": ...}``
    binding object.  Semantic identity must not change at that transport
    boundary.
    """

    if isinstance(value, Mapping):
        return {
            str(key): _normalize_semantic_bindings(item, field_name=str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_semantic_bindings(item) for item in value]
    if isinstance(value, str) and field_name == "statePath":
        return _normalize_pointer_string(value)
    if isinstance(value, str) and (value.startswith("$/") or value.startswith("$state.")):
        return {"path": _normalize_pointer_string(value)}
    return value


def _normalize_pointer_string(value: str) -> str:
    if value == "$":
        return "/"
    if value.startswith("$/"):
        return "/" + value[2:]
    if value.startswith("$state."):
        return "/" + value[7:].replace(".", "/")
    return value


def _catalog_and_actions() -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    from .common import load_catalog

    catalog = load_catalog()
    components = catalog.get("components") if isinstance(catalog.get("components"), Mapping) else {}
    actions = catalog.get("actions") if isinstance(catalog.get("actions"), Mapping) else {}
    return dict(components), dict(actions)


def _allowed_properties(descriptor: Mapping[str, Any]) -> list[str]:
    values = descriptor.get("allowedProperties") or descriptor.get("consumedProps") or descriptor.get("positional") or []
    return [str(value) for value in values]


def _validate_action_map(value: Any, actions: Mapping[str, Mapping[str, Any]], context: str) -> str | None:
    if not isinstance(value, Mapping):
        return f"{context} must be an object"
    for name, action in value.items():
        error = _validate_action(action, actions, f"{context}.{name}")
        if error:
            return error
    return None


def _validate_property_values(element_type: str, props: Mapping[str, Any], context: str) -> str | None:
    """Validate catalog/profile enum values without parser-side normalization."""

    if element_type in {"Stack", "Row", "Column"}:
        direction = props.get("direction")
        if direction is not None and direction not in {"vertical", "horizontal"}:
            return f"{context}.direction must be 'vertical' or 'horizontal'"
        gap = props.get("gap")
        if gap is not None and gap not in {"none", "sm", "md", "lg", "xl"}:
            return f"{context}.gap must be one of none, sm, md, lg, xl"
        wrap = props.get("wrap")
        if wrap is not None and wrap not in {"nowrap", "wrap"}:
            return f"{context}.wrap must be 'nowrap' or 'wrap'"
        justify = props.get("justify")
        if justify is not None and justify not in {
            "start",
            "center",
            "end",
            "stretch",
            "spaceAround",
            "spaceBetween",
            "spaceEvenly",
            "between",
        }:
            return f"{context}.justify must be a supported distribution"
    return None


def _validate_action(value: Any, actions: Mapping[str, Mapping[str, Any]], context: str) -> str | None:
    if isinstance(value, list):
        if not value:
            return f"{context} action list must not be empty"
        for index, item in enumerate(value):
            error = _validate_action(item, actions, f"{context}[{index}]")
            if error:
                return error
        return None
    if not isinstance(value, Mapping):
        return f"{context} must be an action object"
    action_name = value.get("action")
    if not isinstance(action_name, str) or action_name not in actions:
        return f"{context} uses unknown action {action_name!r}"
    params = value.get("params", {})
    if not isinstance(params, Mapping):
        return f"{context}.params must be an object"
    descriptor = actions[action_name]
    allowed = set(str(item) for item in descriptor.get("positional", ()))
    unknown = sorted(set(params) - allowed)
    if unknown and not descriptor.get("allowAdditionalParams", False):
        return f"{context}.params has unsupported field(s): {', '.join(map(str, unknown))}"
    missing = sorted(_REQUIRED_ACTION_PARAMS.get(action_name, set()) - set(params))
    if missing:
        return f"{context}.params is missing required field(s): {', '.join(missing)}"
    return None


def _reachable_cycle(root: str, elements: Mapping[str, Mapping[str, Any]]) -> str | None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(element_id: str) -> str | None:
        if element_id in visiting:
            return element_id
        if element_id in visited:
            return None
        visiting.add(element_id)
        for reference in iter_renderer_references(elements[element_id]):
            cycle = visit(reference.target_id)
            if cycle is not None:
                return cycle
        visiting.remove(element_id)
        visited.add(element_id)
        return None

    return visit(root)


def _set_path(root: Any, path: str, value: Any) -> None:
    current = root
    tokens: list[str | int] = []
    buffer = ""
    i = 0
    while i < len(path):
        ch = path[i]
        if ch == ".":
            if buffer:
                tokens.append(buffer)
                buffer = ""
        elif ch == "[":
            if buffer:
                tokens.append(buffer)
                buffer = ""
            end = path.index("]", i)
            tokens.append(int(path[i + 1:end]))
            i = end
        else:
            buffer += ch
        i += 1
    if buffer:
        tokens.append(buffer)
    if not tokens:
        raise ValueError(f"Empty renderer reference path {path!r}")
    for token in tokens[:-1]:
        current = current[token]
    current[tokens[-1]] = value


def _short_id(index: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    out = ""
    n = index
    while True:
        out = alphabet[n % 26] + out
        n = n // 26 - 1
        if n < 0:
            return out


__all__ = [
    "CanonicalValidation",
    "canonical_graph",
    "rewrite_element_ids",
    "semantic_hash",
    "validate_canonical_graph",
]
