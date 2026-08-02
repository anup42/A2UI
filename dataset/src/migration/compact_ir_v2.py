"""One-time Compact IR v2 (gci2) migration decoder.

The codec reduces syntax only. It never removes a semantic component or an
interaction. Empty/default fields are reconstructed by the decoder before the
renderer contract is evaluated.
"""
from __future__ import annotations
from copy import deepcopy
from typing import Any, Mapping

from pipeline.ir_formats.common import rewrite_element_ids

VERSION = "gci2"


def encode(spec: Mapping[str, Any], *, shorten_ids: bool = True) -> dict[str, Any]:
    source = rewrite_element_ids(spec, shorten=shorten_ids)
    out: dict[str, Any] = {"v": VERSION, "r": source["root"], "e": {}}
    state = source.get("state")
    if isinstance(state, dict) and state:
        out["s"] = deepcopy(state)
    for element_id, raw in source["elements"].items():
        element = deepcopy(raw)
        element_type = str(element.get("type", ""))
        props = element.get("props") if isinstance(element.get("props"), dict) else {}
        props = deepcopy(props)
        # Layout aliases are semantic-preserving and avoid repeating direction.
        if element_type == "Stack" and props.get("direction") in {"horizontal", "row"}:
            compact_type = "Row"; props.pop("direction", None)
        elif element_type == "Stack" and props.get("direction") in {"vertical", "column"}:
            compact_type = "Column"; props.pop("direction", None)
        else:
            compact_type = element_type
        item: dict[str, Any] = {"t": compact_type}
        if props:
            item["p"] = props
        children = element.get("children")
        if isinstance(children, list) and children:
            item["c"] = deepcopy(children)
        if isinstance(element.get("repeat"), dict): item["x"] = deepcopy(element["repeat"])
        if "visible" in element: item["z"] = deepcopy(element["visible"])
        if isinstance(element.get("on"), dict) and element["on"]: item["o"] = deepcopy(element["on"])
        if isinstance(element.get("watch"), dict) and element["watch"]: item["w"] = deepcopy(element["watch"])
        out["e"][element_id] = item
    return out


def decode(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("Compact IR payload must be an object")
    if payload.get("v") != VERSION:
        raise ValueError(f"Compact IR version must be {VERSION!r}")
    root = payload.get("r")
    elements = payload.get("e")
    if not isinstance(root, str) or not root.strip():
        raise ValueError("Compact IR r must be a non-empty string")
    if not isinstance(elements, Mapping) or not elements:
        raise ValueError("Compact IR e must be a non-empty object")
    state = payload.get("s", {})
    if not isinstance(state, Mapping):
        raise ValueError("Compact IR s must be an object")
    decoded: dict[str, Any] = {"root": root, "state": deepcopy(dict(state)), "elements": {}}
    for element_id, raw in elements.items():
        if not isinstance(element_id, str) or not element_id:
            raise ValueError("Compact IR element IDs must be non-empty strings")
        if not isinstance(raw, Mapping):
            raise ValueError(f"Compact IR element {element_id!r} must be an object")
        compact_type = raw.get("t")
        if not isinstance(compact_type, str) or not compact_type:
            raise ValueError(f"Compact IR element {element_id!r} requires t")
        props = raw.get("p", {})
        children = raw.get("c", [])
        if not isinstance(props, Mapping):
            raise ValueError(f"Compact IR element {element_id!r} p must be an object")
        if not isinstance(children, list) or any(not isinstance(x, str) or not x for x in children):
            raise ValueError(f"Compact IR element {element_id!r} c must be an array of non-empty strings")
        props_out = deepcopy(dict(props))
        if compact_type == "Row":
            element_type = "Stack"; props_out.setdefault("direction", "horizontal")
        elif compact_type == "Column":
            element_type = "Stack"; props_out.setdefault("direction", "vertical")
        else:
            element_type = compact_type
        element: dict[str, Any] = {"type": element_type, "props": props_out, "children": deepcopy(children)}
        for compact_key, full_key, expected in (
            ("x", "repeat", Mapping), ("o", "on", Mapping), ("w", "watch", Mapping)
        ):
            if compact_key in raw:
                value = raw[compact_key]
                if not isinstance(value, expected):
                    raise ValueError(f"Compact IR element {element_id!r} {compact_key} has invalid type")
                element[full_key] = deepcopy(dict(value))
        if "z" in raw: element["visible"] = deepcopy(raw["z"])
        unknown = set(raw) - {"t", "p", "c", "x", "z", "o", "w"}
        if unknown:
            raise ValueError(f"Compact IR element {element_id!r} has unknown keys {sorted(unknown)}")
        decoded["elements"][element_id] = element
    unknown_top = set(payload) - {"v", "r", "s", "e"}
    if unknown_top:
        raise ValueError(f"Compact IR has unknown keys {sorted(unknown_top)}")
    return decoded
