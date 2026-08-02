"""GenUICraft custom-catalog A2UI v1 wire codec.

The decoder accepts a create-surface envelope or an ordered v1 message stream.
All messages normalize into the same canonical FlatSpec graph used by Android.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping

from .common import load_catalog, load_manifest, rewrite_element_ids

VERSION = "v1.0"
DEFAULT_SURFACE_ID = "default_surface"


def encode(spec: Mapping[str, Any], *, shorten_ids: bool = True) -> dict[str, Any]:
    source = rewrite_element_ids(spec, shorten=shorten_ids)
    manifest = load_manifest()
    components: list[dict[str, Any]] = []
    for element_id, raw in source["elements"].items():
        component: dict[str, Any] = {
            "id": element_id,
            "component": raw["type"],
        }
        props = raw.get("props") if isinstance(raw.get("props"), dict) else {}
        component.update(deepcopy(props))
        for key in ("children", "repeat", "visible", "on", "watch"):
            value = raw.get(key)
            if value not in (None, {}, []):
                component[key] = deepcopy(value)
        components.append(component)

    create: dict[str, Any] = {
        "surfaceId": DEFAULT_SURFACE_ID,
        "catalogId": manifest["catalogId"],
        "rootId": source["root"],
        "components": components,
    }
    if source.get("state"):
        create["dataModel"] = deepcopy(source["state"])
    return {"version": VERSION, "createSurface": create}


def decode(payload: Any) -> dict[str, Any]:
    messages = _messages(payload)
    elements: dict[str, dict[str, Any]] = {}
    state: dict[str, Any] = {}
    surface_id: str | None = None
    root_id: str | None = None
    deleted = False

    for message in messages:
        if message.get("version") != VERSION:
            raise ValueError(f"A2UI wire version must be {VERSION}")

        if "createSurface" in message:
            create = _mapping(message["createSurface"], "createSurface")
            surface_id = _surface_id(create, surface_id)
            _validate_catalog(create.get("catalogId"))
            root_candidate = create.get("rootId")
            if root_candidate is not None:
                if not isinstance(root_candidate, str) or not root_candidate.strip():
                    raise ValueError("A2UI createSurface.rootId must be a non-empty string")
                root_id = root_candidate
            raw_state = create.get("dataModel", {})
            if not isinstance(raw_state, Mapping):
                raise ValueError("A2UI createSurface.dataModel must be an object")
            state = deepcopy(dict(raw_state))
            if "components" in create:
                for element_id, element in _decode_components(create["components"]):
                    elements[element_id] = element
            deleted = False
            continue

        if "updateComponents" in message:
            update = _mapping(message["updateComponents"], "updateComponents")
            surface_id = _surface_id(update, surface_id)
            for element_id, element in _decode_components(update.get("components")):
                elements[element_id] = element
            continue

        if "updateDataModel" in message:
            update = _mapping(message["updateDataModel"], "updateDataModel")
            surface_id = _surface_id(update, surface_id)
            if "value" not in update:
                raise ValueError("A2UI updateDataModel requires value")
            _set_data_path(state, update.get("path"), deepcopy(update["value"]))
            continue

        if "deleteSurface" in message:
            delete = _mapping(message["deleteSurface"], "deleteSurface")
            surface_id = _surface_id(delete, surface_id)
            deleted = True
            elements.clear()
            state.clear()
            continue

        raise ValueError("Unsupported A2UI v1 wire message")

    if deleted:
        raise ValueError("A2UI surface was deleted before conversion")
    if not elements:
        raise ValueError("A2UI wire payload did not produce any components")
    resolved_root = root_id or ("root" if "root" in elements else next(iter(elements)))
    if resolved_root not in elements:
        raise ValueError(f"A2UI rootId {resolved_root!r} does not exist in components")
    return {
        "root": resolved_root,
        "state": state,
        "elements": elements,
    }


def _messages(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        return [payload]
    if isinstance(payload, list) and payload:
        if all(isinstance(item, Mapping) for item in payload):
            return list(payload)
    raise ValueError("A2UI wire payload must be an object or non-empty message array")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"A2UI {label} must be an object")
    return value


def _surface_id(message: Mapping[str, Any], current: str | None) -> str:
    raw = message.get("surfaceId")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("A2UI message requires a non-empty surfaceId")
    if current is not None and raw != current:
        raise ValueError("A2UI conversion supports one surface per payload")
    return raw


def _validate_catalog(value: Any) -> None:
    if value is None:
        return
    expected = str(load_manifest()["catalogId"])
    if not isinstance(value, str) or value != expected:
        raise ValueError(f"A2UI createSurface.catalogId must be {expected!r}")


def _decode_components(value: Any) -> Iterable[tuple[str, dict[str, Any]]]:
    if not isinstance(value, list) or not value:
        raise ValueError("A2UI components must be a non-empty array")
    allowed = set(load_catalog().get("components", {}))
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError("A2UI component must be an object")
        element_id = raw.get("id")
        component_type = raw.get("component")
        if not isinstance(element_id, str) or not element_id:
            raise ValueError("A2UI component requires a non-empty id")
        if element_id in seen:
            raise ValueError(f"Duplicate A2UI component id {element_id!r}")
        seen.add(element_id)
        if not isinstance(component_type, str) or component_type not in allowed:
            raise ValueError(f"Unsupported A2UI component {component_type!r}")

        metadata = {"id", "component", "children", "repeat", "visible", "on", "watch"}
        props = {key: deepcopy(item) for key, item in raw.items() if key not in metadata}
        children = raw.get("children", [])
        if not isinstance(children, list) or any(not isinstance(item, str) or not item for item in children):
            raise ValueError(f"A2UI component {element_id!r} children must be non-empty strings")
        element: dict[str, Any] = {
            "type": component_type,
            "props": props,
            "children": deepcopy(children),
        }
        for key in ("repeat", "on", "watch"):
            if key in raw:
                item = raw[key]
                if not isinstance(item, Mapping):
                    raise ValueError(f"A2UI component {element_id!r} {key} must be an object")
                element[key] = deepcopy(dict(item))
        if "visible" in raw:
            element["visible"] = deepcopy(raw["visible"])
        yield element_id, element


def _set_data_path(state: dict[str, Any], raw_path: Any, value: Any) -> None:
    if raw_path in (None, "", "/"):
        if not isinstance(value, Mapping):
            raise ValueError("Root A2UI data-model update must be an object")
        state.clear()
        state.update(deepcopy(dict(value)))
        return
    if not isinstance(raw_path, str) or not raw_path.startswith("/"):
        raise ValueError("A2UI updateDataModel.path must be a JSON pointer")
    parts = [
        token.replace("~1", "/").replace("~0", "~")
        for token in raw_path.split("/")[1:]
    ]
    current = state
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value
