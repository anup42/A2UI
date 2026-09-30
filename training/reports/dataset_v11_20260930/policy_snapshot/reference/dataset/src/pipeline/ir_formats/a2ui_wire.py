"""Standard pinned A2UI v0.9 wire codec.

The model-facing representation is A2UI Express.  This module is only the
internal compiler/transport boundary: it emits the upstream v0.9 message
envelopes (``createSurface``, ``updateComponents`` and optional
``updateDataModel``) and lowers canonical graph values to standard data
bindings, child lists, and action structures.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping

from .common import load_catalog, load_manifest, rewrite_element_ids

VERSION = "v0.9"
DEFAULT_SURFACE_ID = "default_surface"
_MISSING = object()
_IMAGE_DIMENSION_TOKENS = {
    "xs": 14,
    "xsmall": 14,
    "extra-small": 14,
    "sm": 18,
    "small": 18,
    "md": 24,
    "medium": 24,
    "lg": 32,
    "large": 32,
    "xl": 40,
    "xlarge": 40,
    "extra-large": 40,
}
_FORMULA_DISPLAY_TOKENS = {
    "block": True,
    "display": True,
    "true": True,
    "inline": False,
    "none": False,
    "false": False,
}


def encode(spec: Mapping[str, Any], *, shorten_ids: bool = True) -> list[dict[str, Any]]:
    source = rewrite_element_ids(spec, shorten=shorten_ids)
    manifest = load_manifest()
    components: list[dict[str, Any]] = []
    for element_id, raw in source["elements"].items():
        component: dict[str, Any] = {
            "id": element_id,
            "component": raw["type"],
        }
        props = raw.get("props") if isinstance(raw.get("props"), dict) else {}
        # Express producers may use ``None`` to mean an omitted optional prop
        # (for example ``Table.statePath``).  The pinned wire schema does not
        # accept JSON null for those fields, so omit null-valued keys at the
        # transport boundary while preserving nulls inside arrays/strings.
        component.update(
            _lower_bindings(
                _normalize_wire_props(
                    _drop_none_props(deepcopy(props)),
                    component_type=raw["type"],
                )
            )
        )
        children = raw.get("children") if isinstance(raw.get("children"), list) else []
        repeat = raw.get("repeat") if isinstance(raw.get("repeat"), Mapping) else None
        child_list = _lower_child_list(children, repeat)
        if child_list:
            component["children"] = child_list
        for key in ("visible", "on", "watch"):
            value = raw.get(key)
            if value not in (None, {}, []):
                component[key] = _lower_bindings(deepcopy(value), action_map=key in {"on", "watch"})
        components.append(component)

    if source["root"] != "root":
        raise ValueError("Compiled A2UI root must be deterministically rewritten to id 'root'")
    surface_id = DEFAULT_SURFACE_ID
    messages: list[dict[str, Any]] = [
        {
            "version": VERSION,
            "createSurface": {
                "surfaceId": surface_id,
                "catalogId": manifest["catalogId"],
                "sendDataModel": False,
            },
        },
        {
            "version": VERSION,
            "updateComponents": {
                "surfaceId": surface_id,
                "components": components,
            },
        },
    ]
    if source.get("state"):
        messages.append(
            {
                "version": VERSION,
                "updateDataModel": {
                    "surfaceId": surface_id,
                    "path": "/",
                    "value": _lower_bindings(deepcopy(source["state"])),
                },
            }
        )
    return messages


def _drop_none_props(value: Any) -> Any:
    """Remove null-valued object properties before wire-schema validation."""

    if isinstance(value, Mapping):
        return {
            key: _drop_none_props(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_drop_none_props(item) for item in value]
    return value


def _normalize_wire_props(value: Any, *, component_type: str) -> Any:
    """Normalize Express-only shorthand before strict wire-schema validation.

    The Android renderer uses the same compact icon-size scale for dimensions
    (xs=14, sm=18, md=24, lg=32, xl=40 dp), while the pinned wire contract
    represents Image width/height as numeric dynamic values.
    """

    if not isinstance(value, dict):
        return value
    if component_type == "Image":
        for key in ("width", "height"):
            raw = value.get(key)
            if isinstance(raw, str):
                token = raw.strip().lower()
                if token in _IMAGE_DIMENSION_TOKENS:
                    value[key] = _IMAGE_DIMENSION_TOKENS[token]
    elif component_type == "Formula":
        raw = value.get("display")
        if isinstance(raw, str):
            token = raw.strip().lower()
            if token in _FORMULA_DISPLAY_TOKENS:
                value["display"] = _FORMULA_DISPLAY_TOKENS[token]
    elif component_type == "List":
        # Some model outputs use items as a numeric count. The pinned wire
        # schema requires dynamicArray here, while children already carries
        # the rendered entries, so omit the invalid count.
        items = value.get("items")
        if isinstance(items, (int, float)) and not isinstance(items, bool):
            value.pop("items", None)
    return value


def decode(payload: Any) -> dict[str, Any]:
    messages = _messages(payload)
    elements: dict[str, dict[str, Any]] = {}
    state: dict[str, Any] = {}
    surface_id: str | None = None
    deleted = False

    for message in messages:
        if message.get("version") != VERSION:
            raise ValueError(f"A2UI wire version must be {VERSION}")

        if "createSurface" in message:
            create = _mapping(message["createSurface"], "createSurface")
            surface_id = _surface_id(create, surface_id)
            _validate_catalog(create.get("catalogId"))
            if "rootId" in create:
                raise ValueError("Non-standard createSurface.rootId is not allowed; the root component id is 'root'")
            if any(key in create for key in ("rootId", "components", "dataModel")):
                raise ValueError("A2UI v0.9 createSurface may not contain components, dataModel, or rootId")
            deleted = False
            continue

        if "updateComponents" in message:
            update = _mapping(message["updateComponents"], "updateComponents")
            surface_id = _surface_id(update, surface_id)
            for element_id, element in _decode_components(update.get("components")):
                if element_id in elements:
                    raise ValueError(f"Duplicate A2UI component id {element_id!r} across updates")
                elements[element_id] = element
            continue

        if "updateDataModel" in message:
            update = _mapping(message["updateDataModel"], "updateDataModel")
            surface_id = _surface_id(update, surface_id)
            _set_data_path(state, update.get("path"), deepcopy(update["value"]) if "value" in update else _MISSING)
            continue

        if "deleteSurface" in message:
            delete = _mapping(message["deleteSurface"], "deleteSurface")
            surface_id = _surface_id(delete, surface_id)
            deleted = True
            elements.clear()
            state.clear()
            continue

        raise ValueError("Unsupported A2UI v0.9 wire message")

    if deleted:
        raise ValueError("A2UI surface was deleted before conversion")
    if not elements:
        raise ValueError("A2UI wire payload did not produce any components")
    resolved_root = "root"
    if resolved_root not in elements:
        raise ValueError("A2UI payload must contain a component with id 'root'")
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

        metadata = {"id", "component", "children", "visible", "on", "watch"}
        descriptor = load_catalog().get("components", {}).get(component_type, {})
        allowed_props = set(descriptor.get("allowedProperties", ()))
        if not allowed_props:
            allowed_props = set(descriptor.get("consumedProps", ())) | set(descriptor.get("positional", ()))
        unknown = [key for key in raw if key not in metadata and key not in allowed_props]
        if unknown:
            raise ValueError(
                f"A2UI component {element_id!r} has unsupported properties: {', '.join(map(str, unknown))}"
            )
        props = {key: deepcopy(item) for key, item in raw.items() if key not in metadata}
        children, repeat = _decode_child_list(raw.get("children", []), element_id)
        element: dict[str, Any] = {
            "type": component_type,
            "props": _raise_bindings(props),
            "children": deepcopy(children),
        }
        if repeat is not None:
            element["repeat"] = repeat
        for key in ("on", "watch"):
            if key in raw:
                item = raw[key]
                if not isinstance(item, Mapping):
                    raise ValueError(f"A2UI component {element_id!r} {key} must be an object")
                element[key] = _raise_bindings(deepcopy(dict(item)), action_map=key in {"on", "watch"})
        if "visible" in raw:
            element["visible"] = _raise_bindings(deepcopy(raw["visible"]))
        yield element_id, element


def _set_data_path(state: dict[str, Any], raw_path: Any, value: Any) -> None:
    if raw_path in (None, "", "/"):
        state.clear()
        if value is not _MISSING:
            if not isinstance(value, Mapping):
                raise ValueError("Root A2UI data-model update must be an object")
            state.update(_raise_bindings(deepcopy(dict(value))))
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
    if value is _MISSING:
        current.pop(parts[-1], None)
    else:
        current[parts[-1]] = _raise_bindings(value)


def _json_pointer_path(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A2UI dynamic path must be a non-empty string")
    path = value.strip()
    if path == "$":
        return "/"
    if path.startswith("$/"):
        return "/" + path[2:]
    if path.startswith("$state."):
        return "/" + path[7:].replace(".", "/")
    if path.startswith("/"):
        return path
    if path.startswith("$"):
        return "/" + path[1:].lstrip("/").replace(".", "/")
    return path


def _lower_child_list(children: list[Any], repeat: Mapping[str, Any] | None) -> Any:
    if repeat:
        template = (
            repeat.get("template")
            or repeat.get("itemTemplate")
            or repeat.get("child")
            or (children[0] if children else None)
        )
        path = repeat.get("statePath") or repeat.get("path")
        if not isinstance(template, str) or not template.strip():
            raise ValueError("A2UI repeat requires template/itemTemplate/child")
        child_list = {"componentId": template, "path": _json_pointer_path(path)}
        if isinstance(repeat.get("key"), str) and repeat["key"].strip():
            child_list["key"] = repeat["key"]
        return child_list
    return _lower_bindings(deepcopy(children)) if children else []


def _decode_child_list(value: Any, element_id: str) -> tuple[list[str], dict[str, Any] | None]:
    if value in (None, []):
        return [], None
    if isinstance(value, list):
        if any(not isinstance(item, str) or not item for item in value):
            raise ValueError(f"A2UI component {element_id!r} children must be non-empty strings")
        return list(value), None
    if isinstance(value, Mapping):
        allowed = {"componentId", "path", "key"}
        if not set(value).issubset(allowed) or not isinstance(value.get("componentId"), str) or not isinstance(value.get("path"), str):
            raise ValueError(f"A2UI component {element_id!r} dynamic children must contain componentId and path")
        repeat = {"statePath": _json_pointer_path(value["path"]), "template": str(value["componentId"])}
        if isinstance(value.get("key"), str) and value["key"].strip():
            repeat["key"] = value["key"]
        return [str(value["componentId"])], repeat
    raise ValueError(f"A2UI component {element_id!r} children must be a ChildList")


def _lower_bindings(value: Any, *, action_map: bool = False) -> Any:
    if isinstance(value, str) and _is_binding_string(value):
        return {"path": _json_pointer_path(value)}
    if isinstance(value, list):
        return [_lower_bindings(item, action_map=action_map) for item in value]
    if isinstance(value, Mapping):
        if action_map and "action" in value and isinstance(value.get("action"), str):
            return _lower_action(value)
        if isinstance(value.get("call"), str) and set(value).issubset({"call", "args", "kwargs"}):
            return _lower_legacy_call(value)
        return {str(key): _lower_bindings(item, action_map=action_map) for key, item in value.items()}
    return value


def _is_binding_string(value: str) -> bool:
    """Recognize only Express data-path forms; dollar-prefixed literals stay literal."""

    return value == "$" or value.startswith("$/") or value.startswith("$state.")


def _lower_legacy_call(value: Mapping[str, Any]) -> dict[str, Any]:
    """Lower Express call shorthand to the pinned wire functionCall shape."""

    args: dict[str, Any] = {}
    positional = value.get("args")
    if isinstance(positional, Mapping):
        args.update({str(key): _lower_bindings(item) for key, item in positional.items()})
    elif isinstance(positional, list):
        args.update({f"arg{index}": _lower_bindings(item) for index, item in enumerate(positional)})
    kwargs = value.get("kwargs")
    if isinstance(kwargs, Mapping):
        args.update({str(key): _lower_bindings(item) for key, item in kwargs.items()})
    function_call: dict[str, Any] = {"call": str(value["call"])}
    if args:
        function_call["args"] = args
    return function_call


def _lower_action(value: Mapping[str, Any]) -> dict[str, Any]:
    action = str(value.get("action"))
    params = value.get("params") if isinstance(value.get("params"), Mapping) else {}
    if action == "emitEvent":
        name = params.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("emitEvent requires name")
        raw_context = params.get("context") if isinstance(params.get("context"), Mapping) else {}
        context = {str(key): _lower_bindings(item) for key, item in raw_context.items()}
        event: dict[str, Any] = {"name": name}
        if context:
            event["context"] = context
        for key in ("wantResponse", "responsePath"):
            if key in params:
                event[key] = _lower_bindings(params[key])
        return {"event": event}
    args = {str(key): _lower_bindings(item) for key, item in params.items()}
    function_call: dict[str, Any] = {"call": action}
    if args:
        function_call["args"] = args
    return {"functionCall": function_call}


def _raise_bindings(value: Any, *, action_map: bool = False) -> Any:
    if isinstance(value, Mapping):
        if set(value) == {"path"} and isinstance(value.get("path"), str):
            return {"path": _json_pointer_path(value["path"])}
        if action_map and ("event" in value or "functionCall" in value):
            return _raise_action(value)
        return {str(key): _raise_bindings(item, action_map=action_map) for key, item in value.items()}
    if isinstance(value, list):
        return [_raise_bindings(item, action_map=action_map) for item in value]
    return value


def _raise_action(value: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value.get("event"), Mapping):
        event = value["event"]
        params: dict[str, Any] = {"name": event.get("name")}
        raw_context = event.get("context") if isinstance(event.get("context"), Mapping) else None
        if raw_context is not None:
            params["context"] = {str(key): _raise_bindings(item) for key, item in raw_context.items()}
        for key in ("wantResponse", "responsePath"):
            if key in event:
                params[key] = _raise_bindings(event[key])
        return {"action": "emitEvent", "params": params}
    function_call = value.get("functionCall")
    if not isinstance(function_call, Mapping) or not isinstance(function_call.get("call"), str):
        raise ValueError("A2UI action must be an event or functionCall")
    params = function_call.get("args") if isinstance(function_call.get("args"), Mapping) else {}
    return {"action": str(function_call["call"]), "params": {str(key): _raise_bindings(item) for key, item in params.items()}}
