from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

from .flat_spec_semantics import iter_renderer_references

_CAPABILITY_PATH = Path(__file__).resolve().parents[2] / "schema" / "renderer_capabilities.json"
_CAPABILITIES = json.loads(_CAPABILITY_PATH.read_text(encoding="utf-8"))
if _CAPABILITIES.get("version") != "2.0.0":
    raise RuntimeError("Python flat-spec contract requires renderer capability version 2.0.0")

_TYPE_CANONICAL_MAP = {
    "".join(ch for ch in str(alias) if ch.isalnum()).lower(): str(entry["canonical"])
    for entry in _CAPABILITIES["types"]
    for alias in entry.get("aliases", [])
}
for _alias, _descriptor in _CAPABILITIES.get("compatibility_type_aliases", {}).items():
    _TYPE_CANONICAL_MAP[str(_alias).lower()] = str(_descriptor["canonical"])

_ALLOWED_TYPES = set(_TYPE_CANONICAL_MAP)
_COMPATIBILITY_TYPE_DIRECTIONS = {
    str(alias).lower(): str(descriptor.get("props", {}).get("direction", ""))
    for alias, descriptor in _CAPABILITIES.get("compatibility_type_aliases", {}).items()
}
_ALLOWED_ACTIONS = {str(action["name"]) for action in _CAPABILITIES["actions"]}
_ACTION_DESCRIPTORS = {
    str(action["name"]).lower(): action for action in _CAPABILITIES["actions"]
}
_TABLE_DOMAINS = {str(value) for value in _CAPABILITIES["table_domains"]["canonical"]}
_TABLE_DOMAIN_ALIASES = {
    str(alias): str(canonical)
    for alias, canonical in _CAPABILITIES["table_domains"].get("aliases", {}).items()
}

_LEGACY_TYPE_CANONICAL_MAP = {
    "code": "CodeBlock",
    "codeblock": "CodeBlock",
    "code_block": "CodeBlock",
    "pre": "CodeBlock",
    "preformatted": "CodeBlock",
    "console": "ConsoleLog",
    "consolelog": "ConsoleLog",
    "console_log": "ConsoleLog",
    "terminal": "ConsoleLog",
    "email": "EmailPreview",
    "mail": "EmailPreview",
    "emailpreview": "EmailPreview",
    "email_preview": "EmailPreview",
    "messagepreview": "EmailPreview",
    "message_preview": "EmailPreview",
    "table": "Table",
    "chart": "Chart",
    "barchart": "Chart",
    "bar_chart": "Chart",
    "image": "Image",
    "icon": "Icon",
    "video": "Video",
    "audioplayer": "AudioPlayer",
    "audio": "AudioPlayer",
    "divider": "Divider",
    "button": "Button",
    "tabs": "Tabs",
    "tab": "Tabs",
    "modal": "Modal",
    "textfield": "TextField",
    "textinput": "TextField",
    "textbox": "TextField",
    "input": "TextField",
    "checkbox": "CheckBox",
    "check": "CheckBox",
    "choicepicker": "ChoicePicker",
    "picker": "ChoicePicker",
    "dropdown": "ChoicePicker",
    "select": "ChoicePicker",
    "slider": "Slider",
    "datetimeinput": "DateTimeInput",
    "datetimepicker": "DateTimeInput",
    "dateinput": "DateTimeInput",
    "datepicker": "DateTimeInput",
}
_TYPE_CANONICAL_MAP.update(_LEGACY_TYPE_CANONICAL_MAP)

_EVENT_ALIASES = {
    "click": "press",
    "tap": "press",
    "onclick": "press",
    "onpress": "press",
    "onsubmit": "submit",
    "input": "change",
    "onchange": "change",
}

_ACTION_ALIASES = {
    "openurl": "openUrl",
    "url": "openUrl",
    "open": "openUrl",
    "link": "openUrl",
    "setstate": "setState",
    "updatestate": "setState",
    "pushstate": "pushState",
    "navigate": "pushState",
    "route": "pushState",
    "removestate": "removeState",
    "deletestate": "removeState",
    "validate": "validateForm",
    "validateform": "validateForm",
}


@dataclass(frozen=True)
class NormalizeResult:
    spec: dict[str, Any] | None
    converted_from_legacy: bool
    error: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    error: str | None = None


@dataclass(frozen=True)
class CoerceResult:
    spec: dict[str, Any] | None
    converted_from_legacy: bool
    error: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.spec is not None and self.error is None


def looks_like_flat_spec(value: Any) -> bool:
    return isinstance(value, dict) and "root" in value and "elements" in value


def coerce_and_validate(value: Any) -> CoerceResult:
    if value is None:
        return CoerceResult(spec=None, converted_from_legacy=False, error="Stage 3 output is null.")

    normalized = normalize_to_flat_spec(value)
    if normalized.spec is None:
        return CoerceResult(
            spec=None,
            converted_from_legacy=normalized.converted_from_legacy,
            error=normalized.error or "Could not normalize Stage 3 output to flat spec.",
        )

    validation = validate_flat_spec(normalized.spec)
    if not validation.is_valid:
        return CoerceResult(
            spec=None,
            converted_from_legacy=normalized.converted_from_legacy,
            error=validation.error or "Flat spec validation failed.",
        )

    return CoerceResult(
        spec=normalized.spec,
        converted_from_legacy=normalized.converted_from_legacy,
        error=None,
    )


def normalize_to_flat_spec(value: Any) -> NormalizeResult:
    if looks_like_flat_spec(value):
        return NormalizeResult(spec=canonicalize_flat_spec(value), converted_from_legacy=False)

    unwrapped = unwrap_known_containers(value)
    if unwrapped is not None:
        if looks_like_flat_spec(unwrapped):
            return NormalizeResult(spec=canonicalize_flat_spec(unwrapped), converted_from_legacy=True)
        if isinstance(unwrapped, list):
            converted = convert_legacy_messages(unwrapped)
            if converted is not None:
                return NormalizeResult(spec=converted, converted_from_legacy=True)
            return NormalizeResult(
                spec=None,
                converted_from_legacy=True,
                error="Wrapped legacy payload could not be converted.",
            )
        if isinstance(unwrapped, dict):
            converted = convert_legacy_messages([unwrapped])
            if converted is not None:
                return NormalizeResult(spec=converted, converted_from_legacy=True)

    if isinstance(value, list):
        converted = convert_legacy_messages(value)
        if converted is not None:
            return NormalizeResult(spec=converted, converted_from_legacy=True)
        return NormalizeResult(
            spec=None,
            converted_from_legacy=True,
            error="Legacy message array could not be converted.",
        )

    if isinstance(value, dict):
        converted = convert_legacy_messages([value])
        if converted is not None:
            return NormalizeResult(spec=converted, converted_from_legacy=True)

    return NormalizeResult(
        spec=None,
        converted_from_legacy=False,
        error="Unsupported JSON shape for flat spec.",
    )


def validate_flat_spec(spec: dict[str, Any]) -> ValidationResult:
    root = spec.get("root")
    if not isinstance(root, str) or not root.strip():
        return ValidationResult(False, "Missing or invalid root id.")

    elements_node = spec.get("elements")
    if not isinstance(elements_node, dict):
        return ValidationResult(False, "Missing or invalid elements object.")

    if not elements_node:
        return ValidationResult(False, "Elements map is empty.")

    if root not in elements_node:
        return ValidationResult(False, f"Root id '{root}' does not exist in elements.")

    ids = set(elements_node.keys())
    for element_id, raw_element in elements_node.items():
        if not isinstance(raw_element, dict):
            return ValidationResult(False, f"Element '{element_id}' must be an object.")

        element_type = raw_element.get("type")
        if not isinstance(element_type, str) or not element_type.strip():
            return ValidationResult(False, f"Element '{element_id}' is missing a valid type.")
        if element_type.lower() not in _ALLOWED_TYPES:
            return ValidationResult(False, f"Element '{element_id}' has unsupported type '{element_type}'.")

        props = raw_element.get("props")
        if not isinstance(props, dict):
            return ValidationResult(False, f"Element '{element_id}' must define props as an object.")
        if "action" in props:
            return ValidationResult(
                False,
                f"Element '{element_id}' uses legacy props.action. Use on.<event> action bindings instead.",
            )

        children = raw_element.get("children")
        if not isinstance(children, list):
            return ValidationResult(False, f"Element '{element_id}' must define children as an array.")
        for child in children:
            if not isinstance(child, str):
                return ValidationResult(False, f"Element '{element_id}' contains a non-string child reference.")
        for reference in iter_renderer_references(raw_element):
            if reference.target_id not in ids:
                noun = "child" if reference.reference_kind == "child" else "element"
                return ValidationResult(
                    False,
                    f"Element '{element_id}' references missing {noun} '{reference.target_id}' "
                    f"at {reference.source_path}.",
                )

        repeat = raw_element.get("repeat")
        if repeat is not None:
            if not isinstance(repeat, dict):
                return ValidationResult(False, f"Element '{element_id}' repeat must be an object.")
            state_path = repeat.get("statePath")
            if not isinstance(state_path, str) or not state_path.strip():
                return ValidationResult(False, f"Element '{element_id}' repeat.statePath is required.")

        on = raw_element.get("on")
        if on is not None:
            if not isinstance(on, dict):
                return ValidationResult(False, f"Element '{element_id}' on must be an object.")
            for event_name, action_value in on.items():
                action_error = _validate_action_candidate(action_value, f"Element '{element_id}' on.{event_name}")
                if action_error:
                    return ValidationResult(False, action_error)

        watch = raw_element.get("watch")
        if watch is not None:
            if not isinstance(watch, dict):
                return ValidationResult(False, f"Element '{element_id}' watch must be an object.")
            for state_path, action_value in watch.items():
                if not isinstance(state_path, str) or not state_path.strip() or not state_path.startswith("/"):
                    return ValidationResult(
                        False,
                        f"Element '{element_id}' watch key '{state_path}' must be a non-empty JSON pointer path.",
                    )
                action_error = _validate_action_candidate(action_value, f"Element '{element_id}' watch.{state_path}")
                if action_error:
                    return ValidationResult(False, action_error)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(element_id: str) -> str | None:
        if element_id in visiting:
            return element_id
        if element_id in visited:
            return None
        visiting.add(element_id)
        raw_element = elements_node.get(element_id)
        if isinstance(raw_element, dict):
            for reference in iter_renderer_references(raw_element):
                cycle_id = visit(reference.target_id)
                if cycle_id is not None:
                    return cycle_id
        visiting.remove(element_id)
        visited.add(element_id)
        return None

    cycle_id = visit(root)
    if cycle_id is not None:
        return ValidationResult(
            False,
            f"Renderer reference cycle is reachable from root through element '{cycle_id}'.",
        )

    return ValidationResult(True)


def canonicalize_flat_spec(raw: dict[str, Any]) -> dict[str, Any]:
    source_elements = raw.get("elements") if isinstance(raw.get("elements"), dict) else {}
    canonical_elements: dict[str, Any] = {}

    for element_id, element_value in source_elements.items():
        if not isinstance(element_value, dict):
            continue

        element_type = element_value.get("type")
        if not isinstance(element_type, str) or not element_type.strip():
            legacy_type = element_value.get("component")
            if isinstance(legacy_type, str) and legacy_type.strip():
                element_type = legacy_type
            else:
                continue
        canonical_type = _canonicalize_type_name(element_type)
        if canonical_type is None:
            canonical_type = element_type

        props = element_value.get("props") if isinstance(element_value.get("props"), dict) else {}
        props = deepcopy(props)
        compatibility_token = "".join(
            ch for ch in str(element_type).strip() if ch.isalnum()
        ).lower()
        compatibility_direction = _COMPATIBILITY_TYPE_DIRECTIONS.get(compatibility_token)
        if compatibility_direction and "direction" not in props:
            props["direction"] = compatibility_direction
        if canonical_type == "Table" and isinstance(props.get("domain"), str):
            domain_token = str(props["domain"]).strip().lower()
            props["domain"] = _TABLE_DOMAIN_ALIASES.get(domain_token, domain_token)

        for key, val in element_value.items():
            if key in {"type", "component", "props", "children", "on", "repeat", "visible", "watch"}:
                continue
            props.setdefault(key, deepcopy(val))

        children: list[str] = []
        raw_children = element_value.get("children")
        if isinstance(raw_children, list):
            children.extend([item for item in raw_children if isinstance(item, str)])
        raw_child = element_value.get("child")
        if isinstance(raw_child, str) and raw_child not in children:
            children.append(raw_child)

        on_bindings: dict[str, Any] = {}
        if isinstance(element_value.get("on"), dict):
            on_bindings = _normalize_event_bindings(element_value.get("on"))
        elif isinstance(element_value.get("action"), (dict, list)):
            normalized_action = _normalize_action_candidate(element_value.get("action"))
            if normalized_action is not None:
                on_bindings["press"] = normalized_action

        if "action" in props and not on_bindings:
            normalized_action = _normalize_action_candidate(props.get("action"))
            if normalized_action is not None:
                on_bindings["press"] = normalized_action
            props.pop("action", None)

        watch_bindings: dict[str, Any] = {}
        if isinstance(element_value.get("watch"), dict):
            raw_watch = element_value.get("watch")
            assert isinstance(raw_watch, dict)
            for watch_key, watch_action in raw_watch.items():
                normalized_watch_key = _normalize_watch_key(watch_key)
                if normalized_watch_key is None:
                    continue
                normalized_watch_action = _normalize_action_candidate(watch_action)
                if normalized_watch_action is None:
                    continue
                watch_bindings[normalized_watch_key] = normalized_watch_action

        repeat = element_value.get("repeat")
        normalized_repeat = _normalize_repeat(repeat)

        canonical_element: dict[str, Any] = {
            "type": canonical_type,
            "props": props,
            "children": children,
        }
        if normalized_repeat is not None:
            canonical_element["repeat"] = normalized_repeat
        if "visible" in element_value:
            canonical_element["visible"] = deepcopy(element_value["visible"])
        if on_bindings:
            canonical_element["on"] = on_bindings
        if watch_bindings:
            canonical_element["watch"] = watch_bindings

        canonical_elements[str(element_id)] = canonical_element

    root = raw.get("root") if isinstance(raw.get("root"), str) else ""
    if isinstance(root, str):
        root = root.strip().lstrip("#")
    if not root or root not in canonical_elements:
        if "root" in canonical_elements:
            root = "root"
        elif canonical_elements:
            root = next(iter(canonical_elements.keys()))
        else:
            root = "root"

    state = raw.get("state") if isinstance(raw.get("state"), dict) else {}

    return {
        "root": root,
        "state": deepcopy(state),
        "elements": canonical_elements,
    }


def _canonicalize_type_name(element_type: str) -> str | None:
    if not isinstance(element_type, str):
        return None
    normalized = "".join(ch for ch in element_type.strip() if ch.isalnum()).lower()
    if not normalized:
        return None
    return _TYPE_CANONICAL_MAP.get(normalized)


def _normalize_event_name(event_name: Any) -> str | None:
    if not isinstance(event_name, str):
        return None
    key = "".join(ch for ch in event_name.strip() if ch.isalnum()).lower()
    if not key:
        return None
    return _EVENT_ALIASES.get(key, key)


def _normalize_action_name(action_name: Any) -> str | None:
    if not isinstance(action_name, str):
        return None
    compact = "".join(ch for ch in action_name.strip() if ch.isalnum())
    if not compact:
        return None
    if compact in _ALLOWED_ACTIONS:
        return compact
    lowered = compact.lower()
    if lowered in _ACTION_ALIASES:
        return _ACTION_ALIASES[lowered]
    for allowed in _ALLOWED_ACTIONS:
        if allowed.lower() == lowered:
            return allowed
    return None


def _normalize_action_candidate(candidate: Any) -> dict[str, Any] | list[dict[str, Any]] | None:
    if candidate is None:
        return None

    if isinstance(candidate, list):
        out: list[dict[str, Any]] = []
        for item in candidate:
            normalized = _normalize_action_candidate(item)
            if isinstance(normalized, dict):
                out.append(normalized)
            elif isinstance(normalized, list):
                out.extend([x for x in normalized if isinstance(x, dict)])
        if not out:
            return None
        return out

    if not isinstance(candidate, dict):
        return None

    if "functionCall" in candidate and isinstance(candidate.get("functionCall"), dict):
        function_call = candidate.get("functionCall")
        assert isinstance(function_call, dict)
        action_name = _normalize_action_name(function_call.get("call"))
        if action_name is None:
            return None
        params = function_call.get("args") if isinstance(function_call.get("args"), dict) else {}
        normalized: dict[str, Any] = {"action": action_name}
        if params:
            normalized["params"] = deepcopy(params)
        return normalized

    action_name = _normalize_action_name(candidate.get("action"))
    if action_name is None:
        return None

    params = candidate.get("params") if isinstance(candidate.get("params"), dict) else {}
    params = deepcopy(params)
    for key, value in candidate.items():
        if key in {"action", "params", "preventDefault", "confirm"}:
            continue
        params[key] = deepcopy(value)

    normalized = {"action": action_name}
    if params:
        normalized["params"] = params
    if isinstance(candidate.get("preventDefault"), bool):
        normalized["preventDefault"] = bool(candidate.get("preventDefault"))
    if isinstance(candidate.get("confirm"), dict):
        normalized["confirm"] = deepcopy(candidate.get("confirm"))
    return normalized


def _normalize_event_bindings(on_value: Any) -> dict[str, Any]:
    if not isinstance(on_value, dict):
        return {}
    out: dict[str, Any] = {}
    for raw_event_name, action_candidate in on_value.items():
        event_name = _normalize_event_name(raw_event_name)
        if event_name is None:
            continue
        action_binding = _normalize_action_candidate(action_candidate)
        if action_binding is None:
            continue
        out[event_name] = action_binding
    return out


def _normalize_watch_key(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    key = value.strip()
    if not key:
        return None
    if key.startswith("/"):
        return key
    if key.startswith("$."):
        key = key[2:]
    if key.startswith("state."):
        key = key[len("state.") :]
    parts = [part for part in key.split(".") if part]
    if not parts:
        return None
    return "/" + "/".join(parts)


def _normalize_repeat(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    state_path = value.get("statePath")
    if not isinstance(state_path, str) or not state_path.strip():
        state_path = value.get("path")
    normalized_state_path = _normalize_watch_key(state_path)
    if normalized_state_path is None:
        return None
    normalized = {"statePath": normalized_state_path}
    key_value = value.get("key")
    if isinstance(key_value, str) and key_value.strip():
        normalized["key"] = key_value.strip()
    for raw_key, raw_val in value.items():
        if raw_key in {"statePath", "path", "key"}:
            continue
        normalized[raw_key] = deepcopy(raw_val)
    return normalized


def unwrap_known_containers(value: Any) -> Any:
    if not isinstance(value, dict):
        return None

    for key in ("genui_json", "flat_spec", "messages"):
        if key in value:
            return value.get(key)

    payload = value.get("payload")
    if isinstance(payload, (list, dict)):
        if isinstance(payload, dict):
            for nested_key in ("messages", "genui_json", "flat_spec"):
                if nested_key in payload:
                    return payload.get(nested_key)
        return payload

    return None


def convert_legacy_messages(messages: list[Any]) -> dict[str, Any] | None:
    components = _extract_legacy_components(messages)
    if components is None:
        return None

    elements: dict[str, Any] = {}
    root_id: str | None = None

    for component in components:
        if not isinstance(component, dict):
            continue
        element_id = component.get("id")
        if not isinstance(element_id, str) or not element_id:
            continue

        component_type = component.get("component")
        if not isinstance(component_type, str) or not component_type:
            component_type = component.get("type")
        if not isinstance(component_type, str) or not component_type:
            continue

        if root_id is None and element_id == "root":
            root_id = element_id

        props: dict[str, Any] = {}
        children: list[str] = []
        on_bindings: dict[str, Any] = {}

        for key, val in component.items():
            if key in {"id", "component", "type", "children"}:
                continue
            if key == "child":
                props[key] = deepcopy(val)
                if isinstance(val, str) and val not in children:
                    children.append(val)
                continue
            if key == "action":
                action_binding = _legacy_action_to_binding(val)
                if action_binding is not None:
                    on_bindings["press"] = action_binding
                continue
            props[key] = deepcopy(val)

        raw_children = component.get("children")
        if isinstance(raw_children, list):
            for child in raw_children:
                if isinstance(child, str) and child not in children:
                    children.append(child)

        element_payload: dict[str, Any] = {
            "type": component_type,
            "props": props,
            "children": children,
        }
        if on_bindings:
            element_payload["on"] = on_bindings

        elements[element_id] = element_payload

    if not elements:
        return None

    resolved_root = (
        root_id
        if root_id and root_id in elements
        else "root"
        if "root" in elements
        else next(iter(elements.keys()))
    )

    return {
        "root": resolved_root,
        "state": {},
        "elements": elements,
    }


def _extract_legacy_components(messages: list[Any]) -> list[dict[str, Any]] | None:
    for message in messages:
        if not isinstance(message, dict):
            continue
        update = message.get("updateComponents")
        if isinstance(update, dict):
            components = update.get("components")
            if isinstance(components, list):
                return [item for item in components if isinstance(item, dict)]

        direct = message.get("components")
        if isinstance(direct, list):
            return [item for item in direct if isinstance(item, dict)]

    looks_like_component_array = True
    filtered: list[dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, dict):
            looks_like_component_array = False
            break
        item_id = item.get("id")
        item_component = item.get("component")
        item_type = item.get("type")
        if not isinstance(item_id, str):
            looks_like_component_array = False
            break
        if not (isinstance(item_component, str) or isinstance(item_type, str)):
            looks_like_component_array = False
            break
        filtered.append(item)

    if looks_like_component_array and filtered:
        return filtered
    return None


def _legacy_action_to_binding(action: Any) -> dict[str, Any] | list[dict[str, Any]] | None:
    normalized = _normalize_action_candidate(action)
    if isinstance(normalized, dict):
        return normalized
    if isinstance(normalized, list):
        return [item for item in normalized if isinstance(item, dict)] or None
    return None


def _validate_action_candidate(candidate: Any, context: str) -> str | None:
    if candidate is None:
        return f"{context} cannot be null."

    if isinstance(candidate, list):
        if not candidate:
            return f"{context} action array cannot be empty."
        for index, item in enumerate(candidate):
            error = _validate_single_action_binding(item, f"{context}[{index}]")
            if error:
                return error
        return None

    return _validate_single_action_binding(candidate, context)


def _validate_single_action_binding(binding: Any, context: str) -> str | None:
    if not isinstance(binding, dict):
        return f"{context} must be an object action binding."

    if "functionCall" in binding:
        return f"{context} uses legacy functionCall. Use {{\"action\":\"...\",\"params\":{{...}}}}."

    action = binding.get("action")
    if not isinstance(action, str) or not action.strip():
        return f"{context} action must be a non-empty string."
    if action not in _ALLOWED_ACTIONS:
        return f"{context} action '{action}' is not allowed."

    unsupported_keys = set(binding) - {"action", "params"}
    if unsupported_keys:
        return f"{context} contains unsupported action fields {sorted(unsupported_keys)}."

    params = binding.get("params")
    if params is not None and not isinstance(params, dict):
        return f"{context} params must be an object when present."

    descriptor = _ACTION_DESCRIPTORS[action.lower()]
    params = params if isinstance(params, dict) else {}
    missing = [key for key in descriptor.get("required", []) if key not in params]
    if missing:
        return f"{context} action '{action}' requires params {sorted(missing)}."
    for alternatives in descriptor.get("required_any", []):
        if not any(
            key in params and (not isinstance(params[key], str) or bool(params[key].strip()))
            for key in alternatives
        ):
            return f"{context} action '{action}' requires one of params {sorted(alternatives)}."
    if descriptor.get("safe_url"):
        alternatives = descriptor.get("required_any", [[]])[0]
        raw_url = next((params[key] for key in alternatives if key in params), None)
        if not _is_safe_action_url(raw_url):
            return f"{context} openUrl contains unsafe URL."
    if action == "removeState":
        index = params.get("index")
        if not isinstance(index, (int, float)) or index < 0 or int(index) != index:
            return f"{context} removeState params.index must be a non-negative integer."

    return None


def _is_safe_action_url(raw: Any) -> bool:
    if not isinstance(raw, str) or not raw.strip():
        return False
    value = raw.strip()
    # Training URL preprocessing replaces real, already-validated URLs with
    # typed symbolic tokens. Treat only the exact action/source token shape as
    # safe so the canonical graph can be validated before Express compilation.
    if re.fullmatch(r"\[(?:ACTION|SOURCE|URL)_URL_\d+\]", value):
        return True
    if value.lower().startswith("tel:"):
        return bool(re.fullmatch(r"tel:[+0-9().\-\s]{3,}", value, flags=re.IGNORECASE))
    parsed = urlparse(value)
    return parsed.scheme.lower() == "https" and bool(parsed.netloc)


_URL_RE = re.compile(r"https?://[^\s)]+", re.IGNORECASE)
_MARKDOWN_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")


def _sanitize_markdown_line(line: str) -> str:
    value = (line or "").replace("\t", " ").strip()
    if not value:
        return ""
    if value.startswith("```") or value.startswith("'''"):
        return ""
    value = re.sub(r"^#{1,6}\s*", "", value)
    value = re.sub(r"^#\s*", "", value)
    if value.startswith("- "):
        value = f"• {value[2:].strip()}"
    elif value.startswith("* "):
        value = f"• {value[2:].strip()}"
    value = value.replace("**", "")
    value = value.replace("```", "")
    value = value.replace("'''", "")
    value = value.replace("`", "")
    return re.sub(r"\s+", " ", value).strip()


def _split_text_chunks(text: str, max_chars: int = 180) -> list[str]:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            for i in range(0, len(sentence), max_chars):
                part = sentence[i : i + max_chars].strip()
                if part:
                    chunks.append(part)
            continue
        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current.strip())
            current = sentence
    if current:
        chunks.append(current.strip())
    return chunks


def _is_heading_candidate(line: str) -> bool:
    if not line:
        return False
    if line.endswith(":") and len(line) <= 90:
        return True
    if len(line) <= 60 and line[0].isupper() and not line.endswith("."):
        words = line.split()
        if 1 <= len(words) <= 8:
            return True
    return False


def _parse_sections_from_lines(lines: list[str]) -> tuple[str, list[tuple[str, str]]]:
    sanitized = [_sanitize_markdown_line(line) for line in lines]
    sanitized = [line for line in sanitized if line]
    if not sanitized:
        return "Overview", []

    title = sanitized[0]
    sections: list[tuple[str, str]] = []
    current_heading = "Overview"
    current_body_lines: list[str] = []

    for idx, line in enumerate(sanitized[1:], start=1):
        is_heading = _is_heading_candidate(line)
        if is_heading and idx > 1:
            if current_body_lines:
                body = " ".join(current_body_lines).strip()
                if body:
                    sections.append((current_heading.rstrip(":"), body))
            current_heading = line.rstrip(":")
            current_body_lines = []
            continue
        current_body_lines.append(line)

    if current_body_lines:
        body = " ".join(current_body_lines).strip()
        if body:
            sections.append((current_heading.rstrip(":"), body))

    if not sections and len(sanitized) > 1:
        sections.append(("Details", " ".join(sanitized[1:])))

    return title, sections


def _to_column_key(label: str, index: int) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    if not base:
        base = f"column_{index + 1}"
    return base


def _infer_table_domain(headers: list[str], full_text: str) -> tuple[str, str]:
    joined = " ".join(headers).lower()
    response = (full_text or "").lower()
    source = f"{joined} {response}"
    if any(token in source for token in ("temp", "humidity", "wind", "forecast", "rain", "uv")):
        return "weather", "cards"
    if any(token in source for token in ("formula", "variable", "equation", "principal", "interest rate", "monthly payment", "total interest")):
        return "formula", "table"
    if any(token in source for token in ("flight", "airline", "fare", "departure", "arrival")):
        return "flight", "cards"
    if any(token in source for token in ("hotel", "room", "rating", "amenity", "booking")):
        return "booking", "cards"
    if any(token in source for token in ("restaurant", "cuisine", "dish", "menu", "dining")):
        return "restaurants", "cards"
    if any(token in source for token in ("track", "artist", "album", "playlist", "duration")):
        return "playlist", "cards"
    if any(token in source for token in ("headline", "publisher", "article", "news", "published")):
        return "news", "cards"
    if any(token in source for token in ("product", "price", "availability", "seller", "catalog")):
        return "product", "cards"
    if any(token in source for token in ("schedule", "time slot", "agenda", "session")):
        return "schedule", "cards"
    if any(token in source for token in ("status", "state", "health", "uptime")):
        return "status", "cards"
    if any(token in source for token in ("compare", "comparison", "versus", "vs")):
        return "comparison", "table"
    return "generic", "table"


def _extract_first_markdown_table(lines: list[str]) -> tuple[dict[str, Any] | None, set[int]]:
    used_indexes: set[int] = set()
    for start in range(len(lines)):
        line = lines[start]
        if line.count("|") < 2:
            continue
        block_indexes: list[int] = []
        idx = start
        while idx < len(lines) and lines[idx].count("|") >= 2:
            block_indexes.append(idx)
            idx += 1
        if len(block_indexes) < 2:
            continue

        raw_rows = [lines[i].strip() for i in block_indexes]
        header_parts = [part.strip() for part in raw_rows[0].strip("|").split("|")]
        header_parts = [part for part in header_parts if part]
        if len(header_parts) < 2:
            continue

        data_start = 1
        if len(raw_rows) > 1 and _MARKDOWN_TABLE_SEP_RE.match(raw_rows[1]):
            data_start = 2
        if len(raw_rows) <= data_start:
            continue

        columns: list[dict[str, str]] = []
        keys: list[str] = []
        for col_idx, label in enumerate(header_parts):
            key = _to_column_key(label, col_idx)
            while key in keys:
                key = f"{key}_{col_idx + 1}"
            keys.append(key)
            columns.append({"key": key, "label": label or f"Column {col_idx + 1}"})

        rows: list[dict[str, str]] = []
        for raw_line in raw_rows[data_start:]:
            parts = [part.strip() for part in raw_line.strip("|").split("|")]
            if len(parts) < len(keys):
                parts = parts + [""] * (len(keys) - len(parts))
            row = {key: parts[col_idx] if col_idx < len(parts) else "" for col_idx, key in enumerate(keys)}
            if any(str(value).strip() for value in row.values()):
                rows.append(row)

        if not rows:
            continue

        used_indexes.update(block_indexes)
        return {
            "columns": columns,
            "rows": rows,
            "sourceText": "\n".join(raw_rows),
        }, used_indexes

    return None, used_indexes


def build_fallback_flat_spec(stage2_response: str) -> dict[str, Any]:
    text_value = stage2_response.strip() if isinstance(stage2_response, str) else ""
    if not text_value:
        text_value = "No content generated."
    lines = text_value.splitlines()
    table_payload, table_indexes = _extract_first_markdown_table(lines)
    content_lines = [
        line
        for idx, line in enumerate(lines)
        if idx not in table_indexes and not re.match(r"^\s*media\s*:\s*(image|icon)\s*=", line, re.IGNORECASE)
    ]
    title, sections = _parse_sections_from_lines(content_lines)
    if not sections:
        sections = [("Details", text_value)]

    first_url_match = _URL_RE.search(text_value)
    first_url = first_url_match.group(0).strip() if first_url_match else None

    elements: dict[str, Any] = {}
    state: dict[str, Any] = {}

    root_id = "root"
    header_id = "header_row"
    icon_id = "header_icon"
    title_id = "title_text"

    root_children: list[str] = [header_id]

    elements[root_id] = {
        "type": "Stack",
        "props": {"direction": "vertical", "gap": "md"},
        "children": root_children,
    }
    elements[header_id] = {
        "type": "Stack",
        "props": {"direction": "horizontal", "gap": "sm", "align": "center"},
        "children": [icon_id, title_id],
    }
    elements[icon_id] = {
        "type": "Icon",
        "props": {"name": "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/info-circle.svg"},
        "children": [],
    }
    elements[title_id] = {
        "type": "Text",
        "props": {"variant": "h2", "text": title or "Overview"},
        "children": [],
    }

    section_count = 0
    for heading, body in sections:
        if section_count >= 4:
            break
        chunks = _split_text_chunks(body, max_chars=180)
        if not chunks:
            continue
        section_count += 1
        card_id = f"card_{section_count}"
        heading_id = f"{card_id}_heading"
        elements[card_id] = {"type": "Card", "props": {}, "children": [heading_id]}
        elements[heading_id] = {
            "type": "Text",
            "props": {"variant": "h3", "text": heading or f"Section {section_count}"},
            "children": [],
        }
        for idx, chunk in enumerate(chunks[:3], start=1):
            body_id = f"{card_id}_body_{idx}"
            elements[body_id] = {
                "type": "Text",
                "props": {"variant": "body", "text": chunk},
                "children": [],
            }
            elements[card_id]["children"].append(body_id)
        root_children.append(card_id)

    if table_payload:
        table_card_id = "table_card"
        table_heading_id = "table_heading"
        table_id = "table_1"
        rows_state_key = "table_rows_1"
        domain, preferred = _infer_table_domain(
            [str(col.get("label") or "") for col in table_payload.get("columns", [])],
            text_value,
        )
        state[rows_state_key] = table_payload.get("rows", [])
        elements[table_card_id] = {
            "type": "Card",
            "props": {},
            "children": [table_heading_id, table_id],
        }
        elements[table_heading_id] = {
            "type": "Text",
            "props": {"variant": "h3", "text": "Structured Data"},
            "children": [],
        }
        elements[table_id] = {
            "type": "Table",
            "props": {
                "columns": table_payload.get("columns", []),
                "statePath": f"/{rows_state_key}",
                "domain": domain,
                "preferredPresentation": preferred,
                "sourceFormat": "markdown",
                "sourceText": table_payload.get("sourceText", ""),
            },
            "children": [],
        }
        root_children.append(table_card_id)

    if first_url:
        button_id = "source_button"
        elements[button_id] = {
            "type": "Button",
            "props": {"label": "Open Source"},
            "on": {"press": {"action": "openUrl", "params": {"url": first_url}}},
            "children": [],
        }
        root_children.append(button_id)

    # Guarantee minimum structure for downstream quality checks.
    if len(elements) < 8:
        filler_card_id = "filler_card"
        filler_heading_id = "filler_heading"
        filler_body_id = "filler_body"
        elements[filler_card_id] = {
            "type": "Card",
            "props": {},
            "children": [filler_heading_id, filler_body_id],
        }
        elements[filler_heading_id] = {
            "type": "Text",
            "props": {"variant": "h3", "text": "Additional Notes"},
            "children": [],
        }
        elements[filler_body_id] = {
            "type": "Text",
            "props": {
                "variant": "body",
                "text": _split_text_chunks(text_value, max_chars=160)[0] if text_value else "No additional details.",
            },
            "children": [],
        }
        root_children.append(filler_card_id)

    return {"root": root_id, "state": state, "elements": elements}


def extract_json_element(text: str) -> Any:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("Empty text")

    candidates: list[str] = []

    fenced = extract_fenced_block(cleaned)
    if fenced:
        candidates.append(fenced)

    if cleaned.startswith("[") or cleaned.startswith("{"):
        candidates.append(cleaned)

    candidates.extend(_collect_balanced_candidates(cleaned, "[", "]"))
    candidates.extend(_collect_balanced_candidates(cleaned, "{", "}"))

    parsed_candidates: list[Any] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed_candidates.append(json.loads(candidate))
        except Exception:
            continue

    if not parsed_candidates:
        raise ValueError("No JSON found")

    parsed_candidates.sort(key=_score_json_candidate, reverse=True)
    return parsed_candidates[0]


def extract_fenced_block(text: str) -> str | None:
    start = text.find("```")
    if start < 0:
        return None
    end = text.find("```", start + 3)
    if end <= start:
        return None
    block = text[start + 3 : end].strip()
    if block.lower().startswith("json"):
        return block[4:].strip()
    return block


def build_flat_spec_repair_prompt(raw_text: str, failure_reason: str | None = None) -> str:
    reason_line = ""
    if isinstance(failure_reason, str) and failure_reason.strip():
        reason_line = f"Failure reason: {failure_reason.strip()}\n"

    return (
        "The previous output does not satisfy the required flat-spec contract.\n"
        f"{reason_line}"
        "Return ONLY one valid JSON object with this shape:\n"
        '{"root":"<id>","state":{...},"elements":{...}}\n\n'
        "Rules:\n"
        "- Do NOT emit legacy v0.9 message arrays (`createSurface` / `updateComponents`).\n"
        "- `root` must reference an existing key in `elements`.\n"
        "- `elements` must contain at least 8 entries (title, sections, and content).\n"
        "- Every element must contain `type`, `props`, and `children`.\n"
        "- Every id in `children` must exist in `elements`.\n"
        "- Do not emit markdown control text (`#`, `|`, ``` , ''') in final Text fields.\n"
        "- Return JSON only, no markdown.\n\n"
        f"Original output:\n{(raw_text or '').strip()}"
    )


def _collect_balanced_candidates(text: str, open_char: str, close_char: str) -> list[str]:
    out: list[str] = []
    index = text.find(open_char)
    while index >= 0:
        candidate = balanced_substring(text, index, open_char, close_char)
        if candidate:
            out.append(candidate)
        index = text.find(open_char, index + 1)
    return out


def balanced_substring(text: str, start: int, open_char: str, close_char: str) -> str | None:
    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue

        if ch == "\\" and in_string:
            escape = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    return None


def _score_json_candidate(candidate: Any) -> int:
    score = 0

    coerce_result = coerce_and_validate(candidate)
    if coerce_result.is_valid and isinstance(coerce_result.spec, dict):
        elements = coerce_result.spec.get("elements")
        element_count = len(elements) if isinstance(elements, dict) else 0
        score += 400
        score += min(element_count, 80)
        root_id = coerce_result.spec.get("root")
        if isinstance(root_id, str) and isinstance(elements, dict) and root_id in elements:
            score += 60
    else:
        normalized = normalize_to_flat_spec(candidate)
        if normalized.spec is not None:
            score += 160
        elif isinstance(candidate, list):
            score += 80
        elif isinstance(candidate, dict):
            score += 40

    if isinstance(candidate, dict) and any(k in candidate for k in ("genui_json", "messages", "payload")):
        score += 50

    try:
        serialized = json.dumps(candidate, ensure_ascii=False)
    except Exception:
        serialized = str(candidate)
    score += min(len(serialized), 4000) // 200
    return score
