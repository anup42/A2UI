"""Pinned GenUICraft A2UI Express codec.

This dependency-free parser follows the pinned Express grammar and the strict
GenUICraft catalog/profile. Legacy graph conversion happens outside this codec.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any, Mapping

from .common import load_catalog, rewrite_element_ids

SENTINEL_OPEN = "<a2ui>"
SENTINEL_CLOSE = "</a2ui>"
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_PATH_RE = re.compile(r"\$[A-Za-z0-9_/]*\Z")
_CALL = "__express_call__"
_REF = "__express_ref__"
_CHECK = "__express_check__"
_SKIPPED = "__express_skipped__"


def encode(
    spec: Mapping[str, Any],
    *,
    shorten_ids: bool = True,
    pretty: bool = False,
) -> str:
    del pretty  # The canonical line-oriented form is already deterministic.
    source = rewrite_element_ids(
        spec,
        shorten=shorten_ids,
        reserve_root=True,
    )
    catalog = load_catalog()
    components = catalog.get("components", {})
    lines = [SENTINEL_OPEN]
    state = source.get("state")
    if isinstance(state, dict) and state:
        lines.append(f"$/={_express_value(state)}")

    for element_id, raw in source["elements"].items():
        element_type = str(raw.get("type", ""))
        props = deepcopy(raw.get("props") if isinstance(raw.get("props"), dict) else {})
        children = deepcopy(raw.get("children") if isinstance(raw.get("children"), list) else [])

        output_type = element_type
        if element_type == "Stack":
            direction = str(props.get("direction") or "").lower()
            if direction in {"horizontal", "row"}:
                output_type = "Row"
                props.pop("direction", None)
            elif direction in {"vertical", "column"}:
                output_type = "Column"
                props.pop("direction", None)

        descriptor = components.get(output_type)
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"Component {output_type!r} is missing from the Express catalog")
        positional = [str(value) for value in descriptor.get("positional", ())]
        defaults = descriptor.get("defaults") if isinstance(descriptor.get("defaults"), Mapping) else {}

        values: list[Any] = []
        for key in positional:
            values.append(children if key == "children" else props.get(key))
        last = -1
        for index, (key, value) in enumerate(zip(positional, values)):
            if value is not None and value != defaults.get(key) and value != []:
                last = index

        args: list[str] = []
        consumed: set[str] = set()
        first_missing = next(
            (index for index, value in enumerate(values)
             if value is None or value == defaults.get(positional[index]) or value == []),
            -1,
        )
        has_positional_gap = first_missing >= 0 and any(
            value is not None and value != defaults.get(positional[index]) and value != []
            for index, value in enumerate(values[first_missing + 1:], first_missing + 1)
        )
        if has_positional_gap:
            # A skipped positional slot may not be followed by another
            # positional value. Emit all populated slots as named args.
            for index, (key, value) in enumerate(zip(positional, values)):
                if value is None or value == defaults.get(key) or value == []:
                    continue
                rendered = _component_reference_array(value) if key == "children" else _express_value(value)
                args.append(f"{key}={rendered}")
                consumed.add(key)
        else:
            for index in range(last + 1):
                key = positional[index]
                value = values[index]
                if value is None or value == defaults.get(key) or value == []:
                    args.append("_")
                    continue
                if key == "children":
                    args.append(_component_reference_array(value))
                else:
                    args.append(_express_value(value))
                consumed.add(key)

        named: list[str] = []
        leftover = {key: value for key, value in props.items() if key not in consumed}
        for key, value in leftover.items():
            if not _allowed_property(descriptor, key):
                raise ValueError(f"Property {key!r} is not supported by component {output_type!r}")
            named.append(f"{key}={_express_value(value)}")
        if children and "children" not in consumed:
            named.append(f"children={_component_reference_array(children)}")

        raw_events = raw.get("on")
        if isinstance(raw_events, Mapping):
            for event_name, action_value in raw_events.items():
                keyword = _event_keyword(str(event_name))
                expression = _action_expression(action_value)
                if not keyword or expression is None:
                    raise ValueError(f"Event {event_name!r} cannot be represented by explicit Express syntax")
                named.append(f"{keyword}={expression}")

        for field, name in (
            ("repeat", "repeat"),
            ("visible", "visible"),
            ("watch", "watch"),
        ):
            if field in raw and raw[field] not in ({}, [], None):
                named.append(f"{name}={_express_value(raw[field])}")

        lines.append(f"{element_id}={output_type}({','.join(args + named)})")

    lines.append(SENTINEL_CLOSE)
    return "\n".join(lines)


def decode(text: Any) -> dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError("A2UI Express payload must be text")
    catalog = load_catalog()
    components = catalog.get("components", {})
    actions = catalog.get("actions", {})
    alias_map = _component_alias_map(components)
    state: dict[str, Any] = {}
    elements: dict[str, Any] = {}
    inline_counter = 0

    def materialize(element_id: str, call: Mapping[str, Any]) -> str:
        nonlocal inline_counter
        if element_id in elements:
            raise ValueError(f"Duplicate A2UI Express component id {element_id!r}")
        raw_component = str(call[_CALL])
        resolved = alias_map.get(raw_component.casefold())
        if resolved is None:
            raise ValueError(f"Unknown A2UI Express component {raw_component!r}")
        descriptor = components[resolved]
        canonical_type = str(descriptor.get("canonicalType") or resolved)
        positional = [str(value) for value in descriptor.get("positional", ())]
        props = deepcopy(
            descriptor.get("implicitProps")
            if isinstance(descriptor.get("implicitProps"), Mapping)
            else {}
        )
        children: list[str] = []
        on: dict[str, Any] = {}

        def add_children(value: Any) -> None:
            nonlocal inline_counter
            if not isinstance(value, list):
                raise ValueError(f"{raw_component}.children must be an array")
            for child in value:
                if _is_marker(child, _REF):
                    child_id = str(child[_REF])
                elif isinstance(child, str):
                    child_id = child
                elif _is_marker(child, _CALL):
                    inline_counter += 1
                    child_id = f"{element_id}_i{inline_counter}"
                    materialize(child_id, child)
                else:
                    raise ValueError(
                        f"{raw_component}.children must contain references or inline components"
                    )
                if not child_id:
                    raise ValueError(f"{raw_component}.children contains an empty reference")
                children.append(child_id)

        assigned: set[str] = set()
        skipped_positional = False
        for index, expression in enumerate(call.get("args", ())):
            if _is_marker(expression, _SKIPPED):
                skipped_positional = True
                continue
            if skipped_positional:
                raise ValueError(
                    f"Positional argument follows a skipped argument for {raw_component}; "
                    "use a named property instead"
                )
            if index >= len(positional):
                raise ValueError(f"Too many positional args for {raw_component}")
            key = positional[index]
            if key in assigned:
                raise ValueError(f"Duplicate component property {key!r} for {raw_component}")
            assigned.add(key)
            if key == "children":
                add_children(expression)
            else:
                props[key] = _plain_value(
                    expression,
                    allow_references=key in {"tabs", "trigger", "content", "child", "template", "itemTemplate"},
                )

        kwargs = call.get("kwargs", {})
        if not isinstance(kwargs, Mapping):
            raise ValueError("Express call kwargs must be an object")
        metadata: dict[str, Any] = {}
        metadata_expr: dict[str, Any] = {}
        for raw_key, expression in kwargs.items():
            key = str(raw_key)
            if key.startswith("_"):
                raise ValueError(
                    f"Opaque Express metadata parameter {key!r} is forbidden; "
                    "use explicit named properties"
                )
            action = _action_from_expression(expression, actions)
            event_name = _event_name_from_keyword(key) if action is not None else None
            if event_name:
                if event_name in on:
                    raise ValueError(f"Duplicate component event {event_name!r}")
                on[event_name] = action
                continue
            if key == "children":
                if key in assigned:
                    raise ValueError(f"Duplicate component property {key!r} for {raw_component}")
                assigned.add(key)
                add_children(expression)
                continue
            if key in {"repeat", "visible", "watch"}:
                if key in metadata:
                    raise ValueError(f"Duplicate component metadata property {key!r}")
                metadata[key] = _plain_value(expression, allow_references=key == "repeat")
                metadata_expr[key] = expression
                continue
            if not _allowed_property(descriptor, key):
                raise ValueError(f"Unknown named property {key!r} for {raw_component}")
            if key in assigned:
                raise ValueError(f"Duplicate component property {key!r} for {raw_component}")
            assigned.add(key)
            props[key] = _plain_value(
                expression,
                allow_references=key in {"tabs", "trigger", "content", "child", "template", "itemTemplate"},
            )

        element: dict[str, Any] = {
            "type": canonical_type,
            "props": props,
            "children": children,
        }
        if on:
            element["on"] = on
        for name in ("repeat", "visible", "watch"):
            if name not in metadata:
                continue
            value = metadata[name]
            if name in {"repeat", "watch"} and not isinstance(value, dict):
                raise ValueError(f"{name} must be a map")
            if name == "watch":
                normalized_watch: dict[str, Any] = {}
                raw_watch = metadata_expr[name]
                if not isinstance(raw_watch, Mapping):
                    raise ValueError("watch must be a map")
                for path, expression in raw_watch.items():
                    action = _action_from_expression(expression, actions)
                    if action is None:
                        raise ValueError(f"watch.{path} is not an action")
                    normalized_watch[str(path)] = action
                value = normalized_watch
            element[name] = deepcopy(value)

        elements[element_id] = element
        return element_id

    for statement in _statements(text):
        lhs, rhs = _split_assignment(statement)
        parser = _Parser(rhs)
        value = parser.value()
        parser.require_complete()
        if lhs.startswith("$"):
            plain = _plain_value(value)
            if lhs in {"$", "$/"}:
                if not isinstance(plain, dict):
                    raise ValueError("Root data-model assignment must be a map")
                state = deepcopy(plain)
                continue
            _set_state_path(state, lhs, plain)
            continue
        if not _IDENTIFIER_RE.fullmatch(lhs):
            raise ValueError(f"Invalid Express assignment target {lhs!r}")
        if not _is_marker(value, _CALL):
            raise ValueError(f"{lhs} must assign a component call")
        materialize(lhs, value)

    if "root" not in elements:
        raise ValueError("A2UI Express requires reserved root component")
    return {"root": "root", "state": state, "elements": elements}


def _component_alias_map(components: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    if not isinstance(components, Mapping):
        return out
    for name, descriptor in components.items():
        out[str(name).casefold()] = str(name)
        if isinstance(descriptor, Mapping):
            for alias in descriptor.get("aliases", ()):
                out[str(alias).casefold()] = str(name)
    return out


def _allowed_property(descriptor: Mapping[str, Any], key: str) -> bool:
    """Return whether an explicit named property is in the pinned catalog."""
    allowed = descriptor.get("allowedProperties")
    if isinstance(allowed, (list, tuple, set)):
        return key in {str(value) for value in allowed}
    positional = descriptor.get("positional")
    if isinstance(positional, (list, tuple, set)) and key in {str(value) for value in positional}:
        return True
    consumed = descriptor.get("consumedProps")
    return isinstance(consumed, (list, tuple, set)) and key in {str(value) for value in consumed}


def _component_reference_array(value: Any) -> str:
    if not isinstance(value, list):
        raise ValueError("Component children must be a list")
    rendered: list[str] = []
    for child in value:
        if not isinstance(child, str) or not _IDENTIFIER_RE.fullmatch(child):
            raise ValueError(f"Component reference {child!r} is not a valid Express identifier")
        rendered.append(child)
    return "[" + ",".join(rendered) + "]"


def _express_value(value: Any) -> str:
    if isinstance(value, str):
        if _PATH_RE.fullmatch(value):
            return value
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    if isinstance(value, list):
        return "[" + ",".join(_express_value(item) for item in value) + "]"
    if isinstance(value, Mapping):
        entries = []
        for key, item in value.items():
            key_text = str(key) if _IDENTIFIER_RE.fullmatch(str(key)) else json.dumps(str(key), ensure_ascii=False)
            entries.append(f"{key_text}:{_express_value(item)}")
        return "{" + ",".join(entries) + "}"
    raise ValueError(f"Unsupported Express value type {type(value).__name__}")


def _event_keyword(event_name: str) -> str | None:
    parts = re.findall(r"[A-Za-z0-9]+", event_name)
    if not parts:
        return None
    suffix = "".join(part[:1].upper() + part[1:] for part in parts)
    keyword = "on" + suffix
    return keyword if _IDENTIFIER_RE.fullmatch(keyword) else None


def _event_name_from_keyword(keyword: str) -> str | None:
    if not keyword.startswith("on") or len(keyword) <= 2:
        return None
    suffix = keyword[2:]
    return suffix[:1].lower() + suffix[1:]


def _action_expression(value: Any) -> str | None:
    if isinstance(value, list):
        rendered = [_action_expression(item) for item in value]
        if any(item is None for item in rendered):
            return None
        return "[" + ",".join(str(item) for item in rendered) + "]"
    if not isinstance(value, Mapping):
        return None
    action = value.get("action")
    if not isinstance(action, str) or not action:
        return None
    params = value.get("params") if isinstance(value.get("params"), Mapping) else {}
    function_name = "Event" if action == "emitEvent" else action
    catalog = load_catalog().get("actions", {})
    descriptor = catalog.get(action, {}) if isinstance(catalog, Mapping) else {}
    positional = [str(item) for item in descriptor.get("positional", ())]
    args: list[str] = []
    last = -1
    for index, name in enumerate(positional):
        if name in params:
            last = index
    consumed: set[str] = set()
    for index in range(last + 1):
        name = positional[index]
        if name not in params:
            args.append("_")
        else:
            args.append(_express_value(params[name]))
            consumed.add(name)
    for name, item in params.items():
        if name not in consumed:
            if not _IDENTIFIER_RE.fullmatch(str(name)):
                return None
            args.append(f"{name}={_express_value(item)}")
    return f"{function_name}({','.join(args)})"


def _action_from_expression(expression: Any, actions: Any) -> Any | None:
    if isinstance(expression, list):
        normalized = [_action_from_expression(item, actions) for item in expression]
        return normalized if normalized and all(item is not None for item in normalized) else None
    if isinstance(expression, Mapping) and isinstance(expression.get("action"), str):
        return deepcopy(dict(expression))
    if not _is_marker(expression, _CALL):
        return None

    raw_name = str(expression[_CALL])
    action_names = {
        str(name).casefold(): str(name)
        for name in actions
    } if isinstance(actions, Mapping) else {}
    action_name = "emitEvent" if raw_name.casefold() == "event" else action_names.get(raw_name.casefold())
    if action_name is None:
        return None
    descriptor = actions.get(action_name, {}) if isinstance(actions, Mapping) else {}
    positional = [str(item) for item in descriptor.get("positional", ())]
    params: dict[str, Any] = {}
    skipped_positional = False
    for index, item in enumerate(expression.get("args", ())):
        if _is_marker(item, _SKIPPED):
            skipped_positional = True
            continue
        if skipped_positional:
            raise ValueError(
                f"Positional parameter follows a skipped parameter for action {raw_name}; "
                "use a named parameter instead"
            )
        if index >= len(positional):
            raise ValueError(f"Too many positional parameters for action {raw_name}")
        if positional[index] in params:
            raise ValueError(f"Duplicate action parameter {positional[index]!r}")
        params[positional[index]] = _plain_value(item)
    for key, item in expression.get("kwargs", {}).items():
        if key in params:
            raise ValueError(f"Duplicate action parameter {key!r}")
        if not descriptor.get("allowAdditionalParams", False) and str(key) not in positional:
            raise ValueError(f"Unknown action parameter {key!r} for {raw_name}")
        params[str(key)] = _plain_value(item)
    return {"action": action_name, "params": params}


def _plain_value(value: Any, *, allow_references: bool = False) -> Any:
    if _is_marker(value, _REF):
        if not allow_references:
            raise ValueError(f"Unresolved Express variable {value[_REF]!r}")
        return str(value[_REF])
    if _is_marker(value, _CHECK):
        return {
            "check": value[_CHECK],
            "args": [_plain_value(item, allow_references=allow_references) for item in value.get("args", ())],
        }
    if _is_marker(value, _CALL):
        return {
            "call": value[_CALL],
            "args": [_plain_value(item, allow_references=allow_references) for item in value.get("args", ())],
            "kwargs": {
                str(key): _plain_value(item, allow_references=allow_references)
                for key, item in value.get("kwargs", {}).items()
            },
        }
    if _is_marker(value, _SKIPPED):
        return None
    if isinstance(value, list):
        return [_plain_value(item, allow_references=allow_references) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _plain_value(item, allow_references=allow_references)
            for key, item in value.items()
        }
    return deepcopy(value)


def _is_marker(value: Any, key: str) -> bool:
    return isinstance(value, Mapping) and key in value


def _set_state_path(state: dict[str, Any], lhs: str, value: Any) -> None:
    raw_path = lhs[2:] if lhs.startswith("$/") else lhs[1:]
    parts = [part for part in raw_path.split("/") if part]
    if not parts:
        if not isinstance(value, Mapping):
            raise ValueError("Root data-model assignment must be a map")
        state.clear()
        state.update(deepcopy(dict(value)))
        return
    current = state
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = deepcopy(value)


def _statements(text: str) -> list[str]:
    if not isinstance(text, str):
        raise ValueError("A2UI Express payload must be text")
    stripped = text.strip()
    if not stripped.startswith(SENTINEL_OPEN) or not stripped.endswith(SENTINEL_CLOSE):
        raise ValueError("A2UI Express payload must contain exactly one <a2ui> block")
    if stripped.count(SENTINEL_OPEN) != 1 or stripped.count(SENTINEL_CLOSE) != 1:
        raise ValueError("A2UI Express payload must contain exactly one <a2ui> block")
    body = stripped[len(SENTINEL_OPEN):-len(SENTINEL_CLOSE)]

    out: list[str] = []
    buffer: list[str] = []
    depth = 0
    index = 0
    string_delimiter: str | None = None
    raw_string = False
    escaped = False

    def flush() -> None:
        value = "".join(buffer).strip()
        buffer.clear()
        if value:
            out.append(value)

    while index < len(body):
        if string_delimiter is not None:
            if body.startswith(string_delimiter, index) and (raw_string or not escaped):
                buffer.append(string_delimiter)
                index += len(string_delimiter)
                string_delimiter = None
                raw_string = False
                escaped = False
                continue
            ch = body[index]
            buffer.append(ch)
            index += 1
            if not raw_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
            continue

        prefix = body[index:index + 4]
        if prefix[:4].lower() == 'r"""':
            buffer.append(body[index:index + 4])
            index += 4
            string_delimiter = '"""'
            raw_string = True
            continue
        if body.startswith('"""', index):
            buffer.append('"""')
            index += 3
            string_delimiter = '"""'
            continue
        if body[index:index + 2].lower() == 'r"':
            buffer.append(body[index:index + 2])
            index += 2
            string_delimiter = '"'
            raw_string = True
            continue
        if body[index] == '"':
            buffer.append('"')
            index += 1
            string_delimiter = '"'
            continue
        if body.startswith("/*", index):
            end = body.find("*/", index + 2)
            if end < 0:
                raise ValueError("Unterminated Express block comment")
            index = end + 2
            continue
        if body.startswith("//", index) or body[index] == "#":
            end = body.find("\n", index)
            index = len(body) if end < 0 else end
            continue

        ch = body[index]
        index += 1
        if ch in "([{":
            depth += 1
            buffer.append(ch)
        elif ch in ")]}":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced Express delimiter")
            buffer.append(ch)
        elif ch in "\n;" and depth == 0:
            flush()
        else:
            buffer.append(ch)

    if string_delimiter is not None or depth != 0:
        raise ValueError("Unterminated Express string or delimiter")
    flush()
    return out


def _split_assignment(statement: str) -> tuple[str, str]:
    depth = 0
    index = 0
    string_delimiter: str | None = None
    escaped = False
    while index < len(statement):
        if string_delimiter is not None:
            if statement.startswith(string_delimiter, index) and not escaped:
                index += len(string_delimiter)
                string_delimiter = None
                continue
            ch = statement[index]
            index += 1
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            continue
        if statement.startswith('"""', index):
            string_delimiter = '"""'
            index += 3
            continue
        ch = statement[index]
        if ch == '"':
            string_delimiter = '"'
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "=" and depth == 0:
            lhs = statement[:index].strip()
            rhs = statement[index + 1:].strip()
            if not lhs or not rhs:
                break
            return lhs, rhs
        index += 1
    raise ValueError(f"Express statement is not an assignment: {statement[:80]}")


class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.i = 0

    def ws(self) -> None:
        while self.i < len(self.text) and self.text[self.i].isspace():
            self.i += 1

    def peek(self) -> str:
        self.ws()
        return self.text[self.i:self.i + 1]

    def consume(self, token: str) -> None:
        self.ws()
        if not self.text.startswith(token, self.i):
            raise ValueError(f"Expected {token!r} at {self.i}")
        self.i += len(token)

    def ident(self) -> str:
        self.ws()
        match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", self.text[self.i:])
        if not match:
            raise ValueError(f"Expected identifier at {self.i}")
        self.i += len(match.group(0))
        return match.group(0)

    def require_complete(self) -> None:
        self.ws()
        if self.i != len(self.text):
            raise ValueError(f"Trailing Express syntax at {self.i}")

    def string(self) -> str:
        self.ws()
        raw = False
        if self.text[self.i:self.i + 1].lower() == "r":
            raw = True
            self.i += 1
        triple = self.text.startswith('"""', self.i)
        delimiter = '"""' if triple else '"'
        if not self.text.startswith(delimiter, self.i):
            raise ValueError(f"Expected string at {self.i}")
        self.i += len(delimiter)
        out: list[str] = []
        escaped = False
        while self.i < len(self.text):
            if self.text.startswith(delimiter, self.i) and (raw or not escaped):
                self.i += len(delimiter)
                return "".join(out)
            ch = self.text[self.i]
            self.i += 1
            if not triple and ch in "\r\n":
                raise ValueError("Single-line Express strings may not contain newlines")
            if raw:
                out.append(ch)
                continue
            if escaped:
                out.append({
                    "n": "\n",
                    "r": "\r",
                    "t": "\t",
                    "b": "\b",
                    "f": "\f",
                    '"': '"',
                    "\\": "\\",
                    "/": "/",
                }.get(ch, ch))
                escaped = False
            elif ch == "\\":
                escaped = True
            else:
                out.append(ch)
        raise ValueError("Unterminated Express string")

    def value(self) -> Any:
        self.ws()
        ch = self.peek()
        if ch == '"' or self.text[self.i:self.i + 2].lower() == 'r"':
            return self.string()
        if self.text.startswith('"""', self.i) or self.text[self.i:self.i + 4].lower() == 'r"""':
            return self.string()
        if ch == "[":
            return self.array()
        if ch == "{":
            return self.mapping()
        if ch == "$":
            match = re.match(r"\$[A-Za-z0-9_/]*", self.text[self.i:])
            if not match:
                raise ValueError(f"Invalid path at {self.i}")
            self.i += len(match.group(0))
            return match.group(0)
        if ch == "?":
            self.consume("?")
            name = self.ident()
            args = self.call_args()[0] if self.peek() == "(" else []
            return {_CHECK: name, "args": args}
        number = re.match(r"-?[0-9]+(?:\.[0-9]+)?", self.text[self.i:])
        if number:
            token = number.group(0)
            self.i += len(token)
            return float(token) if "." in token else int(token)
        name = self.ident()
        if name == "true":
            return True
        if name == "false":
            return False
        if name == "null":
            return None
        if name == "_":
            return {_SKIPPED: True}
        if self.peek() == "(":
            args, kwargs = self.call_args()
            return {_CALL: name, "args": args, "kwargs": kwargs}
        return {_REF: name}

    def array(self) -> list[Any]:
        self.consume("[")
        out: list[Any] = []
        if self.peek() != "]":
            while True:
                out.append(self.value())
                if self.peek() != ",":
                    break
                self.consume(",")
                if self.peek() == "]":
                    break
        self.consume("]")
        return out

    def mapping(self) -> dict[str, Any]:
        self.consume("{")
        out: dict[str, Any] = {}
        if self.peek() != "}":
            while True:
                key = self.string() if (
                    self.peek() == '"'
                    or self.text[self.i:self.i + 2].lower() == 'r"'
                    or self.text.startswith('"""', self.i)
                    or self.text[self.i:self.i + 4].lower() == 'r"""'
                ) else self.ident()
                self.consume(":")
                if key in out:
                    raise ValueError(f"Duplicate Express map key {key!r}")
                out[key] = self.value()
                if self.peek() != ",":
                    break
                self.consume(",")
                if self.peek() == "}":
                    break
        self.consume("}")
        return out

    def call_args(self) -> tuple[list[Any], dict[str, Any]]:
        self.consume("(")
        args: list[Any] = []
        kwargs: dict[str, Any] = {}
        seen_named = False
        if self.peek() != ")":
            while True:
                saved = self.i
                self.ws()
                candidate_match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", self.text[self.i:])
                candidate = candidate_match.group(0) if candidate_match else None
                if candidate is not None:
                    self.i += len(candidate)
                    if self.peek() == "=":
                        self.consume("=")
                        if candidate in kwargs:
                            raise ValueError(f"Duplicate named argument {candidate!r}")
                        kwargs[candidate] = self.value()
                        seen_named = True
                    else:
                        self.i = saved
                        if seen_named:
                            raise ValueError("Positional argument cannot follow a named argument")
                        args.append(self.value())
                else:
                    self.i = saved
                    args.append(self.value())
                if self.peek() != ",":
                    break
                self.consume(",")
                if self.peek() == ")":
                    break
        self.consume(")")
        return args, kwargs
