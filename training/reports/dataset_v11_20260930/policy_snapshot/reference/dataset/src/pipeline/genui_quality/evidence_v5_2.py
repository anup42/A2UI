"""Android-parity dynamic and renderer evidence for GenUI metric v5.2."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import re
import types
from typing import Any, Callable, Mapping

from . import _core
from .config import RewardConfig
from . import evidence_v5_1 as _v51
from .matching_v5_1 import PreparedTextBlock, prepare_text_block


EVIDENCE_POLICY_VERSION = "5.2.0"
DYNAMIC_SEMANTICS_VERSION = "android-flat-expr-5.2.0"
DYNAMIC_PARITY_VECTOR_VERSION = "1.0.0"
_UNKNOWN = _v51._UNKNOWN  # Reuse the sentinel cleaned by the shared interpreter.
_EXPRESSION_KEYS = frozenset(
    {
        "$item",
        "$state",
        "$bindState",
        "$bindItem",
        "$index",
        "$cond",
        "$template",
        "$computed",
        "literalString",
        "literalNumber",
        "literalBoolean",
    }
)


ComputedFunction = Callable[[Mapping[str, Any]], Any]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_behavior_value(
    value: Any,
    *,
    seen: set[int] | None = None,
) -> Any:
    """Serialize callable state without process-specific object addresses."""

    active = seen if seen is not None else set()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"float": "nan"}
        if math.isinf(value):
            return {"float": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, types.CodeType):
        return {
            "kind": "code",
            "argcount": value.co_argcount,
            "posonlyargcount": value.co_posonlyargcount,
            "kwonlyargcount": value.co_kwonlyargcount,
            "nlocals": value.co_nlocals,
            "flags": value.co_flags,
            "bytecode": value.co_code.hex(),
            "constants": [
                _stable_behavior_value(item, seen=active)
                for item in value.co_consts
            ],
            "names": list(value.co_names),
            "varnames": list(value.co_varnames),
            "freevars": list(value.co_freevars),
            "cellvars": list(value.co_cellvars),
        }
    identifier = id(value)
    if identifier in active:
        return {
            "cycle": (
                f"{type(value).__module__}.{type(value).__qualname__}"
            )
        }
    active.add(identifier)
    try:
        if isinstance(value, Mapping):
            items = [
                (
                    _canonical_json(
                        _stable_behavior_value(key, seen=active)
                    ),
                    _stable_behavior_value(nested, seen=active),
                )
                for key, nested in value.items()
            ]
            return {
                "mapping": [
                    [key, nested]
                    for key, nested in sorted(items, key=lambda item: item[0])
                ]
            }
        if isinstance(value, (list, tuple)):
            return {
                "sequence_type": type(value).__name__,
                "items": [
                    _stable_behavior_value(item, seen=active)
                    for item in value
                ],
            }
        if isinstance(value, (set, frozenset)):
            items = [
                _stable_behavior_value(item, seen=active)
                for item in value
            ]
            return {
                "set_type": type(value).__name__,
                "items": sorted(items, key=_canonical_json),
            }
        if callable(value):
            return _callable_behavior_payload(value, seen=active)
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, Mapping):
            return {
                "object_type": (
                    f"{type(value).__module__}.{type(value).__qualname__}"
                ),
                "attributes": _stable_behavior_value(
                    attributes, seen=active
                ),
            }
        return {
            "object_type": (
                f"{type(value).__module__}.{type(value).__qualname__}"
            )
        }
    finally:
        active.remove(identifier)


def _callable_behavior_payload(
    function: ComputedFunction,
    *,
    seen: set[int] | None = None,
) -> dict[str, Any]:
    code = getattr(function, "__code__", None)
    payload: dict[str, Any] = {
        "module": str(getattr(function, "__module__", "")),
        "qualname": str(getattr(function, "__qualname__", "")),
    }
    if code is not None:
        payload.update(
            {
                "code": _stable_behavior_value(code, seen=seen),
                "defaults": _stable_behavior_value(
                    getattr(function, "__defaults__", None), seen=seen
                ),
                "kwdefaults": _stable_behavior_value(
                    getattr(function, "__kwdefaults__", None), seen=seen
                ),
                "closure": [
                    _stable_behavior_value(cell.cell_contents, seen=seen)
                    for cell in (getattr(function, "__closure__", None) or ())
                ],
            }
        )
    else:
        try:
            payload["source"] = inspect.getsource(function)
        except (OSError, TypeError):
            payload["type"] = (
                f"{type(function).__module__}.{type(function).__qualname__}"
            )
    bound_self = getattr(function, "__self__", None)
    if bound_self is not None:
        payload["bound_self"] = _stable_behavior_value(
            bound_self, seen=seen
        )
    return payload


def _callable_behavior_hash(function: ComputedFunction) -> str:
    payload = _callable_behavior_payload(function)
    return _sha256_json(payload)


@dataclass(frozen=True)
class ComputedFunctionRegistryV52:
    registry_id: str
    registry_version: str
    functions: Mapping[str, ComputedFunction]
    manifest_hash: str

    def __post_init__(self) -> None:
        if not str(self.registry_id).strip():
            raise ValueError("computed registry_id must be non-empty")
        if not str(self.registry_version).strip():
            raise ValueError("computed registry_version must be non-empty")
        if not str(self.manifest_hash).strip():
            raise ValueError("computed manifest_hash must be non-empty")
        if any(
            not str(name).strip() or not callable(function)
            for name, function in self.functions.items()
        ):
            raise ValueError(
                "computed functions must have non-empty names and be callable"
            )

    def identity(self) -> dict[str, Any]:
        return {
            "registry_id": self.registry_id,
            "registry_version": self.registry_version,
            "manifest_hash": self.manifest_hash,
            "function_behavior_hashes": {
                str(name): _callable_behavior_hash(function)
                for name, function in sorted(self.functions.items())
            },
        }

    def identity_hash(self) -> str:
        return _sha256_json(self.identity())


def _kotlin_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _builtin_concat(args: Mapping[str, Any]) -> str:
    return "".join(_kotlin_string(value) for value in args.values())


def _builtin_uppercase(args: Mapping[str, Any]) -> str:
    return _kotlin_string(args.get("value")).upper()


def _builtin_lowercase(args: Mapping[str, Any]) -> str:
    return _kotlin_string(args.get("value")).lower()


def _builtin_coalesce(args: Mapping[str, Any]) -> Any:
    values = args.get("values")
    if not isinstance(values, list):
        return None
    for candidate in values:
        if candidate is None:
            continue
        if isinstance(candidate, str) and not candidate.strip():
            continue
        return candidate
    return None


def _builtin_sum(args: Mapping[str, Any]) -> float:
    values = args.get("values")
    if not isinstance(values, list):
        return 0.0
    return float(
        sum(
            float(value)
            for value in values
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        )
    )


_BUILTIN_FUNCTIONS: dict[str, ComputedFunction] = {
    "concat": _builtin_concat,
    "uppercase": _builtin_uppercase,
    "lowercase": _builtin_lowercase,
    "coalesce": _builtin_coalesce,
    "sum": _builtin_sum,
}
BUILTIN_COMPUTED_MANIFEST = {
    "renderer": "FlatSpecRenderer.kt",
    "semantics_version": DYNAMIC_SEMANTICS_VERSION,
    "functions": {
        "concat": "args.values ordered join; null becomes empty string",
        "uppercase": "args.value Kotlin-style string uppercase",
        "lowercase": "args.value Kotlin-style string lowercase",
        "coalesce": "first non-null and non-blank-string list value",
        "sum": "sum numeric list values as Double; Boolean contributes zero",
    },
}
BUILTIN_COMPUTED_MANIFEST_HASH = _sha256_json(BUILTIN_COMPUTED_MANIFEST)
DEFAULT_COMPUTED_REGISTRY_V52 = ComputedFunctionRegistryV52(
    registry_id="android-flat-spec-builtins",
    registry_version="1.0.0",
    functions=_BUILTIN_FUNCTIONS,
    manifest_hash=BUILTIN_COMPUTED_MANIFEST_HASH,
)


def effective_registry_identity(
    custom: ComputedFunctionRegistryV52 | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "builtin": DEFAULT_COMPUTED_REGISTRY_V52.identity(),
        "builtin_identity_hash": (
            DEFAULT_COMPUTED_REGISTRY_V52.identity_hash()
        ),
    }
    if custom is not None:
        payload["custom"] = custom.identity()
        payload["custom_identity_hash"] = custom.identity_hash()
    return payload


def _effective_functions(
    custom: ComputedFunctionRegistryV52 | None,
) -> dict[str, ComputedFunction]:
    result = dict(DEFAULT_COMPUTED_REGISTRY_V52.functions)
    if custom is not None:
        result.update(
            {str(name): function for name, function in custom.functions.items()}
        )
    return result


def _decode_pointer_token(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _encode_pointer_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _pointer_tokens(path: Any) -> list[str]:
    value = str(path if path is not None else "")
    if not value.strip() or value == "/":
        return []
    return [
        _decode_pointer_token(token)
        for token in value.removeprefix("/").split("/")
        if token
    ]


def resolve_renderer_path_v5_2(root: Any, path: Any) -> Any:
    """Mirror ``FlatSpecParser.getAtPath`` including root and slash rules."""

    current = root
    for token in _pointer_tokens(path):
        if isinstance(current, Mapping):
            current = current.get(token)
        elif isinstance(current, list):
            if not re.fullmatch(r"\d+", token):
                return None
            index = int(token)
            if index < 0 or index >= len(current):
                return None
            current = current[index]
        else:
            return None
    return current


def _normalize_pointer(path: Any) -> str:
    value = str(path if path is not None else "").strip()
    if not value:
        return ""
    return value if value.startswith("/") else "/" + value


def _combine_pointer(base: str | None, token: str) -> str | None:
    normalized_base = str(base or "").strip()
    if not normalized_base:
        return None
    return (
        _normalize_pointer(normalized_base).rstrip("/")
        + "/"
        + _encode_pointer_token(str(token))
    )


def _resolve_item_value(item: Any, path: Any) -> Any:
    if item is None:
        return None
    normalized = str(path if path is not None else "").strip()
    if not normalized:
        return item
    resolved = resolve_renderer_path_v5_2(item, normalized)
    if resolved is not None:
        return resolved
    return item if normalized == "value" else None


def _to_double(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _truthy(value: Any) -> bool:
    if value is None or value is _UNKNOWN:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    if isinstance(value, str):
        return bool(value.strip()) and value.casefold() != "false"
    return True


def _deep_equal(left: Any, right: Any) -> bool:
    if left is right:
        return True
    if left is None or right is None:
        return False
    if isinstance(left, bool) != isinstance(right, bool):
        if isinstance(left, (bool, int, float)) and isinstance(
            right, (bool, int, float)
        ):
            return False
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return (
            set(left) == set(right)
            and all(_deep_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _deep_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


class DynamicEvidenceResolverV52:
    """Bounded Python port of Android ``FlatExprResolver``."""

    def __init__(
        self,
        state: Mapping[str, Any],
        config: RewardConfig,
        computed_registry: ComputedFunctionRegistryV52 | None = None,
    ) -> None:
        self.state = state
        self.config = config
        self.computed_registry = computed_registry
        self.computed_functions = _effective_functions(computed_registry)
        self.unknown: list[str] = []

    def _unknown(self, code: str) -> None:
        if code not in self.unknown:
            self.unknown.append(code)

    def _bind_item(self, raw_path: Any, base_path: str | None) -> Any:
        if not str(base_path or "").strip():
            return None
        requested = str(raw_path if raw_path is not None else "").strip()
        resolved_path = (
            _normalize_pointer(base_path)
            if not requested
            else _combine_pointer(base_path, requested)
        )
        if not str(resolved_path or "").strip():
            return None
        return resolve_renderer_path_v5_2(self.state, resolved_path)

    def resolve(
        self,
        value: Any,
        *,
        item: Any = None,
        index: int | None = None,
        base_path: str | None = None,
        depth: int = 0,
        path: str = "value",
    ) -> Any:
        if depth > self.config.max_expression_depth:
            self._unknown(f"expression_depth:{path}")
            return _UNKNOWN
        if isinstance(value, Mapping):
            string_map = {str(key): nested for key, nested in value.items()}
            has_expression = any(
                key in _EXPRESSION_KEYS for key in string_map
            )
            if not has_expression:
                return {
                    key: self.resolve(
                        nested,
                        item=item,
                        index=index,
                        base_path=base_path,
                        depth=depth + 1,
                        path=f"{path}.{key}",
                    )
                    for key, nested in string_map.items()
                }
            if "$item" in string_map:
                return _resolve_item_value(item, string_map.get("$item"))
            if "$state" in string_map:
                return resolve_renderer_path_v5_2(
                    self.state, string_map.get("$state")
                )
            if "$bindState" in string_map:
                return resolve_renderer_path_v5_2(
                    self.state, string_map.get("$bindState")
                )
            if "$bindItem" in string_map:
                return self._bind_item(
                    string_map.get("$bindItem"), base_path
                )
            if "$index" in string_map:
                return index
            if "$cond" in string_map:
                branch = (
                    "$then"
                    if self.evaluate_condition(
                        string_map.get("$cond"),
                        item=item,
                        index=index,
                        base_path=base_path,
                        path=f"{path}.$cond",
                    )
                    else "$else"
                )
                return self.resolve(
                    string_map.get(branch),
                    item=item,
                    index=index,
                    base_path=base_path,
                    depth=depth + 1,
                    path=f"{path}.{branch}",
                )
            if "$template" in string_map:
                return self.resolve_inline_template(
                    str(string_map.get("$template") or ""),
                    item=item,
                    index=index,
                    path=path,
                )
            if "$computed" in string_map:
                name = str(string_map.get("$computed") or "").strip()
                function = self.computed_functions.get(name)
                if function is None:
                    self._unknown(f"computed_function_unknown:{name}")
                    return _UNKNOWN
                raw_args = string_map.get("args")
                args = (
                    {
                        str(key): self.resolve(
                            nested,
                            item=item,
                            index=index,
                            base_path=base_path,
                            depth=depth + 1,
                            path=f"{path}.args.{key}",
                        )
                        for key, nested in raw_args.items()
                    }
                    if isinstance(raw_args, Mapping)
                    else {}
                )
                try:
                    return function(args)
                except Exception as exc:
                    self._unknown(
                        f"computed_function_error:{name}:{type(exc).__name__}"
                    )
                    return _UNKNOWN
            if "literalString" in string_map:
                return string_map.get("literalString")
            if "literalNumber" in string_map:
                return string_map.get("literalNumber")
            if "literalBoolean" in string_map:
                return string_map.get("literalBoolean")
            return {
                key: self.resolve(
                    nested,
                    item=item,
                    index=index,
                    base_path=base_path,
                    depth=depth + 1,
                    path=f"{path}.{key}",
                )
                for key, nested in string_map.items()
            }
        if isinstance(value, list):
            return [
                self.resolve(
                    nested,
                    item=item,
                    index=index,
                    base_path=base_path,
                    depth=depth + 1,
                    path=f"{path}[{position}]",
                )
                for position, nested in enumerate(value)
            ]
        if isinstance(value, str):
            return self.resolve_inline_template(
                value, item=item, index=index, path=path
            )
        return value

    def evaluate_condition(
        self,
        condition: Any,
        *,
        item: Any = None,
        index: int | None = None,
        base_path: str | None = None,
        path: str = "condition",
    ) -> bool:
        if condition is None:
            return True
        if isinstance(condition, bool):
            return condition
        if isinstance(condition, list):
            return all(
                self.evaluate_condition(
                    child,
                    item=item,
                    index=index,
                    base_path=base_path,
                    path=f"{path}[{position}]",
                )
                for position, child in enumerate(condition)
            )
        if not isinstance(condition, Mapping):
            return True
        expr = {str(key): value for key, value in condition.items()}
        if "$and" in expr:
            values = expr.get("$and")
            if not isinstance(values, list):
                return True
            return all(
                self.evaluate_condition(
                    child,
                    item=item,
                    index=index,
                    base_path=base_path,
                    path=f"{path}.$and[{position}]",
                )
                for position, child in enumerate(values)
            )
        if "$or" in expr:
            values = expr.get("$or")
            if not isinstance(values, list):
                return False
            return any(
                self.evaluate_condition(
                    child,
                    item=item,
                    index=index,
                    base_path=base_path,
                    path=f"{path}.$or[{position}]",
                )
                for position, child in enumerate(values)
            )
        if "$state" in expr:
            raw_value = resolve_renderer_path_v5_2(
                self.state, expr.get("$state")
            )
        elif "$item" in expr:
            raw_value = _resolve_item_value(item, expr.get("$item"))
        elif "$index" in expr:
            raw_value = index
        elif "value" in expr:
            raw_value = self.resolve(
                expr.get("value"),
                item=item,
                index=index,
                base_path=base_path,
                path=f"{path}.value",
            )
        else:
            raw_value = None

        def resolved(name: str) -> Any:
            return self.resolve(
                expr.get(name),
                item=item,
                index=index,
                base_path=base_path,
                path=f"{path}.{name}",
            )

        if "eq" in expr:
            result = _deep_equal(raw_value, resolved("eq"))
        elif "neq" in expr:
            result = not _deep_equal(raw_value, resolved("neq"))
        elif "gt" in expr:
            result = _to_double(raw_value) > _to_double(resolved("gt"))
        elif "gte" in expr:
            result = _to_double(raw_value) >= _to_double(resolved("gte"))
        elif "lt" in expr:
            result = _to_double(raw_value) < _to_double(resolved("lt"))
        elif "lte" in expr:
            result = _to_double(raw_value) <= _to_double(resolved("lte"))
        else:
            result = _truthy(raw_value)
        return not result if expr.get("not") is True else result

    def resolve_inline_template(
        self,
        template: str,
        *,
        item: Any,
        index: int | None,
        path: str,
    ) -> str:
        value = str(template)

        def item_replacement(match: re.Match[str]) -> str:
            resolved = _resolve_item_value(item, match.group(1).strip())
            return "" if resolved is None else _kotlin_string(resolved)

        value = re.sub(
            r"\$\{\s*\$item[./]([^}]+?)\s*\}",
            item_replacement,
            value,
        )
        value = re.sub(
            r"\{\{\s*\$item[./]([^}]+?)\s*\}\}",
            item_replacement,
            value,
        )
        value = re.sub(
            r"(?<!\$)\{\s*\$item[./]([^}]+?)\s*\}",
            item_replacement,
            value,
        )

        def legacy_index(match: re.Match[str]) -> str:
            if index is None:
                return ""
            return str(index + 1 if "+" in match.group(0) else index)

        value = re.sub(
            r"(?<!\$)\{\s*\$index\s*(?:\+\s*1)?\s*\}",
            legacy_index,
            value,
        )

        def generic(match: re.Match[str]) -> str:
            raw = match.group(1).strip()
            if raw in {"index", "index_0"}:
                return "" if index is None else str(index)
            if raw == "index_1":
                return "" if index is None else str(index + 1)
            state_value = resolve_renderer_path_v5_2(
                self.state, _normalize_pointer(raw)
            )
            if state_value is not None:
                return _kotlin_string(state_value)
            item_value = _resolve_item_value(item, raw)
            return "" if item_value is None else _kotlin_string(item_value)

        value = re.sub(r"\$\{([^}]+)\}", generic, value)
        if len(value) > self.config.max_string_expansion_length:
            self._unknown(f"string_truncated:{path}")
            value = value[: self.config.max_string_expansion_length]
        return value


class _EvidenceInterpreterV52(_v51._EvidenceInterpreter):
    def __init__(
        self,
        spec: Mapping[str, Any],
        config: RewardConfig,
        computed_registry: ComputedFunctionRegistryV52 | None,
    ) -> None:
        super().__init__(spec, config, {})
        self.resolver = DynamicEvidenceResolverV52(
            self.state, config, computed_registry
        )

    def _collect_element(
        self,
        element_id: str,
        element: Mapping[str, Any],
        props: Mapping[str, Any],
    ) -> None:
        super()._collect_element(element_id, element, props)
        element_type = str(element.get("type") or "").casefold().replace("_", "")
        if element_type in {
            "textfield",
            "checkbox",
            "choicepicker",
            "slider",
            "datetimeinput",
        }:
            self._add_role(
                "form",
                {
                    "component_id": element_id,
                    "title": str(
                        props.get("label")
                        or props.get("title")
                        or props.get("placeholder")
                        or ""
                    ),
                    "control_type": element_type,
                },
            )


@dataclass(frozen=True)
class V52EvidenceResult:
    output: _core.OutputEvidence
    output_tables: tuple[_core.OutputTable, ...]
    output_charts: tuple[_core.OutputTable, ...]
    structured_data_payloads: tuple[_core.OutputTable, ...]
    role_instances: dict[str, tuple[dict[str, Any], ...]]
    prepared_visible_blocks: tuple[PreparedTextBlock, ...]
    unknown_diagnostics: tuple[str, ...]
    dynamic_expression_unknown_count: int
    dynamic_evidence_complete: bool
    expanded_evidence_nodes: int
    truncated: bool
    dynamic_semantics: dict[str, Any]


def collect_output_evidence_v5_2(
    spec: Mapping[str, Any],
    audit: _core.GraphAudit,
    config: RewardConfig,
    *,
    computed_registry: ComputedFunctionRegistryV52 | None = None,
) -> V52EvidenceResult:
    interpreter = _EvidenceInterpreterV52(
        spec, config, computed_registry
    )
    root = spec.get("root")
    if isinstance(root, str):
        interpreter.walk(root)
    static = _core.collect_output_evidence(spec, audit)
    visible_blocks = [
        value
        for value in (
            str(item).strip() for item in interpreter.visible_blocks
        )
        if value
    ]
    content_tokens = Counter(_core.tokenize(" ".join(visible_blocks)))
    exact_values: Counter[str] = Counter()
    for block in visible_blocks:
        exact_values.update(
            _core.normalize_match_text(value)
            for value in _core._EXACT_VALUE_RE.findall(block)  # type: ignore[attr-defined]
        )
        exact_values.update(
            _core.normalize_match_text(value)
            for value in _core._DATE_RE.findall(block)  # type: ignore[attr-defined]
        )
    exact_values.pop("", None)
    structured = [*interpreter.tables, *interpreter.charts]
    reachable_type_counts: Counter[str] = Counter()
    reachable_valid_type_counts: Counter[str] = Counter()
    state = spec.get("state", {})
    for element_id in audit.reachable_ids:
        element = static.elements.get(element_id)
        if not isinstance(element, Mapping):
            continue
        element_type = str(element.get("type") or "")
        reachable_type_counts[element_type] += 1
        if _core._element_contract_score(element, state) >= 1.0:  # type: ignore[attr-defined]
            reachable_valid_type_counts[element_type] += 1
    output = _core.OutputEvidence(
        elements=static.elements,
        reachable_ids=set(audit.reachable_ids),
        visible_blocks=visible_blocks,
        headings=interpreter.headings,
        heading_variants=interpreter.heading_variants,
        tables=structured,
        actions=interpreter.actions,
        media=interpreter.media,
        type_counts=reachable_type_counts,
        content_tokens=content_tokens,
        exact_values=exact_values,
        valid_type_counts=reachable_valid_type_counts,
    )
    unknown = tuple(interpreter.resolver.unknown)
    dynamic_semantics = {
        "parity_version": DYNAMIC_SEMANTICS_VERSION,
        "parity_vector_version": DYNAMIC_PARITY_VECTOR_VERSION,
        "unknown_count": len(unknown),
        "unknown_codes": list(unknown),
        "complete": not unknown and not interpreter.truncated,
        "expanded_evidence_nodes": interpreter.expanded_nodes,
        "truncated": interpreter.truncated,
        "computed_registry": effective_registry_identity(computed_registry),
        "limits": {
            "max_repeat_items": config.max_repeat_items,
            "max_expanded_evidence_nodes": (
                config.max_expanded_evidence_nodes
            ),
            "max_expression_depth": config.max_expression_depth,
            "max_string_expansion_length": (
                config.max_string_expansion_length
            ),
        },
    }
    return V52EvidenceResult(
        output=output,
        output_tables=tuple(interpreter.tables),
        output_charts=tuple(interpreter.charts),
        structured_data_payloads=tuple(structured),
        role_instances={
            role: tuple(values)
            for role, values in sorted(interpreter.role_instances.items())
        },
        prepared_visible_blocks=tuple(
            prepare_text_block(value) for value in visible_blocks
        ),
        unknown_diagnostics=unknown,
        dynamic_expression_unknown_count=len(unknown),
        dynamic_evidence_complete=not unknown and not interpreter.truncated,
        expanded_evidence_nodes=interpreter.expanded_nodes,
        truncated=interpreter.truncated,
        dynamic_semantics=dynamic_semantics,
    )


__all__ = [
    "BUILTIN_COMPUTED_MANIFEST",
    "BUILTIN_COMPUTED_MANIFEST_HASH",
    "ComputedFunctionRegistryV52",
    "DEFAULT_COMPUTED_REGISTRY_V52",
    "DYNAMIC_PARITY_VECTOR_VERSION",
    "DYNAMIC_SEMANTICS_VERSION",
    "DynamicEvidenceResolverV52",
    "EVIDENCE_POLICY_VERSION",
    "V52EvidenceResult",
    "collect_output_evidence_v5_2",
    "effective_registry_identity",
    "resolve_renderer_path_v5_2",
]
