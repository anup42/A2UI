"""Streaming Android-parity renderer evidence for metric v5.3."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import re
from typing import Any, Mapping

from ..flat_spec_semantics import iter_renderer_references
from . import _core
from .applicability_v5_3 import classify_action, classify_media
from .config_v5_3 import RewardConfigV53
from .evidence_v5_1 import _combine_pointer, _normalize_pointer
from .evidence_v5_2 import (
    ComputedFunction,
    ComputedFunctionRegistryV52,
    _EvidenceInterpreterV52,
    _UNKNOWN,
    _callable_behavior_hash,
    _effective_functions,
    _sha256_json,
    _to_double,
    _truthy,
    resolve_renderer_path_v5_2,
)
from .matching_v5_1 import PreparedTextBlock, prepare_text_block


EVIDENCE_POLICY_VERSION = "5.3.0"
DYNAMIC_SEMANTICS_VERSION = "android-flat-expr-5.3.0"
DYNAMIC_PARITY_VECTOR_VERSION = "2.0.0"
ComputedFunctionRegistryV53 = ComputedFunctionRegistryV52


def kotlin_string_v5_3(value: Any, *, nested: bool = False) -> str:
    if value is None:
        return "null" if nested else ""
    if value is _UNKNOWN:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ", ".join(
            kotlin_string_v5_3(item, nested=True) for item in value
        ) + "]"
    if isinstance(value, Mapping):
        return "{" + ", ".join(
            f"{key}={kotlin_string_v5_3(nested_value, nested=True)}"
            for key, nested_value in value.items()
        ) + "}"
    return str(value)


def deep_equal_v5_3(left: Any, right: Any) -> bool:
    """Mirror Kotlin equality, including integral-versus-Double identity."""

    if left is right:
        return True
    if left is None or right is None:
        return False
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, int) or isinstance(right, int):
        return isinstance(left, int) and isinstance(right, int) and left == right
    if isinstance(left, float) or isinstance(right, float):
        return (
            isinstance(left, float)
            and isinstance(right, float)
            and left == right
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return (
            set(left) == set(right)
            and all(deep_equal_v5_3(left[key], right[key]) for key in left)
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            deep_equal_v5_3(a, b) for a, b in zip(left, right)
        )
    return type(left) is type(right) and left == right


def _builtin_concat(args: Mapping[str, Any]) -> str:
    return "".join(kotlin_string_v5_3(value) for value in args.values())


def _builtin_uppercase(args: Mapping[str, Any]) -> str:
    return kotlin_string_v5_3(args.get("value")).upper()


def _builtin_lowercase(args: Mapping[str, Any]) -> str:
    return kotlin_string_v5_3(args.get("value")).lower()


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
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
        )
    )


_BUILTINS: dict[str, ComputedFunction] = {
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
        "concat": (
            "ordered Kotlin toString join; direct null is empty; nested "
            "collections stringify recursively"
        ),
        "uppercase": "Kotlin-compatible value string uppercase",
        "lowercase": "Kotlin-compatible value string lowercase",
        "coalesce": "first non-null and non-blank-string list value",
        "sum": "numeric list values summed as Double; Boolean contributes zero",
    },
}
BUILTIN_COMPUTED_MANIFEST_HASH = _sha256_json(BUILTIN_COMPUTED_MANIFEST)
DEFAULT_COMPUTED_REGISTRY_V53 = ComputedFunctionRegistryV53(
    registry_id="android-flat-spec-builtins",
    registry_version="2.0.0",
    functions=_BUILTINS,
    manifest_hash=BUILTIN_COMPUTED_MANIFEST_HASH,
)


def effective_registry_identity_v5_3(
    custom: ComputedFunctionRegistryV53 | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "builtin": DEFAULT_COMPUTED_REGISTRY_V53.identity(),
        "builtin_identity_hash": DEFAULT_COMPUTED_REGISTRY_V53.identity_hash(),
    }
    if custom is not None:
        payload["custom"] = custom.identity()
        payload["custom_identity_hash"] = custom.identity_hash()
    return payload


class DynamicEvidenceResolverV53:
    """Bounded Python port of the production Android expression resolver."""

    def __init__(
        self,
        state: Mapping[str, Any],
        config: RewardConfigV53,
        computed_registry: ComputedFunctionRegistryV53 | None = None,
    ) -> None:
        self.state = state
        self.config = config
        self.computed_registry = computed_registry
        self.computed_functions = dict(_BUILTINS)
        if computed_registry is not None:
            self.computed_functions.update(computed_registry.functions)
        self.unknown: list[str] = []
        self.expression_evaluations = 0

    def _unknown(self, code: str) -> None:
        if code not in self.unknown:
            self.unknown.append(code)

    def _consume(self, path: str) -> bool:
        self.expression_evaluations += 1
        if self.expression_evaluations > self.config.max_expression_evaluations:
            self._unknown(f"expression_evaluation_budget:{path}")
            return False
        return True

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

    @staticmethod
    def _resolve_item(item: Any, path: Any) -> Any:
        if item is None:
            return None
        normalized = str(path if path is not None else "").strip()
        if not normalized:
            return item
        resolved = resolve_renderer_path_v5_2(item, normalized)
        if resolved is not None:
            return resolved
        return item if normalized == "value" else None

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
        if not self._consume(path):
            return _UNKNOWN
        if depth > self.config.max_expression_depth:
            self._unknown(f"expression_depth:{path}")
            return _UNKNOWN
        if isinstance(value, Mapping):
            string_map = {str(key): nested for key, nested in value.items()}
            expression_keys = {
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
            if not any(key in expression_keys for key in string_map):
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
                return self._resolve_item(item, string_map.get("$item"))
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
        if not self._consume(path):
            return False
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
            raw_value = self._resolve_item(item, expr.get("$item"))
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
            result = deep_equal_v5_3(raw_value, resolved("eq"))
        elif "neq" in expr:
            result = not deep_equal_v5_3(raw_value, resolved("neq"))
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
            resolved = self._resolve_item(item, match.group(1).strip())
            return kotlin_string_v5_3(resolved)

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
                return kotlin_string_v5_3(state_value)
            return kotlin_string_v5_3(self._resolve_item(item, raw))

        value = re.sub(r"\$\{([^}]+)\}", generic, value)
        if len(value) > self.config.max_string_expansion_length:
            self._unknown(f"string_truncated:{path}")
            value = value[: self.config.max_string_expansion_length]
        return value


class _EvidenceInterpreterV53(_EvidenceInterpreterV52):
    def __init__(
        self,
        spec: Mapping[str, Any],
        config: RewardConfigV53,
        computed_registry: ComputedFunctionRegistryV53 | None,
    ) -> None:
        super().__init__(spec, config, None)
        self.resolver = DynamicEvidenceResolverV53(
            self.state, config, computed_registry
        )
        self.work_units = 0
        self.evidence_bytes = 0
        self.block_ownership: list[dict[str, Any]] = []
        self.action_taxonomy: list[dict[str, Any]] = []
        self.media_taxonomy: list[dict[str, Any]] = []

    def _consume_work(self, path: str) -> bool:
        self.work_units += 1
        if self.work_units > self.config.max_repeat_work_units:
            self.truncated = True
            self.resolver._unknown(f"repeat_work_budget:{path}")
            return False
        return True

    def _collect_element(
        self,
        element_id: str,
        element: Mapping[str, Any],
        props: Mapping[str, Any],
    ) -> None:
        before_blocks = len(self.visible_blocks)
        before_actions = len(self.actions)
        before_media = len(self.media)
        super()._collect_element(element_id, element, props)
        element_type = (
            str(element.get("type") or "").casefold().replace("_", "")
        )
        if element_type == "chart":
            x_keys = [
                str(value)
                for value in (
                    [props.get("xKey")]
                    if props.get("xKey") is not None
                    else props.get("xFields") or props.get("x") or []
                )
                if value not in (None, "")
            ]
            y_keys = [
                str(value)
                for value in (
                    [props.get("yKey")]
                    if props.get("yKey") is not None
                    else props.get("yFields") or props.get("y") or []
                )
                if value not in (None, "")
            ]
            chart_keys = list(dict.fromkeys([*x_keys, *y_keys]))
            state_path = str(
                props.get("statePath") or props.get("dataPath") or ""
            )
            raw_rows = resolve_renderer_path_v5_2(
                self.state, _normalize_pointer(state_path)
            )
            rows = (
                [dict(row) for row in raw_rows if isinstance(row, Mapping)]
                if isinstance(raw_rows, list)
                else []
            )
            if chart_keys and rows:
                structured = _core.OutputTable(
                    element_id=element_id,
                    element_type="Chart",
                    headers=chart_keys,
                    keys=chart_keys,
                    rows=rows,
                )
                if self.charts:
                    self.charts[-1] = structured
                else:
                    self.charts.append(structured)
                self.visible_blocks.extend(chart_keys)
                self.visible_blocks.extend(
                    " | ".join(str(row.get(key, "")) for key in chart_keys)
                    for row in rows
                )
            charts = self.role_instances.get("chart") or []
            if (
                charts
                and charts[-1].get("component_id") == element_id
                and str(props.get("title") or props.get("label") or "").strip()
            ):
                charts[-1]["data_identity"] = str(
                    props.get("title") or props.get("label")
                ).strip()
        if element_type == "text":
            owner = "generic_visible_content"
        elif element_type == "table":
            owner = "source_table"
        elif element_type == "chart":
            owner = "source_chart"
        elif element_type == "button":
            owner = (
                "source_action"
                if any(
                    classify_action(item.action_type, item.url)
                    == "external_semantic"
                    for item in self.actions[before_actions:]
                )
                else "local_ui_mechanic"
            )
        elif element_type in {
            "formula",
            "codeblock",
            "consolelog",
            "emailpreview",
        }:
            owner = "semantic_role"
        elif element_type in {
            "textfield",
            "checkbox",
            "choicepicker",
            "slider",
            "datetimeinput",
        }:
            owner = "local_ui_mechanic"
        elif element_type in {"image", "icon", "video", "audioplayer"}:
            owner = "source_media"
        else:
            owner = "generic_visible_content"
        for block in self.visible_blocks[before_blocks:]:
            encoded = str(block).encode("utf-8")
            self.evidence_bytes += len(encoded)
            self.block_ownership.append(
                {
                    "component_id": element_id,
                    "owner": owner,
                    "text": str(block),
                }
            )
        if self.evidence_bytes > self.config.max_dynamic_evidence_bytes:
            self.truncated = True
            self.resolver._unknown("dynamic_evidence_byte_budget")

        for action in self.actions[before_actions:]:
            self.action_taxonomy.append(
                {
                    "component_id": action.element_id,
                    "action_type": action.action_type,
                    "target": action.url,
                    "category": classify_action(
                        action.action_type, action.url
                    ),
                }
            )
        decorative = bool(
            props.get("decorative")
            or props.get("ariaHidden")
            or (
                isinstance(props.get("accessibility"), Mapping)
                and props["accessibility"].get("decorative")
            )
        )
        for media_index in range(before_media, len(self.media)):
            item = self.media[media_index]
            item = replace(item, component_id=element_id)
            self.media[media_index] = item
            category = classify_media(
                item.kind,
                item.alt,
                decorative=decorative,
                renderer_asset=str(item.url).startswith(
                    ("asset://", "file:///android_asset/")
                ),
            )
            self.media_taxonomy.append(
                {
                    "component_id": element_id,
                    "kind": item.kind,
                    "url": item.url,
                    "category": category,
                }
            )

    def walk(
        self,
        element_id: str,
        *,
        item: Any = None,
        index: int | None = None,
        base_path: str | None = None,
        active_path: tuple[str, ...] = (),
    ) -> None:
        if not self._consume_work(f"elements.{element_id}"):
            return
        if element_id in active_path:
            self.resolver._unknown(f"cycle:{element_id}")
            return
        raw = self.elements.get(element_id)
        if not isinstance(raw, Mapping):
            self.resolver._unknown(f"missing_reference:{element_id}")
            return
        if not self.resolver.evaluate_condition(
            raw.get("visible"),
            item=item,
            index=index,
            base_path=base_path,
            path=f"elements.{element_id}.visible",
        ):
            return
        self.expanded_nodes += 1
        props = self.resolver.resolve(
            raw.get("props")
            if isinstance(raw.get("props"), Mapping)
            else {},
            item=item,
            index=index,
            base_path=base_path,
            path=f"elements.{element_id}.props",
        )
        props = self._clean_unknown(props)
        props = props if isinstance(props, Mapping) else {}
        resolved = dict(raw)
        resolved["props"] = dict(props)
        resolved["on"] = self._clean_unknown(
            self.resolver.resolve(
                raw.get("on") if isinstance(raw.get("on"), Mapping) else {},
                item=item,
                index=index,
                base_path=base_path,
                path=f"elements.{element_id}.on",
            )
        )
        self._collect_element(element_id, resolved, props)
        if self.truncated:
            return

        references = iter_renderer_references(raw)
        repeat = raw.get("repeat")
        if isinstance(repeat, Mapping):
            state_path = str(
                repeat.get("statePath") or repeat.get("path") or ""
            )
            items = resolve_renderer_path_v5_2(
                self.state, _normalize_pointer(state_path)
            )
            if not isinstance(items, list):
                self.resolver._unknown(
                    f"repeat_state_not_list:elements.{element_id}"
                )
                return
            for repeat_index, repeat_item in enumerate(items):
                repeat_base = _combine_pointer(
                    _normalize_pointer(state_path), str(repeat_index)
                )
                for reference in references:
                    self.walk(
                        reference.target_id,
                        item=repeat_item,
                        index=repeat_index,
                        base_path=repeat_base,
                        active_path=active_path + (element_id,),
                    )
                    if self.truncated:
                        return
            return
        for reference in references:
            self.walk(
                reference.target_id,
                item=item,
                index=index,
                base_path=base_path,
                active_path=active_path + (element_id,),
            )
            if self.truncated:
                return


@dataclass(frozen=True)
class V53EvidenceResult:
    output: _core.OutputEvidence
    output_tables: tuple[_core.OutputTable, ...]
    output_charts: tuple[_core.OutputTable, ...]
    structured_data_payloads: tuple[_core.OutputTable, ...]
    role_instances: dict[str, tuple[dict[str, Any], ...]]
    prepared_visible_blocks: tuple[PreparedTextBlock, ...]
    prepared_generic_visible_blocks: tuple[PreparedTextBlock, ...]
    generic_content_tokens: Counter[str]
    evidence_ownership: tuple[dict[str, Any], ...]
    action_taxonomy: tuple[dict[str, Any], ...]
    media_taxonomy: tuple[dict[str, Any], ...]
    source_semantic_actions: tuple[_core.OutputAction, ...]
    source_semantic_media: tuple[_core.MediaRef, ...]
    unknown_diagnostics: tuple[str, ...]
    dynamic_expression_unknown_count: int
    dynamic_evidence_complete: bool
    expanded_evidence_nodes: int
    truncated: bool
    dynamic_semantics: dict[str, Any]
    dynamic_evidence_certification: dict[str, Any]


def collect_output_evidence_v5_3(
    spec: Mapping[str, Any],
    audit: _core.GraphAudit,
    config: RewardConfigV53,
    *,
    computed_registry: ComputedFunctionRegistryV53 | None = None,
) -> V53EvidenceResult:
    interpreter = _EvidenceInterpreterV53(
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
    generic_blocks = [
        str(item["text"]).strip()
        for item in interpreter.block_ownership
        if item.get("owner") == "generic_visible_content"
        and str(item.get("text") or "").strip()
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
    complete = not unknown and not interpreter.truncated
    certification = {
        "complete": complete,
        "optimality_certified": complete,
        "work_units": interpreter.work_units,
        "expression_evaluations": (
            interpreter.resolver.expression_evaluations
        ),
        "evidence_bytes": interpreter.evidence_bytes,
        "lower_bound": {
            "visible_blocks": len(visible_blocks),
            "actions": len(interpreter.actions),
            "media": len(interpreter.media),
        },
        "upper_bound": (
            {
                "visible_blocks": len(visible_blocks),
                "actions": len(interpreter.actions),
                "media": len(interpreter.media),
            }
            if complete
            else None
        ),
        "diagnostic_codes": list(unknown),
    }
    dynamic_semantics = {
        "parity_version": DYNAMIC_SEMANTICS_VERSION,
        "parity_vector_version": DYNAMIC_PARITY_VECTOR_VERSION,
        "unknown_count": len(unknown),
        "unknown_codes": list(unknown),
        "complete": complete,
        "expanded_evidence_nodes": interpreter.expanded_nodes,
        "truncated": interpreter.truncated,
        "computed_registry": effective_registry_identity_v5_3(
            computed_registry
        ),
        "limits": {
            "max_repeat_work_units": config.max_repeat_work_units,
            "max_dynamic_evidence_bytes": (
                config.max_dynamic_evidence_bytes
            ),
            "max_expression_evaluations": (
                config.max_expression_evaluations
            ),
            "max_expression_depth": config.max_expression_depth,
            "max_string_expansion_length": (
                config.max_string_expansion_length
            ),
        },
    }
    semantic_actions = tuple(
        action
        for action in interpreter.actions
        if classify_action(action.action_type, action.url)
        == "external_semantic"
    )
    semantic_media = tuple(
        item
        for item, taxonomy in zip(
            interpreter.media, interpreter.media_taxonomy
        )
        if taxonomy.get("category") == "source_semantic_media"
    )
    return V53EvidenceResult(
        output=output,
        output_tables=tuple(interpreter.tables),
        output_charts=tuple(interpreter.charts),
        structured_data_payloads=tuple(structured),
        role_instances={
            role: tuple(values)
            for role, values in sorted(
                interpreter.role_instances.items()
            )
        },
        prepared_visible_blocks=tuple(
            prepare_text_block(value) for value in visible_blocks
        ),
        prepared_generic_visible_blocks=tuple(
            prepare_text_block(value) for value in generic_blocks
        ),
        generic_content_tokens=Counter(
            _core.tokenize(" ".join(generic_blocks))
        ),
        evidence_ownership=tuple(interpreter.block_ownership),
        action_taxonomy=tuple(interpreter.action_taxonomy),
        media_taxonomy=tuple(interpreter.media_taxonomy),
        source_semantic_actions=semantic_actions,
        source_semantic_media=semantic_media,
        unknown_diagnostics=unknown,
        dynamic_expression_unknown_count=len(unknown),
        dynamic_evidence_complete=complete,
        expanded_evidence_nodes=interpreter.expanded_nodes,
        truncated=interpreter.truncated,
        dynamic_semantics=dynamic_semantics,
        dynamic_evidence_certification=certification,
    )


__all__ = [
    "BUILTIN_COMPUTED_MANIFEST",
    "BUILTIN_COMPUTED_MANIFEST_HASH",
    "ComputedFunctionRegistryV53",
    "DEFAULT_COMPUTED_REGISTRY_V53",
    "DYNAMIC_PARITY_VECTOR_VERSION",
    "DYNAMIC_SEMANTICS_VERSION",
    "DynamicEvidenceResolverV53",
    "EVIDENCE_POLICY_VERSION",
    "V53EvidenceResult",
    "collect_output_evidence_v5_3",
    "deep_equal_v5_3",
    "effective_registry_identity_v5_3",
    "kotlin_string_v5_3",
]
