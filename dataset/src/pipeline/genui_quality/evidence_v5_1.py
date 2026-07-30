"""Android-compatible bounded renderer evidence for metric v5.1."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any, Callable, Mapping

from ..flat_spec_semantics import iter_renderer_references
from . import _core
from .config import RewardConfig
from .matching_v5_1 import PreparedTextBlock, prepare_text_block
from .metrics_v5_1 import semantic_signature


EVIDENCE_POLICY_VERSION = "5.1.0"
DYNAMIC_SEMANTICS_VERSION = "5.1.0"
ComputedFunctionRegistry = Mapping[str, Callable[[Mapping[str, Any]], Any]]
_UNKNOWN = object()
_EXPRESSION_KEYS = {
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


@dataclass(frozen=True)
class V51EvidenceResult:
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


def _path_parts(path: str) -> list[str]:
    value = str(path or "").strip()
    if value.startswith("$."):
        value = value[2:]
    if value.startswith("/"):
        value = value[1:]
    return [part for part in re.split(r"[./]", value) if part]


def _resolve_local(value: Any, path: str) -> Any:
    original = value
    current = value
    normalized = str(path or "").strip()
    if not normalized:
        return current
    for part in _path_parts(normalized):
        if isinstance(current, Mapping):
            if part not in current:
                return original if normalized == "value" else None
            current = current[part]
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return original if normalized == "value" else None
        else:
            return original if normalized == "value" else None
    return current


def _normalize_pointer(path: Any) -> str:
    value = str(path or "").strip()
    if not value:
        return ""
    return value if value.startswith("/") else "/" + value


def _encode_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _combine_pointer(base: str | None, token: str) -> str | None:
    if not base:
        return None
    return (
        _normalize_pointer(base).rstrip("/")
        + "/"
        + _encode_pointer_token(str(token))
    )


def _to_double(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
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


class DynamicEvidenceResolver:
    """Port of the deterministic subset in Android ``FlatExprResolver``."""

    def __init__(
        self,
        state: Mapping[str, Any],
        config: RewardConfig,
        computed_functions: ComputedFunctionRegistry | None = None,
    ) -> None:
        self.state = state
        self.config = config
        self.computed_functions = computed_functions or {}
        self.unknown: list[str] = []

    def _unknown(self, code: str) -> None:
        if code not in self.unknown:
            self.unknown.append(code)

    def _bind_item(
        self,
        raw_path: Any,
        base_path: str | None,
    ) -> Any:
        if not base_path:
            return None
        requested = str(raw_path or "").strip()
        pointer = (
            _normalize_pointer(base_path)
            if not requested
            else _combine_pointer(base_path, requested)
        )
        return _core.resolve_json_pointer(self.state, pointer)

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
            has_expression = any(key in _EXPRESSION_KEYS for key in string_map)
            if not has_expression:
                for key in string_map:
                    if key.startswith("$"):
                        self._unknown(f"unsupported_expression_key:{key}")
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
                return _resolve_local(item, str(string_map.get("$item") or ""))
            if "$state" in string_map:
                return _core.resolve_json_pointer(
                    self.state, str(string_map.get("$state") or "")
                )
            if "$bindState" in string_map:
                return _core.resolve_json_pointer(
                    self.state, str(string_map.get("$bindState") or "")
                )
            if "$bindItem" in string_map:
                return self._bind_item(string_map.get("$bindItem"), base_path)
            if "$index" in string_map:
                return index
            if "$cond" in string_map:
                branch = "$then" if self.evaluate_condition(
                    string_map.get("$cond"),
                    item=item,
                    index=index,
                    base_path=base_path,
                    path=f"{path}.$cond",
                ) else "$else"
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
            return string_map
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
                value,
                item=item,
                index=index,
                path=path,
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
            raw_value = _core.resolve_json_pointer(
                self.state, str(expr.get("$state") or "")
            )
        elif "$item" in expr:
            raw_value = _resolve_local(item, str(expr.get("$item") or ""))
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
            result = raw_value == resolved("eq")
        elif "neq" in expr:
            result = raw_value != resolved("neq")
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

        def item_repl(match: re.Match[str]) -> str:
            resolved = _resolve_local(item, match.group(1).strip())
            return "" if resolved is None else str(resolved)

        value = re.sub(
            r"\$\{\s*\$item[./]([^}]+?)\s*\}",
            item_repl,
            value,
        )
        value = re.sub(
            r"\{\{\s*\$item[./]([^}]+?)\s*\}\}",
            item_repl,
            value,
        )
        value = re.sub(
            r"(?<!\$)\{\s*\$item[./]([^}]+?)\s*\}",
            item_repl,
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
            state_value = _core.resolve_json_pointer(
                self.state, _normalize_pointer(raw)
            )
            if state_value is not None:
                return str(state_value)
            item_value = _resolve_local(item, raw)
            return "" if item_value is None else str(item_value)

        value = re.sub(r"\$\{([^}]+)\}", generic, value)
        if len(value) > self.config.max_string_expansion_length:
            self._unknown(f"string_truncated:{path}")
            value = value[: self.config.max_string_expansion_length]
        return value


class _EvidenceInterpreter:
    def __init__(
        self,
        spec: Mapping[str, Any],
        config: RewardConfig,
        computed_functions: ComputedFunctionRegistry | None,
    ) -> None:
        self.spec = spec
        raw_elements = spec.get("elements")
        self.elements: Mapping[str, Any] = (
            raw_elements if isinstance(raw_elements, Mapping) else {}
        )
        self.state = spec.get("state") if isinstance(spec.get("state"), Mapping) else {}
        self.config = config
        self.resolver = DynamicEvidenceResolver(
            self.state, config, computed_functions
        )
        self.expanded_nodes = 0
        self.truncated = False
        self.visible_blocks: list[str] = []
        self.headings: list[str] = []
        self.heading_variants: list[str] = []
        self.tables: list[_core.OutputTable] = []
        self.charts: list[_core.OutputTable] = []
        self.actions: list[_core.OutputAction] = []
        self.media: list[_core.MediaRef] = []
        self.role_instances: dict[str, list[dict[str, Any]]] = {}

    def _add_role(self, role: str, payload: Mapping[str, Any]) -> None:
        item = dict(payload)
        item["signature"] = semantic_signature(
            {key: value for key, value in item.items() if key != "component_id"}
        )
        self.role_instances.setdefault(role, []).append(item)

    @staticmethod
    def _text_values(element_type: str, props: Mapping[str, Any]) -> list[str]:
        keys_by_type = {
            "text": ("text", "content", "value", "label", "title"),
            "formula": ("text", "latex", "formula", "title"),
            "codeblock": ("code", "text", "language", "title"),
            "consolelog": ("code", "text", "output", "title"),
            "emailpreview": ("subject", "title", "to", "from", "body"),
            "button": ("label", "title", "text"),
            "image": ("alt", "description", "title"),
            "video": ("alt", "description", "title"),
            "audioplayer": ("alt", "description", "title"),
            "textfield": ("label", "placeholder", "value"),
            "checkbox": ("label",),
            "choicepicker": ("label",),
            "slider": ("label", "value"),
            "datetimeinput": ("label", "value"),
            "chart": ("title", "label", "description"),
        }
        return [
            str(props[key]).strip()
            for key in keys_by_type.get(element_type, ())
            if isinstance(props.get(key), (str, int, float))
            and str(props.get(key)).strip()
        ]

    def _collect_element(
        self,
        element_id: str,
        element: Mapping[str, Any],
        props: Mapping[str, Any],
    ) -> None:
        element_type = str(element.get("type") or "").casefold().replace("_", "")
        text_values = self._text_values(element_type, props)
        self.visible_blocks.extend(text_values)
        variant = str(props.get("variant") or "").casefold()
        if element_type == "text" and variant in {
            "h1", "h2", "h3", "h4", "title", "heading"
        } and text_values:
            self.headings.append(text_values[0])
            self.heading_variants.append(variant)

        temp = dict(element)
        temp["props"] = dict(props)
        structured: _core.OutputTable | None = None
        if element_type in {"table", "chart"}:
            structured = _core._table_from_element(  # type: ignore[attr-defined]
                element_id, temp, self.state
            )
            if structured is not None:
                if element_type == "table":
                    self.tables.append(structured)
                else:
                    self.charts.append(structured)
                self.visible_blocks.extend(
                    header for header in structured.headers if header
                )
                for row in structured.rows:
                    block = " | ".join(
                        str(row.get(key, "")) for key in structured.keys
                    )
                    if block.strip(" |"):
                        self.visible_blocks.append(block)
        self.actions.extend(
            _core._output_actions(element_id, temp, structured)  # type: ignore[attr-defined]
        )

        if element_type in {"image", "icon", "video", "audioplayer"}:
            kind = {
                "image": "Image",
                "icon": "Icon",
                "video": "Video",
                "audioplayer": "AudioPlayer",
            }[element_type]
            url = str(
                props.get("url")
                or props.get("src")
                or props.get("name")
                or props.get("source")
                or ""
            ).strip()
            alt = str(
                props.get("alt")
                or props.get("description")
                or props.get("accessibilityLabel")
                or ""
            ).strip()
            self.media.append(
                _core.MediaRef(kind=kind, url=_core.normalize_url(url), alt=alt)
            )

        if element_type == "chart":
            self._add_role(
                "chart",
                {
                    "component_id": element_id,
                    "title": str(props.get("title") or props.get("label") or ""),
                    "chart_type": str(
                        props.get("chartType") or props.get("type") or ""
                    ),
                    "x_fields": [
                        str(value)
                        for value in (
                            [props.get("xKey")]
                            if props.get("xKey") is not None
                            else props.get("xFields") or props.get("x") or []
                        )
                        if value not in (None, "")
                    ],
                    "y_fields": [
                        str(value)
                        for value in (
                            [props.get("yKey")]
                            if props.get("yKey") is not None
                            else props.get("yFields") or props.get("y") or []
                        )
                        if value not in (None, "")
                    ],
                    "series": [
                        str(value)
                        for value in (
                            props.get("series")
                            if isinstance(props.get("series"), list)
                            else [props.get("series")]
                        )
                        if value not in (None, "")
                    ],
                    "data_identity": str(
                        props.get("statePath") or props.get("dataPath") or ""
                    ),
                },
            )
        elif element_type in {"formula", "codeblock", "consolelog", "emailpreview"}:
            role = {
                "formula": "formula",
                "codeblock": "code",
                "consolelog": "console",
                "emailpreview": "email",
            }[element_type]
            self._add_role(
                role,
                {
                    "component_id": element_id,
                    "title": str(props.get("title") or props.get("subject") or ""),
                    "content": " ".join(text_values),
                    "language": str(props.get("language") or ""),
                    "subject": str(props.get("subject") or ""),
                    "to": str(props.get("to") or ""),
                    "from": str(props.get("from") or ""),
                },
            )

    @staticmethod
    def _clean_unknown(value: Any) -> Any:
        if value is _UNKNOWN:
            return None
        if isinstance(value, Mapping):
            return {
                str(key): _EvidenceInterpreter._clean_unknown(nested)
                for key, nested in value.items()
            }
        if isinstance(value, list):
            return [_EvidenceInterpreter._clean_unknown(nested) for nested in value]
        return value

    def walk(
        self,
        element_id: str,
        *,
        item: Any = None,
        index: int | None = None,
        base_path: str | None = None,
        active_path: tuple[str, ...] = (),
    ) -> None:
        if self.expanded_nodes >= self.config.max_expanded_evidence_nodes:
            self.truncated = True
            self.resolver._unknown("expanded_evidence_node_limit")
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
            raw.get("props") if isinstance(raw.get("props"), Mapping) else {},
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

        references = iter_renderer_references(raw)
        repeat = raw.get("repeat")
        if isinstance(repeat, Mapping):
            state_path = str(repeat.get("statePath") or repeat.get("path") or "")
            items = _core.resolve_json_pointer(
                self.state, _normalize_pointer(state_path)
            )
            if not isinstance(items, list):
                self.resolver._unknown(
                    f"repeat_state_not_list:elements.{element_id}"
                )
                return
            if len(items) > self.config.max_repeat_items:
                self.truncated = True
                self.resolver._unknown(
                    f"repeat_item_limit:elements.{element_id}"
                )
            for repeat_index, repeat_item in enumerate(
                items[: self.config.max_repeat_items]
            ):
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
            return
        for reference in references:
            self.walk(
                reference.target_id,
                item=item,
                index=index,
                base_path=base_path,
                active_path=active_path + (element_id,),
            )


def collect_output_evidence_v5_1(
    spec: Mapping[str, Any],
    audit: _core.GraphAudit,
    config: RewardConfig,
    *,
    computed_functions: ComputedFunctionRegistry | None = None,
) -> V51EvidenceResult:
    interpreter = _EvidenceInterpreter(spec, config, computed_functions)
    root = spec.get("root")
    if isinstance(root, str):
        interpreter.walk(root)
    static = _core.collect_output_evidence(spec, audit)
    visible_blocks = [
        value
        for value in (str(item).strip() for item in interpreter.visible_blocks)
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
    return V51EvidenceResult(
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
    )


__all__ = [
    "ComputedFunctionRegistry",
    "DYNAMIC_SEMANTICS_VERSION",
    "DynamicEvidenceResolver",
    "EVIDENCE_POLICY_VERSION",
    "V51EvidenceResult",
    "collect_output_evidence_v5_1",
]
