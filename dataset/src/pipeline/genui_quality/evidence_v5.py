"""Bounded renderer-aware evidence extraction for GenUI metric v5."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import re
from typing import Any, Mapping

from ..flat_spec_semantics import iter_renderer_references
from . import _core
from .config import RewardConfig
from .identity import sha256_json


_ITEM_TEMPLATE_RE = re.compile(r"\$\{\s*\$item(?:[./]([^}]+))?\s*\}")
_INDEX_TEMPLATE_RE = re.compile(r"\$\{\s*\$index\s*\}")
_STATE_TEMPLATE_RE = re.compile(r"\$\{\s*(/[^}]+)\s*\}")


@dataclass(frozen=True)
class V5EvidenceResult:
    output: _core.OutputEvidence
    role_signatures: dict[str, tuple[str, ...]]
    unknown_diagnostics: tuple[str, ...]
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
    current = value
    for part in _path_parts(path):
        if isinstance(current, Mapping):
            if part not in current:
                return None
            current = current[part]
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current


class _EvidenceInterpreter:
    def __init__(self, spec: Mapping[str, Any], config: RewardConfig) -> None:
        self.spec = spec
        raw_elements = spec.get("elements")
        self.elements: Mapping[str, Any] = (
            raw_elements if isinstance(raw_elements, Mapping) else {}
        )
        self.state = spec.get("state") if isinstance(spec.get("state"), Mapping) else {}
        self.config = config
        self.unknown: list[str] = []
        self.expanded_nodes = 0
        self.truncated = False
        self.visible_blocks: list[str] = []
        self.headings: list[str] = []
        self.heading_variants: list[str] = []
        self.tables: list[_core.OutputTable] = []
        self.actions: list[_core.OutputAction] = []
        self.media: list[_core.MediaRef] = []
        self.role_signatures: dict[str, list[str]] = {}

    def _unknown(self, code: str) -> None:
        if code not in self.unknown:
            self.unknown.append(code)

    def resolve(
        self,
        value: Any,
        *,
        item: Any,
        index: int | None,
        depth: int = 0,
        path: str = "value",
    ) -> Any:
        if depth > self.config.max_expression_depth:
            self._unknown(f"expression_depth:{path}")
            return None
        if isinstance(value, Mapping):
            if "$state" in value:
                pointer = value.get("$state")
                if not isinstance(pointer, str):
                    self._unknown(f"state_expression:{path}")
                    return None
                return _core.resolve_json_pointer(self.state, pointer)
            if "$item" in value:
                item_path = value.get("$item")
                if item_path in (None, True, "", "."):
                    return item
                if not isinstance(item_path, str):
                    self._unknown(f"item_expression:{path}")
                    return None
                return _resolve_local(item, item_path)
            if "$index" in value:
                return index
            if "$template" in value:
                template = value.get("$template")
                if not isinstance(template, str):
                    self._unknown(f"template_expression:{path}")
                    return None
                return self.resolve(
                    template,
                    item=item,
                    index=index,
                    depth=depth + 1,
                    path=path,
                )
            supported_condition = {"$eq", "$ne", "$and", "$or", "$not"}
            if any(key.startswith("$") and key not in supported_condition for key in value):
                self._unknown(f"unsupported_expression:{path}")
            return {
                str(key): self.resolve(
                    nested,
                    item=item,
                    index=index,
                    depth=depth + 1,
                    path=f"{path}.{key}",
                )
                for key, nested in value.items()
            }
        if isinstance(value, list):
            return [
                self.resolve(
                    nested,
                    item=item,
                    index=index,
                    depth=depth + 1,
                    path=f"{path}[{position}]",
                )
                for position, nested in enumerate(value)
            ]
        if not isinstance(value, str):
            return value

        if value == "$item":
            return item
        if value.startswith("$item.") or value.startswith("$item/"):
            return _resolve_local(item, value[len("$item") :])
        if value == "$index":
            return index

        def item_repl(match: re.Match[str]) -> str:
            resolved = item if not match.group(1) else _resolve_local(item, match.group(1))
            return "" if resolved is None else str(resolved)

        def state_repl(match: re.Match[str]) -> str:
            resolved = _core.resolve_json_pointer(self.state, match.group(1))
            return "" if resolved is None else str(resolved)

        rendered = _ITEM_TEMPLATE_RE.sub(item_repl, value)
        rendered = _INDEX_TEMPLATE_RE.sub("" if index is None else str(index), rendered)
        rendered = _STATE_TEMPLATE_RE.sub(state_repl, rendered)
        if len(rendered) > self.config.max_string_expansion_length:
            self._unknown(f"string_truncated:{path}")
            rendered = rendered[: self.config.max_string_expansion_length]
        if "${" in rendered or "$item" in rendered or "$state" in rendered:
            self._unknown(f"unsupported_template:{path}")
        return rendered

    def visible(
        self,
        value: Any,
        *,
        item: Any,
        index: int | None,
        path: str,
    ) -> bool | None:
        if value is None:
            return True
        if isinstance(value, bool):
            return value
        if isinstance(value, Mapping):
            if "$and" in value:
                raw = value.get("$and")
                if not isinstance(raw, list):
                    self._unknown(f"visibility_and:{path}")
                    return None
                resolved = [
                    self.visible(child, item=item, index=index, path=f"{path}.$and")
                    for child in raw
                ]
                return None if any(child is None for child in resolved) else all(resolved)
            if "$or" in value:
                raw = value.get("$or")
                if not isinstance(raw, list):
                    self._unknown(f"visibility_or:{path}")
                    return None
                resolved = [
                    self.visible(child, item=item, index=index, path=f"{path}.$or")
                    for child in raw
                ]
                return None if any(child is None for child in resolved) else any(resolved)
            if "$not" in value:
                resolved = self.visible(
                    value.get("$not"),
                    item=item,
                    index=index,
                    path=f"{path}.$not",
                )
                return None if resolved is None else not resolved
            for operator in ("$eq", "$ne"):
                if operator in value:
                    operands = value.get(operator)
                    if not isinstance(operands, list) or len(operands) != 2:
                        self._unknown(f"visibility_compare:{path}")
                        return None
                    left = self.resolve(
                        operands[0], item=item, index=index, path=f"{path}.{operator}[0]"
                    )
                    right = self.resolve(
                        operands[1], item=item, index=index, path=f"{path}.{operator}[1]"
                    )
                    return left == right if operator == "$eq" else left != right
        resolved = self.resolve(value, item=item, index=index, path=path)
        if isinstance(resolved, (bool, int, float, str, list, Mapping)):
            if isinstance(resolved, str) and resolved.casefold() in {"false", "0", "none", "null"}:
                return False
            return bool(resolved)
        self._unknown(f"visibility_unknown:{path}")
        return None

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
        values: list[str] = []
        for key in keys_by_type.get(element_type, ()):
            value = props.get(key)
            if isinstance(value, (str, int, float)) and str(value).strip():
                values.append(str(value).strip())
        return values

    def _add_signature(self, role: str, payload: Any) -> None:
        signature = sha256_json(payload)
        values = self.role_signatures.setdefault(role, [])
        if signature not in values:
            values.append(signature)

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
        if element_type == "text" and variant in {"h1", "h2", "h3", "h4", "title", "heading"}:
            if text_values:
                self.headings.append(text_values[0])
                self.heading_variants.append(variant)

        temp_element = dict(element)
        temp_element["props"] = dict(props)
        table: _core.OutputTable | None = None
        if element_type in {"table", "chart"}:
            table = _core._table_from_element(  # type: ignore[attr-defined]
                element_id,
                temp_element,
                self.state,
            )
            if table is not None:
                self.tables.append(table)
                for header in table.headers:
                    if header:
                        self.visible_blocks.append(header)
                for row in table.rows:
                    block = " | ".join(str(row.get(key, "")) for key in table.keys)
                    if block.strip(" |"):
                        self.visible_blocks.append(block)

        self.actions.extend(
            _core._output_actions(  # type: ignore[attr-defined]
                element_id,
                temp_element,
                table,
            )
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
            self.media.append(_core.MediaRef(kind=kind, url=_core.normalize_url(url), alt=alt))

        role = {
            "table": "table",
            "chart": "chart",
            "formula": "formula",
            "codeblock": "code",
            "consolelog": "console",
            "emailpreview": "email",
            "image": "image",
            "video": "video",
            "audioplayer": "audio",
        }.get(element_type)
        if role is not None:
            signature_payload: Any
            if table is not None:
                signature_payload = {
                    "type": element_type,
                    "title": props.get("title"),
                    "headers": table.headers,
                    "rows": table.rows,
                    "xKey": props.get("xKey"),
                    "yKey": props.get("yKey"),
                }
            else:
                signature_payload = {
                    "type": element_type,
                    "content": text_values,
                    "url": props.get("url") or props.get("src"),
                }
            self._add_signature(role, signature_payload)

    def walk(
        self,
        element_id: str,
        *,
        item: Any = None,
        index: int | None = None,
        active_path: tuple[str, ...] = (),
    ) -> None:
        if self.expanded_nodes >= self.config.max_expanded_evidence_nodes:
            self.truncated = True
            self._unknown("expanded_evidence_node_limit")
            return
        if element_id in active_path:
            self._unknown(f"cycle:{element_id}")
            return
        raw = self.elements.get(element_id)
        if not isinstance(raw, Mapping):
            self._unknown(f"missing_reference:{element_id}")
            return

        visible = self.visible(
            raw.get("visible"),
            item=item,
            index=index,
            path=f"elements.{element_id}.visible",
        )
        if visible is None:
            self._unknown(f"visibility_unknown:elements.{element_id}")
            return
        if not visible:
            return

        self.expanded_nodes += 1
        props_raw = raw.get("props")
        props = self.resolve(
            props_raw if isinstance(props_raw, Mapping) else {},
            item=item,
            index=index,
            path=f"elements.{element_id}.props",
        )
        props = props if isinstance(props, Mapping) else {}
        resolved_element = dict(raw)
        resolved_element["props"] = dict(props)
        resolved_element["on"] = self.resolve(
            raw.get("on") if isinstance(raw.get("on"), Mapping) else {},
            item=item,
            index=index,
            path=f"elements.{element_id}.on",
        )
        self._collect_element(element_id, resolved_element, props)

        references = iter_renderer_references(raw)
        repeat = raw.get("repeat")
        if isinstance(repeat, Mapping):
            state_path = repeat.get("statePath") or repeat.get("path")
            items = _core.resolve_json_pointer(self.state, state_path)
            if not isinstance(items, list):
                self._unknown(f"repeat_state_not_list:elements.{element_id}")
                return
            if len(items) > self.config.max_repeat_items:
                self.truncated = True
                self._unknown(f"repeat_item_limit:elements.{element_id}")
            for repeat_index, repeat_item in enumerate(items[: self.config.max_repeat_items]):
                for reference in references:
                    self.walk(
                        reference.target_id,
                        item=repeat_item,
                        index=repeat_index,
                        active_path=active_path + (element_id,),
                    )
            return

        for reference in references:
            self.walk(
                reference.target_id,
                item=item,
                index=index,
                active_path=active_path + (element_id,),
            )


def collect_output_evidence_v5(
    spec: Mapping[str, Any],
    audit: _core.GraphAudit,
    config: RewardConfig,
) -> V5EvidenceResult:
    interpreter = _EvidenceInterpreter(spec, config)
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

    output = _core.OutputEvidence(
        elements=static.elements,
        reachable_ids=static.reachable_ids,
        visible_blocks=visible_blocks,
        headings=interpreter.headings,
        heading_variants=interpreter.heading_variants,
        tables=interpreter.tables,
        actions=interpreter.actions,
        media=interpreter.media,
        type_counts=static.type_counts,
        content_tokens=content_tokens,
        exact_values=exact_values,
        valid_type_counts=static.valid_type_counts,
    )
    return V5EvidenceResult(
        output=output,
        role_signatures={
            role: tuple(signatures)
            for role, signatures in sorted(interpreter.role_signatures.items())
        },
        unknown_diagnostics=tuple(interpreter.unknown),
        expanded_evidence_nodes=interpreter.expanded_nodes,
        truncated=interpreter.truncated,
    )


__all__ = ["V5EvidenceResult", "collect_output_evidence_v5"]
