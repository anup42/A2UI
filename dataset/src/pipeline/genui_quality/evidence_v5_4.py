"""Android-effective dynamic and component evidence for metric v5.4."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import inspect
import json
import math
import types
from typing import Any, Callable, Mapping

from ..renderer_effective_semantics_v5_4 import (
    canonical_media_kind,
    effective_chart,
    effective_media,
    effective_table,
    output_table_from_chart,
    output_table_from_effective,
)
from . import _core
from .applicability_v5_3 import classify_action, classify_media
from .config_v5_3 import RewardConfigV53
from .evidence_v5_2 import (
    ComputedFunction,
    ComputedFunctionRegistryV52,
    _UNKNOWN,
    _sha256_json,
)
from .evidence_v5_3 import (
    DynamicEvidenceResolverV53,
    _EvidenceInterpreterV53,
    deep_equal_v5_3,
    kotlin_string_v5_3,
)
from .matching_v5_1 import PreparedTextBlock, prepare_text_block


EVIDENCE_POLICY_VERSION = "5.4.0"
DYNAMIC_SEMANTICS_VERSION = "a2ui-express-native-semantics-5.4.0"
DYNAMIC_PARITY_VECTOR_VERSION = "3.0.0"
COMPUTED_REGISTRY_IDENTITY_VERSION = "2.0.0"

deep_equal_v5_4 = deep_equal_v5_3
kotlin_string_v5_4 = kotlin_string_v5_3


def _behavior_value(value: Any, seen: set[int]) -> Any:
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
            "bytecode": value.co_code.hex(),
            "constants": [_behavior_value(item, seen) for item in value.co_consts],
            "names": list(value.co_names),
            "varnames": list(value.co_varnames),
            "freevars": list(value.co_freevars),
            "cellvars": list(value.co_cellvars),
        }
    if isinstance(value, type):
        return {"type": f"{value.__module__}.{value.__qualname__}"}
    if isinstance(value, types.ModuleType):
        raise ValueError(
            "computed functions may not depend on module globals; bind an "
            "immutable value explicitly"
        )
    identity = id(value)
    if identity in seen:
        return {"cycle": f"{type(value).__module__}.{type(value).__qualname__}"}
    seen.add(identity)
    try:
        if isinstance(value, tuple):
            return {"tuple": [_behavior_value(item, seen) for item in value]}
        if isinstance(value, frozenset):
            return {
                "frozenset": sorted(
                    (_behavior_value(item, seen) for item in value),
                    key=lambda item: json.dumps(item, sort_keys=True, default=str),
                )
            }
        if isinstance(value, (list, dict, set)):
            raise ValueError(
                "computed function behavior depends on a mutable global/nonlocal"
            )
        if callable(value):
            return _callable_behavior_payload_v5_4(value, seen=seen)
        attributes = getattr(value, "__dict__", None)
        if attributes:
            raise ValueError(
                "computed function behavior depends on mutable object state"
            )
        return {"object_type": f"{type(value).__module__}.{type(value).__qualname__}"}
    finally:
        seen.remove(identity)


def _callable_behavior_payload_v5_4(
    function: ComputedFunction,
    *,
    seen: set[int] | None = None,
) -> dict[str, Any]:
    active = seen if seen is not None else set()
    code = getattr(function, "__code__", None)
    payload: dict[str, Any] = {
        "module": str(getattr(function, "__module__", "")),
        "qualname": str(getattr(function, "__qualname__", "")),
    }
    if code is not None:
        payload["code"] = _behavior_value(code, active)
        payload["defaults"] = _behavior_value(
            getattr(function, "__defaults__", None), active
        )
        kwdefaults = getattr(function, "__kwdefaults__", None)
        payload["kwdefaults"] = (
            {
                str(name): _behavior_value(value, active)
                for name, value in sorted(kwdefaults.items())
            }
            if isinstance(kwdefaults, Mapping)
            else None
        )
        closure = inspect.getclosurevars(function)
        payload["referenced_globals"] = {
            name: _behavior_value(value, active)
            for name, value in sorted(closure.globals.items())
        }
        payload["referenced_nonlocals"] = {
            name: _behavior_value(value, active)
            for name, value in sorted(closure.nonlocals.items())
        }
        payload["unbound"] = sorted(closure.unbound)
    else:
        try:
            payload["source"] = inspect.getsource(function)
        except (OSError, TypeError):
            payload["type"] = (
                f"{type(function).__module__}.{type(function).__qualname__}"
            )
    bound = getattr(function, "__self__", None)
    if bound is not None:
        payload["bound_self"] = _behavior_value(bound, active)
    return payload


def callable_behavior_hash_v5_4(function: ComputedFunction) -> str:
    return _sha256_json(_callable_behavior_payload_v5_4(function))


@dataclass(frozen=True)
class ComputedFunctionRegistryV54(ComputedFunctionRegistryV52):
    def identity(self) -> dict[str, Any]:
        computed = {
            str(name): callable_behavior_hash_v5_4(function)
            for name, function in sorted(self.functions.items())
        }
        declared = str(self.manifest_hash)
        return {
            "registry_id": self.registry_id,
            "registry_version": self.registry_version,
            "manifest_hash": declared,
            "identity_policy_version": COMPUTED_REGISTRY_IDENTITY_VERSION,
            "function_behavior_hashes": computed,
            "computed_behavior_manifest_hash": _sha256_json(computed),
            "declared_manifest_verified": bool(declared),
        }

    def identity_hash(self) -> str:
        return _sha256_json(self.identity())


def _builtin_concat(args: Mapping[str, Any]) -> str:
    return "".join(kotlin_string_v5_4(value) for value in args.values())


def _builtin_uppercase(args: Mapping[str, Any]) -> str:
    return kotlin_string_v5_4(args.get("value")).upper()


def _builtin_lowercase(args: Mapping[str, Any]) -> str:
    return kotlin_string_v5_4(args.get("value")).lower()


def _builtin_coalesce(args: Mapping[str, Any]) -> Any:
    values = args.get("values")
    if not isinstance(values, list):
        return None
    return next(
        (
            value
            for value in values
            if value is not None
            and not (isinstance(value, str) and not value.strip())
        ),
        None,
    )


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


_BUILTINS: dict[str, Callable[[Mapping[str, Any]], Any]] = {
    "concat": _builtin_concat,
    "uppercase": _builtin_uppercase,
    "lowercase": _builtin_lowercase,
    "coalesce": _builtin_coalesce,
    "sum": _builtin_sum,
}
BUILTIN_COMPUTED_MANIFEST = {
    "renderer": "FlatSpecRenderer.kt",
    "semantics_version": DYNAMIC_SEMANTICS_VERSION,
    "identity_policy": COMPUTED_REGISTRY_IDENTITY_VERSION,
    "functions": sorted(_BUILTINS),
}
BUILTIN_COMPUTED_MANIFEST_HASH = _sha256_json(BUILTIN_COMPUTED_MANIFEST)
DEFAULT_COMPUTED_REGISTRY_V54 = ComputedFunctionRegistryV54(
    registry_id="a2ui-express-native-builtins",
    registry_version="3.0.0",
    functions=_BUILTINS,
    manifest_hash=BUILTIN_COMPUTED_MANIFEST_HASH,
)


def effective_registry_identity_v5_4(
    custom: ComputedFunctionRegistryV54 | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "identity_policy_version": COMPUTED_REGISTRY_IDENTITY_VERSION,
        "builtin": DEFAULT_COMPUTED_REGISTRY_V54.identity(),
        "builtin_identity_hash": DEFAULT_COMPUTED_REGISTRY_V54.identity_hash(),
    }
    if custom is not None:
        payload["custom"] = custom.identity()
        payload["custom_identity_hash"] = custom.identity_hash()
    return payload


class DynamicEvidenceResolverV54(DynamicEvidenceResolverV53):
    def __init__(
        self,
        state: Mapping[str, Any],
        config: RewardConfigV53,
        computed_registry: ComputedFunctionRegistryV54 | None = None,
    ) -> None:
        super().__init__(state, config, None)
        self.computed_registry = computed_registry
        self.computed_functions = dict(_BUILTINS)
        if computed_registry is not None:
            self.computed_functions.update(computed_registry.functions)


class _EvidenceInterpreterV54(_EvidenceInterpreterV53):
    def __init__(
        self,
        spec: Mapping[str, Any],
        config: RewardConfigV53,
        computed_registry: ComputedFunctionRegistryV54 | None,
    ) -> None:
        super().__init__(spec, config, None)
        self.resolver = DynamicEvidenceResolverV54(
            self.state, config, computed_registry
        )
        self.effective_components: list[dict[str, Any]] = []

    def _collect_element(
        self,
        element_id: str,
        element: Mapping[str, Any],
        props: Mapping[str, Any],
    ) -> None:
        before = {
            "visible": len(self.visible_blocks),
            "tables": len(self.tables),
            "charts": len(self.charts),
            "actions": len(self.actions),
            "media": len(self.media),
            "ownership": len(self.block_ownership),
            "action_taxonomy": len(self.action_taxonomy),
            "media_taxonomy": len(self.media_taxonomy),
            "chart_roles": len(self.role_instances.get("chart", ())),
        }
        super()._collect_element(element_id, element, props)
        token = str(element.get("type") or "").casefold().replace("_", "")
        semantic_owner = {
            "formula": "source_formula",
            "codeblock": "source_code",
            "consolelog": "source_console",
            "emailpreview": "source_email",
        }.get(token)
        if semantic_owner is not None:
            for item in self.block_ownership[before["ownership"] :]:
                item["owner"] = semantic_owner
        if token in {"table", "chart"}:
            del self.visible_blocks[before["visible"] :]
            del self.tables[before["tables"] :]
            del self.charts[before["charts"] :]
            del self.actions[before["actions"] :]
            del self.block_ownership[before["ownership"] :]
            del self.action_taxonomy[before["action_taxonomy"] :]
            if token == "chart":
                roles = self.role_instances.get("chart", [])
                del roles[before["chart_roles"] :]

            structured: _core.OutputTable | None
            owner: str
            component_detail: dict[str, Any]
            if token == "table":
                model = effective_table(props, self.state)
                structured = output_table_from_effective(
                    element_id, "Table", model
                )
                self.tables.append(structured)
                owner = "source_table"
                component_detail = {
                    "component_id": element_id,
                    "kind": "table",
                    "columns": [
                        {"key": item.key, "label": item.label}
                        for item in model.columns
                    ],
                    "rows": [list(row) for row in model.rows],
                    "complete": model.complete,
                    "diagnostics": list(model.diagnostics),
                }
            else:
                chart = effective_chart(props, self.state)
                structured = output_table_from_chart(element_id, chart)
                if structured is not None:
                    self.charts.append(structured)
                owner = "source_chart"
                component_detail = {
                    "component_id": element_id,
                    "kind": "chart",
                    "columns": [
                        {"key": item.key, "label": item.label}
                        for item in chart.columns
                    ],
                    "x_labels": list(chart.x_labels),
                    "y_values": list(chart.y_values),
                    "y_display_values": list(chart.y_display_values),
                    "dataset_identity": chart.dataset_identity,
                    "complete": chart.complete,
                    "diagnostics": list(chart.diagnostics),
                }
                x_fields: list[str] = []
                y_fields: list[str] = []
                if chart.x_index is not None:
                    column = chart.columns[chart.x_index]
                    x_fields = list(dict.fromkeys([column.key, column.label]))
                if chart.y_index is not None:
                    column = chart.columns[chart.y_index]
                    y_fields = list(dict.fromkeys([column.key, column.label]))
                self._add_role(
                    "chart",
                    {
                        "component_id": element_id,
                        "title": chart.title,
                        "chart_type": chart.chart_type,
                        "x_fields": x_fields,
                        "y_fields": y_fields,
                        "x_labels": list(chart.x_labels),
                        "y_values": list(chart.y_display_values),
                        "column_identities": [
                            column.key for column in chart.columns
                        ],
                        "data_identity": chart.dataset_identity,
                    },
                )
                if chart.title:
                    self.visible_blocks.append(chart.title)

            if structured is not None:
                self.visible_blocks.extend(
                    value for value in structured.headers if value
                )
                self.visible_blocks.extend(
                    " | ".join(str(row.get(key, "")) for key in structured.keys)
                    for row in structured.rows
                    if any(str(row.get(key, "")).strip() for key in structured.keys)
                )
            temp = dict(element)
            temp["props"] = dict(props)
            self.actions.extend(
                _core._output_actions(  # type: ignore[attr-defined]
                    element_id, temp, structured
                )
            )
            for block in self.visible_blocks[before["visible"] :]:
                self.block_ownership.append(
                    {
                        "component_id": element_id,
                        "owner": owner,
                        "text": str(block),
                    }
                )
            for action in self.actions[before["actions"] :]:
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
            self.effective_components.append(component_detail)

        if token in {"image", "icon", "video", "audioplayer"}:
            del self.media[before["media"] :]
            del self.media_taxonomy[before["media_taxonomy"] :]
            model = effective_media(token, props)
            alt = str(
                props.get("alt")
                or props.get("description")
                or props.get("accessibilityLabel")
                or ""
            ).strip()
            item = _core.MediaRef(
                kind={
                    "image": "Image",
                    "icon": "Icon",
                    "video": "Video",
                    "audio": "AudioPlayer",
                }.get(canonical_media_kind(token), str(element.get("type") or "")),
                url=_core.normalize_url(model.url),
                alt=alt,
                component_id=element_id,
            )
            self.media.append(item)
            decorative = bool(
                props.get("decorative")
                or props.get("ariaHidden")
                or (
                    isinstance(props.get("accessibility"), Mapping)
                    and props["accessibility"].get("decorative")
                )
            )
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
            self.effective_components.append(
                {
                    "component_id": element_id,
                    "kind": canonical_media_kind(token),
                    "url": item.url,
                    "complete": model.complete,
                    "diagnostics": list(model.diagnostics),
                }
            )


@dataclass(frozen=True)
class V54EvidenceResult:
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
    effective_components: tuple[dict[str, Any], ...]


def collect_output_evidence_v5_4(
    spec: Mapping[str, Any],
    audit: _core.GraphAudit,
    config: RewardConfigV53,
    *,
    computed_registry: ComputedFunctionRegistryV54 | None = None,
) -> V54EvidenceResult:
    interpreter = _EvidenceInterpreterV54(spec, config, computed_registry)
    root = spec.get("root")
    if isinstance(root, str):
        interpreter.walk(root)
    static = _core.collect_output_evidence(spec, audit)
    visible = [
        text
        for text in (str(item).strip() for item in interpreter.visible_blocks)
        if text
    ]
    generic = [
        str(item.get("text") or "").strip()
        for item in interpreter.block_ownership
        if item.get("owner") == "generic_visible_content"
        and str(item.get("text") or "").strip()
    ]
    exact: Counter[str] = Counter()
    for block in visible:
        exact.update(
            _core.normalize_match_text(value)
            for value in _core._EXACT_VALUE_RE.findall(block)  # type: ignore[attr-defined]
        )
        exact.update(
            _core.normalize_match_text(value)
            for value in _core._DATE_RE.findall(block)  # type: ignore[attr-defined]
        )
    exact.pop("", None)
    type_counts: Counter[str] = Counter()
    for element_id in audit.reachable_ids:
        element = static.elements.get(element_id)
        if isinstance(element, Mapping):
            type_counts[str(element.get("type") or "")] += 1
    structured = [*interpreter.tables, *interpreter.charts]
    output = _core.OutputEvidence(
        elements=static.elements,
        reachable_ids=set(audit.reachable_ids),
        visible_blocks=visible,
        headings=interpreter.headings,
        heading_variants=interpreter.heading_variants,
        tables=structured,
        actions=interpreter.actions,
        media=interpreter.media,
        type_counts=type_counts,
        content_tokens=Counter(_core.tokenize(" ".join(visible))),
        exact_values=exact,
        valid_type_counts=Counter(type_counts),
    )
    unknown = tuple(interpreter.resolver.unknown)
    complete = not unknown and not interpreter.truncated
    registry = effective_registry_identity_v5_4(computed_registry)
    certification = {
        "complete": complete,
        "optimality_certified": complete,
        "work_units": interpreter.work_units,
        "expression_evaluations": interpreter.resolver.expression_evaluations,
        "evidence_bytes": interpreter.evidence_bytes,
        "diagnostic_codes": list(unknown),
    }
    dynamic = {
        "parity_version": DYNAMIC_SEMANTICS_VERSION,
        "parity_vector_version": DYNAMIC_PARITY_VECTOR_VERSION,
        "unknown_count": len(unknown),
        "unknown_codes": list(unknown),
        "complete": complete,
        "expanded_evidence_nodes": interpreter.expanded_nodes,
        "truncated": interpreter.truncated,
        "computed_registry": registry,
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
    return V54EvidenceResult(
        output=output,
        output_tables=tuple(interpreter.tables),
        output_charts=tuple(interpreter.charts),
        structured_data_payloads=tuple(structured),
        role_instances={
            role: tuple(values)
            for role, values in sorted(interpreter.role_instances.items())
        },
        prepared_visible_blocks=tuple(prepare_text_block(value) for value in visible),
        prepared_generic_visible_blocks=tuple(
            prepare_text_block(value) for value in generic
        ),
        generic_content_tokens=Counter(_core.tokenize(" ".join(generic))),
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
        dynamic_semantics=dynamic,
        dynamic_evidence_certification=certification,
        effective_components=tuple(interpreter.effective_components),
    )


__all__ = [
    "BUILTIN_COMPUTED_MANIFEST",
    "BUILTIN_COMPUTED_MANIFEST_HASH",
    "COMPUTED_REGISTRY_IDENTITY_VERSION",
    "ComputedFunctionRegistryV54",
    "DEFAULT_COMPUTED_REGISTRY_V54",
    "DYNAMIC_PARITY_VECTOR_VERSION",
    "DYNAMIC_SEMANTICS_VERSION",
    "DynamicEvidenceResolverV54",
    "EVIDENCE_POLICY_VERSION",
    "V54EvidenceResult",
    "callable_behavior_hash_v5_4",
    "collect_output_evidence_v5_4",
    "deep_equal_v5_4",
    "effective_registry_identity_v5_4",
    "kotlin_string_v5_4",
]
