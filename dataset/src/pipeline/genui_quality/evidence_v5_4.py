"""Android-effective dynamic and component evidence for metric v5.4."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
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
from .list_evidence_v5_4 import literal_list_evidence, parse_source_links, self_source_cue, source_heading
from .pilot_source_card_v2 import source_card_link_presence_v2


EVIDENCE_POLICY_VERSION = "5.4.2"
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
    def _text_values(self, element_type: str, props: Mapping[str, Any]) -> list[str]:
        if getattr(self.config, "pilot_fidelity_v2", False) and element_type == "formula":
            from .formula_evidence_v2 import normalize_formula_evidence_v2

            # RenderFormula's Elvis chain is null-only, and an empty expression
            # returns before the title, subtitle or result can render.
            raw = next((props[key] for key in ("latex", "formula", "text", "value") if props.get(key) is not None), "")
            formula = normalize_formula_evidence_v2(kotlin_string_v5_4(raw))
            if not formula:
                return []
            return [value for value in (
                kotlin_string_v5_4(props.get("title")).strip(),
                kotlin_string_v5_4(props.get("subtitle")).strip(),
                formula,
                kotlin_string_v5_4(props.get("result")).strip(),
            ) if value]
        if getattr(self.config, "pilot_fidelity_v2", False) and element_type == "alert":
            # RenderAlert in FlatIntentTemplateComposables.kt:67-148. The
            # fallback is null-only: an explicitly blank message hides text.
            # Metadata is one rendered Text block; unknown/unused props never
            # gain credit. Parent collection retains visibility, bindings,
            # repeat scopes, ownership and evidence-budget enforcement.
            def display(key: str) -> str:
                return kotlin_string_v5_4(props.get(key)).strip()

            message_key = "message" if props.get("message") is not None else "text"
            metadata = " • ".join(value for value in (display("timestamp"), display("source")) if value)
            return [value for value in (display("title"), display(message_key), metadata) if value]
        if getattr(self.config, "pilot_fidelity_v2", False) and element_type == "emailpreview":
            return [*super()._text_values(element_type, props),
                    *([props["date"].strip()] if isinstance(props.get("date"), str) and props["date"].strip() else [])]
        if getattr(self.config, "pilot_fidelity_v2", False) and element_type == "checklist":
            # The native Checklist consumes maps with labels and renders at
            # most twenty. Never treat a string item as visible evidence.
            items = props.get("items")
            rows = [item for item in items or () if isinstance(item, Mapping)
                    and isinstance(item.get("label"), str) and item["label"].strip()] if isinstance(items, list) else []
            if not isinstance(items, list) or len(rows) != len(items) or len(rows) > 20:
                self.resolver._unknown("pilot_checklist_items_not_renderable")
            visible = [kotlin_string_v5_4(props.get("title")).strip()]
            for item in rows[:20]:
                visible.extend((item["label"].strip() + (" *" if item.get("required") is True else ""),
                                kotlin_string_v5_4(item.get("detail")).strip()))
            visible.extend(kotlin_string_v5_4(props.get(key)).strip() for key in ("disclaimer", "source"))
            return [text for text in visible if text]
        return super()._text_values(element_type, props)

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
        self.source_section_nodes: set[str] = set()
        # A template/component ID can be rendered repeatedly or shared by
        # multiple parents. Content association must follow render occurrences,
        # never merge all occurrences bearing the same component ID.
        self._card_instances: list[str] = []
        self._next_card_instance = 0
        self._card_headers: dict[str, tuple[str, ...]] = {}
        self._card_header_children: dict[str, tuple[str, ...]] = {}
        self._source_cards: set[str] = set()
        self._pilot_render_stack: list[dict[str, Any]] = []
        self._next_pilot_render_instance = 0

    def _card_headers_may_be_intercepted(
        self, children: tuple[str, ...], *, item: Any,
        index: int | None, base_path: str | None,
        source_card: bool = False,
    ) -> bool:
        """Conservative header gate for Android's early-return Card routes.

        Source cues resolve direct child Text bindings. Email interception
        scans descendant Text even when ``visible=false``. Thus ordinary
        visible evidence cannot certify whether declared Card headers render.
        This bounded scan grants no content credit and deliberately withholds
        header credit when route resolution is uncertain.
        """
        texts: list[str] = []
        pilot = bool(getattr(self.config, "pilot_fidelity_v2", False))
        source_links_proven = False
        source_links_possible = False
        pending = [(child, (), True, True) for child in reversed(children)]
        while pending:
            child_id, ancestors, direct, source_owned = pending.pop()
            if not self._consume_work(f"card_header_route.{child_id}"):
                return True
            child = self.elements.get(child_id)
            if child_id in ancestors or not isinstance(child, Mapping):
                return True
            before_unknown = len(self.resolver.unknown)
            props = self.resolver.resolve(
                child.get("props") if isinstance(child.get("props"), Mapping) else {},
                item=item, index=index, base_path=base_path,
                path=f"card_header_route.{child_id}.props",
            )
            if len(self.resolver.unknown) != before_unknown or not isinstance(props, Mapping):
                return True
            token = str(child.get("type") or "").casefold()
            if direct and self_source_cue(child_id, props):
                if not pilot:
                    return True
                source_card = True
            if direct and token == "table":
                table = effective_table(props, self.state)
                if any(self_source_cue("table", {"title": column.label}) for column in table.columns):
                    if not pilot:
                        return True
                    source_card = True
            if token == "text":
                values = [
                    kotlin_string_v5_4(props[key]).strip()
                    for key in ("text", "title", "label", "content", "value")
                    if props.get(key) is not None and not isinstance(props[key], (Mapping, list))
                ]
                value = next((value for value in values if value), "")
                if direct and any(source_heading(line) for line in value.splitlines()):
                    if not pilot:
                        return True
                    source_card = True
                if value:
                    texts.append(value)
            if pilot and source_owned:
                on = self.resolver.resolve(
                    child.get("on") or {}, item=item, index=index, base_path=base_path,
                    path=f"card_header_route.{child_id}.on",
                ) if token == "button" else {}
                proven, possible = source_card_link_presence_v2(token, props, on)
                source_links_proven |= proven
                source_links_possible |= possible
            nested_children = child.get("children") or []
            if pilot and not nested_children and isinstance(props.get("child"), str):
                nested_children = [props["child"]]
            pending.extend(
                (child_id_next, (*ancestors, child_id), False,
                 source_owned and token not in {"text", "button", "table"})
                for child_id_next in reversed(nested_children)
                if isinstance(child_id_next, str)
            )
        if pilot and source_card:
            if source_links_proven:
                return True
            if source_links_possible:
                self.resolver._unknown("pilot_source_card_interception_unproven")
                return True
        return (
            len(texts) >= 5
            and any(
                text.casefold().startswith("subject:")
                and (text.partition(":")[2].strip() or position + 1 < len(texts))
                for position, text in enumerate(texts)
            )
            and any(
                text.casefold().startswith(("dear ", "hello ", "best regards", "sincerely", "regards", "thank you"))
                for text in texts
            )
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
        raw = self.elements.get(element_id)
        is_card = isinstance(raw, Mapping) and str(raw.get("type") or "").casefold() == "card"
        start_visible = len(self.visible_blocks)
        start_owned = len(self.block_ownership)
        pilot = bool(getattr(self.config, "pilot_fidelity_v2", False))
        prior_source_nodes = set(self.source_section_nodes) if pilot and is_card else None
        if pilot:
            if self._pilot_render_stack:
                parent_scope = self._pilot_render_stack[-1]
                scope = (index, base_path)
                if parent_scope.get("repeat_scope") != scope:
                    parent_scope["source_heading"] = False
                    parent_scope["repeat_scope"] = scope
            self._pilot_render_stack.append({
                "kind": str(raw.get("type") or "").casefold() if isinstance(raw, Mapping) else "",
                "source_heading": False,
                "instance": f"render:{self._next_pilot_render_instance}:{element_id}",
            })
            self._next_pilot_render_instance += 1
        if is_card:
            instance = f"card:{self._next_card_instance}:{element_id}"
            self._next_card_instance += 1
            self._card_instances.append(instance)
        try:
            super().walk(
                element_id, item=item, index=index, base_path=base_path,
                active_path=active_path,
            )
            if not is_card:
                return
            headers = self._card_headers.pop(instance, ())
            children = self._card_header_children.pop(instance, ())
            # Android's source-link/email Card routes return before ordinary
            # declared headers. Never credit text hidden by those routes.
            intercepted = headers and (
                (not pilot and instance in self._source_cards)
                or self._card_headers_may_be_intercepted(
                    children, item=item, index=index, base_path=base_path,
                    source_card=instance in self._source_cards,
                )
            )
            if headers and not intercepted:
                self.visible_blocks[start_visible:start_visible] = headers
                self.block_ownership[start_owned:start_owned] = [
                    {
                        "component_id": element_id,
                        "component_type": "card",
                        "card_instance": instance,
                        "card_ancestors": tuple(self._card_instances),
                        "owner": "generic_visible_content",
                        "text": text,
                    }
                    for text in headers
                ]
                self.evidence_bytes += sum(len(text.encode("utf-8")) for text in headers)
                if self.evidence_bytes > self.config.max_dynamic_evidence_bytes:
                    self.truncated = True
                    self.resolver._unknown("dynamic_evidence_byte_budget")
        finally:
            if prior_source_nodes is not None:
                self.source_section_nodes = prior_source_nodes
            if pilot:
                self._pilot_render_stack.pop()
            if is_card:
                self._card_instances.pop()
                self._source_cards.discard(instance)

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
            "formula_roles": len(self.role_instances.get("formula", ())),
        }
        super()._collect_element(element_id, element, props)
        token = str(element.get("type") or "").casefold().replace("_", "")
        if getattr(self.config, "pilot_fidelity_v2", False) and token == "text" and len(self._pilot_render_stack) >= 2:
            # Ordinary List has no native source interception. Its semantic
            # source role requires a preceding visible sibling source heading,
            # never an invisible component ID or unused List.title property.
            parent_scope = self._pilot_render_stack[-2]
            if parent_scope["kind"] in {"stack", "column", "row"}:
                blocks = self.visible_blocks[before["visible"] :]
                parent_scope["source_heading"] = bool(blocks and source_heading(blocks[0]))
        if token == "formula" and getattr(self.config, "pilot_fidelity_v2", False) and len(self.visible_blocks) == before["visible"]:
            del self.role_instances.get("formula", [])[before["formula_roles"] :]
        elif token == "formula" and getattr(self.config, "pilot_fidelity_v2", False):
            from .formula_evidence_v2 import formula_syntax_errors_v2

            raw_formula = next((props[key] for key in ("latex", "formula", "text", "value")
                                if props.get(key) is not None), "")
            for error in formula_syntax_errors_v2(kotlin_string_v5_4(raw_formula)):
                self.resolver._unknown(f"pilot_formula_syntax:{element_id}:{error}")
        if token == "alert" and getattr(self.config, "pilot_fidelity_v2", False):
            title = kotlin_string_v5_4(props.get("title")).strip()
            if title:
                self.headings.append(title)
                self.heading_variants.append("heading")
        if token == "card" and self._card_instances:
            self._card_headers[self._card_instances[-1]] = tuple(
                kotlin_string_v5_4(props[key]).strip()
                for key in ("title", "subtitle")
                if props.get(key) is not None and kotlin_string_v5_4(props[key]).strip()
            )
            self._card_header_children[self._card_instances[-1]] = tuple(
                child for child in (element.get("children") or [props.get("child")])
                if isinstance(child, str)
            )
        children = element.get("children") or []
        if token in {"stack", "column", "row", "card"} and (
            not getattr(self.config, "pilot_fidelity_v2", False) or token == "card"
        ) and (
            self_source_cue(element_id, props)
            or (token == "card" and any(
                self_source_cue(str(child), self.elements.get(child, {}).get("props", {}))
                or source_heading(self.elements.get(child, {}).get("props", {}).get("text"))
                for child in children if isinstance(self.elements.get(child), Mapping)
            ))
        ):
            if token == "card" and self._card_instances:
                self._source_cards.add(self._card_instances[-1])
            pending = list(children)
            while pending:
                child = pending.pop()
                if not isinstance(child, str) or child in self.source_section_nodes:
                    continue
                self.source_section_nodes.add(child)
                nested = self.elements.get(child)
                if isinstance(nested, Mapping):
                    pending.extend(nested.get("children") or [])
        source_context = element_id in self.source_section_nodes
        source_role_context = source_context or (
            getattr(self.config, "pilot_fidelity_v2", False) and token == "list"
            and len(self._pilot_render_stack) >= 2
            and self._pilot_render_stack[-2].get("source_heading") is True
        )
        if token == "list":
            pilot_list = bool(getattr(self.config, "pilot_fidelity_v2", False))
            if pilot_list and source_context and isinstance(props.get("items"), list):
                for position, entry in enumerate(props["items"]):
                    if not isinstance(entry, Mapping):
                        continue
                    targets = {entry[key].strip() for key in ("url", "href", "link", "source")
                               if isinstance(entry.get(key), str) and entry[key].strip()}
                    if len(targets) > 1:
                        # Native picks the first URL passing its public-host
                        # sanitizer, which this collector does not certify.
                        # Retain legacy evidence and block completeness rather
                        # than hide a later, potentially rendered destination.
                        self.resolver._unknown(f"pilot_source_link_destination_ambiguity:{element_id}[{position}]")
                        pilot_list = False
            blocks, links = literal_list_evidence(props.get("items"), source_context=source_context, pilot_fidelity_v2=pilot_list)
            # Parent interpreters do not own literal List.items. Add only
            # renderer-supported fields; never serialize arbitrary state/maps.
            for block in blocks:
                self.visible_blocks.append(block)
                source_label = source_role_context and any(block == label for label, _ in links)
                self.block_ownership.append({"component_id": element_id, "owner": "source_action" if (source_context and links) or source_label else "generic_visible_content", "text": block})
        elif token == "text" and source_context:
            blocks = [str(next((props[key] for key in ("text", "title", "label", "content", "value") if props.get(key) is not None), ""))]
            links = [link for block in blocks for link in parse_source_links(block, pilot_fidelity_v2=getattr(self.config, "pilot_fidelity_v2", False))]
        else:
            links = []
        for index, (label, target) in enumerate(links):
            action = _core.OutputAction(f"{element_id}[{index}]", label, target, "source_link" if source_role_context else "list.item", "openUrl")
            self.actions.append(action)
            self.action_taxonomy.append({"component_id": action.element_id, "action_type": "openUrl", "target": target, "category": "external_semantic"})
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
            axes = []
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
                    "chart_type": chart.chart_type,
                    "series": list(chart.series),
                    "x_values": list(chart.x_values),
                    "x_type": chart.x_type,
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
                y_fields = list(dict.fromkeys(value for series in chart.series for value in (series["key"], series["label"])))
                if chart.complete:
                    self._add_role(
                        "chart",
                        {
                            "component_id": element_id,
                            "title": chart.title,
                            "chart_type": chart.chart_type,
                            "series": list(chart.series),
                            "x_values": list(chart.x_values),
                            "x_type": chart.x_type,
                            "x_fields": x_fields,
                            "y_fields": y_fields,
                            "x_labels": list(chart.x_labels),
                            "y_values": [value for series in chart.series for value in series["displays"]],
                            "column_identities": [
                                column.key for column in chart.columns
                            ],
                            "data_identity": chart.dataset_identity,
                        },
                    )
                    if chart.title:
                        self.visible_blocks.append(chart.title)
                    if getattr(self.config, "pilot_fidelity_v2", False):
                        for field, key in (("x", "xLabel"), ("y", "yLabel"), ("right y", "rightYLabel")):
                            raw_axis = props.get(key)
                            if isinstance(raw_axis, str) and raw_axis.strip():
                                label = raw_axis.strip()
                                if field == "x" and chart.x_type == "time":
                                    label += " (UTC)"
                                axes.append((field, label))

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
            for field, text in axes:
                self.visible_blocks.append(text)
                self.block_ownership.append({"component_id": element_id, "owner": owner, "text": text,
                                             "semantic_field": field, "chart_title": chart.title})
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
            if getattr(self.config, "pilot_fidelity_v2", False):
                self.media_taxonomy[-1]["raw_url"] = model.url
                if self._card_instances:
                    self.media_taxonomy[-1]["card_instance"] = self._card_instances[-1]
            self.effective_components.append(
                {
                    "component_id": element_id,
                    "kind": canonical_media_kind(token),
                    "url": item.url,
                    "complete": model.complete,
                    "diagnostics": list(model.diagnostics),
                }
            )

        if getattr(self.config, "pilot_fidelity_v2", False):
            # Legacy collection canonicalizes URLs and drops fragments. Bind
            # each actual occurrence back to its already-resolved event/row;
            # do not inspect hidden props or unrelated state for a replacement.
            raw_events = defaultdict(deque)
            on = element.get("on")
            if isinstance(on, Mapping):
                for event, payload in on.items():
                    for raw_action in _core._iter_action_candidates(payload):
                        action_type = str(raw_action.get("action") or "")
                        params = raw_action.get("params")
                        if action_type == "openUrl" and isinstance(params, Mapping):
                            raw_events[(f"on.{event}", action_type)].append(
                                params.get("url") or params.get("href") or params.get("link") or params.get("targetUrl"))
            for action in self.actions[before["actions"] :]:
                if action.action_type != "openUrl":
                    continue
                target = None
                if action.source.startswith("on."):
                    pending = raw_events[(action.source, action.action_type)]
                    target = pending.popleft() if pending else None
                elif action.source.startswith("row.") and token in {"table", "chart"}:
                    try:
                        position = int(action.element_id.rsplit("[", 1)[1].rstrip("]"))
                        target = structured.rows[position].get(action.source.split(".", 1)[1]) if structured is not None else None
                    except (ValueError, IndexError):
                        target = None
                elif action.source in {"source_link", "list.item"}:
                    target = action.url
                if isinstance(target, str) and target.strip():
                    action.url = target.strip()
                else:
                    self.resolver._unknown(f"pilot_action_raw_target_unproven:{action.element_id}:{action.source}")
            for offset, taxonomy in enumerate(self.action_taxonomy[before["action_taxonomy"] :]):
                action_index = before["actions"] + offset
                if action_index < len(self.actions):
                    taxonomy["target"] = self.actions[action_index].url

        for row_index, row in enumerate(self.block_ownership[before["ownership"] :]):
            row["component_type"] = token
            if getattr(self.config, "pilot_fidelity_v2", False) and self._pilot_render_stack:
                row["render_instance"] = self._pilot_render_stack[-1]["instance"]
                if len(self._pilot_render_stack) >= 2:
                    row["parent_instance"] = self._pilot_render_stack[-2]["instance"]
                    row["parent_kind"] = self._pilot_render_stack[-2]["kind"]
                if token == "table" and len(self.tables) > before["tables"]:
                    row["table_header"] = row_index < len(self.tables[before["tables"]].headers)
                if token == "emailpreview":
                    fields = [name for name in ("subject", "title", "to", "from", "date", "body")
                              if isinstance(props.get(name), str) and props[name].strip() == row.get("text")]
                    if len(fields) == 1:
                        row["semantic_field"] = fields[0]
            if getattr(self.config, "pilot_fidelity_v2", False) and row_index == 0:
                text_heading = token == "text" and str(props.get("variant") or "").casefold() in {"h1", "h2", "h3", "h4", "title", "heading"}
                alert_heading = token == "alert" and bool(kotlin_string_v5_4(props.get("title")).strip())
                checklist_heading = token == "checklist" and bool(kotlin_string_v5_4(props.get("title")).strip())
                if text_heading or alert_heading or checklist_heading:
                    row["rendered_heading"] = True
            if self._card_instances:
                row["card_instance"] = self._card_instances[-1]
                row["card_ancestors"] = tuple(self._card_instances)
        if getattr(self.config, "grpo_entity_extraction", False):
            for offset, row in enumerate(self.action_taxonomy[before["action_taxonomy"] :]):
                action_index = before["actions"] + offset
                if action_index < len(self.actions):
                    row["label"] = self.actions[action_index].label
                if self._card_instances:
                    row["card_instance"] = self._card_instances[-1]


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
