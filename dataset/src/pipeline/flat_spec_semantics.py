"""Authoritative Android FlatSpec renderer reference semantics.

This module intentionally contains no metric code.  The production contract,
graph auditor, evidence collector, and reward boundary all import the same
reference inventory so renderer traversal cannot drift between call sites.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping


RENDERER_SEMANTICS_VERSION = "1.0.0"


@dataclass(frozen=True)
class RendererReference:
    target_id: str
    source_path: str
    reference_kind: str


def _non_empty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _append(
    out: list[RendererReference],
    value: Any,
    source_path: str,
    reference_kind: str,
) -> None:
    target = _non_empty_string(value)
    if target is None:
        return
    item = RendererReference(target, source_path, reference_kind)
    if item not in out:
        out.append(item)


def _first_reference(
    mapping: Mapping[str, Any],
    keys: Iterable[str],
) -> tuple[str, str] | None:
    """Match Kotlin's left-to-right Elvis-chain reference resolution."""
    for key in keys:
        value = _non_empty_string(mapping.get(key))
        if value is not None:
            return key, value
    return None


def iter_renderer_references(
    element: Mapping[str, Any],
) -> tuple[RendererReference, ...]:
    """Return every element ID the Android FlatSpec renderer may traverse.

    Inventory audited against ``FlatSpecRenderer.kt``:

    * ordinary ``children`` plus legacy top-level ``child``;
    * parser-supported ``template``/``itemTemplate``/``child`` properties;
    * the same template aliases in a repeat object for generation-contract
      compatibility;
    * ``Tabs.props.tabs[]`` using the renderer's exact alias precedence; and
    * ``Modal.props.trigger`` and ``Modal.props.content``.

    Repeat iteration changes scope, not the referenced element inventory.  Its
    template is therefore represented by the ordinary/template edges above.
    """

    out: list[RendererReference] = []
    children = element.get("children")
    if isinstance(children, list):
        for index, child in enumerate(children):
            _append(out, child, f"children[{index}]", "child")

    _append(out, element.get("child"), "child", "legacy_child")
    _append(out, element.get("template"), "template", "template")
    _append(out, element.get("itemTemplate"), "itemTemplate", "item_template")

    props = element.get("props")
    if not isinstance(props, Mapping):
        props = {}
    for key, kind in (
        ("template", "template"),
        ("itemTemplate", "item_template"),
        ("child", "property_child"),
    ):
        _append(out, props.get(key), f"props.{key}", kind)

    repeat = element.get("repeat")
    if not isinstance(repeat, Mapping):
        repeat_value = props.get("repeat")
        repeat = repeat_value if isinstance(repeat_value, Mapping) else {}
    for key, kind in (
        ("template", "repeat_template"),
        ("itemTemplate", "repeat_item_template"),
        ("child", "repeat_child"),
    ):
        _append(out, repeat.get(key), f"repeat.{key}", kind)

    element_type = str(element.get("type") or element.get("component") or "").casefold()
    if element_type in {"tabs", "tab", "tabgroup"}:
        tabs = props.get("tabs")
        if isinstance(tabs, list):
            for index, raw_tab in enumerate(tabs):
                if not isinstance(raw_tab, Mapping):
                    continue
                selected = _first_reference(raw_tab, ("child", "content", "id", "element"))
                if selected is not None:
                    key, target = selected
                    _append(
                        out,
                        target,
                        f"props.tabs[{index}].{key}",
                        "tab_content",
                    )

    if element_type == "modal":
        _append(out, props.get("trigger"), "props.trigger", "modal_trigger")
        _append(out, props.get("content"), "props.content", "modal_content")

    return tuple(out)


def renderer_reference_inventory_hash() -> str:
    """Stable fingerprint input for score identity and stale-score detection."""
    probes = (
        {
            "type": "Stack",
            "props": {
                "template": "p_template",
                "itemTemplate": "p_item",
                "child": "p_child",
            },
            "children": ["child"],
            "repeat": {
                "template": "r_template",
                "itemTemplate": "r_item",
                "child": "r_child",
            },
        },
        {
            "type": "Tabs",
            "props": {
                "tabs": [
                    {"child": "tab_child"},
                    {"content": "tab_content"},
                    {"id": "tab_id"},
                    {"element": "tab_element"},
                ]
            },
            "children": [],
        },
        {
            "type": "Modal",
            "props": {"trigger": "trigger", "content": "content"},
            "children": [],
        },
    )
    payload = {
        "version": RENDERER_SEMANTICS_VERSION,
        "references": [
            [asdict(reference) for reference in iter_renderer_references(probe)]
            for probe in probes
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "RENDERER_SEMANTICS_VERSION",
    "RendererReference",
    "iter_renderer_references",
    "renderer_reference_inventory_hash",
]
