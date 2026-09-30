"""Literal List.items contract shared conceptually with FlatListItems.kt.

Component calls belong in children, never opaque maps inside data items.
"""
from collections.abc import Mapping
from typing import Any

LIST_TEXT_FIELDS = ("text", "label", "title", "name", "content", "value", "description")
LIST_LINK_FIELDS = ("url", "href", "link", "source")


def validate_list_items(value: Any) -> str | None:
    if not isinstance(value, list):
        return "items must be an array of text or text/link objects; use children for components"
    for index, item in enumerate(value):
        if isinstance(item, str):
            continue
        if not isinstance(item, Mapping) or not item:
            return f"items[{index}] must be text or a nonempty text/link object"
        unknown = set(item) - set(LIST_TEXT_FIELDS + LIST_LINK_FIELDS)
        if unknown:
            return f"items[{index}] has unsupported fields {sorted(unknown)}; use children for component calls"
        if any(not isinstance(v, str) for v in item.values()):
            return f"items[{index}] text/link fields must be strings"
    return None
