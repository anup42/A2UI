"""Separate prose punctuation from references without breaking balanced URL paths."""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_TOKEN_START = re.compile(r"\[(?:IMAGE_URL|ICON_URL|MEDIA_URL|ACTION_URL|SOURCE_URL|URL|IMAGE_ASSET|ICON_ASSET|MEDIA_ASSET)_", re.IGNORECASE)


def reference_destination_errors(graph: Any, references: Mapping[str, str]) -> list[str]:
    """Reject damaged/unbound tokens in destinations, not quoted display text.

    No URL synthesis or graph rewriting takes place. Exact supplied tokens pass
    and are restored by the existing post-parse reference binding step.
    """
    errors: list[str] = []

    state = graph.get("state", {}) if isinstance(graph, dict) else {}

    def check(value: Any, path: str) -> None:
        # Resolve only direct absolute initial-state bindings. Do not treat
        # arbitrary state/table fields called 'url' as executable destinations.
        if isinstance(value, str) and value.startswith("$/"):
            resolved = state
            for part in value[2:].split("/"):
                if isinstance(resolved, dict):
                    resolved = resolved.get(part)
                elif isinstance(resolved, list) and part.isdigit() and int(part) < len(resolved):
                    resolved = resolved[int(part)]
                else:
                    resolved = None
                    break
            value = resolved
        if isinstance(value, str) and _TOKEN_START.search(value) and value not in references:
            errors.append(f"invalid_reference_destination:{path}")

    def actions(value: Any, path: str) -> None:
        if isinstance(value, dict):
            if value.get("action") == "openUrl" and isinstance(value.get("params"), dict):
                check(value["params"].get("url"), f"{path}.params.url")
            for key, item in value.items():
                # Event context is inert supplied data, not an executable action.
                if key not in {"params", "context"} and isinstance(item, (dict, list)):
                    actions(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                actions(item, f"{path}[{index}]")

    if isinstance(graph, dict):
        for key, element in graph.get("elements", {}).items():
            if not isinstance(element, dict):
                continue
            path = f"elements.{key}"
            if element.get("type") in {"Image", "Icon", "Video"}:
                for prop, value in element.get("props", {}).items():
                    if prop in {"url", "src", "source", "name", "icon", "poster", "posterUrl", "thumbnail", "thumbnailUrl"}:
                        check(value, f"{path}.props.{prop}")
            actions(element.get("on", {}), f"{path}.on")
            actions(element.get("watch", {}), f"{path}.watch")
    return errors


def split_reference_suffix(value: str) -> tuple[str, str]:
    """Keep balanced delimiters (e.g. Wikipedia titles) inside a reference.

    An unmatched closing delimiter belongs to surrounding prose/Markdown. Plain
    sentence punctuation retains the existing masking convention. Callers with
    explicitly quoted paths should preserve the entire quoted value instead.
    """
    end = len(value)
    closing = {")": "(", "]": "[", "}": "{"}
    while end:
        char = value[end - 1]
        if char in closing:
            prefix = value[:end]
            if prefix.count(char) <= prefix.count(closing[char]):
                break
        elif char not in ".,;:!?":
            break
        end -= 1
    return value[:end], value[end:]
