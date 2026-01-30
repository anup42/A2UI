from __future__ import annotations

from pathlib import Path
from typing import Any


def _parse_scalar(value: str) -> Any:
    v = value.strip()
    if v == "null" or v == "~":
        return None
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(x.strip()) for x in inner.split(",")]
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        return v


def _prep_lines(text: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        lines.append((indent, raw.strip()))
    return lines


def _load_yaml_minimal(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(0, root)]
    lines = _prep_lines(text)

    def next_is_list(idx: int, indent: int) -> bool:
        for j in range(idx + 1, len(lines)):
            n_indent, n_content = lines[j]
            if n_indent <= indent:
                return False
            if n_content.startswith("- "):
                return True
            return False
        return False

    for idx, (indent, content) in enumerate(lines):
        while stack and indent < stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if content.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError("YAML list item without list parent")
            item_content = content[2:].strip()
            if not item_content:
                new_item: dict[str, Any] = {}
                parent.append(new_item)
                stack.append((indent + 2, new_item))
                continue
            key, sep, value = item_content.partition(":")
            if sep:
                key = key.strip()
                value = value.strip()
                new_item = {key: _parse_scalar(value)} if value else {key: {}}
                parent.append(new_item)
                if value == "" or next_is_list(idx, indent):
                    stack.append((indent + 2, new_item))
                continue
            parent.append(_parse_scalar(item_content))
            continue

        key, sep, value = content.partition(":")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if value == "":
            if next_is_list(idx, indent):
                new_list: list[Any] = []
                parent[key] = new_list
                stack.append((indent + 2, new_list))
            else:
                new_dict: dict[str, Any] = {}
                parent[key] = new_dict
                stack.append((indent + 2, new_dict))
            continue
        parent[key] = _parse_scalar(value)

    return root


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None

    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text)
        if data is None:
            return {}
        return data
    if text.lstrip().startswith("{"):
        import json
        return json.loads(text)
    return _load_yaml_minimal(text)
