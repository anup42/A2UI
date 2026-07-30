from __future__ import annotations

from typing import Any


def root_spec(
    elements: dict[str, dict[str, Any]],
    children: list[str],
    *,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "root": "root",
        "state": dict(state or {}),
        "elements": {
            "root": {
                "type": "Column",
                "props": {},
                "children": list(children),
            },
            **elements,
        },
    }


def text(value: str) -> dict[str, Any]:
    return {
        "type": "Text",
        "props": {"text": value},
        "children": [],
    }


def table() -> dict[str, Any]:
    return {
        "type": "Table",
        "props": {
            "columns": [
                {"key": "city", "label": "City"},
                {"key": "temp", "label": "Temp"},
            ],
            "rows": [
                {"city": "Paris", "temp": "21 C"},
                {"city": "Rome", "temp": "25 C"},
            ],
        },
        "children": [],
    }


TABLE_SOURCE = (
    "| City | Temp |\n"
    "|---|---|\n"
    "| Paris | 21 C |\n"
    "| Rome | 25 C |"
)
