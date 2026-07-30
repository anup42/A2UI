from __future__ import annotations

import copy
import json
from typing import Any

from pipeline.genui_quality import (
    extract_expected_ui_contract_v5_1,
    score_genui_completion_v5_1,
)


def simple_spec(text: str = "Hello world") -> dict[str, Any]:
    return {
        "root": "root",
        "state": {},
        "elements": {
            "root": {
                "type": "Stack",
                "props": {"direction": "vertical"},
                "children": ["body"],
            },
            "body": {
                "type": "Text",
                "props": {"text": text, "variant": "body"},
                "children": [],
            },
        },
    }


def contract(source: str, *, intent: str | None = None, **updates: Any) -> dict[str, Any]:
    value = extract_expected_ui_contract_v5_1(source, intent=intent)
    value.update(copy.deepcopy(updates))
    return value


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def score(
    spec: Any,
    source: str,
    *,
    intent: str | None = None,
    expected: dict[str, Any] | None = None,
):
    return score_genui_completion_v5_1(
        compact(spec),
        source,
        intent=intent,
        expected_ui_contract=expected,
    )
