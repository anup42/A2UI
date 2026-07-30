from __future__ import annotations

import copy
import json
from typing import Any

from pipeline.genui_quality import (
    extract_expected_ui_contract_v5_2,
    score_genui_completion_v5_2 as score_genui_completion,
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


def contract(
    source: str,
    *,
    intent: str | None = None,
    **updates: Any,
) -> dict[str, Any]:
    value = extract_expected_ui_contract_v5_2(source, intent=intent)
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
    **kwargs: Any,
):
    return score_genui_completion(
        compact(spec) if isinstance(spec, (dict, list)) else spec,
        source,
        intent=intent,
        expected_ui_contract=expected,
        **kwargs,
    )


def action_spec(
    actions: list[tuple[str, str, str]]
) -> dict[str, Any]:
    children: list[str] = []
    elements: dict[str, Any] = {
        "root": {"type": "Stack", "props": {}, "children": children}
    }
    for index, (label, target, action_type) in enumerate(actions):
        element_id = f"action_{index}"
        params = (
            {"url": target}
            if action_type == "openUrl"
            else {"statePath": target}
        )
        elements[element_id] = {
            "type": "Button",
            "props": {"label": label},
            "children": [],
            "on": {
                "press": {
                    "action": action_type,
                    "params": params,
                }
            },
        }
        children.append(element_id)
    return {"root": "root", "state": {}, "elements": elements}
