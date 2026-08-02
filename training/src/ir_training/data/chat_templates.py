from __future__ import annotations

import json
from typing import Any

DEFAULT_SYSTEM_PROMPT = (
    "You convert response text into compact Android GenUI IR without removing "
    "meaningful components or interactions. Return only the requested format."
)


def minify_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _target_instruction(target_format: str | None) -> str:
    if target_format == "a2ui_express_v1":
        return "Create A2UI Express v1 GenUI IR for this response:"
    if target_format == "compact_ir_v2":
        return "Create Compact IR v2 JSON for this response:"
    return "Create GenUI IR for this response:"


def build_messages(
    system_prompt: str,
    response_text: str,
    completion_json: Any | None = None,
    *,
    target_format: str | None = None,
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": system_prompt.strip() or DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": _target_instruction(target_format) + "\n\n" + response_text.strip()},
    ]
    if completion_json is not None:
        completion = completion_json if isinstance(completion_json, str) else minify_json(completion_json)
        messages.append({"role": "assistant", "content": completion})
    return messages


def build_prompt(
    system_prompt: str,
    response_text: str,
    *,
    target_format: str | None = None,
) -> str:
    return (
        f"System:\n{system_prompt.strip() or DEFAULT_SYSTEM_PROMPT}\n\n"
        f"User:\n{_target_instruction(target_format)}\n\n"
        f"{response_text.strip()}\n\nAssistant:\n"
    )
