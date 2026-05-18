from __future__ import annotations

import json
from typing import Any

DEFAULT_SYSTEM_PROMPT = (
    "You convert response text into compact Android flat-spec GenUI IR. "
    "Return JSON only with top-level keys root, state, and elements."
)


def minify_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_messages(system_prompt: str, response_text: str, completion_json: Any | None = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": system_prompt.strip() or DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": "Create GenUI flat-spec IR for this response:\n\n" + response_text.strip()},
    ]
    if completion_json is not None:
        messages.append({"role": "assistant", "content": minify_json(completion_json)})
    return messages


def build_prompt(system_prompt: str, response_text: str) -> str:
    return (
        f"System:\n{system_prompt.strip() or DEFAULT_SYSTEM_PROMPT}\n\n"
        "User:\nCreate GenUI flat-spec IR for this response:\n\n"
        f"{response_text.strip()}\n\nAssistant:\n"
    )
