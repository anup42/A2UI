from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_shared_express_contract() -> str:
    contract_path = (
        Path(__file__).resolve().parents[4]
        / "dataset"
        / "prompts"
        / "genui_gen_mobile_a2ui_express_v1.md"
    )
    try:
        return contract_path.read_text(encoding="utf-8").replace(
            "{response_text}", "[RESPONSE_TEXT_IS_PROVIDED_IN_THE_USER_MESSAGE]"
        ).strip()
    except OSError:
        # Keep imports usable for packaged training environments that do not
        # ship the repository prompt; normal repository runs use the generated
        # contract above and the prompt drift test catches missing copies.
        return (
            "You convert response text into A2UI Express v1. Return exactly one "
            "strict <a2ui>...</a2ui> block and preserve all meaningful UI semantics."
        )


DEFAULT_SYSTEM_PROMPT = _load_shared_express_contract()


def minify_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _target_instruction(target_format: str | None) -> str:
    if target_format == "a2ui_express_v1":
        return "Create A2UI Express v1 GenUI IR for this response:"
    if target_format not in {None, "a2ui_express_v1"}:
        raise ValueError("Only a2ui_express_v1 is an active training target")
    return "Create A2UI Express v1 GenUI IR for this response:"


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
