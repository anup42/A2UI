"""Read text SFT records without inferring Arrow types for arbitrary provenance.

Raw metadata/IR stays untouched and is available to the model adapter. Only
formatted text or final integer tensors cross the Arrow boundary in SFT.
"""
from __future__ import annotations

import json
from pathlib import Path


def _reject_json_constant(value: str):
    raise ValueError(f"non-standard JSON constant {value}")


def validate_sft_messages(row: dict) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("messages must contain a prompt and an assistant completion")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{index}] must be an object")
        if not isinstance(message.get("role"), str) or not message["role"].strip():
            raise ValueError(f"messages[{index}].role must be a nonempty string")
        if not isinstance(message.get("content"), str):
            raise ValueError(f"messages[{index}].content must be a string for text SFT")
    if messages[-1]["role"] != "assistant" or not messages[-1]["content"].strip():
        raise ValueError("messages must end with a nonempty assistant completion")
    if not any(message["content"].strip() for message in messages[:-1]):
        raise ValueError("messages must include nonempty prompt content")
    if row.get("completion") is not None and not isinstance(row["completion"], str):
        raise ValueError("completion must be a string when provided")


def iter_sft_rows(path: Path):
    """Yield original records, with actionable file/line errors; never skip bad data."""
    path = Path(path)
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=_reject_json_constant)
                if not isinstance(row, dict):
                    raise ValueError("expected a JSON object")
                validate_sft_messages(row)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            yield row
