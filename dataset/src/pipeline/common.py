from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_json_array_re = re.compile(r"\[.*\]", re.DOTALL)
_json_object_re = re.compile(r"\{.*\}", re.DOTALL)


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def render_prompt(template: str, **kwargs: Any) -> str:
    return template.format(**kwargs)


def extract_json(text: str) -> Any:
    text = text.strip()
    if not text:
        raise ValueError("Empty text")
    if text[0] in "[{":
        return json.loads(text)
    match = _json_array_re.search(text)
    if match:
        return json.loads(match.group(0))
    match = _json_object_re.search(text)
    if match:
        return json.loads(match.group(0))
    raise ValueError("No JSON found")


def minify_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
