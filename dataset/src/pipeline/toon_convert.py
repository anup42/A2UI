from __future__ import annotations

import base64
import json
from typing import Any


def canonicalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: canonicalize_json(value[k]) for k in sorted(value.keys())}
    if isinstance(value, list):
        return [canonicalize_json(item) for item in value]
    return value


def encode_toon(a2ui_json: Any) -> str:
    canonical = canonicalize_json(a2ui_json)
    payload = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    b64 = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")
    return f"TOON({b64})"


def decode_toon(toon: str) -> Any:
    if not toon.startswith("TOON(") or not toon.endswith(")"):
        raise ValueError("Invalid TOON wrapper")
    b64 = toon[len("TOON("):-1]
    raw = base64.urlsafe_b64decode(b64.encode("ascii")).decode("utf-8")
    return json.loads(raw)


def roundtrip_ok(a2ui_json: Any, toon: str) -> bool:
    try:
        decoded = decode_toon(toon)
    except Exception:
        return False
    return canonicalize_json(decoded) == canonicalize_json(a2ui_json)
