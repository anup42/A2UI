from __future__ import annotations

from typing import Any

from toon_format import ToonDecodeError, decode as toon_decode, encode as toon_encode


def canonicalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: canonicalize_json(value[k]) for k in sorted(value.keys())}
    if isinstance(value, list):
        return [canonicalize_json(item) for item in value]
    return value


def _is_primitive(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _prepare_for_toon(value: Any) -> Any:
    """Prepare a JSON-like value for stable, decoder-friendly TOON encoding.

    The TOON encoder/decoder are both strict about list item shapes. In practice,
    we found that emitting list items where the first mapping key is a nested
    object can produce output that some decoders reject (e.g. a list item
    rendered as a standalone '-' line followed by an indented object).

    To avoid this, we reorder dict keys so that primitive values come first,
    followed by non-primitives. Keys are sorted within those groups to keep the
    encoding stable across runs.
    """
    if isinstance(value, dict):
        prim_keys: list[str] = []
        nonprim_keys: list[str] = []
        for k, v in value.items():
            if _is_primitive(v):
                prim_keys.append(str(k))
            else:
                nonprim_keys.append(str(k))
        ordered_keys = sorted(prim_keys) + sorted(nonprim_keys)
        return {k: _prepare_for_toon(value[k]) for k in ordered_keys}
    if isinstance(value, list):
        return [_prepare_for_toon(item) for item in value]
    return value


def encode_toon(genui_json: Any) -> str:
    prepared = _prepare_for_toon(genui_json)
    return toon_encode(prepared, options={"indent": 2})


def decode_toon(toon: str) -> Any:
    # Accept raw TOON (v1.x) strings (no custom wrapper).
    return toon_decode(toon)


def roundtrip_ok(genui_json: Any, toon: str) -> bool:
    try:
        decoded = decode_toon(toon)
    except (ToonDecodeError, Exception):
        return False
    return canonicalize_json(decoded) == canonicalize_json(genui_json)
