"""Production A2UI Express boundary plus read-only legacy graph conversion."""
from __future__ import annotations
from dataclasses import dataclass
import json
from typing import Any, Mapping

from ..flat_spec_contract import coerce_and_validate, looks_like_flat_spec
from . import express, a2ui_wire
from .canonical import canonical_graph
from .common import codec_identity, semantic_hash

FLAT_SPEC_V1='flat_spec_v1'  # read-only migration source
A2UI_EXPRESS_V1='a2ui_express_v1'
A2UI_V1_WIRE='a2ui_v1_wire'

@dataclass(frozen=True)
class DecodedIr:
    source_format: str
    flat_spec: dict[str,Any]


def detect_format(value: Any) -> str:
    if isinstance(value,str):
        text=value.strip()
        if '<a2ui>' in text or ('=' in text and resembles_express(text)): return A2UI_EXPRESS_V1
        try:
            value=json.loads(text)
        except Exception as exc:
            raise ValueError('Unable to detect IR text format') from exc
        if isinstance(value, str):
            return detect_format(value)
    if looks_like_flat_spec(value): return FLAT_SPEC_V1
    if isinstance(value,Mapping) and value.get('v')=='gci2':
        raise ValueError('Compact IR is migration-only and is not an active format')
    if isinstance(value,Mapping) and value.get('version')=='v1.0' and 'createSurface' in value: return A2UI_V1_WIRE
    if isinstance(value,list) and value and all(isinstance(item,Mapping) and item.get('version')=='v1.0' for item in value):
        return A2UI_V1_WIRE
    raise ValueError('Unsupported or ambiguous IR format')


def resembles_express(text: str) -> bool:
    return any(
        line.strip().startswith(('root=', 'root =', '$=', '$ =', '$/=', '$/ ='))
        for line in text.splitlines()
    )


def decode_to_flat_spec(value: Any, *, format_hint: str|None=None) -> DecodedIr:
    format_id=format_hint or detect_format(value)
    if format_id==FLAT_SPEC_V1:
        if isinstance(value,str): value=json.loads(value)
        result=coerce_and_validate(value)
        if not result.is_valid or result.spec is None: raise ValueError(result.error or 'Invalid FlatSpec')
        return DecodedIr(format_id,result.spec)
    if format_id==A2UI_EXPRESS_V1:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith('"'):
                decoded_string = json.loads(stripped)
                if isinstance(decoded_string, str):
                    value = decoded_string
        raw=express.decode(value)
    elif format_id==A2UI_V1_WIRE:
        if isinstance(value,str): value=json.loads(value)
        raw=a2ui_wire.decode(value)
    else: raise ValueError(f'Unsupported production format {format_id}')
    # Active formats must be checked by the strict canonical graph contract.
    # Do not pass model output through the legacy FlatSpec coercer, which can
    # normalize malformed values (for example, invalid Stack gap enums).
    return DecodedIr(format_id, canonical_graph(raw))


def encode_from_flat_spec(spec: Mapping[str,Any], target_format: str, *, shorten_ids: bool=True, pretty: bool=False) -> Any:
    result=coerce_and_validate(dict(spec))
    if not result.is_valid or result.spec is None: raise ValueError(result.error or 'Invalid FlatSpec')
    if target_format==FLAT_SPEC_V1: return result.spec
    if target_format==A2UI_EXPRESS_V1: return express.encode(result.spec,shorten_ids=shorten_ids,pretty=pretty)
    if target_format==A2UI_V1_WIRE: return a2ui_wire.encode(result.spec,shorten_ids=shorten_ids)
    if target_format == 'compact_ir_v2':
        raise ValueError('Compact IR is migration-only and cannot be generated')
    raise ValueError(f'Unsupported production format {target_format}')


def semantic_equivalent(left: Any,right: Any) -> bool:
    return semantic_hash(decode_to_flat_spec(left).flat_spec)==semantic_hash(decode_to_flat_spec(right).flat_spec)


def serialized_text(value: Any, *, pretty: bool=False) -> str:
    if isinstance(value,str): return value.strip()
    return json.dumps(value,ensure_ascii=False,indent=2 if pretty else None,separators=None if pretty else (',',':'),sort_keys=False)
