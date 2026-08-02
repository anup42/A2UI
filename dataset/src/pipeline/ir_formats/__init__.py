"""Public production IR API. Legacy FlatSpec is read-only input only."""
from .codec import (
    A2UI_EXPRESS_V1, A2UI_V1_WIRE, FLAT_SPEC_V1,
    DecodedIr, decode_to_flat_spec, detect_format, encode_from_flat_spec,
    semantic_equivalent, serialized_text,
)
from .common import codec_identity, semantic_hash

__all__=[
    'A2UI_EXPRESS_V1','A2UI_V1_WIRE','FLAT_SPEC_V1',
    'DecodedIr','decode_to_flat_spec','detect_format','encode_from_flat_spec',
    'semantic_equivalent','serialized_text','codec_identity','semantic_hash',
]
