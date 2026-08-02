"""Public production IR API. Legacy FlatSpec is read-only input only."""
from .codec import (
    A2UI_EXPRESS_V1, A2UI_V1_WIRE, FLAT_SPEC_V1,
    DecodedIr, decode_to_flat_spec, detect_format, encode_from_flat_spec,
    semantic_equivalent, serialized_text,
)
from .common import codec_identity, semantic_hash
from .active import (
    ACTIVE_FORMAT_ID,
    WIRE_FORMAT_ID,
    ActiveValidation,
    compile_express_to_wire,
    decode_express_completion,
    encode_express_completion,
    validate_express_completion,
)

__all__=[
    'A2UI_EXPRESS_V1','A2UI_V1_WIRE','FLAT_SPEC_V1',
    'DecodedIr','decode_to_flat_spec','detect_format','encode_from_flat_spec',
    'semantic_equivalent','serialized_text','codec_identity','semantic_hash',
    'ACTIVE_FORMAT_ID','WIRE_FORMAT_ID','ActiveValidation',
    'compile_express_to_wire','decode_express_completion',
    'encode_express_completion','validate_express_completion',
]
