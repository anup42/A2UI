from __future__ import annotations

import json
import sys
from typing import Any, Mapping

from ir_training.common.config import repo_root

A2UI_EXPRESS_V1 = "a2ui_express_v1"
SUPPORTED_TARGETS = (A2UI_EXPRESS_V1,)

_ALIASES = {
    "express": A2UI_EXPRESS_V1,
    "a2ui_express": A2UI_EXPRESS_V1,
}


def canonical_graph_from_source(value: Any, source_format: str | None = None) -> dict[str, Any]:
    """Return the internal graph after an explicit source-format boundary.

    Active training records are decoded as A2UI Express text.  A legacy
    FlatSpec/Compact record is accepted only when its source format is marked
    for migration; it is never treated as a completion target.
    """
    api = _codec_api()
    hint = _normalize_source_format(source_format)
    if hint == A2UI_EXPRESS_V1 or (hint is None and isinstance(value, str)):
        return api["decode_express_completion"](value)
    if hint is None:
        raise ValueError(
            "A structured UI source requires an explicit source_format; "
            "use a2ui_express_v1 for active records or flat_spec_v1 for migration"
        )
    raise ValueError(
        "Legacy sources must be decoded by ir_training.data.legacy_targets "
        "before entering the active Express target builder"
    )


def canonical_flat_spec(value: Any, source_format: str | None = None) -> dict[str, Any]:
    """Compatibility shim; import the explicitly isolated legacy module."""

    from .legacy_targets import canonical_flat_spec as _legacy_canonical_flat_spec

    return _legacy_canonical_flat_spec(value, source_format=source_format)


def materialize_completion_targets(canonical_graph: Mapping[str, Any]) -> dict[str, Any]:
    api = _codec_api()
    expected = api["semantic_hash"](canonical_graph)
    payload = api["encode_express_completion"](canonical_graph)
    decoded = api["decode_express_completion"](payload)
    if api["semantic_hash"](decoded) != expected:
        raise ValueError("Semantic target mismatch for a2ui_express_v1")
    return {A2UI_EXPRESS_V1: payload}


def resolve_target_formats(run_cfg: Mapping[str, Any], row: Mapping[str, Any]) -> list[str]:
    configured = run_cfg.get("target_formats")
    if configured is None:
        configured = run_cfg.get("target_format")
    if configured is not None:
        raw_values = configured if isinstance(configured, list) else [configured]
        normalized: list[str] = []
        for value in raw_values:
            token = str(value).strip().lower()
            if token in {"both", "dual", "compact", "compact_ir", "gci2", "flat_spec", "flat_spec_v1"}:
                raise ValueError("Only a2ui_express_v1 is an active training target")
            resolved = _ALIASES.get(token, token)
            if resolved not in SUPPORTED_TARGETS:
                raise ValueError(
                    f"Unsupported training target {value!r}; choose {', '.join(SUPPORTED_TARGETS)}"
                )
            if resolved not in normalized:
                normalized.append(resolved)
        return normalized

    return [A2UI_EXPRESS_V1]


def serialize_completion(payload: Any, target_format: str) -> str:
    if target_format != A2UI_EXPRESS_V1:
        raise ValueError("Only a2ui_express_v1 completion targets are supported")
    if not isinstance(payload, str):
        raise ValueError("Express completion target must be text")
    return payload.strip()


def semantic_hash(canonical_graph: Mapping[str, Any]) -> str:
    return str(_codec_api()["semantic_hash"](canonical_graph))


def _normalize_source_format(value: str | None) -> str | None:
    if not value:
        return None
    token = str(value).strip().lower()
    return _ALIASES.get(token, token)


def _codec_api() -> dict[str, Any]:
    dataset_src = repo_root() / "dataset" / "src"
    if str(dataset_src) not in sys.path:
        sys.path.insert(0, str(dataset_src))
    from pipeline.ir_formats import (
        decode_express_completion,
        encode_express_completion,
        semantic_hash as codec_semantic_hash,
    )

    return {
        "decode_express_completion": decode_express_completion,
        "encode_express_completion": encode_express_completion,
        "semantic_hash": codec_semantic_hash,
    }


__all__ = [
    "A2UI_EXPRESS_V1",
    "SUPPORTED_TARGETS",
    "canonical_graph_from_source",
    "canonical_flat_spec",
    "materialize_completion_targets",
    "resolve_target_formats",
    "serialize_completion",
    "semantic_hash",
]
