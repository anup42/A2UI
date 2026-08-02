from __future__ import annotations

import json
import sys
from typing import Any, Mapping

from ir_training.common.config import repo_root

FLAT_SPEC_V1 = "flat_spec_v1"
COMPACT_IR_V2 = "compact_ir_v2"
A2UI_EXPRESS_V1 = "a2ui_express_v1"
SUPPORTED_TARGETS = (COMPACT_IR_V2, A2UI_EXPRESS_V1)

_ALIASES = {
    "compact": COMPACT_IR_V2,
    "compact_ir": COMPACT_IR_V2,
    "gci2": COMPACT_IR_V2,
    "express": A2UI_EXPRESS_V1,
    "a2ui_express": A2UI_EXPRESS_V1,
}


def canonical_flat_spec(value: Any, source_format: str | None = None) -> dict[str, Any]:
    api = _codec_api()
    hint = _normalize_source_format(source_format)
    try:
        return api["decode_to_flat_spec"](value, format_hint=hint).flat_spec
    except ValueError:
        if hint is not None:
            raise
        result = api["coerce_and_validate"](value)
        if not result.is_valid or result.spec is None:
            raise ValueError(result.error or "Invalid renderer graph")
        return result.spec


def materialize_completion_targets(flat_spec: Mapping[str, Any]) -> dict[str, Any]:
    api = _codec_api()
    expected = api["semantic_hash"](flat_spec)
    targets: dict[str, Any] = {}
    for target_format in SUPPORTED_TARGETS:
        payload = api["encode_from_flat_spec"](
            flat_spec,
            target_format,
            shorten_ids=True,
        )
        decoded = api["decode_to_flat_spec"](
            payload,
            format_hint=target_format,
        ).flat_spec
        if api["semantic_hash"](decoded) != expected:
            raise ValueError(f"Semantic target mismatch for {target_format}")
        targets[target_format] = payload
    return targets


def resolve_target_formats(run_cfg: Mapping[str, Any], row: Mapping[str, Any]) -> list[str]:
    configured = run_cfg.get("target_formats")
    if configured is None:
        configured = run_cfg.get("target_format")
    if configured is not None:
        raw_values = configured if isinstance(configured, list) else [configured]
        normalized: list[str] = []
        for value in raw_values:
            token = str(value).strip().lower()
            if token in {"both", "dual"}:
                for target in SUPPORTED_TARGETS:
                    if target not in normalized:
                        normalized.append(target)
                continue
            resolved = _ALIASES.get(token, token)
            if resolved not in SUPPORTED_TARGETS:
                raise ValueError(
                    f"Unsupported training target {value!r}; choose {', '.join(SUPPORTED_TARGETS)}"
                )
            if resolved not in normalized:
                normalized.append(resolved)
        return normalized

    source_format = str(row.get("source_format") or "").strip()
    if source_format in SUPPORTED_TARGETS:
        return [source_format]
    # Historical FlatSpec rows produce one Compact IR example by default so
    # importing an old run never silently doubles the dataset.
    return [COMPACT_IR_V2]


def serialize_completion(payload: Any, target_format: str) -> str:
    if target_format == A2UI_EXPRESS_V1:
        if not isinstance(payload, str):
            raise ValueError("Express completion target must be text")
        return payload.strip()
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def semantic_hash(flat_spec: Mapping[str, Any]) -> str:
    return str(_codec_api()["semantic_hash"](flat_spec))


def _normalize_source_format(value: str | None) -> str | None:
    if not value:
        return None
    token = str(value).strip().lower()
    return _ALIASES.get(token, token)


def _codec_api() -> dict[str, Any]:
    dataset_src = repo_root() / "dataset" / "src"
    if str(dataset_src) not in sys.path:
        sys.path.insert(0, str(dataset_src))
    from pipeline.flat_spec_contract import coerce_and_validate
    from pipeline.ir_formats import (
        decode_to_flat_spec,
        encode_from_flat_spec,
        semantic_hash as codec_semantic_hash,
    )

    return {
        "coerce_and_validate": coerce_and_validate,
        "decode_to_flat_spec": decode_to_flat_spec,
        "encode_from_flat_spec": encode_from_flat_spec,
        "semantic_hash": codec_semantic_hash,
    }


__all__ = [
    "FLAT_SPEC_V1",
    "COMPACT_IR_V2",
    "A2UI_EXPRESS_V1",
    "SUPPORTED_TARGETS",
    "canonical_flat_spec",
    "materialize_completion_targets",
    "resolve_target_formats",
    "serialize_completion",
    "semantic_hash",
]
