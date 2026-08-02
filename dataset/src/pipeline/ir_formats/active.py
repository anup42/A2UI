"""The single active model-facing A2UI Express boundary.

Legacy graph-shaped payloads are intentionally handled by ``migration`` or
offline comparison callers.  Training, evaluation, and generation code should
use this module so a completion is always validated as Express text first and
only then lowered to the internal canonical graph used by the native renderer.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..flat_spec_contract import coerce_and_validate
from . import a2ui_wire, express


ACTIVE_FORMAT_ID = "a2ui_express_v1"
WIRE_FORMAT_ID = "a2ui_v1_wire"


@dataclass(frozen=True)
class ActiveValidation:
    """Separate raw validity from validity after an explicit repair."""

    raw_valid: bool
    repaired_valid: bool
    canonical_graph: dict[str, Any] | None
    errors: tuple[str, ...]
    semantic_hash: str | None
    repair_applied: bool = False

    @property
    def production_valid(self) -> bool:
        return self.raw_valid or self.repaired_valid


def decode_express_completion(value: Any) -> dict[str, Any]:
    """Strictly decode one active Express completion to the canonical graph."""

    if not isinstance(value, str):
        raise ValueError("A2UI Express completion must be text")
    text = value.strip()
    if not text.startswith(express.SENTINEL_OPEN) or not text.endswith(express.SENTINEL_CLOSE):
        raise ValueError("A2UI Express completion must contain exactly one <a2ui> block")
    graph = express.decode(text)
    validation = coerce_and_validate(graph)
    if not validation.is_valid or validation.spec is None:
        raise ValueError(validation.error or "Decoded A2UI Express graph is invalid")
    return validation.spec


def encode_express_completion(graph: Mapping[str, Any]) -> str:
    """Encode the canonical graph as the only active completion format."""

    validation = coerce_and_validate(dict(graph))
    if not validation.is_valid or validation.spec is None:
        raise ValueError(validation.error or "Canonical UI graph is invalid")
    return express.encode(validation.spec, shorten_ids=True)


def validate_express_completion(
    raw_completion: Any,
    repaired_completion: Any | None = None,
) -> ActiveValidation:
    """Validate raw and optional repaired text without conflating the results."""

    errors: list[str] = []
    graph: dict[str, Any] | None = None
    raw_valid = False
    try:
        graph = decode_express_completion(raw_completion)
        raw_valid = True
    except Exception as exc:
        errors.append(f"raw:{type(exc).__name__}:{exc}")

    repaired_valid = False
    repair_applied = repaired_completion is not None
    if repaired_completion is not None:
        try:
            repaired_graph = decode_express_completion(repaired_completion)
            repaired_valid = True
            if graph is None:
                graph = repaired_graph
        except Exception as exc:
            errors.append(f"repaired:{type(exc).__name__}:{exc}")

    semantic = None
    if graph is not None:
        semantic = hashlib.sha256(
            json.dumps(
                graph,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    return ActiveValidation(
        raw_valid=raw_valid,
        repaired_valid=repaired_valid,
        canonical_graph=graph,
        errors=tuple(errors),
        semantic_hash=semantic,
        repair_applied=repair_applied,
    )


def compile_express_to_wire(graph_or_completion: Mapping[str, Any] | str) -> Any:
    """Compile validated Express into the pinned standard A2UI wire stream."""

    graph = (
        decode_express_completion(graph_or_completion)
        if isinstance(graph_or_completion, str)
        else dict(graph_or_completion)
    )
    payload = a2ui_wire.encode(graph, shorten_ids=True)
    _validate_standard_wire_schema(payload)
    decoded = a2ui_wire.decode(payload)
    validation = coerce_and_validate(decoded)
    if not validation.is_valid:
        raise ValueError(validation.error or "Compiled A2UI wire payload is invalid")
    return payload


def _validate_standard_wire_schema(payload: Any) -> None:
    schema_path = Path(__file__).resolve().parents[3] / "schema" / "genuicraft_a2ui_v1_wire.schema.json"
    try:
        import jsonschema  # type: ignore
    except ImportError as exc:
        raise ValueError("jsonschema is required for standard A2UI wire validation") from exc
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(payload)


__all__ = [
    "ACTIVE_FORMAT_ID",
    "WIRE_FORMAT_ID",
    "ActiveValidation",
    "compile_express_to_wire",
    "decode_express_completion",
    "encode_express_completion",
    "validate_express_completion",
]
