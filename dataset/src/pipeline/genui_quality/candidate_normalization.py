"""Shared candidate parsing, production canonicalization, and strict validation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..flat_spec_contract import (
    coerce_and_validate,
    extract_json_element,
    normalize_to_flat_spec,
)
from ..ir_formats import decode_to_flat_spec
from ._core import completion_to_text
from .graph import audit_renderer_graph


NORMALIZATION_POLICY_VERSION = "1.0.1"
LEGACY_METRIC_CANONICALIZATION = "legacy_metric_v1"
RENDERER_V2_CANONICALIZATION = "renderer_v2"
DEFAULT_STRICT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[3] / "schema" / "genui_flatspec.schema.json"
)
STRICT_TOP_LEVEL_PROPERTIES = frozenset({"root", "state", "elements"})
_LEGACY_LAYOUT_ALIASES = {
    "column": ("Column", "vertical"),
    "row": ("Row", "horizontal"),
}


@dataclass(frozen=True)
class CandidateNormalizationResult:
    raw_parse_ok: bool
    canonical_spec: dict[str, Any] | None
    production_valid: bool
    strict_schema_valid: bool
    converted_from_legacy: bool
    raw_format_utility: float
    errors: tuple[str, ...]
    raw_hash: str
    canonical_hash: str | None


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _raw_identity(completion: Any) -> tuple[str, str]:
    if isinstance(completion, (Mapping, list, tuple)):
        try:
            value = _canonical_json(completion)
            return value, _hash_text(value)
        except (TypeError, ValueError):
            pass
    value = completion_to_text(completion)
    return value, _hash_text(value)


@lru_cache(maxsize=8)
def _load_schema_at_state(path: str, mtime_ns: int, size: int) -> dict[str, Any]:
    del mtime_ns, size
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _schema_state(path: str) -> tuple[str, int, int]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return str(resolved), stat.st_mtime_ns, stat.st_size


def _load_schema(path: str) -> dict[str, Any]:
    return _load_schema_at_state(*_schema_state(path))


@lru_cache(maxsize=8)
def _validator_at_state(path: str, mtime_ns: int, size: int) -> Any:
    try:
        import jsonschema  # type: ignore
    except ImportError:
        return None
    return jsonschema.Draft202012Validator(
        _load_schema_at_state(path, mtime_ns, size)
    )


def _validator(path: str) -> Any:
    return _validator_at_state(*_schema_state(path))


def load_strict_schema(path: Path | None = None) -> dict[str, Any]:
    resolved = str((path or DEFAULT_STRICT_SCHEMA_PATH).resolve())
    return dict(_load_schema(resolved))


def _strict_validate(
    value: Any,
    schema: Mapping[str, Any] | None,
) -> tuple[bool, list[str]]:
    if not isinstance(value, Mapping):
        return False, ["strict_schema.non_object"]
    try:
        import jsonschema  # type: ignore
    except ImportError:
        required = {"root", "elements"}
        unknown = set(value) - STRICT_TOP_LEVEL_PROPERTIES
        ok = required.issubset(value) and not unknown
        return ok, ([] if ok else ["strict_schema.validator_unavailable_shape_failure"])

    if schema is None:
        validator = _validator(str(DEFAULT_STRICT_SCHEMA_PATH.resolve()))
    else:
        validator = jsonschema.Draft202012Validator(dict(schema))
    assert validator is not None
    failures = sorted(validator.iter_errors(dict(value)), key=lambda item: list(item.path))
    return (
        not failures,
        [
            "strict_schema."
            + (".".join(str(part) for part in failure.path) or "root")
            + ":"
            + failure.validator
            for failure in failures
        ],
    )


def _layout_alias_token(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return "".join(character for character in value.strip() if character.isalnum()).lower()


def _strict_legacy_candidate(value: Any) -> Any:
    """Represent legacy layout aliases in the renderer-v2 strict schema.

    Historical metric versions accepted ``Row`` and ``Column`` as strict
    element types. The renderer-v2 schema expresses those as compatibility
    aliases for ``Stack``. Translating only for schema validation preserves the
    historical acceptance decision without weakening any other constraint.
    """

    if not isinstance(value, Mapping):
        return value
    candidate = deepcopy(dict(value))
    elements = candidate.get("elements")
    if not isinstance(elements, dict):
        return candidate
    for element in elements.values():
        if not isinstance(element, dict):
            continue
        raw_type = element.get("type") or element.get("component")
        # Preserve the historical metric contract for the explicit legacy
        # `Row`/`Column` aliases, while lower-case DSL/web aliases remain a
        # strict-format failure.
        if raw_type not in {"Row", "Column"}:
            continue
        token = _layout_alias_token(raw_type)
        descriptor = _LEGACY_LAYOUT_ALIASES.get(token)
        if descriptor is None:
            continue
        _, direction = descriptor
        if "type" in element:
            element["type"] = "Stack"
        elif "component" in element:
            element["component"] = "Stack"
        props = element.get("props")
        props = dict(props) if isinstance(props, Mapping) else {}
        props.setdefault("direction", direction)
        element["props"] = props
    return candidate


def _restore_legacy_layout_aliases(
    parsed: Any,
    canonical_spec: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Restore the canonical payload shape used by metric v5-v5.3."""

    if canonical_spec is None or not isinstance(parsed, Mapping):
        return canonical_spec
    raw_elements = parsed.get("elements")
    if not isinstance(raw_elements, Mapping):
        return canonical_spec
    restored = deepcopy(canonical_spec)
    canonical_elements = restored.get("elements")
    if not isinstance(canonical_elements, dict):
        return restored
    for element_id, raw_element in raw_elements.items():
        if not isinstance(raw_element, Mapping):
            continue
        token = _layout_alias_token(raw_element.get("type") or raw_element.get("component"))
        descriptor = _LEGACY_LAYOUT_ALIASES.get(token)
        canonical_element = canonical_elements.get(str(element_id))
        if descriptor is None or not isinstance(canonical_element, dict):
            continue
        legacy_type, direction = descriptor
        canonical_element["type"] = legacy_type
        raw_props = raw_element.get("props")
        direction_was_explicit = isinstance(raw_props, Mapping) and "direction" in raw_props
        props = canonical_element.get("props")
        if not direction_was_explicit and isinstance(props, dict) and props.get("direction") == direction:
            props.pop("direction", None)
    return restored


def _parse_completion(completion: Any, raw_text: str) -> tuple[Any, bool, str | None]:
    if not raw_text.strip() and not isinstance(completion, (Mapping, list, tuple)):
        return None, False, "parse.empty_completion"
    # All model-facing formats cross the same canonical boundary before metrics.
    try:
        candidate: Any = completion
        if isinstance(completion, tuple):
            candidate = list(completion)
        # Preserve the original canonical graph for strict validation. In
        # particular, do not let a decoder silently drop unknown top-level
        # fields or aliases before the strict gate sees them.
        if isinstance(candidate, Mapping) and {
            "root",
            "elements",
        }.issubset(candidate):
            return dict(candidate), True, None
        if isinstance(candidate, str) and candidate.lstrip().startswith(("{", "[")):
            parsed_json = extract_json_element(candidate)
            if isinstance(parsed_json, Mapping) and {"root", "elements"}.issubset(parsed_json):
                return dict(parsed_json), True, None
        decoded = decode_to_flat_spec(candidate)
        return decoded.flat_spec, True, None
    except Exception:
        pass
    if isinstance(completion, Mapping):
        return dict(completion), True, None
    if isinstance(completion, (list, tuple)):
        return list(completion), True, None
    try:
        parsed = extract_json_element(raw_text)
        if isinstance(parsed, Mapping) and {"root", "elements"}.issubset(parsed):
            return dict(parsed), True, None
        try:
            return decode_to_flat_spec(parsed).flat_spec, True, None
        except Exception:
            return parsed, True, None
    except Exception as exc:
        return None, False, f"parse.invalid_ir:{type(exc).__name__}"


def normalize_and_validate_candidate(
    completion: Any,
    *,
    strict_schema: Mapping[str, Any] | None = None,
    canonicalization_profile: str = LEGACY_METRIC_CANONICALIZATION,
) -> CandidateNormalizationResult:
    """Apply the candidate boundary pinned to the caller's metric version."""

    if canonicalization_profile not in {
        LEGACY_METRIC_CANONICALIZATION,
        RENDERER_V2_CANONICALIZATION,
    }:
        raise ValueError(
            f"Unsupported canonicalization profile: {canonicalization_profile}"
        )

    raw_text, raw_hash = _raw_identity(completion)
    parsed, raw_parse_ok, parse_error = _parse_completion(completion, raw_text)
    errors: list[str] = []
    if parse_error:
        errors.append(parse_error)
    if not raw_parse_ok or not isinstance(parsed, (Mapping, list)):
        if raw_parse_ok:
            errors.append("parse.non_object")
        return CandidateNormalizationResult(
            raw_parse_ok=False,
            canonical_spec=None,
            production_valid=False,
            strict_schema_valid=False,
            converted_from_legacy=False,
            raw_format_utility=0.0,
            errors=tuple(errors),
            raw_hash=raw_hash,
            canonical_hash=None,
        )

    strict_candidate = (
        _strict_legacy_candidate(parsed)
        if canonicalization_profile == LEGACY_METRIC_CANONICALIZATION
        else parsed
    )
    strict_valid, strict_errors = _strict_validate(strict_candidate, strict_schema)
    errors.extend(strict_errors)
    if isinstance(parsed, Mapping):
        for key in sorted(set(parsed) - STRICT_TOP_LEVEL_PROPERTIES):
            errors.append(f"strict_schema.unknown_top_level_property:{key}")

    coerce_result = coerce_and_validate(parsed)
    canonical_spec = coerce_result.spec
    if canonicalization_profile == LEGACY_METRIC_CANONICALIZATION:
        canonical_spec = _restore_legacy_layout_aliases(parsed, canonical_spec)
    production_valid = coerce_result.is_valid
    converted_from_legacy = bool(coerce_result.converted_from_legacy)
    if not production_valid:
        errors.append(f"production_validation:{coerce_result.error or 'failed'}")
        normalized = normalize_to_flat_spec(parsed)
        canonical_spec = normalized.spec
        converted_from_legacy = bool(normalized.converted_from_legacy)

    if canonical_spec is not None:
        audit = audit_renderer_graph(canonical_spec)
        if not audit.root_exists:
            production_valid = False
            errors.append("renderer_reference.missing_root")
        if audit.missing_references:
            production_valid = False
            for target in sorted(audit.missing_references):
                errors.append(f"renderer_reference.missing:{target}")
        if audit.cycle_edges:
            production_valid = False
            for source, target in sorted(audit.cycle_edges):
                errors.append(f"renderer_reference.cycle:{source}->{target}")

    canonical_hash: str | None = None
    if canonical_spec is not None:
        try:
            canonical_hash = _hash_text(_canonical_json(canonical_spec))
        except (TypeError, ValueError):
            canonical_spec = None
            production_valid = False
            errors.append("canonicalization.non_finite_or_non_json")

    raw_format_utility = (
        1.0
        if strict_valid and production_valid
        else 0.80
        if production_valid
        else 0.35
        if canonical_spec is not None
        else 0.0
    )
    return CandidateNormalizationResult(
        raw_parse_ok=True,
        canonical_spec=canonical_spec,
        production_valid=production_valid,
        strict_schema_valid=strict_valid,
        converted_from_legacy=converted_from_legacy,
        raw_format_utility=raw_format_utility,
        errors=tuple(dict.fromkeys(errors)),
        raw_hash=raw_hash,
        canonical_hash=canonical_hash,
    )


__all__ = [
    "CandidateNormalizationResult",
    "DEFAULT_STRICT_SCHEMA_PATH",
    "LEGACY_METRIC_CANONICALIZATION",
    "NORMALIZATION_POLICY_VERSION",
    "RENDERER_V2_CANONICALIZATION",
    "STRICT_TOP_LEVEL_PROPERTIES",
    "load_strict_schema",
    "normalize_and_validate_candidate",
]
