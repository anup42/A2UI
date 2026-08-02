"""Raw-envelope evidence layered over the shared production candidate boundary."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

from ._core import completion_to_text
from .candidate_normalization import (
    CandidateNormalizationResult,
    RENDERER_V2_CANONICALIZATION,
    _canonical_json,
    _hash_text,
    _strict_validate,
    normalize_and_validate_candidate,
)
from .graph import audit_renderer_graph


NORMALIZATION_POLICY_VERSION_V54 = "2.0.0"
RAW_ENVELOPE_POLICY_VERSION = "1.0.0"


@dataclass(frozen=True)
class RawJsonEnvelopeEvidence:
    exact_single_json_value: bool
    consumed_start: int
    consumed_end: int
    leading_non_whitespace: bool
    trailing_non_whitespace: bool
    markdown_fence_present: bool
    extra_json_value_present: bool


@dataclass(frozen=True)
class RawExpressEnvelopeEvidence:
    exact_single_express_block: bool
    consumed_start: int
    consumed_end: int
    leading_non_whitespace: bool
    trailing_non_whitespace: bool
    markdown_fence_present: bool
    extra_express_block_present: bool

    @property
    def exact_single_json_value(self) -> bool:
        # Compatibility field for older dashboards; active code uses the
        # explicit Express field above.
        return self.exact_single_express_block

    @property
    def extra_json_value_present(self) -> bool:
        return self.extra_express_block_present


@dataclass(frozen=True)
class CandidateNormalizationResultV54:
    boundary: CandidateNormalizationResult
    raw_envelope: RawJsonEnvelopeEvidence

    def __getattr__(self, name: str) -> Any:
        return getattr(self.boundary, name)

    @property
    def raw_valid(self) -> bool:
        return bool(self.boundary.raw_parse_ok)

    @property
    def repaired_valid(self) -> bool:
        # Repair is an explicit caller-owned second pass; normalization never
        # silently repairs a candidate.
        return False


def raw_express_envelope_evidence(completion: Any) -> RawExpressEnvelopeEvidence:
    text = completion_to_text(completion)
    open_token = "<a2ui>"
    close_token = "</a2ui>"
    start = text.find(open_token)
    close_start = text.find(close_token, start + len(open_token)) if start >= 0 else -1
    end = close_start + len(close_token) if close_start >= 0 else -1
    leading = bool(text[:start].strip()) if start >= 0 else bool(text.strip())
    trailing = bool(text[end:].strip()) if end >= 0 else False
    remainder = (text[:start] + text[end:]) if start >= 0 and end >= 0 else ""
    extra = open_token in remainder or close_token in remainder
    fenced = "```" in text
    exact = (
        isinstance(completion, str)
        and start == 0
        and end == len(text)
        and not leading
        and not trailing
        and not fenced
        and not extra
    )
    return RawExpressEnvelopeEvidence(
        exact_single_express_block=exact,
        consumed_start=start,
        consumed_end=end,
        leading_non_whitespace=leading,
        trailing_non_whitespace=trailing,
        markdown_fence_present=fenced,
        extra_express_block_present=extra,
    )


def normalize_and_validate_express_candidate_v5_4(
    completion: Any,
    *,
    strict_schema: Mapping[str, Any] | None = None,
) -> CandidateNormalizationResultV54:
    """Strict active Express scoring boundary (no JSON extraction/fallback)."""
    from ..ir_formats import compile_express_to_wire, validate_express_completion

    raw_text = completion_to_text(completion)
    validation = validate_express_completion(completion)
    envelope = raw_express_envelope_evidence(completion)
    errors = list(validation.errors)
    canonical = validation.canonical_graph if validation.raw_valid else None
    standard_valid = False
    if canonical is not None:
        try:
            compile_express_to_wire(completion)
            standard_valid = True
        except Exception as exc:
            errors.append(f"standard_a2ui:{type(exc).__name__}:{exc}")
        audit = audit_renderer_graph(canonical)
        if audit.missing_references:
            standard_valid = False
            errors.extend(f"renderer_reference.missing:{item}" for item in sorted(audit.missing_references))
        if audit.cycle_edges:
            standard_valid = False
            errors.extend(f"renderer_reference.cycle:{source}->{target}" for source, target in sorted(audit.cycle_edges))
    strict_valid, strict_errors = _strict_validate(canonical, strict_schema)
    errors.extend(strict_errors)
    production_valid = bool(validation.raw_valid and standard_valid and strict_valid)
    canonical_hash = _hash_text(_canonical_json(canonical)) if canonical is not None else None
    if not production_valid and not errors:
        errors.append("express.production_invalid")
    return CandidateNormalizationResultV54(
        boundary=CandidateNormalizationResult(
            raw_parse_ok=validation.raw_valid,
            canonical_spec=canonical,
            production_valid=production_valid,
            strict_schema_valid=strict_valid,
            converted_from_legacy=False,
            raw_format_utility=1.0 if envelope.exact_single_express_block and production_valid else 0.0,
            errors=tuple(dict.fromkeys(errors)),
            raw_hash=_hash_text(raw_text),
            canonical_hash=canonical_hash,
        ),
        raw_envelope=envelope,
    )


def _json_values(text: str) -> list[tuple[int, int, Any]]:
    decoder = json.JSONDecoder()
    values: list[tuple[int, int, Any]] = []
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, consumed = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        end = index + consumed
        if any(index >= start and end <= previous_end for start, previous_end, _ in values):
            continue
        values.append((index, end, value))
    return values


def _candidate_rank(value: Any) -> tuple[int, int]:
    if not isinstance(value, Mapping):
        return (0, 0)
    keys = set(value)
    return (
        int("root" in keys and "elements" in keys),
        int("elements" in keys) + int("root" in keys) + int("state" in keys),
    )


def raw_json_envelope_evidence(completion: Any) -> RawJsonEnvelopeEvidence:
    if isinstance(completion, Mapping):
        return RawJsonEnvelopeEvidence(
            exact_single_json_value=True,
            consumed_start=0,
            consumed_end=0,
            leading_non_whitespace=False,
            trailing_non_whitespace=False,
            markdown_fence_present=False,
            extra_json_value_present=False,
        )
    text = completion_to_text(completion)
    values = _json_values(text)
    if not values:
        return RawJsonEnvelopeEvidence(
            exact_single_json_value=False,
            consumed_start=-1,
            consumed_end=-1,
            leading_non_whitespace=bool(text.strip()),
            trailing_non_whitespace=False,
            markdown_fence_present="```" in text,
            extra_json_value_present=False,
        )
    selected = max(
        enumerate(values),
        key=lambda item: (
            _candidate_rank(item[1][2]),
            item[1][1] - item[1][0],
            -item[0],
        ),
    )[1]
    start, end, _ = selected
    leading = bool(text[:start].strip())
    trailing = bool(text[end:].strip())
    extra = any(
        (other_end <= start or other_start >= end)
        and isinstance(value, (Mapping, list))
        for other_start, other_end, value in values
        if (other_start, other_end) != (start, end)
    )
    fenced = "```" in text
    exact = not leading and not trailing and not fenced and not extra
    return RawJsonEnvelopeEvidence(
        exact_single_json_value=exact,
        consumed_start=start,
        consumed_end=end,
        leading_non_whitespace=leading,
        trailing_non_whitespace=trailing,
        markdown_fence_present=fenced,
        extra_json_value_present=extra,
    )


def normalize_and_validate_candidate_v5_4(
    completion: Any,
    *,
    strict_schema: Mapping[str, Any] | None = None,
) -> CandidateNormalizationResultV54:
    return CandidateNormalizationResultV54(
        boundary=normalize_and_validate_candidate(
            completion,
            strict_schema=strict_schema,
            canonicalization_profile=RENDERER_V2_CANONICALIZATION,
        ),
        raw_envelope=raw_json_envelope_evidence(completion),
    )


__all__ = [
    "CandidateNormalizationResultV54",
    "NORMALIZATION_POLICY_VERSION_V54",
    "RAW_ENVELOPE_POLICY_VERSION",
    "RawJsonEnvelopeEvidence",
    "RawExpressEnvelopeEvidence",
    "normalize_and_validate_express_candidate_v5_4",
    "normalize_and_validate_candidate_v5_4",
    "raw_json_envelope_evidence",
    "raw_express_envelope_evidence",
]
