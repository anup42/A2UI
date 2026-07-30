"""Raw-envelope evidence layered over the shared production candidate boundary."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

from ._core import completion_to_text
from .candidate_normalization import (
    CandidateNormalizationResult,
    normalize_and_validate_candidate,
)


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
class CandidateNormalizationResultV54:
    boundary: CandidateNormalizationResult
    raw_envelope: RawJsonEnvelopeEvidence

    def __getattr__(self, name: str) -> Any:
        return getattr(self.boundary, name)


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
            completion, strict_schema=strict_schema
        ),
        raw_envelope=raw_json_envelope_evidence(completion),
    )


__all__ = [
    "CandidateNormalizationResultV54",
    "NORMALIZATION_POLICY_VERSION_V54",
    "RAW_ENVELOPE_POLICY_VERSION",
    "RawJsonEnvelopeEvidence",
    "normalize_and_validate_candidate_v5_4",
    "raw_json_envelope_evidence",
]
