"""Symmetric source/output semantic evidence ownership for metric v5.4."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Mapping, Sequence

from . import _core


OWNERSHIP_POLICY_VERSION = "5.4.0"
SUPPORTED_OWNERS = frozenset(
    {
        "generic_prose",
        "heading",
        "table_header",
        "table_row",
        "table_cell",
        "chart_directive",
        "chart_data",
        "formula",
        "code",
        "console",
        "email",
        "action",
        "media",
        "form",
        "local_ui_mechanic",
    }
)

_FENCE = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)
_HEADING = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$")
_TABLE_SEPARATOR = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
_DIRECTIVE = re.compile(
    r"(?i)^\s*(?:[-*+]\s*)?(?:#{1,6}\s*)?"
    r"(chart|graph|plot|visualization|formula|equation|"
    r"code(?:\s+block)?|console(?:\s+(?:log|output))?|"
    r"email(?:\s+preview)?|action|media|form|controls?)\b"
)
_ROLE_SECTION = re.compile(
    r"(?i)^\s*(?:#{1,6}\s*)?"
    r"(?:charts?|graphs?|plots?|visualizations?|formulas?|equations?|"
    r"code(?:\s+(?:blocks?|examples?))?|console(?:\s+(?:logs?|outputs?))?|"
    r"email(?:\s+previews?)?|forms?|controls?)"
    r"(?:\s+to\s+include)?\s*:?\s*$"
)
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.+?)\s*$")
_CHART_REQUEST = re.compile(
    r"(?i)(?:\(\s*(?:as\s+(?:a\s+)?)?"
    r"(?:stacked\s+bar|horizontal\s+bar|vertical\s+bar|bar|line|column|"
    r"scatter|pie|area|donut|doughnut)?\s*chart\s*\)\s*$)|"
    r"(?:\b(?:create|show|include|provide|display|use)\s+"
    r"(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"\s+charts?\s*:)"
)


@dataclass(frozen=True)
class SemanticEvidenceUnit:
    unit_id: str
    owner: str
    text: str
    normalized_text: str
    exact_values: tuple[str, ...]
    source_span: tuple[int, int] | None
    semantic_key: str | None


def _normalized(value: Any) -> str:
    return _core.normalize_match_text(value)


def _unit(
    owner: str,
    text: Any,
    *,
    source_span: tuple[int, int] | None,
    semantic_key: str | None,
    ordinal: int,
) -> SemanticEvidenceUnit | None:
    raw = str(text or "").strip()
    normalized = _normalized(raw)
    if owner not in SUPPORTED_OWNERS or not normalized:
        return None
    exact = [
        _normalized(value)
        for value in _core._EXACT_VALUE_RE.findall(raw)  # type: ignore[attr-defined]
    ]
    exact.extend(
        _normalized(value)
        for value in _core._DATE_RE.findall(raw)  # type: ignore[attr-defined]
    )
    digest = hashlib.sha256(
        f"{owner}\0{semantic_key or ''}\0{ordinal}\0{normalized}".encode("utf-8")
    ).hexdigest()[:20]
    return SemanticEvidenceUnit(
        unit_id=f"{owner}_{digest}",
        owner=owner,
        text=raw,
        normalized_text=normalized,
        exact_values=tuple(value for value in exact if value),
        source_span=source_span,
        semantic_key=semantic_key,
    )


def _line_spans(text: str) -> list[tuple[int, int, str]]:
    result: list[tuple[int, int, str]] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        end = cursor + len(line)
        result.append((cursor, end, line.rstrip("\r\n")))
        cursor = end
    if cursor < len(text):
        result.append((cursor, len(text), text[cursor:]))
    return result


def _markdown_table_spans(text: str) -> list[tuple[int, int]]:
    lines = _line_spans(text)
    result: list[tuple[int, int]] = []
    for index, (_, _, line) in enumerate(lines):
        if not _TABLE_SEPARATOR.match(line) or index == 0:
            continue
        start_index = index - 1
        end_index = index + 1
        while end_index < len(lines) and "|" in lines[end_index][2]:
            end_index += 1
        result.append((lines[start_index][0], lines[end_index - 1][1]))
    return result


def _span_for_index(
    spans: Sequence[tuple[int, int]],
    index: int,
) -> tuple[int, int] | None:
    return spans[index] if index < len(spans) else None


def build_source_ownership(
    source_text: str,
    contract: Mapping[str, Any],
) -> tuple[SemanticEvidenceUnit, ...]:
    """Assign every source-side fact to exactly one scoring channel."""

    text = str(source_text or "")
    claimed: list[tuple[int, int]] = []
    units: list[SemanticEvidenceUnit] = []
    ordinal = 0

    def add(
        owner: str,
        value: Any,
        *,
        span: tuple[int, int] | None = None,
        key: str | None = None,
    ) -> None:
        nonlocal ordinal
        item = _unit(
            owner,
            value,
            source_span=span,
            semantic_key=key,
            ordinal=ordinal,
        )
        ordinal += 1
        if item is not None:
            units.append(item)

    fences = [match.span() for match in _FENCE.finditer(text)]
    for index, raw in enumerate(contract.get("code_blocks") or []):
        if isinstance(raw, Mapping):
            body = raw.get("code")
        else:
            body = raw
        span = _span_for_index(fences, index)
        add("code", body, span=span, key=f"code:{index}")
        if span is not None:
            claimed.append(span)

    table_spans = _markdown_table_spans(text)
    for table_index, raw in enumerate(contract.get("tables") or []):
        if not isinstance(raw, Mapping):
            continue
        span = _span_for_index(table_spans, table_index)
        semantic_key = str(raw.get("id") or f"table:{table_index}")
        headers = raw.get("headers") or []
        rows = raw.get("rows") or []
        for header_index, header in enumerate(headers):
            add(
                "table_header",
                header,
                span=span,
                key=f"{semantic_key}:header:{header_index}",
            )
        for row_index, row in enumerate(rows):
            values = (
                list(row.values())
                if isinstance(row, Mapping)
                else list(row)
                if isinstance(row, Sequence)
                and not isinstance(row, (str, bytes, bytearray))
                else [row]
            )
            add(
                "table_row",
                " | ".join(str(value) for value in values),
                span=span,
                key=f"{semantic_key}:row:{row_index}",
            )
            for cell_index, cell in enumerate(values):
                add(
                    "table_cell",
                    cell,
                    span=span,
                    key=f"{semantic_key}:cell:{row_index}:{cell_index}",
                )
        if span is not None:
            claimed.append(span)

    role_owner = {
        "chart": "chart_directive",
        "formula": "formula",
        "code": "code",
        "console": "console",
        "email": "email",
        "form": "form",
    }
    for role, requirements in (contract.get("role_requirements") or {}).items():
        owner = role_owner.get(str(role).casefold())
        if owner is None or not isinstance(requirements, Sequence):
            continue
        for index, requirement in enumerate(requirements):
            if not isinstance(requirement, Mapping):
                continue
            payload = " ".join(
                str(value)
                for key, value in requirement.items()
                if key
                not in {
                    "id",
                    "required",
                    "minimum_count",
                    "interchangeable",
                    "source_section",
                    "expected_component_id",
                }
                and value not in (None, "", [], {})
            )
            add(
                owner,
                payload,
                key=str(requirement.get("id") or f"{role}:{index}"),
            )

    for index, action in enumerate(contract.get("actions") or []):
        if isinstance(action, Mapping):
            add(
                "action",
                f"{action.get('label', '')} {action.get('target', '')}",
                key=str(action.get("id") or f"action:{index}"),
            )
    for index, media in enumerate(contract.get("media") or []):
        if isinstance(media, Mapping):
            add(
                "media",
                f"{media.get('kind', '')} {media.get('url', '')} {media.get('alt', '')}",
                key=str(media.get("id") or f"media:{index}"),
            )

    lines = _line_spans(text)
    in_role_section = False
    for start, end, raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            in_role_section = False
            continue
        if any(start < claim_end and end > claim_start for claim_start, claim_end in claimed):
            continue
        heading = _HEADING.match(raw_line)
        if heading:
            if _ROLE_SECTION.match(raw_line) or _CHART_REQUEST.search(raw_line):
                in_role_section = True
                owner = (
                    "chart_directive"
                    if re.search(
                        r"(?i)charts?|graphs?|plots?|visualizations?",
                        raw_line,
                    )
                    else "form"
                )
                add(owner, heading.group(1), span=(start, end))
            else:
                add("heading", heading.group(1), span=(start, end))
            claimed.append((start, end))
            continue
        if _ROLE_SECTION.match(raw_line):
            in_role_section = True
            owner = "chart_directive" if re.search(
                r"(?i)charts?|graphs?|plots?|visualizations?", raw_line
            ) else "form"
            add(owner, stripped, span=(start, end))
            claimed.append((start, end))
            continue
        if in_role_section and _BULLET.match(raw_line):
            owner = "chart_directive"
            add(owner, _BULLET.match(raw_line).group(1), span=(start, end))  # type: ignore[union-attr]
            claimed.append((start, end))
            continue
        if _CHART_REQUEST.search(raw_line):
            add("chart_directive", stripped, span=(start, end))
            claimed.append((start, end))
            continue
        directive = _DIRECTIVE.match(raw_line)
        if directive:
            token = directive.group(1).casefold()
            owner = (
                "chart_directive"
                if token.startswith(("chart", "graph", "plot", "visual"))
                else "formula"
                if token.startswith(("formula", "equation"))
                else "code"
                if token.startswith("code")
                else "console"
                if token.startswith("console")
                else "email"
                if token.startswith("email")
                else "action"
                if token.startswith("action")
                else "media"
                if token.startswith("media")
                else "form"
            )
            add(owner, stripped, span=(start, end))
            claimed.append((start, end))
            continue
        add("generic_prose", stripped, span=(start, end))

    return tuple(units)


def generic_source_texts(
    source_text: str,
    contract: Mapping[str, Any],
) -> tuple[str, ...]:
    return tuple(
        item.text
        for item in build_source_ownership(source_text, contract)
        if item.owner == "generic_prose"
    )


def build_output_ownership(
    evidence_result: Any,
) -> tuple[SemanticEvidenceUnit, ...]:
    units: list[SemanticEvidenceUnit] = []
    owner_map = {
        "generic_visible_content": "generic_prose",
        "source_table": "table_row",
        "source_chart": "chart_data",
        "source_action": "action",
        "source_media": "media",
        "source_formula": "formula",
        "source_code": "code",
        "source_console": "console",
        "source_email": "email",
        "semantic_role": "formula",
        "local_ui_mechanic": "local_ui_mechanic",
    }
    for index, raw in enumerate(evidence_result.evidence_ownership):
        owner = owner_map.get(str(raw.get("owner") or ""), "generic_prose")
        item = _unit(
            owner,
            raw.get("text"),
            source_span=None,
            semantic_key=str(raw.get("component_id") or ""),
            ordinal=index,
        )
        if item is not None:
            units.append(item)
    return tuple(units)


def _token_similarity(left: str, right: str) -> float:
    return _token_set_similarity(
        set(_core.tokenize(left)),
        set(_core.tokenize(right)),
    )


def _token_set_similarity(
    left_tokens: set[str],
    right_tokens: set[str],
) -> float:
    if not left_tokens or not right_tokens:
        return 0.0
    precision = len(left_tokens & right_tokens) / len(right_tokens)
    recall = len(left_tokens & right_tokens) / len(left_tokens)
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def cross_channel_duplication_utility(
    source_units: Sequence[SemanticEvidenceUnit],
    output_units: Sequence[SemanticEvidenceUnit],
) -> tuple[float, dict[str, Any]]:
    dedicated_owners = {
        "table_header",
        "table_row",
        "table_cell",
        "chart_directive",
        "chart_data",
        "formula",
        "code",
        "console",
        "email",
        "action",
        "media",
        "form",
    }
    dedicated = [
        item for item in output_units if item.owner in dedicated_owners
    ]
    generic = [item for item in output_units if item.owner == "generic_prose"]
    required_generic = [
        item for item in source_units if item.owner == "generic_prose"
    ]
    required_exact = {
        item.normalized_text for item in required_generic
    }
    required_token_sets = [
        set(_core.tokenize(item.normalized_text))
        for item in required_generic
    ]
    inverted: dict[str, list[int]] = {}
    for index, tokens in enumerate(required_token_sets):
        for token in sorted(tokens):
            inverted.setdefault(token, []).append(index)
    max_postings_per_token = 64
    max_similarity_candidates = 64
    dedicated_tokens = {
        token
        for item in dedicated
        for token in _core.tokenize(item.normalized_text)
    }
    dedicated_exact_values = {
        value for item in dedicated for value in item.exact_values
    }
    redundant: list[str] = []
    bounded_candidate_search_count = 0
    for candidate in generic:
        if candidate.normalized_text in required_exact:
            continue
        candidate_tokens = set(_core.tokenize(candidate.normalized_text))
        overlaps: dict[int, int] = {}
        for token in sorted(candidate_tokens):
            postings = inverted.get(token, ())
            if len(postings) > max_postings_per_token:
                bounded_candidate_search_count += 1
            for index in postings[:max_postings_per_token]:
                overlaps[index] = overlaps.get(index, 0) + 1
        comparison_indices = [
            index
            for index, _ in sorted(
                overlaps.items(),
                key=lambda item: (-item[1], item[0]),
            )[:max_similarity_candidates]
        ]
        if any(
            _token_set_similarity(
                required_token_sets[index], candidate_tokens
            )
            >= 0.70
            for index in comparison_indices
        ):
            continue
        dedicated_similarity = _token_set_similarity(
            dedicated_tokens, candidate_tokens
        )
        if dedicated_similarity >= 0.80 or (
            candidate.exact_values
            and set(candidate.exact_values).issubset(
                dedicated_exact_values
            )
            and dedicated_similarity >= 0.60
        ):
            redundant.append(candidate.unit_id)
    utility = 1.0 - len(redundant) / max(1, len(generic))
    return max(0.0, min(1.0, utility)), {
        "policy_version": OWNERSHIP_POLICY_VERSION,
        "generic_output_count": len(generic),
        "dedicated_output_count": len(dedicated),
        "redundant_generic_unit_ids": redundant,
        "required_generic_count": len(required_generic),
        "matching_policy": "exact_then_bounded_token_inverted_index",
        "max_postings_per_token": max_postings_per_token,
        "max_similarity_candidates": max_similarity_candidates,
        "bounded_candidate_search_count": (
            bounded_candidate_search_count
        ),
    }


__all__ = [
    "OWNERSHIP_POLICY_VERSION",
    "SUPPORTED_OWNERS",
    "SemanticEvidenceUnit",
    "build_output_ownership",
    "build_source_ownership",
    "cross_channel_duplication_utility",
    "generic_source_texts",
]
