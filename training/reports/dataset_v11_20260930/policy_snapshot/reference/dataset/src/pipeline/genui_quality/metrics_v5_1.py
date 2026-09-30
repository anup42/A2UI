"""Association-preserving full-set metrics for GenUI quality v5.1."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from . import _core
from .matching_v5_1 import (
    AssignmentResult,
    PreparedTextBlock,
    build_candidate_edges,
    maximum_weight_assignment,
    prepare_text_block,
)


SCORING_POLICY_VERSION = "5.1.0"


def clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def f_score(precision: float, recall: float, beta: float = 1.0) -> float:
    return _core.f_beta(clamp01(precision), clamp01(recall), beta)


def _counter_fbeta(
    expected: Counter[str],
    actual: Counter[str],
    beta: float,
) -> float:
    precision, recall = _core.counter_pr(actual, expected)
    return f_score(precision, recall, beta)


def _lcs_norm(left: Sequence[str], right: Sequence[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    previous = [0] * (len(right) + 1)
    for left_value in left:
        current = [0]
        for index, right_value in enumerate(right, start=1):
            current.append(
                previous[index - 1] + 1
                if left_value == right_value
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1] / max(len(left), len(right))


def prepared_block_similarity(
    source: PreparedTextBlock,
    output: PreparedTextBlock,
) -> float:
    unigram = _counter_fbeta(source.counter(), output.counter(), 2.0)
    bigram = _counter_fbeta(
        Counter(source.bigrams),
        Counter(output.bigrams),
        1.0,
    )
    lcs = _lcs_norm(source.tokens, output.tokens)
    return clamp01(0.45 * unigram + 0.35 * bigram + 0.20 * lcs)


def _cheap_text_similarity(
    source: PreparedTextBlock,
    output: PreparedTextBlock,
) -> float:
    if source.normalized == output.normalized:
        return 1.0
    left = set(source.tokens)
    right = set(output.tokens)
    return len(left & right) / max(1, len(left | right))


def _assignment_diagnostics(result: AssignmentResult) -> dict[str, Any]:
    return {
        "expected_count": result.expected_count,
        "actual_count": result.actual_count,
        "evaluated_edge_count": result.evaluated_edge_count,
        "candidate_edge_count": result.candidate_edge_count,
        "exact": result.exact,
        "complete": result.complete,
        "diagnostic_codes": list(result.diagnostic_codes),
    }


def content_fidelity_v5_1(
    source_units: Sequence[str] | Sequence[PreparedTextBlock],
    output_blocks: Sequence[str] | Sequence[PreparedTextBlock],
    *,
    exact_dense_limit: int,
    max_edges: int,
    top_k: int,
) -> tuple[dict[str, float | None], dict[str, Any]]:
    prepared_source = tuple(
        value if isinstance(value, PreparedTextBlock) else prepare_text_block(str(value))
        for value in source_units
    )
    prepared_output = tuple(
        value if isinstance(value, PreparedTextBlock) else prepare_text_block(str(value))
        for value in output_blocks
    )
    if not prepared_source and not prepared_output:
        return {
            "content_unit_fidelity": None,
            "content_order_preservation": None,
            "output_block_precision": None,
        }, {
            "matches": [],
            "matching": {
                "expected_count": 0,
                "actual_count": 0,
                "complete": True,
            },
        }

    edges, edge_codes = build_candidate_edges(
        prepared_source,
        prepared_output,
        score=lambda left, right: prepared_block_similarity(
            prepared_source[left], prepared_output[right]
        ),
        cheap_score=lambda left, right: _cheap_text_similarity(
            prepared_source[left], prepared_output[right]
        ),
        max_edges=max_edges,
        top_k=top_k,
    )
    assignment = maximum_weight_assignment(
        len(prepared_source),
        len(prepared_output),
        edges,
        exact_dense_limit=exact_dense_limit,
    )
    complete = assignment.complete
    mass = sum(edge.score for edge in assignment.matches)
    source_recall = mass / len(prepared_source) if prepared_source else 1.0
    output_precision = mass / len(prepared_output) if prepared_output else 1.0
    meaningful = [
        (edge.expected_index, edge.actual_index)
        for edge in assignment.matches
        if edge.score >= 0.15
    ]
    ordered = sorted(meaningful, key=lambda pair: pair[1])
    order_score = (
        _lcs_norm(
            [str(source) for source, _ in ordered],
            [str(source) for source in sorted(source for source, _ in meaningful)],
        )
        if len(meaningful) >= 2
        else 1.0
        if meaningful
        else 0.0
    )
    values: dict[str, float | None] = {
        "content_unit_fidelity": (
            f_score(output_precision, source_recall, 2.0) if complete else None
        ),
        "content_order_preservation": (
            order_score if prepared_source and complete else None
        ),
        "output_block_precision": (
            output_precision if prepared_output and complete else None
        ),
    }
    diagnostics = {
        "source_unit_recall": source_recall,
        "output_block_precision": output_precision,
        "assignment_truncated": False,
        "matching": {
            **_assignment_diagnostics(assignment),
            "diagnostic_codes": list(
                dict.fromkeys([*edge_codes, *assignment.diagnostic_codes])
            ),
        },
        "matches": [
            {
                "source_index": edge.expected_index,
                "output_index": edge.actual_index,
                "similarity": edge.score,
            }
            for edge in assignment.matches
        ],
    }
    return values, diagnostics


@dataclass(frozen=True)
class TablePairResult:
    score: float
    column_fbeta: float
    row_fbeta: float
    associated_cell_fbeta: float
    order_score: float | None
    key_precision: float | None
    key_recall: float | None
    key_exact_match_rate: float | None
    complete: bool
    diagnostic_codes: tuple[str, ...] = ()


def _output_rows(table: _core.OutputTable) -> list[list[str]]:
    return [
        [str(row.get(key, "")) for key in table.keys]
        for row in table.rows
    ]


def _normalized_key(value: Any) -> str:
    return _core.normalize_match_text(value)


def _find_header(headers: Sequence[str], name: str) -> int | None:
    normalized = _normalized_key(name)
    for index, header in enumerate(headers):
        if _normalized_key(header) == normalized:
            return index
    return None


def table_pair_metrics_v5_1(
    source: _core.SourceTable,
    output: _core.OutputTable,
    *,
    beta: float,
    exact_dense_limit: int,
    max_edges: int,
    top_k: int,
) -> TablePairResult:
    source_headers = tuple(prepare_text_block(value) for value in source.headers)
    output_headers = tuple(prepare_text_block(value) for value in output.headers)
    column_edges, column_codes = build_candidate_edges(
        source_headers,
        output_headers,
        score=lambda left, right: _core.text_similarity(
            source_headers[left].raw, output_headers[right].raw
        ),
        cheap_score=lambda left, right: _cheap_text_similarity(
            source_headers[left], output_headers[right]
        ),
        max_edges=max_edges,
        top_k=top_k,
    )
    columns = maximum_weight_assignment(
        len(source_headers),
        len(output_headers),
        column_edges,
        exact_dense_limit=exact_dense_limit,
    )
    column_mass = sum(edge.score for edge in columns.matches)
    column_precision = column_mass / len(output.headers) if output.headers else 0.0
    column_recall = column_mass / len(source.headers) if source.headers else 1.0
    column_f = f_score(column_precision, column_recall, beta)
    aligned = [
        (edge.expected_index, edge.actual_index)
        for edge in columns.matches
        if edge.score >= 0.20
    ]

    key_pairs: list[tuple[int, int]] = []
    key_diagnostics: list[str] = []
    for key_name in source.row_key:
        source_index = _find_header(source.headers, key_name)
        if source_index is None:
            key_diagnostics.append(f"row_key_source_column_missing:{key_name}")
            continue
        output_index = next(
            (
                actual_index
                for expected_index, actual_index in aligned
                if expected_index == source_index
            ),
            None,
        )
        if output_index is None:
            key_diagnostics.append(f"row_key_output_column_missing:{key_name}")
            continue
        key_pairs.append((source_index, output_index))

    source_rows = tuple(tuple(str(cell) for cell in row) for row in source.rows)
    output_rows = tuple(tuple(row) for row in _output_rows(output))
    non_key_aligned = [
        pair for pair in aligned if pair[0] not in {value[0] for value in key_pairs}
    ]

    def key_compatible(left: int, right: int) -> tuple[float, float]:
        if not source.row_key:
            return 1.0, 1.0
        if len(key_pairs) != len(source.row_key):
            return 0.0, 0.0
        similarities = [
            float(
                _normalized_key(
                    source_rows[left][source_index]
                    if source_index < len(source_rows[left])
                    else ""
                )
                == _normalized_key(
                    output_rows[right][output_index]
                    if output_index < len(output_rows[right])
                    else ""
                )
            )
            for source_index, output_index in key_pairs
        ]
        exact = float(all(value == 1.0 for value in similarities))
        return exact, sum(similarities) / len(similarities)

    def row_similarity(left: int, right: int) -> float:
        if not aligned:
            return 0.0
        gate, key_score = key_compatible(left, right)
        if gate <= 0.0:
            return 0.0
        pairs = non_key_aligned or key_pairs or aligned
        non_key = [
            _core.text_similarity(
                source_rows[left][source_index]
                if source_index < len(source_rows[left])
                else "",
                output_rows[right][output_index]
                if output_index < len(output_rows[right])
                else "",
            )
            for source_index, output_index in pairs
        ]
        content_score = sum(non_key) / len(non_key) if non_key else 1.0
        return clamp01(
            gate
            * (
                0.75 * content_score + 0.25 * key_score
                if source.row_key
                else content_score
            )
        )

    def cheap_row(left: int, right: int) -> float:
        gate, _ = key_compatible(left, right)
        if gate <= 0.0:
            return 0.0
        return _core.text_similarity(
            " ".join(source_rows[left]),
            " ".join(output_rows[right]),
        )

    row_edges, row_codes = build_candidate_edges(
        source_rows,
        output_rows,
        score=row_similarity,
        cheap_score=cheap_row,
        max_edges=max_edges,
        top_k=top_k,
    )
    rows = maximum_weight_assignment(
        len(source_rows),
        len(output_rows),
        row_edges,
        exact_dense_limit=exact_dense_limit,
    )
    row_mass = sum(edge.score for edge in rows.matches)
    row_precision = row_mass / len(output_rows) if output_rows else 0.0
    row_recall = row_mass / len(source_rows) if source_rows else 1.0
    row_f = f_score(row_precision, row_recall, beta)

    associated_scores: list[float] = []
    key_exact = 0
    for edge in rows.matches:
        gate, _ = key_compatible(edge.expected_index, edge.actual_index)
        key_exact += int(gate == 1.0) if source.row_key else 0
        for source_column, output_column in aligned:
            associated_scores.append(
                _core.text_similarity(
                    source_rows[edge.expected_index][source_column]
                    if source_column < len(source_rows[edge.expected_index])
                    else "",
                    output_rows[edge.actual_index][output_column]
                    if output_column < len(output_rows[edge.actual_index])
                    else "",
                )
                * gate
            )
    expected_cells = len(source_rows) * max(1, len(aligned))
    actual_cells = len(output_rows) * max(1, len(aligned))
    cell_mass = sum(associated_scores)
    cell_precision = cell_mass / actual_cells if actual_cells else 0.0
    cell_recall = cell_mass / expected_cells if expected_cells else 1.0
    cell_f = f_score(cell_precision, cell_recall, beta)

    order_score: float | None = None
    if source.order_sensitive and source.rows:
        ordered = [
            edge.expected_index
            for edge in sorted(rows.matches, key=lambda edge: edge.actual_index)
            if edge.score >= 0.20
        ]
        order_score = _lcs_norm(
            [str(value) for value in ordered],
            [str(value) for value in sorted(ordered)],
        )

    complete = columns.complete and rows.complete
    parts = [(0.20, column_f), (0.30, row_f), (0.45, cell_f)]
    if order_score is not None:
        parts.append((0.05, order_score))
    else:
        parts[2] = (0.50, cell_f)
    score = sum(weight * value for weight, value in parts) if complete else 0.0
    key_precision = (
        key_exact / len(output_rows) if source.row_key and output_rows else 0.0
    ) if source.row_key else None
    key_recall = (
        key_exact / len(source_rows) if source.row_key and source_rows else 1.0
    ) if source.row_key else None
    return TablePairResult(
        clamp01(score),
        column_f,
        row_f,
        cell_f,
        order_score,
        key_precision,
        key_recall,
        (
            key_exact / max(1, len(rows.matches))
            if source.row_key
            else None
        ),
        complete,
        tuple(
            dict.fromkeys(
                [
                    *column_codes,
                    *columns.diagnostic_codes,
                    *row_codes,
                    *rows.diagnostic_codes,
                    *key_diagnostics,
                ]
            )
        ),
    )


def _table_type_allowed(source: _core.SourceTable, output: _core.OutputTable) -> bool:
    policy = source.representation_policy.casefold()
    output_type = output.element_type.casefold()
    if policy == "table_only":
        return output_type == "table"
    if policy == "chart_only":
        return output_type == "chart"
    if policy in {"structured_equivalent", "either_table_or_chart"}:
        return output_type in {"table", "chart"}
    return output_type == "table"


def table_fidelity_v5_1(
    expected: Sequence[_core.SourceTable],
    output_tables: Sequence[_core.OutputTable],
    output_charts: Sequence[_core.OutputTable],
    *,
    beta: float,
    exact_dense_limit: int,
    max_edges: int,
    top_k: int,
) -> tuple[float | None, dict[str, Any]]:
    expanded = [
        item
        for item in expected
        if item.required and item.minimum_count > 0
        for _ in range(max(1, item.minimum_count))
    ]
    actual = [
        item
        for item in [*output_tables, *output_charts]
        if any(_table_type_allowed(source, item) for source in expanded)
    ]
    if not expanded and not actual:
        return None, {
            "required_count": 0,
            "matched_required_count": 0,
            "matching_complete": True,
        }
    details: dict[tuple[int, int], TablePairResult] = {}

    def pair_score(left: int, right: int) -> float:
        if not _table_type_allowed(expanded[left], actual[right]):
            return 0.0
        result = table_pair_metrics_v5_1(
            expanded[left],
            actual[right],
            beta=beta,
            exact_dense_limit=exact_dense_limit,
            max_edges=max_edges,
            top_k=top_k,
        )
        details[(left, right)] = result
        return result.score

    edges, edge_codes = build_candidate_edges(
        expanded,
        actual,
        score=pair_score,
        cheap_score=lambda left, right: float(
            _table_type_allowed(expanded[left], actual[right])
        ),
        max_edges=max_edges,
        top_k=top_k,
    )
    assignment = maximum_weight_assignment(
        len(expanded),
        len(actual),
        edges,
        exact_dense_limit=exact_dense_limit,
    )
    complete = assignment.complete and all(
        details.get((edge.expected_index, edge.actual_index), TablePairResult(
            0, 0, 0, 0, None, None, None, None, False
        )).complete
        for edge in assignment.matches
        if edge.score > 0
    )
    mass = sum(edge.score for edge in assignment.matches)
    precision = mass / len(actual) if actual else 0.0
    recall = mass / len(expanded) if expanded else 1.0
    matched = sum(edge.score >= 0.60 for edge in assignment.matches)
    return (
        f_score(precision, recall, beta) if complete else None
    ), {
        "required_count": len(expanded),
        "matched_required_count": matched,
        "precision": precision,
        "recall": recall,
        "assignment_truncated": False,
        "matching_complete": complete,
        "matching": {
            **_assignment_diagnostics(assignment),
            "diagnostic_codes": list(
                dict.fromkeys([*edge_codes, *assignment.diagnostic_codes])
            ),
        },
        "matches": [
            {
                "expected_index": edge.expected_index,
                "output_index": edge.actual_index,
                "output_type": actual[edge.actual_index].element_type,
                "score": edge.score,
                "column_fbeta": details[(edge.expected_index, edge.actual_index)].column_fbeta,
                "row_fbeta": details[(edge.expected_index, edge.actual_index)].row_fbeta,
                "associated_cell_fbeta": details[(edge.expected_index, edge.actual_index)].associated_cell_fbeta,
                "order_score": details[(edge.expected_index, edge.actual_index)].order_score,
                "key_precision": details[(edge.expected_index, edge.actual_index)].key_precision,
                "key_recall": details[(edge.expected_index, edge.actual_index)].key_recall,
                "key_exact_match_rate": details[(edge.expected_index, edge.actual_index)].key_exact_match_rate,
                "diagnostic_codes": list(details[(edge.expected_index, edge.actual_index)].diagnostic_codes),
            }
            for edge in assignment.matches
            if (edge.expected_index, edge.actual_index) in details
        ],
    }


def action_match_score(
    expected: _core.ActionRef,
    actual: _core.OutputAction,
    aliases: Mapping[str, set[str]],
) -> float:
    if expected.action_type.casefold() != actual.action_type.casefold():
        return 0.0
    if not _core._action_target_equivalent(expected, actual, aliases):  # type: ignore[attr-defined]
        return 0.0
    label = _core.text_similarity(expected.label, actual.label) if expected.label else 1.0
    return clamp01(0.60 + 0.40 * label)


def _expanded_actions(expected: Sequence[_core.ActionRef]) -> list[_core.ActionRef]:
    result: list[_core.ActionRef] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in expected:
        if not item.required or item.minimum_count <= 0:
            continue
        key = (
            _core.normalize_match_text(item.label),
            _core.normalize_url(item.url),
            item.action_type.casefold(),
            item.id,
        )
        if key in seen:
            continue
        seen.add(key)
        result.extend([item] * max(1, item.minimum_count))
    return result


def action_fidelity_v5_1(
    expected: Sequence[_core.ActionRef],
    actual: Sequence[_core.OutputAction],
    *,
    aliases: Mapping[str, set[str]],
    exact_dense_limit: int,
    max_edges: int,
    top_k: int,
) -> tuple[float | None, dict[str, Any]]:
    expanded = _expanded_actions(expected)
    if not expanded and not actual:
        return None, {
            "required_count": 0,
            "matched_required_count": 0,
            "matching_complete": True,
        }
    edges, edge_codes = build_candidate_edges(
        expanded,
        actual,
        score=lambda left, right: action_match_score(
            expanded[left], actual[right], aliases
        ),
        cheap_score=lambda left, right: float(
            expanded[left].action_type.casefold()
            == actual[right].action_type.casefold()
        ),
        max_edges=max_edges,
        top_k=top_k,
    )
    assignment = maximum_weight_assignment(
        len(expanded),
        len(actual),
        edges,
        exact_dense_limit=exact_dense_limit,
    )
    mass = sum(edge.score for edge in assignment.matches)
    precision = mass / len(actual) if actual else 0.0
    recall = mass / len(expanded) if expanded else 1.0
    matched = sum(edge.score >= 0.75 for edge in assignment.matches)
    return (
        f_score(precision, recall, 2.0) if assignment.complete else None
    ), {
        "required_count": len(expanded),
        "matched_required_count": matched,
        "precision": precision,
        "recall": recall,
        "role_coverage": matched / len(expanded) if expanded else None,
        "assignment_truncated": False,
        "matching_complete": assignment.complete,
        "matching": {
            **_assignment_diagnostics(assignment),
            "diagnostic_codes": list(
                dict.fromkeys([*edge_codes, *assignment.diagnostic_codes])
            ),
        },
        "matches": [
            {
                "expected_index": edge.expected_index,
                "output_index": edge.actual_index,
                "score": edge.score,
            }
            for edge in assignment.matches
        ],
    }


def _media_match_score(
    expected: _core.MediaRef,
    actual: _core.MediaRef,
    aliases: Mapping[str, set[str]],
) -> float:
    if expected.kind.casefold() != actual.kind.casefold():
        return 0.0
    url = (
        1.0
        if not expected.url
        else float(
            _core._url_equivalent(expected.url, actual.url, aliases)  # type: ignore[attr-defined]
        )
    )
    if url <= 0.0:
        return 0.0
    alt = _core.text_similarity(expected.alt, actual.alt) if expected.alt else 1.0
    return clamp01(0.75 * url + 0.25 * alt)


def media_fidelity_v5_1(
    expected: Sequence[_core.MediaRef],
    actual: Sequence[_core.MediaRef],
    *,
    aliases: Mapping[str, set[str]],
    exact_dense_limit: int,
    max_edges: int,
    top_k: int,
) -> tuple[float | None, dict[str, Any]]:
    exact: list[_core.MediaRef] = []
    representative: dict[str, _core.MediaRef] = {}
    for item in expected:
        if not item.required or item.minimum_count <= 0:
            continue
        if item.media_policy.casefold() == "representative":
            representative.setdefault(item.kind.casefold(), item)
        else:
            exact.extend([item] * max(1, item.minimum_count))
    expanded = exact + list(representative.values())
    if not expanded and not actual:
        return None, {
            "required_count": 0,
            "matched_required_count": 0,
            "matching_complete": True,
        }
    edges, edge_codes = build_candidate_edges(
        expanded,
        actual,
        score=lambda left, right: _media_match_score(
            expanded[left], actual[right], aliases
        ),
        cheap_score=lambda left, right: float(
            expanded[left].kind.casefold() == actual[right].kind.casefold()
        ),
        max_edges=max_edges,
        top_k=top_k,
    )
    assignment = maximum_weight_assignment(
        len(expanded),
        len(actual),
        edges,
        exact_dense_limit=exact_dense_limit,
    )
    mass = sum(edge.score for edge in assignment.matches)
    precision = mass / len(actual) if actual else 0.0
    recall = mass / len(expanded) if expanded else 1.0
    matched = sum(edge.score >= 0.75 for edge in assignment.matches)
    return (
        f_score(precision, recall, 2.0) if assignment.complete else None
    ), {
        "required_count": len(expanded),
        "matched_required_count": matched,
        "precision": precision,
        "recall": recall,
        "assignment_truncated": False,
        "matching_complete": assignment.complete,
        "matching": {
            **_assignment_diagnostics(assignment),
            "diagnostic_codes": list(
                dict.fromkeys([*edge_codes, *assignment.diagnostic_codes])
            ),
        },
        "matches": [
            {
                "expected_index": edge.expected_index,
                "output_index": edge.actual_index,
                "score": edge.score,
            }
            for edge in assignment.matches
        ],
    }


def _role_payload(requirement: Mapping[str, Any]) -> dict[str, Any]:
    ignored = {"id", "required", "minimum_count", "interchangeable"}
    return {
        str(key): value
        for key, value in requirement.items()
        if key not in ignored and value not in (None, "", [], {})
    }


def role_instance_similarity(
    requirement: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> float:
    fields = _role_payload(requirement)
    if not fields:
        return 1.0
    scores: list[float] = []
    for key, expected in sorted(fields.items()):
        observed = actual.get(key)
        if isinstance(expected, Sequence) and not isinstance(
            expected, (str, bytes, bytearray)
        ):
            expected_values = {
                _core.normalize_match_text(value) for value in expected
            }
            actual_values = (
                {
                    _core.normalize_match_text(value)
                    for value in observed
                }
                if isinstance(observed, Sequence)
                and not isinstance(observed, (str, bytes, bytearray))
                else {_core.normalize_match_text(observed)}
            )
            scores.append(
                len(expected_values & actual_values) / max(1, len(expected_values))
            )
        else:
            scores.append(_core.text_similarity(str(expected), str(observed or "")))
    return sum(scores) / len(scores)


def semantic_role_coverage_v5_1(
    expected: Mapping[str, Sequence[Mapping[str, Any]]],
    actual: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    threshold: float,
    exact_dense_limit: int,
    max_edges: int,
    top_k: int,
) -> tuple[float | None, dict[str, float | None], dict[str, Any]]:
    role_values: dict[str, float | None] = {}
    diagnostics: dict[str, Any] = {}
    for role in sorted(set(expected) | set(actual)):
        requirements = [
            dict(item)
            for item in expected.get(role, ())
            if bool(item.get("required", True))
            and int(item.get("minimum_count", 1) or 0) > 0
        ]
        expanded = [
            item
            for item in requirements
            for _ in range(max(1, int(item.get("minimum_count", 1) or 1)))
        ]
        candidates = [dict(item) for item in actual.get(role, ())]
        if not expanded:
            role_values[role] = None
            continue
        edges, edge_codes = build_candidate_edges(
            expanded,
            candidates,
            score=lambda left, right: role_instance_similarity(
                expanded[left], candidates[right]
            ),
            cheap_score=lambda left, right: _core.text_similarity(
                str(expanded[left].get("title") or expanded[left].get("content") or ""),
                str(candidates[right].get("title") or candidates[right].get("content") or ""),
            ),
            max_edges=max_edges,
            top_k=top_k,
        )
        assignment = maximum_weight_assignment(
            len(expanded),
            len(candidates),
            edges,
            exact_dense_limit=exact_dense_limit,
        )
        accepted = []
        used_noninterchangeable_signatures: set[str] = set()
        for edge in sorted(
            assignment.matches,
            key=lambda item: (-item.score, item.expected_index, item.actual_index),
        ):
            if edge.score < threshold:
                continue
            requirement = expanded[edge.expected_index]
            signature = str(candidates[edge.actual_index].get("signature") or "")
            if (
                not bool(requirement.get("interchangeable", False))
                and signature
                and signature in used_noninterchangeable_signatures
            ):
                continue
            if not bool(requirement.get("interchangeable", False)) and signature:
                used_noninterchangeable_signatures.add(signature)
            accepted.append(edge)
        coverage = len(accepted) / len(expanded)
        role_values[role] = coverage if assignment.complete else None
        diagnostics[role] = {
            "required_count": len(expanded),
            "actual_count": len(candidates),
            "matched_required_count": len(accepted),
            "coverage": coverage,
            "matching_complete": assignment.complete,
            "matching": {
                **_assignment_diagnostics(assignment),
                "diagnostic_codes": list(
                    dict.fromkeys([*edge_codes, *assignment.diagnostic_codes])
                ),
            },
            "matches": [
                {
                    "expected_index": edge.expected_index,
                    "output_index": edge.actual_index,
                    "score": edge.score,
                }
                for edge in accepted
            ],
        }
    applicable = [float(value) for value in role_values.values() if value is not None]
    return (
        sum(applicable) / len(applicable) if applicable else None,
        role_values,
        diagnostics,
    )


def semantic_signature(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "SCORING_POLICY_VERSION",
    "TablePairResult",
    "action_fidelity_v5_1",
    "clamp01",
    "content_fidelity_v5_1",
    "f_score",
    "media_fidelity_v5_1",
    "prepared_block_similarity",
    "role_instance_similarity",
    "semantic_role_coverage_v5_1",
    "semantic_signature",
    "table_fidelity_v5_1",
    "table_pair_metrics_v5_1",
]
