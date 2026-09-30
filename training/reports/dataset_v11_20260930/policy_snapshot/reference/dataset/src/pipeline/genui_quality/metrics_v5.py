"""Association-preserving atomic metrics for GenUI representation quality v5."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
import math
from typing import Any, Callable, Mapping, Sequence

from . import _core


def clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def f_score(precision: float, recall: float, beta: float = 1.0) -> float:
    return _core.f_beta(clamp01(precision), clamp01(recall), beta)


def _counter_fbeta(left: Sequence[str], right: Sequence[str], beta: float) -> float:
    left_counter = Counter(left)
    right_counter = Counter(right)
    precision, recall = _core.counter_pr(right_counter, left_counter)
    return f_score(precision, recall, beta)


def _bigrams(tokens: Sequence[str]) -> list[str]:
    return [f"{tokens[index]}\0{tokens[index + 1]}" for index in range(len(tokens) - 1)]


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


def block_similarity(source: str, output: str) -> float:
    source_tokens = _core.tokenize(source)
    output_tokens = _core.tokenize(output)
    unigram = _counter_fbeta(source_tokens, output_tokens, beta=2.0)
    bigram = _counter_fbeta(_bigrams(source_tokens), _bigrams(output_tokens), beta=1.0)
    lcs = _lcs_norm(source_tokens, output_tokens)
    return clamp01(0.45 * unigram + 0.35 * bigram + 0.20 * lcs)


def _greedy_assignment(matrix: Sequence[Sequence[float]]) -> list[tuple[int, int, float]]:
    candidates = [
        (clamp01(value), row, column)
        for row, values in enumerate(matrix)
        for column, value in enumerate(values)
    ]
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    rows: set[int] = set()
    columns: set[int] = set()
    result: list[tuple[int, int, float]] = []
    for score, row, column in candidates:
        if row in rows or column in columns:
            continue
        rows.add(row)
        columns.add(column)
        result.append((row, column, score))
    return sorted(result)


def maximum_assignment(
    matrix: Sequence[Sequence[float]],
    *,
    max_size: int = 64,
) -> list[tuple[int, int, float]]:
    """Deterministic maximum-weight one-to-one assignment.

    Uses the O(n^3) Hungarian algorithm for bounded inputs and a deterministic
    greedy fallback when a caller exceeds the configured safe bound.
    """

    rows = len(matrix)
    columns = max((len(row) for row in matrix), default=0)
    if rows == 0 or columns == 0:
        return []
    normalized = [
        [clamp01(row[column]) if column < len(row) else 0.0 for column in range(columns)]
        for row in matrix
    ]
    if max(rows, columns) > max_size:
        return _greedy_assignment(normalized)

    transposed = rows > columns
    working = (
        [[normalized[row][column] for row in range(rows)] for column in range(columns)]
        if transposed
        else normalized
    )
    n = len(working)
    m = len(working[0])
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minimum = [float("inf")] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cost = 1.0 - working[i0 - 1][j - 1]
                current = cost - u[i0] - v[j]
                if current < minimum[j] - 1e-15:
                    minimum[j] = current
                    way[j] = j0
                if minimum[j] < delta - 1e-15:
                    delta = minimum[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    result: list[tuple[int, int, float]] = []
    for column in range(1, m + 1):
        if p[column] == 0:
            continue
        left = p[column] - 1
        right = column - 1
        row, col = (right, left) if transposed else (left, right)
        if row < rows and col < columns:
            result.append((row, col, normalized[row][col]))
    return sorted(result)


def content_fidelity(
    source_units: Sequence[str],
    output_blocks: Sequence[str],
    *,
    max_size: int,
) -> tuple[dict[str, float | None], dict[str, Any]]:
    if not source_units and not output_blocks:
        return {
            "content_unit_fidelity": None,
            "content_order_preservation": None,
            "output_block_precision": None,
        }, {"matches": []}
    bounded_sources = list(source_units[:max_size])
    bounded_outputs = list(output_blocks[:max_size])
    matrix = [
        [block_similarity(source, output) for output in bounded_outputs]
        for source in bounded_sources
    ]
    matches = maximum_assignment(matrix, max_size=max_size)
    source_recall = (
        sum(score for _, _, score in matches) / len(source_units)
        if source_units
        else 1.0
    )
    output_precision = (
        sum(score for _, _, score in matches) / len(output_blocks)
        if output_blocks
        else 1.0
    )
    unit_f = f_score(output_precision, source_recall, beta=2.0)
    meaningful = [(source, output) for source, output, score in matches if score >= 0.15]
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
    return {
        "content_unit_fidelity": unit_f,
        "content_order_preservation": order_score if source_units else None,
        "output_block_precision": output_precision if output_blocks else 0.0,
    }, {
        "source_unit_recall": source_recall,
        "output_block_precision": output_precision,
        "assignment_truncated": (
            len(source_units) > max_size or len(output_blocks) > max_size
        ),
        "matches": [
            {"source_index": source, "output_index": output, "similarity": score}
            for source, output, score in matches
        ],
    }


@dataclass(frozen=True)
class TablePairResult:
    score: float
    column_fbeta: float
    row_fbeta: float
    associated_cell_fbeta: float
    order_score: float | None


def _output_rows(table: _core.OutputTable) -> list[list[str]]:
    return [
        [str(row.get(key, "")) for key in table.keys]
        for row in table.rows
    ]


def table_pair_metrics(
    source: _core.SourceTable,
    output: _core.OutputTable,
    *,
    beta: float,
    max_size: int,
) -> TablePairResult:
    bounded_source_headers = list(source.headers[:max_size])
    bounded_output_headers = list(output.headers[:max_size])
    column_matrix = [
        [_core.text_similarity(header, actual) for actual in bounded_output_headers]
        for header in bounded_source_headers
    ]
    column_matches = maximum_assignment(column_matrix, max_size=max_size)
    column_mass = sum(score for _, _, score in column_matches)
    column_precision = column_mass / len(output.headers) if output.headers else 0.0
    column_recall = column_mass / len(source.headers) if source.headers else 1.0
    column_f = f_score(column_precision, column_recall, beta)

    aligned = [(source_index, output_index) for source_index, output_index, score in column_matches if score >= 0.20]
    output_rows = _output_rows(output)
    bounded_source_rows = list(source.rows[:max_size])
    bounded_output_rows = output_rows[:max_size]

    def row_similarity(source_row: Sequence[str], output_row: Sequence[str]) -> float:
        if not aligned:
            return 0.0
        values = [
            _core.text_similarity(
                source_row[source_index] if source_index < len(source_row) else "",
                output_row[output_index] if output_index < len(output_row) else "",
            )
            for source_index, output_index in aligned
        ]
        return sum(values) / len(values)

    row_matrix = [
        [row_similarity(source_row, output_row) for output_row in bounded_output_rows]
        for source_row in bounded_source_rows
    ]
    row_matches = maximum_assignment(row_matrix, max_size=max_size)
    row_mass = sum(score for _, _, score in row_matches)
    row_precision = row_mass / len(output_rows) if output_rows else 0.0
    row_recall = row_mass / len(source.rows) if source.rows else 1.0
    row_f = f_score(row_precision, row_recall, beta)

    associated_scores: list[float] = []
    for source_row_index, output_row_index, _ in row_matches:
        if source_row_index >= len(bounded_source_rows) or output_row_index >= len(bounded_output_rows):
            continue
        for source_column, output_column in aligned:
            associated_scores.append(
                _core.text_similarity(
                    bounded_source_rows[source_row_index][source_column]
                    if source_column < len(bounded_source_rows[source_row_index])
                    else "",
                    bounded_output_rows[output_row_index][output_column]
                    if output_column < len(bounded_output_rows[output_row_index])
                    else "",
                )
            )
    expected_cells = len(source.rows) * max(1, len(aligned))
    actual_cells = len(output_rows) * max(1, len(aligned))
    cell_mass = sum(associated_scores)
    cell_precision = cell_mass / actual_cells if actual_cells else 0.0
    cell_recall = cell_mass / expected_cells if expected_cells else 1.0
    cell_f = f_score(cell_precision, cell_recall, beta)

    order_score: float | None = None
    if source.order_sensitive and source.rows:
        ordered = [
            source_index
            for source_index, _, score in sorted(row_matches, key=lambda item: item[1])
            if score >= 0.20
        ]
        order_score = _lcs_norm(
            [str(value) for value in ordered],
            [str(value) for value in sorted(ordered)],
        )

    parts = [(0.20, column_f), (0.30, row_f), (0.45, cell_f)]
    if order_score is not None:
        parts.append((0.05, order_score))
    else:
        parts[2] = (0.50, cell_f)
    score = sum(weight * value for weight, value in parts)
    return TablePairResult(
        clamp01(score),
        column_f,
        row_f,
        cell_f,
        order_score,
    )


def table_fidelity_v5(
    expected: Sequence[_core.SourceTable],
    actual: Sequence[_core.OutputTable],
    *,
    beta: float,
    max_size: int,
) -> tuple[float | None, dict[str, Any]]:
    required = [item for item in expected if item.required and item.minimum_count > 0]
    expanded = [
        item
        for item in required
        for _ in range(max(1, item.minimum_count))
    ]
    if not expanded and not actual:
        return None, {"required_count": 0, "matched_required_count": 0}
    bounded_expected = expanded[:max_size]
    bounded_actual = list(actual[:max_size])
    matrix: list[list[float]] = []
    details: dict[tuple[int, int], TablePairResult] = {}
    for source_index, source in enumerate(bounded_expected):
        row: list[float] = []
        for output_index, output in enumerate(bounded_actual):
            result = table_pair_metrics(
                source,
                output,
                beta=beta,
                max_size=max_size,
            )
            details[(source_index, output_index)] = result
            row.append(result.score)
        matrix.append(row)
    matches = maximum_assignment(matrix, max_size=max_size)
    mass = sum(score for _, _, score in matches)
    precision = mass / len(actual) if actual else 0.0
    recall = mass / len(expanded) if expanded else 1.0
    matched_required = sum(score >= 0.60 for _, _, score in matches)
    return f_score(precision, recall, beta), {
        "required_count": len(expanded),
        "matched_required_count": matched_required,
        "precision": precision,
        "recall": recall,
        "assignment_truncated": len(expanded) > max_size or len(actual) > max_size,
        "matches": [
            {
                "expected_index": source,
                "output_index": output,
                "score": score,
                "column_fbeta": details[(source, output)].column_fbeta,
                "row_fbeta": details[(source, output)].row_fbeta,
                "associated_cell_fbeta": details[(source, output)].associated_cell_fbeta,
                "order_score": details[(source, output)].order_score,
            }
            for source, output, score in matches
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
    label = (
        _core.text_similarity(expected.label, actual.label)
        if expected.label
        else 1.0
    )
    return clamp01(0.60 + 0.40 * label)


def action_fidelity_v5(
    expected: Sequence[_core.ActionRef],
    actual: Sequence[_core.OutputAction],
    *,
    aliases: Mapping[str, set[str]],
    max_size: int,
) -> tuple[float | None, dict[str, Any]]:
    required = [item for item in expected if item.required and item.minimum_count > 0]
    expanded = [
        item
        for item in required
        for _ in range(max(1, item.minimum_count))
    ]
    if not expanded and not actual:
        return None, {"required_count": 0, "matched_required_count": 0}
    bounded_expected = expanded[:max_size]
    bounded_actual = list(actual[:max_size])
    matrix = [
        [action_match_score(source, output, aliases) for output in bounded_actual]
        for source in bounded_expected
    ]
    matches = maximum_assignment(matrix, max_size=max_size)
    mass = sum(score for _, _, score in matches)
    precision = mass / len(actual) if actual else 0.0
    recall = mass / len(expanded) if expanded else 1.0
    matched = sum(score >= 0.75 for _, _, score in matches)
    return f_score(precision, recall, 2.0), {
        "required_count": len(expanded),
        "matched_required_count": matched,
        "precision": precision,
        "recall": recall,
        "assignment_truncated": len(expanded) > max_size or len(actual) > max_size,
        "matches": [
            {"expected_index": source, "output_index": output, "score": score}
            for source, output, score in matches
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


def media_fidelity_v5(
    expected: Sequence[_core.MediaRef],
    actual: Sequence[_core.MediaRef],
    *,
    aliases: Mapping[str, set[str]],
    max_size: int,
) -> tuple[float | None, dict[str, Any]]:
    required_exact: list[_core.MediaRef] = []
    representative_by_kind: dict[str, _core.MediaRef] = {}
    for item in expected:
        if not item.required or item.minimum_count <= 0:
            continue
        if item.media_policy.casefold() == "representative":
            representative_by_kind.setdefault(item.kind.casefold(), item)
        else:
            required_exact.extend([item] * max(1, item.minimum_count))
    expanded = required_exact + list(representative_by_kind.values())
    if not expanded and not actual:
        return None, {"required_count": 0, "matched_required_count": 0}
    bounded_expected = expanded[:max_size]
    bounded_actual = list(actual[:max_size])
    matrix = [
        [_media_match_score(source, output, aliases) for output in bounded_actual]
        for source in bounded_expected
    ]
    matches = maximum_assignment(matrix, max_size=max_size)
    mass = sum(score for _, _, score in matches)
    precision = mass / len(actual) if actual else 0.0
    recall = mass / len(expanded) if expanded else 1.0
    matched = sum(score >= 0.75 for _, _, score in matches)
    return f_score(precision, recall, 2.0), {
        "required_count": len(expanded),
        "matched_required_count": matched,
        "precision": precision,
        "recall": recall,
        "assignment_truncated": len(expanded) > max_size or len(actual) > max_size,
        "matches": [
            {"expected_index": source, "output_index": output, "score": score}
            for source, output, score in matches
        ],
    }


def distinct_role_coverage(
    required_counts: Mapping[str, int],
    role_signatures: Mapping[str, Sequence[str]],
    *,
    action_matched: int,
    table_matched: int,
    media_matched_by_role: Mapping[str, int],
) -> tuple[float | None, dict[str, float | None]]:
    values: dict[str, float | None] = {}
    for role, raw_count in sorted(required_counts.items()):
        count = max(0, int(raw_count))
        if count <= 0:
            values[role] = None
            continue
        if role == "action":
            present = action_matched
        elif role == "table":
            present = table_matched
        elif role in {"image", "video", "audio"}:
            present = int(media_matched_by_role.get(role, 0))
        else:
            present = len(set(role_signatures.get(role, ())))
        values[role] = min(1.0, present / count)
    applicable = [float(value) for value in values.values() if value is not None]
    return (
        (sum(applicable) / len(applicable) if applicable else None),
        values,
    )


__all__ = [
    "TablePairResult",
    "action_fidelity_v5",
    "block_similarity",
    "clamp01",
    "content_fidelity",
    "distinct_role_coverage",
    "f_score",
    "maximum_assignment",
    "media_fidelity_v5",
    "table_fidelity_v5",
    "table_pair_metrics",
]
