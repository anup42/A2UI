"""Deterministic full-set matching primitives for GenUI metric v5.1."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any, Callable, Sequence

from . import _core


MATCHING_POLICY_VERSION = "5.1.0"


@dataclass(frozen=True)
class MatchEdge:
    expected_index: int
    actual_index: int
    score: float


@dataclass(frozen=True)
class AssignmentResult:
    matches: tuple[MatchEdge, ...]
    expected_count: int
    actual_count: int
    evaluated_edge_count: int
    candidate_edge_count: int
    exact: bool
    complete: bool
    diagnostic_codes: tuple[str, ...]


@dataclass(frozen=True)
class PreparedTextBlock:
    raw: str
    normalized: str
    tokens: tuple[str, ...]
    token_counter: tuple[tuple[str, int], ...]
    bigrams: tuple[str, ...]

    def counter(self) -> Counter[str]:
        return Counter(dict(self.token_counter))


@lru_cache(maxsize=32768)
def prepare_text_block(value: str) -> PreparedTextBlock:
    raw = str(value or "")
    normalized = _core.normalize_match_text(raw)
    tokens = tuple(_core.tokenize(raw))
    counter = tuple(sorted(Counter(tokens).items()))
    bigrams = tuple(
        f"{tokens[index]}\0{tokens[index + 1]}"
        for index in range(max(0, len(tokens) - 1))
    )
    return PreparedTextBlock(raw, normalized, tokens, counter, bigrams)


def _clamp(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _hungarian(
    expected_count: int,
    actual_count: int,
    scores: dict[tuple[int, int], float],
) -> tuple[MatchEdge, ...]:
    matrix = [
        [_clamp(scores.get((row, column), 0.0)) for column in range(actual_count)]
        for row in range(expected_count)
    ]
    transposed = expected_count > actual_count
    working = (
        [
            [matrix[row][column] for row in range(expected_count)]
            for column in range(actual_count)
        ]
        if transposed
        else matrix
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
                current = (1.0 - working[i0 - 1][j - 1]) - u[i0] - v[j]
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

    matches: list[MatchEdge] = []
    for column in range(1, m + 1):
        if p[column] == 0:
            continue
        left = p[column] - 1
        right = column - 1
        expected, actual = (right, left) if transposed else (left, right)
        if expected < expected_count and actual < actual_count:
            matches.append(
                MatchEdge(expected, actual, matrix[expected][actual])
            )
    return tuple(sorted(matches, key=lambda edge: (edge.expected_index, edge.actual_index)))


def maximum_weight_assignment(
    expected_count: int,
    actual_count: int,
    edges: Sequence[MatchEdge],
    *,
    exact_dense_limit: int,
) -> AssignmentResult:
    """Assign over the complete item sets without truncating either side."""

    if expected_count < 0 or actual_count < 0:
        raise ValueError("assignment counts must be non-negative")
    candidate_edge_count = expected_count * actual_count
    unique: dict[tuple[int, int], float] = {}
    diagnostics: list[str] = []
    for edge in edges:
        if not (
            0 <= edge.expected_index < expected_count
            and 0 <= edge.actual_index < actual_count
        ):
            diagnostics.append("invalid_edge_index")
            continue
        key = (edge.expected_index, edge.actual_index)
        unique[key] = max(unique.get(key, 0.0), _clamp(edge.score))

    if expected_count == 0 or actual_count == 0:
        return AssignmentResult(
            (),
            expected_count,
            actual_count,
            len(unique),
            candidate_edge_count,
            True,
            True,
            tuple(dict.fromkeys(diagnostics)),
        )

    expected_covered = {left for left, _ in unique}
    actual_covered = {right for _, right in unique}
    complete = (
        len(expected_covered) == expected_count
        and len(actual_covered) == actual_count
    )
    if not complete:
        diagnostics.append("matching_budget_exceeded")

    exact = (
        max(expected_count, actual_count) <= exact_dense_limit
        and len(unique) == candidate_edge_count
    )
    if exact:
        matches = _hungarian(expected_count, actual_count, unique)
    else:
        ordered = sorted(
            (
                MatchEdge(left, right, score)
                for (left, right), score in unique.items()
            ),
            key=lambda edge: (
                -edge.score,
                edge.expected_index,
                edge.actual_index,
            ),
        )
        used_expected: set[int] = set()
        used_actual: set[int] = set()
        selected: list[MatchEdge] = []
        for edge in ordered:
            if (
                edge.expected_index in used_expected
                or edge.actual_index in used_actual
            ):
                continue
            used_expected.add(edge.expected_index)
            used_actual.add(edge.actual_index)
            selected.append(edge)
        matches = tuple(
            sorted(selected, key=lambda edge: (edge.expected_index, edge.actual_index))
        )
        diagnostics.append("deterministic_sparse_assignment")

    return AssignmentResult(
        matches,
        expected_count,
        actual_count,
        len(unique),
        candidate_edge_count,
        exact,
        complete,
        tuple(dict.fromkeys(diagnostics)),
    )


def build_candidate_edges(
    expected: Sequence[Any],
    actual: Sequence[Any],
    *,
    score: Callable[[int, int], float],
    cheap_score: Callable[[int, int], float],
    max_edges: int,
    top_k: int,
) -> tuple[tuple[MatchEdge, ...], tuple[str, ...]]:
    """Evaluate all pairs when bounded, otherwise cover every item sparsely."""

    expected_count = len(expected)
    actual_count = len(actual)
    pair_count = expected_count * actual_count
    if pair_count == 0:
        return (), ()
    if pair_count <= max_edges:
        return (
            tuple(
                MatchEdge(left, right, score(left, right))
                for left in range(expected_count)
                for right in range(actual_count)
            ),
            (),
        )

    minimum_coverage = max(expected_count, actual_count)
    if max_edges < minimum_coverage:
        pairs = [
            (index % expected_count, index % actual_count)
            for index in range(max_edges)
        ]
        return (
            tuple(MatchEdge(left, right, score(left, right)) for left, right in pairs),
            ("matching_budget_exceeded",),
        )

    pairs: set[tuple[int, int]] = set()
    per_side = max(1, min(top_k, max_edges // max(1, expected_count + actual_count)))
    for left in range(expected_count):
        ranked = sorted(
            range(actual_count),
            key=lambda right: (-_clamp(cheap_score(left, right)), right),
        )
        pairs.update((left, right) for right in ranked[:per_side])
    for right in range(actual_count):
        ranked = sorted(
            range(expected_count),
            key=lambda left: (-_clamp(cheap_score(left, right)), left),
        )
        pairs.update((left, right) for left in ranked[:per_side])
    if len(pairs) > max_edges:
        pairs = set(
            sorted(
                pairs,
                key=lambda pair: (
                    -_clamp(cheap_score(pair[0], pair[1])),
                    pair[0],
                    pair[1],
                ),
            )[:max_edges]
        )
    return (
        tuple(
            MatchEdge(left, right, score(left, right))
            for left, right in sorted(pairs)
        ),
        ("sparse_candidate_generation",),
    )


__all__ = [
    "AssignmentResult",
    "MATCHING_POLICY_VERSION",
    "MatchEdge",
    "PreparedTextBlock",
    "build_candidate_edges",
    "maximum_weight_assignment",
    "prepare_text_block",
]
