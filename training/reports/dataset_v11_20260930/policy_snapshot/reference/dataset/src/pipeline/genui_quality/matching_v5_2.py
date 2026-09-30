"""Certified full-set assignment primitives for GenUI metric v5.2.

The v5.1 matcher used sparse top-k candidates and then labelled a greedy
assignment as complete whenever every vertex appeared in an edge.  V5.2 keeps
those concepts separate:

* exact identity matches are allocated one-to-one before fuzzy work;
* residual candidate generation reports whether it enumerated every pair;
* each sparse connected component is solved exactly when it fits a
  deterministic budget; and
* any approximation is explicit and can never be presented as certified.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Callable, Hashable, Iterable, Sequence


MATCHING_POLICY_VERSION = "5.2.0"
_TOLERANCE = 1e-12


@dataclass(frozen=True)
class MatchEdge:
    expected_index: int
    actual_index: int
    score: float


@dataclass(frozen=True)
class AssignmentCertification:
    vertex_edge_coverage_complete: bool
    full_cardinality_matching_exists: bool
    optimality_certified: bool
    approximate_matching_used: bool
    candidate_generation_complete: bool
    lower_bound_weight: float
    upper_bound_weight: float | None
    diagnostic_codes: tuple[str, ...]


@dataclass(frozen=True)
class AssignmentResultV52:
    matches: tuple[MatchEdge, ...]
    expected_count: int
    actual_count: int
    evaluated_edge_count: int
    candidate_edge_count: int
    certification: AssignmentCertification
    exact_preallocated_count: int = 0
    fuzzy_residual_count: int = 0


ExactKey = Hashable | None
ExactKeyFunction = Callable[[int], ExactKey]
BlockKeyFunction = Callable[[int], Iterable[Hashable]]
PairScoreFunction = Callable[[int, int], float]


def _clamp(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return max(0.0, min(1.0, numeric))


def normalized_text_hash(value: str) -> str:
    """Return a stable exact-key digest without retaining large source text."""

    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _deduplicate_edges(
    expected_count: int,
    actual_count: int,
    edges: Sequence[MatchEdge],
) -> tuple[dict[tuple[int, int], float], list[str]]:
    unique: dict[tuple[int, int], float] = {}
    diagnostics: list[str] = []
    for edge in edges:
        if not (
            0 <= int(edge.expected_index) < expected_count
            and 0 <= int(edge.actual_index) < actual_count
        ):
            diagnostics.append("invalid_edge_index")
            continue
        key = (int(edge.expected_index), int(edge.actual_index))
        unique[key] = max(unique.get(key, 0.0), _clamp(edge.score))
    return unique, diagnostics


def _connected_components(
    unique: dict[tuple[int, int], float],
) -> list[tuple[list[int], list[int], dict[tuple[int, int], float]]]:
    left_to_right: dict[int, set[int]] = defaultdict(set)
    right_to_left: dict[int, set[int]] = defaultdict(set)
    for left, right in unique:
        left_to_right[left].add(right)
        right_to_left[right].add(left)

    seen_left: set[int] = set()
    seen_right: set[int] = set()
    components: list[
        tuple[list[int], list[int], dict[tuple[int, int], float]]
    ] = []
    for start in sorted(left_to_right):
        if start in seen_left:
            continue
        pending: deque[tuple[str, int]] = deque([("left", start)])
        component_left: set[int] = set()
        component_right: set[int] = set()
        while pending:
            side, index = pending.popleft()
            if side == "left":
                if index in seen_left:
                    continue
                seen_left.add(index)
                component_left.add(index)
                for right in sorted(left_to_right.get(index, ())):
                    if right not in seen_right:
                        pending.append(("right", right))
            else:
                if index in seen_right:
                    continue
                seen_right.add(index)
                component_right.add(index)
                for left in sorted(right_to_left.get(index, ())):
                    if left not in seen_left:
                        pending.append(("left", left))
        component_edges = {
            (left, right): score
            for (left, right), score in unique.items()
            if left in component_left and right in component_right
        }
        components.append(
            (
                sorted(component_left),
                sorted(component_right),
                component_edges,
            )
        )
    return components


def _hungarian_component(
    expected_ids: Sequence[int],
    actual_ids: Sequence[int],
    scores: dict[tuple[int, int], float],
) -> tuple[MatchEdge, ...]:
    """Maximum-cardinality, then maximum-weight, exact dense assignment."""

    if not expected_ids or not actual_ids:
        return ()
    transposed = len(expected_ids) > len(actual_ids)
    rows = list(actual_ids if transposed else expected_ids)
    columns = list(expected_ids if transposed else actual_ids)
    cardinality_bonus = float(min(len(expected_ids), len(actual_ids)) + 1)

    def original_pair(row: int, column: int) -> tuple[int, int]:
        return (column, row) if transposed else (row, column)

    values: list[list[float]] = []
    for row in rows:
        current: list[float] = []
        for column in columns:
            pair = original_pair(row, column)
            current.append(
                cardinality_bonus + scores[pair]
                if pair in scores
                else 0.0
            )
        values.append(current)
    maximum = cardinality_bonus + 1.0
    n = len(rows)
    m = len(columns)
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
                current = (
                    maximum - values[i0 - 1][j - 1] - u[i0] - v[j]
                )
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

    selected: list[MatchEdge] = []
    for column_position in range(1, m + 1):
        if p[column_position] == 0:
            continue
        row = rows[p[column_position] - 1]
        column = columns[column_position - 1]
        left, right = original_pair(row, column)
        if (left, right) in scores:
            selected.append(MatchEdge(left, right, scores[(left, right)]))
    return tuple(
        sorted(
            selected,
            key=lambda edge: (edge.expected_index, edge.actual_index),
        )
    )


@dataclass
class _FlowArc:
    to: int
    reverse: int
    capacity: int
    cost: float


def _sparse_flow_component(
    expected_ids: Sequence[int],
    actual_ids: Sequence[int],
    scores: dict[tuple[int, int], float],
    *,
    relaxation_budget: int,
) -> tuple[tuple[MatchEdge, ...], bool]:
    """Exact sparse min-cost maximum-cardinality matching within a budget."""

    left_position = {value: index for index, value in enumerate(expected_ids)}
    right_position = {value: index for index, value in enumerate(actual_ids)}
    source = 0
    left_offset = 1
    right_offset = left_offset + len(expected_ids)
    sink = right_offset + len(actual_ids)
    graph: list[list[_FlowArc]] = [[] for _ in range(sink + 1)]
    tracked: dict[tuple[int, int], _FlowArc] = {}

    def add_arc(start: int, end: int, capacity: int, cost: float) -> _FlowArc:
        forward = _FlowArc(end, len(graph[end]), capacity, cost)
        reverse = _FlowArc(start, len(graph[start]), 0, -cost)
        graph[start].append(forward)
        graph[end].append(reverse)
        return forward

    for left in expected_ids:
        add_arc(source, left_offset + left_position[left], 1, 0.0)
    for right in actual_ids:
        add_arc(right_offset + right_position[right], sink, 1, 0.0)
    cardinality_bonus = float(min(len(expected_ids), len(actual_ids)) + 1)
    for (left, right), score in sorted(scores.items()):
        tracked[(left, right)] = add_arc(
            left_offset + left_position[left],
            right_offset + right_position[right],
            1,
            -(cardinality_bonus + score),
        )

    relaxations = 0
    while True:
        distance = [float("inf")] * len(graph)
        previous_node = [-1] * len(graph)
        previous_arc = [-1] * len(graph)
        queued = [False] * len(graph)
        queue: deque[int] = deque([source])
        distance[source] = 0.0
        queued[source] = True
        budget_exceeded = False
        while queue and not budget_exceeded:
            node = queue.popleft()
            queued[node] = False
            for arc_index, arc in enumerate(graph[node]):
                if arc.capacity <= 0:
                    continue
                relaxations += 1
                if relaxations > relaxation_budget:
                    budget_exceeded = True
                    break
                candidate = distance[node] + arc.cost
                if candidate < distance[arc.to] - 1e-15:
                    distance[arc.to] = candidate
                    previous_node[arc.to] = node
                    previous_arc[arc.to] = arc_index
                    if not queued[arc.to]:
                        queue.append(arc.to)
                        queued[arc.to] = True
        if budget_exceeded:
            return _selected_flow_edges(tracked, scores), False
        if previous_node[sink] < 0:
            break
        node = sink
        while node != source:
            parent = previous_node[node]
            arc_index = previous_arc[node]
            arc = graph[parent][arc_index]
            arc.capacity -= 1
            graph[node][arc.reverse].capacity += 1
            node = parent
    return _selected_flow_edges(tracked, scores), True


def _selected_flow_edges(
    tracked: dict[tuple[int, int], _FlowArc],
    scores: dict[tuple[int, int], float],
) -> tuple[MatchEdge, ...]:
    return tuple(
        MatchEdge(left, right, scores[(left, right)])
        for (left, right), arc in sorted(tracked.items())
        if arc.capacity == 0
    )


def _greedy_component(
    scores: dict[tuple[int, int], float],
) -> tuple[MatchEdge, ...]:
    used_left: set[int] = set()
    used_right: set[int] = set()
    selected: list[MatchEdge] = []
    for (left, right), score in sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    ):
        if left in used_left or right in used_right:
            continue
        used_left.add(left)
        used_right.add(right)
        selected.append(MatchEdge(left, right, score))
    return tuple(
        sorted(
            selected,
            key=lambda edge: (edge.expected_index, edge.actual_index),
        )
    )


def maximum_weight_assignment(
    expected_count: int,
    actual_count: int,
    edges: Sequence[MatchEdge],
    *,
    exact_dense_limit: int = 64,
    candidate_generation_complete: bool | None = None,
    max_hungarian_work: int = 50_000_000,
    max_sparse_relaxations: int = 50_000_000,
) -> AssignmentResultV52:
    """Return a truthfully certified maximum-weight assignment.

    ``exact_dense_limit`` remains accepted for call-site compatibility, but it
    is not a semantic cliff.  A 65x65 component is still solved exactly.
    """

    del exact_dense_limit
    if expected_count < 0 or actual_count < 0:
        raise ValueError("assignment counts must be non-negative")
    unique, diagnostics = _deduplicate_edges(
        expected_count, actual_count, edges
    )
    full_pair_count = expected_count * actual_count
    generation_complete = (
        len(unique) == full_pair_count
        if candidate_generation_complete is None
        else bool(candidate_generation_complete)
    )
    expected_covered = {left for left, _ in unique}
    actual_covered = {right for _, right in unique}
    vertex_coverage = (
        (expected_count == 0 or len(expected_covered) == expected_count)
        and (actual_count == 0 or len(actual_covered) == actual_count)
    )

    selected: list[MatchEdge] = []
    approximate = False
    exact_solver = True
    for left_ids, right_ids, component_edges in _connected_components(unique):
        dense_work = (
            min(len(left_ids), len(right_ids))
            * min(len(left_ids), len(right_ids))
            * max(len(left_ids), len(right_ids))
        )
        sparse_work = (
            max(1, min(len(left_ids), len(right_ids)))
            * max(1, len(component_edges))
        )
        if dense_work <= max_hungarian_work:
            selected.extend(
                _hungarian_component(left_ids, right_ids, component_edges)
            )
        elif sparse_work <= max_sparse_relaxations:
            flow_matches, finished = _sparse_flow_component(
                left_ids,
                right_ids,
                component_edges,
                relaxation_budget=max_sparse_relaxations,
            )
            selected.extend(flow_matches)
            if not finished:
                exact_solver = False
                approximate = True
                diagnostics.append("sparse_solver_budget_exceeded")
        else:
            selected.extend(_greedy_component(component_edges))
            exact_solver = False
            approximate = True
            diagnostics.append("matching_solver_budget_exceeded")

    matches = tuple(
        sorted(
            selected,
            key=lambda edge: (edge.expected_index, edge.actual_index),
        )
    )
    lower_bound = sum(edge.score for edge in matches)
    theoretical_upper = float(min(expected_count, actual_count))
    saturated = abs(lower_bound - theoretical_upper) <= _TOLERANCE
    optimality_certified = exact_solver and (
        generation_complete or saturated
    )
    upper_bound = lower_bound if optimality_certified else theoretical_upper
    full_cardinality = len(matches) == min(expected_count, actual_count)
    if not generation_complete:
        diagnostics.append("candidate_generation_incomplete")
    if not vertex_coverage:
        diagnostics.append("vertex_edge_coverage_incomplete")
    if not full_cardinality:
        diagnostics.append("full_cardinality_not_established")
    if approximate:
        diagnostics.append("approximate_matching_used")
    if optimality_certified:
        diagnostics.append("optimality_certified")
    else:
        diagnostics.append("optimality_uncertified")

    return AssignmentResultV52(
        matches=matches,
        expected_count=expected_count,
        actual_count=actual_count,
        evaluated_edge_count=len(unique),
        candidate_edge_count=len(unique),
        certification=AssignmentCertification(
            vertex_edge_coverage_complete=vertex_coverage,
            full_cardinality_matching_exists=full_cardinality,
            optimality_certified=optimality_certified,
            approximate_matching_used=approximate,
            candidate_generation_complete=generation_complete,
            lower_bound_weight=lower_bound,
            upper_bound_weight=upper_bound,
            diagnostic_codes=tuple(dict.fromkeys(diagnostics)),
        ),
    )


def _candidate_pairs(
    expected_indices: Sequence[int],
    actual_indices: Sequence[int],
    *,
    expected_block_keys: BlockKeyFunction,
    actual_block_keys: BlockKeyFunction,
    cheap_score: PairScoreFunction,
    max_edges: int,
    top_k: int,
) -> tuple[set[tuple[int, int]], bool, tuple[str, ...]]:
    pair_count = len(expected_indices) * len(actual_indices)
    if pair_count <= max_edges:
        return (
            {
                (left, right)
                for left in expected_indices
                for right in actual_indices
            },
            True,
            (),
        )

    actual_by_key: dict[Hashable, list[int]] = defaultdict(list)
    for right in actual_indices:
        for key in dict.fromkeys(actual_block_keys(right)):
            actual_by_key[key].append(right)
    pairs: set[tuple[int, int]] = set()
    width = max(1, top_k)
    for position, left in enumerate(expected_indices):
        keys = sorted(
            dict.fromkeys(expected_block_keys(left)),
            key=lambda key: (len(actual_by_key.get(key, ())), repr(key)),
        )
        pool: list[int] = []
        seen: set[int] = set()
        for key in keys:
            for right in actual_by_key.get(key, ())[: width * 8]:
                if right not in seen:
                    seen.add(right)
                    pool.append(right)
            if len(pool) >= width * 8:
                break
        if not pool and actual_indices:
            pool = [actual_indices[position % len(actual_indices)]]
        ranked = sorted(
            pool,
            key=lambda right: (-_clamp(cheap_score(left, right)), right),
        )
        pairs.update((left, right) for right in ranked[:width])

    covered_right = {right for _, right in pairs}
    for position, right in enumerate(actual_indices):
        if right not in covered_right and expected_indices:
            left = expected_indices[position % len(expected_indices)]
            pairs.add((left, right))

    if len(pairs) > max_edges:
        mandatory: set[tuple[int, int]] = set()
        for position, left in enumerate(expected_indices):
            if actual_indices:
                mandatory.add(
                    (left, actual_indices[position % len(actual_indices)])
                )
        for position, right in enumerate(actual_indices):
            if expected_indices:
                mandatory.add(
                    (expected_indices[position % len(expected_indices)], right)
                )
        remaining = sorted(
            pairs - mandatory,
            key=lambda pair: (
                -_clamp(cheap_score(pair[0], pair[1])),
                pair[0],
                pair[1],
            ),
        )
        keep = max(0, max_edges - len(mandatory))
        pairs = mandatory | set(remaining[:keep])
    diagnostics = ["blocked_candidate_generation"]
    if max_edges < max(len(expected_indices), len(actual_indices)):
        diagnostics.append("candidate_edge_budget_below_vertex_coverage")
    return pairs, False, tuple(diagnostics)


def certified_assignment(
    expected_count: int,
    actual_count: int,
    *,
    score: PairScoreFunction,
    cheap_score: PairScoreFunction,
    expected_exact_key: ExactKeyFunction,
    actual_exact_key: ExactKeyFunction,
    expected_block_keys: BlockKeyFunction,
    actual_block_keys: BlockKeyFunction,
    max_edges: int,
    top_k: int,
    exact_dense_limit: int = 64,
    max_hungarian_work: int = 50_000_000,
    max_sparse_relaxations: int = 50_000_000,
) -> AssignmentResultV52:
    """Preallocate exact identities, then certify residual fuzzy assignment."""

    if min(expected_count, actual_count, max_edges, top_k) < 0:
        raise ValueError("matching counts and budgets must be non-negative")
    actual_groups: dict[Hashable, deque[int]] = defaultdict(deque)
    for right in range(actual_count):
        key = actual_exact_key(right)
        if key is not None:
            actual_groups[key].append(right)

    exact_matches: list[MatchEdge] = []
    used_expected: set[int] = set()
    used_actual: set[int] = set()
    evaluated = 0
    for left in range(expected_count):
        key = expected_exact_key(left)
        if key is None:
            continue
        queue = actual_groups.get(key)
        while queue:
            right = queue.popleft()
            if right in used_actual:
                continue
            value = _clamp(score(left, right))
            evaluated += 1
            if value >= 1.0 - _TOLERANCE:
                exact_matches.append(MatchEdge(left, right, 1.0))
                used_expected.add(left)
                used_actual.add(right)
                break

    residual_expected = [
        index for index in range(expected_count) if index not in used_expected
    ]
    residual_actual = [
        index for index in range(actual_count) if index not in used_actual
    ]
    pairs, generation_complete, edge_codes = _candidate_pairs(
        residual_expected,
        residual_actual,
        expected_block_keys=expected_block_keys,
        actual_block_keys=actual_block_keys,
        cheap_score=cheap_score,
        max_edges=max_edges,
        top_k=top_k,
    )
    residual_edges = [
        MatchEdge(left, right, score(left, right))
        for left, right in sorted(pairs)
    ]
    evaluated += len(residual_edges)

    if residual_expected and residual_actual:
        left_remap = {
            original: local
            for local, original in enumerate(residual_expected)
        }
        right_remap = {
            original: local
            for local, original in enumerate(residual_actual)
        }
        local_edges = [
            MatchEdge(
                left_remap[edge.expected_index],
                right_remap[edge.actual_index],
                edge.score,
            )
            for edge in residual_edges
        ]
        residual_result = maximum_weight_assignment(
            len(residual_expected),
            len(residual_actual),
            local_edges,
            exact_dense_limit=exact_dense_limit,
            candidate_generation_complete=generation_complete,
            max_hungarian_work=max_hungarian_work,
            max_sparse_relaxations=max_sparse_relaxations,
        )
        fuzzy_matches = [
            MatchEdge(
                residual_expected[edge.expected_index],
                residual_actual[edge.actual_index],
                edge.score,
            )
            for edge in residual_result.matches
        ]
        residual_certification = residual_result.certification
    else:
        fuzzy_matches = []
        residual_lower = 0.0
        residual_certification = AssignmentCertification(
            vertex_edge_coverage_complete=True,
            full_cardinality_matching_exists=True,
            optimality_certified=True,
            approximate_matching_used=False,
            candidate_generation_complete=True,
            lower_bound_weight=residual_lower,
            upper_bound_weight=residual_lower,
            diagnostic_codes=("empty_residual_certified",),
        )

    matches = tuple(
        sorted(
            [*exact_matches, *fuzzy_matches],
            key=lambda edge: (edge.expected_index, edge.actual_index),
        )
    )
    lower = sum(edge.score for edge in matches)
    theoretical_upper = float(min(expected_count, actual_count))
    saturated = abs(lower - theoretical_upper) <= _TOLERANCE
    optimality = residual_certification.optimality_certified or saturated
    full_cardinality = len(matches) == min(expected_count, actual_count)
    covered_expected = {edge.expected_index for edge in matches} | {
        edge.expected_index for edge in residual_edges
    }
    covered_actual = {edge.actual_index for edge in matches} | {
        edge.actual_index for edge in residual_edges
    }
    vertex_coverage = (
        (expected_count == 0 or len(covered_expected) == expected_count)
        and (actual_count == 0 or len(covered_actual) == actual_count)
    )
    diagnostics = list(edge_codes)
    diagnostics.extend(residual_certification.diagnostic_codes)
    if exact_matches:
        diagnostics.append("exact_identity_preallocation")
    if optimality:
        diagnostics.append("global_optimality_certified")
    else:
        diagnostics.append("global_optimality_uncertified")
    return AssignmentResultV52(
        matches=matches,
        expected_count=expected_count,
        actual_count=actual_count,
        evaluated_edge_count=evaluated,
        candidate_edge_count=len(exact_matches) + len(residual_edges),
        certification=AssignmentCertification(
            vertex_edge_coverage_complete=vertex_coverage,
            full_cardinality_matching_exists=full_cardinality,
            optimality_certified=optimality,
            approximate_matching_used=(
                residual_certification.approximate_matching_used
            ),
            candidate_generation_complete=(
                residual_certification.candidate_generation_complete
            ),
            lower_bound_weight=lower,
            upper_bound_weight=lower if optimality else theoretical_upper,
            diagnostic_codes=tuple(dict.fromkeys(diagnostics)),
        ),
        exact_preallocated_count=len(exact_matches),
        fuzzy_residual_count=len(residual_expected) + len(residual_actual),
    )


def certification_to_mapping(
    certification: AssignmentCertification,
) -> dict[str, Any]:
    return {
        "vertex_edge_coverage_complete": (
            certification.vertex_edge_coverage_complete
        ),
        "full_cardinality_matching_exists": (
            certification.full_cardinality_matching_exists
        ),
        "optimality_certified": certification.optimality_certified,
        "approximate_matching_used": (
            certification.approximate_matching_used
        ),
        "candidate_generation_complete": (
            certification.candidate_generation_complete
        ),
        "lower_bound_weight": certification.lower_bound_weight,
        "upper_bound_weight": certification.upper_bound_weight,
        "diagnostic_codes": list(certification.diagnostic_codes),
    }


def assignment_to_mapping(result: AssignmentResultV52) -> dict[str, Any]:
    return {
        "expected_count": result.expected_count,
        "actual_count": result.actual_count,
        "evaluated_edge_count": result.evaluated_edge_count,
        "candidate_edge_count": result.candidate_edge_count,
        "exact_preallocated_count": result.exact_preallocated_count,
        "fuzzy_residual_count": result.fuzzy_residual_count,
        "certification": certification_to_mapping(result.certification),
    }


__all__ = [
    "AssignmentCertification",
    "AssignmentResultV52",
    "MATCHING_POLICY_VERSION",
    "MatchEdge",
    "assignment_to_mapping",
    "certification_to_mapping",
    "certified_assignment",
    "maximum_weight_assignment",
    "normalized_text_hash",
]
