"""Prepared-index certified matching for GenUI metric v5.3."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Hashable, Mapping, Sequence

from .matching_v5_2 import (
    AssignmentCertification,
    AssignmentResultV52,
    BlockKeyFunction,
    ExactKeyFunction,
    MatchEdge,
    PairScoreFunction,
    _candidate_pairs,
    _clamp,
    assignment_to_mapping,
    certification_to_mapping,
    maximum_weight_assignment,
    normalized_text_hash,
)


MATCHING_POLICY_VERSION = "5.3.0"
_TOLERANCE = 1e-12


def group_exact_indices(
    keys: Sequence[Hashable | None],
) -> dict[Hashable, tuple[int, ...]]:
    grouped: dict[Hashable, list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        if key is not None:
            grouped[key].append(index)
    return {
        key: tuple(values)
        for key, values in sorted(grouped.items(), key=lambda item: repr(item[0]))
    }


def certified_assignment_v5_3(
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
    expected_exact_index: Mapping[Hashable, Sequence[int]] | None = None,
    exact_dense_limit: int = 64,
    max_hungarian_work: int = 50_000_000,
    max_sparse_relaxations: int = 50_000_000,
) -> AssignmentResultV52:
    """Preallocate exact identities using the prepared source index.

    Requirement IDs never participate unless a key builder explicitly includes
    ``expected_component_id``. Duplicate multiplicity is preserved by queues.
    """

    if min(expected_count, actual_count, max_edges, top_k) < 0:
        raise ValueError("matching counts and budgets must be non-negative")
    actual_groups: dict[Hashable, deque[int]] = defaultdict(deque)
    for right in range(actual_count):
        key = actual_exact_key(right)
        if key is not None:
            actual_groups[key].append(right)
    source_index = (
        {
            key: tuple(int(index) for index in values)
            for key, values in expected_exact_index.items()
        }
        if expected_exact_index is not None
        else group_exact_indices(
            [expected_exact_key(index) for index in range(expected_count)]
        )
    )

    exact_matches: list[MatchEdge] = []
    used_expected: set[int] = set()
    used_actual: set[int] = set()
    evaluated = 0
    for key in sorted(source_index, key=repr):
        queue = actual_groups.get(key)
        if not queue:
            continue
        for left in source_index[key]:
            if left < 0 or left >= expected_count or left in used_expected:
                continue
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
        residual_certification = AssignmentCertification(
            vertex_edge_coverage_complete=True,
            full_cardinality_matching_exists=True,
            optimality_certified=True,
            approximate_matching_used=False,
            candidate_generation_complete=True,
            lower_bound_weight=0.0,
            upper_bound_weight=0.0,
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
    if expected_exact_index is not None:
        diagnostics.append("prepared_source_exact_index_used")
    if exact_matches:
        diagnostics.append("exact_identity_preallocation")
    diagnostics.append(
        "global_optimality_certified"
        if optimality
        else "global_optimality_uncertified"
    )
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


__all__ = [
    "AssignmentCertification",
    "AssignmentResultV52",
    "MATCHING_POLICY_VERSION",
    "MatchEdge",
    "assignment_to_mapping",
    "certification_to_mapping",
    "certified_assignment_v5_3",
    "group_exact_indices",
    "normalized_text_hash",
]
