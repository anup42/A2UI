"""Constrained semantic matching and ownership-aware metrics for v5.4."""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any, Hashable, Mapping, Sequence

from . import _core
from .matching_v5_2 import assignment_to_mapping
from .matching_v5_3 import certified_assignment_v5_3
from .metrics_v5_3 import (
    clamp01,
    role_instance_similarity_v5_3,
    role_payload_v5_3,
    semantic_signature_v5_3,
    _action_match_score,
    _media_match_score,
)
from .applicability_v5_3 import classify_action, classify_media
from .ownership_v5_4 import (
    build_output_ownership,
    build_source_ownership,
    cross_channel_duplication_utility,
)


SCORING_POLICY_VERSION = "5.4.1"
ROLE_MATCHING_POLICY_VERSION = "5.4.1"


def normalize_formula_wrapper(value: Any) -> Any:
    """Remove one balanced display/inline wrapper, preserving math contents."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    for opening, closing in (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$")):
        if len(text) <= len(opening) + len(closing):
            continue
        if text.startswith(opening) and text.endswith(closing):
            inner = text[len(opening):-len(closing)]
            # Multiple separate expressions and escaped closing dollars are
            # not a single wrapper around an equivalent formula.
            if opening.startswith("$") and (re.search(r"(?<!\\)\$", inner) or inner.endswith("\\")):
                continue
            if opening.startswith("\\") and (opening in inner or closing in inner):
                continue
            return inner.strip()
    return text


def unsupported_external_addition_precision_v5_4(
    expected_actions: Sequence[_core.ActionRef],
    actual_actions: Sequence[_core.OutputAction],
    expected_media: Sequence[_core.MediaRef],
    actual_media: Sequence[_core.MediaRef],
    *,
    aliases: Mapping[str, set[str]],
) -> tuple[float, dict[str, Any]]:
    """Source-origin icons remain supported when rendered as labelled images.

    All source media (including decorative icons) are passed here. This does
    not make decorative media required, and matching still consumes instances.
    """
    actions = [item for item in actual_actions if classify_action(item.action_type, item.url) == "external_semantic"]
    media = [item for item in actual_media if classify_media(item.kind, item.alt) == "source_semantic_media"]
    remaining_actions = list(actions)
    for expected in expected_actions:
        for _ in range(max(0, expected.minimum_count)):
            for index, actual in enumerate(remaining_actions):
                if _action_match_score(expected, actual, aliases) >= 0.75:
                    remaining_actions.pop(index)
                    break
    remaining_media = list(media)
    icon_matches = 0
    for expected in expected_media:
        for _ in range(max(0, expected.minimum_count)):
            for index, actual in enumerate(remaining_media):
                source_icon = (
                    expected.kind.casefold() == "icon"
                    and actual.kind.casefold() in {"icon", "image"}
                    and bool(expected.url)
                    and _core._url_equivalent(expected.url, actual.url, aliases)
                    and (not expected.expected_component_id or expected.expected_component_id == actual.component_id)
                )
                if source_icon or _media_match_score(expected, actual, aliases) >= 0.75:
                    remaining_media.pop(index)
                    icon_matches += int(source_icon)
                    break
    count = len(actions) + len(media)
    unsupported = len(remaining_actions) + len(remaining_media)
    value = (count - unsupported) / count if count else 1.0
    return value, {
        "actual_external_action_count": len(actions),
        "unsupported_external_action_count": len(remaining_actions),
        "actual_semantic_media_count": len(media),
        "unsupported_semantic_media_count": len(remaining_media),
        "source_icon_supported_count": icon_matches,
        "supported_count": count - unsupported,
        "candidate_count": count,
        "precision": value,
    }


def _expanded_requirements(
    values: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    semantic: list[dict[str, Any]] = []
    count_only = 0
    for raw in values:
        item = dict(raw)
        if not bool(item.get("required", True)):
            continue
        count = max(0, int(item.get("minimum_count", 1) or 0))
        if count <= 0:
            continue
        if role_payload_v5_3(item):
            semantic.extend(dict(item) for _ in range(count))
        else:
            count_only += count
    return semantic, count_only


def _candidate_signature_groups(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        groups[semantic_signature_v5_3(candidate)].append(index)
    return dict(sorted(groups.items()))


def semantic_role_coverage_v5_4(
    expected: Mapping[str, Sequence[Mapping[str, Any]]],
    actual: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    expected_exact_indices: Mapping[
        str, Mapping[Hashable, Sequence[int]]
    ] | None,
    threshold: float,
    exact_dense_limit: int,
    max_edges: int,
    top_k: int,
    max_hungarian_work: int = 50_000_000,
    max_sparse_relaxations: int = 50_000_000,
) -> tuple[
    float | None,
    float | None,
    dict[str, float | None],
    dict[str, float | None],
    dict[str, Any],
    float,
]:
    del expected_exact_indices
    count_values: dict[str, float | None] = {}
    semantic_values: dict[str, float | None] = {}
    diagnostics: dict[str, Any] = {}
    total_required = 0
    total_semantic = 0

    for role in sorted(expected):
        requirements = [
            dict(item)
            for item in expected.get(role, ())
            if bool(item.get("required", True))
            and int(item.get("minimum_count", 1) or 0) > 0
        ]
        candidates = [dict(item) for item in actual.get(role, ())]
        if role == "formula":
            for item in [*requirements, *candidates]:
                if "content" in item:
                    item["content"] = normalize_formula_wrapper(item["content"])
        semantic, count_only = _expanded_requirements(requirements)
        required_count = len(semantic) + count_only
        total_required += required_count
        total_semantic += len(semantic)
        if not semantic:
            count_coverage = (
                min(1.0, len(candidates) / count_only)
                if count_only
                else None
            )
            count_values[role] = count_coverage
            semantic_values[role] = None
            diagnostics[role] = {
                "required_count": required_count,
                "semantic_required_count": 0,
                "count_only_required_count": count_only,
                "actual_count": len(candidates),
                "role_count_coverage": count_coverage,
                "count_only_role_count_coverage": count_coverage,
                "semantic_role_instance_fidelity": None,
                "count_only": count_only > 0,
                "matching": {},
                "constraint_model": "count_only_low_specificity",
            }
            continue

        groups = _candidate_signature_groups(candidates)
        interchangeable_capacity = sum(
            1
            for item in semantic
            if bool(item.get("interchangeable", False))
        )
        slots: list[tuple[str, int | None]] = []
        for signature, members in groups.items():
            capacity = min(
                len(members),
                max(1, interchangeable_capacity),
            )
            if capacity == 1:
                slots.append((signature, None))
            else:
                slots.extend((signature, member) for member in members[:capacity])

        def score(left: int, right: int) -> float:
            signature, member = slots[right]
            indices = (
                [member]
                if member is not None
                else groups.get(signature, ())
            )
            return max(
                (
                    role_instance_similarity_v5_3(
                        semantic[left], candidates[index]
                    )
                    for index in indices
                    if index is not None
                ),
                default=0.0,
            )

        assignment = certified_assignment_v5_3(
            len(semantic),
            len(slots),
            score=score,
            cheap_score=score,
            expected_exact_key=lambda index: None,
            actual_exact_key=lambda index: None,
            expected_block_keys=lambda index: (
                ("role", role),
                *(
                    ("token", token)
                    for token in _core.tokenize(
                        " ".join(
                            str(value)
                            for value in role_payload_v5_3(
                                semantic[index]
                            ).values()
                        )
                    )
                ),
            ),
            actual_block_keys=lambda index: (("role", role),),
            expected_exact_index=None,
            max_edges=max_edges,
            top_k=top_k,
            exact_dense_limit=exact_dense_limit,
            max_hungarian_work=max_hungarian_work,
            max_sparse_relaxations=max_sparse_relaxations,
        )
        accepted = [
            edge for edge in assignment.matches if edge.score >= threshold
        ]
        accepted_count = len(accepted)
        semantic_count_coverage = accepted_count / len(semantic)
        semantic_fidelity = (
            sum(edge.score for edge in accepted) / len(semantic)
            if assignment.certification.optimality_certified
            else None
        )
        count_only_coverage = (
            min(
                1.0,
                max(0, len(candidates) - accepted_count) / count_only,
            )
            if count_only
            else None
        )
        count_values[role] = (
            (
                accepted_count
                + (
                    count_only * float(count_only_coverage)
                    if count_only_coverage is not None
                    else 0.0
                )
            )
            / required_count
            if required_count
            else None
        )
        semantic_values[role] = semantic_fidelity
        diagnostics[role] = {
            "required_count": required_count,
            "semantic_required_count": len(semantic),
            "count_only_required_count": count_only,
            "actual_count": len(candidates),
            "distinct_candidate_signature_count": len(groups),
            "semantic_capacity_slot_count": len(slots),
            "matched_required_count": accepted_count,
            "role_count_coverage": count_values[role],
            "semantic_count_coverage": semantic_count_coverage,
            "count_only_role_count_coverage": count_only_coverage,
            "semantic_role_instance_fidelity": semantic_fidelity,
            "count_only": False,
            "matching": assignment_to_mapping(assignment),
            "constraint_model": (
                "signature_capacity_inside_certified_assignment"
            ),
            "optimality_certifies_constrained_problem": bool(
                assignment.certification.optimality_certified
            ),
            "matches": [
                {
                    "expected_index": edge.expected_index,
                    "capacity_slot_index": edge.actual_index,
                    "signature": slots[edge.actual_index][0],
                    "score": edge.score,
                }
                for edge in accepted
            ],
        }

    semantic_applicable = [
        float(value)
        for value in semantic_values.values()
        if value is not None
    ]
    count_applicable = [
        float(value) for value in count_values.values() if value is not None
    ]
    specificity = total_semantic / total_required if total_required else 1.0
    return (
        (
            sum(semantic_applicable) / len(semantic_applicable)
            if semantic_applicable
            else None
        ),
        (
            sum(count_applicable) / len(count_applicable)
            if count_applicable
            else None
        ),
        count_values,
        semantic_values,
        diagnostics,
        specificity,
    )


def semantic_duplication_score_v5_4(
    prepared: Any,
    evidence_result: Any,
    output: _core.OutputEvidence,
    *,
    cross_channel_utility: float | None = None,
) -> float:
    historical = _core._semantic_duplicate_utility(output)  # type: ignore[attr-defined]
    if cross_channel_utility is None:
        source_units = build_source_ownership(
            prepared.source_text, prepared.expected_contract
        )
        output_units = build_output_ownership(evidence_result)
        cross_channel_utility, _ = cross_channel_duplication_utility(
            source_units, output_units
        )
    return clamp01(min(historical, float(cross_channel_utility)))


__all__ = [
    "ROLE_MATCHING_POLICY_VERSION",
    "SCORING_POLICY_VERSION",
    "semantic_duplication_score_v5_4",
    "semantic_role_coverage_v5_4",
]
