"""Source-applicable, prepared-index association metrics for v5.3."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json
import math
from typing import Any, Hashable, Mapping, Sequence
from urllib.parse import urlsplit

from . import _core
from .applicability_v5_3 import (
    canonical_representation_policy,
    classify_action,
    classify_media,
)
from .matching_v5_1 import PreparedTextBlock, prepare_text_block
from .matching_v5_2 import (
    AssignmentCertification,
    AssignmentResultV52,
    assignment_to_mapping,
)
from .matching_v5_3 import (
    certified_assignment_v5_3,
    group_exact_indices,
    normalized_text_hash,
)
from .metrics_v5_2 import (
    clamp01,
    combine_certifications,
    f_score,
    prepared_block_similarity,
    table_fidelity_v5_2,
)


SCORING_POLICY_VERSION = "5.3.0"


def _normalized(value: Any) -> str:
    return _core.normalize_match_text(value)


def _canonical_action_type(value: Any) -> str:
    return str(value or "").strip().casefold().replace("_", "")


def _canonical_alias(
    value: str,
    aliases: Mapping[str, set[str]],
) -> str:
    normalized = _core.normalize_url(value)
    equivalents = {normalized}
    for key, members in aliases.items():
        group = {
            _core.normalize_url(key),
            *(_core.normalize_url(member) for member in members),
        }
        if normalized in group:
            equivalents.update(group)
    return min(equivalents) if equivalents else normalized


def _url_block_keys(value: str) -> tuple[Hashable, ...]:
    normalized = _core.normalize_url(value)
    parsed = urlsplit(normalized)
    path = parsed.path.rstrip("/")
    return (
        ("url_host", parsed.hostname or ""),
        ("url_path", path),
        ("url_file", path.rsplit("/", 1)[-1]),
    )


def _cheap_text_similarity(
    source: PreparedTextBlock,
    output: PreparedTextBlock,
) -> float:
    if source.normalized == output.normalized:
        return 1.0
    left = set(source.tokens)
    right = set(output.tokens)
    return len(left & right) / max(1, len(left | right))


def _text_exact_key(block: PreparedTextBlock) -> str | None:
    return (
        normalized_text_hash(block.normalized)
        if block.normalized
        else None
    )


def _text_block_keys(block: PreparedTextBlock) -> tuple[Hashable, ...]:
    keys: list[Hashable] = [
        ("length", min(64, len(block.tokens) // 4))
    ]
    if block.tokens:
        keys.extend(
            [
                ("first", block.tokens[0]),
                ("last", block.tokens[-1]),
            ]
        )
        keys.extend(
            ("token", token) for token in sorted(set(block.tokens))
        )
    return tuple(keys)


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


def content_exact_key_v5_3(block: PreparedTextBlock) -> str | None:
    return _text_exact_key(block)


def content_fidelity_v5_3(
    source_units: Sequence[str] | Sequence[PreparedTextBlock],
    output_blocks: Sequence[str] | Sequence[PreparedTextBlock],
    *,
    expected_exact_index: Mapping[Hashable, Sequence[int]] | None,
    exact_dense_limit: int,
    max_edges: int,
    top_k: int,
    max_hungarian_work: int = 50_000_000,
    max_sparse_relaxations: int = 50_000_000,
) -> tuple[dict[str, float | None], dict[str, Any]]:
    source = tuple(
        value
        if isinstance(value, PreparedTextBlock)
        else prepare_text_block(str(value))
        for value in source_units
    )
    output = tuple(
        value
        if isinstance(value, PreparedTextBlock)
        else prepare_text_block(str(value))
        for value in output_blocks
    )
    if not source and not output:
        empty = AssignmentResultV52(
            matches=(),
            expected_count=0,
            actual_count=0,
            evaluated_edge_count=0,
            candidate_edge_count=0,
            certification=AssignmentCertification(
                True,
                True,
                True,
                False,
                True,
                0.0,
                0.0,
                ("empty_assignment_certified",),
            ),
        )
        return {
            "content_unit_fidelity": None,
            "content_order_preservation": None,
            "output_block_precision": None,
        }, {
            "matches": [],
            "matching": assignment_to_mapping(empty),
        }
    assignment = certified_assignment_v5_3(
        len(source),
        len(output),
        score=lambda left, right: prepared_block_similarity(
            source[left], output[right]
        ),
        cheap_score=lambda left, right: _cheap_text_similarity(
            source[left], output[right]
        ),
        expected_exact_key=lambda index: _text_exact_key(source[index]),
        actual_exact_key=lambda index: _text_exact_key(output[index]),
        expected_block_keys=lambda index: _text_block_keys(source[index]),
        actual_block_keys=lambda index: _text_block_keys(output[index]),
        expected_exact_index=expected_exact_index,
        max_edges=max_edges,
        top_k=top_k,
        exact_dense_limit=exact_dense_limit,
        max_hungarian_work=max_hungarian_work,
        max_sparse_relaxations=max_sparse_relaxations,
    )
    certified = assignment.certification.optimality_certified
    mass = sum(edge.score for edge in assignment.matches)
    recall = mass / len(source) if source else 1.0
    precision = mass / len(output) if output else 1.0
    meaningful = [
        (edge.expected_index, edge.actual_index)
        for edge in assignment.matches
        if edge.score >= 0.15
    ]
    actual_order = [
        left
        for left, _ in sorted(meaningful, key=lambda pair: pair[1])
    ]
    expected_order = sorted(left for left, _ in meaningful)
    order_score = (
        _lcs_norm(
            [str(value) for value in actual_order],
            [str(value) for value in expected_order],
        )
        if len(meaningful) >= 2
        else 1.0
        if meaningful
        else 0.0
    )
    return {
        "content_unit_fidelity": (
            f_score(precision, recall, 2.0) if certified else None
        ),
        "content_order_preservation": (
            order_score if source and certified else None
        ),
        "output_block_precision": (
            precision if output and certified else None
        ),
    }, {
        "source_unit_recall": recall,
        "output_block_precision": precision,
        "matching": assignment_to_mapping(assignment),
        "prepared_exact_index_used": expected_exact_index is not None,
        "matches": [
            {
                "source_index": edge.expected_index,
                "output_index": edge.actual_index,
                "similarity": edge.score,
            }
            for edge in assignment.matches
        ],
    }


def table_fidelity_v5_3(
    expected: Sequence[_core.SourceTable],
    output_tables: Sequence[_core.OutputTable],
    output_charts: Sequence[_core.OutputTable],
    **kwargs: Any,
) -> tuple[float | None, dict[str, Any]]:
    expanded_policy: list[_core.SourceTable] = []
    for item in expected:
        policy = canonical_representation_policy(item.representation_policy)
        if policy == "both_required":
            expanded_policy.append(
                replace(
                    item,
                    representation_policy="table_only",
                    minimum_count=item.minimum_count,
                )
            )
            expanded_policy.append(
                replace(
                    item,
                    representation_policy="chart_only",
                    minimum_count=item.minimum_count,
                )
            )
        else:
            expanded_policy.append(
                replace(item, representation_policy=policy)
            )
    return table_fidelity_v5_2(
        expanded_policy, output_tables, output_charts, **kwargs
    )


def action_exact_key_expected_v5_3(
    item: _core.ActionRef,
    aliases: Mapping[str, set[str]],
) -> Hashable:
    base: tuple[Hashable, ...] = (
        _canonical_action_type(item.action_type),
        _canonical_alias(item.url, aliases),
        _normalized(item.label),
    )
    return (
        (*base, ("component", item.expected_component_id))
        if item.expected_component_id
        else base
    )


def action_exact_key_actual_v5_3(
    item: _core.OutputAction,
    aliases: Mapping[str, set[str]],
    constrained_ids: set[str],
) -> Hashable:
    base: tuple[Hashable, ...] = (
        _canonical_action_type(item.action_type),
        _canonical_alias(item.url, aliases),
        _normalized(item.label),
    )
    return (
        (*base, ("component", item.element_id))
        if item.element_id in constrained_ids
        else base
    )


def _expanded_actions(
    expected: Sequence[_core.ActionRef],
) -> list[_core.ActionRef]:
    return [
        item
        for item in expected
        if item.required and item.minimum_count > 0
        for _ in range(max(1, item.minimum_count))
    ]


def _action_match_score(
    expected: _core.ActionRef,
    actual: _core.OutputAction,
    aliases: Mapping[str, set[str]],
) -> float:
    if _canonical_action_type(expected.action_type) != _canonical_action_type(
        actual.action_type
    ):
        return 0.0
    if not _core._action_target_equivalent(expected, actual, aliases):  # type: ignore[attr-defined]
        return 0.0
    if (
        expected.expected_component_id
        and expected.expected_component_id != actual.element_id
    ):
        return 0.0
    label = (
        _core.text_similarity(expected.label, actual.label)
        if expected.label
        else 1.0
    )
    return clamp01(0.60 + 0.40 * label)


def action_fidelity_v5_3(
    expected: Sequence[_core.ActionRef],
    actual: Sequence[_core.OutputAction],
    *,
    aliases: Mapping[str, set[str]],
    expected_exact_index: Mapping[Hashable, Sequence[int]] | None,
    exact_dense_limit: int,
    max_edges: int,
    top_k: int,
    max_hungarian_work: int = 50_000_000,
    max_sparse_relaxations: int = 50_000_000,
) -> tuple[float | None, dict[str, Any]]:
    expanded = _expanded_actions(expected)
    if not expanded:
        return None, {
            "required_count": 0,
            "matched_required_count": 0,
            "ignored_candidate_count": len(actual),
            "matching": {},
        }
    constrained = {
        item.expected_component_id
        for item in expanded
        if item.expected_component_id
    }
    assignment = certified_assignment_v5_3(
        len(expanded),
        len(actual),
        score=lambda left, right: _action_match_score(
            expanded[left], actual[right], aliases
        ),
        cheap_score=lambda left, right: float(
            _canonical_action_type(expanded[left].action_type)
            == _canonical_action_type(actual[right].action_type)
        )
        * (
            0.5
            + 0.5
            * _core.text_similarity(
                expanded[left].label, actual[right].label
            )
        ),
        expected_exact_key=lambda index: action_exact_key_expected_v5_3(
            expanded[index], aliases
        ),
        actual_exact_key=lambda index: action_exact_key_actual_v5_3(
            actual[index], aliases, constrained
        ),
        expected_block_keys=lambda index: (
            (
                "action_type",
                _canonical_action_type(expanded[index].action_type),
            ),
            *_url_block_keys(expanded[index].url),
            *[
                ("label_token", token)
                for token in _core.tokenize(expanded[index].label)
            ],
        ),
        actual_block_keys=lambda index: (
            (
                "action_type",
                _canonical_action_type(actual[index].action_type),
            ),
            *_url_block_keys(actual[index].url),
            *[
                ("label_token", token)
                for token in _core.tokenize(actual[index].label)
            ],
        ),
        expected_exact_index=expected_exact_index,
        max_edges=max_edges,
        top_k=top_k,
        exact_dense_limit=exact_dense_limit,
        max_hungarian_work=max_hungarian_work,
        max_sparse_relaxations=max_sparse_relaxations,
    )
    mass = sum(edge.score for edge in assignment.matches)
    precision = mass / len(actual) if actual else 0.0
    recall = mass / len(expanded)
    matched = sum(edge.score >= 0.75 for edge in assignment.matches)
    certified = assignment.certification.optimality_certified
    return (
        f_score(precision, recall, 2.0) if certified else None
    ), {
        "required_count": len(expanded),
        "matched_required_count": matched,
        "precision": precision,
        "recall": recall,
        "role_coverage": matched / len(expanded),
        "matching": assignment_to_mapping(assignment),
        "prepared_exact_index_used": expected_exact_index is not None,
        "matches": [
            {
                "expected_index": edge.expected_index,
                "output_index": edge.actual_index,
                "score": edge.score,
            }
            for edge in assignment.matches
        ],
    }


def _expanded_media(
    expected: Sequence[_core.MediaRef],
) -> list[_core.MediaRef]:
    exact: list[_core.MediaRef] = []
    representative: dict[str, _core.MediaRef] = {}
    for item in expected:
        if not item.required or item.minimum_count <= 0:
            continue
        if item.media_policy.casefold() == "representative":
            representative.setdefault(item.kind.casefold(), item)
        else:
            exact.extend([item] * max(1, item.minimum_count))
    return exact + list(representative.values())


def media_exact_key_expected_v5_3(
    item: _core.MediaRef,
    aliases: Mapping[str, set[str]],
) -> Hashable:
    base: tuple[Hashable, ...] = (
        item.kind.casefold(),
        _canonical_alias(item.url, aliases),
        _normalized(item.alt) if item.alt else None,
    )
    return (
        (*base, ("component", item.expected_component_id))
        if item.expected_component_id
        else base
    )


def media_exact_key_actual_v5_3(
    item: _core.MediaRef,
    aliases: Mapping[str, set[str]],
    *,
    alt_required: bool,
    constrained_ids: set[str],
) -> Hashable:
    base: tuple[Hashable, ...] = (
        item.kind.casefold(),
        _canonical_alias(item.url, aliases),
        _normalized(item.alt) if alt_required else None,
    )
    return (
        (*base, ("component", item.component_id))
        if item.component_id in constrained_ids
        else base
    )


def _media_match_score(
    expected: _core.MediaRef,
    actual: _core.MediaRef,
    aliases: Mapping[str, set[str]],
) -> float:
    if expected.kind.casefold() != actual.kind.casefold():
        return 0.0
    if expected.url and not _core._url_equivalent(  # type: ignore[attr-defined]
        expected.url, actual.url, aliases
    ):
        return 0.0
    if (
        expected.expected_component_id
        and expected.expected_component_id != actual.component_id
    ):
        return 0.0
    alt = (
        _core.text_similarity(expected.alt, actual.alt)
        if expected.alt
        else 1.0
    )
    return clamp01(0.75 + 0.25 * alt)


def media_fidelity_v5_3(
    expected: Sequence[_core.MediaRef],
    actual: Sequence[_core.MediaRef],
    *,
    aliases: Mapping[str, set[str]],
    expected_exact_index: Mapping[Hashable, Sequence[int]] | None,
    exact_dense_limit: int,
    max_edges: int,
    top_k: int,
    max_hungarian_work: int = 50_000_000,
    max_sparse_relaxations: int = 50_000_000,
) -> tuple[float | None, dict[str, Any]]:
    expanded = _expanded_media(expected)
    if not expanded:
        return None, {
            "required_count": 0,
            "matched_required_count": 0,
            "ignored_candidate_count": len(actual),
            "matching": {},
        }
    alt_required = all(bool(item.alt) for item in expanded)
    constrained = {
        item.expected_component_id
        for item in expanded
        if item.expected_component_id
    }
    assignment = certified_assignment_v5_3(
        len(expanded),
        len(actual),
        score=lambda left, right: _media_match_score(
            expanded[left], actual[right], aliases
        ),
        cheap_score=lambda left, right: float(
            expanded[left].kind.casefold() == actual[right].kind.casefold()
        )
        * (
            0.75
            + 0.25
            * _core.text_similarity(
                expanded[left].alt, actual[right].alt
            )
        ),
        expected_exact_key=lambda index: media_exact_key_expected_v5_3(
            expanded[index], aliases
        ),
        actual_exact_key=lambda index: media_exact_key_actual_v5_3(
            actual[index],
            aliases,
            alt_required=alt_required,
            constrained_ids=constrained,
        ),
        expected_block_keys=lambda index: (
            ("media_kind", expanded[index].kind.casefold()),
            *_url_block_keys(expanded[index].url),
            *[
                ("alt_token", token)
                for token in _core.tokenize(expanded[index].alt)
            ],
        ),
        actual_block_keys=lambda index: (
            ("media_kind", actual[index].kind.casefold()),
            *_url_block_keys(actual[index].url),
            *[
                ("alt_token", token)
                for token in _core.tokenize(actual[index].alt)
            ],
        ),
        expected_exact_index=expected_exact_index,
        max_edges=max_edges,
        top_k=top_k,
        exact_dense_limit=exact_dense_limit,
        max_hungarian_work=max_hungarian_work,
        max_sparse_relaxations=max_sparse_relaxations,
    )
    mass = sum(edge.score for edge in assignment.matches)
    precision = mass / len(actual) if actual else 0.0
    recall = mass / len(expanded)
    matched = sum(edge.score >= 0.75 for edge in assignment.matches)
    certified = assignment.certification.optimality_certified
    return (
        f_score(precision, recall, 2.0) if certified else None
    ), {
        "required_count": len(expanded),
        "matched_required_count": matched,
        "precision": precision,
        "recall": recall,
        "role_coverage": matched / len(expanded),
        "matching": assignment_to_mapping(assignment),
        "prepared_exact_index_used": expected_exact_index is not None,
        "matches": [
            {
                "expected_index": edge.expected_index,
                "output_index": edge.actual_index,
                "score": edge.score,
            }
            for edge in assignment.matches
        ],
    }


_ROLE_META_FIELDS = frozenset(
    {
        "id",
        "required",
        "minimum_count",
        "interchangeable",
        "signature",
        "component_id",
        "requirement_id",
        "expected_component_id",
        # Extraction provenance is retained in diagnostics but is not a
        # renderer-observable semantic field.
        "source_section",
    }
)


def role_payload_v5_3(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): nested
        for key, nested in value.items()
        if key not in _ROLE_META_FIELDS
        and nested not in (None, "", [], {})
    }


def semantic_signature_v5_3(value: Mapping[str, Any]) -> str:
    normalized: dict[str, Any] = {}
    for key, nested in sorted(role_payload_v5_3(value).items()):
        if isinstance(nested, Sequence) and not isinstance(
            nested, (str, bytes, bytearray)
        ):
            normalized[key] = sorted(
                _normalized(item)
                for item in nested
                if _normalized(item)
            )
        else:
            normalized[key] = _normalized(nested)
    return hashlib.sha256(
        json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def role_exact_key_expected_v5_3(
    role: str,
    item: Mapping[str, Any],
) -> Hashable:
    base: tuple[Hashable, ...] = (
        role.casefold(),
        semantic_signature_v5_3(item),
    )
    component = str(item.get("expected_component_id") or "")
    return (*base, ("component", component)) if component else base


def role_exact_key_actual_v5_3(
    role: str,
    item: Mapping[str, Any],
    constrained_ids: set[str],
) -> Hashable:
    base: tuple[Hashable, ...] = (
        role.casefold(),
        semantic_signature_v5_3(item),
    )
    component = str(item.get("component_id") or "")
    return (
        (*base, ("component", component))
        if component in constrained_ids
        else base
    )


def role_instance_similarity_v5_3(
    requirement: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> float:
    expected_component = str(
        requirement.get("expected_component_id") or ""
    )
    if expected_component and expected_component != str(
        actual.get("component_id") or ""
    ):
        return 0.0
    fields = role_payload_v5_3(requirement)
    if not fields:
        return 0.0
    scores: list[float] = []
    for key, expected in sorted(fields.items()):
        observed = actual.get(key)
        if isinstance(expected, Sequence) and not isinstance(
            expected, (str, bytes, bytearray)
        ):
            expected_values = {
                _normalized(value)
                for value in expected
                if _normalized(value)
            }
            actual_values = (
                {
                    _normalized(value)
                    for value in observed
                    if _normalized(value)
                }
                if isinstance(observed, Sequence)
                and not isinstance(observed, (str, bytes, bytearray))
                else {_normalized(observed)}
            )
            scores.append(
                len(expected_values & actual_values)
                / max(1, len(expected_values))
            )
        else:
            scores.append(
                _core.text_similarity(
                    str(expected), str(observed or "")
                )
            )
    return sum(scores) / len(scores)


def semantic_role_coverage_v5_3(
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
        required_count = sum(
            max(1, int(item.get("minimum_count", 1) or 1))
            for item in requirements
        )
        total_required += required_count
        count_values[role] = (
            min(1.0, len(candidates) / required_count)
            if required_count
            else None
        )
        semantic_requirements = [
            item for item in requirements if role_payload_v5_3(item)
        ]
        expanded = [
            item
            for item in semantic_requirements
            for _ in range(max(1, int(item.get("minimum_count", 1) or 1)))
        ]
        total_semantic += len(expanded)
        if not expanded:
            semantic_values[role] = None
            diagnostics[role] = {
                "required_count": required_count,
                "semantic_required_count": 0,
                "actual_count": len(candidates),
                "role_count_coverage": count_values[role],
                "semantic_role_instance_fidelity": None,
                "count_only": required_count > 0,
                "matching": {},
            }
            continue
        constrained = {
            str(item.get("expected_component_id") or "")
            for item in expanded
            if str(item.get("expected_component_id") or "")
        }
        assignment = certified_assignment_v5_3(
            len(expanded),
            len(candidates),
            score=lambda left, right: role_instance_similarity_v5_3(
                expanded[left], candidates[right]
            ),
            cheap_score=lambda left, right: _core.text_similarity(
                " ".join(
                    str(value)
                    for value in role_payload_v5_3(
                        expanded[left]
                    ).values()
                ),
                " ".join(
                    str(value)
                    for value in role_payload_v5_3(
                        candidates[right]
                    ).values()
                ),
            ),
            expected_exact_key=lambda index: role_exact_key_expected_v5_3(
                role, expanded[index]
            ),
            actual_exact_key=lambda index: role_exact_key_actual_v5_3(
                role, candidates[index], constrained
            ),
            expected_block_keys=lambda index: (
                ("role", role),
                *[
                    ("role_token", token)
                    for token in _core.tokenize(
                        " ".join(
                            str(value)
                            for value in role_payload_v5_3(
                                expanded[index]
                            ).values()
                        )
                    )
                ],
            ),
            actual_block_keys=lambda index: (
                ("role", role),
                *[
                    ("role_token", token)
                    for token in _core.tokenize(
                        " ".join(
                            str(value)
                            for value in role_payload_v5_3(
                                candidates[index]
                            ).values()
                        )
                    )
                ],
            ),
            expected_exact_index=(
                expected_exact_indices.get(role)
                if expected_exact_indices is not None
                else None
            ),
            max_edges=max_edges,
            top_k=top_k,
            exact_dense_limit=exact_dense_limit,
            max_hungarian_work=max_hungarian_work,
            max_sparse_relaxations=max_sparse_relaxations,
        )
        accepted = []
        used_noninterchangeable_signatures: set[str] = set()
        for edge in sorted(
            assignment.matches,
            key=lambda item: (
                -item.score,
                item.expected_index,
                item.actual_index,
            ),
        ):
            if edge.score < threshold:
                continue
            requirement = expanded[edge.expected_index]
            signature = semantic_signature_v5_3(
                candidates[edge.actual_index]
            )
            if (
                not bool(requirement.get("interchangeable", False))
                and signature in used_noninterchangeable_signatures
            ):
                continue
            if not bool(requirement.get("interchangeable", False)):
                used_noninterchangeable_signatures.add(signature)
            accepted.append(edge)
        coverage = len(accepted) / len(expanded)
        semantic_values[role] = (
            coverage
            if assignment.certification.optimality_certified
            else None
        )
        diagnostics[role] = {
            "required_count": required_count,
            "semantic_required_count": len(expanded),
            "actual_count": len(candidates),
            "matched_required_count": len(accepted),
            "role_count_coverage": count_values[role],
            "semantic_role_instance_fidelity": semantic_values[role],
            "count_only": False,
            "matching": assignment_to_mapping(assignment),
            "prepared_exact_index_used": (
                expected_exact_indices is not None
            ),
            "matches": [
                {
                    "expected_index": edge.expected_index,
                    "output_index": edge.actual_index,
                    "score": edge.score,
                }
                for edge in accepted
            ],
        }
    counts = [
        float(value) for value in count_values.values() if value is not None
    ]
    semantics = [
        float(value)
        for value in semantic_values.values()
        if value is not None
    ]
    specificity = (
        total_semantic / total_required if total_required else 1.0
    )
    return (
        sum(semantics) / len(semantics) if semantics else None,
        sum(counts) / len(counts) if counts else None,
        count_values,
        semantic_values,
        diagnostics,
        specificity,
    )


def unsupported_external_addition_precision_v5_3(
    expected_actions: Sequence[_core.ActionRef],
    actual_actions: Sequence[_core.OutputAction],
    expected_media: Sequence[_core.MediaRef],
    actual_media: Sequence[_core.MediaRef],
    *,
    aliases: Mapping[str, set[str]],
) -> tuple[float, dict[str, Any]]:
    external_actions = [
        item
        for item in actual_actions
        if classify_action(item.action_type, item.url)
        == "external_semantic"
    ]
    supported_action_count = 0
    remaining = list(external_actions)
    for expected in expected_actions:
        for index, actual in enumerate(remaining):
            if _action_match_score(expected, actual, aliases) >= 0.75:
                supported_action_count += 1
                remaining.pop(index)
                break
    semantic_media = [
        item
        for item in actual_media
        if classify_media(item.kind, item.alt)
        == "source_semantic_media"
    ]
    supported_media_count = 0
    remaining_media = list(semantic_media)
    for expected in expected_media:
        for index, actual in enumerate(remaining_media):
            if _media_match_score(expected, actual, aliases) >= 0.75:
                supported_media_count += 1
                remaining_media.pop(index)
                break
    actual_count = len(external_actions) + len(semantic_media)
    supported = supported_action_count + supported_media_count
    value = supported / actual_count if actual_count else 1.0
    return value, {
        "actual_external_action_count": len(external_actions),
        "unsupported_external_action_count": len(remaining),
        "actual_semantic_media_count": len(semantic_media),
        "unsupported_semantic_media_count": len(remaining_media),
        "supported_count": supported,
        "candidate_count": actual_count,
        "precision": value,
    }


__all__ = [
    "SCORING_POLICY_VERSION",
    "action_exact_key_expected_v5_3",
    "action_fidelity_v5_3",
    "clamp01",
    "combine_certifications",
    "content_exact_key_v5_3",
    "content_fidelity_v5_3",
    "f_score",
    "group_exact_indices",
    "media_exact_key_expected_v5_3",
    "media_fidelity_v5_3",
    "role_exact_key_expected_v5_3",
    "role_instance_similarity_v5_3",
    "role_payload_v5_3",
    "semantic_role_coverage_v5_3",
    "semantic_signature_v5_3",
    "table_fidelity_v5_3",
    "unsupported_external_addition_precision_v5_3",
]
