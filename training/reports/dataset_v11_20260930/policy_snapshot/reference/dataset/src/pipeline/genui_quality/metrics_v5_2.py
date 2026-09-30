"""Certified association-preserving metrics for GenUI quality v5.2."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Hashable, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

from . import _core
from .matching_v5_1 import PreparedTextBlock, prepare_text_block
from .matching_v5_2 import (
    AssignmentCertification,
    AssignmentResultV52,
    assignment_to_mapping,
    certified_assignment,
    normalized_text_hash,
)


SCORING_POLICY_VERSION = "5.2.0"


def clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return max(0.0, min(1.0, numeric))


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
        Counter(source.bigrams), Counter(output.bigrams), 1.0
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


def _text_exact_key(block: PreparedTextBlock) -> str | None:
    return (
        normalized_text_hash(block.normalized)
        if block.normalized
        else None
    )


def _text_block_keys(block: PreparedTextBlock) -> tuple[Hashable, ...]:
    tokens = block.tokens
    keys: list[Hashable] = [
        ("length", min(64, len(tokens) // 4)),
    ]
    if tokens:
        keys.extend(
            [
                ("first", tokens[0]),
                ("last", tokens[-1]),
            ]
        )
        keys.extend(("token", token) for token in sorted(set(tokens)))
    return tuple(keys)


def content_fidelity_v5_2(
    source_units: Sequence[str] | Sequence[PreparedTextBlock],
    output_blocks: Sequence[str] | Sequence[PreparedTextBlock],
    *,
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
                True, True, True, False, True, 0.0, 0.0,
                ("empty_assignment_certified",),
            ),
        )
        return {
            "content_unit_fidelity": None,
            "content_order_preservation": None,
            "output_block_precision": None,
        }, {"matches": [], "matching": assignment_to_mapping(empty)}

    assignment = certified_assignment(
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
        max_edges=max_edges,
        top_k=top_k,
        exact_dense_limit=exact_dense_limit,
        max_hungarian_work=max_hungarian_work,
        max_sparse_relaxations=max_sparse_relaxations,
    )
    certified = assignment.certification.optimality_certified
    mass = sum(edge.score for edge in assignment.matches)
    source_recall = mass / len(source) if source else 1.0
    output_precision = mass / len(output) if output else 1.0
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
    values = {
        "content_unit_fidelity": (
            f_score(output_precision, source_recall, 2.0)
            if certified
            else None
        ),
        "content_order_preservation": (
            order_score if source and certified else None
        ),
        "output_block_precision": (
            output_precision if output and certified else None
        ),
    }
    return values, {
        "source_unit_recall": source_recall,
        "output_block_precision": output_precision,
        "matching": assignment_to_mapping(assignment),
        "matches": [
            {
                "source_index": edge.expected_index,
                "output_index": edge.actual_index,
                "similarity": edge.score,
            }
            for edge in assignment.matches
        ],
    }


def _output_rows(table: _core.OutputTable) -> list[list[str]]:
    return [
        [str(row.get(key, "")) for key in table.keys]
        for row in table.rows
    ]


def _normalized(value: Any) -> str:
    return _core.normalize_match_text(value)


def _find_header(headers: Sequence[str], name: str) -> int | None:
    expected = _normalized(name)
    for index, header in enumerate(headers):
        if _normalized(header) == expected:
            return index
    return None


def _row_signature(
    row: Sequence[str],
    columns: Sequence[int],
) -> tuple[str, ...]:
    return tuple(
        _normalized(row[index] if index < len(row) else "")
        for index in columns
    )


@dataclass(frozen=True)
class TablePairResultV52:
    score: float | None
    column_fbeta: float | None
    row_fbeta: float | None
    associated_cell_fbeta: float | None
    order_score: float | None
    key_precision: float | None
    key_recall: float | None
    key_exact_match_rate: float | None
    certification: dict[str, Any]
    diagnostic_codes: tuple[str, ...] = ()


def table_pair_metrics_v5_2(
    source: _core.SourceTable,
    output: _core.OutputTable,
    *,
    beta: float,
    exact_dense_limit: int,
    max_edges: int,
    top_k: int,
    max_hungarian_work: int = 50_000_000,
    max_sparse_relaxations: int = 50_000_000,
) -> TablePairResultV52:
    source_headers = tuple(
        prepare_text_block(value) for value in source.headers
    )
    output_headers = tuple(
        prepare_text_block(value) for value in output.headers
    )
    columns = certified_assignment(
        len(source_headers),
        len(output_headers),
        score=lambda left, right: _core.text_similarity(
            source_headers[left].raw, output_headers[right].raw
        ),
        cheap_score=lambda left, right: _cheap_text_similarity(
            source_headers[left], output_headers[right]
        ),
        expected_exact_key=lambda index: _text_exact_key(
            source_headers[index]
        ),
        actual_exact_key=lambda index: _text_exact_key(
            output_headers[index]
        ),
        expected_block_keys=lambda index: _text_block_keys(
            source_headers[index]
        ),
        actual_block_keys=lambda index: _text_block_keys(
            output_headers[index]
        ),
        max_edges=max_edges,
        top_k=top_k,
        exact_dense_limit=exact_dense_limit,
        max_hungarian_work=max_hungarian_work,
        max_sparse_relaxations=max_sparse_relaxations,
    )
    column_mass = sum(edge.score for edge in columns.matches)
    column_precision = (
        column_mass / len(output.headers) if output.headers else 0.0
    )
    column_recall = (
        column_mass / len(source.headers) if source.headers else 1.0
    )
    column_f = f_score(column_precision, column_recall, beta)
    aligned = [
        (edge.expected_index, edge.actual_index)
        for edge in columns.matches
        if edge.score >= 0.20
    ]

    key_pairs: list[tuple[int, int]] = []
    diagnostics: list[str] = []
    for key_name in source.row_key:
        source_index = _find_header(source.headers, key_name)
        if source_index is None:
            diagnostics.append(f"row_key_source_column_missing:{key_name}")
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
            diagnostics.append(f"row_key_output_column_missing:{key_name}")
            continue
        key_pairs.append((source_index, output_index))

    source_rows = tuple(
        tuple(str(cell) for cell in row) for row in source.rows
    )
    output_rows = tuple(tuple(row) for row in _output_rows(output))
    key_source_columns = {value[0] for value in key_pairs}
    non_key_aligned = [
        pair for pair in aligned if pair[0] not in key_source_columns
    ]

    def key_compatible(left: int, right: int) -> tuple[float, float]:
        if not source.row_key:
            return 1.0, 1.0
        if len(key_pairs) != len(source.row_key):
            return 0.0, 0.0
        comparisons = [
            float(
                _normalized(
                    source_rows[left][source_column]
                    if source_column < len(source_rows[left])
                    else ""
                )
                == _normalized(
                    output_rows[right][output_column]
                    if output_column < len(output_rows[right])
                    else ""
                )
            )
            for source_column, output_column in key_pairs
        ]
        return float(all(value == 1.0 for value in comparisons)), (
            sum(comparisons) / len(comparisons)
        )

    def row_similarity(left: int, right: int) -> float:
        if not aligned:
            return 0.0
        gate, key_score = key_compatible(left, right)
        if gate <= 0.0:
            return 0.0
        pairs = non_key_aligned or key_pairs or aligned
        similarities = [
            _core.text_similarity(
                source_rows[left][source_column]
                if source_column < len(source_rows[left])
                else "",
                output_rows[right][output_column]
                if output_column < len(output_rows[right])
                else "",
            )
            for source_column, output_column in pairs
        ]
        content_score = (
            sum(similarities) / len(similarities)
            if similarities
            else 1.0
        )
        return clamp01(
            gate
            * (
                0.75 * content_score + 0.25 * key_score
                if source.row_key
                else content_score
            )
        )

    def row_exact_key(left: int, *, source_side: bool) -> Hashable | None:
        if source_side:
            row = source_rows[left]
            columns_to_use = (
                [pair[0] for pair in key_pairs]
                if source.row_key and len(key_pairs) == len(source.row_key)
                else [pair[0] for pair in aligned]
            )
        else:
            row = output_rows[left]
            columns_to_use = (
                [pair[1] for pair in key_pairs]
                if source.row_key and len(key_pairs) == len(source.row_key)
                else [pair[1] for pair in aligned]
            )
        return _row_signature(row, columns_to_use) if columns_to_use else None

    def row_blocks(index: int, *, source_side: bool) -> tuple[Hashable, ...]:
        row = source_rows[index] if source_side else output_rows[index]
        tokens = tuple(
            token
            for cell in row
            for token in _core.tokenize(str(cell))
        )
        keys: list[Hashable] = [
            ("row_length", len(row)),
            *[("row_token", token) for token in sorted(set(tokens))],
        ]
        exact = row_exact_key(index, source_side=source_side)
        if exact is not None:
            keys.append(("row_signature", exact))
        return tuple(keys)

    rows = certified_assignment(
        len(source_rows),
        len(output_rows),
        score=row_similarity,
        cheap_score=lambda left, right: _core.text_similarity(
            " ".join(source_rows[left]), " ".join(output_rows[right])
        )
        * key_compatible(left, right)[0],
        expected_exact_key=lambda index: row_exact_key(
            index, source_side=True
        ),
        actual_exact_key=lambda index: row_exact_key(
            index, source_side=False
        ),
        expected_block_keys=lambda index: row_blocks(
            index, source_side=True
        ),
        actual_block_keys=lambda index: row_blocks(
            index, source_side=False
        ),
        max_edges=max_edges,
        top_k=top_k,
        exact_dense_limit=exact_dense_limit,
        max_hungarian_work=max_hungarian_work,
        max_sparse_relaxations=max_sparse_relaxations,
    )
    certified = (
        columns.certification.optimality_certified
        and rows.certification.optimality_certified
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
                    if source_column
                    < len(source_rows[edge.expected_index])
                    else "",
                    output_rows[edge.actual_index][output_column]
                    if output_column
                    < len(output_rows[edge.actual_index])
                    else "",
                )
                * gate
            )
    expected_cells = len(source_rows) * max(1, len(aligned))
    actual_cells = len(output_rows) * max(1, len(aligned))
    cell_mass = sum(associated_scores)
    cell_precision = (
        cell_mass / actual_cells if actual_cells else 0.0
    )
    cell_recall = (
        cell_mass / expected_cells if expected_cells else 1.0
    )
    cell_f = f_score(cell_precision, cell_recall, beta)
    order_score: float | None = None
    if source.order_sensitive and source.rows:
        ordered = [
            edge.expected_index
            for edge in sorted(
                rows.matches, key=lambda edge: edge.actual_index
            )
            if edge.score >= 0.20
        ]
        order_score = _lcs_norm(
            [str(value) for value in ordered],
            [str(value) for value in sorted(ordered)],
        )
    parts = [(0.20, column_f), (0.30, row_f), (0.45, cell_f)]
    if order_score is None:
        parts[2] = (0.50, cell_f)
    else:
        parts.append((0.05, order_score))
    score = (
        clamp01(sum(weight * value for weight, value in parts))
        if certified
        else None
    )
    key_precision = (
        key_exact / len(output_rows)
        if source.row_key and output_rows
        else 0.0
    ) if source.row_key else None
    key_recall = (
        key_exact / len(source_rows)
        if source.row_key and source_rows
        else 1.0
    ) if source.row_key else None
    combined = combine_certifications(
        {
            "columns": assignment_to_mapping(columns),
            "rows": assignment_to_mapping(rows),
        }
    )
    return TablePairResultV52(
        score=score,
        column_fbeta=column_f if certified else None,
        row_fbeta=row_f if certified else None,
        associated_cell_fbeta=cell_f if certified else None,
        order_score=order_score if certified else None,
        key_precision=key_precision if certified else None,
        key_recall=key_recall if certified else None,
        key_exact_match_rate=(
            key_exact / max(1, len(rows.matches))
            if source.row_key and certified
            else None
        ),
        certification=combined,
        diagnostic_codes=tuple(dict.fromkeys(diagnostics)),
    )


def _table_type_allowed(
    source: _core.SourceTable, output: _core.OutputTable
) -> bool:
    policy = source.representation_policy.casefold()
    output_type = output.element_type.casefold()
    if policy == "table_only":
        return output_type == "table"
    if policy == "chart_only":
        return output_type == "chart"
    if policy in {"structured_equivalent", "either_table_or_chart"}:
        return output_type in {"table", "chart"}
    return output_type == "table"


def table_fidelity_v5_2(
    expected: Sequence[_core.SourceTable],
    output_tables: Sequence[_core.OutputTable],
    output_charts: Sequence[_core.OutputTable],
    *,
    beta: float,
    exact_dense_limit: int,
    max_edges: int,
    top_k: int,
    max_hungarian_work: int = 50_000_000,
    max_sparse_relaxations: int = 50_000_000,
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
        if not expanded
        or any(_table_type_allowed(source, item) for source in expanded)
    ]
    if not expanded and not actual:
        return None, {
            "required_count": 0,
            "matched_required_count": 0,
            "matching": _empty_matching_mapping(),
        }
    details: dict[tuple[int, int], TablePairResultV52] = {}

    def pair_score(left: int, right: int) -> float:
        if not _table_type_allowed(expanded[left], actual[right]):
            return 0.0
        result = table_pair_metrics_v5_2(
            expanded[left],
            actual[right],
            beta=beta,
            exact_dense_limit=exact_dense_limit,
            max_edges=max_edges,
            top_k=top_k,
            max_hungarian_work=max_hungarian_work,
            max_sparse_relaxations=max_sparse_relaxations,
        )
        details[(left, right)] = result
        return clamp01(result.score)

    assignment = certified_assignment(
        len(expanded),
        len(actual),
        score=pair_score,
        cheap_score=lambda left, right: float(
            _table_type_allowed(expanded[left], actual[right])
        ),
        expected_exact_key=lambda index: None,
        actual_exact_key=lambda index: None,
        expected_block_keys=lambda index: (
            ("table_policy", expanded[index].representation_policy),
            *[("header", _normalized(value)) for value in expanded[index].headers],
        ),
        actual_block_keys=lambda index: (
            ("table_policy", actual[index].element_type.casefold()),
            *[("header", _normalized(value)) for value in actual[index].headers],
        ),
        max_edges=max_edges,
        top_k=top_k,
        exact_dense_limit=exact_dense_limit,
        max_hungarian_work=max_hungarian_work,
        max_sparse_relaxations=max_sparse_relaxations,
    )
    nested_certified = all(
        details.get((edge.expected_index, edge.actual_index)) is not None
        and bool(
            details[(edge.expected_index, edge.actual_index)]
            .certification.get("optimality_certified")
        )
        for edge in assignment.matches
        if edge.score > 0.0
    )
    certified = (
        assignment.certification.optimality_certified
        and nested_certified
    )
    mass = sum(edge.score for edge in assignment.matches)
    precision = mass / len(actual) if actual else 0.0
    recall = mass / len(expanded) if expanded else 1.0
    matched = sum(edge.score >= 0.60 for edge in assignment.matches)
    matching = assignment_to_mapping(assignment)
    matching["nested_optimality_certified"] = nested_certified
    matching["certification"]["optimality_certified"] = certified
    return (
        f_score(precision, recall, beta) if certified else None
    ), {
        "required_count": len(expanded),
        "matched_required_count": matched,
        "precision": precision,
        "recall": recall,
        "matching": matching,
        "matches": [
            {
                "expected_index": edge.expected_index,
                "output_index": edge.actual_index,
                "output_type": actual[edge.actual_index].element_type,
                "score": edge.score,
                "column_fbeta": details[
                    (edge.expected_index, edge.actual_index)
                ].column_fbeta,
                "row_fbeta": details[
                    (edge.expected_index, edge.actual_index)
                ].row_fbeta,
                "associated_cell_fbeta": details[
                    (edge.expected_index, edge.actual_index)
                ].associated_cell_fbeta,
                "order_score": details[
                    (edge.expected_index, edge.actual_index)
                ].order_score,
                "key_exact_match_rate": details[
                    (edge.expected_index, edge.actual_index)
                ].key_exact_match_rate,
                "certification": details[
                    (edge.expected_index, edge.actual_index)
                ].certification,
            }
            for edge in assignment.matches
            if (edge.expected_index, edge.actual_index) in details
        ],
    }


def _canonical_alias(
    value: str, aliases: Mapping[str, set[str]]
) -> str:
    normalized = _core.normalize_url(value)
    equivalents = {normalized}
    for key, values in aliases.items():
        group = {
            _core.normalize_url(key),
            *(_core.normalize_url(item) for item in values),
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


def action_match_score_v5_2(
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


def _expanded_actions(
    expected: Sequence[_core.ActionRef],
) -> list[_core.ActionRef]:
    result: list[_core.ActionRef] = []
    for item in expected:
        if item.required and item.minimum_count > 0:
            result.extend([item] * max(1, item.minimum_count))
    return result


def action_fidelity_v5_2(
    expected: Sequence[_core.ActionRef],
    actual: Sequence[_core.OutputAction],
    *,
    aliases: Mapping[str, set[str]],
    exact_dense_limit: int,
    max_edges: int,
    top_k: int,
    max_hungarian_work: int = 50_000_000,
    max_sparse_relaxations: int = 50_000_000,
) -> tuple[float | None, dict[str, Any]]:
    expanded = _expanded_actions(expected)
    if not expanded and not actual:
        return None, {
            "required_count": 0,
            "matched_required_count": 0,
            "matching": _empty_matching_mapping(),
        }
    explicit_ids = {item.id for item in expanded if item.id}

    def expected_key(index: int) -> Hashable:
        item = expanded[index]
        return (
            item.action_type.casefold(),
            _canonical_alias(item.url, aliases),
            _normalized(item.label),
            item.id,
        )

    def actual_key(index: int) -> Hashable:
        item = actual[index]
        return (
            item.action_type.casefold(),
            _canonical_alias(item.url, aliases),
            _normalized(item.label),
            item.element_id if item.element_id in explicit_ids else "",
        )

    assignment = certified_assignment(
        len(expanded),
        len(actual),
        score=lambda left, right: action_match_score_v5_2(
            expanded[left], actual[right], aliases
        ),
        cheap_score=lambda left, right: (
            float(
                expanded[left].action_type.casefold()
                == actual[right].action_type.casefold()
            )
            * (
                0.5
                + 0.5
                * _core.text_similarity(
                    expanded[left].label, actual[right].label
                )
            )
        ),
        expected_exact_key=expected_key,
        actual_exact_key=actual_key,
        expected_block_keys=lambda index: (
            ("action_type", expanded[index].action_type.casefold()),
            *_url_block_keys(expanded[index].url),
            *[
                ("label_token", token)
                for token in _core.tokenize(expanded[index].label)
            ],
        ),
        actual_block_keys=lambda index: (
            ("action_type", actual[index].action_type.casefold()),
            *_url_block_keys(actual[index].url),
            *[
                ("label_token", token)
                for token in _core.tokenize(actual[index].label)
            ],
        ),
        max_edges=max_edges,
        top_k=top_k,
        exact_dense_limit=exact_dense_limit,
        max_hungarian_work=max_hungarian_work,
        max_sparse_relaxations=max_sparse_relaxations,
    )
    mass = sum(edge.score for edge in assignment.matches)
    precision = mass / len(actual) if actual else 0.0
    recall = mass / len(expanded) if expanded else 1.0
    matched = sum(edge.score >= 0.75 for edge in assignment.matches)
    certified = assignment.certification.optimality_certified
    return (
        f_score(precision, recall, 2.0) if certified else None
    ), {
        "required_count": len(expanded),
        "matched_required_count": matched,
        "precision": precision,
        "recall": recall,
        "role_coverage": matched / len(expanded) if expanded else None,
        "matching": assignment_to_mapping(assignment),
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
    url_score = (
        1.0
        if not expected.url
        else float(
            _core._url_equivalent(expected.url, actual.url, aliases)  # type: ignore[attr-defined]
        )
    )
    if url_score <= 0.0:
        return 0.0
    alt = (
        _core.text_similarity(expected.alt, actual.alt)
        if expected.alt
        else 1.0
    )
    return clamp01(0.75 * url_score + 0.25 * alt)


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


def media_fidelity_v5_2(
    expected: Sequence[_core.MediaRef],
    actual: Sequence[_core.MediaRef],
    *,
    aliases: Mapping[str, set[str]],
    exact_dense_limit: int,
    max_edges: int,
    top_k: int,
    max_hungarian_work: int = 50_000_000,
    max_sparse_relaxations: int = 50_000_000,
) -> tuple[float | None, dict[str, Any]]:
    expanded = _expanded_media(expected)
    if not expanded and not actual:
        return None, {
            "required_count": 0,
            "matched_required_count": 0,
            "matching": _empty_matching_mapping(),
        }
    explicit_ids = {item.id for item in expanded if item.id}
    assignment = certified_assignment(
        len(expanded),
        len(actual),
        score=lambda left, right: _media_match_score(
            expanded[left], actual[right], aliases
        ),
        cheap_score=lambda left, right: (
            float(
                expanded[left].kind.casefold()
                == actual[right].kind.casefold()
            )
            * (
                0.75
                + 0.25
                * _core.text_similarity(
                    expanded[left].alt, actual[right].alt
                )
            )
        ),
        expected_exact_key=lambda index: (
            expanded[index].kind.casefold(),
            _canonical_alias(expanded[index].url, aliases),
            expanded[index].id,
        ),
        actual_exact_key=lambda index: (
            actual[index].kind.casefold(),
            _canonical_alias(actual[index].url, aliases),
            actual[index].id if actual[index].id in explicit_ids else "",
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
        max_edges=max_edges,
        top_k=top_k,
        exact_dense_limit=exact_dense_limit,
        max_hungarian_work=max_hungarian_work,
        max_sparse_relaxations=max_sparse_relaxations,
    )
    mass = sum(edge.score for edge in assignment.matches)
    precision = mass / len(actual) if actual else 0.0
    recall = mass / len(expanded) if expanded else 1.0
    matched = sum(edge.score >= 0.75 for edge in assignment.matches)
    certified = assignment.certification.optimality_certified
    return (
        f_score(precision, recall, 2.0) if certified else None
    ), {
        "required_count": len(expanded),
        "matched_required_count": matched,
        "precision": precision,
        "recall": recall,
        "role_coverage": matched / len(expanded) if expanded else None,
        "matching": assignment_to_mapping(assignment),
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
    }
)


def _role_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): nested
        for key, nested in value.items()
        if key not in _ROLE_META_FIELDS
        and nested not in (None, "", [], {})
    }


def _normalized_role_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, nested in sorted(_role_payload(value).items()):
        if isinstance(nested, Sequence) and not isinstance(
            nested, (str, bytes, bytearray)
        ):
            payload[key] = sorted(
                _normalized(item) for item in nested if _normalized(item)
            )
        else:
            payload[key] = _normalized(nested)
    return payload


def semantic_signature_v5_2(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _normalized_role_payload(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def role_instance_similarity_v5_2(
    requirement: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> float:
    fields = _role_payload(requirement)
    if not fields:
        return 0.0
    scores: list[float] = []
    for key, expected in sorted(fields.items()):
        observed = actual.get(key)
        if isinstance(expected, Sequence) and not isinstance(
            expected, (str, bytes, bytearray)
        ):
            expected_values = {
                _normalized(value) for value in expected if _normalized(value)
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


def semantic_role_coverage_v5_2(
    expected: Mapping[str, Sequence[Mapping[str, Any]]],
    actual: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
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
    for role in sorted(set(expected) | set(actual)):
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
            item
            for item in requirements
            if _role_payload(item)
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
                "matching": _empty_matching_mapping(),
            }
            continue
        explicit_ids = {
            str(item.get("id"))
            for item in expanded
            if str(item.get("id") or "")
        }
        assignment = certified_assignment(
            len(expanded),
            len(candidates),
            score=lambda left, right: role_instance_similarity_v5_2(
                expanded[left], candidates[right]
            ),
            cheap_score=lambda left, right: _core.text_similarity(
                str(
                    expanded[left].get("title")
                    or expanded[left].get("content")
                    or expanded[left].get("subject")
                    or ""
                ),
                str(
                    candidates[right].get("title")
                    or candidates[right].get("content")
                    or candidates[right].get("subject")
                    or ""
                ),
            ),
            expected_exact_key=lambda index: (
                str(expanded[index].get("id") or ""),
                semantic_signature_v5_2(expanded[index]),
            ),
            actual_exact_key=lambda index: (
                (
                    str(candidates[index].get("component_id") or "")
                    if str(
                        candidates[index].get("component_id") or ""
                    )
                    in explicit_ids
                    else ""
                ),
                semantic_signature_v5_2(candidates[index]),
            ),
            expected_block_keys=lambda index: (
                ("role", role),
                *[
                    ("role_token", token)
                    for token in _core.tokenize(
                        " ".join(
                            str(value)
                            for value in _role_payload(
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
                            for value in _role_payload(
                                candidates[index]
                            ).values()
                        )
                    )
                ],
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
            signature = semantic_signature_v5_2(
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
            "matches": [
                {
                    "expected_index": edge.expected_index,
                    "output_index": edge.actual_index,
                    "score": edge.score,
                }
                for edge in accepted
            ],
        }
    applicable_counts = [
        float(value) for value in count_values.values() if value is not None
    ]
    applicable_semantics = [
        float(value)
        for value in semantic_values.values()
        if value is not None
    ]
    specificity = (
        total_semantic / total_required if total_required else 1.0
    )
    return (
        (
            sum(applicable_semantics) / len(applicable_semantics)
            if applicable_semantics
            else None
        ),
        (
            sum(applicable_counts) / len(applicable_counts)
            if applicable_counts
            else None
        ),
        count_values,
        semantic_values,
        diagnostics,
        specificity,
    )


def _empty_matching_mapping() -> dict[str, Any]:
    return {
        "expected_count": 0,
        "actual_count": 0,
        "evaluated_edge_count": 0,
        "candidate_edge_count": 0,
        "exact_preallocated_count": 0,
        "fuzzy_residual_count": 0,
        "certification": {
            "vertex_edge_coverage_complete": True,
            "full_cardinality_matching_exists": True,
            "optimality_certified": True,
            "approximate_matching_used": False,
            "candidate_generation_complete": True,
            "lower_bound_weight": 0.0,
            "upper_bound_weight": 0.0,
            "diagnostic_codes": ["empty_assignment_certified"],
        },
    }


def _iter_matching_mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        certification = value.get("certification")
        if isinstance(certification, Mapping):
            yield value
        for nested in value.values():
            yield from _iter_matching_mappings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_matching_mappings(nested)


def combine_certifications(
    domains: Mapping[str, Any],
) -> dict[str, Any]:
    mappings = list(_iter_matching_mappings(domains))
    certifications = [
        value["certification"]
        for value in mappings
        if isinstance(value.get("certification"), Mapping)
    ]
    if not certifications:
        empty = _empty_matching_mapping()["certification"]
        return {**empty, "domains": dict(domains)}
    upper_values = [
        value.get("upper_bound_weight")
        for value in certifications
    ]
    upper = (
        sum(float(value) for value in upper_values)
        if all(value is not None for value in upper_values)
        else None
    )
    return {
        "vertex_edge_coverage_complete": all(
            bool(value.get("vertex_edge_coverage_complete"))
            for value in certifications
        ),
        "full_cardinality_matching_exists": all(
            bool(value.get("full_cardinality_matching_exists"))
            for value in certifications
        ),
        "optimality_certified": all(
            bool(value.get("optimality_certified"))
            for value in certifications
        ),
        "approximate_matching_used": any(
            bool(value.get("approximate_matching_used"))
            for value in certifications
        ),
        "candidate_generation_complete": all(
            bool(value.get("candidate_generation_complete"))
            for value in certifications
        ),
        "lower_bound_weight": sum(
            float(value.get("lower_bound_weight") or 0.0)
            for value in certifications
        ),
        "upper_bound_weight": upper,
        "evaluated_edge_count": sum(
            int(value.get("evaluated_edge_count") or 0)
            for value in mappings
        ),
        "candidate_edge_count": sum(
            int(value.get("candidate_edge_count") or 0)
            for value in mappings
        ),
        "exact_preallocated_count": sum(
            int(value.get("exact_preallocated_count") or 0)
            for value in mappings
        ),
        "fuzzy_residual_count": sum(
            int(value.get("fuzzy_residual_count") or 0)
            for value in mappings
        ),
        "diagnostic_codes": list(
            dict.fromkeys(
                str(code)
                for value in certifications
                for code in value.get("diagnostic_codes") or []
            )
        ),
        "domains": dict(domains),
    }


__all__ = [
    "SCORING_POLICY_VERSION",
    "TablePairResultV52",
    "action_fidelity_v5_2",
    "action_match_score_v5_2",
    "clamp01",
    "combine_certifications",
    "content_fidelity_v5_2",
    "f_score",
    "media_fidelity_v5_2",
    "prepared_block_similarity",
    "role_instance_similarity_v5_2",
    "semantic_role_coverage_v5_2",
    "semantic_signature_v5_2",
    "table_fidelity_v5_2",
    "table_pair_metrics_v5_2",
]
