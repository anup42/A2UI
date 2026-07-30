from __future__ import annotations

from pathlib import Path
import sys

import pytest


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))

from pipeline.genui_quality import _core  # noqa: E402
from pipeline.genui_quality.matching_v5_2 import (  # noqa: E402
    MatchEdge,
    maximum_weight_assignment,
)
from pipeline.genui_quality.metrics_v5_2 import (  # noqa: E402
    action_fidelity_v5_2,
    content_fidelity_v5_2,
    media_fidelity_v5_2,
    semantic_role_coverage_v5_2,
    table_pair_metrics_v5_2,
)


MATCHING_ARGS = {
    "exact_dense_limit": 64,
    "max_edges": 65536,
    "top_k": 16,
}


@pytest.mark.parametrize("count", [64, 65, 100, 256, 257, 300, 1000])
def test_exact_content_full_set_is_one_and_certified(count: int) -> None:
    values = [
        f"Exact content unit {index} value {index * 17}"
        for index in range(count)
    ]
    atomics, diagnostics = content_fidelity_v5_2(
        values, list(reversed(values)), **MATCHING_ARGS
    )
    certification = diagnostics["matching"]["certification"]
    assert atomics["content_unit_fidelity"] == 1.0
    assert diagnostics["matching"]["exact_preallocated_count"] == count
    assert certification["optimality_certified"]
    assert "complete" not in diagnostics["matching"]


@pytest.mark.parametrize("count", [64, 65, 100, 256, 257, 300, 1000])
def test_exact_actions_and_media_do_not_collapse(count: int) -> None:
    expected_actions = [
        _core.ActionRef(
            label=f"Open item {index}",
            url=f"https://example.com/{index}",
        )
        for index in range(count)
    ]
    actual_actions = [
        _core.OutputAction(
            element_id=f"button_{index}",
            label=f"Open item {index}",
            url=f"https://example.com/{index}",
            source="on.press",
        )
        for index in range(count)
    ]
    action_value, action_diagnostics = action_fidelity_v5_2(
        expected_actions, actual_actions, aliases={}, **MATCHING_ARGS
    )
    assert action_value == 1.0
    assert action_diagnostics["matched_required_count"] == count
    assert action_diagnostics["matching"]["certification"][
        "optimality_certified"
    ]

    media = [
        _core.MediaRef(
            kind="Image",
            url=f"https://example.com/{index}.png",
            alt=f"Image {index}",
        )
        for index in range(count)
    ]
    media_value, media_diagnostics = media_fidelity_v5_2(
        media, list(reversed(media)), aliases={}, **MATCHING_ARGS
    )
    assert media_value == 1.0
    assert media_diagnostics["matched_required_count"] == count
    assert media_diagnostics["matching"]["certification"][
        "optimality_certified"
    ]


@pytest.mark.parametrize("count", [64, 65, 100, 256, 257, 300, 1000])
def test_exact_table_rows_preserve_associations(count: int) -> None:
    source = _core.SourceTable(
        headers=("ID", "Value"),
        rows=tuple(
            (f"id-{index}", f"value-{index}")
            for index in range(count)
        ),
        row_key=("ID",),
    )
    output = _core.OutputTable(
        "table",
        "Table",
        ["ID", "Value"],
        ["id", "value"],
        [
            {"id": f"id-{index}", "value": f"value-{index}"}
            for index in reversed(range(count))
        ],
    )
    result = table_pair_metrics_v5_2(
        source, output, beta=2.0, **MATCHING_ARGS
    )
    assert result.score == 1.0
    assert result.row_fbeta == 1.0
    assert result.associated_cell_fbeta == 1.0
    assert result.certification["optimality_certified"]


def test_swapping_values_between_row_keys_lowers_associated_cells() -> None:
    source = _core.SourceTable(
        headers=("ID", "Value"),
        rows=(("a", "red"), ("b", "blue")),
        row_key=("ID",),
    )
    correct = _core.OutputTable(
        "table",
        "Table",
        ["ID", "Value"],
        ["id", "value"],
        [{"id": "a", "value": "red"}, {"id": "b", "value": "blue"}],
    )
    swapped = _core.OutputTable(
        "table",
        "Table",
        ["ID", "Value"],
        ["id", "value"],
        [{"id": "a", "value": "blue"}, {"id": "b", "value": "red"}],
    )
    good = table_pair_metrics_v5_2(
        source, correct, beta=2.0, **MATCHING_ARGS
    )
    bad = table_pair_metrics_v5_2(
        source, swapped, beta=2.0, **MATCHING_ARGS
    )
    assert bad.associated_cell_fbeta < good.associated_cell_fbeta
    assert bad.score < good.score


def test_two_by_two_counterexample_finds_weight_1_65() -> None:
    result = maximum_weight_assignment(
        2,
        2,
        [
            MatchEdge(0, 0, 0.90),
            MatchEdge(0, 1, 0.80),
            MatchEdge(1, 0, 0.85),
            MatchEdge(1, 1, 0.00),
        ],
        candidate_generation_complete=True,
    )
    assert sum(edge.score for edge in result.matches) == pytest.approx(1.65)
    assert result.certification.optimality_certified
    assert not result.certification.approximate_matching_used


def test_sixty_five_counterexample_is_not_greedy() -> None:
    edges = [MatchEdge(index, index, 1.0) for index in range(2, 65)]
    edges.extend(
        [
            MatchEdge(0, 0, 1.0),
            MatchEdge(0, 1, 0.65),
            MatchEdge(1, 0, 1.0),
            MatchEdge(1, 1, 0.0),
        ]
    )
    result = maximum_weight_assignment(
        65,
        65,
        edges,
        candidate_generation_complete=True,
    )
    assert sum(edge.score for edge in result.matches) == pytest.approx(
        64.65
    )
    assert result.certification.optimality_certified


@pytest.mark.parametrize("count", [64, 65, 100, 256, 257, 300, 1000])
def test_exact_semantic_role_instances_are_certified(count: int) -> None:
    requirements = {
        "chart": [
            {
                "id": f"chart_{index}",
                "required": True,
                "title": f"Metric {index}",
            }
            for index in range(count)
        ]
    }
    actual = {
        "chart": [
            {
                "component_id": f"output_{index}",
                "title": f"Metric {index}",
            }
            for index in range(count)
        ]
    }
    semantic, count, _, _, diagnostics, _ = (
        semantic_role_coverage_v5_2(
            requirements,
            actual,
            threshold=0.70,
            **MATCHING_ARGS,
        )
    )
    assert semantic == 1.0
    assert count == 1.0
    assert diagnostics["chart"]["matched_required_count"] == len(
        requirements["chart"]
    )
    assert diagnostics["chart"]["matching"]["certification"][
        "optimality_certified"
    ]


@pytest.mark.parametrize("count", [64, 65, 100, 256, 257, 300, 1000])
def test_tie_heavy_duplicate_content_preserves_multiplicity(
    count: int,
) -> None:
    values = [f"Repeated bucket {index % 7}" for index in range(count)]
    atomics, diagnostics = content_fidelity_v5_2(
        values, list(reversed(values)), **MATCHING_ARGS
    )
    assert atomics["content_unit_fidelity"] == 1.0
    assert diagnostics["matching"]["exact_preallocated_count"] == count
    assert diagnostics["matching"]["certification"][
        "optimality_certified"
    ]


def test_missing_extra_and_one_wrong_item_have_directional_fidelity() -> None:
    expected = [f"Exact unit {index}" for index in range(1000)]
    exact, _ = content_fidelity_v5_2(
        expected, list(expected), **MATCHING_ARGS
    )
    one_wrong, _ = content_fidelity_v5_2(
        expected,
        [*expected[:-1], "Completely unrelated replacement"],
        **MATCHING_ARGS,
    )
    missing, _ = content_fidelity_v5_2(
        expected, expected[:-1], **MATCHING_ARGS
    )
    extra, _ = content_fidelity_v5_2(
        expected, [*expected, "Unrelated extra output"], **MATCHING_ARGS
    )
    assert exact["content_unit_fidelity"] == 1.0
    assert one_wrong["content_unit_fidelity"] < 1.0
    assert missing["content_unit_fidelity"] < 1.0
    assert extra["content_unit_fidelity"] < 1.0
