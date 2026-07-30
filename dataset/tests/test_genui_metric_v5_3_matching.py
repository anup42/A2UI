from __future__ import annotations

from pathlib import Path
import sys

import pytest

DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))

from pipeline.genui_quality.matching_v5_3 import (  # noqa: E402
    certified_assignment_v5_3,
    group_exact_indices,
)
from pipeline.genui_quality import _core  # noqa: E402
from pipeline.genui_quality.metrics_v5_3 import (  # noqa: E402
    action_exact_key_expected_v5_3,
    action_fidelity_v5_3,
    media_exact_key_expected_v5_3,
    media_fidelity_v5_3,
    role_exact_key_expected_v5_3,
    semantic_role_coverage_v5_3,
)


def _assignment(expected: list[str], actual: list[str]):
    prepared = group_exact_indices(expected)
    return certified_assignment_v5_3(
        len(expected),
        len(actual),
        score=lambda left, right: float(expected[left] == actual[right]),
        cheap_score=lambda left, right: float(
            expected[left] == actual[right]
        ),
        expected_exact_key=lambda index: expected[index],
        actual_exact_key=lambda index: actual[index],
        expected_block_keys=lambda index: (expected[index],),
        actual_block_keys=lambda index: (actual[index],),
        expected_exact_index=prepared,
        max_edges=65_536,
        top_k=16,
    )


@pytest.mark.parametrize("count", [257, 300, 1000])
def test_large_perfect_sets_preallocate_and_certify(count: int) -> None:
    values = [f"semantic-{index}" for index in range(count)]
    result = _assignment(values, list(reversed(values)))
    assert result.certification.optimality_certified
    assert result.exact_preallocated_count == count
    assert len(result.matches) == count


@pytest.mark.parametrize("count", [257, 300, 1000])
def test_one_error_is_proportional_not_a_cliff(count: int) -> None:
    expected = [f"semantic-{index}" for index in range(count)]
    actual = list(expected)
    actual[-1] = "wrong"
    result = _assignment(expected, actual)
    assert result.certification.optimality_certified
    assert sum(item.score for item in result.matches) == pytest.approx(
        count - 1
    )
    assert sum(item.score for item in result.matches) / count > 0.99


def test_generic_requirement_ids_do_not_block_action_preallocation() -> None:
    expected = [
        _core.ActionRef(
            label=f"Open {index}",
            url=f"https://example.com/{index}",
            id=f"requirement-{index}",
        )
        for index in range(300)
    ]
    actual = [
        _core.OutputAction(
            element_id=f"button-{index}",
            label=f"Open {index}",
            url=f"https://example.com/{index}",
            source="on.press",
        )
        for index in range(300)
    ]
    index = group_exact_indices(
        [
            action_exact_key_expected_v5_3(item, {})
            for item in expected
        ]
    )
    value, diagnostics = action_fidelity_v5_3(
        expected,
        actual,
        aliases={},
        expected_exact_index=index,
        exact_dense_limit=64,
        max_edges=65_536,
        top_k=16,
    )
    assert value == 1.0
    assert diagnostics["matching"]["exact_preallocated_count"] == 300
    assert diagnostics["matching"]["certification"][
        "optimality_certified"
    ]


@pytest.mark.parametrize("count", [257, 300, 1000])
@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_large_one_item_cardinality_mutations_remain_certified(
    count: int,
    mutation: str,
) -> None:
    expected = [f"semantic-{index}" for index in range(count)]
    actual = list(expected)
    if mutation == "missing":
        actual.pop()
    else:
        actual.append("unrelated-extra")
    result = _assignment(expected, actual)
    assert result.certification.optimality_certified
    assert sum(edge.score for edge in result.matches) == pytest.approx(
        count - 1 if mutation == "missing" else count
    )


@pytest.mark.parametrize("count", [257, 300, 1000])
def test_large_duplicate_multiplicity_remains_certified(count: int) -> None:
    expected = ["duplicate", "duplicate"] + [
        f"semantic-{index}" for index in range(count - 2)
    ]
    actual = list(expected)
    actual[1] = "unrelated"
    result = _assignment(expected, actual)
    assert result.certification.optimality_certified
    assert sum(edge.score for edge in result.matches) == pytest.approx(
        count - 1
    )


def _long_action_result(count: int):
    expected = [
        _core.ActionRef(
            label=f"Open {index}",
            url=f"https://example.com/{index}",
            id=f"requirement-{index}",
        )
        for index in range(count)
    ]
    actual = [
        _core.OutputAction(
            element_id=f"button-{index}",
            label=f"Open {index}",
            url=f"https://example.com/{index}",
            source="on.press",
        )
        for index in range(count)
    ]
    prepared = group_exact_indices(
        [
            action_exact_key_expected_v5_3(item, {})
            for item in expected
        ]
    )
    return action_fidelity_v5_3(
        expected,
        actual,
        aliases={},
        expected_exact_index=prepared,
        exact_dense_limit=64,
        max_edges=65_536,
        top_k=16,
    )


def _long_media_result(count: int):
    expected = [
        _core.MediaRef(
            kind="image",
            url=f"https://example.com/{index}.png",
            alt=f"Image {index}",
            id=f"requirement-{index}",
        )
        for index in range(count)
    ]
    actual = [
        _core.MediaRef(
            kind="image",
            url=f"https://example.com/{index}.png",
            alt=f"Image {index}",
            component_id=f"image-{index}",
        )
        for index in range(count)
    ]
    prepared = group_exact_indices(
        [
            media_exact_key_expected_v5_3(item, {})
            for item in expected
        ]
    )
    return media_fidelity_v5_3(
        expected,
        actual,
        aliases={},
        expected_exact_index=prepared,
        exact_dense_limit=64,
        max_edges=65_536,
        top_k=16,
    )


def _long_role_result(count: int):
    expected = {
        "chart": [
            {
                "id": f"requirement-{index}",
                "required": True,
                "minimum_count": 1,
                "chart_type": "bar",
                "title": f"Revenue {index}",
            }
            for index in range(count)
        ]
    }
    actual = {
        "chart": [
            {
                "component_id": f"chart-{index}",
                "chart_type": "bar",
                "title": f"Revenue {index}",
            }
            for index in range(count)
        ]
    }
    prepared = {
        "chart": group_exact_indices(
            [
                role_exact_key_expected_v5_3("chart", item)
                for item in expected["chart"]
            ]
        )
    }
    return semantic_role_coverage_v5_3(
        expected,
        actual,
        expected_exact_indices=prepared,
        threshold=0.55,
        exact_dense_limit=64,
        max_edges=65_536,
        top_k=16,
    )


@pytest.mark.parametrize("count", [257, 300, 1000])
@pytest.mark.parametrize("taxonomy", ["action", "media", "role"])
def test_long_production_taxonomies_ignore_generic_requirement_ids(
    count: int,
    taxonomy: str,
) -> None:
    if taxonomy == "action":
        value, diagnostics = _long_action_result(count)
        matching = diagnostics["matching"]
    elif taxonomy == "media":
        value, diagnostics = _long_media_result(count)
        matching = diagnostics["matching"]
    else:
        _, value, _, _, diagnostics, _ = _long_role_result(count)
        matching = diagnostics["chart"]["matching"]
    assert value == 1.0
    assert matching["exact_preallocated_count"] == count
    assert matching["certification"]["optimality_certified"]
