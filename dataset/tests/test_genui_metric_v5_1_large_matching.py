from __future__ import annotations

from pathlib import Path
import sys

import pytest


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))

from pipeline.genui_quality import _core  # noqa: E402
from pipeline.genui_quality.metrics_v5_1 import (  # noqa: E402
    action_fidelity_v5_1,
    content_fidelity_v5_1,
    media_fidelity_v5_1,
    table_pair_metrics_v5_1,
)


@pytest.mark.parametrize("count", [64, 65, 100])
def test_exact_large_content_sets_are_not_truncated(count: int) -> None:
    values = [f"Unit {index} exact value {index * 13}" for index in range(count)]
    atomics, diagnostics = content_fidelity_v5_1(
        values,
        values,
        exact_dense_limit=64,
        max_edges=65536,
        top_k=16,
    )
    assert atomics["content_unit_fidelity"] == 1.0
    assert atomics["output_block_precision"] == 1.0
    assert diagnostics["matching"]["complete"]
    assert not diagnostics["assignment_truncated"]


def test_exact_100_row_table_preserves_row_and_cell_fidelity() -> None:
    rows = tuple((f"id-{index}", f"value-{index}") for index in range(100))
    source = _core.SourceTable(
        headers=("ID", "Value"), rows=rows, row_key=("ID",)
    )
    output = _core.OutputTable(
        "table",
        "Table",
        ["ID", "Value"],
        ["id", "value"],
        [
            {"id": f"id-{index}", "value": f"value-{index}"}
            for index in range(100)
        ],
    )
    result = table_pair_metrics_v5_1(
        source,
        output,
        beta=2.0,
        exact_dense_limit=64,
        max_edges=65536,
        top_k=16,
    )
    assert result.complete
    assert result.row_fbeta == 1.0
    assert result.associated_cell_fbeta == 1.0
    assert result.key_exact_match_rate == 1.0
    assert result.score == 1.0


def test_exact_100_actions_have_full_one_to_one_coverage() -> None:
    expected = [
        _core.ActionRef(
            label=f"Open item {index}",
            url=f"https://example.com/{index}",
            id=f"action-{index}",
        )
        for index in range(100)
    ]
    actual = [
        _core.OutputAction(
            element_id=f"button-{index}",
            label=f"Open item {index}",
            url=f"https://example.com/{index}",
            source="on.press",
            action_type="openUrl",
        )
        for index in range(100)
    ]
    value, diagnostics = action_fidelity_v5_1(
        expected,
        actual,
        aliases={},
        exact_dense_limit=64,
        max_edges=65536,
        top_k=16,
    )
    assert value == 1.0
    assert diagnostics["required_count"] == 100
    assert diagnostics["matched_required_count"] == 100
    assert diagnostics["role_coverage"] == 1.0
    assert diagnostics["matching_complete"]


def test_exact_100_media_requirements_match_one_to_one() -> None:
    expected = [
        _core.MediaRef(
            kind="Image",
            url=f"https://example.com/{index}.png",
            alt=f"Image {index}",
            id=f"media-{index}",
        )
        for index in range(100)
    ]
    value, diagnostics = media_fidelity_v5_1(
        expected,
        list(expected),
        aliases={},
        exact_dense_limit=64,
        max_edges=65536,
        top_k=16,
    )
    assert value == 1.0
    assert diagnostics["required_count"] == 100
    assert diagnostics["matched_required_count"] == 100
    assert diagnostics["matching_complete"]


def test_budget_exhaustion_is_explicit_not_false_low_fidelity() -> None:
    values = [f"Unit {index}" for index in range(100)]
    atomics, diagnostics = content_fidelity_v5_1(
        values,
        values,
        exact_dense_limit=64,
        max_edges=50,
        top_k=4,
    )
    assert atomics["content_unit_fidelity"] is None
    assert not diagnostics["matching"]["complete"]
    assert "matching_budget_exceeded" in diagnostics["matching"][
        "diagnostic_codes"
    ]


def test_no_v5_1_matcher_slices_by_assignment_limit() -> None:
    source = (
        DATASET_ROOT / "src" / "pipeline" / "genui_quality" / "metrics_v5_1.py"
    ).read_text(encoding="utf-8")
    assert "[:max_assignment_size]" not in source
    assert "[:exact_dense_limit]" not in source
