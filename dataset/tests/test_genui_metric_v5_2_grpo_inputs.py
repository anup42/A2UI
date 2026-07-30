from __future__ import annotations

from pathlib import Path
import sys

import pytest


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from genui_metric_v5_2_support import simple_spec  # noqa: E402
from pipeline.genui_quality import (  # noqa: E402
    genui_grpo_reward,
    score_genui_completion,
)
from pipeline.genui_quality.grpo_reward import (  # noqa: E402
    AssetCollection,
    PerCompletionAssetCollections,
    normalize_asset_rows,
    normalize_contract_rows,
    normalize_scalar_or_vector_text,
)


def test_scalar_and_vector_source_shapes_are_explicit() -> None:
    assert normalize_scalar_or_vector_text(
        "source", 3, field="response_text"
    ) == ["source"] * 3
    assert normalize_scalar_or_vector_text(
        ["a", "b"], 2, field="response_text"
    ) == ["a", "b"]
    with pytest.raises(ValueError, match="response_text"):
        normalize_scalar_or_vector_text(
            ["only"], 2, field="response_text"
        )


def test_one_collection_of_eight_assets_broadcasts_to_eight() -> None:
    assets = [
        {"id": f"asset_{index}", "url": f"https://example.com/{index}"}
        for index in range(8)
    ]
    rows = normalize_asset_rows(assets, 8)
    assert len(rows) == 8
    assert all(len(row or []) == 8 for row in rows)
    assert all((row or [])[5]["id"] == "asset_5" for row in rows)


def test_explicit_per_completion_asset_collections_remain_distinct() -> None:
    nested = [
        [{"id": f"asset_{index}", "url": f"https://example.com/{index}"}]
        for index in range(8)
    ]
    rows = normalize_asset_rows(nested, 8)
    assert [(row or [])[0]["id"] for row in rows] == [
        f"asset_{index}" for index in range(8)
    ]

    wrapped = PerCompletionAssetCollections(
        tuple(
            AssetCollection(({"id": f"wrapped_{index}"},))
            for index in range(8)
        )
    )
    assert [(row or [])[0]["id"] for row in normalize_asset_rows(wrapped, 8)] == [
        f"wrapped_{index}" for index in range(8)
    ]


def test_malformed_mixed_asset_shape_fails_before_scoring() -> None:
    with pytest.raises(ValueError, match="bare mapping"):
        normalize_asset_rows([{"id": "bare"}, []], 2)
    with pytest.raises(ValueError, match="per-completion"):
        genui_grpo_reward(
            [simple_spec(), simple_spec()],
            "Hello world",
            assets=[[{"id": "one"}]],
        )


def test_contract_mapping_and_vector_shapes_are_unambiguous() -> None:
    contract = {"contract_version": "test"}
    assert normalize_contract_rows(contract, 2) == [contract, contract]
    vector = [contract, None]
    assert normalize_contract_rows(vector, 2) == vector
    with pytest.raises(ValueError, match="expected_ui_contract"):
        normalize_contract_rows([contract], 2)


def test_group_and_scalar_rewards_are_equal_for_identical_rows() -> None:
    completions = [
        simple_spec("Hello world"),
        simple_spec("Wrong content"),
    ]
    grouped = genui_grpo_reward(completions, "Hello world")
    scalar = [
        score_genui_completion(value, "Hello world").reward
        for value in completions
    ]
    assert grouped == scalar
