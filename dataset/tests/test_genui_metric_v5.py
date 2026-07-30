from __future__ import annotations

import copy
import math
from pathlib import Path
import sys


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from genui_metric_v5_support import contract, score, simple_spec  # noqa: E402
from pipeline.flat_spec_contract import validate_flat_spec  # noqa: E402
from pipeline.genui_quality import score_genui_completion_v5_1  # noqa: E402


def test_public_output_schema_and_bounds() -> None:
    result = score(simple_spec(), "Hello world")
    assert result.metric_name == "GenUI Representation Quality"
    assert result.metric_version == "5.1.0"
    assert len(result.metric_fingerprint) == 64
    assert 0.0 <= result.quality_0_1 <= 1.0
    assert result.quality_0_100 == 100.0 * result.quality_0_1
    assert result.reward == 2.0 * result.quality_0_1 - 1.0
    assert math.isfinite(result.base_quality_before_caps)
    assert set(result.identity) == {
        "source_hash",
        "raw_candidate_hash",
        "canonical_candidate_hash",
        "expected_contract_hash",
    }


def test_unsupported_type_among_100_valid_nodes_is_non_dilutable() -> None:
    candidate = simple_spec()
    for index in range(100):
        candidate["elements"][f"valid_{index}"] = {
            "type": "Text",
            "props": {"text": "padding"},
            "children": [],
        }
    candidate["elements"]["invalid"] = {
        "type": "ProductionUnsupported",
        "props": {},
        "children": [],
    }
    candidate["elements"]["root"]["children"].append("invalid")
    result = score(candidate, "Hello world")
    assert not result.normalization["production_valid"]
    assert result.quality_0_1 <= 0.25
    assert any(cap["name"] == "production_invalid" for cap in result.active_caps)


def test_forbidden_top_level_property_activates_strict_policy() -> None:
    candidate = simple_spec()
    candidate["debug"] = True
    result = score(candidate, "Hello world")
    assert result.normalization["production_valid"]
    assert not result.normalization["strict_schema_valid"]
    assert result.normalization["unknown_top_level_properties"] == ["debug"]
    assert any(cap["name"] == "strict_format" for cap in result.active_caps)


def test_tabs_and_modal_nested_references_are_validated_and_reachable() -> None:
    candidate = {
        "root": "tabs",
        "state": {},
        "elements": {
            "tabs": {
                "type": "Tabs",
                "props": {
                    "tabs": [
                        {"title": "Overview", "child": "overview"},
                        {"title": "Details", "content": "modal"},
                    ]
                },
                "children": [],
            },
            "overview": {
                "type": "Text",
                "props": {"text": "Overview"},
                "children": [],
            },
            "modal": {
                "type": "Modal",
                "props": {"trigger": "open", "content": "details"},
                "children": [],
            },
            "open": {
                "type": "Button",
                "props": {"label": "Open"},
                "children": [],
            },
            "details": {
                "type": "Text",
                "props": {"text": "Details"},
                "children": [],
            },
        },
    }
    assert validate_flat_spec(candidate).is_valid
    result = score(candidate, "Overview\nDetails")
    assert set(result.evidence["output"]["reachable_ids"]) == set(candidate["elements"])

    missing = copy.deepcopy(candidate)
    missing["elements"]["modal"]["props"]["content"] = "absent"
    validation = validate_flat_spec(missing)
    assert not validation.is_valid
    assert "props.content" in str(validation.error)


def test_nested_reference_cycle_activates_cap() -> None:
    candidate = {
        "root": "tabs",
        "state": {},
        "elements": {
            "tabs": {
                "type": "Tabs",
                "props": {"tabs": [{"child": "panel"}]},
                "children": [],
            },
            "panel": {
                "type": "Stack",
                "props": {},
                "children": ["tabs"],
            },
        },
    }
    result = score(candidate, "Panel")
    assert result.quality_0_1 <= 0.25
    assert result.evidence["output"]["cycles"]


def test_table_row_association_mutation_lowers_score() -> None:
    source = "Item | Price\n--- | ---\nA | 10\nB | 20"
    expected = contract(source)
    candidate = {
        "root": "root",
        "state": {
            "rows": [
                {"item": "A", "price": "10"},
                {"item": "B", "price": "20"},
            ]
        },
        "elements": {
            "root": {"type": "Stack", "props": {}, "children": ["table"]},
            "table": {
                "type": "Table",
                "props": {
                    "columns": [
                        {"key": "item", "label": "Item"},
                        {"key": "price", "label": "Price"},
                    ],
                    "statePath": "/rows",
                },
                "children": [],
            },
        },
    }
    original = score(candidate, source, expected=expected)
    swapped = copy.deepcopy(candidate)
    swapped["state"]["rows"][0]["price"], swapped["state"]["rows"][1]["price"] = (
        swapped["state"]["rows"][1]["price"],
        swapped["state"]["rows"][0]["price"],
    )
    changed = score(swapped, source, expected=expected)
    assert (
        changed.atomics["fidelity"]["markdown_table_fidelity"]
        < original.atomics["fidelity"]["markdown_table_fidelity"]
    )
    assert changed.quality_0_1 < original.quality_0_1


def test_optional_requirements_do_not_activate_caps() -> None:
    source = "Optional information"
    expected = contract(
        source,
        actions=[
            {
                "id": "optional-action",
                "label": "Open",
                "target": "https://example.com",
                "required": False,
            }
        ],
        tables=[
            {
                "id": "optional-table",
                "headers": ["A"],
                "rows": [["1"]],
                "required": False,
            }
        ],
        media=[
            {
                "id": "optional-media",
                "kind": "Image",
                "url": "https://example.com/a.png",
                "required": False,
                "media_policy": "exact",
            }
        ],
        required_roles={"action": 0, "table": 0, "image": 0},
        expected_role_counts={"action": 0, "table": 0, "image": 0},
    )
    result = score(simple_spec(source), source, expected=expected)
    semantic_cap_names = {
        "missing_action",
        "missing_table",
        "missing_media",
        "partial_action",
        "partial_table",
        "partial_media",
    }
    assert not semantic_cap_names.intersection(
        cap["name"] for cap in result.active_caps
    )


def test_wrong_action_target_cannot_cover_required_action() -> None:
    source = "Action: [Button: Open account] https://example.com/open"
    expected = contract(source)
    candidate = simple_spec("Open account")
    candidate["elements"]["button"] = {
        "type": "Button",
        "props": {"label": "Open account"},
        "children": [],
        "on": {
            "press": {
                "action": "openUrl",
                "params": {"url": "https://wrong.invalid/"},
            }
        },
    }
    candidate["elements"]["root"]["children"].append("button")
    result = score(candidate, source, expected=expected)
    assert result.evidence["action_matching"]["matched_required_count"] == 0
    assert result.evidence["role_values"]["action"] == 0.0
    assert any(cap["name"] == "missing_action" for cap in result.active_caps)


def test_duplicate_charts_do_not_cover_distinct_chart_instances() -> None:
    source = "Quarterly charts"
    expected = contract(
        source,
        required_roles={"chart": 2},
        expected_role_counts={"chart": 2},
    )
    candidate = simple_spec(source)
    candidate["state"]["series"] = [{"quarter": "Q1", "value": 10}]
    for index in range(3):
        candidate["elements"][f"chart_{index}"] = {
            "type": "Chart",
            "props": {
                "title": "Same chart",
                "columns": [
                    {"key": "quarter", "label": "Quarter"},
                    {"key": "value", "label": "Value"},
                ],
                "statePath": "/series",
                "xKey": "quarter",
                "yKey": "value",
            },
            "children": [],
        }
        candidate["elements"]["root"]["children"].append(f"chart_{index}")
    result = score(candidate, source, expected=expected)
    assert result.evidence["role_values"]["chart"] == 0.5


def test_one_image_cannot_cover_two_required_media_instances() -> None:
    source = (
        "Media: Image = https://example.com/a.png Alt = A\n"
        "Media: Image = https://example.com/b.png Alt = B"
    )
    expected = contract(source)
    candidate = simple_spec("A and B")
    candidate["elements"]["image"] = {
        "type": "Image",
        "props": {"url": "https://example.com/a.png", "alt": "A"},
        "children": [],
    }
    candidate["elements"]["root"]["children"].append("image")
    result = score(candidate, source, expected=expected)
    assert result.evidence["media_matching"]["required_count"] == 2
    assert result.evidence["media_matching"]["matched_required_count"] == 1
    assert result.evidence["role_values"]["image"] == 0.5


def test_malformed_results_are_finite_and_bounded() -> None:
    for completion in ("", '{"root":', "42", None):
        result = score_genui_completion_v5_1(completion, "Hello")
        assert result.quality_0_1 == 0.0
        assert result.reward == -1.0
        assert all(
            math.isfinite(value)
            for value in (result.quality_0_1, result.quality_0_100, result.reward)
        )
