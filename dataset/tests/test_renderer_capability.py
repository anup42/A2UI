"""Guards DoD #7: renderer coverage is a reported number, not an impression."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline import flat_spec_contract  # noqa: E402
from pipeline.renderer_capability import (  # noqa: E402
    capability_manifest,
    dropped_prop_histogram,
    renderer_coverage,
    renderer_program_coverage,
)


def _spec(elements: dict) -> dict:
    return {"root": "main", "state": {}, "elements": elements}


def test_manifest_aliases_cover_the_python_contract_allow_list() -> None:
    """The manifest must not claim support the contract does not permit.

    Catches a type being added to the renderer manifest without being added to
    the contract, and vice versa.
    """
    manifest_aliases = {
        alias
        for entry in capability_manifest()["types"]
        for alias in entry["aliases"]
    }
    manifest_aliases.update(capability_manifest()["compatibility_type_aliases"])
    contract_types = {
        name.lower()
        for name in flat_spec_contract._ALLOWED_TYPES  # noqa: SLF001
    }
    missing_from_manifest = contract_types - manifest_aliases
    assert not missing_from_manifest, (
        "Contract allows types the renderer capability manifest does not list: "
        f"{sorted(missing_from_manifest)}"
    )


def test_manifest_actions_match_the_contract() -> None:
    manifest_actions = {
        str(action["name"]).lower()
        for action in capability_manifest()["actions"]
    }
    contract_actions = {
        name.lower()
        for name in flat_spec_contract._ALLOWED_ACTIONS  # noqa: SLF001
    }
    assert manifest_actions == contract_actions


def test_manifest_matches_json_schema_types_and_actions() -> None:
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "schema" / "genui_flatspec.schema.json")
        .read_text(encoding="utf-8")
    )
    manifest = capability_manifest()
    schema_types = set(schema["$defs"]["element"]["properties"]["type"]["enum"])
    manifest_types = {str(entry["canonical"]) for entry in manifest["types"]}
    assert schema_types == manifest_types

    schema_actions = set(schema["$defs"]["action"]["properties"]["action"]["enum"])
    manifest_actions = {str(entry["name"]) for entry in manifest["actions"]}
    assert schema_actions == manifest_actions


def test_no_action_is_a_stub() -> None:
    """`validateForm` now evaluates real rules, so nothing should be stubbed.

    If an action regresses to a hardcoded result, add it to `_STUB_ACTIONS` and
    `action_stub_rate` will surface it per run.
    """
    assert capability_manifest()["stub_actions"] == []

    spec = _spec(
        {
            "main": {"type": "Stack", "props": {}, "children": ["btn"]},
            "btn": {
                "type": "Button",
                "props": {"label": "Check"},
                "children": [],
                "on": {"press": {"action": "validateForm", "params": {}}},
            },
        }
    )
    assert renderer_coverage(spec)["action_stub_rate"] == 0.0


def test_unimplemented_action_is_counted_as_a_stub() -> None:
    """`showMessage`/`showSurface` were documented but never wired up."""
    spec = _spec(
        {
            "main": {"type": "Stack", "props": {}, "children": ["btn"]},
            "btn": {
                "type": "Button",
                "props": {"label": "Go"},
                "children": [],
                "on": {"press": {"action": "showMessage", "params": {}}},
            },
        }
    )
    assert renderer_coverage(spec)["action_stub_rate"] == 1.0


def test_unsupported_type_lowers_support_rate() -> None:
    spec = _spec(
        {
            "main": {"type": "Stack", "props": {}, "children": ["x"]},
            "x": {"type": "Hologram", "props": {}, "children": []},
        }
    )
    metrics = renderer_coverage(spec)
    assert metrics["renderer_support_rate"] == 0.5
    assert metrics["renderer_unsupported_count"] == 1.0


def test_unconsumed_prop_is_counted_and_attributed() -> None:
    spec = _spec(
        {
            "main": {"type": "Stack", "props": {}, "children": ["t"]},
            "t": {
                "type": "Text",
                "props": {"text": "hi", "sparkle": True},
                "children": [],
            },
        }
    )
    assert renderer_coverage(spec)["dropped_prop_count"] == 1.0
    assert dropped_prop_histogram([spec]) == {"Text.sparkle": 1}


def test_audited_table_props_report_unknown_values_as_dropped() -> None:
    spec = _spec(
        {
            "main": {"type": "Stack", "props": {}, "children": ["tbl"]},
            "tbl": {
                "type": "Table",
                "props": {"columns": [], "rows": [], "anythingAtAll": 1},
                "children": [],
            },
        }
    )
    metrics = renderer_coverage(spec)
    assert metrics["dropped_prop_count"] == 1.0
    assert metrics["prop_consumption_rate"] == 2.0 / 3.0


def test_coverage_dimensions_are_independent() -> None:
    spec = _spec(
        {
            "main": {"type": "Stack", "props": {}, "children": ["table", "button"]},
            "table": {
                "type": "Table",
                "props": {"domain": "shopping", "columns": [], "rows": []},
                "children": [],
            },
            "button": {
                "type": "Button",
                "props": {"label": "Open"},
                "children": [],
                "on": {"press": {"action": "openUrl", "params": {"url": "http://unsafe.example"}}},
            },
        }
    )
    metrics = renderer_coverage(spec)
    assert metrics["renderer_structural_type_coverage"] == 1.0
    assert metrics["renderer_domain_route_coverage"] == 1.0
    assert metrics["renderer_action_valid_coverage"] == 0.0


def test_program_coverage_keeps_visual_and_end_to_end_dimensions_separate() -> None:
    record = _spec(
        {"main": {"type": "Text", "props": {"text": "ok"}, "children": []}}
    )
    coverage = renderer_program_coverage(
        [record],
        expected_intents={"a", "b"},
        visual_fixture_intents={"a", "b"},
        end_to_end_passed_intents={"a"},
    )
    assert coverage["renderer_visual_fixture_coverage"] == 1.0
    assert coverage["renderer_end_to_end_intent_coverage"] == 0.5
    assert coverage["renderer_structural_type_coverage"] == 1.0


def test_dangling_reference_lowers_resolution_rate() -> None:
    spec = _spec(
        {"main": {"type": "Stack", "props": {}, "children": ["ghost"]}}
    )
    assert renderer_coverage(spec)["reference_resolution_rate"] == 0.0


def test_non_flat_spec_input_returns_empty_metrics() -> None:
    """Callers merge unconditionally, so bad input must not raise."""
    assert renderer_coverage(None) == {}
    assert renderer_coverage({"not": "a spec"}) == {}
    assert renderer_coverage([1, 2, 3]) == {}
