from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.flat_spec_contract import coerce_and_validate
from pipeline.flat_spec_semantics import iter_renderer_references
from pipeline.genui_quality.candidate_normalization import (
    RENDERER_V2_CANONICALIZATION,
    normalize_and_validate_candidate,
)
from pipeline.ir_formats import (
    A2UI_EXPRESS_V1,
    A2UI_V1_WIRE,
    FLAT_SPEC_V1,
    decode_to_flat_spec,
    detect_format,
    encode_from_flat_spec,
    semantic_equivalent,
    semantic_hash,
)
from pipeline.metrics import compute_ui_metrics


def reference_spec() -> dict:
    return {
        "root": "layout",
        "state": {"items": [{"id": "one"}], "selectedId": "one"},
        "elements": {
            "layout": {
                "type": "Stack",
                "props": {"direction": "vertical"},
                "children": ["tabs", "modal", "list", "continue"],
            },
            "tabs": {
                "type": "Tabs",
                "props": {
                    "tabs": [
                        {"id": "summary", "child": "summary_panel"},
                        {"id": "details", "content": "details_panel"},
                    ]
                },
                "children": [],
            },
            "summary_panel": {
                "type": "Text",
                "props": {"text": "Summary", "variant": "h2"},
                "children": [],
            },
            "details_panel": {
                "type": "Text",
                "props": {"text": "Details"},
                "children": [],
            },
            "modal": {
                "type": "Modal",
                "props": {"trigger": "open", "content": "dialog", "title": "Review"},
                "children": [],
            },
            "open": {
                "type": "Button",
                "props": {"label": "Open"},
                "children": [],
            },
            "dialog": {
                "type": "Card",
                "props": {},
                "children": ["dialog_text"],
            },
            "dialog_text": {
                "type": "Text",
                "props": {"text": "Dialog"},
                "children": [],
            },
            "list": {
                "type": "List",
                "props": {},
                "children": ["row_template"],
                "repeat": {"statePath": "/items", "key": "id", "template": "row_template"},
            },
            "row_template": {
                "type": "Text",
                "props": {"text": {"path": "/name"}},
                "children": [],
            },
            "continue": {
                "type": "Button",
                "props": {"label": "Continue", "accessibilityLabel": "Continue flow"},
                "children": [],
                "on": {
                    "press": {
                        "action": "emitEvent",
                        "params": {
                            "name": "continue",
                            "context": {"selectedId": "$/selectedId"},
                            "wantResponse": True,
                            "responsePath": "/eventResponse",
                        },
                    }
                },
            },
        },
    }


def test_reference_inventory_fixture() -> None:
    fixture = json.loads(
        (ROOT / "tests" / "fixtures" / "flat_spec_reference_inventory_v1.json").read_text(
            encoding="utf-8"
        )
    )
    for case in fixture["cases"]:
        actual = [
            [item.target_id, item.source_path, item.reference_kind]
            for item in iter_renderer_references(case["element"])
        ]
        assert actual == case["references"], case["name"]


def test_all_formats_round_trip_reference_graph() -> None:
    spec = reference_spec()
    expected = semantic_hash(spec)
    for format_id in (FLAT_SPEC_V1, A2UI_EXPRESS_V1, A2UI_V1_WIRE):
        encoded = encode_from_flat_spec(spec, format_id, shorten_ids=True)
        decoded = decode_to_flat_spec(encoded, format_hint=format_id).flat_spec
        assert semantic_hash(decoded) == expected, format_id
        assert len(decoded["elements"]) == len(spec["elements"])


def test_catalog_covers_every_renderer_component() -> None:
    capabilities = json.loads(
        (ROOT / "schema" / "renderer_capabilities.json").read_text(encoding="utf-8")
    )
    component_types = sorted({entry["canonical"] for entry in capabilities["types"]})
    spec = {
        "root": "root",
        "state": {},
        "elements": {"root": {"type": "Stack", "props": {}, "children": []}},
    }
    for index, component_type in enumerate(component_types):
        if component_type == "Stack":
            continue
        element_id = f"component_{index}"
        spec["elements"]["root"]["children"].append(element_id)
        spec["elements"][element_id] = {
            "type": component_type,
            "props": {},
            "children": [],
        }
    for format_id in (A2UI_EXPRESS_V1, A2UI_V1_WIRE):
        encoded = encode_from_flat_spec(spec, format_id)
        assert semantic_equivalent(spec, encoded), format_id


def test_compact_payload_is_rejected_by_active_codec() -> None:
    with pytest.raises(ValueError, match="migration-only"):
        detect_format({"v": "gci2", "r": "root", "e": {"root": {"t": "Text"}}})


def test_express_advanced_syntax_inline_components_and_event() -> None:
    text = r'''<a2ui>
/* pinned grammar features */
$/={selectedId:"one",items:[{id:"one"}],}
root=Column([tabs,modal,list,action,inline],gap="md",)
tabs=Tabs([{id:"summary",child:"summary_panel"},{id:"details",content:"details_panel"}],"summary")
summary_panel=Text("""Summary
panel""","h2")
details_panel=Text(r"Details\nraw","body")
modal=Modal(open,dialog,"Review")
open=Button("Open")
dialog=Card([dialog_text])
dialog_text=Text("Dialog")
list=List([row_template],repeat={statePath:"/items",key:"id",template:"row_template"})
row_template=Text("Row")
action=Button("Continue",onPress=Event("continue",{selectedId:$/selectedId},true,$/eventResponse))
inline=Card([Text("Inline one"),Text("Inline two"),],)
</a2ui>'''
    decoded = decode_to_flat_spec(text, format_hint=A2UI_EXPRESS_V1).flat_spec
    assert decoded["elements"]["root"]["type"] == "Stack"
    assert decoded["elements"]["root"]["props"]["direction"] == "vertical"
    action = decoded["elements"]["action"]["on"]["press"]
    assert action["action"] == "emitEvent"
    assert action["params"]["name"] == "continue"
    assert action["params"]["context"] == {"selectedId": "$/selectedId"}
    assert len(decoded["elements"]["inline"]["children"]) == 2
    assert len(decoded["elements"]) == 14


def test_wire_message_stream_and_unknown_prop_preservation() -> None:
    catalog_id = "https://genui.samsung.com/a2ui/catalogs/genuicraft-mobile/v1"
    stream = [
        {
            "version": "v0.9",
            "createSurface": {
                "surfaceId": "s",
                "catalogId": catalog_id,
                "sendDataModel": False,
            },
        },
        {
            "version": "v0.9",
            "updateComponents": {
                "surfaceId": "s",
                "components": [
                    {"id": "root", "component": "Stack", "children": ["text"]},
                    {"id": "text", "component": "Text", "text": "After"},
                ],
            },
        },
        {
            "version": "v0.9",
            "updateDataModel": {"surfaceId": "s", "path": "/nested/value", "value": 2},
        },
    ]
    decoded = decode_to_flat_spec(stream, format_hint=A2UI_V1_WIRE).flat_spec
    assert decoded["root"] == "root"
    assert decoded["elements"]["text"]["props"]["text"] == "After"
    assert decoded["state"]["nested"]["value"] == 2


def test_detection_handles_json_string_wrapped_express() -> None:
    express = encode_from_flat_spec(reference_spec(), A2UI_EXPRESS_V1)
    wrapped = json.dumps(express)
    assert detect_format(wrapped) == A2UI_EXPRESS_V1
    assert semantic_equivalent(reference_spec(), wrapped)


def test_renderer_contract_accepts_emit_event() -> None:
    result = coerce_and_validate(reference_spec())
    assert result.is_valid, result.error


def test_metrics_are_format_invariant_and_raw_express_is_normalized() -> None:
    spec = reference_spec()
    response_text = "# Review\nContinue with the selected item."
    expected_metrics = compute_ui_metrics(response_text, spec)

    for format_id in (A2UI_EXPRESS_V1, A2UI_V1_WIRE):
        encoded = encode_from_flat_spec(spec, format_id, shorten_ids=True)
        assert compute_ui_metrics(response_text, encoded) == expected_metrics

    express = encode_from_flat_spec(spec, A2UI_EXPRESS_V1, shorten_ids=True)
    normalized = normalize_and_validate_candidate(
        express,
        canonicalization_profile=RENDERER_V2_CANONICALIZATION,
    )
    assert normalized.raw_parse_ok
    assert normalized.production_valid
    assert normalized.canonical_spec is not None
    assert semantic_hash(normalized.canonical_spec) == semantic_hash(spec)


def test_conversion_cli_emits_production_formats(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    output_dir = tmp_path / "converted"
    input_path.write_text(json.dumps(reference_spec()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "convert_ir.py"),
            str(input_path),
            "--all-formats",
            "--output",
            str(output_dir),
        ],
        cwd=ROOT.parent,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    express_payload = (output_dir / "output.express.a2ui").read_text(encoding="utf-8")
    wire_payload = json.loads((output_dir / "output.a2ui.json").read_text(encoding="utf-8"))
    report = json.loads((output_dir / "conversion_report.json").read_text(encoding="utf-8"))
    assert semantic_equivalent(reference_spec(), express_payload)
    assert semantic_equivalent(reference_spec(), wire_payload)
    assert set(report) == {FLAT_SPEC_V1, A2UI_EXPRESS_V1, A2UI_V1_WIRE}


def test_shared_express_conformance_corpus() -> None:
    corpus = json.loads(
        (ROOT / "tests" / "fixtures" / "a2ui_express_conformance_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert corpus["version"] == "a2ui-express-conformance-v1"
    for case in corpus["valid"]:
        decoded = decode_to_flat_spec(case["program"], format_hint=A2UI_EXPRESS_V1).flat_spec
        assert decoded == case["canonical"], case["name"]
        assert semantic_hash(decoded) == semantic_hash(case["canonical"])
    for case in corpus["invalid"]:
        with pytest.raises(ValueError):
            decode_to_flat_spec(case["program"], format_hint=A2UI_EXPRESS_V1)


def test_benchmark_cli_reports_semantic_roundtrip(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.json"
    output_json = tmp_path / "benchmark.json"
    output_csv = tmp_path / "benchmark.csv"
    corpus_path.write_text(
        json.dumps({"fixtures": [{"spec": reference_spec()}]}),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "benchmark_ir_formats.py"),
            str(corpus_path),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        cwd=ROOT.parent,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["corpus_samples_discovered"] == 1
    assert report["corpus_samples_benchmarked"] == 1
    assert report["failures"] == []
    assert all(details["roundtrip_passed"] == 1 for details in report["formats"].values())
    csv_text = output_csv.read_text(encoding="utf-8")
    assert csv_text.count("\n") >= 2
    assert str(tmp_path) not in csv_text
    assert "corpus.json:$.fixtures[0].spec" in csv_text
