from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_a2ui_express_prompt.py"
PROMPT_PATHS = [
    ROOT / "prompts" / "genui_gen_mobile_a2ui_express_v1.md",
    ROOT / ".." / "android" / "app" / "src" / "main" / "assets" / "pipeline_prompts" / "genui_gen.md",
    ROOT / ".." / "android" / "app" / "src" / "main" / "assets" / "pipeline_prompts" / "genui_gen_a2ui_express_v1.md",
]


def test_generated_prompt_copies_are_drift_free() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=ROOT.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    contents = [path.resolve().read_bytes() for path in PROMPT_PATHS]
    assert len({content for content in contents}) == 1


def test_generated_prompt_is_catalog_and_quality_complete() -> None:
    text = PROMPT_PATHS[0].read_text(encoding="utf-8")
    folded = text.lower()
    for token in (
        "grammar=",
        "catalog=",
        "profile=",
        "<a2ui>",
        "Table(columns, statePath, rows, title, domain, preferredPresentation)",
        "emitEvent(name, context, wantResponse, responsePath)",
        "preserve every requested fact",
        "_props",
    ):
        assert token.lower() in folded


def test_generated_prompt_teaches_grounded_field_extraction_not_card_padding() -> None:
    text = PROMPT_PATHS[0].read_text(encoding="utf-8")
    for instruction in (
        "Treat bullets as source formatting",
        "one record\n  per entity",
        'preferredPresentation="cards"',
        "Merely wrapping the unchanged bullet in a Card is not field extraction",
        "attached to its original entity and section",
        "do not swap values across",
        "infer missing values, invent facts or unsupported field meanings",
        "Plain steps, checklists, and unstructured text items remain `List`",
        'preferredPresentation="table"',
        "add empty\n  cards to increase component counts",
    ):
        assert instruction in text


def test_field_extraction_example_compiles_to_separate_source_grounded_fields() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from pipeline.ir_formats.active import validate_express_completion

    text = PROMPT_PATHS[0].read_text(encoding="utf-8")
    example_section = text.split("Field extraction example", 1)[1]
    example = "<a2ui>" + example_section.split("<a2ui>", 1)[1].split("</a2ui>", 1)[0] + "</a2ui>"
    result = validate_express_completion(example)
    assert result.raw_valid, result.errors
    graph = result.canonical_graph
    assert graph is not None
    for identifier, title, fields in (
        ("cedar", "Hotel: Cedar", ["City: Pune", "Price: INR 4200/night", "Rating: 4.6/5"]),
        ("elm", "Hotel: Elm", ["City: Kochi", "Price: INR 5300/night", "Rating: 4.8/5"]),
    ):
        card = graph["elements"][identifier]
        assert card["type"] == "Card"
        assert card["props"]["title"] == title
        assert [graph["elements"][child]["props"]["text"] for child in card["children"]] == fields
