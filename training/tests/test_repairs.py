from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "dataset" / "src"))

from ir_training.data.build_pairs import prepare_dataset
from ir_training.data.repairs import normalize_gap, repair_graph


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_gap_normalization_is_deterministic() -> None:
    assert normalize_gap("xs") == "sm"
    assert normalize_gap("extra-large") == "xl"
    assert normalize_gap(0) == "none"
    assert normalize_gap(12) == "md"
    assert normalize_gap("unrecognized") == "md"


def test_repairs_preserve_layout_and_semantic_content() -> None:
    graph = {
        "root": "root",
        "state": {},
        "elements": {
            "root": {
                "type": "Stack",
                "props": {"gap": "xs", "padding": 16, "paddingTop": 8},
                "children": ["email", "button"],
            },
            "email": {
                "type": "EmailPreview",
                "props": {"body": "Hello", "signature": "Thanks"},
                "children": [],
            },
            "button": {
                "type": "Button",
                "props": {"label": "Open"},
                "children": [],
                "on": {
                    "press": {
                        "action": "openUrl",
                        "params": {"url": "javascript:alert(1)"},
                    }
                },
            },
        },
    }

    result = repair_graph(graph)

    assert result.graph["elements"]["root"]["props"] == {"gap": "sm", "padding": 16}
    assert result.graph["elements"]["email"]["props"]["body"] == "Hello\n\nThanks"
    assert "signature" not in result.graph["elements"]["email"]["props"]
    assert "on" not in result.graph["elements"]["button"]
    assert {change.kind for change in result.changes} == {
        "gap_normalized",
        "layout_prop_dropped",
        "semantic_prop_mapped",
        "unsafe_url_action_removed",
    }


def test_prepare_dataset_records_repairs_and_emits_strict_express(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    response = {
        "response_id": "r1",
        "query_id": "q1",
        "response_text": "Open the email.",
        "intent_bucket": "email",
    }
    spec = {
        "root": "root",
        "state": {},
        "elements": {
            "root": {
                "type": "Stack",
                "props": {"direction": "vertical", "gap": "xs", "padding": 16},
                "children": ["button"],
            },
            "button": {
                "type": "Button",
                "props": {"label": "Open"},
                "children": [],
                "on": {
                    "press": {
                        "action": "openUrl",
                        "params": {"url": "http://unsafe.example"},
                    }
                },
            },
        },
    }
    _write_jsonl(run_dir / "responses.jsonl", [response])
    _write_jsonl(
        run_dir / "genui.jsonl",
        [{"response_id": "r1", "ui_id": "u1", "genui_json": spec}],
    )

    out_dir = tmp_path / "prepared"
    manifest = prepare_dataset(
        {
            "run": {
                "source_run_dir": str(run_dir),
                "output_dir": str(out_dir),
                "system_prompt": "Return Express.",
            },
            "filters": {"max_input_chars": 1000, "max_output_chars": 10000},
            "split": {"train": 1, "val": 0, "test": 0, "stratify_by": "intent_bucket"},
        }
    )

    assert manifest["counts"]["accepted"] == 1
    assert manifest["repair"]["accepted_records_with_repairs"] == 1
    assert manifest["repair"]["change_counts"]["gap_normalized"] == 1
    assert manifest["repair"]["change_counts"]["unsafe_url_action_removed"] == 1
    row = json.loads((out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["repair"]["applied"] is True
    assert row["canonical_graph"]["elements"]["root"]["props"]["gap"] == "sm"
    assert "openUrl" not in row["completion"]
