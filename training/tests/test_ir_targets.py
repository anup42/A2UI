from __future__ import annotations

import json
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.data.build_pairs import prepare_dataset
from ir_training.data.ir_targets import (
    A2UI_EXPRESS_V1,
    canonical_flat_spec,
    materialize_completion_targets,
    resolve_target_formats,
)
from ir_training.data.legacy_targets import canonical_graph_from_legacy_source


def _spec() -> dict:
    return {
        "root": "layout",
        "state": {"selected": "one"},
        "elements": {
            "layout": {"type": "Stack", "props": {"direction": "vertical"}, "children": ["text"]},
            "text": {"type": "Text", "props": {"text": "Hello"}, "children": []},
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_materialized_targets_round_trip_to_same_graph() -> None:
    targets = materialize_completion_targets(_spec())
    assert set(targets) == {A2UI_EXPRESS_V1}
    assert targets[A2UI_EXPRESS_V1].startswith("<a2ui>")
    for target in targets.values():
        assert canonical_flat_spec(target)["elements"]["root"]["type"] == "Stack"


def test_target_policy_preserves_native_format_and_does_not_double_history() -> None:
    assert resolve_target_formats({}, {}) == [A2UI_EXPRESS_V1]
    assert resolve_target_formats({}, {"source_format": A2UI_EXPRESS_V1}) == [A2UI_EXPRESS_V1]
    with pytest.raises(ValueError, match="Only a2ui_express_v1"):
        resolve_target_formats({"target_formats": ["both"]}, {})


def test_compact_source_is_rejected_outside_the_isolated_migration_tool() -> None:
    compact = {
        "v": "gci2",
        "r": "root",
        "e": {"root": {"t": "Text", "p": {"text": "legacy"}, "c": []}},
    }
    with pytest.raises(ValueError, match="migration-only"):
        canonical_graph_from_legacy_source(compact, source_format="compact_ir_v2")


def test_prepare_dataset_emits_only_express_target(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_jsonl(
        run_dir / "responses.jsonl",
        [{"response_id": "r1", "query_id": "q1", "response_text": "Say hello"}],
    )
    _write_jsonl(
        run_dir / "genui.jsonl",
        [{"response_id": "r1", "ui_id": "u1", "genui_json": _spec()}],
    )
    output_dir = tmp_path / "prepared"
    manifest = prepare_dataset(
        {
            "run": {
                "source_run_dir": str(run_dir),
                "output_dir": str(output_dir),
                "target_formats": [A2UI_EXPRESS_V1],
            },
            "filters": {
                "require_strict_flat_spec": True,
                "max_input_chars": 1000,
                "max_output_chars": 1000,
            },
            "split": {"train": 1, "val": 0, "test": 0, "stratify_by": "intent_bucket"},
        }
    )
    assert manifest["counts"]["accepted"] == 1
    rows = [json.loads(line) for line in (output_dir / "all.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["target_format"] for row in rows} == {A2UI_EXPRESS_V1}
    for row in rows:
        assert set(row["completion_targets"]) == {A2UI_EXPRESS_V1}
        assert row["messages"][-1]["content"] == row["completion"]
