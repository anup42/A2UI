import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "training/src"), str(ROOT / "training/scripts")]
from ir_training.data.archive_semantic_review import (
    pointer, process_graph, repair_graph, review_graph, visible_strings,
)
from ir_training.data.express_preparation import _api, serialize_checked
from ir_training.data.url_preprocess import preprocess_training_urls, restore_url_placeholders

FIXTURES = json.loads((Path(__file__).parent / "fixtures/archive_semantic_review_cases.json").read_text(encoding="utf-8"))


def graph(elements=None, state=None):
    nodes = deepcopy(elements or {"root": {"type": "Text", "props": {"text": "A valid label"}}})
    for node in nodes.values():
        node.setdefault("children", [])
    return {"root": "root", "state": state or {}, "elements": nodes}


def test_pointer_resolver_matches_android_and_repairs_only_unresolved_literal_keys():
    g = graph({"root": {"type": "Table", "props": {"statePath": "/a/b", "columns": [{"key": "x", "label": "X"}]}}}, {"/a/b": [{"x": "one"}]})
    before = deepcopy(g)
    result = process_graph("one", g, reviewed=[])
    assert pointer(g["state"], "/a/b") is None
    assert result["graph"]["elements"]["root"]["props"]["statePath"] == "/~1a~1b"
    assert "one" in visible_strings(result["graph"])
    assert g == before
    assert not process_graph("one", result["graph"], reviewed=[])["proofs"]
    g["state"]["a"] = {"b": [{"x": "actual nested value"}]}
    assert not repair_graph("one", g)[3]


def test_unresolvable_pointer_without_literal_data_is_quarantined():
    g = graph({"root": {"type": "Table", "props": {"statePath": "/absent", "columns": ["x"]}}})
    assert "unresolved_table_state_path" in {x["code"] for x in review_graph("hello", g, reviewed=[])}


def test_only_empty_unsupported_media_columns_are_removed():
    g = graph({"root": {"type": "Table", "props": {"rows": [{"meal": "Lunch"}], "columns": [{"key": "meal"}, {"key": "image"}, {"key": "imageAlt"}]}}})
    assert len(repair_graph("Lunch", g)[1]["elements"]["root"]["props"]["columns"]) == 1
    assert not repair_graph("Media: Image=[IMAGE_URL_1]", g)[3]
    g["elements"]["root"]["props"]["primaryColumn"] = "image"
    assert any(c.get("key") == "image" for c in repair_graph("Lunch", g)[1]["elements"]["root"]["props"]["columns"])


def test_no_unused_state_credit_for_source_actions_and_media():
    g = graph({"root": {"type": "Table", "props": {"columns": [{"key": "title", "label": "Title"}], "statePath": "/items"}}}, {"items": [{"title": "Item", "url": "[ACTION_URL_1]", "icon": "[ICON_URL_1]"}]})
    codes = {x["code"] for x in review_graph("[Button: Remove] <[ACTION_URL_1]>\nMedia: Icon=[ICON_URL_1]", g, reviewed=[])}
    assert {"inline_action_not_bound", "source_media_not_bound"} <= codes


def test_exact_source_clause_restore_is_unique_and_does_not_synthesize():
    short = "Choose the pads for your Civic and inspect the rotors before installation."
    full = "Choose the mid-range ceramic pads for your 2018 Civic and inspect the vented rotors before installation."
    g = graph({"root": {"type": "Text", "props": {"text": short}}})
    result = process_graph(full, g, reviewed=[])
    assert result["graph"]["elements"]["root"]["props"]["text"] == full
    assert result["changes"]["unique_source_clause_restoration"] == 1
    ambiguous = full + "\n" + full.replace("2018", "2020")
    assert not repair_graph(ambiguous, g)[3]
    assert not repair_graph("Totally unrelated response.", g)[3]


def test_false_asset_prose_is_not_masked_but_real_extensionless_asset_is():
    source = "Signed asset/liability sheet and assets/liabilities summary.\nMedia: Asset=assets/models/gemma"
    result = preprocess_training_urls(source, {"text": source})
    assert "asset/liability" in result.response_text
    assert "assets/liabilities" in result.response_text
    assert "assets/models/gemma" not in result.response_text
    assert restore_url_placeholders(result.response_text, result.url_map) == source


def test_known_false_mask_requires_proven_map_and_never_unmasks_an_action():
    g = graph({"root": {"type": "Text", "props": {"text": "Signed [MEDIA_ASSET_1] sheet"}}})
    mapping = {"[MEDIA_ASSET_1]": {"url": "asset/liability", "kind": "local_asset"}}
    source, fixed, cleaned, proofs = repair_graph("Signed [MEDIA_ASSET_1] sheet", g, mapping)
    assert source == "Signed asset/liability sheet" and not cleaned and proofs
    assert not repair_graph(source, fixed, cleaned)[3]
    assert not repair_graph("Signed [MEDIA_ASSET_1] sheet", g, {})[3]


def test_unbound_checkbox_is_not_flagged_as_inert():
    g = graph({"root": {"type": "CheckBox", "props": {"label": "Finish task", "value": False}}})
    assert not review_graph("Finish task", g, reviewed=[])


@pytest.mark.parametrize("value", ["asset/liability", "asset/liability.", "Asset/Liability,", "assets/liabilities!"])
def test_standalone_false_asset_prose_and_punctuation_remain_literal(value):
    assert preprocess_training_urls(value, {"text": value}).response_text == value


def test_generic_url_placeholder_is_recognized_in_a_real_action():
    g = graph({"root": {"type": "Button", "props": {"label": "Open"}, "on": {"press": {"action": "openUrl", "params": {"url": "[URL_1]"}}}}})
    assert not review_graph("[Button: Open] <[URL_1]>", g, reviewed=[])


def test_literal_description_in_field_is_rejected():
    g = graph({"root": {"type": "TextField", "props": {"label": "Candidate", "value": "Auto-filled from QR"}}})
    assert any(x["code"] == "description_used_as_input_value" for x in review_graph("Candidate", g, reviewed=[]))


@pytest.mark.parametrize("case", FIXTURES, ids=lambda x: f"manual-{x['sample']:03}")
def test_manual_regressions_are_either_source_proven_repaired_or_quarantined(case):
    active, express, _, _ = _api()
    original = active.decode_express_completion(case["target"])
    snapshot = deepcopy(original)
    result = process_graph(case["source"], original, url_map=case["url_map"], original_source_sha256=case["original_source_sha256"])
    assert original == snapshot
    if case["previous_verdict"] != "KEEP":
        assert result["issues"] or result["proofs"], "Known issue silently kept unchanged"
    if case["sample"] in {11,29,90,97}:
        assert not result["issues"]
    if result["issues"]:
        return
    checked = serialize_checked(express.encode(result["graph"], shorten_ids=False), "root-first")
    assert checked.graph == result["graph"]
    again = process_graph(result["source"], checked.graph, url_map=result["url_map"], original_source_sha256=case["original_source_sha256"])
    assert not again["proofs"] and not again["issues"]


def test_export_reconciles_repairs_quarantine_and_immutable_inputs(tmp_path, monkeypatch):
    import csv
    import finalize_archive_semantics as runner
    from ir_training.data.archive_recovery import text_sha256
    from ir_training.data.express_preparation import TASK_PREFIX
    from verify_recovered_archive import main as verify_main

    _, express, semantic_hash, _ = _api()
    base, output, report = tmp_path / "v9", tmp_path / "v10", tmp_path / "report"
    base.mkdir()
    table = graph({"root": {"type": "Table", "props": {"rows": [{"meal": "Lunch"}], "columns": [{"key": "meal", "label": "Meal"}, {"key": "image", "label": "Image"}]}}})
    items = [("train", 1, "Lunch", table),
             ("train", 2, "Sources:\n- Guide: <[SOURCE_URL_1]>", graph({"root": {"type": "Text", "props": {"text": "Guide"}}})),
             ("val", 1, "Dinner", graph({"root": {"type": "Text", "props": {"text": "Dinner"}}}))]
    rows = {"train": [], "val": []}
    decisions = []
    for split, line, source, g in items:
        target = serialize_checked(express.encode(g, shorten_ids=False), "root-first").text
        family = "family-" + text_sha256(source)
        recovery = {"coordinate": f"{split}:{line}", "original_source_sha256": text_sha256(source),
                    "effective_source_sha256": text_sha256(source), "effective_target_sha256": text_sha256(target),
                    "effective_semantic_sha256": semantic_hash(g), "transformations": [], "repair_metrics": {}}
        rows[split].append({"id": f"{split}-{line}", "source_id": family, "response_text": source, "completion": target,
                           "messages": [{"role": "user", "content": TASK_PREFIX + source}, {"role": "assistant", "content": target}],
                           "repair": {"applied": False, "changes": []},
                           "metadata": {"source_id": family, "assigned_split": split, "archive_recovery": recovery,
                                        "archive_refinement": {"assigned_split": split}}})
        decisions.append({"split": split, "line": line, "category": "KEEP", "reason": "fixture", "assigned_split": split,
                          "prior_category": "KEEP", "prior_reason": "fixture", "source_family": family, "warnings": ""})
    for split, records in rows.items():
        (base / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    for filename in ("decisions.csv", "quarantine.csv"):
        with (base / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(decisions[0]))
            writer.writeheader()
            if filename == "decisions.csv":
                writer.writerows(decisions)
    manifest = {"schema_version": 5, "status": "candidate_export_complete", "output_rows": {"train": 2, "val": 1},
                "all_rows_reconciled": 3, "split_rebuild": True, "immutable_input_sha256": {}, "implementation_sha256": {},
                "benchmark_file_sha256": runner.load_goldens()[2],
                "outputs": {p.name: runner.file_sha256(p) for p in base.iterdir()}}
    (base / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    snapshots = {p.name: p.read_bytes() for p in base.iterdir()}
    monkeypatch.setattr(runner, "_init_worker", lambda: None)
    monkeypatch.setattr(runner, "check_export_space", lambda *args: {"test": True})
    result = runner.export(base, output, report, workers=1)
    assert result["output_rows"] == {"train": 1, "val": 1}
    assert result["categories"]["combined"] == {"REPAIR": 1, "QUARANTINE": 1, "KEEP": 1}
    assert result["semantic_review"]["outcomes_from_v9"]["train"] == {"REPAIR": 1, "QUARANTINE": 1}
    assert snapshots == {p.name: p.read_bytes() for p in base.iterdir()}
    monkeypatch.setattr(sys, "argv", ["verify_recovered_archive.py", "--dataset-dir", str(output), "--strict-sample-size", "3"])
    assert verify_main() == 0
    with pytest.raises(FileExistsError):
        runner.export(base, output, report, workers=1)
