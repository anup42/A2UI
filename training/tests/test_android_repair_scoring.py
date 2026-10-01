from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "dataset/src")]

from ir_training.eval.android_repair import repair_batch, resolve_runtime, source_digest
from ir_training.eval.compare_to_baseline import evaluate_predictions

VALID = '<a2ui>\nroot=Text("Hello")\n</a2ui>'


def test_repair_scores_keep_raw_results_and_rejected_rows(tmp_path, monkeypatch):
    predictions = tmp_path / "predictions.jsonl"
    original_rows = [
        {"id": "fenced", "response_text": "Hello", "generated_text": "```a2ui\n" + VALID + "\n```"},
        {"id": "broken", "response_text": "Do not import this reference into repair", "generated_text": "<a2ui>\n$/{$/}", "expected": VALID},
    ]
    predictions.write_text("".join(json.dumps(row) + "\n" for row in original_rows), encoding="utf-8")
    observed = []
    def fake_repair(texts, runtime):
        observed.extend(texts)
        return [
            {"index": 0, "success": True, "express": VALID, "repair_kind": "STRUCTURAL", "repair_seconds": .01},
            {"index": 1, "success": False, "repair_kind": "REJECTED", "repair_seconds": .02},
        ], {"accepted_count": 1, "rejected_count": 1}
    monkeypatch.setattr("ir_training.eval.android_repair.repair_batch", fake_repair)
    aggregate = evaluate_predictions(predictions, output_dir=tmp_path / "scores",
        android_repair_config=tmp_path / "runtime.json", metric_version="dual")
    rows = [json.loads(line) for line in (tmp_path / "scores/scored_predictions.jsonl").read_text().splitlines()]
    assert observed == [row["generated_text"] for row in original_rows]
    assert all(row["metrics"]["native_syntax_valid"] is False for row in rows)
    assert rows[0]["android_repaired_metrics"]["native_syntax_valid"] is True
    assert rows[1]["android_repaired_metrics"]["native_syntax_valid"] is False
    assert rows[1]["android_repaired_text"] == ""
    assert aggregate["count"] == 2
    assert aggregate["android_repaired_diagnostics"]["count"] == 2
    assert aggregate["android_repaired_diagnostics"]["native_syntax_valid_avg"] == .5
    assert aggregate["android_repair_available"] is True
    assert "android_repaired_generation_reward_v5_4" in aggregate
    assert "Do not import" not in "".join(observed)


def test_stale_runtime_is_rejected_before_execution(tmp_path):
    manifest = tmp_path / "runtime.json"
    manifest.write_text(json.dumps({"profile": "TRAINED_E2B_V10_W4", "source_sha256": "old", "command": ["never-run"]}))
    with pytest.raises(ValueError, match="stale"):
        repair_batch([VALID], manifest)
    assert len(source_digest()) == 64


def test_source_digest_includes_trained_converter_policy(tmp_path, monkeypatch):
    from ir_training.eval import android_repair
    bridge = tmp_path / "bridge.kt"
    bridge.write_text("bridge", encoding="utf-8")
    monkeypatch.setattr(android_repair, "source_files", lambda root: [bridge])
    sdk = tmp_path / android_repair.SDK
    for relative in ("GenUiSession.kt", "GenUiTrainedConverter.kt", "internal/pipeline/FlatSpecContract.kt"):
        path = sdk / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original", encoding="utf-8")
    before = source_digest(tmp_path)
    (sdk / "GenUiTrainedConverter.kt").write_text("changed repair policy", encoding="utf-8")
    assert source_digest(tmp_path) != before


@pytest.mark.parametrize("changes", [
    {"repair_seconds": True}, {"repair_seconds": -1}, {"repair_seconds": float("nan")},
    {"repair_seconds": float("inf")}, {"index": False}, {"express": " "},
    {"success": False, "repair_kind": "NONE"},
])
def test_malformed_bridge_evidence_cannot_be_scored(tmp_path, monkeypatch, changes):
    from ir_training.eval import android_repair
    manifest = tmp_path / "runtime.json"
    manifest.write_text(json.dumps({"profile": android_repair.PROFILE,
        "source_sha256": source_digest(), "command": ["fake-bridge"]}), encoding="utf-8")
    row = {"index": 0, "success": True, "repair_kind": "NONE", "express": VALID,
           "repair_seconds": .01, **changes}
    monkeypatch.setattr(android_repair.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(returncode=0, stdout=json.dumps(row) + "\n", stderr=""))
    with pytest.raises(ValueError, match="Android repair bridge"):
        repair_batch([VALID], manifest)


def test_unconfigured_repair_is_explicit(tmp_path, monkeypatch):
    monkeypatch.delenv("A2UI_ANDROID_REPAIR_RUNTIME", raising=False)
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({"response_text": "Hello", "generated_text": VALID}) + "\n")
    aggregate = evaluate_predictions(predictions)
    assert aggregate["android_repair_available"] is False
    assert "not configured" in aggregate["android_repair_status"]
    assert "android_repaired_diagnostics" not in aggregate


@pytest.mark.skipif(not os.environ.get("A2UI_ANDROID_REPAIR_RUNTIME"), reason="Build JVM bridge and set A2UI_ANDROID_REPAIR_RUNTIME for exact Android parity")
def test_actual_android_repair_matches_sdk_regressions():
    # Cases from GenUiCompilerTest.kt; executes those very compiler/repair files.
    texts = [VALID, "```a2ui\n" + VALID + "\n```", "Here is your UI: " + VALID,
        '<a2ui>\nroot=Column(children=[table],Gap="Ã©â€œÂº")\ntable=Table(columns=[{Name:"Name","Value"}],rows=[["A","1"]])\n</a2ui>',
        '<a2ui>\n$/={heading:"Preserved answer",detail:"Second generated fact",broken:[\n',
        '<a2ui>\n$/{$/}']
    results, evidence = repair_batch(texts, resolve_runtime())
    assert [row["repair_kind"] for row in results] == ["NONE", "STRUCTURAL",
        "GENERATED_DSL_REPAIR", "GENERATED_DSL_REPAIR", "GENERATED_DSL_REPAIR", "REJECTED"]
    assert results[0]["express"] == results[1]["express"] == VALID
    assert 'columns=["Name","Value"]' in results[3]["express"]
    assert "Preserved answer" in results[4]["express"]
    assert "Second generated fact" in results[4]["express"]
    assert evidence["source_or_reference_supplied_to_repair"] is False
    assert evidence["accepted_count"] == 5
    assert evidence["rejected_count"] == 1
    assert all(row["repair_seconds"] >= 0 for row in results)


@pytest.mark.skipif(not os.environ.get("A2UI_ANDROID_REPAIR_RUNTIME"), reason="Requires the Kotlin bridge")
def test_actual_android_repair_preserves_unicode_and_current_chart_catalog():
    label = "हैलो · 안녕하세요 · 日本語 · 😀"
    text = '<a2ui>\nroot=Text(' + json.dumps(label, ensure_ascii=False) + ')\n</a2ui>'
    chart = '<a2ui>\nroot=Chart(chartType="radar",columns=["Category","Value"],rows=[["A",1],["B",2],["C",3]])\n</a2ui>'
    results, _ = repair_batch([text, chart], resolve_runtime())
    assert [row["repair_kind"] for row in results] == ["NONE", "NONE"]
    assert label in results[0]["express"]
    assert results[1]["express"] == chart


def test_minimal_tensorboard_keeps_after_repair_headlines():
    from ir_training.eval.tensorboard_logging import select_tensorboard_metrics
    selected = select_tensorboard_metrics({
        "generation_reward_v5_4_avg": 55.0,
        "android_repaired_generation_reward_v5_4": 74.5,
        "android_repaired_generation_reward_v5_4_avg": 74.5,
        "android_repaired_native_syntax_valid_avg": .96875,
        "android_repair": {"batch_wall_seconds": .9, "mean_repair_seconds": .017},
    }, detail="minimal")
    assert selected["android_repaired_generation_reward_v5_4_avg"] == 74.5
    assert "android_repaired_generation_reward_v5_4" not in selected
    assert selected["android_repaired_native_syntax_valid_avg"] == .96875
    assert selected["android_repair/batch_wall_seconds"] == .9
