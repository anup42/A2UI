from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

DATASET_ROOT = Path(os.environ.get("A2UI_TEST_DATASET_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(DATASET_ROOT / "src"))
from pipeline import aggregate_background as background
from pipeline.metrics import aggregate_metrics, compute_media_score


def request_for(tmp_path, monkeypatch):
    rows = [{"ui_id": "u1", "response_id": "r1", "intent": "generic",
             "validation": {"schema_valid_strict": True},
             "genui_json": {"root": "title", "state": {}, "elements": {
                 "title": {"type": "Text", "props": {"text": "Train costs $24."}, "children": []}}}}]
    genui, responses = tmp_path / "genui.jsonl", tmp_path / "responses.jsonl"
    genui.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    responses.write_text(json.dumps({"response_id": "r1", "response_text": "Train costs $24."}) + "\n", encoding="utf-8")
    monkeypatch.setattr(background.subprocess, "Popen", lambda *a, **kw: type("Proc", (), {"pid": 123})())
    receipt = background.schedule_aggregate(
        genui_path=genui, responses_path=responses, aggregates_path=tmp_path / "aggregates.json",
        aggregate_weights={}, metric_kwargs={"metric_version": "v5_4"}, work_dir=tmp_path / "workers")
    state = Path(receipt["state_dir"])
    request = pickle.loads((state / "request.pkl").read_bytes())
    return request, rows, state


def test_background_scores_equal_synchronous_scores(tmp_path, monkeypatch):
    request, rows, _ = request_for(tmp_path, monkeypatch)
    rows[0]["response_text"] = "Train costs $24."
    expected = aggregate_metrics(rows, render_rows_by_ui_id={}, metric_version="v5_4")
    expected["media_score"] = compute_media_score(expected)
    actual = background.compute_snapshot(request)
    for key, value in expected.items():
        assert actual[key] == value, key
    assert actual["aggregated_sample_count"] == 1


def test_snapshot_excludes_later_records_and_partial_append(tmp_path, monkeypatch):
    request, _, _ = request_for(tmp_path, monkeypatch)
    genui = Path(request["genui_path"])
    with genui.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"ui_id": "later"}) + "\n")
        handle.write('{"ui_id": "partial')
    assert len(list(background._snapshot_rows(genui, request["snapshot_bytes"][str(genui)]))) == 1
    assert len(list(background._snapshot_rows(genui, genui.stat().st_size))) == 2


def test_submission_preserves_runtime_configs_and_returns_without_scoring(tmp_path, monkeypatch):
    monkeypatch.setattr(background, "compute_snapshot", lambda _: pytest.fail("submission must not compute"))
    request, _, _ = request_for(tmp_path, monkeypatch)
    assert request["metric_kwargs"] == {"metric_version": "v5_4"}
    assert request["snapshot_bytes"][request["genui_path"]] > 0


def test_final_completion_requires_report_for_all_written_bytes(tmp_path, monkeypatch):
    request, _, _ = request_for(tmp_path, monkeypatch)
    monkeypatch.setattr(background.subprocess, "run", lambda *a, **kw: None)
    output = Path(request["aggregates_path"])
    output.write_text(json.dumps({"aggregated_input_bytes": 0}))
    arguments = {"genui_path": request["genui_path"], "aggregates_path": output,
                 "work_dir": tmp_path / "workers"}
    with pytest.raises(RuntimeError, match="complete generated corpus"):
        background.wait_for_aggregate(**arguments)
    output.write_text(json.dumps({"aggregated_input_bytes": request["snapshot_bytes"][request["genui_path"]]}))
    background.wait_for_aggregate(**arguments)


def test_stage3_finishes_without_waiting_for_full_aggregate(tmp_path, monkeypatch):
    import logging

    from pipeline import stage3_genui
    from pipeline.cache import PromptCache
    from pipeline.ir_formats import A2UI_EXPRESS_V1
    from test_stage3_rolling_dispatch import Adapter
    from utils.rate_limit import RateLimiter

    # Exercise the POSIX scheduling branch without spawning a platform worker.
    monkeypatch.setattr(stage3_genui, "os", SimpleNamespace(
        name="posix", environ=os.environ, getenv=os.getenv, getpid=os.getpid,
    ))
    monkeypatch.setenv("GENUI_STAGE3_AGGREGATE_WORK_DIR", str(tmp_path / "workers"))
    monkeypatch.setenv("GENUI_STAGE3_ROLLING_DISPATCH", "1")
    monkeypatch.setenv("LOCAL_ALLOW_HTTP_ENDPOINT", "1")
    monkeypatch.setenv("LOCAL_VLLM_STRICT_OFFLINE", "0")
    monkeypatch.setenv("LOCAL_VLLM_RETRY_RESULT_ERRORS", "0")
    monkeypatch.setenv("STAGE3_FINAL_REGEN_ATTEMPTS", "0")
    scheduled = []
    monkeypatch.setattr(background, "schedule_aggregate", lambda **kwargs: scheduled.append(kwargs) or {"pid": 123})
    original_aggregate = stage3_genui.aggregate_metrics
    def per_record_only(*args, **kwargs):
        if "v4_config" in kwargs:
            pytest.fail("Stage3 must not block on full metrics")
        return original_aggregate(*args, **kwargs)
    monkeypatch.setattr(stage3_genui, "aggregate_metrics", per_record_only)
    responses = tmp_path / "responses.jsonl"
    responses.write_text(json.dumps({"response_id": "r_0", "query_id": "q_0", "n_idx": 1,
                                     "response_text": "Trip train at 09:30 costs $24. Direct service MARKER_0."}) + "\n")
    queries = tmp_path / "queries.jsonl"
    queries.write_text(json.dumps({"query_id": "q_0", "intent": "travel", "tags": []}) + "\n")
    stage3_genui.run_stage3(
        queries_path=queries, responses_path=responses, adapter=Adapter(),
        prompt_path=DATASET_ROOT / "prompts/genui_gen_mobile_a2ui_express_v1.md",
        schema_path=DATASET_ROOT / "schema/canonical_ui_graph_v1.schema.json",
        genui_path=tmp_path / "genui.jsonl", artifacts_dir=tmp_path / "artifacts",
        candidates_per_response=1, max_repair_attempts=0, max_tokens=2048,
        prompt_max_tokens=None, batch_size=1, seed=1, rate_limiter=RateLimiter(0),
        cache=PromptCache(tmp_path / "cache.jsonl", enabled=False),
        logger=logging.getLogger("background-aggregate-test"), max_total=1, max_attempts=1,
        aggregates_path=tmp_path / "aggregates.json", metric_version="legacy",
        ir_formats=[A2UI_EXPRESS_V1],
    )
    assert scheduled
    assert len((tmp_path / "genui.jsonl").read_text().splitlines()) == 1
    assert scheduled[-1]["genui_path"].stat().st_size > 0


@pytest.mark.skipif(os.name != "posix", reason="Space worker uses a POSIX OS lock")
def test_worker_coalesces_newer_snapshot_and_deduplicates_waiters(tmp_path, monkeypatch):
    request, _, state = request_for(tmp_path, monkeypatch)
    calls = []
    def compute(current):
        calls.append(current["revision"])
        if len(calls) == 1:
            newer = dict(request, revision="newest")
            background._atomic_write(state / "request.pkl", pickle.dumps(newer))
        return {"aggregated_sample_count": len(calls), "aggregated_input_bytes": len(calls)}
    monkeypatch.setattr(background, "compute_snapshot", compute)
    background.run_worker(state)
    background.run_worker(state)
    assert calls == [request["revision"], "newest"]
    assert json.loads(Path(request["aggregates_path"]).read_text())["aggregated_sample_count"] == 2
    assert json.loads((state / "status.json").read_text())["status"] == "completed"


@pytest.mark.skipif(os.name != "posix", reason="Space worker uses a POSIX OS lock")
def test_failure_preserves_previous_report(tmp_path, monkeypatch):
    request, _, state = request_for(tmp_path, monkeypatch)
    output = Path(request["aggregates_path"])
    output.write_text('{"previous": true}')
    def failed(_):
        raise RuntimeError("scoring failed")
    monkeypatch.setattr(background, "compute_snapshot", failed)
    with pytest.raises(RuntimeError, match="scoring failed"):
        background.run_worker(state)
    assert json.loads(output.read_text()) == {"previous": True}
    assert json.loads((state / "status.json").read_text())["status"] == "failed"
