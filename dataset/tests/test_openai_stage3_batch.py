"""Offline regressions: actual Stage 3 gates plus mocked paid Batch transport."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "dataset/src"), str(ROOT / "training/src")]

from ir_training.data.express_preparation import TASK_PREFIX
from pipeline import openai_stage3_batch as pilot
from pipeline import openai_stage3_batch_api as api
from pipeline import openai_stage3_batch_runner as runner
from pipeline.v11_regeneration_io import (
    SplitInput,
    canonical,
    digest,
    file_hash,
)

SOURCE = "- Alder; Price: $120; Breakfast: not included\n- Birch; Price: $80; Breakfast: included"
GOOD = '''<a2ui>
root=Column([a,b])
a=Card([ap,ab],title="Alder")
ap=Text("Price: $120")
ab=Text("Breakfast: not included")
b=Card([bp,bb],title="Birch")
bp=Text("Price: $80")
bb=Text("Breakfast: included")
</a2ui>'''


def row(identifier, source=SOURCE, split="train"):
    target = "<a2ui>root=Text(" + json.dumps(source) + ")</a2ui>"
    return {"id": identifier, "response_text": source, "completion": target, "a2ui_express": target,
            "completion_targets": {"a2ui_express_v1": target},
            "messages": [{"role": "system", "content": "Render faithfully."},
                         {"role": "user", "content": TASK_PREFIX + source},
                         {"role": "assistant", "content": target}],
            "metadata": {"assigned_split": split, "query_text": "Display the response."}}


def inputs(tmp_path, rows=None):
    directory = tmp_path / "input"
    directory.mkdir()
    runner.write_rows(directory / "train.jsonl", rows or [row("train-a")])
    runner.write_rows(directory / "val.jsonl", [row("validation", "Meeting at 11:00.", "val")])
    return directory


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    # No networking is permitted for prepare, including accidentally using an SDK.
    monkeypatch.setattr(api, "client", lambda: pytest.fail("Unexpected API client"))
    directory = inputs(tmp_path)
    output = tmp_path / "pilot"
    before = file_hash(directory / "train.jsonl")
    pilot.prepare(directory, output, samples=1)
    assert file_hash(directory / "train.jsonl") == before
    return output


def response(custom_id, text=GOOD, *, status="completed", model="gpt-6.1-sol"):
    return {"custom_id": custom_id, "error": None, "response": {"status_code": 200,
            "body": {"id": "resp-1", "status": status, "model": model,
                     "output": [{"type": "message", "role": "assistant", "status": "completed",
                                 "content": [{"type": "output_text", "text": text}]}],
                     "usage": {"input_tokens": 10000, "output_tokens": 3000,
                               "input_tokens_details": {"cached_tokens": 2000},
                               "output_tokens_details": {"reasoning_tokens": 1000}}}}}


class FakeAPI:
    def __init__(self, output, *, text=GOOD, fail_create=False):
        self.creates = 0
        self.uploads = 0
        self.downloads = 0
        self.fail_create = fail_create
        self.contract = pilot.load(output)
        self.records = [response(key, text) for key in reversed(self.contract["bindings"])]
        self.files = SimpleNamespace(create=self.upload, content=self.content)
        self.batches = SimpleNamespace(create=self.create, retrieve=self.retrieve)
        self.batch = None

    def upload(self, **kwargs):
        self.uploads += 1
        assert kwargs["purpose"] == "batch"
        assert kwargs["file"].read()
        return SimpleNamespace(id="file-input")

    def create(self, **kwargs):
        self.creates += 1
        self.batch = dict(kwargs, id="batch-1", status="completed", output_file_id="file-output", error_file_id=None)
        if self.fail_create:
            raise TimeoutError("Connection closed after server accepted the batch")
        return SimpleNamespace(model_dump=lambda **_: self.batch)

    def retrieve(self, identifier):
        assert identifier == "batch-1"
        return SimpleNamespace(model_dump=lambda **_: self.batch)

    def content(self, identifier):
        self.downloads += 1
        assert identifier == "file-output"
        return SimpleNamespace(content=b"".join(canonical(item) + b"\n" for item in self.records))


def test_prepare_captures_canonical_prompt_and_cost_without_target_or_api(prepared):
    contract = pilot.load(prepared)
    request = runner.read_rows(prepared / "requests.jsonl")[0]
    assert request["url"] == "/v1/responses"
    assert request["body"]["reasoning"] == {"effort": "medium"}
    assert request["body"]["store"] is False
    assert request["body"]["max_output_tokens"] == 8192
    messages = request["body"]["input"]
    assert SOURCE in messages[-1]["content"]
    assert row("train-a")["completion"] not in messages[-1]["content"]
    assert "same entity, section, event, and action context" in messages[0]["content"]
    assert "temperature" not in request["body"]
    assert contract["bindings"][request["custom_id"]].startswith("p-")
    costs = json.loads((prepared / "estimate.json").read_text())
    assert costs["estimated_input_tokens"] > 1000
    assert costs["output_token_cap_including_reasoning"] == 8192


def test_selection_deterministic_excludes_validation_sources_and_duplicates(tmp_path):
    directory = inputs(tmp_path, [row("a"), row("b", "Meeting at 11:00."),
                                 row("c", "Rain expected at 09:00."), row("duplicate-a")])
    first, counts, _ = pilot.select(SplitInput(directory), 2, 123)
    second, _, _ = pilot.select(SplitInput(directory), 2, 123)
    assert first == second
    assert {item["source_id"] for item in first} == {"a", "c"}
    assert counts["heldout_overlap_excluded"] == 1
    assert counts["duplicate_sources_excluded"] == 1


def test_benchmark_identity_is_rejected(tmp_path):
    directory = inputs(tmp_path, [row("BXP-1")])
    with pytest.raises(ValueError, match="evaluation-only"):
        pilot.select(SplitInput(directory), 1, 123)


def test_missing_evaluator_dependencies_fail_during_offline_prepare(tmp_path, monkeypatch):
    from pipeline.genui_quality import validation_v5_4

    def unavailable():
        raise RuntimeError("evaluator dependencies missing")

    monkeypatch.setattr(validation_v5_4, "ensure_v5_4_validation_ready", unavailable)
    directory = inputs(tmp_path)
    with pytest.raises(RuntimeError, match="dependencies missing"):
        pilot.prepare(directory, tmp_path / "not-created", samples=1)
    assert not (tmp_path / "not-created").exists()


def test_unbound_action_source_is_excluded_before_paid_preparation(tmp_path):
    directory = inputs(tmp_path, [row("invalid-action", "Action: [Button: Open] [ACTION_URL_1]"), row("valid")])
    selected, counts, _ = pilot.select(SplitInput(directory), 1, 123)
    assert selected[0]["source_id"] == "valid"
    assert counts["sampled_source_contract_excluded"] == 1


def test_identical_masked_prompts_have_distinct_bindings_and_out_of_order_replay():
    config = {"model": "gpt-6.1-sol", "reasoning_effort": "medium", "max_output_tokens": 8192}
    capture = runner.CaptureAdapter(config)
    for _ in range(2):
        capture.generate(prompt="Open [URL_1]", system="Convert faithfully")
    keys = list(capture.requests)
    assert len(keys) == 2
    assert capture.requests[keys[0]]["body"] == capture.requests[keys[1]]["body"]
    records = pilot.index_results([response(keys[1], "second"), response(keys[0], "first")], set(keys))
    replay = runner.ReplayAdapter(config, capture.requests, records)
    assert replay.generate(prompt="Open [URL_1]", system="Convert faithfully").text == "first"
    assert replay.generate(prompt="Open [URL_1]", system="Convert faithfully").text == "second"
    with pytest.raises(ValueError, match="drift or duplicate"):
        replay.generate(prompt="Open [URL_1]", system="Convert faithfully")


def test_no_paid_submission_above_estimate_allowance_or_modified_request(prepared):
    fake = FakeAPI(prepared)
    with pytest.raises(ValueError, match="exceeds"):
        api.submit(prepared, 0.000001, api=fake)
    assert fake.creates == fake.uploads == 0
    with (prepared / "requests.jsonl").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="file changed"):
        api.submit(prepared, 100, api=fake)
    assert fake.creates == 0


def test_uncertain_submission_requires_verified_recovery_not_paid_retry(prepared):
    fake = FakeAPI(prepared, fail_create=True)
    with pytest.raises(RuntimeError, match="outcome is unknown"):
        api.submit(prepared, 100, api=fake)
    with pytest.raises(ValueError, match="outcome is unknown"):
        api.submit(prepared, 100, api=fake)
    assert fake.creates == 1
    fake.batch["metadata"]["pilot_contract"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        api.status(prepared, "batch-1", api=fake)
    fake.batch["metadata"]["pilot_contract"] = digest(fake.contract)
    assert api.status(prepared, "batch-1", api=fake)["status"] == "completed"
    assert api.submit(prepared, 100, api=fake)["status"] == "already_submitted"
    assert fake.creates == 1


def test_real_canonical_stage3_and_training_admission_collect_idempotently(prepared):
    fake = FakeAPI(prepared)
    api.submit(prepared, 100, api=fake)
    report = api.collect(prepared, api=fake)
    assert report["accepted"] == 1, report
    assert report["training_input_ready"] is False
    accepted = runner.read_rows(prepared / "pilot_accepted.jsonl")[0]
    assert accepted["response_text"] == SOURCE
    assert accepted["id"] == "train-a"
    assert accepted["metadata"]["openai_stage3_batch"]["original_split"] == "train"
    assert api.collect(prepared) == report  # No client/key required for completed collection.
    assert fake.downloads == fake.creates == 1
    assert report["usage"]["output_tokens"] == 3000  # Reasoning not double-counted.
    assert report["usage"]["reasoning_tokens"] == 1000


@pytest.mark.parametrize("text", [GOOD.replace("$120", "$80"), GOOD.replace('ab=Text("Breakfast: not included")', 'ab=Text("Breakfast: included")'),
                                  '<a2ui>root=Text("Alder and Birch")</a2ui>', "not a valid program"])
def test_fidelity_failures_never_enter_accepted_output(prepared, text):
    fake = FakeAPI(prepared, text=text)
    api.submit(prepared, 100, api=fake)
    report = api.collect(prepared, api=fake)
    assert report["accepted"] == 0
    assert report["rejected_or_missing"] == 1
    assert report["usage"]["output_tokens"] == 3000
    assert not (prepared / "pilot_accepted.jsonl").read_bytes()


@pytest.mark.parametrize("mutation", ["incomplete", "refusal", "wrong_model", "missing"])
def test_incomplete_refused_wrong_model_missing_results_fail_closed(prepared, mutation):
    fake = FakeAPI(prepared)
    body = fake.records[0]["response"]["body"]
    if mutation == "incomplete":
        body["status"] = "incomplete"
    elif mutation == "refusal":
        body["output"][0]["content"] = [{"type": "refusal", "refusal": "No"}]
    elif mutation == "wrong_model":
        body["model"] = "other-model"
    else:
        fake.records = []
    api.submit(prepared, 100, api=fake)
    assert api.collect(prepared, api=fake)["accepted"] == 0


def test_out_of_order_ids_and_unknown_duplicate_results():
    indexed = pilot.index_results([response("b"), response("a")], {"a", "b"})
    assert indexed["a"]["custom_id"] == "a"
    for records in ([response("a"), response("a")], [response("unknown")]):
        with pytest.raises(ValueError, match="Unknown or duplicate"):
            pilot.index_results(records, {"a"})


def test_prompt_code_drift_blocks_submission(prepared, monkeypatch):
    monkeypatch.setattr(pilot, "fingerprint", dict)
    with pytest.raises(ValueError, match="code/prompt/schema"):
        pilot.load(prepared)


@pytest.mark.parametrize("destination,accepted", [("[URL_1]", 1), ("[URL_2]", 0)])
def test_reference_masking_and_strict_binding(tmp_path, destination, accepted):
    source = "Action: [Button: Open] [ACTION_URL_1]"
    directory = tmp_path / "reference-input"
    directory.mkdir()
    original = row("link", source)
    original["metadata"]["url_preprocessing"] = {"url_map": {"[ACTION_URL_1]": {"url": "https://example.com/guide"}}}
    runner.write_rows(directory / "train.jsonl", [original])
    runner.write_rows(directory / "val.jsonl", [row("val", "Tomorrow: rain.", "val")])
    output = tmp_path / "references"
    pilot.prepare(directory, output, samples=1)
    request = runner.read_rows(output / "requests.jsonl")[0]
    user = request["body"]["input"][-1]["content"]
    assert "https://example.com/guide" not in user
    assert "[URL_1]" in user
    assert "https://example.com/guide" in (output / "responses.jsonl").read_text()
    # A valid binding round-trips; a fabricated placeholder fails the SAME gate.
    fake = FakeAPI(output, text=f'<a2ui>root=Button("Open",onPress=openUrl("{destination}"))</a2ui>')
    api.submit(output, 100, api=fake)
    report = api.collect(output, api=fake)
    assert report["accepted"] == accepted, report
    if accepted:
        updated = runner.read_rows(output / "pilot_accepted.jsonl")[0]
        assert updated["response_text"] == source
        assert "[ACTION_URL_1]" in updated["completion"]
        assert "https://" not in updated["completion"]


def test_environment_restored_on_error(monkeypatch):
    monkeypatch.setenv("STAGE3_PROMPT_MODE", "inline")
    import os
    with pytest.raises(RuntimeError), runner.offline_environment():
        assert os.environ["STAGE3_PROMPT_MODE"] == "system_prefix"
        raise RuntimeError("test")
    assert os.environ["STAGE3_PROMPT_MODE"] == "inline"


def test_reported_usage_includes_failed_requests_without_double_counting_reasoning():
    records = [response("good"), response("truncated", status="incomplete"),
               {"custom_id": "error", "response": None, "error": {"code": "batch_expired"}}]
    report = api.usage_report(records, {"model": "gpt-6.1-sol"})
    assert report["input_tokens"] == 20000
    assert report["output_tokens"] == 6000
    assert report["reasoning_tokens"] == 2000
    assert report["records_without_usage"] == 1
    assert report["batch_cost_usd_from_reported_tokens_estimated"] == pytest.approx(0.0462)


def test_expired_partial_batch_keeps_valid_results_and_reports_missing(tmp_path):
    directory = inputs(tmp_path, [row("a"), row("b", "Rain expected tomorrow.")])
    output = tmp_path / "partial"
    pilot.prepare(directory, output, samples=2)
    fake = FakeAPI(output)
    rows = {item["response_id"]: item for item in runner.read_rows(output / "selected.jsonl")}
    good_id = next(key for key, identity in fake.contract["bindings"].items() if rows[identity]["source_id"] == "a")
    fake.records = [response(good_id)]
    api.submit(output, 100, api=fake)
    fake.batch["status"] = "expired"
    report = api.collect(output, api=fake)
    assert (report["accepted"], report["missing_results"], report["batch_status"]) == (1, 1, "expired")
    assert report["training_input_ready"] is False


def test_collection_detects_modified_download_and_completed_evidence(prepared):
    fake = FakeAPI(prepared)
    api.submit(prepared, 100, api=fake)
    api.collect(prepared, api=fake)
    with (prepared / "output_file_id.jsonl").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="evidence changed"):
        api.collect(prepared, api=fake)
    assert fake.downloads == 1


def test_real_openai_sdk_batch_and_file_contract_offline(prepared):
    import httpx
    from openai import OpenAI

    contract = pilot.load(prepared)
    events = []
    remote = {"id": "batch-sdk", "object": "batch", "status": "completed", "endpoint": "/v1/responses",
              "input_file_id": "file-sdk", "completion_window": "24h", "created_at": 0,
              "metadata": {"pilot_contract": digest(contract)}, "output_file_id": "out-sdk"}

    def handle(request):
        events.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path == "/v1/files":
            return httpx.Response(200, json={"id": "file-sdk", "object": "file", "bytes": 1,
                                  "created_at": 0, "filename": "requests.jsonl", "purpose": "batch"})
        if request.method == "POST" and request.url.path == "/v1/batches":
            payload = json.loads(request.content)
            assert payload["endpoint"] == "/v1/responses"
            assert payload["completion_window"] == "24h"
            assert payload["metadata"]["pilot_contract"] == digest(contract)
            return httpx.Response(200, json=remote)
        if request.url.path == "/v1/batches/batch-sdk":
            return httpx.Response(200, json=remote)
        if request.url.path == "/v1/files/out-sdk/content":
            return httpx.Response(200, content=canonical(response(next(iter(contract["bindings"])))) + b"\n")
        raise AssertionError(f"Unexpected SDK request: {request.method} {request.url.path}")

    with OpenAI(api_key="offline-fixture", base_url="https://api.openai.com/v1", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handle))) as sdk:
        api.submit(prepared, 100, api=sdk)
        assert api.collect(prepared, api=sdk)["accepted"] == 1
    assert events.count(("POST", "/v1/batches")) == 1
