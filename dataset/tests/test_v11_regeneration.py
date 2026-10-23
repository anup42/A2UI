"""Offline regeneration lifecycle: no Muse server, training or benchmark labels."""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "dataset/src"), str(ROOT / "training/src")]

from ir_training.data.express_preparation import TASK_PREFIX, prepare_row
from pipeline import v11_regeneration as api
from pipeline.v11_regeneration_io import SplitInput, canonical, checked, file_hash

SOURCE = "- Alder; Price: $120; Breakfast: not included\n- Birch; Price: $80; Breakfast: included"
BAD = '<a2ui>\nroot=Text(' + json.dumps(SOURCE) + ')\n</a2ui>'
GOOD = '''<a2ui>
root=Column([a,b])
a=Card([ap,ab],title="Alder")
ap=Text("Price: $120")
ab=Text("Breakfast: not included")
b=Card([bp,bb],title="Birch")
bp=Text("Price: $80")
bb=Text("Breakfast: included")
</a2ui>'''


def make_row(row_id, split="train", *, bad=False, held=False):
    target = BAD if bad else GOOD
    return {"id": row_id, "response_text": SOURCE, "completion": target,
            "completion_targets": {"a2ui_express_v1": target}, "a2ui_express": target,
            "messages": [{"role": "system", "content": "Render faithfully."},
                         {"role": "user", "content": TASK_PREFIX + SOURCE},
                         {"role": "assistant", "content": target}],
            "metadata": {"assigned_split": split, "query_text": "Compare the hotels.", "test_hold": held}}


def stage3_record(response, target=GOOD):
    return {"response_id": response["response_id"], "response_text": response["response_text"],
            "a2ui_express": target, "record_status": "accepted", "source_format": "a2ui_express_v1",
            "training_acceptance": {"eligible": True, "blocking_reasons": [], "review_reasons": []},
            "validation": {"schema_valid_strict": True, "standard_a2ui_valid": True},
            "gen": {"error": None, "completion_complete": True}}


@pytest.fixture
def run(tmp_path, monkeypatch):
    source = tmp_path / "input"
    source.mkdir()
    rows = {"train": [make_row("change-1", bad=True), make_row("preserve-1")],
            "val": [make_row("preserve-val", "val")]}
    for split, values in rows.items():
        (source / f"{split}.jsonl").write_bytes(b"".join(canonical(row) + b"\n" for row in values))
    output = tmp_path / "output"
    calls = []
    guard = SimpleNamespace(
        guard_identity=lambda: {"version": "test-guard"}, source_text=lambda row: row["response_text"],
        assess_source=lambda row: {"eligible": not row["metadata"].get("test_hold"), "reasons": ["known_source_defect"]},
        assess_candidate=lambda row, target: {"eligible": target != BAD, "reasons": ["entity_fields_not_structured"] if target == BAD else []})
    monkeypatch.setattr(api, "_guard", lambda: guard)
    monkeypatch.setattr(api, "_benchmark_exclusions", lambda: (set(), set(), {}))

    def runner(**kwargs):
        calls.append(kwargs)
        values = [json.loads(line) for line in kwargs["responses_path"].read_text().splitlines()]
        kwargs["genui_path"].write_bytes(b"".join(canonical(stage3_record(row)) + b"\n" for row in values))

    def generate(**kwargs):
        return api.regenerate(source, output, adapter=object(), stage3_runner=kwargs.pop("stage3_runner", runner), **kwargs)
    return SimpleNamespace(source=source, output=output, rows=rows, calls=calls, generate=generate, runner=runner, guard=guard)


def test_default_plan_is_read_only_and_no_teacher(run):
    report = run.generate()
    assert report["generation_started"] is False
    assert report["selected"] == 1
    assert not run.output.exists()
    assert not run.calls


def test_execute_preserves_splits_ids_unaffected_bytes_and_preparation_contract(run):
    before = {name: (run.source / name).read_bytes() for name in ("train.jsonl", "val.jsonl")}
    report = run.generate(execute=True)
    assert report["counts"]["accepted"] == 1
    assert report["training_input_ready"] is True
    actual = (run.output / "train.jsonl").read_bytes().splitlines(keepends=True)
    assert actual[1] == before["train.jsonl"].splitlines(keepends=True)[1]
    assert (run.output / "val.jsonl").read_bytes() == before["val.jsonl"]
    changed = json.loads(actual[0])
    assert changed["id"] == "change-1"
    assert changed["response_text"] == SOURCE
    assert changed["messages"][:-1] == run.rows["train"][0]["messages"][:-1]
    prepared, target, _ = prepare_row(changed, "root-first")
    assert prepared["completion"] == target.text == changed["completion"]
    assert changed["canonical_graph"] == target.graph
    assert all((run.source / name).read_bytes() == raw for name, raw in before.items())
    assert run.calls[0]["prompt_supplement_path"] == api.SUPPLEMENT
    assert run.calls[0]["max_attempts"] == 1


def test_completed_resume_reuses_work_with_no_scoring_or_teacher(run, monkeypatch):
    first = run.generate(execute=True)
    snapshot = (run.output / "train.jsonl").read_bytes()
    monkeypatch.setattr(api, "_scan", lambda *_: pytest.fail("completed resume rescanned guards"))
    second = run.generate(execute=True, resume=True)
    assert first == second
    assert len(run.calls) == 1
    assert (run.output / "train.jsonl").read_bytes() == snapshot


@pytest.mark.parametrize("field,value", [("seed", 124), ("max_attempts", 2), ("max_tokens", 11000), ("batch_size", 2)])
def test_resume_rejects_changed_configuration(run, field, value):
    run.generate(execute=True)
    with pytest.raises(ValueError, match="Resume contract mismatch"):
        run.generate(execute=True, resume=True, **{field: value})
    assert len(run.calls) == 1


def test_resume_rejects_changed_input(run):
    run.generate(execute=True)
    path = run.source / "train.jsonl"
    path.write_bytes(path.read_bytes().replace(b"Compare the hotels", b"Compare these hotels"))
    with pytest.raises(ValueError, match="source changed"):
        run.generate(execute=True, resume=True)


@pytest.mark.parametrize("artifact", ["train.jsonl", "quarantine.jsonl", "stage3"])
def test_resume_rejects_changed_published_or_generation_evidence(run, artifact):
    run.generate(execute=True)
    path = next((run.output / "attempts").glob("*/genui.jsonl")) if artifact == "stage3" else run.output / artifact
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        run.generate(execute=True, resume=True)


def test_bounded_failures_quarantine_never_fallback_to_old_target(run):
    def reject(**kwargs):
        run.calls.append(kwargs)
        rows = [json.loads(line) for line in kwargs["responses_path"].read_text().splitlines()]
        kwargs["genui_path"].write_bytes(b"".join(canonical(stage3_record(row, BAD)) + b"\n" for row in rows))
    report = run.generate(execute=True, stage3_runner=reject, max_attempts=2)
    assert report["counts"]["rejected"] == 1
    assert len(run.calls) == 2
    assert [json.loads(line)["id"] for line in (run.output / "train.jsonl").read_text().splitlines()] == ["preserve-1"]
    quarantined = json.loads((run.output / "quarantine.jsonl").read_text())
    assert quarantined["original_row"]["id"] == "change-1"
    assert len(quarantined["attempts"]) == 2
    feedback = json.loads(run.calls[1]["responses_path"].read_text())["stage3_repair_feedback"]
    assert feedback["review_reasons"] == ["entity_fields_not_structured"]
    run.generate(execute=True, resume=True, stage3_runner=reject, max_attempts=2)
    assert len(run.calls) == 2


def test_unsafe_source_is_quarantined_without_teacher_call(run):
    held = make_row("held", held=True)
    with (run.source / "train.jsonl").open("ab") as stream:
        stream.write(canonical(held) + b"\n")
    report = run.generate(execute=True)
    assert report["counts"]["source_held"] == 1
    assert len(run.calls) == 1
    assert "held" not in run.calls[0]["responses_path"].read_text()
    quarantine = json.loads((run.output / "quarantine.jsonl").read_text())
    assert quarantine["status"] == "source_held"
    assert quarantine["original_row"] == held


def test_interrupted_batch_recovers_successful_outputs_after_partial_journal_save(run, monkeypatch):
    with (run.source / "train.jsonl").open("ab") as stream:
        stream.write(canonical(make_row("change-2", bad=True)) + b"\n")
    real_save = api.save
    failed = False

    def crash(path, value):
        nonlocal failed
        if path.parent.name == "journal" and value.get("status") == "accepted" and value["entry"]["id"] == "change-2" and not failed:
            failed = True
            raise KeyboardInterrupt("simulated process death")
        return real_save(path, value)
    monkeypatch.setattr(api, "save", crash)
    with pytest.raises(KeyboardInterrupt):
        run.generate(execute=True)
    assert len(run.calls) == 1
    monkeypatch.setattr(api, "save", real_save)
    report = run.generate(execute=True, resume=True)
    assert report["counts"]["accepted"] == 2
    assert len(run.calls) == 1


def test_interrupted_before_teacher_response_consumes_bounded_attempt_then_retries(run):
    def crash(**kwargs):
        run.calls.append(kwargs)
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run.generate(execute=True, stage3_runner=crash)
    report = run.generate(execute=True, resume=True)
    assert report["counts"]["accepted"] == 1
    state = checked(next((run.output / "journal").glob("*.json")))
    assert len(state["attempts"]) == 2
    assert len(run.calls) == 2


def test_pilot_cannot_masquerade_as_complete_training_input(run):
    report = run.generate(execute=True, max_samples=1)
    assert report["status"] == "pilot_partial"
    assert report["training_input_ready"] is False
    assert not (run.output / "train.jsonl").exists()
    assert not (run.output / "val.jsonl").exists()
    assert (run.output / "pilot_train.jsonl").exists()


def test_duplicate_split_ids_benchmark_ids_and_context_overflow_fail_before_output(run):
    with (run.source / "val.jsonl").open("ab") as stream:
        stream.write(canonical(make_row("BXP-024", "val")) + b"\n")
    with pytest.raises(ValueError, match="Golden/Bixby"):
        run.generate(execute=True)
    assert not run.output.exists()
    with pytest.raises(ValueError, match="server context"):
        run.generate(context_tokens=16000)


def test_reference_restoration_then_exact_rebinding_preserves_original_source():
    source = "Action: [Button: Open] [ACTION_URL_1]"
    row = {"response_text": source, "metadata": {"url_preprocessing": {"url_map": {
        "[ACTION_URL_1]": {"url": "https://example.org/a?x=1&y=2"}}}}}
    assert api._generation_source(row) == "Action: [Button: Open] https://example.org/a?x=1&y=2"
    target = '<a2ui>\nroot=Button("Open",onPress=openUrl("https://example.org/a?x=1&y=2"))\n</a2ui>'
    rebound = api._candidate_target(row, {"a2ui_express": target})
    assert "[ACTION_URL_1]" in rebound and "https://" not in rebound
    assert row["response_text"] == source


def test_fresh_output_refusal_and_process_lock(run):
    run.output.mkdir()
    with pytest.raises(ValueError, match="new output"):
        run.generate(execute=True)
    with api._exclusive(run.output), pytest.raises(ValueError, match="Another regeneration"), api._exclusive(run.output):
        pytest.fail("lock unexpectedly acquired")


def test_bundle_reader_preserves_utf8_and_lines_across_part_boundaries(tmp_path):
    files = []
    expected = {}
    for split in ("train", "val"):
        raw = canonical({"id": split, "text": "café"}) + b"\n" + canonical({"id": split + "2"}) + b"\n"
        expected[split] = raw
        cut = raw.index("é".encode()) + 1
        parts = []
        for number, part in enumerate((raw[:cut], raw[cut:])):
            name = f"{split}.part-{number:05d}.gz"
            compressed = gzip.compress(part)
            (tmp_path / name).write_bytes(compressed)
            parts.append({"name": name, "bytes": len(compressed), "raw_bytes": len(part),
                          "sha256": hashlib.sha256(compressed).hexdigest()})
        files.append({"name": f"{split}.jsonl", "bytes": len(raw), "rows": 2,
                      "sha256": hashlib.sha256(raw).hexdigest(), "parts": parts})
    manifest = {"dataset": "test", "outputs": {item["name"]: item for item in files}, "rows": {"train": 2, "val": 2}}
    (tmp_path / "manifest.json").write_bytes(canonical(manifest))
    (tmp_path / "prompt_scaffold.json").write_bytes(b"{}")
    sidecars = [{"name": name, "bytes": (tmp_path / name).stat().st_size, "sha256": file_hash(tmp_path / name)}
                for name in ("manifest.json", "prompt_scaffold.json")]
    (tmp_path / "bundle.json").write_bytes(canonical({"format": "a2ui-training-splits-gzip-v2",
        "dataset": "test", "files": files, "sidecars": sidecars}))
    reader = SplitInput(tmp_path)
    for split, raw in expected.items():
        assert b"".join(line for _, line, _ in reader.rows(split)) == raw
    (tmp_path / "train.part-00000.gz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        list(reader.rows("train"))


def test_real_guard_selects_whole_entity_blob_and_accepts_extracted_fields(run, monkeypatch):
    from pipeline import v11_regeneration_guard
    monkeypatch.setattr(api, "_guard", lambda: v11_regeneration_guard)
    report = run.generate(execute=True)
    assert report["counts"]["selected"] == report["counts"]["accepted"] == 1


def test_benchmark_source_identity_cannot_be_hidden_by_renaming(run, monkeypatch):
    monkeypatch.setattr(api, "_benchmark_exclusions", lambda: (set(), {api.digest(" ".join(SOURCE.split()))}, {}))
    with pytest.raises(ValueError, match="Pinned Golden/Bixby source"):
        run.generate(execute=True)
    assert not run.output.exists()


def test_published_manifest_edit_is_not_silently_overwritten(run):
    run.generate(execute=True)
    path = run.output / "manifest.json"
    value = json.loads(path.read_text())
    value["counts"]["accepted"] = 999
    path.write_bytes(canonical(value) + b"\n")
    with pytest.raises(ValueError, match="Published manifest changed"):
        run.generate(execute=True, resume=True)


def test_completed_run_missing_candidate_proof_fails_closed(run):
    run.generate(execute=True)
    path = next((run.output / "journal").glob("*.json"))
    path.unlink()
    with pytest.raises(ValueError, match="journal is missing"):
        run.generate(execute=True, resume=True)


def test_declared_restored_split_hash_is_enforced_before_generation(run):
    (run.source / "manifest.json").write_bytes(canonical({"outputs": {"train.jsonl": {"sha256": "0" * 64}}}))
    with pytest.raises(ValueError, match="input manifest"):
        run.generate(execute=True)
    assert not run.output.exists()


def test_duplicate_id_across_splits_is_rejected(run):
    (run.source / "val.jsonl").write_bytes(canonical(make_row("change-1", "val")) + b"\n")
    with pytest.raises(ValueError, match="Duplicate row id"):
        run.generate(execute=True)
    assert not run.output.exists()


def test_real_pinned_benchmark_exclusion_registry_supports_historical_masked_aliases():
    ids, sources, files = api._benchmark_exclusions()
    assert "BXP-024" in ids
    assert len(sources) >= 100
    assert len(files) == 3


def test_infrastructure_failure_keeps_completed_records_and_requires_explicit_resume(run):
    def failure(**kwargs):
        run.runner(**kwargs)
        raise RuntimeError("server unavailable after partial successful batch")
    with pytest.raises(RuntimeError, match="durable evidence"):
        run.generate(execute=True, stage3_runner=failure)
    assert not (run.output / "manifest.json").exists()
    report = run.generate(execute=True, resume=True)
    assert report["counts"]["accepted"] == 1
    assert len(run.calls) == 1


def test_input_modified_during_generation_is_not_published(run):
    def mutate(**kwargs):
        run.runner(**kwargs)
        path = run.source / "val.jsonl"
        path.write_bytes(path.read_bytes().replace(b"preserve-val", b"modified-val"))
    with pytest.raises(ValueError, match="Input changed"):
        run.generate(execute=True, stage3_runner=mutate)
    assert not (run.output / "manifest.json").exists()


def test_reference_aliases_do_not_break_source_or_benchmark_restoration():
    row = {"response_text": "[SOURCE_URL_1] [ACTION_URL_1]", "url_map": {
        "[SOURCE_URL_1]": "https://example.org/a", "[ACTION_URL_1]": "https://example.org/a"}}
    assert api._generation_source(row) == "https://example.org/a https://example.org/a"
    target = '<a2ui>\nroot=Button("Open",onPress=openUrl("https://example.org/a"))\n</a2ui>'
    with pytest.raises(ValueError, match="Ambiguous inverse"):
        api._candidate_target(row, {"a2ui_express": target})


@pytest.mark.parametrize("bad_record", [b"null\n", b"[]\n", b"{truncated\n", b'{"response_id":"foreign"}\n',
                                      b'{"response_id":[]}\n', b'{"response_id":"change-1","gen":"malformed"}\n'])
def test_malformed_and_unknown_stage3_records_retry_boundedly(run, bad_record):
    def corrupt_first(**kwargs):
        if not run.calls:
            run.calls.append(kwargs)
            kwargs["genui_path"].write_bytes(bad_record)
        else:
            run.runner(**kwargs)
    result = run.generate(execute=True, stage3_runner=corrupt_first)
    assert result["counts"]["accepted"] == 1
    assert len(run.calls) == 2


def test_kill_during_attempt_input_preparation_is_resumable_without_consuming_attempt(run, monkeypatch):
    original = api.atomic_bytes
    killed = False

    def interrupt(path, value):
        nonlocal killed
        if path.name == "responses.jsonl" and not killed:
            killed = True
            raise KeyboardInterrupt
        return original(path, value)
    monkeypatch.setattr(api, "atomic_bytes", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run.generate(execute=True)
    assert not run.calls
    monkeypatch.setattr(api, "atomic_bytes", original)
    result = run.generate(execute=True, resume=True)
    assert result["counts"]["accepted"] == 1
    assert len(run.calls) == 1
    assert len(checked(next((run.output / "journal").glob("*.json")))["attempts"]) == 1


def test_canonical_release_hold_catalogue_is_bound_and_cached_policy_refreshed():
    from ir_training.data.release_review import review_policy
    review_policy()
    review_policy()
    config = api._config(teacher_model=api.DEFAULT_TEACHER, max_samples=0, max_attempts=3,
                         batch_size=8, max_tokens=12288, prompt_max_tokens=16000,
                         context_tokens=32768, seed=123)
    path = "training/data/quality/v11_review_holds_20261001.json"
    assert config["code_prompt_schema_hashes"][path] == file_hash(ROOT / path)
    assert review_policy.cache_info().misses == 1
    assert review_policy.cache_info().hits == 0
