"""Sealed tiny archives exercise offline reassessment without paid transport."""
from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "dataset/src"), str(ROOT / "training/src")]

from pipeline import openai_stage3_rescore as rescore
from pipeline.v11_regeneration_io import canonical, checked, digest, file_hash, save

REAL_ASSESS = rescore.guard.assess_candidate


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")


@pytest.fixture
def archive(tmp_path, monkeypatch):
    monkeypatch.setattr(socket, "create_connection", lambda *a, **kw: pytest.fail("Networking forbidden"))
    monkeypatch.setattr(rescore.pilot, "load", lambda *a: pytest.fail("Historical pilot.load forbidden"))
    monkeypatch.setattr(rescore.runner, "run", lambda *a, **kw: pytest.fail("Stage 3 replay forbidden"))
    monkeypatch.setattr(rescore.pilot, "fingerprint", lambda: {"current.py": "b" * 64})
    monkeypatch.setattr(rescore.guard, "guard_identity", lambda: {"current": "guard"})
    calls = []

    def assess(row, target, *, pilot_fidelity_v2=False):
        calls.append((row["id"], target, pilot_fidelity_v2))
        return {"eligible": pilot_fidelity_v2, "reasons": [] if pilot_fidelity_v2 else ["legacy_gate"],
                **({"score_v5_4": 97.0} if pilot_fidelity_v2 else {})}

    monkeypatch.setattr(rescore.guard, "assess_candidate", assess)
    root = tmp_path / "pilot"
    root.mkdir()
    source = "Meeting at 11:00."
    target = '<a2ui>root=Text("Meeting at 11:00.")</a2ui>'
    row = {"id": "source-1", "response_text": source, "completion": target}
    request_body = {"model": "gpt-6.1-sol"}
    custom_id = rescore.runner.request_id(request_body) + "-0000"
    response_id = "p-" + digest(row)[:24]
    entry = {"source_id": row["id"], "response_id": response_id, "row": row,
             "original_split": "train", "selection_bucket": "test", "line": 1,
             "original_row_sha256": digest(row)}
    write(root, "selected.jsonl", entry)
    write(root, "requests.jsonl", {"custom_id": custom_id, "method": "POST", "url": "/v1/responses", "body": request_body})
    write(root, "responses.jsonl", {"response_id": response_id, "query_id": response_id, "response_text": source})
    write(root, "queries.jsonl", {"query_id": response_id, "query_text": "Render faithfully"})
    contract = {"version": rescore.pilot.VERSION, "config": {"samples": 1, "model": "gpt-6.1-sol"},
                "bindings": {custom_id: response_id}, "code_hashes": {"old.py": "a" * 64},
                "guard": {"archived": "guard"},
                "files": {n: file_hash(root / n) for n in ("selected.jsonl", "requests.jsonl", "responses.jsonl", "queries.jsonl")}}
    save(root / "contract.json", contract)
    save(root / "submission.json", {"batch_id": "batch-1", "contract_sha256": digest(contract)})
    write(root, "report.json", {"contract_sha256": digest(contract), "stage3_artifacts": "v-1/stage3"})
    write(root, "assessments.jsonl", {"source_id": "source-1", "custom_id": custom_id, "eligible": False, "reasons": ["legacy_gate"]})
    write(root, "output_file_id.jsonl", {"custom_id": custom_id, "response": {"status_code": 200,
          "body": {"model": "gpt-6.1-sol", "status": "completed", "output": [{"type": "message",
                   "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": target}]}]}}})
    write(root, "v-1/stage3/genui.jsonl", {"response_id": response_id, "query_id": response_id,
          "response_text": source, "a2ui_express": target, "genui_raw_completion": target, "source_format": "a2ui_express_v1",
          "record_status": "accepted", "validation": {"schema_valid_strict": True, "standard_a2ui_valid": True},
          "gen": {"completion_complete": True},
          "training_acceptance": {"eligible": False, "blocking_reasons": [], "review_reasons": ["old_quality"]}})
    seal(root)
    return root, calls


def seal(root):
    contract = checked(root / "contract.json")
    files = {p.relative_to(root).as_posix(): file_hash(p) for p in root.rglob("*")
             if p.is_file() and p.name not in {"contract.json", "submission.json", "collection.json"}
             and p.relative_to(root).as_posix() not in contract["files"]}
    save(root / "collection.json", {"batch_id": "batch-1", "contract_sha256": digest(contract), "files": files})


def chatgpt_archive(root):
    from pipeline import chatgpt_stage3
    contract = checked(root / "contract.json")
    contract["config"]["transport"] = "chatgpt"
    save(root / "contract.json", contract)
    identity = digest(contract)
    request = rescore._rows(root / "requests.jsonl")[0]
    identifier = request["custom_id"]
    target = rescore._rows(root / "v-1/stage3/genui.jsonl")[0]["genui_raw_completion"]
    record_path = root / "chatgpt_records" / (identifier + ".json")
    record_path.parent.mkdir()
    save(record_path, {"custom_id": identifier, "contract_sha256": identity,
        "request_sha256": digest(request), "transport": "chatgpt", "status": "completed",
        "model": contract["config"]["model"], "thread_id": "thread-1", "turn_id": "turn-1", "text": target})
    save(root / "generation.json", {"version": chatgpt_stage3.VERSION, "contract_sha256": identity,
        "status": "validated", "client_version": "historical-runtime",
        "attempts": {identifier: {"status": "completed", "record_sha256": file_hash(record_path)}}})
    write(root, "report.json", {"contract_sha256": identity, "stage3_artifacts": "v-1/stage3"})
    chatgpt_seal(root)
    return identifier, record_path


def chatgpt_seal(root):
    from pipeline import chatgpt_stage3
    contract = checked(root / "contract.json")
    files = {p.relative_to(root).as_posix(): file_hash(p) for p in root.rglob("*")
             if p.is_file() and p.name not in {"contract.json", "submission.json", "collection.json",
                                              "chatgpt_collection.json"}
             and p.relative_to(root).as_posix() not in contract["files"]}
    save(root / "chatgpt_collection.json", {"version": chatgpt_stage3.VERSION,
        "contract_sha256": digest(contract), "files": files})


def test_chatgpt_rescore_is_offline_and_preserves_all_sealed_bytes(archive, monkeypatch):
    from pipeline import chatgpt_stage3
    root, calls = archive
    chatgpt_archive(root)
    monkeypatch.setattr(chatgpt_stage3, "load", lambda *a: pytest.fail("No live fingerprint/load"))
    monkeypatch.setattr(chatgpt_stage3, "run", lambda *a: pytest.fail("No generation"))
    monkeypatch.setattr(chatgpt_stage3, "collect", lambda *a: pytest.fail("No replay"))
    before = rescore._snapshot(root)
    summary = rescore.rescore(root, root.with_name("rescored"))
    assert summary["transport"] == "chatgpt" and summary["batch_id"] is None
    assert summary["overall"]["canonical_valid_candidates"] == summary["overall"]["v2_accepted"] == 1
    assert len(calls) == 3 and rescore._snapshot(root) == before


@pytest.mark.parametrize("change", [
    {"request_sha256": "f" * 64}, {"contract_sha256": "f" * 64},
    {"model": "wrong-model"}, {"thread_id": None}, {"custom_id": "other"},
    {"status": "unknown"}, {"status": "skipped"},
])
def test_resealed_chatgpt_binding_or_status_mutations_fail(archive, change):
    root, calls = archive
    identifier, path = chatgpt_archive(root)
    record = checked(path)
    save(path, {**record, **change})
    state = checked(root / "generation.json")
    state["attempts"][identifier]["record_sha256"] = file_hash(path)
    save(root / "generation.json", state)
    chatgpt_seal(root)
    with pytest.raises(ValueError):
        rescore.rescore(root, root.with_name("rescored"))
    assert not calls


@pytest.mark.parametrize("mutation", ["missing_attempt", "no_hash", "unsealed_record", "unsealed_state", "tampered_record"])
def test_chatgpt_missing_or_unsealed_evidence_fails_before_scoring(archive, mutation):
    root, calls = archive
    identifier, path = chatgpt_archive(root)
    state = checked(root / "generation.json")
    if mutation == "missing_attempt":
        state["attempts"] = {}
    if mutation == "no_hash":
        state["attempts"][identifier].pop("record_sha256")
    save(root / "generation.json", state)
    chatgpt_seal(root)
    receipt = checked(root / "chatgpt_collection.json")
    if mutation == "unsealed_record":
        receipt["files"].pop(path.relative_to(root).as_posix())
    if mutation == "unsealed_state":
        receipt["files"].pop("generation.json")
    save(root / "chatgpt_collection.json", receipt)
    if mutation == "tampered_record":
        path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        rescore.rescore(root, root.with_name("rescored"))
    assert not calls


def test_chatgpt_raw_output_cannot_be_replaced_by_canonical_target(archive):
    root, _ = archive
    identifier, path = chatgpt_archive(root)
    record = checked(path)
    record["text"] = '<a2ui>root=Text("Meeting at 12:00.")</a2ui>'
    save(path, record)
    state = checked(root / "generation.json")
    state["attempts"][identifier]["record_sha256"] = file_hash(path)
    save(root / "generation.json", state)
    chatgpt_seal(root)
    output = root.with_name("rescored")
    summary = rescore.rescore(root, output)
    assert summary["overall"]["v2_accepted"] == 0
    assert rescore._rows(output / "assessments.jsonl")[0]["v2"]["reasons"] == [
        "canonical_candidate_paid_output_mismatch"]


def test_rescore_preserves_archive_and_recomputes_old_quality(archive):
    root, calls = archive
    before = rescore._snapshot(root)
    output = root.with_name("rescored")
    summary = rescore.rescore(root, output)
    assert rescore._snapshot(root) == before
    assert summary["diagnostic_only"] and not summary["training_input_ready"]
    assert summary["overall"]["v2_accepted"] == summary["overall"]["baseline_v2_accepted"] == 1
    assert summary["overall"]["legacy_accepted"] == 0
    assert summary["overall"]["official_to_v2"] == {"rejected->accepted": 1}
    assert len(calls) == 3 and [c[2] for c in calls] == [True, False, True]
    provenance = checked(output / "rescore.json")
    assert provenance["archived_code_hashes"] != provenance["current_code_hashes"]
    assert provenance["archive_hashes_before"] == provenance["archive_hashes_after"] == before
    row = rescore._rows(output / "assessments.jsonl")[0]
    assert "review:old_quality" in row["archived_admission_errors"]
    assert {p.name for p in output.iterdir()} == {"assessments.jsonl", "summary.json", "rescore.json"}


@pytest.mark.parametrize("change,reason", [
    ({"record_status": "format_rejected"}, "stage3_not_accepted"),
    ({"response_text": "Different source"}, "source_binding_mismatch"),
    ({"validation": {"schema_valid_strict": False, "standard_a2ui_valid": True}}, "schema_valid_strict"),
    ({"gen": {"completion_complete": False}}, "generation_failed_or_incomplete"),
    ({"a2ui_express": "invalid"}, ""),
])
def test_invalid_canonical_candidates_never_rescued(archive, change, reason):
    root, calls = archive
    name = "v-1/stage3/genui.jsonl"
    candidate = rescore._rows(root / name)[0]
    candidate.update(change)
    write(root, name, candidate)
    seal(root)
    output = root.with_name("rescored")
    summary = rescore.rescore(root, output)
    assert summary["overall"]["v2_accepted"] == 0
    assert len(calls) == 1  # Only the original label is scored.
    assert reason in rescore._rows(output / "assessments.jsonl")[0]["v2"]["reasons"][0]


def test_missing_canonical_candidate_is_not_reconstructed(archive):
    root, calls = archive
    (root / "v-1/stage3/genui.jsonl").unlink()
    write(root, "v-1/stage3/error.json", {"error": "rejected"})
    seal(root)
    summary = rescore.rescore(root, root.with_name("rescored"))
    assert summary["overall"]["v2_reasons"] == {"canonical_stage3_rejected": 1}
    assert len(calls) == 1


@pytest.mark.parametrize("name", ["selected.jsonl", "output_file_id.jsonl", "v-1/stage3/genui.jsonl"])
def test_hash_tampering_is_rejected_before_scoring(archive, name):
    root, calls = archive
    with (root / name).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="Archive file changed"):
        rescore.rescore(root, root.with_name("rescored"))
    assert calls == []


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/escape", "v-1/../escape", "v-1\\escape"])
def test_manifest_paths_cannot_escape_archive(archive, name):
    root, calls = archive
    receipt = checked(root / "collection.json")
    receipt["files"][name] = "a" * 64
    save(root / "collection.json", receipt)
    with pytest.raises(ValueError, match="Unsafe archive path"):
        rescore.rescore(root, root.with_name("rescored"))
    assert calls == []


@pytest.mark.parametrize("name,key", [("assessments.jsonl", "source_id"), ("v-1/stage3/genui.jsonl", "response_id"), ("output_file_id.jsonl", "custom_id")])
def test_duplicate_identities_rejected_even_when_resealed(archive, name, key):
    root, calls = archive
    path = root / name
    path.write_bytes(path.read_bytes() * 2)
    seal(root)
    with pytest.raises(ValueError, match="duplicate|Duplicate"):
        rescore.rescore(root, root.with_name("rescored"))
    assert calls == []


def test_batch_mismatch_and_unsealed_stage_path_fail(archive):
    root, _ = archive
    receipt = checked(root / "collection.json")
    receipt["batch_id"] = "another-batch"
    save(root / "collection.json", receipt)
    with pytest.raises(ValueError, match="batch binding"):
        rescore.rescore(root, root.with_name("rescored"))
    seal(root)
    receipt = checked(root / "collection.json")
    del receipt["files"]["v-1/stage3/genui.jsonl"]
    save(root / "collection.json", receipt)
    with pytest.raises(ValueError, match="Unsealed Stage 3"):
        rescore.rescore(root, root.with_name("rescored"))


def test_archive_change_during_scoring_prevents_publication(archive, monkeypatch):
    root, _ = archive
    original = rescore.guard.assess_candidate

    def mutate(*args, **kwargs):
        (root / "new-file").write_text("changed")
        return original(*args, **kwargs)

    monkeypatch.setattr(rescore.guard, "assess_candidate", mutate)
    output = root.with_name("rescored")
    with pytest.raises(ValueError, match="Archive changed during"):
        rescore.rescore(root, output)
    assert not output.exists()


def test_output_must_be_fresh_sibling(archive):
    root, calls = archive
    for output in (root, root / "analysis", root.parent):
        with pytest.raises(ValueError, match="fresh sibling"):
            rescore.rescore(root, output)
    assert calls == []


def test_actual_guard_and_reference_binding_on_tiny_archive(archive, monkeypatch):
    root, _ = archive
    monkeypatch.setattr(rescore.guard, "assess_candidate", REAL_ASSESS)
    summary = rescore.rescore(root, root.with_name("rescored"))
    assert summary["overall"]["v2_accepted"] == 1
    assert summary["overall"]["baseline_v2_scores"]["count"] == 1


@pytest.mark.parametrize("name", ["contract.json", "submission.json", "collection.json"])
def test_broken_seals_rejected(archive, name):
    root, calls = archive
    envelope = json.loads((root / name).read_text())
    envelope["sha256"] = "0" * 64
    write(root, name, envelope)
    with pytest.raises(ValueError, match="Journal integrity"):
        rescore.rescore(root, root.with_name("rescored"))
    assert calls == []


def test_duplicate_json_keys_are_not_hidden_by_parser(archive):
    root, _ = archive
    path = root / "contract.json"
    path.write_text('{"payload":{},"payload":{},"sha256":"bad"}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        rescore.rescore(root, root.with_name("rescored"))


@pytest.mark.parametrize("change", [{"source_id": "wrong"}, {"response_id": "p-wrong"}, {"original_split": "val"}, {"line": False}])
def test_resealed_malformed_selection_is_rejected(archive, change):
    root, calls = archive
    entry = rescore._rows(root / "selected.jsonl")[0]
    entry.update(change)
    write(root, "selected.jsonl", entry)
    contract = checked(root / "contract.json")
    contract["files"]["selected.jsonl"] = file_hash(root / "selected.jsonl")
    save(root / "contract.json", contract)
    save(root / "submission.json", {"batch_id": "batch-1", "contract_sha256": digest(contract)})
    write(root, "report.json", {"contract_sha256": digest(contract), "stage3_artifacts": "v-1/stage3"})
    seal(root)
    with pytest.raises(ValueError):
        rescore.rescore(root, root.with_name("rescored"))
    assert calls == []


@pytest.mark.parametrize("also_replace_raw_copy", [False, True])
def test_substituted_paid_output_cannot_credit_an_unrelated_candidate(archive, also_replace_raw_copy):
    root, calls = archive
    raw = rescore._rows(root / "output_file_id.jsonl")[0]
    substituted = '<a2ui>root=Text("Meeting at 12:00.")</a2ui>'
    raw["response"]["body"]["output"][0]["content"][0]["text"] = substituted
    write(root, "output_file_id.jsonl", raw)
    if also_replace_raw_copy:
        candidate = rescore._rows(root / "v-1/stage3/genui.jsonl")[0]
        candidate["genui_raw_completion"] = substituted
        write(root, "v-1/stage3/genui.jsonl", candidate)
    seal(root)
    output = root.with_name("rescored")
    summary = rescore.rescore(root, output)
    assert summary["overall"]["v2_accepted"] == 0
    assert summary["overall"]["canonical_valid_candidates"] == 0
    reason = "canonical_candidate_paid_semantics_mismatch" if also_replace_raw_copy else "canonical_candidate_paid_output_mismatch"
    assert summary["overall"]["v2_reasons"] == {reason: 1}
    assert len(calls) == 1  # Only original label, never substituted candidate.


@pytest.mark.parametrize("change", [{"method": "GET"}, {"url": "/v1/other"},
                                    {"body": {"model": "another-model"}},
                                    {"body": {"model": "gpt-6.1-sol", "input": "substituted"}}])
def test_resealed_request_mutation_does_not_reuse_original_identity(archive, change):
    root, calls = archive
    request = rescore._rows(root / "requests.jsonl")[0]
    request.update(change)
    write(root, "requests.jsonl", request)
    contract = checked(root / "contract.json")
    contract["files"]["requests.jsonl"] = file_hash(root / "requests.jsonl")
    save(root / "contract.json", contract)
    save(root / "submission.json", {"batch_id": "batch-1", "contract_sha256": digest(contract)})
    write(root, "report.json", {"contract_sha256": digest(contract), "stage3_artifacts": "v-1/stage3"})
    seal(root)
    with pytest.raises(ValueError, match="request transport or identity"):
        rescore.rescore(root, root.with_name("rescored"))
    assert calls == []


def test_unknown_archived_pilot_profile_is_rejected(archive):
    root, calls = archive
    contract = checked(root / "contract.json")
    contract["config"]["pilot_profile"] = "unknown"
    save(root / "contract.json", contract)
    with pytest.raises(ValueError, match="Unknown OpenAI pilot profile"):
        rescore.rescore(root, root.with_name("rescored"))
    assert calls == []
