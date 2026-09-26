"""Offline recovery, reference binding, immutable evidence and legacy migration."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline import training_augmentation as api
from pipeline import training_augmentation_resume as engine
from test_training_augmentation_generation import FakeTeacher, stage3_row


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def run(tmp_path):
    donors = tmp_path / "donors.jsonl"
    donors.write_text("".join(json.dumps({"donor_id": f"id/{index}", "source_group_id": f"family:{index}",
        "split": "train", "response_text": f"Original {index}"}) + "\n" for index in range(3)), encoding="utf-8")
    output = tmp_path / "generated"
    calls = []

    def stage3(**kwargs):
        calls.append(kwargs)
        kwargs["genui_path"].write_text("".join(json.dumps(stage3_row(row)) + "\n" for row in api.read_jsonl(kwargs["responses_path"])), encoding="utf-8")

    def generate(**kwargs):
        return api.generate_training_augmentations(donors, output, max_new_samples=kwargs.pop("max_new_samples", 3),
            adapter=kwargs.pop("adapter", FakeTeacher()), stage3_runner=kwargs.pop("stage3_runner", stage3), **kwargs)
    return donors, output, calls, generate


def test_completed_resume_uses_zero_teacher_calls_and_identical_projection(run):
    _, output, calls, generate = run
    first = generate()
    before = (output / "accepted_genui.jsonl").read_bytes()
    teacher = FakeTeacher()
    second = generate(resume=True, adapter=teacher)
    assert teacher.calls == []
    assert len(calls) == 1
    assert second["attempted_rows"] == first["attempted_rows"] == 3
    assert (output / "accepted_genui.jsonl").read_bytes() == before


def test_rejected_labels_retry_new_stage3_directory_without_source_calls(run):
    _, output, calls, generate = run
    def reject_one(**kwargs):
        calls.append(kwargs)
        rows = [stage3_row(row) for row in api.read_jsonl(kwargs["responses_path"])]
        rows[0]["training_acceptance"]["review_reasons"] = ["content_unit_fidelity"]
        kwargs["genui_path"].write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert generate(stage3_runner=reject_one)["accepted_rows"] == 2
    preserved = (output / "stage3_attempts/000001/genui.jsonl").read_bytes()
    teacher = FakeTeacher()
    manifest = generate(resume=True, adapter=teacher)
    assert teacher.calls == []
    assert manifest["accepted_rows"] == manifest["attempted_rows"] == 3
    assert len(api.read_jsonl(calls[-1]["responses_path"])) == 1
    assert calls[0]["genui_path"] != calls[1]["genui_path"]
    assert (output / "stage3_attempts/000001/genui.jsonl").read_bytes() == preserved
    assert len({row["response_id"] for row in api.read_jsonl(output / "accepted_genui.jsonl")}) == 3


def test_interrupted_stage3_recovers_completed_prefix_and_retries_only_missing(run):
    _, output, calls, generate = run
    def interrupted(**kwargs):
        rows = api.read_jsonl(kwargs["responses_path"])
        kwargs["genui_path"].write_text(json.dumps(stage3_row(rows[0])) + "\n", encoding="utf-8")
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        generate(stage3_runner=interrupted)
    assert json.loads((output / "manifest.json").read_text())["status"] == "running"
    teacher = FakeTeacher()
    assert generate(resume=True, adapter=teacher)["accepted_rows"] == 3
    assert teacher.calls == []
    assert len(api.read_jsonl(calls[-1]["responses_path"])) == 2


def test_review_transport_failure_reuses_source(run):
    _, _, _, generate = run
    class InterruptedReview(FakeTeacher):
        def generate(self, **kwargs):
            if kwargs["prompt"].startswith("TASK: REVIEW_SOURCE"):
                raise KeyboardInterrupt
            return super().generate(**kwargs)
    with pytest.raises(KeyboardInterrupt):
        generate(adapter=InterruptedReview(), max_new_samples=1)
    teacher = FakeTeacher()
    assert generate(resume=True, adapter=teacher, max_new_samples=1)["accepted_rows"] == 1
    assert len(teacher.calls) == 1
    assert teacher.calls[0]["prompt"].startswith("TASK: REVIEW_SOURCE")


def test_rejected_source_regenerated_with_new_seed_and_attempts_preserved(run):
    _, output, _, generate = run
    with pytest.raises(RuntimeError):
        generate(adapter=FakeTeacher(review_override=lambda value: dict(value, coherent=False)), max_new_samples=1)
    artifacts = {str(path): path.read_bytes() for path in (output / "source_attempts").rglob("*.json")}
    teacher = FakeTeacher()
    manifest = generate(resume=True, adapter=teacher, max_new_samples=1)
    assert manifest["attempted_rows"] == manifest["accepted_rows"] == 1
    assert teacher.calls[0]["seed"] == 124
    assert all(Path(path).read_bytes() == data for path, data in artifacts.items())


@pytest.mark.parametrize("change", ["seed", "budget", "donors", "recipes", "projection", "journal", "attempt", "stage3_attempt"])
def test_resume_rejects_contract_or_evidence_drift_before_calls(run, change):
    donors, output, _, generate = run
    generate()
    teacher = FakeTeacher()
    kwargs = {"resume": True, "adapter": teacher}
    if change == "seed":
        kwargs["seed"] = 456
    elif change == "budget":
        kwargs["max_new_samples"] = 2
    elif change == "donors":
        donors.write_text(donors.read_text() + "\n", encoding="utf-8")
    elif change == "projection":
        (output / "accepted_genui.jsonl").write_text("", encoding="utf-8")
    elif change == "journal":
        next((output / "candidates").glob("*.json")).write_text("{}", encoding="utf-8")
    elif change == "attempt":
        next((output / "source_attempts").rglob("*.json")).write_text("{}", encoding="utf-8")
    elif change == "stage3_attempt":
        (output / "stage3_attempts/000001/genui.jsonl").write_text("{}", encoding="utf-8")
    with patch.dict(api.RECIPES, {api.CATEGORIES[0]: "changed"} if change == "recipes" else {}), pytest.raises(ValueError):
        generate(**kwargs)
    assert teacher.calls == []


def sidecar(donors, tmp_path):
    rows = api.read_jsonl(donors)
    rows[0]["response_text"] = "Original: Source [URL_1]"
    donors.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    path = tmp_path / "references.json"
    write_json(path, {"version": 1, "donors_sha256": engine._digest(donors), "bindings": {
        row["donor_id"]: {"source_sha256": api.text_hash(row["response_text"]), "reference_map": {"[URL_1]": "https://example.org/source"} if index == 0 else {}}
        for index, row in enumerate(rows)}})
    return path


def test_sidecar_restores_candidate_source_and_is_resume_bound(run, tmp_path):
    donors, output, _, generate = run
    bindings = sidecar(donors, tmp_path)
    def source(value, context):
        references = context["references"]["supplied_references"]
        if references:
            value["response_text"] += "\nSource: " + references[0]["token"]
        return value
    manifest = generate(adapter=FakeTeacher(source_override=source), reference_bindings_path=bindings)
    assert manifest["reference_bindings_sha256"] == engine._digest(bindings)
    assert any("https://example.org/source" in row["response_text"] for row in api.read_jsonl(output / "accepted_genui.jsonl"))
    assert "[URL_1]" in donors.read_text()
    with pytest.raises(ValueError, match="contract mismatch"):
        generate(resume=True)
    assert generate(resume=True, reference_bindings_path=bindings)["accepted_rows"] == 3


def test_restored_donor_is_not_accepted_as_changed_source(run, tmp_path):
    donors, _, _, generate = run
    bindings = sidecar(donors, tmp_path)
    def source(value, context):
        value["response_text"] = context["donor_response_text"]
        return value
    with pytest.raises(RuntimeError, match="No eligible"):
        generate(adapter=FakeTeacher(source_override=source), reference_bindings_path=bindings)


def test_invalid_sidecar_rejected_before_calls(run, tmp_path):
    donors, _, _, generate = run
    bindings = sidecar(donors, tmp_path)
    value = json.loads(bindings.read_text())
    value["bindings"].pop("id/1")
    write_json(bindings, value)
    teacher = FakeTeacher()
    with pytest.raises(ValueError, match="exactly"):
        generate(adapter=teacher, reference_bindings_path=bindings)
    assert teacher.calls == []


def legacy_fixture(run, tmp_path):
    donors, output, _, generate = run
    generate()
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    for name in (*engine.PROJECTIONS, "augmentation_audit.jsonl", "manifest.json"):
        shutil.copyfile(output / name, legacy / name)
    manifest = json.loads((legacy / "manifest.json").read_text())
    for field in ("resume_schema", "resume_contract_sha256", "reference_policy_version", "reference_bindings_sha256", "projection_hashes", "journal_hashes"):
        manifest.pop(field, None)
    manifest["code_hashes"] = {name: min(values) for name, values in engine.LEGACY_CODE_HASHES.items()}
    manifest["status"] = "failed"
    write_json(legacy / "manifest.json", manifest)
    return donors, legacy


@pytest.mark.parametrize("status", ["failed", "completed"])
def test_known_legacy_upgrade_readonly_preflight_and_reuses_valid_completed(run, tmp_path, status):
    donors, legacy = legacy_fixture(run, tmp_path)
    manifest = json.loads((legacy / "manifest.json").read_text())
    manifest["status"] = status
    write_json(legacy / "manifest.json", manifest)
    before = {path.name: path.read_bytes() for path in legacy.iterdir()}
    with patch("llm.factory.build_adapter", side_effect=AssertionError("must not create teacher")):
        result = api.validate_generation_resume(donors, legacy, max_new_samples=3)
    assert result["legacy_upgrade"] is True
    assert {path.name: path.read_bytes() for path in legacy.iterdir()} == before
    teacher = FakeTeacher()
    result = api.generate_training_augmentations(donors, legacy, max_new_samples=3, adapter=teacher, resume=True,
        stage3_runner=lambda **_: pytest.fail("Completed legacy rows must be reused"))
    assert result["accepted_rows"] == 3
    assert teacher.calls == []
    assert all((legacy / "legacy_v1" / name).read_bytes() == data for name, data in before.items())
    assert api.validate_generation_resume(donors, legacy, max_new_samples=3)["legacy_upgrade"] is False


@pytest.mark.parametrize("change", ["missing_manifest", "code", "prompt", "seed", "accepted_hash"])
def test_legacy_upgrade_refuses_unproven_identity(run, tmp_path, change):
    donors, legacy = legacy_fixture(run, tmp_path)
    path = legacy / "manifest.json"
    manifest = json.loads(path.read_text())
    if change == "missing_manifest":
        path.unlink()
    else:
        if change == "code":
            manifest["code_hashes"]["src/pipeline/training_augmentation.py"] = "unknown"
        elif change == "prompt":
            manifest["teacher_prompt_sha256"] = "unknown"
        elif change == "seed":
            manifest["seed"] = 321
        else:
            manifest["accepted_genui_sha256"] = "unknown"
        write_json(path, manifest)
    with pytest.raises(ValueError):
        api.validate_generation_resume(donors, legacy, max_new_samples=3)


def test_legacy_unbound_sources_regenerated_without_target_rewrite(run, tmp_path):
    donors, legacy = legacy_fixture(run, tmp_path)
    accepted = api.read_jsonl(legacy / "accepted_genui.jsonl")
    responses = api.read_jsonl(legacy / "responses.jsonl")
    audits = api.read_jsonl(legacy / "augmentation_audit.jsonl")
    candidate_id = accepted[0]["response_id"]
    source = "Synthetic unknown media [IMAGE_URL_999]"
    for row in accepted:
        if row["response_id"] == candidate_id:
            row["response_text"] = source
            row["augmentation"]["source_sha256"] = api.text_hash(source)
    for row in responses:
        if row["response_id"] == candidate_id:
            row["response_text"] = source
    for audit in audits:
        if audit.get("candidate_id") == candidate_id:
            audit["augmentation"]["source_sha256"] = api.text_hash(source)
            if audit["status"] == "source_approved":
                generated = json.loads(audit["source_generation"]["text"])
                generated["response_text"] = source
                audit["source_generation"]["text"] = json.dumps(generated)
    for name, rows in (("accepted_genui.jsonl", accepted), ("responses.jsonl", responses), ("augmentation_audit.jsonl", audits)):
        (legacy / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = json.loads((legacy / "manifest.json").read_text())
    manifest["accepted_genui_sha256"] = engine._digest(legacy / "accepted_genui.jsonl")
    write_json(legacy / "manifest.json", manifest)
    teacher = FakeTeacher()
    def runner(**kwargs):
        kwargs["genui_path"].write_text("".join(json.dumps(stage3_row(row)) + "\n" for row in api.read_jsonl(kwargs["responses_path"])), encoding="utf-8")
    result = api.generate_training_augmentations(donors, legacy, max_new_samples=3, adapter=teacher, stage3_runner=runner, resume=True)
    assert result["accepted_rows"] == 3
    assert len(teacher.calls) == 2
    assert "[IMAGE_URL_999]" in (legacy / "legacy_v1/accepted_genui.jsonl").read_text()
    assert "[IMAGE_URL_999]" not in (legacy / "accepted_genui.jsonl").read_text()
