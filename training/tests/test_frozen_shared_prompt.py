"""An unchanged prepared prompt survives live checkout drift, never artifact tampering."""
import json
from copy import deepcopy

import pytest
import test_dual_golden_prepared_contract as fixtures
from ir_training.data import shared_prompt as prompt
from ir_training.eval.prepared_contract import (
    checked_preparation_manifest,
    file_sha256,
    verify_evaluation_prepared_contract,
)

prepared_dual = fixtures.prepared_dual


def live_builder_unavailable(*args, **kwargs):
    raise AssertionError("A bound runtime must not regenerate the saved production prompt")


def test_frozen_contract_and_inference_do_not_read_live_prompt(tmp_path, monkeypatch):
    saved = prompt.create_shared_prompt_contract()
    path = tmp_path / "shared_prompt.json"
    path.write_text(json.dumps(saved), encoding="utf-8")
    monkeypatch.setattr(prompt, "create_shared_prompt_contract", live_builder_unavailable)
    assert prompt.validate_shared_prompt_snapshot(saved) == saved
    assert prompt.load_shared_prompt_contract(path) == saved
    messages = prompt.build_inference_messages("  New response  ", saved)
    assert messages[:-1] == saved["scaffold"]["messages"]
    assert messages[-1] == {"role": "user", "content": saved["scaffold"]["task_prefix"] + "New response"}
    messages[0]["content"] = "caller changed its copy"
    assert prompt.load_shared_prompt_contract(path) == saved
    with pytest.raises(AssertionError, match="must not regenerate"):
        prompt.load_shared_prompt_contract(path, require_current=True)


def test_new_preparation_rejects_real_production_drift_with_diagnostic(tmp_path, monkeypatch):
    saved = prompt.create_shared_prompt_contract()
    original = (prompt.repo_root() / prompt.PRODUCTION_PROMPT).read_text(encoding="utf-8")
    current_source = tmp_path / prompt.PRODUCTION_PROMPT
    current_source.parent.mkdir(parents=True)
    current_source.write_text(original + "\nUpdated production instruction.\n", encoding="utf-8")
    monkeypatch.setattr(prompt, "repo_root", lambda: tmp_path)
    current = prompt.create_shared_prompt_contract()
    assert saved["contract_sha256"] != current["contract_sha256"]
    assert prompt.validate_shared_prompt_snapshot(saved) == saved
    with pytest.raises(ValueError, match="source_text_sha256.*saved=.*current=.*fresh preparation"):
        prompt.validate_shared_prompt_contract(saved)
    from ir_training.pipeline.golden_training import prepare_data
    # Reject a stale new plan before even looking up source/model paths or cache.
    with pytest.raises(ValueError, match="versus current production"):
        prepare_data({"shared_prompt": saved})
    assert prompt.validate_shared_prompt_contract(current) == current


@pytest.mark.parametrize("part", ["contract", "scaffold", "system", "source", "version", "ordering", "roles", "missing", "prefix"])
def test_snapshot_does_not_accept_tampering(part):
    saved = prompt.create_shared_prompt_contract()
    corrupted = deepcopy(saved)
    if part == "contract":
        corrupted["contract_sha256"] = "0" * 64
    elif part == "scaffold":
        corrupted["scaffold"]["messages"][2]["content"] += "changed"
    elif part == "system":
        corrupted["scaffold"]["messages"][0]["content"] += "changed"
    elif part == "source":
        corrupted["source_text_sha256"] = "0" * 64
    elif part == "version":
        corrupted["schema_version"] = True
    elif part == "ordering":
        corrupted["serialization_order"] = []
    elif part == "roles":
        corrupted["scaffold"]["messages"][1]["role"] = "assistant"
    elif part == "missing":
        del corrupted["system_prompt_sha256"]
    else:
        corrupted["scaffold"]["task_prefix"] = ""
    with pytest.raises(ValueError, match="stale or changed"):
        prompt.validate_shared_prompt_snapshot(corrupted)


def test_bound_launch_and_both_golden_evaluations_use_frozen_prompt(prepared_dual, tmp_path, monkeypatch):
    directory, manifest = prepared_dual
    config_path, _, _ = fixtures.bound_run(tmp_path, directory)
    before = {name: (directory / name).read_bytes() for name in ("manifest.json", "shared_prompt.json", "train.jsonl")}
    monkeypatch.setattr(prompt, "create_shared_prompt_contract", live_builder_unavailable)
    assert checked_preparation_manifest(directory)["shared_prompt"] == manifest["shared_prompt"]
    fixtures.script("launch_review_training").verify_launch_binding(config_path)
    for count in (32, 35):
        checked = verify_evaluation_prepared_contract(config_path, directory / f"golden{count}.jsonl",
            required_rows=count, max_input_tokens=4096)
        assert checked["required_rows"] == count
    assert before == {name: (directory / name).read_bytes() for name in before}


@pytest.mark.parametrize("part", ["prompt_scaffolds", "inference_prompt", "shared_prompt"])
def test_artifacts_must_agree_even_if_file_hashes_are_refreshed(prepared_dual, part):
    directory, manifest = prepared_dual
    path = directory / f"{part}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if part == "prompt_scaffolds":
        value[0]["messages"][0]["content"] += "changed"
    elif part == "inference_prompt":
        value["task_prefix"] += "changed"
    else:
        value["scaffold"]["messages"][0]["content"] += "changed"
        manifest["shared_prompt"] = value
    path.write_text(json.dumps(value), encoding="utf-8")
    manifest[f"{part}_sha256"] = file_sha256(path)
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from|checksum mismatch"):
        checked_preparation_manifest(directory)


def test_prompt_tampering_fails_before_large_corpus_scan(prepared_dual, monkeypatch):
    directory, _ = prepared_dual
    with (directory / "shared_prompt.json").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    script = fixtures.script("prepare_review_training")
    monkeypatch.setattr(script, "rows", lambda *a: pytest.fail("Corpus scan must not run"))
    with pytest.raises(ValueError, match="Saved shared_prompt changed"):
        script.verify_prepared(directory, directory / "golden32.jsonl", max_sequence=4096, max_prompt=4096)
