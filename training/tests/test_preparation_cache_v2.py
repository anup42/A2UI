"""Persistent cache safety tests; only tiny synthetic artifacts, no model load."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ir_training.pipeline import preparation_cache as cache


def artifacts(output: Path) -> Path:
    for name in cache.REQUIRED_ARTIFACTS:
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"synthetic_artifact": name}) + "\n", encoding="utf-8")
    return output


@pytest.fixture
def stored(tmp_path):
    root = tmp_path / "shared-cache"
    output = artifacts(tmp_path / "first-run")
    binding = {"version": 2, "source_files": {"train.jsonl": "fixture"}, "tokenizer_assets": {"tokenizer.json": "fixture"}}
    cache.publish(root, output, binding)
    entry = root / "prepared-v2" / cache._digest(binding)
    return root, output, binding, entry


def test_owned_artifacts_survive_origin_deletion_and_mutated_restored_run(stored, tmp_path):
    root, original, binding, entry = stored
    expected = (original / "prepared/train.jsonl").read_bytes()
    # Deleting only this pytest-owned temporary fixture must not evict the cache.
    assert original.parent == tmp_path
    shutil.rmtree(original)
    second = tmp_path / "second-run"
    assert cache.restore(root, second, binding)
    assert (second / "prepared/train.jsonl").read_bytes() == expected
    (second / "prepared/train.jsonl").write_text("changed restored copy\n", encoding="utf-8")
    third = tmp_path / "third-run"
    assert cache.restore(root, third, binding)
    assert (third / "prepared/train.jsonl").read_bytes() == expected
    assert (entry / "prepared/train.jsonl").read_bytes() == expected
    assert json.loads((third / "preparation_receipt.json").read_text())["identity"] == binding
    assert json.loads((third / "cache_reuse.json").read_text())["cache_entry"] == str(entry)


def test_empty_and_legacy_entries_have_clear_miss_reasons(tmp_path, capsys):
    root = tmp_path / "cache"
    assert not cache.restore(root, tmp_path / "new", {"version": 2})
    assert "CACHE MISS" in capsys.readouterr().out
    (root / "old-v1.json").write_text(json.dumps({"output": str(tmp_path / "do-not-read")}), encoding="utf-8")
    assert not cache.restore(root, tmp_path / "newer", {"version": 2})
    assert "legacy run-pointer" in capsys.readouterr().out


def test_changed_identity_explains_fields(stored, tmp_path, capsys):
    root, _, binding, _ = stored
    updated = deepcopy(binding)
    updated["tokenizer_assets"]["tokenizer.json"] = "changed"
    assert not cache.restore(root, tmp_path / "new", updated)
    assert "identity changed: tokenizer_assets" in capsys.readouterr().out


@pytest.mark.parametrize("damage", ["content", "missing", "manifest", "incomplete", "version", "identity", "checksum"])
def test_corrupt_or_incomplete_entry_is_never_installed(stored, tmp_path, damage, capsys):
    root, _, binding, entry = stored
    path = entry / "prepared/train.jsonl"
    manifest = entry / "manifest.json"
    if damage == "content":
        path.write_text("corrupt\n", encoding="utf-8")
    elif damage == "missing":
        path.unlink()
    elif damage == "manifest":
        manifest.write_text("not-json", encoding="utf-8")
    else:
        saved = json.loads(manifest.read_text())
        if damage == "incomplete":
            del saved["files"]["prepared/train.jsonl"]
        elif damage == "version":
            saved["version"] = 1
        elif damage == "identity":
            saved["identity"]["tokenizer_assets"] = {}
        else:
            saved["files"]["prepared/train.jsonl"] = "invalid sha"
        manifest.write_text(json.dumps(saved), encoding="utf-8")
    output = tmp_path / "new"
    assert not cache.restore(root, output, binding)
    assert not (output / "prepared").exists()
    assert "CACHE MISS" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["../secret.json", "prepared/../../secret.json", "/prepared/train.jsonl", "prepared\\train.jsonl",
                                  "prepared//train.jsonl", "prepared/./train.jsonl", "prepared/train.jsonl:secret", "C:/secret.json", "prepared/train.txt"])
def test_manifest_paths_cannot_escape_or_alias(stored, tmp_path, name):
    root, _, binding, entry = stored
    manifest = entry / "manifest.json"
    saved = json.loads(manifest.read_text())
    saved["files"][name] = "0" * 64
    manifest.write_text(json.dumps(saved), encoding="utf-8")
    assert not cache.restore(root, tmp_path / "new", binding)
    assert not (tmp_path / "new/prepared").exists()


def test_existing_restore_destination_is_not_overwritten(stored, tmp_path):
    root, _, binding, _ = stored
    output = artifacts(tmp_path / "occupied")
    before = (output / "prepared/train.jsonl").read_bytes()
    with pytest.raises(FileExistsError, match="fresh preparation destination"):
        cache.restore(root, output, binding)
    assert (output / "prepared/train.jsonl").read_bytes() == before


def test_symbolic_artifact_rejected_even_if_inside_entry(stored, tmp_path, monkeypatch):
    root, _, binding, entry = stored
    target = entry / "prepared/train.jsonl"
    real_check = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda self: self == target or real_check(self))
    assert not cache.restore(root, tmp_path / "new", binding)


def test_symbolic_source_is_never_published(tmp_path, monkeypatch):
    output = artifacts(tmp_path / "run")
    target = output / "prepared/train.jsonl"
    real_check = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda self: self == target or real_check(self))
    with pytest.raises(ValueError, match="Symlink"):
        cache.publish(tmp_path / "cache", output, {"version": 2})


def test_concurrent_publishers_share_one_complete_owned_entry(tmp_path):
    root, binding = tmp_path / "cache", {"version": 2, "source": "fixture"}
    first, second = artifacts(tmp_path / "first"), artifacts(tmp_path / "second")
    with ThreadPoolExecutor(max_workers=2) as executor:
        tasks = [executor.submit(cache.publish, root, output, binding, interval=0.05) for output in (first, second)]
        for task in tasks:
            task.result(timeout=30)
    entries = [path for path in (root / "prepared-v2").iterdir() if path.name == cache._digest(binding)]
    assert len(entries) == 1
    assert cache.restore(root, tmp_path / "third", binding)
    assert not list((root / "prepared-v2").glob(".prepared-publish-*"))


def test_interrupted_publish_does_not_expose_partial_entry(tmp_path, monkeypatch):
    root, binding = tmp_path / "cache", {"version": 2}
    output = artifacts(tmp_path / "run")
    copy = cache._copy_checked
    calls = []
    def fail(source, destination, expected, *, interval):
        calls.append(source)
        if len(calls) == 3:
            raise OSError("synthetic disk write failure")
        return copy(source, destination, expected, interval=interval)
    monkeypatch.setattr(cache, "_copy_checked", fail)
    with pytest.raises(OSError, match="synthetic disk"):
        cache.publish(root, output, binding)
    assert not (root / "prepared-v2" / cache._digest(binding)).exists()
    assert not (output / "preparation_receipt.json").exists()
    assert not list((root / "prepared-v2").glob(".prepared-publish-*"))


def test_existing_entry_cannot_certify_different_prepared_output(stored, tmp_path):
    root, _, binding, entry = stored
    output = artifacts(tmp_path / "divergent")
    (output / "prepared/train.jsonl").write_text("different\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the existing cache"):
        cache.publish(root, output, binding)
    assert not (output / "preparation_receipt.json").exists()
    assert (entry / "prepared/train.jsonl").read_text(encoding="utf-8") != "different\n"


def test_partial_restore_install_failure_is_fatal_not_a_miss(stored, tmp_path, monkeypatch):
    root, _, binding, _ = stored
    atomic = cache._atomic_json
    def fail_receipt(path, value):
        if path.name == "preparation_receipt.json":
            raise OSError("synthetic receipt write failure")
        return atomic(path, value)
    monkeypatch.setattr(cache, "_atomic_json", fail_receipt)
    with pytest.raises(RuntimeError, match="fresh output directory"):
        cache.restore(root, tmp_path / "incomplete-install", binding)


def test_corrupt_entry_rebuilt_without_mutating_source(stored, tmp_path):
    root, output, binding, entry = stored
    expected = (output / "prepared/train.jsonl").read_bytes()
    (entry / "prepared/train.jsonl").write_text("damaged", encoding="utf-8")
    cache.publish(root, output, binding)
    assert cache.restore(root, tmp_path / "new", binding)
    assert (output / "prepared/train.jsonl").read_bytes() == expected
    assert (entry / "prepared/train.jsonl").read_bytes() == expected
    assert len(list(entry.parent.glob(".rejected-*"))) == 1


def test_build_lock_serializes_lookup_and_only_one_writer_builds(tmp_path):
    root, binding = tmp_path / "cache", {"version": 2, "source": "shared"}
    built = []
    def run(index):
        output = tmp_path / f"run-{index}"
        with cache.lock(root, binding, interval=0.05):
            if not cache.restore(root, output, binding, interval=0.05):
                artifacts(output)
                built.append(index)
                cache.publish(root, output, binding, interval=0.05)
        return output
    with ThreadPoolExecutor(max_workers=2) as executor:
        outputs = list(executor.map(run, (1, 2)))
    assert len(built) == 1
    assert all((output / "preparation_receipt.json").is_file() for output in outputs)


def test_preparation_identity_excludes_training_code_and_follows_codec_dependencies():
    implementation = cache._implementation(cache.repo_root())
    assert "training/src/ir_training/data/express_preparation.py" in implementation
    assert "training/src/ir_training/data/audit_filter.py" in implementation
    assert "dataset/src/pipeline/ir_formats/express.py" in implementation
    assert "dataset/src/pipeline/ir_formats/canonical.py" in implementation
    assert "dataset/src/pipeline/renderer_semantics.py" in implementation
    assert "training/src/ir_training/pipeline/golden_training.py:_group_split" in implementation
    assert "training/scripts/prepare_review_training.py:verify_prepared" in implementation
    for path in ("training/src/ir_training/train/sft.py", "training/src/ir_training/train/lora_targets.py",
                 "training/src/ir_training/train/lora_config.py", "training/src/ir_training/common/progress.py",
                 "training/scripts/prepare_review_training.py:build_config", "training/src/ir_training/pipeline/golden_training.py"):
        assert path not in implementation


def test_actual_preprocessing_changes_invalidate_but_recipe_and_trainer_edits_do_not(tmp_path):
    source = cache.repo_root()
    root = tmp_path / "checkout"
    for relative in ("training/src/ir_training", "dataset/src/pipeline", "dataset/schema"):
        shutil.copytree(source / relative, root / relative, ignore=shutil.ignore_patterns("__pycache__"))
    (root / "training/scripts").mkdir(parents=True)
    shutil.copy2(source / "training/scripts/prepare_review_training.py", root / "training/scripts/prepare_review_training.py")
    before = cache._implementation(root)
    for relative in ("training/src/ir_training/train/sft.py", "training/src/ir_training/train/lora_targets.py"):
        (root / relative).write_text("TRAINING_ONLY_CHANGE = True\n", encoding="utf-8")
    recipe = root / "training/scripts/prepare_review_training.py"
    recipe.write_text(recipe.read_text(encoding="utf-8") + "\nUNRELATED_LORA_CONFIG = 99\n", encoding="utf-8")
    assert cache._implementation(root) == before
    grouping = root / "training/src/ir_training/pipeline/golden_training.py"
    grouping.write_text(grouping.read_text(encoding="utf-8").replace("stratified_split(grouped, .9, .1", "stratified_split(grouped, .8, .2"), encoding="utf-8")
    after_grouping = cache._implementation(root)
    assert after_grouping != before
    filtering = root / "training/src/ir_training/data/audit_filter.py"
    filtering.write_text(filtering.read_text(encoding="utf-8") + "\nFILTER_POLICY_VERSION = 999\n", encoding="utf-8")
    assert cache._implementation(root) != after_grouping


def test_identity_tracks_data_goldens_prompt_tokenizer_packages_and_preparation_settings(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    tokenizer_file = model / "tokenizer.json"
    tokenizer_file.write_text("{}", encoding="utf-8")
    plan = {"options": {"model_dir": str(model), "profile": "e2b", "input_dir": "/source", "source_run_dir": None,
                        "seed": 42, "max_seq_length": 4096, "max_input_tokens": 4096, "epochs": 1, "logging_steps": 10},
            "shared_prompt": {"contract": "fixture"}, "goldens": {"golden32": "fixed32", "golden35": "fixed35"}}
    monkeypatch.setattr(cache, "_implementation", lambda root: {"fixture.py": "hash"})
    before = cache.identity(plan, {"train": "train-sha", "val": "val-sha"})
    training_only = deepcopy(plan)
    training_only["options"].update(epochs=3, logging_steps=1, learning_rate=0.01, output_dir="new-output", devices="0,1,2,3")
    assert cache.identity(training_only, before["source_files"]) == before
    assert cache.identity(plan, {"train": "changed", "val": "val-sha"}) != before
    for category, key, value in (("shared_prompt", "contract", "new"), ("goldens", "golden32", "new"),
                                 ("goldens", "golden35", "new"), ("options", "seed", 7),
                                 ("options", "max_seq_length", 8192), ("options", "max_input_tokens", 8192)):
        changed = deepcopy(plan)
        changed[category][key] = value
        assert cache.identity(changed, before["source_files"]) != before
    tokenizer_file.write_text('{"changed": true}', encoding="utf-8")
    assert cache.identity(plan, before["source_files"]) != before
    tokenizer_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cache, "version", lambda name: "changed-version")
    assert cache.identity(plan, before["source_files"]) != before
