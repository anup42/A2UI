from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.train import callbacks


def _digest_tree(path: Path) -> dict[str, str]:
    return {
        item.name: hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(path.iterdir())
        if item.is_file()
    }


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, full_model=False):
    monkeypatch.setitem(
        sys.modules, "transformers", types.SimpleNamespace(TrainerCallback=object)
    )
    split = tmp_path / "golden.jsonl"
    split.write_text('{"id":"one","response_text":"source","completion":"{}"}\n')
    scores = iter([42.0, 10.0])

    def generate(*, output_path, **kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('{"id":"one","generated_text":"{}"}\n')
        return 1

    def evaluate(*, output_dir, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "scored_predictions.jsonl").write_text('{"id":"one","score":1}\n')
        return {"overall_score": next(scores), "count": 1}

    monkeypatch.setattr(callbacks, "_generate_predictions_with_model", generate)
    monkeypatch.setattr(callbacks, "evaluate_predictions", evaluate)

    class Model:
        payload = b"prior-best"

        def save_pretrained(self, directory):
            target = Path(directory)
            target.mkdir(parents=True, exist_ok=True)
            (target / ("config.json" if full_model else "adapter_config.json")).write_text("{}")
            (target / ("pytorch_model.bin" if full_model else "adapter_model.safetensors")).write_bytes(self.payload)

    class Tokenizer:
        chat_template = "fixed-template"

        def save_pretrained(self, directory):
            (Path(directory) / "tokenizer_config.json").write_text("{}")

    def build(output: Path, *, resume=None, relocate=False):
        return callbacks.build_golden_set_eval_callback(
            enabled=True,
            split_path=split,
            output_dir=output,
            adapter=SimpleNamespace(config={}),
            tokenizer=Tokenizer(),
            max_rows=1,
            required_rows=1,
            trigger="evaluate",
            interval=1,
            metric_for_best_model="overall_score",
            best_checkpoint_dir=output / "best_golden_checkpoint",
            resume_checkpoint=resume,
            resume_relocate_best=relocate,
        )

    old_output = tmp_path / "old"
    initial = build(old_output)
    state = SimpleNamespace(global_step=10, epoch=1.0)
    initial.on_evaluate(None, state, None, model=Model())
    checkpoint = old_output / "checkpoint-10"
    checkpoint.mkdir()
    initial.on_save(SimpleNamespace(output_dir=old_output), state, None)
    return build, Model(), checkpoint, old_output / "best_golden_checkpoint"


def test_opted_in_resume_copies_previous_best_and_worse_score_stays_local(
    tmp_path, monkeypatch
):
    build, model, checkpoint, old_best = _fixture(tmp_path, monkeypatch)
    source_before = _digest_tree(old_best)
    new_output = tmp_path / "continuation"

    resumed = build(new_output, resume=checkpoint, relocate=True)
    local_best = new_output / "best_golden_checkpoint"
    assert resumed.summary()["checkpoint_dir"] == str(local_best.resolve())
    assert _digest_tree(local_best) == source_before

    resumed.on_evaluate(
        None, SimpleNamespace(global_step=20, epoch=2.0), None, model=model
    )
    assert resumed.summary()["metric_value"] == 42.0
    assert resumed.summary()["step"] == 10
    assert resumed.summary()["checkpoint_dir"] == str(local_best.resolve())
    assert _digest_tree(old_best) == source_before


def test_default_resume_keeps_existing_best_location(tmp_path, monkeypatch):
    build, _, checkpoint, old_best = _fixture(tmp_path, monkeypatch)
    resumed = build(tmp_path / "continuation", resume=checkpoint)
    assert resumed.summary()["checkpoint_dir"] == str(old_best)
    assert not (tmp_path / "continuation/best_golden_checkpoint").exists()


@pytest.mark.parametrize("damage", ["missing", "mutated"])
def test_relocation_rejects_missing_or_mutated_previous_best(
    tmp_path, monkeypatch, damage
):
    build, _, checkpoint, _ = _fixture(tmp_path, monkeypatch)
    saved = json.loads((checkpoint / "golden_callback_state.json").read_text())
    adapter = Path(saved["best_checkpoint_snapshot"]) / "adapter_model.safetensors"
    if damage == "missing":
        adapter.unlink()
    else:
        adapter.write_bytes(b"changed")

    with pytest.raises(ValueError, match="missing or changed"):
        build(tmp_path / "continuation", resume=checkpoint, relocate=True)
    assert not (tmp_path / "continuation/best_golden_checkpoint").exists()


def test_old_checkpoint_keeps_its_selected_best_after_later_improvement(tmp_path, monkeypatch):
    build, model, checkpoint, old_best = _fixture(tmp_path, monkeypatch)
    saved = json.loads((checkpoint / "golden_callback_state.json").read_text())
    snapshot = Path(saved["best_checkpoint_snapshot"])
    original_snapshot = _digest_tree(snapshot)
    resumed = build(checkpoint.parent, resume=checkpoint)
    monkeypatch.setattr(callbacks, "evaluate_predictions", lambda **_: {"overall_score": 60.0})
    model.payload = b"new-best"
    state = SimpleNamespace(global_step=20, epoch=2.0)
    resumed.on_evaluate(None, state, None, model=model)
    assert (old_best / "adapter_model.safetensors").read_bytes() == b"new-best"
    assert _digest_tree(snapshot) == original_snapshot
    next_checkpoint = checkpoint.parent / "checkpoint-20"
    next_checkpoint.mkdir()
    resumed.on_save(SimpleNamespace(output_dir=checkpoint.parent), state, None)

    # The old checkpoint must recover its own best, not the later improved one.
    earlier = build(tmp_path / "earlier-resume", resume=checkpoint, relocate=True)
    assert earlier.summary()["step"] == 10
    assert earlier.summary()["metric_value"] == 42.0
    assert (Path(earlier.summary()["checkpoint_dir"]) / "adapter_model.safetensors").read_bytes() == b"prior-best"
    later = build(tmp_path / "later-resume", resume=next_checkpoint)
    assert later.summary()["step"] == 20
    assert later.summary()["metric_value"] == 60.0
    # A non-relocating rewind may read the frozen best, but improvement must
    # write the new run's public alias, never overwrite that frozen source.
    rewind_output = tmp_path / "rewind"
    rewind = build(rewind_output, resume=checkpoint)
    assert Path(rewind.summary()["checkpoint_dir"]) == snapshot
    model.payload = b"rewind-improved"
    rewind.on_evaluate(None, SimpleNamespace(global_step=15, epoch=1.5), None, model=model)
    assert Path(rewind.summary()["checkpoint_dir"]) == rewind_output / "best_golden_checkpoint"
    assert _digest_tree(snapshot) == original_snapshot


def test_same_best_snapshot_is_reused_and_changed_snapshot_is_rejected(tmp_path, monkeypatch):
    build, _, checkpoint, _ = _fixture(tmp_path, monkeypatch)
    resumed = build(checkpoint.parent, resume=checkpoint)
    saved = json.loads((checkpoint / "golden_callback_state.json").read_text())
    snapshot = Path(saved["best_checkpoint_snapshot"])
    before = _digest_tree(snapshot)
    copy_calls = []
    monkeypatch.setattr(callbacks.shutil, "copy2", lambda *a, **kw: copy_calls.append(a))
    next_checkpoint = checkpoint.parent / "checkpoint-20"
    next_checkpoint.mkdir()
    state = SimpleNamespace(global_step=20, epoch=2.0)
    resumed.on_save(SimpleNamespace(output_dir=checkpoint.parent), state, None)
    assert copy_calls == []
    assert _digest_tree(snapshot) == before
    second = json.loads((next_checkpoint / "golden_callback_state.json").read_text())
    assert second["best_checkpoint_snapshot"] == saved["best_checkpoint_snapshot"]
    (snapshot / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="snapshot is missing or changed"):
        resumed.on_save(SimpleNamespace(output_dir=checkpoint.parent), state, None)


def test_legacy_selector_state_without_snapshot_is_still_hash_verified(tmp_path, monkeypatch):
    build, _, checkpoint, old_best = _fixture(tmp_path, monkeypatch)
    state_path = checkpoint / "golden_callback_state.json"
    saved = json.loads(state_path.read_text())
    saved.pop("best_checkpoint_snapshot")
    state_path.write_text(json.dumps(saved))
    resumed = build(tmp_path / "legacy-resume", resume=checkpoint)
    assert resumed.summary()["checkpoint_dir"] == str(old_best)
    (old_best / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="missing or changed"):
        build(tmp_path / "legacy-resume", resume=checkpoint)


def test_resumed_frozen_best_cannot_be_rebound_after_tampering(tmp_path, monkeypatch):
    build, _, checkpoint, old_best = _fixture(tmp_path, monkeypatch)
    saved = json.loads((checkpoint / "golden_callback_state.json").read_text())
    snapshot = Path(saved["best_checkpoint_snapshot"])
    (old_best / "adapter_model.safetensors").write_bytes(b"later-best")
    output = tmp_path / "resumed"
    resumed = build(output, resume=checkpoint)
    assert Path(resumed.summary()["checkpoint_dir"]) == snapshot
    (snapshot / "adapter_model.safetensors").write_bytes(b"tampered")
    (output / "checkpoint-20").mkdir(parents=True)
    with pytest.raises(ValueError, match="missing or changed before snapshot"):
        resumed.on_save(SimpleNamespace(output_dir=output), SimpleNamespace(global_step=20), None)


@pytest.mark.parametrize("full_model,extra", [
    (False, "adapter_model.bin"),
    (True, "model.safetensors"),
    (True, "model.safetensors.index.json"),
])
def test_resume_rejects_unbound_added_weights_or_index(tmp_path, monkeypatch, full_model, extra):
    build, _, checkpoint, old_best = _fixture(tmp_path, monkeypatch, full_model=full_model)
    saved = json.loads((checkpoint / "golden_callback_state.json").read_text())
    snapshot = Path(saved["best_checkpoint_snapshot"])
    for directory in (snapshot, old_best):
        (directory / extra).write_bytes(b"unbound payload can override loader selection")
    with pytest.raises(ValueError, match="snapshot is missing or changed"):
        build(tmp_path / "resume", resume=checkpoint)


def test_relocation_rejects_existing_destination_without_touching_source(
    tmp_path, monkeypatch
):
    build, _, checkpoint, old_best = _fixture(tmp_path, monkeypatch)
    source_before = _digest_tree(old_best)
    destination = tmp_path / "continuation/best_golden_checkpoint"
    destination.mkdir(parents=True)
    (destination / "keep.txt").write_text("owned")

    with pytest.raises(FileExistsError, match="already exists"):
        build(tmp_path / "continuation", resume=checkpoint, relocate=True)
    assert (destination / "keep.txt").read_text() == "owned"
    assert _digest_tree(old_best) == source_before


def test_relocation_rejects_symlinked_source_artifact(tmp_path, monkeypatch):
    build, _, checkpoint, old_best = _fixture(tmp_path, monkeypatch)
    external = tmp_path / "external.txt"
    external.write_text("outside")
    link = old_best / "unexpected-link"
    try:
        link.symlink_to(external)
    except OSError:
        pytest.skip("symlinks are unavailable on this host")

    with pytest.raises(ValueError, match="contains a symlink"):
        build(tmp_path / "continuation", resume=checkpoint, relocate=True)
    assert external.read_text() == "outside"
