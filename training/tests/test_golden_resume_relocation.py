from __future__ import annotations

import hashlib
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


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
        def save_pretrained(self, directory):
            target = Path(directory)
            target.mkdir(parents=True, exist_ok=True)
            (target / "adapter_config.json").write_text("{}")
            (target / "adapter_model.safetensors").write_bytes(b"prior-best")

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
    build, _, checkpoint, old_best = _fixture(tmp_path, monkeypatch)
    adapter = old_best / "adapter_model.safetensors"
    if damage == "missing":
        adapter.unlink()
    else:
        adapter.write_bytes(b"changed")

    with pytest.raises(ValueError, match="missing or changed"):
        build(tmp_path / "continuation", resume=checkpoint, relocate=True)
    assert not (tmp_path / "continuation/best_golden_checkpoint").exists()


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
