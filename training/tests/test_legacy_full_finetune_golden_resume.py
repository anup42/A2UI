"""Legacy standalone runs must not reset enabled Golden selection on resume."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("resumed", [False, True])
def test_standalone_golden_builder_forwards_resume_checkpoint(tmp_path, monkeypatch, resumed):
    script = Path(__file__).resolve().parents[1] / "scripts/train_full_finetune_qat.py"
    spec = importlib.util.spec_from_file_location("legacy_full_finetune_resume_test", script)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "build_golden_set_eval_callback", lambda **kwargs: kwargs)
    checkpoint = tmp_path / "checkpoint-10" if resumed else None
    result = module._build_golden_eval_callback(
        config={"golden_eval": {"enabled": True, "split_path": str(tmp_path / "golden.jsonl")}},
        output_dir=tmp_path / "run", adapter=object(), tokenizer=object(),
        model_cfg={}, training_cfg={}, trainer=SimpleNamespace(),
        resume_checkpoint=checkpoint,
    )
    assert result["resume_checkpoint"] == checkpoint


def test_disabled_legacy_golden_still_does_not_construct_callback(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/train_full_finetune_qat.py"
    spec = importlib.util.spec_from_file_location("legacy_full_finetune_disabled_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._build_golden_eval_callback(
        config={}, output_dir=tmp_path, adapter=None, tokenizer=None,
        model_cfg={}, training_cfg={}, trainer=None,
        resume_checkpoint=tmp_path / "checkpoint-10",
    ) is None
