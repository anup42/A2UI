from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training" / "src"))
from ir_training.train.lora_config import resolve_lora_config_targets


def _model():
    nn = pytest.importorskip("torch.nn")

    class WrappedLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2, bias=False)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Module()
            self.language_model.q_proj = WrappedLinear()
            self.language_model.v_proj = nn.Linear(2, 2, bias=False)
            self.language_model.o_proj = nn.Linear(2, 2, bias=False)
            self.language_model.norm = nn.LayerNorm(2)
            self.vision_model = nn.Module()
            self.vision_model.q_proj = nn.Linear(2, 2, bias=False)
            self.embeddings = nn.Embedding(4, 2)
            self.lm_head = nn.Linear(2, 4, bias=False)
            self.config = SimpleNamespace(model_type="mock_model")

        def get_output_embeddings(self):
            return self.lm_head

    return Model()


def test_language_qv_regex_resolves_actual_inner_linear_without_modality_widening():
    config = SimpleNamespace(target_modules=r"language_model\.(q_proj|v_proj)")
    names = resolve_lora_config_targets(config, _model())
    assert names == {"language_model.q_proj.linear", "language_model.v_proj"}
    assert config.target_modules == names


def test_suffix_list_resolves_wrapper_and_direct_linear():
    config = SimpleNamespace(target_modules=["q_proj", "v_proj"])
    assert resolve_lora_config_targets(config, _model()) == {
        "language_model.q_proj.linear", "language_model.v_proj", "vision_model.q_proj"}


def test_all_linear_excludes_output_head_and_embedding_and_deduplicates_inner():
    config = SimpleNamespace(target_modules="all-linear")
    names = resolve_lora_config_targets(config, _model())
    assert names == {"language_model.q_proj.linear", "language_model.v_proj",
                     "language_model.o_proj", "vision_model.q_proj"}
    assert "lm_head" not in names
    assert "embeddings" not in names


def test_exact_materialized_names_resolve_idempotently_for_resume():
    config = SimpleNamespace(target_modules=r"language_model\.(q_proj|v_proj)(\.linear)?")
    model = _model()
    first = resolve_lora_config_targets(config, model)
    assert resolve_lora_config_targets(config, model) == first
    assert first == {"language_model.q_proj.linear", "language_model.v_proj"}


def test_default_mapping_is_materialized_before_adapter_construction(monkeypatch):
    constants = ModuleType("peft.utils.constants")
    constants.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {"mock_model": ["q_proj", "v_proj"]}
    monkeypatch.setitem(sys.modules, "peft.utils.constants", constants)
    config = SimpleNamespace(target_modules=None)
    assert resolve_lora_config_targets(config, _model()) == {
        "language_model.q_proj.linear", "language_model.v_proj", "vision_model.q_proj"}


def test_zero_unsupported_and_invalid_regex_fail_without_model_forward():
    for selector in (["missing"], ["embeddings"], ["norm"]):
        with pytest.raises(ValueError, match="no supported"):
            resolve_lora_config_targets(SimpleNamespace(target_modules=selector), _model())
    with pytest.raises(Exception, match="unterminated"):
        resolve_lora_config_targets(SimpleNamespace(target_modules="["), _model())


def test_explicit_exclusions_apply_at_wrapper_and_inner_names():
    config = SimpleNamespace(target_modules="all-linear", exclude_modules=["q_proj"])
    names = resolve_lora_config_targets(config, _model())
    assert "language_model.q_proj.linear" not in names
    assert "vision_model.q_proj" not in names


def test_all_linear_excludes_output_embedding_with_nonstandard_name():
    model = _model()
    output = model.lm_head
    del model.lm_head
    model.custom_output = output
    model.get_output_embeddings = lambda: output
    names = resolve_lora_config_targets(SimpleNamespace(target_modules="all-linear"), model)
    assert "custom_output" not in names


def test_explicit_output_head_selection_is_not_silently_overridden():
    config = SimpleNamespace(target_modules=["lm_head"])
    assert resolve_lora_config_targets(config, _model()) == {"lm_head"}


def test_unknown_peft_default_fails_with_actionable_message(monkeypatch):
    constants = ModuleType("peft.utils.constants")
    constants.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {}
    monkeypatch.setitem(sys.modules, "peft.utils.constants", constants)
    with pytest.raises(ValueError, match="choose explicit language targets"):
        resolve_lora_config_targets(SimpleNamespace(target_modules=None), _model())
