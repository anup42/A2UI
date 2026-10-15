"""Offline merge-source routing checks with no model download or PEFT runtime."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.export import merge_lora


@pytest.mark.parametrize("separate_tokenizer", [False, True])
@pytest.mark.parametrize("explicit_base", [False, True])
def test_tokenizer_fallback_uses_local_source(tmp_path, monkeypatch, separate_tokenizer, explicit_base):
    base = tmp_path / "local-base"
    base.mkdir()
    tokenizer = tmp_path / "local-tokenizer" if separate_tokenizer else base
    tokenizer.mkdir(exist_ok=True)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter fixture")
    config_path = tmp_path / "training.yaml"
    config_path.write_text(yaml.safe_dump({"model": {"model_source": str(base)}}), encoding="utf-8")
    sources = []

    class Merged:
        def save_pretrained(self, output, **kwargs):
            Path(output, "model.safetensors").write_bytes(b"merged fixture")

    class Wrapped:
        def merge_and_unload(self):
            return Merged()

    class Peft:
        @staticmethod
        def from_pretrained(model, source):
            return Wrapped()

    class Processor:
        @staticmethod
        def from_pretrained(source, **kwargs):
            sources.append(source)
            raise ValueError("Text-only model has no processor")

    class Tokenizer:
        @staticmethod
        def from_pretrained(source, **kwargs):
            sources.append(source)
            if source == str(adapter):
                raise ValueError("Adapter has no tokenizer assets")
            assert source == str(tokenizer)
            return Tokenizer()

        def save_pretrained(self, output):
            Path(output, "tokenizer.json").write_text("{}", encoding="utf-8")

    def load_model(source, config):
        assert source == str(base)
        return object()

    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=Peft))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoProcessor=Processor, AutoTokenizer=Tokenizer))
    monkeypatch.setattr(merge_lora, "load_hf_model", load_model)
    merge_lora.merge_lora_adapter(
        "google/gemma-3-270m-it", adapter, tmp_path / "merged",
        base_model_source=base if explicit_base else None, training_config_path=config_path,
        processor_model_id=str(tokenizer) if separate_tokenizer else None,
    )
    assert sources == [str(adapter), str(adapter), str(tokenizer), str(tokenizer)]
