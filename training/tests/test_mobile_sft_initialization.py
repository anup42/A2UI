"""Verify exact loaded weights and fail closed if a source file changes."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ir_training.train import mobile_sft_init


def test_sft_initialization_verifies_loaded_tensor_values_and_binding(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import save_file

    path = tmp_path / "adapter_model.safetensors"
    saved = {f"layer_{index}.{matrix}": torch.tensor([index + 1], dtype=torch.bfloat16)
             for index in range(205) for matrix in ("lora_A", "lora_B")}
    save_file(saved, str(path))
    live = {name: value.float() for name, value in saved.items()}
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(get_peft_model_state_dict=lambda *_a, **_k: live))
    identity = mobile_sft_init.file_identity(path)
    cfg = tmp_path / "adapter_config.json"
    cfg.write_text("{}", encoding="utf-8")
    record = {"checkpoint": str(tmp_path), "files": {
        "adapter_weights": identity, "adapter_config": mobile_sft_init.file_identity(cfg),
    }}
    loaded = {"checkpoint": str(tmp_path), "adapter_name": "default", "adapter_sha256": identity["sha256"],
              "adapter_pair_count": 205, "finite_adapter_pairs": 205, "nonzero_adapter_pairs": 205}
    mobile_sft_init.require_loaded_weights(object(), record, loaded)
    live["layer_10.lora_B"] += 1
    with pytest.raises(ValueError, match="tensor differs"):
        mobile_sft_init.require_loaded_weights(object(), record, loaded)
    live["layer_10.lora_B"] -= 1
    cfg.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="files changed during loading"):
        mobile_sft_init.require_loaded_weights(object(), record, loaded)
