from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.qat import mobile_seed_architecture as architecture


class _FakeTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.device = SimpleNamespace(type="meta")


class Gemma4TextConfig:
    model_type = "gemma4_text"
    architectures = ["Gemma4ForCausalLM"]


class Gemma4ForCausalLM:
    def __init__(self, inventory: dict[str, tuple[int, ...]]) -> None:
        self._inventory = inventory

    def state_dict(self) -> dict[str, _FakeTensor]:
        return {key: _FakeTensor(shape) for key, shape in self._inventory.items()}


class _FakeMetaDevice:
    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None


class _FakeSlice:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self._shape = shape

    def get_shape(self) -> tuple[int, ...]:
        return self._shape

    def get_dtype(self) -> str:
        return "BF16"


class _FakeSafeOpen:
    def __init__(self, inventory: dict[str, tuple[int, ...]]) -> None:
        self._inventory = inventory

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def keys(self):
        return self._inventory.keys()

    def get_slice(self, key: str) -> _FakeSlice:
        return _FakeSlice(self._inventory[key])


def _fake_transformers(inventory: dict[str, tuple[int, ...]]):
    class AutoConfig:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return Gemma4TextConfig()

    class AutoModelForCausalLM:
        @staticmethod
        def from_config(*_args, **_kwargs):
            return Gemma4ForCausalLM(inventory)

    return SimpleNamespace(
        __version__="5.10.1",
        AutoConfig=AutoConfig,
        AutoModelForCausalLM=AutoModelForCausalLM,
    )


def test_compare_architecture_inventories_is_exact_and_fail_closed():
    exact = architecture.compare_architecture_inventories(
        {"a": (2, 3), "b": (4,)},
        {"a": (2, 3), "b": (4,)},
    )
    assert exact["exact"] is True
    assert exact["checkpoint_inventory_sha256"] == exact[
        "framework_inventory_sha256"
    ]

    mismatch = architecture.compare_architecture_inventories(
        {"a": (2, 4), "unexpected": (1,)},
        {"a": (2, 3), "missing": (1,)},
    )
    assert mismatch["exact"] is False
    assert mismatch["missing_checkpoint_keys"] == ["missing"]
    assert mismatch["unexpected_checkpoint_keys"] == ["unexpected"]
    assert mismatch["mismatched_shapes"][0]["key"] == "a"


def test_checkpoint_inventory_rejects_wrong_index_to_shard_mapping(tmp_path):
    first = tmp_path / "model-00001-of-00002.safetensors"
    second = tmp_path / "model-00002-of-00002.safetensors"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"weight": first.name, "other": second.name}}
        ),
        encoding="utf-8",
    )

    def safe_open_fn(path, **_kwargs):
        return (
            _FakeSafeOpen({"weight": (1,)})
            if Path(path) == second
            else _FakeSafeOpen({"other": (1,)})
        )

    with pytest.raises(
        architecture.MobileSeedArchitectureError,
        match="wrong shard",
    ):
        architecture._checkpoint_inventory(
            tmp_path,
            safe_open_fn=safe_open_fn,
        )


def test_mobile_seed_architecture_preflight_compares_all_541_headers(
    tmp_path, monkeypatch
):
    inventory = {
        f"model.weight_{index:03d}": (index + 1, 2)
        for index in range(architecture.EXPECTED_TENSOR_COUNT)
    }
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"header-only-test-placeholder")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {key: shard.name for key in inventory}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        architecture,
        "verify_configured_mobile_training_seed",
        lambda *_args, **_kwargs: {
            "required": True,
            "verified": True,
            "manifest_sha256": "a" * 64,
            "checks": {"verified_fixture": True},
        },
    )
    torch_module = SimpleNamespace(device=lambda _name: _FakeMetaDevice())

    report = architecture.validate_mobile_seed_architecture(
        {
            "model_id": "google/gemma-4-E2B-it-qat-mobile-transformers",
            "model_source": str(tmp_path),
            "architecture_preflight_required": True,
            "require_exact_checkpoint_keys": True,
        },
        base=tmp_path,
        torch_module=torch_module,
        transformers_module=_fake_transformers(inventory),
        safe_open_fn=lambda *_args, **_kwargs: _FakeSafeOpen(inventory),
    )

    assert report["verified"] is True
    assert report["model_class"] == "Gemma4ForCausalLM"
    assert report["comparison"]["checkpoint_tensor_count"] == 541
    assert report["comparison"]["framework_state_count"] == 541
    assert report["comparison"]["exact"] is True
    assert report["model_weights_loaded"] is False
    assert report["forward_executed"] is False
    assert report["training_executed"] is False


def test_mobile_seed_architecture_preflight_rejects_old_transformers(
    tmp_path, monkeypatch
):
    inventory = {
        f"model.weight_{index:03d}": (1,)
        for index in range(architecture.EXPECTED_TENSOR_COUNT)
    }
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"placeholder")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: shard.name for key in inventory}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        architecture,
        "verify_configured_mobile_training_seed",
        lambda *_args, **_kwargs: {
            "required": True,
            "verified": True,
            "manifest_sha256": "b" * 64,
            "checks": {"verified_fixture": True},
        },
    )
    transformers_module = _fake_transformers(inventory)
    transformers_module.__version__ = "5.9.0"

    report = architecture.validate_mobile_seed_architecture(
        {
            "model_id": "google/gemma-4-E2B-it-qat-mobile-transformers",
            "model_source": str(tmp_path),
            "architecture_preflight_required": True,
            "require_exact_checkpoint_keys": True,
        },
        base=tmp_path,
        torch_module=SimpleNamespace(device=lambda _name: _FakeMetaDevice()),
        transformers_module=transformers_module,
        safe_open_fn=lambda *_args, **_kwargs: _FakeSafeOpen(inventory),
    )

    assert report["verified"] is False
    assert report["checks"]["transformers_minimum_version"] is False
