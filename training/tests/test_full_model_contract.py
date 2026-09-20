from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.config import load_yaml
from ir_training.qat import full_model_contract as contract
from ir_training.qat.fake_quant import QATSpec, prepare_qat_model
from ir_training.qat.numeric_preflight import (
    numeric_preflight_provenance,
    resolve_numeric_policy,
)
from ir_training.qat.workflow import validate_qat_config
from ir_training.train.recipe import validate_sft_recipe
from ir_training.train.sft import (
    _compare_initial_greedy_reports,
    _compare_initial_numeric_reports,
)


def full_config():
    return contract.configure_full_qat(load_yaml(ROOT / "configs/models/gemma4_e2b_mobile_seed_ir_qat_sft.yaml"))


def numeric_evidence(config):
    def probe(loss, ids):
        return {"completion_loss": loss, "row_losses": [loss], "rows_checked": 1,
                "completion_tokens": 8, "logits_finite": True, "top1_probe_ids": ids,
                "top_token_probe_sha256": str(loss)}

    def greedy(ids, repeats):
        return {"rows_checked": 1, "repeats": repeats, "min_new_tokens": 8,
                "max_new_tokens": 32, "deterministic": True,
                "generated_token_ids": [[ids.copy()] for _ in range(repeats)],
                "generated_token_counts": [[len(ids)] for _ in range(repeats)]}

    before, after = list(range(8)), list(range(8, 16))
    result = _compare_initial_numeric_reports(probe(1.0, before), probe(1.4, after),
        preflight_cfg=config["preflight"], qat_enabled=True)
    result["greedy_generation"] = _compare_initial_greedy_reports(greedy(before, 1), greedy(after, 2),
        preflight_cfg=config["preflight"], qat_enabled=True)
    result.update(adapter_initialization_mode="full_model", zero_adapter_initialization={
        "required": False, "reason": "full_finetune", "verified_zero_delta": False})
    return result


def test_recipe_is_isolated_and_original_unchanged():
    original = load_yaml(ROOT / "configs/models/gemma4_e2b_mobile_seed_ir_qat_sft.yaml")
    snapshot = copy.deepcopy(original)
    config = contract.configure_full_qat(original)
    assert original == snapshot
    assert original["training"]["method"] == "qat_lora_sft"
    assert config["training"]["method"] == "full_finetune_qat"
    assert "lora" not in config and "qat_mtp" not in config
    assert validate_sft_recipe(config) == "full_finetune_qat"
    assert validate_qat_config(config) == []
    assert resolve_numeric_policy(config) == contract.NUMERIC_POLICY
    assert config["model"]["dtype"] == "float32"
    assert config["training"]["mixed_precision"] == "bf16"
    assert config["training"]["optim"] == "adafactor"


@pytest.mark.parametrize("section,key,value", [
    ("training", "method", "qat_lora_sft"), ("training", "full_parameter_training", False),
    ("training", "mixed_precision", "auto"), ("training", "optim", "adamw_torch"),
    ("training", "max_grad_norm", 1.0), ("training", "backward_preflight", False),
    ("model", "dtype", "bfloat16"), ("model", "load_in_4bit", True),
    ("qat", "scale_mode", "retained_mobile"), ("qat", "quantize_embeddings", False),
    ("qat", "effective_lora_only", True), ("qat", "exclude_modules", ["lm_head"]),
    ("preflight", "numeric_policy", "retained_mobile_safety_v1"),
    ("preflight", "max_initial_completion_loss", float("inf")),
])
def test_recipe_fail_closed(section, key, value):
    config = full_config()
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_sft_recipe(config)
    assert validate_qat_config(config)[0].severity == "error"


def test_numeric_safety_requires_real_evidence_but_not_cross_mode_parity():
    config = full_config()
    report = numeric_evidence(config)
    assert report["cross_mode_comparison"] == "diagnostic"
    assert report["top1_probe_match_fraction"] == 0
    assert numeric_preflight_provenance(config, report)["verified"]
    report["greedy_generation"]["qat_on"]["generated_token_ids"][1][0][0] = 999
    assert not numeric_preflight_provenance(config, report)["verified"]


def test_weight_allocation_matches_dense_export_roles():
    spec = QATSpec.from_config(full_config())
    assert spec.activation_bits == 32
    assert spec.group_size is None
    for layer in range(35):
        for part in ("gate", "up", "down"):
            name = f"model.layers.{layer}.mlp.{part}_proj"
            assert spec.weight_bits_for_module(name) == (4 if layer < 15 else 2)
        for part in ("q", "k", "v", "o"):
            assert spec.weight_bits_for_module(f"model.layers.{layer}.self_attn.{part}_proj") == 4
    for name, bits in {"model.embed_tokens": 2, "model.embed_tokens_per_layer": 4,
                       "lm_head": 2, "model.per_layer_model_projection": 8,
                       "model.layers.34.per_layer_projection": 8,
                       "model.layers.34.per_layer_input_gate": 8}.items():
        assert spec.weight_bits_for_module(name) == bits
        assert spec.group_size_for_module(name) is None


def test_all_parameter_qat_updates_embeddings_norm_head_and_projection():
    torch = pytest.importorskip("torch")
    from ir_training.train.full_parameters import enable_full_parameter_training

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(8, 4)
            self.model.embed_tokens_per_layer = torch.nn.Embedding(8, 4)
            self.model.per_layer_model_projection = torch.nn.Linear(4, 4, bias=False)
            self.model.norm = torch.nn.LayerNorm(4)
            self.lm_head = torch.nn.Linear(4, 8, bias=False)

        def forward(self, ids):
            x = self.model.embed_tokens(ids) + self.model.embed_tokens_per_layer(ids)
            return self.lm_head(self.model.norm(self.model.per_layer_model_projection(x)))

    torch.manual_seed(17)
    model = Tiny()
    enable_full_parameter_training(model)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    controller = prepare_qat_model(model, full_config())
    coverage = contract.verify_full_qat_coverage(model, controller.summary())
    assert coverage["matrix_count"] == 4
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    output = model(torch.tensor([[1, 2, 3, 4]]))
    torch.nn.functional.cross_entropy(output.reshape(-1, 8), torch.tensor([3, 4, 5, 6])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert all(not torch.equal(before[n], p) for n, p in model.named_parameters())
    controller.restore()


def test_probe_receipt_is_config_rank_and_world_size_bound(tmp_path, monkeypatch):
    config = full_config()
    config["run"]["output_dir"] = str(tmp_path / "training")
    path = tmp_path / "config.yaml"
    path.write_text("bound", encoding="utf-8")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="preflight"):
        contract.require_optimizer_preflight(config, path)
    probe = {"passed": True, "disposable_optimizer_steps": 1, "checkpoint_writes": 0,
             "model_must_not_be_reused": True}
    report = contract.write_optimizer_preflight(config, path, probe)
    assert contract.require_optimizer_preflight(config, path) == report
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="preflight"):
        contract.require_optimizer_preflight(config, path)


def test_export_full_inventory_checks_all_541_tensors_and_norm_changes(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from ir_training.qat import mobile_training_seed
    from safetensors.torch import save_file

    config = full_config()
    monkeypatch.setattr(mobile_training_seed, "verify_configured_mobile_training_seed", lambda *a, **k: {"verified": True})
    # A complete tiny state fixture: the test changes a norm, not just matrices.
    matrices = {
        "model.embed_tokens.weight": [8, 4],
        "model.embed_tokens_per_layer.weight": [8, 4],
        "model.per_layer_model_projection.weight": [4, 4],
        "lm_head.weight": [8, 4],
    }
    shapes = {f"param_{i}": [2] for i in range(536)} | {"norm": [2]} | matrices
    monkeypatch.setattr(contract, "seed_shapes", lambda cfg: shapes)
    tensors = {name: torch.ones(shape) for name, shape in shapes.items()}
    tensors["norm"].add_(0.125)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    parameters = [{"canonical_name": name, "aliases": [name], "source_dtype": "bfloat16",
                   "trainable_dtype": "float32", "numel": tensor.numel()}
                  for name, tensor in tensors.items()]
    trainable_numel = sum(item["numel"] for item in parameters)
    scope = {"schema_version": 1, "scope": "all_model_parameters", "verified": True,
             "master_trainable_dtype": "float32", "unique_parameter_count": len(parameters),
             "named_parameter_count": len(parameters), "tied_alias_count": 0,
             "trainable_numel": trainable_numel, "frozen_parameter_count": 0,
             "adapter_parameter_count": 0, "parameters": parameters}
    allocation = contract._seed_matrix_allocation(shapes)
    metadata = {"checkpoint_kind": "full_model", "full_parameter_scope": scope,
        "trainable_parameter_names": list(shapes),
        "trainable_parameter_counts": {"trainable": trainable_numel, "total": trainable_numel},
        "qat": {"true_fake_quant": True,
                "spec": json.loads(json.dumps(QATSpec.from_config(config).to_dict())),
                "wrapped_effective_lora_count": 0, "retained_qparams_binding_count": 0,
                "wrapped_weight_bits_by_module": allocation},
        "full_qat_coverage": {"verified": True, "matrices": allocation,
                              "matrix_count": len(allocation)},
        "numeric_preflight": numeric_evidence(config), "backward_preflight": {"status": "passed"},
        "full_model_inventory": {"verified": True, "state_tensor_count": 541,
                                 "state_shapes_sha256": contract._shape_digest(shapes)},
        "training_config_sha256": "a" * 64,
        "full_optimizer_preflight": {"training_config_sha256": "a" * 64,
            "probe": {"passed": True, "disposable_optimizer_steps": 1,
                "checkpoint_writes": 0, "model_must_not_be_reused": True,
                "disposable_worker_required": True,
                "optimizer": {"name": "Adafactor", "scale_parameter": False,
                    "relative_step": False, "warmup_init": False, "weight_decay": 0.0,
                    "external_max_grad_norm_required": 0.0, "state_tensor_count": 4,
                    "state_numel": trainable_numel},
                "scope": {"unique_parameter_count": 541, "trainable_numel": trainable_numel,
                    "all_trainable_fp32": True, "all_gradients_finite": True,
                    "all_parameters_finite_after_step": True}}}}
    # Exercise the exact JSON round-trip used by checkpoint provenance.
    metadata = json.loads(json.dumps(metadata))
    report = contract.validate_full_qat_checkpoint(config, metadata, tmp_path)
    assert report["verified"] and report["state_tensor_count"] == 541
    assert report["official_graph"] is False
    incomplete_scope = copy.deepcopy(metadata)
    incomplete_scope["full_parameter_scope"]["parameters"].pop()
    with pytest.raises(ValueError, match="complete seed inventory"):
        contract.validate_full_qat_checkpoint(config, incomplete_scope, tmp_path)
    wrong_spec = copy.deepcopy(metadata)
    wrong_spec["qat"]["spec"]["eps"] = 0.5
    with pytest.raises(ValueError, match="QAT coverage"):
        contract.validate_full_qat_checkpoint(config, wrong_spec, tmp_path)
    missing_matrix = copy.deepcopy(metadata)
    missing_matrix["full_qat_coverage"]["matrices"].pop("lm_head")
    with pytest.raises(ValueError, match="QAT coverage"):
        contract.validate_full_qat_checkpoint(config, missing_matrix, tmp_path)
    weak_optimizer = copy.deepcopy(metadata)
    weak_optimizer["full_optimizer_preflight"]["probe"]["optimizer"]["state_tensor_count"] = 0
    with pytest.raises(ValueError, match="optimizer preflight"):
        contract.validate_full_qat_checkpoint(config, weak_optimizer, tmp_path)
    bad = dict(tensors)
    bad.pop("norm")
    save_file(bad, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="lost/changed"):
        contract.validate_full_qat_checkpoint(config, metadata, tmp_path)
    bad["norm"] = torch.ones(2, dtype=torch.bfloat16)
    save_file(bad, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="FP32"):
        contract.validate_full_qat_checkpoint(config, metadata, tmp_path)


def test_scope_seed_validation_accepts_tied_alias_partition():
    shapes = {"model.embed_tokens.weight": [8, 4], "lm_head.weight": [8, 4]}
    scope = {"parameters": [{"canonical_name": "model.embed_tokens.weight",
                              "aliases": ["model.embed_tokens.weight", "lm_head.weight"],
                              "trainable_dtype": "float32", "numel": 32}],
             "unique_parameter_count": 1, "named_parameter_count": 2,
             "tied_alias_count": 1, "trainable_numel": 32}
    metadata = {"trainable_parameter_names": ["model.embed_tokens.weight"],
                "trainable_parameter_counts": {"trainable": 32, "total": 32}}
    contract._verify_scope_against_seed(scope, shapes, metadata)
    scope["parameters"][0]["aliases"] = ["model.embed_tokens.weight"]
    with pytest.raises(ValueError, match="complete seed inventory"):
        contract._verify_scope_against_seed(scope, shapes, metadata)


def test_runtime_inventory_rejects_state_buffer_not_present_as_parameter(monkeypatch):
    torch = pytest.importorskip("torch")
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.ones(2)))
    model.register_buffer("buffer", torch.ones(2))
    shapes = {"weight": [2], "buffer": [2]}
    monkeypatch.setattr(contract, "seed_shapes", lambda config: shapes)
    with pytest.raises(ValueError, match="exactly of named model parameters"):
        contract.verify_full_model_inventory(model, {})
