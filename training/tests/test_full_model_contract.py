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


def optimizer_probe(*, accumulation_steps=4, world_size=2, trainable_numel=100):
    microsteps = contract.OPTIMIZER_PREFLIGHT_STEPS * accumulation_steps
    return {
        "schema_version": 2,
        "probe": contract.OPTIMIZER_PREFLIGHT_KIND,
        "passed": True,
        "disposable_worker_required": True,
        "model_must_not_be_reused": True,
        "checkpoint_writes": 0,
        "disposable_optimizer_steps": contract.OPTIMIZER_PREFLIGHT_STEPS,
        "optimizer": {
            "name": "Adafactor", "learning_rate": 1e-4, "beta1": None,
            "scale_parameter": False, "relative_step": False,
            "warmup_init": False, "weight_decay": 0.0,
            "clip_threshold": 1.0,
            "external_max_grad_norm_required": 0.0,
            "state_tensor_count": 4, "state_numel": trainable_numel,
        },
        "scope": {
            "unique_parameter_count": 506, "trainable_numel": trainable_numel,
            "all_trainable_fp32": True, "all_gradients_finite": True,
            "all_parameters_finite_after_step": True,
        },
        "ddp_probe": {
            "world_size": world_size, "all_reduce_exercised": world_size > 1,
            "gradient_as_bucket_view": True, "sync_each_batch": True,
            "trainer_backend": "transformers", "trainer_path": "checked_causal_lm_trainer",
            "gradient_accumulation_steps": accumulation_steps, "microsteps": microsteps,
            "synchronized_microsteps": microsteps,
            "optimizer_steps": contract.OPTIMIZER_PREFLIGHT_STEPS,
        },
        "memory": {
            "device": "cuda:0", "cuda_local_rank_only": True,
            "ddp_collectives_certified": world_size > 1,
            "baseline_allocated_bytes": 100, "baseline_reserved_bytes": 200,
            "peak_allocated_bytes": 500, "peak_reserved_bytes": 600,
            "device_total_bytes": 1000, "device_free_bytes_min": 200,
            "peak_reserved_fraction": 0.6, "max_reserved_fraction": 0.90,
        },
    }


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
    assert config["training"]["ddp_sync_each_batch"] is True


@pytest.mark.parametrize("section,key,value", [
    ("training", "method", "qat_lora_sft"), ("training", "full_parameter_training", False),
    ("training", "mixed_precision", "auto"), ("training", "optim", "adamw_torch"),
    ("training", "max_grad_norm", 1.0), ("training", "backward_preflight", False),
    ("training", "ddp_sync_each_batch", False),
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
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="preflight"):
        contract.require_optimizer_preflight(config, path)
    probe = optimizer_probe()
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
    shapes = ({f"param_{i}": [2] for i in range(501)} | {"norm": [2]} | matrices
              | {name: [1] for name in contract.EXPECTED_PERSISTENT_BUFFER_NAMES})
    monkeypatch.setattr(contract, "seed_shapes", lambda cfg: shapes)
    tensors = {name: torch.ones(shape) for name, shape in shapes.items()}
    tensors["norm"].add_(0.125)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    parameters = [{"canonical_name": name, "aliases": [name], "source_dtype": "bfloat16",
                   "trainable_dtype": "float32", "numel": tensor.numel()}
                  for name, tensor in tensors.items() if name not in contract.EXPECTED_PERSISTENT_BUFFER_NAMES]
    trainable_numel = sum(item["numel"] for item in parameters)
    scope = {"schema_version": 1, "scope": "all_model_parameters", "verified": True,
             "master_trainable_dtype": "float32", "unique_parameter_count": len(parameters),
             "named_parameter_count": len(parameters), "tied_alias_count": 0,
             "trainable_numel": trainable_numel, "frozen_parameter_count": 0,
             "adapter_parameter_count": 0, "parameters": parameters}
    allocation = contract._seed_matrix_allocation(shapes)
    metadata = {"checkpoint_kind": "full_model", "full_parameter_scope": scope,
        "trainable_parameter_names": [item["canonical_name"] for item in parameters],
        "trainable_parameter_counts": {"trainable": trainable_numel, "total": trainable_numel},
        "qat": {"true_fake_quant": True,
                "spec": json.loads(json.dumps(QATSpec.from_config(config).to_dict())),
                "wrapped_effective_lora_count": 0, "retained_qparams_binding_count": 0,
                "wrapped_weight_bits_by_module": allocation},
        "full_qat_coverage": {"verified": True, "matrices": allocation,
                              "matrix_count": len(allocation)},
        "numeric_preflight": numeric_evidence(config), "backward_preflight": {"status": "passed"},
        "full_model_inventory": contract._model_inventory_evidence(shapes, {
            name: contract._buffer_value_sha256(tensors[name]) for name in contract.EXPECTED_PERSISTENT_BUFFER_NAMES}),
        "training_config_sha256": "a" * 64,
        "full_optimizer_preflight": {"training_config_sha256": "a" * 64,
            "rank": 0, "world_size": 2,
            "probe": optimizer_probe(trainable_numel=trainable_numel)}}
    # Exercise the exact JSON round-trip used by checkpoint provenance.
    metadata = json.loads(json.dumps(metadata))
    report = contract.validate_full_qat_checkpoint(config, metadata, tmp_path)
    assert report["verified"] and report["state_tensor_count"] == 541
    assert report["named_parameter_count"] == 506
    assert report["persistent_buffer_count"] == 35
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
    with pytest.raises(ValueError, match="[Oo]ptimizer preflight"):
        contract.validate_full_qat_checkpoint(config, weak_optimizer, tmp_path)
    for mutate in (
        lambda probe: probe.update(disposable_optimizer_steps=1),
        lambda probe: probe.pop("ddp_probe"),
        lambda probe: probe["ddp_probe"].update(gradient_accumulation_steps=8),
        lambda probe: probe["memory"].update(peak_reserved_fraction=float("nan")),
        lambda probe: probe["memory"].update(
            peak_reserved_bytes=900, peak_reserved_fraction=0.9
        ),
        lambda probe: probe.update(schema_version=2.0),
        lambda probe: probe.update(optimizer=[]),
        lambda probe: probe["memory"].update(device="cuda:1"),
    ):
        invalid = copy.deepcopy(metadata)
        mutate(invalid["full_optimizer_preflight"]["probe"])
        with pytest.raises(ValueError, match="optimizer preflight|Optimizer preflight"):
            contract.validate_full_qat_checkpoint(config, invalid, tmp_path)
    buffer_as_parameter = copy.deepcopy(metadata)
    buffer_name = "model.layers.0.layer_scalar"
    buffer_as_parameter["full_parameter_scope"]["parameters"].append({
        "canonical_name": buffer_name, "aliases": [buffer_name], "numel": 1, "trainable_dtype": "float32"})
    with pytest.raises(ValueError, match="non-seed alias"):
        contract.validate_full_qat_checkpoint(config, buffer_as_parameter, tmp_path)
    bad_buffer_evidence = copy.deepcopy(metadata)
    bad_buffer_evidence["full_model_inventory"]["persistent_buffer_value_sha256"].pop(buffer_name)
    with pytest.raises(ValueError, match="inventory evidence"):
        contract.validate_full_qat_checkpoint(config, bad_buffer_evidence, tmp_path)
    for replacement in (torch.tensor([2.0]), torch.tensor([1.0, 1.0]), torch.tensor([float("nan")]),
                        torch.tensor([1.0], dtype=torch.bfloat16)):
        bad = dict(tensors)
        bad[buffer_name] = replacement
        save_file(bad, str(tmp_path / "model.safetensors"))
        with pytest.raises(ValueError):
            contract.validate_full_qat_checkpoint(config, metadata, tmp_path)
    for remove, add in ((buffer_name, None), (None, "unexpected_buffer")):
        bad = dict(tensors)
        if remove:
            bad.pop(remove)
        if add:
            bad[add] = torch.ones(1)
        save_file(bad, str(tmp_path / "model.safetensors"))
        with pytest.raises(ValueError, match="lost/changed"):
            contract.validate_full_qat_checkpoint(config, metadata, tmp_path)
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


def _runtime_inventory_fixture(torch):
    model = torch.nn.Module()
    for index in range(506):
        model.register_parameter(f"weight_{index}", torch.nn.Parameter(torch.ones(2)))
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(35)])
    for index, layer in enumerate(model.model.layers):
        layer.register_buffer("layer_scalar", torch.tensor([1.0 + index / 64]))
    for index in range(6):
        model.register_buffer(f"derived_{index}", torch.ones(2), persistent=False)
    return model


def test_runtime_inventory_partitions_parameters_and_persistent_buffers(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from ir_training.train.full_parameters import enable_full_parameter_training
    from safetensors.torch import load_file, save_file

    model = _runtime_inventory_fixture(torch)
    shapes = {name: list(tensor.shape) for name, tensor in model.state_dict().items()}
    monkeypatch.setattr(contract, "seed_shapes", lambda config: shapes)
    buffers_before = {name: (id(value), value.clone()) for name, value in model.named_buffers()}
    inventory = contract.verify_full_model_inventory(model, {})
    assert inventory["state_tensor_count"] == 541
    assert inventory["named_parameter_count"] == 506
    assert inventory["persistent_buffer_count"] == 35
    assert len(list(model.named_buffers(remove_duplicate=False))) == 41
    scope = enable_full_parameter_training(model)
    assert scope["named_parameter_count"] == scope["unique_parameter_count"] == 506
    assert scope["trainable_numel"] == 1012  # Buffers do not enter the optimizer/gradient scope.
    for name, buffer in model.named_buffers():
        assert id(buffer) == buffers_before[name][0]
        assert torch.equal(buffer, buffers_before[name][1])
        assert buffer.requires_grad is False
    saved = tmp_path / "model.safetensors"
    save_file(model.state_dict(), str(saved))
    state = load_file(saved)
    assert len(state) == 541 and not any(name.startswith("derived_") for name in state)
    restored = _runtime_inventory_fixture(torch)
    restored.load_state_dict(state, strict=True)
    assert contract.verify_full_model_inventory(restored, {}) == inventory


@pytest.mark.parametrize("mutation", [
    "missing_buffer", "extra_buffer", "wrong_layer", "wrong_shape", "buffer_parameter",
    "parameter_buffer", "nonpersistent_scalar", "persistent_derived", "nonfinite", "wrong_dtype", "requires_grad",
])
def test_runtime_inventory_rejects_incorrect_state_partition(monkeypatch, mutation):
    torch = pytest.importorskip("torch")

    model = _runtime_inventory_fixture(torch)
    shapes = {name: list(tensor.shape) for name, tensor in model.state_dict().items()}
    monkeypatch.setattr(contract, "seed_shapes", lambda config: shapes)
    layer = model.model.layers[0]
    if mutation == "missing_buffer":
        del layer.layer_scalar
    elif mutation == "extra_buffer":
        model.register_buffer("unexpected", torch.ones(1))
    elif mutation == "wrong_layer":
        del layer.layer_scalar
        model.model.layers.append(torch.nn.Module())
        model.model.layers[-1].register_buffer("layer_scalar", torch.ones(1))
    elif mutation == "wrong_shape":
        layer.layer_scalar = torch.ones(2)
    elif mutation == "buffer_parameter":
        layer.layer_scalar = torch.nn.Parameter(layer.layer_scalar)
    elif mutation == "parameter_buffer":
        old = model.weight_0.detach()
        del model.weight_0
        model.register_buffer("weight_0", old)
    elif mutation == "nonpersistent_scalar":
        layer._non_persistent_buffers_set.add("layer_scalar")
    elif mutation == "persistent_derived":
        model._non_persistent_buffers_set.remove("derived_0")
    elif mutation == "nonfinite":
        layer.layer_scalar.fill_(float("nan"))
    elif mutation == "wrong_dtype":
        layer.layer_scalar = layer.layer_scalar.to(torch.bfloat16)
    else:
        layer.layer_scalar.requires_grad_(True)
    with pytest.raises(ValueError):
        contract.verify_full_model_inventory(model, {})


def test_seed_partition_rejects_arbitrary_persistent_buffer_schema():
    shapes = ({f"param_{index}": [2] for index in range(506)}
              | {name: [1] for name in contract.EXPECTED_PERSISTENT_BUFFER_NAMES})
    parameters, buffers = contract.split_full_model_state_shapes(shapes)
    assert len(parameters) == 506 and len(buffers) == 35
    for bad in (shapes | {"extra": [1]}, shapes | {"model.layers.0.layer_scalar": []},
                {name: shape for name, shape in shapes.items() if name != "model.layers.0.layer_scalar"}):
        with pytest.raises(ValueError, match="35 persistent"):
            contract.split_full_model_state_shapes(bad)
