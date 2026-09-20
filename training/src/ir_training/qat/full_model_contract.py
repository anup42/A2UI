"""Isolated all-parameter E2B QAT contract, not the retained-mobile LoRA lane.

Every text-model parameter is trainable. Weight fake quantization uses the
experimental dense W248 export allocation and new channelwise scales. This is
deliberately NOT Google's static-A8 graph or its private training recipe.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ir_training.qat.mobile_training_seed import (
    EXPECTED_TENSOR_COUNT,
    OFFICIAL_MOBILE_MODEL_ID,
)

WORKFLOW = "e2b_all_parameter_qat_v1"
NUMERIC_POLICY = "all_parameter_qat_safety_v1"
PROFILE = "e2b_all_parameter_dynamic_w248"
EXPECTED_PARAMETER_TENSOR_COUNT = 506
EXPECTED_PERSISTENT_BUFFER_NAMES = frozenset(
    f"model.layers.{layer}.layer_scalar" for layer in range(35)
)
# HF module paths, not converter operation names. These implement the SAME
# allocation as export/gemma4_mixed248.py, including channelwise embeddings.
MODULE_BITS = {
    r"(^|\.)embed_tokens_per_layer(?:\.linear)?$": 4,
    r"(^|\.)embed_tokens(?:\.linear)?$": 2,
    r"(^|\.)lm_head(?:\.linear)?$": 2,
    r"\.layers\.(?:1[5-9]|2[0-9]|3[0-4])\.mlp\.(?:gate|up|down)_proj(?:\.linear)?$": 2,
    r"\.layers\.(?:[0-9]|1[0-4])\.mlp\.(?:gate|up|down)_proj(?:\.linear)?$": 4,
    r"\.self_attn\.(?:q|k|v|o)_proj(?:\.linear)?$": 4,
    r"(^|\.)per_layer_(?:model_projection|input_gate|projection)(?:\.linear)?$": 8,
}


def is_full_qat(config: dict) -> bool:
    return (config.get("run") or {}).get("purpose") == WORKFLOW


def full_parameter_autocast(model: Any):
    """Match BF16 training compute during this lane's probes and generation."""
    from contextlib import nullcontext
    if not getattr(model, "_a2ui_full_parameter_amp", False):
        return nullcontext()
    import torch
    device = next(model.parameters()).device
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def _preflight_path(config: dict, rank: int) -> Path:
    from ir_training.common.config import resolve_path, training_root
    return resolve_path(config["run"]["output_dir"], training_root()) / f"full_optimizer_preflight_rank{rank}.json"


def write_optimizer_preflight(config: dict, config_path: Path | None, probe: dict) -> dict:
    import os

    from ir_training.pipeline.golden_training import _write, sha256
    if config_path is None or probe.get("passed") is not True:
        raise ValueError("Disposable optimizer preflight requires a file-bound config and passing probe")
    rank = int(os.environ.get("RANK", "0"))
    report = {"training_config_sha256": sha256(config_path), "rank": rank,
              "world_size": int(os.environ.get("WORLD_SIZE", "1")), "probe": probe}
    path = _preflight_path(config, rank)
    _write(path, report)
    return {"path": str(path), "sha256": sha256(path), **report}


def require_optimizer_preflight(config: dict, config_path: Path | None) -> dict:
    """Each real training rank must consume its own fresh-process probe evidence."""
    import os

    from ir_training.pipeline.golden_training import sha256
    rank = int(os.environ.get("RANK", "0"))
    path = _preflight_path(config, rank)
    report = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    probe = report.get("probe") or {}
    if (config_path is None or report.get("training_config_sha256") != sha256(config_path)
            or report.get("rank") != rank
            or report.get("world_size") != int(os.environ.get("WORLD_SIZE", "1"))
            or probe.get("passed") is not True or probe.get("disposable_optimizer_steps") != 1
            or probe.get("checkpoint_writes") != 0 or probe.get("model_must_not_be_reused") is not True):
        raise ValueError("Full-parameter training requires a matching disposable optimizer preflight on every rank")
    return {"path": str(path), "sha256": sha256(path), **report}


def _qat_config() -> dict:
    return {
        "enabled": True, "profile": PROFILE, "quantizer": "ste_ai_edge",
        "scale_mode": "dynamic", "weight_bits": 8, "activation_bits": 32,
        "weight_symmetric": True, "activation_symmetric": True,
        "weight_per_channel": True, "weight_axis": 0, "group_size": None,
        "quantize_embeddings": True, "exclude_modules": [],
        "only_base_layers": True, "effective_merged_weight": False,
        "effective_lora_only": False, "ste_gradient": "identity", "eps": 1e-9,
        "module_quant_configs": dict(MODULE_BITS),
        "final_quantization": "gemma4_dense_mixed248_channelwise_experimental",
        "final_runtime_validation_required": True,
    }


def configure_full_qat(config: dict) -> dict:
    """Copy a resolved mobile config; never mutate its LoRA source/defaults."""
    result = copy.deepcopy(config)
    result.setdefault("run", {})["purpose"] = WORKFLOW
    result.pop("lora", None)
    result.pop("qat_mtp", None)
    result["model"].update(dtype="float32", load_in_4bit=False, device_map="none",
                           architecture_preflight_required=True, require_exact_checkpoint_keys=True)
    result["training"].update(
        method="full_finetune_qat", full_parameter_training=True,
        mixed_precision="bf16", optim="adafactor", weight_decay=0.0, max_grad_norm=0.0,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        backward_preflight=True, lora_diagnostics_steps=0, refuse_resume=True,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
    )
    result["qat"] = _qat_config()
    result.setdefault("preflight", {}).update(
        numeric_policy=NUMERIC_POLICY, require_zero_adapter_parity=False,
        require_initial_loss_gate=True, require_greedy_determinism=True,
        max_initial_completion_loss=15.0,
    )
    validate_full_qat_config(result)
    return result


def validate_full_qat_config(config: dict) -> None:
    """Narrow allowlist. No fallback to a LoRA or a subset-of-parameters run."""
    if not is_full_qat(config):
        raise ValueError("All-parameter QAT requires its separate workflow")
    model, training = config.get("model") or {}, config.get("training") or {}
    if config.get("lora") or config.get("qat_mtp"):
        raise ValueError("All-parameter QAT cannot contain LoRA or MTP settings")
    required_model = {"model_id": OFFICIAL_MOBILE_MODEL_ID, "dtype": "float32",
                      "load_in_4bit": False, "architecture_preflight_required": True,
                      "require_exact_checkpoint_keys": True, "device_map": "none"}
    required_training = {
        "method": "full_finetune_qat", "full_parameter_training": True,
        "mixed_precision": "bf16", "optim": "adafactor", "weight_decay": 0.0, "max_grad_norm": 0.0,
        "gradient_checkpointing": True, "backward_preflight": True,
        "per_device_train_batch_size": 1, "per_device_eval_batch_size": 1,
        "refuse_resume": True,
    }
    for section, expected in ((model, required_model), (training, required_training)):
        for key, value in expected.items():
            if section.get(key) != value or isinstance(section.get(key), bool) != isinstance(value, bool):
                raise ValueError(f"All-parameter QAT requires {key}={value!r}")
    for key in ("mobile_training_seed_manifest", "mobile_qparams_contract", "model_source"):
        if not model.get(key):
            raise ValueError(f"All-parameter QAT requires verified seed {key}")
    if training.get("gradient_checkpointing_kwargs") != {"use_reentrant": False}:
        raise ValueError("All-parameter QAT requires non-reentrant gradient checkpointing")
    if training.get("resume_from_checkpoint") or config.get("resume_from_checkpoint"):
        raise ValueError("Use a fresh full-parameter run; LoRA resume/optimizer state is incompatible")
    if config.get("qat") != _qat_config():
        raise ValueError("All-parameter QAT requires the exact dense dynamic W248 contract")
    preflight = config.get("preflight") or {}
    if preflight.get("numeric_policy") != NUMERIC_POLICY:
        raise ValueError("All-parameter QAT requires its explicit safety policy")
    if preflight.get("require_zero_adapter_parity") is not False:
        raise ValueError("Full-parameter training has no zero-adapter initialization")
    for name in ("require_initial_loss_gate", "require_greedy_determinism"):
        if preflight.get(name) is not True:
            raise ValueError(f"All-parameter QAT requires {name}")
    limit = preflight.get("max_initial_completion_loss")
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or not 0 < limit <= 15:
        raise ValueError("All-parameter QAT requires an absolute finite loss ceiling <= 15")
    for name, minimum in (("rows", 1), ("logit_probe_tokens", 1), ("greedy_probe_rows", 1),
                          ("greedy_probe_new_tokens", 8), ("min_greedy_tokens", 8)):
        if type(preflight.get(name)) is not int or preflight[name] < minimum:
            raise ValueError(f"All-parameter QAT requires {name} >= {minimum}")
    if preflight["min_greedy_tokens"] > preflight["greedy_probe_new_tokens"]:
        raise ValueError("Greedy minimum exceeds probe token budget")


def verify_full_qat_coverage(model: Any, summary: dict) -> dict:
    """Every dense Linear/Embedding, no frozen parameters or uncovered weights."""
    import re

    import torch
    expected = {}
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            matches = [bits for pattern, bits in MODULE_BITS.items() if re.search(pattern, name)]
            if len(matches) != 1:
                raise ValueError(f"Unclassified/ambiguous full-QAT matrix: {name}")
            expected[name] = matches[0]
    if not expected or summary.get("wrapped_weight_bits_by_module") != expected:
        raise ValueError("Full-QAT matrix coverage/bit allocation is incomplete")
    if summary.get("wrapped_effective_lora_count") != 0 or summary.get("retained_qparams_binding_count") != 0:
        raise ValueError("Full-QAT cannot use retained LoRA wrappers")
    if any(not p.requires_grad or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Full-QAT froze or downcast a trainable parameter")
    return {"verified": True, "matrices": expected, "matrix_count": len(expected)}


def _shape_digest(shapes: dict) -> str:
    return hashlib.sha256(json.dumps(shapes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def seed_shapes(config: dict) -> dict:
    from ir_training.common.config import resolve_path, training_root
    path = resolve_path(config["model"]["mobile_training_seed_manifest"], training_root())
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mappings = manifest["transformation"]["tensor_mappings"]
    shapes = {item["output_key"]: list(item["output_shape"]) for item in mappings}
    if len(shapes) != EXPECTED_TENSOR_COUNT or len(mappings) != EXPECTED_TENSOR_COUNT:
        raise ValueError("Full-QAT requires the complete 541-tensor text seed")
    return shapes


def split_full_model_state_shapes(shapes: dict) -> tuple[dict, dict]:
    """Partition the pinned text seed, not arbitrary state, into parameters/buffers."""
    buffers = {name: shape for name, shape in shapes.items() if name in EXPECTED_PERSISTENT_BUFFER_NAMES}
    if (len(shapes) != EXPECTED_TENSOR_COUNT
            or set(buffers) != EXPECTED_PERSISTENT_BUFFER_NAMES
            or any(shape != [1] for shape in buffers.values())
            or {name for name in shapes if name.endswith(".layer_scalar")} != set(buffers)):
        raise ValueError("Full-QAT requires exactly 35 persistent layer_scalar buffers of shape [1] in 541 state tensors")
    parameters = {name: shape for name, shape in shapes.items() if name not in buffers}
    if len(parameters) != EXPECTED_PARAMETER_TENSOR_COUNT:
        raise ValueError("Full-QAT requires exactly 506 named parameter state entries")
    return parameters, buffers


def _buffer_value_sha256(tensor: Any) -> str:
    import torch

    if (isinstance(tensor, torch.nn.Parameter) or tensor.dtype != torch.float32
            or tensor.requires_grad or list(tensor.shape) != [1]
            or not bool(torch.isfinite(tensor).all().item())):
        raise ValueError("Full-QAT layer_scalar buffers must be finite non-trainable FP32 tensors of shape [1]")
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()).hexdigest()


def _model_inventory_evidence(shapes: dict, buffer_hashes: dict[str, str]) -> dict:
    parameters, buffers = split_full_model_state_shapes(shapes)
    if (set(buffer_hashes) != set(buffers)
            or any(not isinstance(value, str) or len(value) != 64
                   or any(char not in "0123456789abcdef" for char in value) for value in buffer_hashes.values())):
        raise ValueError("Full-QAT requires value hashes for exactly the expected persistent buffers")
    return {"verified": True, "state_tensor_count": len(shapes), "state_shapes_sha256": _shape_digest(shapes),
            "named_parameter_count": len(parameters), "parameter_shapes_sha256": _shape_digest(parameters),
            "persistent_buffer_count": len(buffers), "persistent_buffer_shapes_sha256": _shape_digest(buffers),
            "persistent_buffer_value_sha256": dict(sorted(buffer_hashes.items()))}


def verify_full_model_inventory(model: Any, config: dict) -> dict:
    expected = seed_shapes(config)
    expected_parameters, expected_buffers = split_full_model_state_shapes(expected)
    actual = {name: list(tensor.shape) for name, tensor in model.state_dict().items()}
    if actual != expected:
        raise ValueError("Loaded full model is not the complete reconstructed text seed")
    try:
        named_parameters = list(model.named_parameters(remove_duplicate=False))
    except TypeError as exc:  # pragma: no cover - unsupported old torch
        raise ValueError("Full-QAT inventory requires alias-aware named parameters") from exc
    parameter_shapes = {name: list(parameter.shape) for name, parameter in named_parameters}
    if len(parameter_shapes) != len(named_parameters) or parameter_shapes != expected_parameters:
        raise ValueError(
            "Full-QAT requires exactly the 506 named parameter entries/aliases, "
            "excluding the 35 persistent layer_scalar buffers"
        )
    # Inspect registration rather than treating every named buffer as persistent:
    # Gemma4 also has derived rotary/embed-scale buffers that are not saved.
    persistent = {}
    for prefix, module in model.named_modules(remove_duplicate=False):
        for local_name, tensor in module._buffers.items():
            if local_name not in module._non_persistent_buffers_set:
                name = f"{prefix}.{local_name}" if prefix else local_name
                persistent[name] = tensor
    if set(persistent) != set(expected_buffers):
        raise ValueError("Full-QAT persistent buffer registrations differ from the exact 35 layer_scalar buffers")
    buffer_hashes = {name: _buffer_value_sha256(tensor) for name, tensor in persistent.items()}
    return _model_inventory_evidence(actual, buffer_hashes)


def _seed_matrix_allocation(shapes: dict[str, list[int]]) -> dict[str, int]:
    """Derive the exact dense W248 matrix inventory from the verified seed."""
    import re

    matrices: dict[str, int] = {}
    for weight_name, shape in shapes.items():
        if not weight_name.endswith(".weight") or len(shape) != 2:
            continue
        module_name = weight_name.removesuffix(".weight")
        matches = [bits for pattern, bits in MODULE_BITS.items() if re.search(pattern, module_name)]
        if len(matches) != 1:
            raise ValueError(f"Unclassified/ambiguous full-QAT seed matrix: {module_name}")
        matrices[module_name] = matches[0]
    if not matrices:
        raise ValueError("Full-QAT seed contains no classified dense matrices")
    return matrices


def _verify_scope_against_seed(scope: dict, shapes: dict[str, list[int]], metadata: dict) -> None:
    """Prove scope covers each named parameter shape (not persistent buffers)."""
    records = scope.get("parameters")
    if not isinstance(records, list) or not records:
        raise ValueError("Checkpoint full-parameter scope has no parameter records")
    aliases_seen: set[str] = set()
    canonical: list[str] = []
    unique_numel = 0
    for item in records:
        if not isinstance(item, dict):
            raise TypeError("Checkpoint full-parameter scope has an invalid parameter record")
        name = item.get("canonical_name")
        aliases = item.get("aliases")
        if not isinstance(name, str) or not isinstance(aliases, list) or not aliases or aliases[0] != name:
            raise ValueError("Checkpoint full-parameter scope has invalid canonical aliases")
        if len(aliases) != len(set(aliases)) or aliases_seen.intersection(aliases):
            raise ValueError("Checkpoint full-parameter scope repeats a parameter alias")
        if any(alias not in shapes for alias in aliases):
            raise ValueError("Checkpoint full-parameter scope contains a non-seed alias")
        alias_shapes = {tuple(shapes[alias]) for alias in aliases}
        if len(alias_shapes) != 1:
            raise ValueError("Tied parameter aliases have inconsistent seed shapes")
        expected_numel = math.prod(next(iter(alias_shapes)))
        if (item.get("numel") != expected_numel
                or item.get("trainable_dtype") != "float32"):
            raise ValueError("Checkpoint full-parameter scope has incorrect numel or dtype")
        aliases_seen.update(aliases)
        canonical.append(name)
        unique_numel += expected_numel
    if aliases_seen != set(shapes):
        raise ValueError("Checkpoint trainable scope does not cover the complete seed inventory")
    if (scope.get("unique_parameter_count") != len(records)
            or scope.get("named_parameter_count") != len(aliases_seen)
            or scope.get("tied_alias_count") != len(aliases_seen) - len(records)
            or scope.get("trainable_numel") != unique_numel):
        raise ValueError("Checkpoint full-parameter scope counts are inconsistent with the seed")
    counts = metadata.get("trainable_parameter_counts") or {}
    if (len(canonical) != len(set(canonical))
            or set(canonical) != set(metadata.get("trainable_parameter_names") or [])
            or counts.get("trainable") != unique_numel
            or counts.get("total") != unique_numel):
        raise ValueError("Checkpoint trainable parameter metadata is incomplete")


def validate_full_qat_checkpoint(config: dict, metadata: dict, checkpoint: Path) -> dict:
    """Additional dense-export gate; callers still verify config/data/file hashes."""
    from safetensors import safe_open

    from ir_training.qat.mobile_training_seed import (
        verify_configured_mobile_training_seed,
    )
    from ir_training.qat.numeric_preflight import numeric_preflight_provenance
    validate_full_qat_config(config)
    seed = verify_configured_mobile_training_seed(config["model"], require_materialized=True)
    if not seed.get("verified") or metadata.get("checkpoint_kind") != "full_model":
        raise ValueError("Full-QAT export requires verified seed and a full-model checkpoint")
    scope = metadata.get("full_parameter_scope") or {}
    if (scope.get("scope") != "all_model_parameters" or scope.get("verified") is not True
            or scope.get("master_trainable_dtype") != "float32"
            or scope.get("frozen_parameter_count") != 0 or scope.get("adapter_parameter_count") != 0):
        raise ValueError("Checkpoint lacks verified all-parameter FP32 scope")
    expected = seed_shapes(config)
    parameter_shapes, buffer_shapes = split_full_model_state_shapes(expected)
    _verify_scope_against_seed(scope, parameter_shapes, metadata)
    names = [item["canonical_name"] for item in scope["parameters"]]
    qat = metadata.get("qat") or {}
    coverage = metadata.get("full_qat_coverage") or {}
    from ir_training.qat.fake_quant import QATSpec
    # Training metadata is persisted as JSON, which converts tuples in the
    # dataclass representation to lists. Compare the canonical persisted form.
    expected_spec = json.loads(json.dumps(QATSpec.from_config(config).to_dict()))
    expected_matrices = _seed_matrix_allocation(expected)
    if (not coverage.get("verified") or not coverage.get("matrices")
            or coverage.get("matrices") != expected_matrices
            or coverage.get("matrix_count") != len(expected_matrices)
            or qat.get("wrapped_weight_bits_by_module") != expected_matrices
            or qat.get("spec") != expected_spec
            or qat.get("true_fake_quant") is not True
            or qat.get("wrapped_effective_lora_count") != 0
            or qat.get("retained_qparams_binding_count") != 0):
        raise ValueError("Checkpoint lacks complete all-parameter QAT coverage")
    numeric = numeric_preflight_provenance(config, metadata.get("numeric_preflight") or {})
    if not numeric.get("verified") or (metadata.get("backward_preflight") or {}).get("status") != "passed":
        raise ValueError("Full-QAT numeric/backward preflight evidence is missing or failed")
    optimizer = metadata.get("full_optimizer_preflight") or {}
    probe = optimizer.get("probe") or {}
    optimizer_spec = probe.get("optimizer") or {}
    optimizer_scope = probe.get("scope") or {}
    if (optimizer.get("training_config_sha256") != metadata.get("training_config_sha256")
            or probe.get("passed") is not True or probe.get("disposable_optimizer_steps") != 1
            or probe.get("checkpoint_writes") != 0
            or probe.get("model_must_not_be_reused") is not True
            or probe.get("disposable_worker_required") is not True
            or optimizer_spec.get("name") != "Adafactor"
            or optimizer_spec.get("scale_parameter") is not False
            or optimizer_spec.get("relative_step") is not False
            or optimizer_spec.get("warmup_init") is not False
            or optimizer_spec.get("weight_decay") != 0.0
            or optimizer_spec.get("external_max_grad_norm_required") != 0.0
            or not isinstance(optimizer_spec.get("state_tensor_count"), int)
            or optimizer_spec.get("state_tensor_count", 0) <= 0
            or optimizer_spec.get("state_numel", 0) <= 0
            or optimizer_scope.get("unique_parameter_count") != scope.get("unique_parameter_count")
            or optimizer_scope.get("trainable_numel") != scope.get("trainable_numel")
            or optimizer_scope.get("all_trainable_fp32") is not True
            or optimizer_scope.get("all_gradients_finite") is not True
            or optimizer_scope.get("all_parameters_finite_after_step") is not True):
        raise ValueError("Full-QAT checkpoint lacks a bound successful optimizer preflight")
    actual = {}
    buffer_hashes = {}
    for path in sorted(checkpoint.glob("*.safetensors")):
        if not path.name.startswith("model"):
            raise ValueError("Full checkpoint contains unexpected adapter/auxiliary weights")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():  # noqa: SIM118 -- Safetensors is not an iterable mapping
                view = handle.get_slice(name)
                if name in actual or view.get_dtype() != "F32":
                    raise ValueError("Full checkpoint must contain each FP32 tensor exactly once")
                actual[name] = list(view.get_shape())
                if name in buffer_shapes:
                    buffer_hashes[name] = _buffer_value_sha256(handle.get_tensor(name))
    inventory = metadata.get("full_model_inventory") or {}
    if (actual != expected or inventory != _model_inventory_evidence(expected, buffer_hashes)):
        raise ValueError("Full checkpoint lost/changed a seed tensor or inventory evidence")
    if not set(names).issubset(actual):
        raise ValueError("Trainable parameter absent from saved full checkpoint")
    return {"verified": True, "workflow": WORKFLOW, "state_tensor_count": len(actual),
            "named_parameter_count": len(parameter_shapes), "persistent_buffer_count": len(buffer_shapes),
            "state_shapes_sha256": _shape_digest(actual), "trainable_numel": scope["trainable_numel"],
            "official_graph": False, "official_retained_scale_export": False, "mtp_exported": False}
