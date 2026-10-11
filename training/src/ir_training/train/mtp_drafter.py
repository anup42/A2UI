"""Train the public Gemma 4 E2B assistant against a frozen target model.

This is a public reconstruction of the target-conditioned autoregressive MTP
objective described by Google.  It deliberately trains only the 23 matrices
that can be transplanted into the released mobile drafter graph.  The target,
RMSNorms, cluster selector, token ordering, and every other assistant value are
frozen so exact-topology export can retain the official non-inventory bytes.

The implementation is explicit and opt-in.  Importing or planning it never
downloads a checkpoint and never starts training.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import random
import re
from collections.abc import Mapping
from contextlib import ExitStack, nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

from ir_training.common.config import load_yaml, resolve_path, training_root
from ir_training.eval.tensorboard_logging import resolve_tensorboard_root
from ir_training.mtp.drafter_contract import (
    OFFICIAL_GEMMA4_E2B_ASSISTANT,
    contract_summary,
    deployment_weight_specs,
    module_bits,
    source_keys,
)
from ir_training.qat.fake_quant import (
    QATController,
    QATSpec,
    fake_quantize_activation,
    fake_quantize_weight,
    qat_numeric_contract,
)


class MTPDrafterTrainingError(RuntimeError):
    """Raised when the public drafter workflow cannot run safely."""


OFFICIAL_MOBILE_MTP_PURPOSE = "official_mobile_mtp_training_v1"


def _official_mobile(config: dict[str, Any]) -> bool:
    return _section(config, "run").get("purpose") == OFFICIAL_MOBILE_MTP_PURPOSE


def _official_target_config(config: dict[str, Any]) -> dict[str, Any]:
    target = _section(config, "target")
    path = resolve_path(str(target.get("training_config") or ""), training_root())
    if not path.is_file():
        raise MTPDrafterTrainingError("Official MTP requires the resolved target training config")
    resolved = load_yaml(path)
    qat = _section(resolved, "qat")
    if not (qat.get("enabled") is True and qat.get("scale_mode") == "retained_mobile"
            and qat.get("expected_effective_lora_modules") == 205
            and qat.get("expected_frozen_activation_modules") == 70
            and qat.get("simulate_frozen_activations") is True
            and qat.get("require_lora_trainable_scope") is True):
        raise MTPDrafterTrainingError("Official MTP requires the unchanged 205-LoRA/70-frozen retained QAT contract")
    training = _section(config, "training")
    if (training.get("require_cuda") is not True
            or training.get("overflow_policy") != "error"
            or training.get("mixed_precision") != "bf16"):
        raise MTPDrafterTrainingError("Official MTP requires CUDA, BF16 compute, and overflow_policy=error")
    model = _section(resolved, "model")
    if training.get("chat_template_kwargs", {}) != (model.get("chat_template_kwargs") or {}):
        raise MTPDrafterTrainingError("MTP chat template kwargs differ from target training")
    if str(target.get("tokenizer_id_or_path") or "") != str(model.get("tokenizer_source") or ""):
        raise MTPDrafterTrainingError("MTP tokenizer must be the target training tokenizer")
    if not target.get("adapter_checkpoint"):
        raise MTPDrafterTrainingError("Official MTP requires the selected target adapter checkpoint")
    return resolved


def official_mobile_source_bindings(config: dict[str, Any]) -> dict[str, str]:
    """Hash local conditioning inputs; recompute before and after adaptation."""
    if not _official_mobile(config):
        raise MTPDrafterTrainingError("Source bindings require the explicit official MTP purpose")
    resolved = _official_target_config(config)
    base = training_root()
    target, run = (_section(config, name) for name in ("target", "run"))
    model = _section(resolved, "model")
    paths = {resolve_path(str(target["training_config"]), base)}
    for key in ("mobile_training_seed_manifest", "mobile_qparams_contract"):
        path = resolve_path(str(model.get(key) or ""), base)
        if not path.is_file():
            raise MTPDrafterTrainingError(f"Required MTP target binding missing: {key}")
        paths.add(path)
    for value in (model.get("model_source"), target.get("adapter_checkpoint"), target.get("model_id_or_path")):
        directory = resolve_path(str(value or ""), base)
        if not directory.is_dir():
            raise MTPDrafterTrainingError(f"Required MTP checkpoint directory missing: {directory}")
        files = [item for item in directory.iterdir() if item.is_file()]
        if not any(item.suffix == ".safetensors" for item in files):
            raise MTPDrafterTrainingError(f"MTP checkpoint has no safetensors: {directory}")
        paths.update(files)
    prepared = resolve_path(str(run["dataset_dir"]), base)
    for name in ("train.jsonl", "val.jsonl", "manifest.json"):
        path = prepared / name
        if not path.is_file():
            raise MTPDrafterTrainingError(f"Required MTP data binding missing: {path}")
        paths.add(path)
    return {str(path.resolve()): _sha256(path) for path in sorted(paths)}


def probe_official_mobile_mtp_environment() -> dict[str, Any]:
    """No weights loaded/downloaded: verify runtime API and complete cached assistant."""
    try:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import (
            Gemma4AssistantForCausalLM,
            Gemma4ForCausalLM,
            Gemma4TextModel,
        )
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4CausalLMOutputWithPast,
        )

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise MTPDrafterTrainingError("Official MTP training requires a CUDA GPU with native BF16")
        target_parameters = inspect.signature(Gemma4ForCausalLM.forward).parameters
        assistant_parameters = inspect.signature(Gemma4AssistantForCausalLM.forward).parameters
        if ("logits_to_keep" not in target_parameters
                or "shared_kv_states" not in Gemma4CausalLMOutputWithPast.__annotations__
                or "shared_kv_states" not in assistant_parameters
                or not callable(Gemma4TextModel.forward)):
            raise MTPDrafterTrainingError("Installed Transformers lacks the required Gemma4 shared-KV/assistant API")
        snapshot = Path(snapshot_download(OFFICIAL_GEMMA4_E2B_ASSISTANT, local_files_only=True))
        if not (snapshot / "config.json").is_file():
            raise MTPDrafterTrainingError("Cached official assistant config is missing")
        cached_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
        if (cached_config.get("model_type") != "gemma4_assistant"
                or (cached_config.get("text_config") or {}).get("num_hidden_layers") != 4):
            raise MTPDrafterTrainingError("Cached official assistant does not declare the four-layer Gemma4 contract")
        index = snapshot / "model.safetensors.index.json"
        if index.is_file():
            names = set(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).values())
        else:
            names = {"model.safetensors"}
        if not names or any(Path(name).is_absolute() or ".." in Path(name).parts
                            or not (snapshot / name).is_file() or (snapshot / name).stat().st_size == 0 for name in names):
            raise MTPDrafterTrainingError("Cached official assistant weights are incomplete")
        return {"passed": True, "model_loaded": False, "downloaded": False,
                "assistant_snapshot": str(snapshot), "assistant_weight_files": sorted(names),
                "shared_kv_api": True, "logits_to_keep": True}
    except (ImportError, OSError, ValueError) as exc:
        raise MTPDrafterTrainingError(
            "Official MTP requires compatible Transformers and the complete cached "
            f"{OFFICIAL_GEMMA4_E2B_ASSISTANT}. Download it explicitly on the training host first. {exc}"
        ) from exc


def _load_official_target(config: dict[str, Any], device: Any) -> tuple[Any, Any, Any]:
    """Load the original seed + frozen selected LoRA, never QAT-wrap dense merge."""
    from peft import PeftModel

    from ir_training.models.registry import create_adapter
    from ir_training.qat.fake_quant import prepare_qat_model
    from ir_training.train.prepared_binding import verify_tokenizer_binding

    resolved = _official_target_config(config)
    model_config = dict(_section(resolved, "model"))
    model_config["device_map"] = None
    adapter = create_adapter(model_config)
    tokenizer = adapter.load_tokenizer()
    verify_tokenizer_binding(resolve_path(config["run"]["dataset_dir"], training_root()),
                             tokenizer, model_config, required=True)
    target = PeftModel.from_pretrained(adapter.load_model(),
                                      config["target"]["adapter_checkpoint"], is_trainable=False)
    base = target.get_base_model()
    if base.__class__.__name__ != "Gemma4ForCausalLM" or getattr(base.config, "model_type", None) != "gemma4_text":
        raise MTPDrafterTrainingError("Official MTP requires the reconstructed Gemma4Text causal target")
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    target.eval().to(device)
    controller = prepare_qat_model(target, resolved)
    try:
        summary = controller.summary()
        if (summary.get("true_fake_quant") is not True
                or summary.get("wrapped_effective_lora_count") != 205
                or summary.get("frozen_activation_module_count") != 70
                or (summary.get("trainable_scope") or {}).get("all_parameters_frozen") is not True):
            raise MTPDrafterTrainingError("Frozen target retained QAT scope is not exactly 205 + 70")
    except BaseException:
        controller.restore()
        raise
    return target, tokenizer, controller


def _check_finite_gradients(model: Any) -> None:
    from ir_training.train.tensor_checks import tensor_finite_and_nonzero

    any_nonzero = False
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        finite, nonzero = tensor_finite_and_nonzero(parameter.grad)
        if not finite:
            raise MTPDrafterTrainingError(f"MTP gradient is nonfinite: {name}")
        any_nonzero = any_nonzero or nonzero
    if not any_nonzero:
        raise MTPDrafterTrainingError("MTP backward produced no nonzero trainable gradients")


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    return value if isinstance(value, dict) else {}


def _looks_local(value: str) -> bool:
    candidate = Path(value).expanduser()
    return bool(
        candidate.is_absolute()
        or value.startswith((".", "outputs/", "outputs\\", "runs/", "runs\\"))
    )


def _model_source(value: Any, base: Path) -> str:
    text = str(value or "").strip()
    if text and _looks_local(text):
        return str(resolve_path(text, base))
    return text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_drafter_training_plan(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    target_model_override: str | Path | None = None,
) -> dict[str, Any]:
    """Build a no-download, no-training plan and validate the mobile contract."""

    base = training_root()
    run_cfg = _section(config, "run")
    target_cfg = _section(config, "target")
    assistant_cfg = _section(config, "assistant")
    training_cfg = _section(config, "training")
    qat_cfg = _section(config, "qat")

    dataset_dir = resolve_path(
        str(run_cfg.get("dataset_dir", "outputs/datasets/stage3_folder_90_10")),
        base,
    )
    output_dir = resolve_path(
        str(run_cfg.get("output_dir", "runs/gemma4_e2b_mtp_drafter_qat")),
        base,
    )
    target_model = _model_source(
        target_model_override
        if target_model_override is not None
        else target_cfg.get("model_id_or_path"),
        base,
    )
    assistant_model = _model_source(
        assistant_cfg.get("model_id_or_path", OFFICIAL_GEMMA4_E2B_ASSISTANT),
        base,
    )
    tokenizer_source = _model_source(
        target_cfg.get("tokenizer_id_or_path") or target_model,
        base,
    )
    best_checkpoint = resolve_path(
        str(
            assistant_cfg.get("best_checkpoint_dir")
            or (output_dir / "best_checkpoint")
        ),
        base,
    )
    final_checkpoint = output_dir / "final_checkpoint"
    train_path = dataset_dir / "train.jsonl"
    val_path = dataset_dir / "val.jsonl"
    issues: list[dict[str, Any]] = []

    if not target_model:
        issues.append(
            {
                "code": "target_model_required",
                "message": "Set target.model_id_or_path to the merged fine-tuned target.",
            }
        )
    elif _looks_local(target_model) and not Path(target_model).exists():
        issues.append(
            {"code": "target_model_missing", "path": target_model}
        )
    if not assistant_model:
        issues.append({"code": "assistant_model_required"})
    elif assistant_model != OFFICIAL_GEMMA4_E2B_ASSISTANT:
        issues.append(
            {
                "code": "assistant_base_must_match_official_public_checkpoint",
                "expected": OFFICIAL_GEMMA4_E2B_ASSISTANT,
                "observed": assistant_model,
            }
        )
    if not train_path.is_file():
        issues.append({"code": "train_split_missing", "path": str(train_path)})
    if _official_mobile(config):
        try:
            _official_target_config(config)
        except (OSError, ValueError, MTPDrafterTrainingError) as exc:
            issues.append({"code": "official_mobile_target_contract", "message": str(exc)})
        if not val_path.is_file():
            issues.append({"code": "official_mtp_validation_required", "path": str(val_path)})

    method = str(training_cfg.get("method") or "")
    if method != "target_conditioned_autoregressive_qat":
        issues.append(
            {
                "code": "unsupported_training_method",
                "expected": "target_conditioned_autoregressive_qat",
                "observed": method,
            }
        )
    draft_steps = int(training_cfg.get("draft_steps", 4) or 0)
    if draft_steps != 4:
        issues.append(
            {
                "code": "draft_steps_must_match_official_verifier",
                "expected": 4,
                "observed": draft_steps,
            }
        )
    if not bool(training_cfg.get("teacher_forcing", True)):
        issues.append(
            {
                "code": "teacher_forcing_required",
                "message": "The reproducible public objective requires teacher_forcing=true.",
            }
        )
    if not bool(training_cfg.get("completion_only_loss", True)):
        issues.append(
            {
                "code": "completion_only_loss_required",
                "message": "Keep drafter adaptation scoped to supervised completions.",
            }
        )
    if int(training_cfg.get("positions_per_sequence", 0) or 0) < 1:
        issues.append({"code": "positions_per_sequence_must_be_positive"})
    batch_size = int(training_cfg.get("per_device_train_batch_size", 1) or 0)
    if batch_size != 1:
        issues.append(
            {
                "code": "batch_size_must_be_one",
                "observed": batch_size,
            }
        )
    gradient_accumulation_steps = int(
        training_cfg.get("gradient_accumulation_steps", 4) or 0
    )
    if gradient_accumulation_steps < 1:
        issues.append({"code": "gradient_accumulation_must_be_positive"})
    epochs = int(training_cfg.get("epochs", 1) or 0)
    if epochs < 1:
        issues.append({"code": "epochs_must_be_positive"})
    max_steps = int(training_cfg.get("max_steps", 0) or 0)
    if max_steps < 0:
        issues.append({"code": "max_steps_cannot_be_negative"})
    max_seq_length = int(training_cfg.get("max_seq_length", 2048) or 0)
    if max_seq_length < 2:
        issues.append({"code": "max_seq_length_must_be_at_least_two"})
    mixed_precision = str(training_cfg.get("mixed_precision", "bf16")).lower()
    if mixed_precision not in {"no", "fp16", "bf16"}:
        issues.append(
            {
                "code": "unsupported_mixed_precision",
                "observed": mixed_precision,
            }
        )
    learning_rate = float(training_cfg.get("learning_rate", 1e-5) or 0.0)
    if learning_rate <= 0.0:
        issues.append({"code": "learning_rate_must_be_positive"})
    if not best_checkpoint.is_relative_to(output_dir):
        issues.append(
            {
                "code": "best_checkpoint_must_be_inside_output_dir",
                "path": str(best_checkpoint),
            }
        )

    precision_assignments: list[dict[str, Any]] = []
    try:
        qat_spec = QATSpec.from_config(config)
        for weight in deployment_weight_specs():
            observed_bits = qat_spec.weight_bits_for_module(weight.module_name)
            precision_assignments.append(
                {
                    "ordinal": weight.ordinal,
                    "module_name": weight.module_name,
                    "official_bits": weight.bits,
                    "qat_bits": observed_bits,
                    "match": observed_bits == weight.bits,
                }
            )
    except Exception as exc:  # noqa: BLE001 - plan should report malformed QAT
        qat_spec = None
        precision_assignments = []
        issues.append({"code": "qat_config_invalid", "error": str(exc)})
    if not bool(qat_cfg.get("enabled", False)):
        issues.append({"code": "qat_must_be_enabled"})
    if str(qat_cfg.get("quantizer") or "").strip().lower() != "ste_ai_edge":
        issues.append({"code": "public_ai_edge_qat_range_required"})
    if int(qat_cfg.get("activation_bits", 0) or 0) != 8:
        issues.append({"code": "activation_int8_qat_required"})
    qat_contract_checks = {
        "weight_symmetric": bool(qat_spec and qat_spec.weight_symmetric),
        "activation_symmetric": bool(qat_spec and qat_spec.activation_symmetric),
        "weight_per_channel": bool(qat_spec and qat_spec.weight_per_channel),
        "weight_axis_zero": bool(qat_spec and qat_spec.weight_axis == 0),
        "no_group_quantization": bool(qat_spec and qat_spec.group_size is None),
        "embedding_qat_enabled": bool(qat_spec and qat_spec.quantize_embeddings),
        "no_excluded_modules": bool(qat_spec and not qat_spec.exclude_modules),
        "only_base_layers": bool(qat_spec and qat_spec.only_base_layers),
        "effective_merged_weight_disabled": bool(
            qat_spec and not qat_spec.effective_merged_weight
        ),
        "public_ai_edge_numeric_contract": bool(
            qat_spec
            and qat_numeric_contract(qat_spec)[
                "public_ai_edge_numeric_contract"
            ]
        ),
    }
    if not all(qat_contract_checks.values()):
        issues.append(
            {
                "code": "qat_numeric_contract_mismatch",
                "checks": qat_contract_checks,
            }
        )
    if not precision_assignments or not all(
        item["match"] for item in precision_assignments
    ):
        issues.append(
            {
                "code": "qat_precision_does_not_match_official_mtp_inventory",
                "assignments": precision_assignments,
            }
        )

    return {
        "schema_version": 1,
        "run_id": str(run_cfg.get("id", "gemma4_e2b_mtp_drafter_qat")),
        "config_path": str(Path(config_path).expanduser().resolve())
        if config_path
        else None,
        "dataset": {
            "directory": str(dataset_dir),
            "train": str(train_path),
            "validation": str(val_path) if val_path.is_file() else None,
            "completion_only": True,
        },
        "target": {
            "model_id_or_path": target_model or None,
            "tokenizer_id_or_path": tokenizer_source or None,
            "frozen": True,
            "conditioning": [
                "target_last_layer_activation",
                "target_shared_kv_states",
                "target_token_embedding",
            ],
        },
        "assistant": {
            "base_model_id_or_path": assistant_model or None,
            "official_reference_id": OFFICIAL_GEMMA4_E2B_ASSISTANT,
            "best_checkpoint": str(best_checkpoint),
            "final_checkpoint": str(final_checkpoint),
            "unmapped_parameters_frozen": True,
            "training_unmapped_constants_source": "public_assistant_checkpoint",
            "deployment_unmapped_constants_source": "official_litertlm_artifact",
            "trainable_source_keys": source_keys(),
        },
        "objective": {
            "method": method,
            "loss": "teacher_forced_completion_cross_entropy",
            "claims_google_distillation_loss": False,
            "draft_steps": draft_steps,
            "teacher_forcing": bool(training_cfg.get("teacher_forcing", True)),
            "constant_position_ids": True,
            "fixed_target_kv_during_each_draft_rollout": True,
            "positions_per_sequence": int(
                training_cfg.get("positions_per_sequence", 4)
            ),
        },
        "optimizer": {
            "epochs": epochs,
            "max_steps": max_steps,
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "mixed_precision": mixed_precision,
            "max_seq_length": max_seq_length,
        },
        "qat": {
            "enabled": bool(qat_cfg.get("enabled", False)),
            "quantizer": qat_cfg.get("quantizer"),
            "activation_bits": qat_cfg.get("activation_bits"),
            "precision_assignments": precision_assignments,
            "numeric_contract_checks": qat_contract_checks,
            "exact_official_inventory_precision": bool(
                precision_assignments
                and all(item["match"] for item in precision_assignments)
            ),
        },
        "mobile_contract": contract_summary(),
        "output_dir": str(output_dir),
        "training_executed": False,
        "private_google_training_recipe_recovered": False,
        "ready": not issues,
        "issues": issues,
        "limitations": [
            "Google does not publish its exact drafter data mixture, loss weights, optimizer, or QAT observer schedule.",
            "This public objective follows the released architecture and target-conditioning behavior, then relies on MTP acceptance and Android throughput gates.",
            "Training starts from the public assistant checkpoint; export retains artifact-native frozen constants outside the 23 mapped matrices, so on-device acceptance validation is mandatory.",
        ],
    }


class DrafterQATController:
    """Restrict STE fake quantization to the 23 deployable assistant matrices."""

    def __init__(self, config: dict[str, Any]):
        self.base_spec = QATSpec.from_config(config)
        self.controller: QATController | None = None
        self._masked_embedder: Any | None = None
        self._masked_forward: Any | None = None

    def prepare(self, model: Any) -> DrafterQATController:
        from torch import nn

        allowed = module_bits()
        # The ordered vocabulary projection reads the tied lm_head weight
        # directly, so keep its alias in the controller and wrap that direct
        # read separately below.
        allowed["lm_head"] = 4
        blocked: list[str] = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)) and name not in allowed:
                blocked.append(rf"^{re.escape(name)}$")
        module_rules = tuple(
            (rf"^{re.escape(name)}$", bits) for name, bits in allowed.items()
        )
        spec = replace(
            self.base_spec,
            module_quant_configs=module_rules,
            modules_to_not_convert=tuple(blocked),
            quantize_embeddings=True,
            exclude_modules=(),
        )
        self.controller = QATController(spec).prepare(model)
        expected_wrapped = set(allowed)
        observed_wrapped = set(self.controller.wrapped_names)
        if observed_wrapped != expected_wrapped:
            self.controller.restore()
            raise MTPDrafterTrainingError(
                "Drafter QAT did not wrap the exact deployable module set: "
                f"missing={sorted(expected_wrapped - observed_wrapped)}, "
                f"unexpected={sorted(observed_wrapped - expected_wrapped)}."
            )

        masked = getattr(model, "masked_embedding", None)
        lm_head = getattr(model, "lm_head", None)
        if masked is None or lm_head is None:
            self.controller.restore()
            raise MTPDrafterTrainingError(
                "Gemma 4 E2B assistant must expose masked_embedding and lm_head."
            )
        original = masked.forward
        head_spec = replace(spec, weight_bits=4)

        def qat_masked_forward(hidden_states: Any, lm_head_weight: Any) -> Any:
            quantized_hidden = fake_quantize_activation(hidden_states, head_spec)
            quantized_weight = fake_quantize_weight(lm_head_weight, head_spec)
            return original(quantized_hidden, quantized_weight)

        masked.forward = qat_masked_forward
        self._masked_embedder = masked
        self._masked_forward = original
        return self

    def restore(self) -> None:
        if self._masked_embedder is not None and self._masked_forward is not None:
            self._masked_embedder.forward = self._masked_forward
        if self.controller is not None:
            self.controller.restore()

    def summary(self) -> dict[str, Any]:
        base = self.controller.summary() if self.controller is not None else {}
        return {
            **base,
            "deployment_weight_count": len(deployment_weight_specs()),
            "deployment_bit_histogram": contract_summary()["bit_histogram"],
            "tied_ordered_lm_head_direct_fake_quant": self._masked_forward is not None,
            "unmapped_quantizable_modules_blocked": True,
        }


def _deployment_trainable_parameters(model: Any) -> dict[str, Any]:
    try:
        parameters = dict(model.named_parameters(remove_duplicate=False))
    except TypeError:  # pragma: no cover - older torch compatibility
        parameters = dict(model.named_parameters())
    missing = [key for key in source_keys() if key not in parameters]
    if missing:
        raise MTPDrafterTrainingError(
            f"Assistant checkpoint is missing deployable parameters: {missing}"
        )
    lm_head = parameters.get("lm_head.weight")
    embedding = parameters["model.embed_tokens.weight"]
    if lm_head is None or id(lm_head) != id(embedding):
        raise MTPDrafterTrainingError(
            "The assistant lm_head must remain tied to model.embed_tokens so the "
            "single deployed vocabulary matrix receives all gradients."
        )
    trainable_ids = {id(parameters[key]) for key in source_keys()}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in trainable_ids)
    return {key: parameters[key] for key in source_keys()}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise MTPDrafterTrainingError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if isinstance(row, dict):
                rows.append(row)
    if not rows:
        raise MTPDrafterTrainingError(f"Dataset split contains no JSON objects: {path}")
    return rows


def _token_list(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value] if isinstance(value, list) else []


def _longest_common_prefix(left: list[int], right: list[int]) -> int:
    length = 0
    for first, second in zip(left, right):
        if first != second:
            break
        length += 1
    return length


def _encode_row(
    row: dict[str, Any], tokenizer: Any, max_length: int, *,
    overflow_policy: str | None = None, chat_template_kwargs: dict[str, Any] | None = None,
) -> dict[str, list[int]]:
    if isinstance(row.get("input_ids"), list):
        input_ids = _token_list(row["input_ids"])
        if not isinstance(row.get("labels"), list):
            raise MTPDrafterTrainingError(
                "Pretokenized completion-only rows must provide explicit labels."
            )
        labels = _token_list(row["labels"])
    else:
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise MTPDrafterTrainingError(
                "Each drafter row needs input_ids or a non-empty messages list."
            )
        full_ids = _token_list(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                **(chat_template_kwargs or {}),
            )
        )
        last_message = messages[-1]
        if not isinstance(last_message, dict):
            raise MTPDrafterTrainingError(
                "The final chat message must be an object with role/content fields."
            )
        if last_message.get("role") != "assistant":
            raise MTPDrafterTrainingError(
                "Completion-only drafter rows must end with an assistant message."
            )
        prompt_messages = messages[:-1]
        if prompt_messages:
            prompt_ids = _token_list(
                tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    **(chat_template_kwargs or {}),
                )
            )
            prompt_length = _longest_common_prefix(prompt_ids, full_ids)
        else:
            prompt_length = 0
        input_ids = full_ids
        labels = [-100] * prompt_length + full_ids[prompt_length:]
    if len(input_ids) != len(labels) or len(input_ids) < 2:
        raise MTPDrafterTrainingError(
            "Encoded drafter row must have equal input/label lengths of at least two."
        )
    if len(input_ids) > max_length:
        if overflow_policy == "error":
            raise MTPDrafterTrainingError(
                f"MTP row {row.get('id')} has {len(input_ids)} tokens, exceeding {max_length}; no truncation permitted"
            )
        # Retain the supervised suffix and as much prompt context as fits.
        supervised = next(
            (index for index, label in enumerate(labels) if label >= 0),
            len(labels),
        )
        completion_ids = input_ids[supervised:]
        completion_labels = labels[supervised:]
        if len(completion_ids) >= max_length:
            input_ids = completion_ids[:max_length]
            labels = completion_labels[:max_length]
        else:
            prompt_budget = max_length - len(completion_ids)
            input_ids = input_ids[max(0, supervised - prompt_budget) : supervised] + completion_ids
            labels = [-100] * min(prompt_budget, supervised) + completion_labels
    if not any(label >= 0 for label in labels[1:]):
        raise MTPDrafterTrainingError(
            "Encoded row has no supervised next-token position after truncation."
        )
    return {"input_ids": input_ids, "labels": labels}


def _anchor_positions(
    labels: list[int],
    *,
    positions_per_sequence: int,
    rng: random.Random,
) -> list[int]:
    candidates = [index for index in range(len(labels) - 1) if labels[index + 1] >= 0]
    if len(candidates) <= positions_per_sequence:
        return candidates
    return sorted(rng.sample(candidates, positions_per_sequence))


def _rollout_loss(
    *,
    target_model: Any,
    assistant_model: Any,
    input_ids: list[int],
    labels: list[int],
    anchors: list[int],
    draft_steps: int,
    device: Any,
    efficient_target: bool = False,
) -> Any:
    import torch
    from torch.nn import functional

    losses = []
    target_embedding = target_model.get_input_embeddings()
    sequence = torch.tensor([input_ids], dtype=torch.long, device=device)
    full_attention_mask = torch.ones_like(sequence)
    with torch.no_grad():
        target_outputs = target_model(
            input_ids=sequence,
            attention_mask=full_attention_mask,
            output_hidden_states=True,
            return_shared_kv_states=True,
            use_cache=False,
            **({"logits_to_keep": 1} if efficient_target else {}),
        )
        hidden_states = getattr(target_outputs, "hidden_states", None)
        all_shared_kv_states = getattr(target_outputs, "shared_kv_states", None)
        if not hidden_states or not isinstance(all_shared_kv_states, Mapping):
            raise MTPDrafterTrainingError(
                "Target model did not return hidden_states and shared_kv_states; "
                "use a current Transformers Gemma4ForCausalLM checkpoint."
            )
    for anchor in anchors:
        current_length = anchor + 1
        attention_mask = full_attention_mask[:, :current_length]
        with torch.no_grad():
            recurrent_state = hidden_states[-1][:, anchor : anchor + 1, :].detach()
            try:
                shared_kv_states = {
                    key: (
                        value[0][:, :, :current_length, :].detach(),
                        value[1][:, :, :current_length, :].detach(),
                    )
                    for key, value in all_shared_kv_states.items()
                }
            except (AttributeError, IndexError, TypeError) as exc:
                raise MTPDrafterTrainingError(
                    "Target shared_kv_states did not use the expected "
                    "[batch, heads, sequence, dim] tuple contract."
                ) from exc
        current_token = sequence[:, anchor : anchor + 1]
        position_ids = torch.tensor([[anchor]], dtype=torch.long, device=device)
        for offset in range(1, draft_steps + 1):
            target_index = anchor + offset
            if target_index >= len(input_ids) or labels[target_index] < 0:
                break
            expected = torch.tensor(
                [input_ids[target_index]], dtype=torch.long, device=device
            )
            with torch.no_grad():
                token_embedding = target_embedding(current_token).detach()
            assistant_inputs = torch.cat(
                [token_embedding, recurrent_state], dim=-1
            )
            outputs = assistant_model(
                inputs_embeds=assistant_inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                shared_kv_states=shared_kv_states,
                use_cache=False,
            )
            logits = outputs.logits[:, -1, :].float()
            losses.append(functional.cross_entropy(logits, expected))
            # Teacher forcing keeps the public objective deterministic. The
            # projected assistant state remains recurrent, matching inference.
            current_token = expected[:, None]
            recurrent_state = outputs.last_hidden_state
    if not losses:
        raise MTPDrafterTrainingError("No drafter loss tokens were produced for this row.")
    return torch.stack(losses).mean()


def _checkpoint_manifest(directory: Path) -> list[dict[str, Any]]:
    files = sorted(
        item
        for pattern in ("*.safetensors", "config.json", "generation_config.json")
        for item in directory.glob(pattern)
        if item.is_file()
    )
    return [
        {
            "path": item.name,
            "size": int(item.stat().st_size),
            "sha256": _sha256(item),
        }
        for item in files
    ]


def train_mtp_drafter(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    target_model_override: str | Path | None = None,
) -> dict[str, Any]:
    """Execute the opt-in public target-conditioned drafter QAT workflow."""
    with ExitStack() as cleanup:
        return _train_mtp_drafter_impl(config, config_path=config_path,
                                       target_model_override=target_model_override, cleanup=cleanup)


def _train_mtp_drafter_impl(
    config: dict[str, Any], *, config_path: str | Path,
    target_model_override: str | Path | None, cleanup: ExitStack,
) -> dict[str, Any]:

    plan = build_drafter_training_plan(
        config,
        config_path=config_path,
        target_model_override=target_model_override,
    )
    if not plan["ready"]:
        raise MTPDrafterTrainingError(
            "Drafter training plan is not executable: "
            + json.dumps(plan["issues"], ensure_ascii=False)
        )
    try:
        import torch
        from accelerate import Accelerator
        from accelerate.utils import set_seed
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, Dataset
        from torch.utils.tensorboard import SummaryWriter
        from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
    except Exception as exc:  # pragma: no cover - dependency environment
        raise MTPDrafterTrainingError(
            "Install training/requirements-training.txt with current Transformers "
            "and Accelerate before executing drafter training."
        ) from exc

    training_cfg = _section(config, "training")
    model_cfg = _section(config, "model")
    official = _official_mobile(config)
    source_bindings: dict[str, str] = {}
    target_qat: dict[str, Any] = {}
    if official:
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise MTPDrafterTrainingError("Official MTP adaptation currently requires a single process/GPU")
        if target_model_override is not None and str(Path(target_model_override).resolve()) != str(Path(config["target"]["model_id_or_path"]).resolve()):
            raise MTPDrafterTrainingError("Official MTP target override cannot change the bound merged checkpoint")
        probe_official_mobile_mtp_environment()
        source_bindings = official_mobile_source_bindings(config)
    seed = int(training_cfg.get("seed", 42))
    output_dir = Path(plan["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise MTPDrafterTrainingError(
            f"Refusing to train into non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    mixed_precision = str(training_cfg.get("mixed_precision", "bf16")).lower()
    if mixed_precision not in {"no", "fp16", "bf16"}:
        raise MTPDrafterTrainingError(
            "training.mixed_precision must be no, fp16, or bf16."
        )
    accelerator = Accelerator(
        gradient_accumulation_steps=int(
            training_cfg.get("gradient_accumulation_steps", 4)
        ),
        mixed_precision=mixed_precision,
    )
    if official and (accelerator.num_processes != 1 or accelerator.device.type != "cuda"):
        raise MTPDrafterTrainingError("Official MTP requires one CUDA process; CPU/distributed fallback is forbidden")
    set_seed(seed, device_specific=True)
    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "no": torch.float32,
    }[mixed_precision]
    load_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
        "low_cpu_mem_usage": True,
    }
    if official:
        target, tokenizer, target_controller = _load_official_target(config, accelerator.device)
        cleanup.callback(target_controller.restore)
        target_qat = target_controller.summary()
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            plan["target"]["tokenizer_id_or_path"],
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
            use_fast=True,
        )
        target = AutoModelForCausalLM.from_pretrained(
            plan["target"]["model_id_or_path"], **load_kwargs
        )
    assistant_load_kwargs = {**load_kwargs, "torch_dtype": torch.float32, "local_files_only": True} if official else load_kwargs
    assistant = AutoModelForCausalLM.from_pretrained(
        plan["assistant"]["base_model_id_or_path"], **assistant_load_kwargs
    )
    if not official and "Gemma4" not in target.__class__.__name__:
        raise MTPDrafterTrainingError(
            "Target must load as a Gemma 4 causal LM with shared-KV outputs; "
            f"observed {target.__class__.__name__}."
        )
    if "Gemma4Assistant" not in assistant.__class__.__name__:
        raise MTPDrafterTrainingError(
            "Assistant must load as Gemma4AssistantForCausalLM; observed "
            f"{assistant.__class__.__name__}."
        )
    if int(getattr(assistant.config.get_text_config(), "num_hidden_layers", 0)) != 4:
        raise MTPDrafterTrainingError("The official E2B assistant must have four layers.")
    target.eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    target.to(accelerator.device)
    trainable = _deployment_trainable_parameters(assistant)
    if official and any(parameter.dtype != torch.float32 for parameter in trainable.values()):
        raise MTPDrafterTrainingError("Official MTP requires FP32 trainable assistant master weights")
    qat_controller = DrafterQATController(config)
    cleanup.callback(qat_controller.restore)
    qat_controller.prepare(assistant)
    qat_summary = qat_controller.summary()

    max_length = int(training_cfg.get("max_seq_length", 2048))
    train_rows = _read_jsonl(Path(plan["dataset"]["train"]))
    validation_rows = (
        _read_jsonl(Path(plan["dataset"]["validation"]))
        if plan["dataset"]["validation"]
        else []
    )

    class EncodedDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]):
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            return _encode_row(self.rows[index], tokenizer, max_length,
                               **({"overflow_policy": "error", "chat_template_kwargs": training_cfg.get("chat_template_kwargs", {})}
                                  if official else {}))

    def collate(rows: list[dict[str, list[int]]]) -> dict[str, list[int]]:
        if len(rows) != 1:
            raise MTPDrafterTrainingError(
                "The target-conditioned drafter currently requires batch_size=1."
            )
        return rows[0]

    batch_size = int(training_cfg.get("per_device_train_batch_size", 1))
    if batch_size != 1:
        raise MTPDrafterTrainingError(
            "Set per_device_train_batch_size=1; positions are sampled inside each sequence."
        )
    train_loader = DataLoader(
        EncodedDataset(train_rows),
        batch_size=1,
        shuffle=True,
        collate_fn=collate,
    )
    validation_loader = (
        DataLoader(
            EncodedDataset(validation_rows),
            batch_size=1,
            shuffle=False,
            collate_fn=collate,
        )
        if validation_rows
        else None
    )
    optimizer = AdamW(
        list({id(value): value for value in trainable.values()}.values()),
        lr=float(training_cfg.get("learning_rate", 1e-5)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
    )
    epochs = int(training_cfg.get("epochs", 1))
    max_steps = int(training_cfg.get("max_steps", 0) or 0)
    total_updates = max(
        1,
        math.ceil(len(train_loader) * epochs / accelerator.gradient_accumulation_steps),
    )
    if max_steps > 0:
        total_updates = min(total_updates, max_steps)
    scheduler = get_scheduler(
        str(training_cfg.get("lr_scheduler_type", "cosine")),
        optimizer=optimizer,
        num_warmup_steps=int(total_updates * float(training_cfg.get("warmup_ratio", 0.03))),
        num_training_steps=total_updates,
    )
    if validation_loader is None:
        if official:
            raise MTPDrafterTrainingError("Official MTP requires nonempty held-out validation data")
        assistant, optimizer, train_loader, scheduler = accelerator.prepare(
            assistant, optimizer, train_loader, scheduler
        )
    else:
        assistant, optimizer, train_loader, validation_loader, scheduler = accelerator.prepare(
            assistant, optimizer, train_loader, validation_loader, scheduler
        )
    positions_per_sequence = int(training_cfg.get("positions_per_sequence", 4))
    draft_steps = int(training_cfg.get("draft_steps", 4))
    rng = random.Random(seed + accelerator.process_index)
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    global_step = 0
    best_loss = float("inf")
    history: list[dict[str, Any]] = []
    tensorboard_subdir = str(training_cfg.get("tensorboard_subdir") or "mtp_training")
    tensorboard_log_dir = (
        resolve_tensorboard_root(training_cfg.get("tensorboard_root") or "tensorboard")
        / str(plan["run_id"])
        / tensorboard_subdir
    )
    writer = (
        SummaryWriter(log_dir=str(tensorboard_log_dir))
        if accelerator.is_main_process
        else None
    )
    logging_steps = max(1, int(training_cfg.get("logging_steps", 20)))

    def evaluate() -> float:
        if validation_loader is None:
            return float("nan")
        assistant.eval()
        losses = []
        evaluation_rng = random.Random(seed)
        max_rows = int(training_cfg.get("max_eval_rows", 64))
        with torch.no_grad():
            for row_index, row in enumerate(validation_loader):
                if max_rows > 0 and row_index >= max_rows:
                    break
                anchors = _anchor_positions(
                    row["labels"],
                    positions_per_sequence=positions_per_sequence,
                    rng=evaluation_rng,
                )
                with accelerator.autocast() if official else nullcontext():
                    loss = _rollout_loss(
                        target_model=target,
                        assistant_model=assistant,
                        input_ids=row["input_ids"],
                        labels=row["labels"],
                        anchors=anchors,
                        draft_steps=draft_steps,
                        device=accelerator.device,
                        **({"efficient_target": True} if official else {}),
                    )
                if official and not bool(torch.isfinite(loss).item()):
                    raise MTPDrafterTrainingError("MTP validation loss is nonfinite")
                losses.append(accelerator.gather_for_metrics(loss.detach()).mean())
        assistant.train()
        return (
            float(torch.stack(losses).mean().item()) if losses else float("nan")
        )

    def save_checkpoint(path: Path) -> None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(assistant)
            unwrapped.save_pretrained(
                path,
                safe_serialization=True,
                save_function=accelerator.save,
            )
            tokenizer.save_pretrained(path)
        accelerator.wait_for_everyone()

    assistant.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        for epoch in range(epochs):
            epoch_losses: list[float] = []
            for row in train_loader:
                anchors = _anchor_positions(
                    row["labels"],
                    positions_per_sequence=positions_per_sequence,
                    rng=rng,
                )
                with accelerator.accumulate(assistant):
                    with accelerator.autocast() if official else nullcontext():
                        loss = _rollout_loss(
                            target_model=target,
                            assistant_model=assistant,
                            input_ids=row["input_ids"],
                            labels=row["labels"],
                            anchors=anchors,
                            draft_steps=draft_steps,
                            device=accelerator.device,
                            **({"efficient_target": True} if official else {}),
                        )
                    if official and not bool(torch.isfinite(loss).item()):
                        raise MTPDrafterTrainingError("MTP training loss is nonfinite")
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        if official:
                            _check_finite_gradients(assistant)
                        gradient_norm = accelerator.clip_grad_norm_(assistant.parameters(), max_grad_norm)
                        if official and not bool(torch.isfinite(gradient_norm).item()):
                            raise MTPDrafterTrainingError("MTP gradient norm is nonfinite")
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                epoch_losses.append(float(loss.detach().item()))
                if accelerator.sync_gradients:
                    global_step += 1
                    if writer is not None and global_step % logging_steps == 0:
                        writer.add_scalar(
                            "mtp_training/train_loss_step",
                            float(loss.detach().item()),
                            global_step,
                        )
                        writer.add_scalar(
                            "mtp_training/learning_rate",
                            float(scheduler.get_last_lr()[0]),
                            global_step,
                        )
                if max_steps > 0 and global_step >= max_steps:
                    break
            train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            validation_loss = evaluate()
            if official and not math.isfinite(validation_loss):
                raise MTPDrafterTrainingError("Official MTP selection requires finite held-out validation loss")
            selection_loss = validation_loss if math.isfinite(validation_loss) else train_loss
            history.append(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss
                    if math.isfinite(validation_loss)
                    else None,
                    "selection_loss": selection_loss,
                }
            )
            if writer is not None:
                writer.add_scalar(
                    "mtp_training/train_loss_epoch", train_loss, global_step
                )
                writer.add_scalar(
                    "mtp_training/selection_loss", selection_loss, global_step
                )
                if math.isfinite(validation_loss):
                    writer.add_scalar(
                        "mtp_training/validation_loss",
                        validation_loss,
                        global_step,
                    )
                writer.flush()
            if selection_loss < best_loss:
                best_loss = selection_loss
                save_checkpoint(Path(plan["assistant"]["best_checkpoint"]))
            if max_steps > 0 and global_step >= max_steps:
                break
        save_checkpoint(Path(plan["assistant"]["final_checkpoint"]))
    finally:
        qat_controller.restore()
        if writer is not None:
            writer.flush()
            writer.close()

    accelerator.wait_for_everyone()
    if official and official_mobile_source_bindings(config) != source_bindings:
        raise MTPDrafterTrainingError("Official MTP source bindings changed during training")
    result = {
        **plan,
        "training_executed": True,
        "global_steps": global_step,
        "best_selection_loss": best_loss,
        "history": history,
        "qat_runtime": qat_summary,
        "tensorboard_log_dir": str(tensorboard_log_dir),
        **({"source_bindings": source_bindings, "target_qat": target_qat,
            "purpose": OFFICIAL_MOBILE_MTP_PURPOSE, "assistant_master_dtype": "float32",
            "selection_metric": "held_out_completion_cross_entropy"} if official else {}),
    }
    if accelerator.is_main_process:
        config_hash = _sha256(Path(config_path).expanduser().resolve())
        for role, checkpoint in (
            ("best", Path(plan["assistant"]["best_checkpoint"])),
            ("final", Path(plan["assistant"]["final_checkpoint"])),
        ):
            metadata = {
                "manifest_version": 1,
                "role": role,
                "training_method": "target_conditioned_autoregressive_qat",
                "loss": "teacher_forced_completion_cross_entropy",
                "private_google_training_recipe_recovered": False,
                "target_model_id_or_path": plan["target"]["model_id_or_path"],
                "target_frozen": True,
                "assistant_base_model_id_or_path": plan["assistant"][
                    "base_model_id_or_path"
                ],
                "training_config": str(Path(config_path).expanduser().resolve()),
                "training_config_sha256": config_hash,
                "draft_steps": draft_steps,
                "teacher_forcing": True,
                "constant_position_ids": True,
                "fixed_target_kv_during_each_draft_rollout": True,
                "unmapped_parameters_frozen": True,
                "training_unmapped_constants_source": "public_assistant_checkpoint",
                "deployment_unmapped_constants_source": "official_litertlm_artifact",
                "trainable_source_keys": source_keys(),
                "mobile_contract": contract_summary(),
                "qat": qat_summary,
                "tensorboard_log_dir": str(tensorboard_log_dir),
                "checkpoint_files": _checkpoint_manifest(checkpoint),
                "history": history,
                **({"source_bindings": source_bindings, "target_qat": target_qat,
                    "purpose": OFFICIAL_MOBILE_MTP_PURPOSE, "assistant_master_dtype": "float32"} if official else {}),
            }
            (checkpoint / "mtp_drafter_training_metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        (output_dir / "mtp_drafter_training_report.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()
    return result
