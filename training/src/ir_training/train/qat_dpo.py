"""Opt-in offline DPO on the *original SFT + QAT* LoRA policy.

The shared QAT loader, numeric gates, Golden selection and adapter export are
deliberately reused. Reference likelihoods are computed before any update and
persisted as an immutable, identity-bound cache; disabling LoRA is never a
reference policy. Imports stay CPU/tooling safe.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DPO_HEALTH_WINDOW = 20
DPO_REFERENCE_SCHEMA = "sft_qat_completion_logps_v1"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def dpo_loss(chosen: Any, rejected: Any, reference_chosen: Any,
             reference_rejected: Any, beta: float) -> Any:
    """Unnormalised completion-sequence, untempered sigmoid DPO objective."""
    from torch.nn import functional
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(beta) or beta <= 0:
        raise ValueError("DPO beta must be finite and positive")
    return -functional.logsigmoid(beta * ((chosen - rejected) - (reference_chosen - reference_rejected)))


def completion_log_probability(model: Any, row: dict, side: str, *, device: Any = None) -> Any:
    """Sum causal log probabilities only over the validated completion + EOS.

    Candidates are not concatenated into a doubled batch. FP32 cross-entropy is
    checkpointed in bounded token blocks so a vocabulary-sized FP32 temporary is
    not retained for every token of both candidates.
    """
    import torch
    from torch.nn import functional
    from torch.utils.checkpoint import checkpoint
    if side not in {"chosen", "rejected"}:
        raise ValueError("Unknown DPO candidate side")
    ids = row[f"{side}_input_ids"]
    labels = row[f"{side}_labels"]
    mask = row[f"{side}_completion_mask"]
    if (len(ids) < 2 or len(labels) != len(ids) or len(mask) != len(ids)
            or mask[0] or not any(mask) or any(bool(m) != (label != -100) for m, label in zip(mask, labels))
            or any(label != token for token, label, m in zip(ids, labels, mask) if m)):
        raise ValueError("Malformed DPO completion-only causal labels")
    if device is None:
        device = next(model.parameters()).device
    inputs = torch.tensor([ids], dtype=torch.long, device=device)
    attention = torch.tensor([row[f"{side}_attention_mask"]], dtype=torch.long, device=device)
    if attention.shape != inputs.shape or not bool(attention.all()):
        raise ValueError("DPO candidates must be unpadded, complete token sequences")
    outputs = model(input_ids=inputs, attention_mask=attention, use_cache=False)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
    if logits.shape[:2] != inputs.shape:
        raise ValueError("DPO model did not return every causal token logit")
    targets = torch.tensor(labels[1:], dtype=torch.long, device=device)
    indices = targets.ne(-100).nonzero(as_tuple=True)[0]

    def block_loss(block, target):
        return functional.cross_entropy(block.float(), target, reduction="sum")

    total = logits.new_zeros((), dtype=torch.float32)
    for start in range(0, len(indices), 128):
        selected = indices[start:start + 128]
        block, target = logits[0, selected, :], targets[selected]
        loss = (checkpoint(block_loss, block, target, use_reentrant=False)
                if torch.is_grad_enabled() and block.requires_grad else block_loss(block, target))
        total = total - loss
    if not bool(torch.isfinite(total).detach()):
        raise ValueError("Nonfinite DPO completion log probability")
    return total


@dataclass(frozen=True)
class ReferenceLogps:
    pair_id: str
    chosen: float
    rejected: float


def precompute_reference_cache(model: Any, pairs: list[dict], path: Path, binding: dict,
                               *, autocast_factory=nullcontext, policy_assert=None) -> tuple[ReferenceLogps, ...]:
    """Score the initial loaded SFT policy once; never mutate/rewrite a cache."""
    import torch
    tokens = [{key: row[key] for key in ("pair_id", "prompt_input_ids", "chosen_input_ids", "chosen_labels",
                                       "rejected_input_ids", "rejected_labels")} for row in pairs]
    expected = {"schema": DPO_REFERENCE_SCHEMA, "binding": binding, "tokenized_pairs_sha256": _digest(tokens),
                "pairs": len(pairs), "temperature": 1.0, "completion_reduction": "sum_with_eos"}
    if path.exists():
        raise ValueError(f"DPO reference cache already exists; use a fresh run directory: {path}")
    if not pairs or len({row["pair_id"] for row in pairs}) != len(pairs):
        raise ValueError("DPO reference requires nonempty unique pair IDs")
    modules = [(module, module.training) for module in model.modules()]
    versions = [(parameter, parameter._version) for parameter in model.parameters()]
    if any(parameter.grad is not None for parameter, _ in versions):
        raise ValueError("DPO reference must be computed before optimizer/training gradients exist")
    rows = []
    try:
        model.eval()
        with torch.no_grad(), autocast_factory():
            if policy_assert is not None:
                policy_assert()
            for pair in pairs:
                rows.append(ReferenceLogps(pair["pair_id"],
                    float(completion_log_probability(model, pair, "chosen")),
                    float(completion_log_probability(model, pair, "rejected"))))
        if any(parameter._version != version or parameter.grad is not None for parameter, version in versions):
            raise ValueError("DPO reference computation changed policy parameters or gradients")
    finally:
        for module, training in modules:
            module.training = training
    payload = {**expected, "rows": [vars(row) for row in rows]}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    return tuple(rows)


def serial_dpo_backward(model: Any, pairs: list[dict], references: dict[str, ReferenceLogps], *,
                        beta: float, backward: Any, scale: float = 1.0,
                        autocast_factory=nullcontext) -> dict[str, float]:
    """Exact sigmoid-DPO gradient with one candidate graph resident at a time.

    With dropout disabled, a no-grad coefficient pass followed by two serial
    forward/backward passes has exactly the same derivative as the paired loss.
    Accelerate owns scaling/DDP, and the caller owns batch/accumulation scaling.
    """
    import torch
    if not pairs or not math.isfinite(scale) or scale <= 0:
        raise ValueError("DPO serial update needs nonempty pairs and positive scaling")
    sums = {"loss": 0.0, "margin": 0.0, "preference_accuracy": 0.0}
    for pair in pairs:
        reference = references[pair["pair_id"]]
        with torch.no_grad(), autocast_factory():
            chosen = completion_log_probability(model, pair, "chosen")
            rejected = completion_log_probability(model, pair, "rejected")
            margin = (chosen - rejected) - (reference.chosen - reference.rejected)
            loss = dpo_loss(chosen, rejected, reference.chosen, reference.rejected, beta)
            coefficient = -beta * torch.sigmoid(-beta * margin)
            sums["loss"] += float(loss)
            sums["margin"] += float(margin)
            sums["preference_accuracy"] += float(margin > 0)
        for side, sign in (("chosen", 1), ("rejected", -1)):
            with autocast_factory():
                logp = completion_log_probability(model, pair, side)
                surrogate = sign * coefficient * logp * (scale / len(pairs))
            backward(surrogate)
            del logp, surrogate
    return {key: value / len(pairs) for key, value in sums.items()}


def health_snapshot(rows: list[dict], step: int) -> dict:
    required = ("loss", "margin", "preference_accuracy", "gradient_l2_norm", "finite_parameters", "parameter_update_max_abs")
    valid = (isinstance(rows, list) and type(step) is int and len(rows) == DPO_HEALTH_WINDOW and step >= DPO_HEALTH_WINDOW
             and all(isinstance(row, dict) and set(row) == set(required) and all(type(row[k]) in (int, float) and math.isfinite(row[k]) for k in required)
                     and row["loss"] >= 0 and 0 <= row["preference_accuracy"] <= 1
                     and row["gradient_l2_norm"] > 0 and row["finite_parameters"] == 1
                     and row["parameter_update_max_abs"] >= 0 for row in rows)
             and any(row["parameter_update_max_abs"] > 0 for row in rows))
    aggregatable = (isinstance(rows, list) and bool(rows) and all(isinstance(row, dict)
        and all(type(row.get(key)) in (int, float) and math.isfinite(row[key]) for key in required) for row in rows))
    aggregates = {key: sum(row[key] for row in rows) / len(rows) for key in required} if aggregatable else {}
    return {"schema": "dpo_health_v1", "passed_window": bool(valid), "optimizer_steps": step,
            "last_step": step, "window_steps": DPO_HEALTH_WINDOW, "window_updates": len(rows),
            "window_metrics": rows, "aggregates": aggregates,
            "scope": "finite optimizer health and actual LoRA update; Golden fidelity/nonregression gates remain mandatory"}


def snapshot_trainable_parameters(model: Any) -> dict[str, Any]:
    """CPU copies of adapters only; never duplicate the frozen multi-billion base."""
    result = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if "lora_" not in name:
                raise ValueError(f"DPO may update only LoRA parameters: {name}")
            result[name] = parameter.detach().to(device="cpu", copy=True)
    if not result:
        raise ValueError("DPO has no trainable adapter parameters")
    return result


def parameter_update_max_abs(model: Any, snapshot: dict[str, Any]) -> float:
    import torch
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if names != set(snapshot):
        raise ValueError("DPO trainable scope changed during an optimizer update")
    maximum = 0.0
    for name, parameter in model.named_parameters():
        if name in snapshot:
            value = float((parameter.detach().cpu().float() - snapshot[name].float()).abs().max())
            if not torch.isfinite(torch.tensor(value)):
                raise ValueError(f"Nonfinite DPO parameter update: {name}")
            maximum = max(maximum, value)
    return maximum


def verify_dpo_evidence(config: dict, metadata: dict) -> dict[str, bool]:
    """Recompute DPO-specific export evidence without importing GPU packages."""
    checks = {"dpo_objective_identified": False, "dpo_reference_bound": False,
              "dpo_preference_manifest_bound": False, "selected_step_dpo_health_window_passed": False}
    try:
        grpo, configured = metadata["grpo"], config["grpo"]
        dpo = grpo["dpo"]
        checks["dpo_objective_identified"] = (configured.get("objective") == "dpo" and grpo.get("algorithm") == "dpo"
            and dpo["beta"] == configured["dpo"]["beta"] and dpo["beta"] > 0)
        identity = dpo["reference_cache"]
        path = Path(identity["path"])
        cache = json.loads(path.read_text(encoding="utf-8"))
        manifest_identity = dpo["preference_manifest"]
        manifest_path = Path(manifest_identity["path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        from ir_training.common.config import resolve_path, training_root
        from ir_training.train.qat_posttrain_data import verify_admission_manifest
        verify_admission_manifest(manifest, kind="qat_dpo",
            reward_config=resolve_path(configured.get("reward_config", "../dataset/configs/genui_metric_v5_4.yaml"), training_root()),
            expected_limits={"max_prompt_length": configured["max_prompt_length"],
                             "max_completion_length": configured["max_completion_length"],
                             "max_sequence_length": config["training"]["max_seq_length"]})
        checks["dpo_preference_manifest_bound"] = (_identity(manifest_path) == manifest_identity
            and manifest["raw_pairs"] == _identity(resolve_path(configured["dpo"]["preferences"], training_root()))
            and manifest["raw_pairs"]["sha256"] == manifest["pairs_sha256"] == configured["dpo"]["preferences_sha256"])
        binding = cache["binding"]
        cache_rows = cache["rows"]
        checks["dpo_reference_bound"] = (identity == _identity(path) and cache["schema"] == DPO_REFERENCE_SCHEMA
            and cache["temperature"] == 1.0 and cache["completion_reduction"] == "sum_with_eos"
            and binding["source_lineage"] == grpo["source_lineage"]
            and binding["training_config_sha256"] == metadata["training_config_sha256"]
            and binding["preference_manifest_sha256"] == manifest_identity["sha256"]
            and binding["tokenizer_binding"] == metadata["prepared_dataset_binding"]
            and cache["tokenized_pairs_sha256"] == dpo["tokenized_pairs_sha256"]
            and cache["pairs"] == manifest["rows"] == len(cache_rows) > 0
            and len({row["pair_id"] for row in cache_rows}) == len(cache_rows)
            and all(type(row[k]) in (float, int) and math.isfinite(row[k]) and row[k] <= 0
                    for row in cache_rows for k in ("chosen", "rejected")))
        health = grpo["health"]
        checks["selected_step_dpo_health_window_passed"] = (metadata["checkpoint_step"] == health["last_step"]
            and health_snapshot(health["window_metrics"], health["last_step"]) == health
            and health["passed_window"] is True)
    except (KeyError, OSError, TypeError, ValueError, OverflowError):
        pass
    return checks


def build_dpo_trainer(base_trainer: Any, bundle: Any, references: tuple[ReferenceLogps, ...], beta: float) -> Any:
    from ir_training.train.qat_grpo import assert_qat_policy_active
    reference_map = {row.pair_id: row for row in references}

    class QATDPOTrainer(base_trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = False
            self.dpo_update_metrics = []

        def training_step(self, model, inputs, num_items_in_batch=None):
            import torch
            if self.is_deepspeed_enabled or self.is_fsdp_enabled:
                raise ValueError("QAT DPO currently supports single-device/DDP only")
            model.train()
            if callable(getattr(self.optimizer, "train", None)):
                self.optimizer.train()
            assert_qat_policy_active(model, bundle.controller)
            pairs = inputs["pairs"]
            accumulation = getattr(self, "current_gradient_accumulation_steps", self.args.gradient_accumulation_steps)
            # Accelerator.backward divides by its own GAS. Compensate explicitly
            # so there is exactly one division by Trainer's actual accumulation.
            accelerator_gas = self.accelerator.gradient_accumulation_steps
            metrics = serial_dpo_backward(model, pairs, reference_map, beta=beta,
                backward=self.accelerator.backward, scale=accelerator_gas / accumulation,
                autocast_factory=self.accelerator.autocast)
            self.dpo_update_metrics.append(metrics)
            return torch.tensor(metrics["loss"] / accumulation, device=self.accelerator.device)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            import torch
            assert_qat_policy_active(model, bundle.controller)
            losses = []
            for pair in inputs["pairs"]:
                reference = reference_map[pair["pair_id"]]
                with self.accelerator.autocast():
                    losses.append(dpo_loss(completion_log_probability(model, pair, "chosen"),
                        completion_log_probability(model, pair, "rejected"), reference.chosen, reference.rejected, beta))
            loss = torch.stack(losses).mean()
            return (loss, {"loss": loss}) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            import torch
            # The collator intentionally has no top-level labels: the two
            # candidates have independent completion masks and token lengths.
            with torch.no_grad(), self.accelerator.autocast():
                loss = self.compute_loss(model, inputs)
            return loss.detach(), None, None

        def save_model(self, output_dir=None, _internal_call=False):
            generation = getattr(self.model, "generation_config", None)
            sentinel = object()
            previous = getattr(generation, "cache_implementation", sentinel)
            if previous is sentinel:
                return super().save_model(output_dir, _internal_call=_internal_call)
            generation.cache_implementation = None
            try:
                return super().save_model(output_dir, _internal_call=_internal_call)
            finally:
                generation.cache_implementation = previous

    return QATDPOTrainer


def collate_dpo_pairs(rows: list[dict]) -> dict:
    """Top-level/picklable collator keeps candidates separate and unpadded."""
    return {"pairs": rows}


def train_qat_dpo(config: dict, config_path: Path, *, dependency_preflight_only=False,
                  runtime_preflight_only=False) -> dict:
    """Train from the same verified SFT adapter and publish shared export evidence."""
    from ir_training.common.config import resolve_path, training_root
    from ir_training.train import sft
    from ir_training.train.grpo_runtime import dependency_report
    from ir_training.train.qat_grpo import (
        _bind_selected_health,
        _script_module,
        assert_qat_policy_active,
        load_qat_grpo_model,
        run_qat_grpo_preflight,
        validate_qat_grpo_config,
        verify_prepared_training_contract,
    )
    validate_qat_grpo_config(config)
    training, grpo, run = config["training"], config["grpo"], config["run"]
    if grpo.get("objective") != "dpo":
        raise ValueError("DPO requires explicit grpo.objective=dpo")
    pair_path = resolve_path(grpo["dpo"]["preferences"], training_root())
    if _identity(pair_path)["sha256"] != grpo["dpo"].get("preferences_sha256"):
        raise ValueError("DPO preference file changed after config preparation")
    output = resolve_path(run["output_dir"], training_root())
    if runtime_preflight_only:
        if dependency_preflight_only:
            raise ValueError("Choose dependency or runtime preflight, not both")
        output = output.parent / "runtime_preflight"
    output.mkdir(parents=True, exist_ok=True)
    rank, world = max(0, int(os.environ.get("RANK", "0"))), max(1, int(os.environ.get("WORLD_SIZE", "1")))
    dependencies = dependency_report()
    dependency_path = output / f"grpo_dependencies.rank{rank}.json"
    dependency_path.write_text(json.dumps(dependencies, indent=2), encoding="utf-8")
    errors = [name for name, info in dependencies["packages"].items() if info.get("import_error")]
    if errors or dependencies["packages"]["trl"].get("outside_distribution_root"):
        raise RuntimeError(f"DPO dependency/source preflight failed: {errors}; see {dependency_path}")
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments
    if not hasattr(TrainerCallback, "on_pre_optimizer_step"):
        raise RuntimeError("DPO health requires TrainerCallback.on_pre_optimizer_step")
    if dependency_preflight_only:
        return {"dependencies": dependencies, "training_executed": False}
    if not torch.cuda.is_available():
        raise RuntimeError("QAT DPO training requires CUDA")
    limit = sft._training_limit_config(training)
    if limit["max_optimizer_steps"] is not None and limit["max_optimizer_steps"] < DPO_HEALTH_WINDOW:
        raise ValueError("DPO max_steps must cover the complete 20-update safety window")
    from ir_training.train.grpo_probe import (
        runtime_arguments,
        verify_local_probe_device,
        verify_probe_reports,
    )
    if grpo.get("runtime_preflight_required") and not runtime_preflight_only:
        verify_probe_reports(output.parent / "runtime_preflight", config_path, world)
        verify_local_probe_device(output.parent / "runtime_preflight" / f"grpo_runtime_probe.rank{rank}.json",
                                  torch, int(os.environ.get("LOCAL_RANK", "0")))
    prepared_contract = verify_prepared_training_contract(config)
    bundle = load_qat_grpo_model(config, config_path=config_path)
    try:
        from ir_training.train.prepared_binding import verify_tokenizer_binding
        from ir_training.train.qat_posttrain_data import prepare_dpo_pairs
        dataset_dir = resolve_path(run["dataset_dir"], training_root())
        binding = verify_tokenizer_binding(dataset_dir, bundle.tokenizer, config["model"], required=True)
        shared_data = _script_module("train_grpo")
        source_rows = shared_data.load_training_dataset(str(dataset_dir / "train.jsonl"), None,
            tokenizer=bundle.tokenizer, prompt_source="prepared",
            chat_template_kwargs=config["model"].get("chat_template_kwargs") or {})
        validation_rows = shared_data.load_training_dataset(str(dataset_dir / "val.jsonl"), None,
            tokenizer=bundle.tokenizer, prompt_source="prepared",
            chat_template_kwargs=config["model"].get("chat_template_kwargs") or {})
        context = sft._model_position_limit(bundle.model) or int(config["model"]["max_context_tokens"])
        if int(training["max_seq_length"]) > context:
            raise ValueError("DPO supervised sequence budget exceeds the actual model context")
        from pipeline.genui_quality import (
            ensure_v5_4_validation_ready,
            load_reward_config_v5_4,
        )
        ensure_v5_4_validation_ready()
        reward_path = resolve_path(grpo["reward_config"], training_root())
        pairs, manifest = prepare_dpo_pairs(pair_path, source_rows, bundle.tokenizer,
            max_prompt_length=grpo["max_prompt_length"], max_completion_length=grpo["max_completion_length"],
            max_sequence_length=int(training["max_seq_length"]),
            reward_config=load_reward_config_v5_4(reward_path), forbidden_rows=validation_rows)
        manifest["raw_pairs"] = _identity(pair_path)
        if not (manifest.get("pairs_sha256") == manifest["raw_pairs"]["sha256"]
                == grpo["dpo"]["preferences_sha256"]):
            raise ValueError("DPO preference file changed during pair admission")
        manifest_path = output / f"dpo_preferences.rank{rank}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        micro, accumulation = training["per_device_train_batch_size"], training["gradient_accumulation_steps"]
        effective_batch = micro * accumulation * world
        if len(pairs) < effective_batch:
            raise ValueError("DPO requires at least one complete effective batch of preference pairs")
        if runtime_preflight_only:
            pairs = sorted(pairs, key=lambda row: (-max(len(row["chosen_input_ids"]),
                len(row["rejected_input_ids"])), row["pair_id"]))[:effective_batch]
        initial_lineage = bundle.metadata["grpo"]["source_lineage"]
        numeric = run_qat_grpo_preflight(bundle, source_rows,
            max_seq_length=min(context, int(training["max_seq_length"])))
        preflight_path = output / f"grpo_preflight.rank{rank}.json"
        preflight_path.write_text(json.dumps({"schema_version": 1,
            "training_config_sha256": bundle.metadata["training_config_sha256"], "numeric_preflight": numeric,
            "source_lineage": initial_lineage}, indent=2), encoding="utf-8")
        metadata = bundle.metadata
        metadata["grpo"].update(algorithm="dpo", objective="dpo", preflight=_identity(preflight_path))
        reference_path = output / f"dpo_reference.rank{rank}.json"
        precision = sft._training_precision_flags(metadata["model"]["dtype"], training)
        references = precompute_reference_cache(bundle.model, pairs, reference_path,
            {"source_lineage": metadata["grpo"]["source_lineage"],
             "training_config_sha256": metadata["training_config_sha256"],
             "preference_manifest_sha256": _identity(manifest_path)["sha256"], "tokenizer_binding": binding},
            autocast_factory=lambda: torch.autocast("cuda",
                dtype=torch.bfloat16 if precision.get("bf16") else torch.float16,
                enabled=bool(precision.get("bf16") or precision.get("fp16"))),
            policy_assert=lambda: assert_qat_policy_active(bundle.model, bundle.controller))
        cache = json.loads(reference_path.read_text(encoding="utf-8"))
        metadata["grpo"]["dpo"] = {"beta": grpo["dpo"]["beta"], "reference_cache": _identity(reference_path),
            "preference_manifest": _identity(manifest_path), "tokenized_pairs_sha256": cache["tokenized_pairs_sha256"]}
        from ir_training.train.qat_grpo_reward import (
            QAT_GRPO_REWARD_VERSION,
            reward_implementation_identity,
        )
        metadata["grpo"]["reward"] = {"version": QAT_GRPO_REWARD_VERSION, "config": _identity(reward_path),
                                     "implementation": reward_implementation_identity(reward_path)}
        parameters = inspect.signature(TrainingArguments.__init__).parameters
        kwargs = dict(output_dir=str(output), learning_rate=float(training.get("learning_rate", 5e-6)),
            num_train_epochs=limit["num_train_epochs"], max_steps=limit["max_optimizer_steps"] or -1,
            per_device_train_batch_size=micro, per_device_eval_batch_size=1,
            gradient_accumulation_steps=accumulation, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False}, ddp_broadcast_buffers=False,
            ddp_find_unused_parameters=False, dataloader_drop_last=True, logging_steps=1,
            logging_nan_inf_filter=False, save_strategy="steps", save_steps=int(training.get("save_steps", 100)),
            eval_strategy="steps", eval_steps=int(training.get("eval_steps", 100)),
            save_total_limit=int(training.get("save_total_limit", 2)), report_to=training.get("report_to", "none"),
            remove_unused_columns=False, seed=metadata["initialization_seed"], prediction_loss_only=True,
            max_grad_norm=float(training.get("max_grad_norm", 1.0)), weight_decay=float(training.get("weight_decay", 0)),
            **sft._training_precision_flags(metadata["model"]["dtype"], training))
        kwargs.update(runtime_arguments(training, parameters, limit["max_optimizer_steps"]))
        sft._apply_training_data_seed(training, kwargs, parameters)
        if "warmup_steps" in training:
            kwargs["warmup_steps"] = training["warmup_steps"]
        elif "warmup_ratio" in parameters:
            kwargs["warmup_ratio"] = training.get("warmup_ratio", .03)
        logging_dir = sft._resolve_training_tensorboard_dir(training, run_id=str(run.get("id") or output.name))
        if logging_dir is not None and "logging_dir" in parameters:
            kwargs["logging_dir"] = str(logging_dir)
        if runtime_preflight_only:
            kwargs.update(max_steps=2, save_strategy="no", eval_strategy="no", report_to="none")
        trainer_class = build_dpo_trainer(Trainer, bundle, references, float(grpo["dpo"]["beta"]))
        with sft._training_tensorboard_environment(logging_dir):
            trainer = trainer_class(model=bundle.model, args=TrainingArguments(**kwargs), train_dataset=pairs,
                # DPO's periodic loss is explicitly a training-pair diagnostic;
                # only the separate reserved Golden callback selects checkpoints.
                eval_dataset=pairs[:min(32, len(pairs))], processing_class=bundle.tokenizer,
                data_collator=collate_dpo_pairs)
        (output / f"grpo_runtime_config.rank{rank}.json").write_text(json.dumps({"objective": "dpo",
            "actual_trainer_arguments": trainer.args.to_dict(), "effective_pair_batch": effective_batch,
            "periodic_loss_scope": "training_pair_diagnostic_not_validation",
            "runtime_preflight_only": runtime_preflight_only}, indent=2, default=str), encoding="utf-8")
        if runtime_preflight_only:
            from ir_training.train.grpo_probe import run_disposable_probe
            return run_disposable_probe(trainer, output=output, config_path=config_path,
                objective="dpo",
                selected_rows=[{"pair_id": row["pair_id"], "source_id": row["source_id"],
                                "sequence_tokens": max(len(row["chosen_input_ids"]), len(row["rejected_input_ids"]))}
                               for row in pairs])
        metadata.update(run_id=run.get("id", output.name), dataset_dir=str(dataset_dir),
            prepared_training_contract=prepared_contract, prepared_dataset_binding=binding,
            golden_eval=config.get("golden_eval") or {}, effective_batch_size=effective_batch,
            trainable_parameter_names=[n for n, p in trainer.model.named_parameters() if p.requires_grad],
            trainable_parameter_counts=dict(zip(("trainable", "total"), sft._trainable_parameter_count(trainer.model))))
        snapshots, window = {}, deque(maxlen=DPO_HEALTH_WINDOW)

        class DPOHealth(TrainerCallback):
            gradient_norm = None
            before_update = None

            def on_pre_optimizer_step(self, args, state, control, model=None, **unused):
                from ir_training.train.tensor_checks import tensor_finite_and_nonzero
                square = 0.0
                for name, parameter in model.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    if parameter.grad is None or not tensor_finite_and_nonzero(parameter.grad)[0]:
                        raise ValueError(f"Missing/nonfinite DPO gradient: {name}")
                    square += float(parameter.grad.detach().float().square().sum())
                self.gradient_norm = math.sqrt(square)
                if not math.isfinite(self.gradient_norm) or self.gradient_norm <= 0:
                    raise ValueError("DPO optimizer has no finite nonzero gradient")
                self.before_update = snapshot_trainable_parameters(model)
                return control

            def on_step_end(self, args, state, control, model=None, **unused):
                from ir_training.train.tensor_checks import tensor_finite_and_nonzero
                if not trainer.dpo_update_metrics:
                    raise ValueError("DPO optimizer step lacks preference-loss evidence")
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad and not tensor_finite_and_nonzero(parameter.detach())[0]:
                        raise ValueError(f"Nonfinite updated DPO parameter: {name}")
                metrics = trainer.dpo_update_metrics
                change = parameter_update_max_abs(model, self.before_update)
                self.before_update = None
                values = torch.tensor([sum(row[key] for row in metrics) / len(metrics)
                    for key in ("loss", "margin", "preference_accuracy")] + [self.gradient_norm, 1.0, change],
                    dtype=torch.float64, device=trainer.accelerator.device)
                values = trainer.accelerator.reduce(values, reduction="mean").cpu().tolist()
                row = dict(zip(("loss", "margin", "preference_accuracy", "gradient_l2_norm", "finite_parameters",
                                "parameter_update_max_abs"), values))
                if any(not math.isfinite(value) for value in values):
                    raise ValueError("Nonfinite distributed DPO health evidence")
                window.append(row)
                snapshot = health_snapshot(list(window), int(state.global_step))
                metadata["grpo"]["health"] = snapshot
                snapshots[int(state.global_step)] = snapshot
                trainer.dpo_update_metrics.clear()
                with (output / f"dpo_health.rank{rank}.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": int(state.global_step), **row}, allow_nan=False) + "\n")
                trainer.log({f"dpo/{key}": value for key, value in row.items()})
                return control

        trainer.add_callback(DPOHealth())
        if bundle.controller.saturation_monitor is not None:
            from ir_training.qat.saturation import build_saturation_trainer_callback
            saturation = build_saturation_trainer_callback(bundle.controller.saturation_monitor,
                report_path=output / "saturation_telemetry.json", log_dir=logging_dir)
            if saturation is not None:
                trainer.add_callback(saturation)
        golden = sft._build_optional_golden_callback(golden_eval_cfg=config.get("golden_eval") or {},
            base=training_root(), output_dir=output, adapter=bundle.adapter, tokenizer=bundle.tokenizer,
            model_cfg=config["model"], training_cfg=training, metric_logger=trainer.log,
            tensorboard_root=sft._resolve_training_tensorboard_root(training), tensorboard_run_id=str(run.get("id") or output.name))
        if golden is None or not (grpo.get("quality_control") or {}).get("enabled"):
            raise ValueError("DPO requires existing Golden baseline/nonregression checkpoint gates")
        trainer.add_callback(golden)
        from ir_training.train.grpo_experiment import (
            QualityPolicy,
            make_quality_callback,
        )
        golden_cfg = dict(config["golden_eval"])
        baseline_dir = output / "sft_starting_baseline"
        baseline = sft._build_optional_golden_callback(golden_eval_cfg={**golden_cfg, "output_dir": str(baseline_dir),
            "save_best_checkpoint": False, "log_to_trainer": False}, base=training_root(), output_dir=output,
            adapter=bundle.adapter, tokenizer=bundle.tokenizer, model_cfg=config["model"], training_cfg=training)
        quality = make_quality_callback(accelerator=trainer.accelerator, baseline_callback=baseline,
            golden_callback=golden, baseline_dir=baseline_dir,
            golden_dir=resolve_path(golden_cfg.get("output_dir", output / "golden_eval"), training_root()),
            output=output, config_path=config_path,
            policy=QualityPolicy(**((grpo["quality_control"].get("policy")) or {})))
        trainer.add_callback(quality)
        from ir_training.train.callbacks import build_checkpoint_provenance_callback
        trainer.add_callback(build_checkpoint_provenance_callback(output_dir=output, metadata=metadata,
            tokenizer=bundle.tokenizer, generation_eos_ids=bundle.generation_eos_ids, config_path=config_path,
            golden_summary_provider=golden.summary))

        class SelectedHealth(TrainerCallback):
            def on_save(self, args, state, control, **unused):
                if state.is_world_process_zero:
                    _bind_selected_health(golden.summary(), snapshots)
                return control

        trainer.add_callback(SelectedHealth())
        trainer.train()
        metadata["grpo"]["quality_gate"] = quality.finalize()
        if not metadata["grpo"].get("health", {}).get("passed_window"):
            raise ValueError("DPO did not complete its 20-update finite-health window")
        final_dir = output / "final_adapter"
        bundle.controller.restore()
        trainer.save_model(str(final_dir))
        if trainer.is_world_process_zero():
            bundle.tokenizer.save_pretrained(str(final_dir))
            metadata["final_adapter"] = str(final_dir)
            checkpoints = [("final", final_dir)]
            summary = golden.summary()
            if summary and summary.get("checkpoint_dir"):
                metadata["best_golden_eval"] = summary
                checkpoints.append(("best_golden", Path(summary["checkpoint_dir"])))
            sft._write_final_checkpoint_metadata(output_dir=output, metadata=metadata, checkpoint_paths=checkpoints,
                final_step=int(trainer.state.global_step), final_epoch=trainer.state.epoch, config_path=config_path)
            _bind_selected_health(summary, snapshots)
        sft._barrier_if_distributed()
        return metadata
    finally:
        if bundle.controller is not None:
            bundle.controller.restore()
