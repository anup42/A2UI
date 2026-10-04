"""GRPO on the existing QAT LoRA policy, with shared SFT/export contracts.

Imports are deliberately CPU/tooling safe. The GPU runner uses the same seed,
LoRA target resolution, QAT wrappers, numeric probes and checkpoint writers as
SFT; only sampling, rewards and the optimization objective belong to GRPO.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def validate_qat_grpo_config(config: dict[str, Any]) -> None:
    from ir_training.qat.workflow import validate_qat_config
    from ir_training.train.qat_grpo_contract import (
        validate_qat_grpo_config as validate_contract,
    )
    validate_contract(config)
    training, grpo = config.get("training") or {}, config.get("grpo") or {}
    if training.get("method") != "qat_lora_grpo" or (config.get("run") or {}).get("purpose") != "qat_lora_grpo_v1":
        raise ValueError("QAT GRPO requires its explicit method and run purpose")
    if grpo.get("family") not in {"e2b", "270m"}:
        raise ValueError("grpo.family must be e2b or 270m")
    if grpo.get("beta", 0) != 0:
        raise ValueError("QAT GRPO requires beta=0: disabling the active adapter is not a QAT reference policy")
    if grpo.get("use_vllm", False) or training.get("distributed_backend", "ddp") != "ddp":
        raise ValueError("QAT GRPO supports HF generation with single-device/DDP only")
    if (grpo.get("top_p", 1.0) != 1.0 or grpo.get("top_k", 0) != 0
            or grpo.get("repetition_penalty", 1.0) != 1.0 or grpo.get("num_iterations", 1) != 1):
        raise ValueError("QAT GRPO requires unfiltered sampling and one iteration per fresh generation batch")
    temperature = grpo.get("temperature", 1.0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("grpo.temperature must be finite and positive")
    if training.get("resume_from_checkpoint") or config.get("resume_from_checkpoint"):
        raise ValueError("QAT GRPO starts a new optimizer from grpo.sft_checkpoint; optimizer resume is not supported")
    if training.get("ddp_broadcast_buffers", False) is not False:
        raise ValueError("QAT GRPO requires ddp_broadcast_buffers=false")
    if training.get("gradient_checkpointing_kwargs", {"use_reentrant": False}) != {"use_reentrant": False}:
        raise ValueError("QAT GRPO requires non-reentrant gradient checkpointing")
    if training.get("logging_steps", 1) != 1:
        raise ValueError("QAT GRPO health monitoring requires logging_steps=1")
    for key, default in (("num_generations", 4), ("max_prompt_length", 4096), ("max_completion_length", 2048)):
        value = grpo.get(key, default)
        if type(value) is not int or value < (2 if key == "num_generations" else 1):
            raise ValueError(f"grpo.{key} has an invalid value")
    for key in ("per_device_train_batch_size", "gradient_accumulation_steps"):
        if type(training.get(key)) is not int or training[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    errors = [issue for issue in validate_qat_config(config) if issue.severity == "error"]
    if errors:
        raise ValueError("QAT GRPO configuration failed: " + "; ".join(f"{item.code}: {item.message}" for item in errors))


def disable_policy_dropout(model: Any) -> dict[str, Any]:
    """Keep policy likelihoods consistent across rollout, old logprobs and loss.

Attention implementations can use functional dropout driven by a numeric
attribute, so disabling only nn.Dropout would be insufficient.
    """
    from torch import nn
    changes = []
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.dropout._DropoutNd) and module.p:
            changes.append({"module": name, "field": "p", "previous": float(module.p)})
            module.p = 0.0
        for key in ("attention_dropout", "hidden_dropout", "activation_dropout", "resid_pdrop", "embd_pdrop", "attn_pdrop"):
            value = getattr(module, key, None)
            if isinstance(value, (float, int)) and not isinstance(value, bool) and value:
                changes.append({"module": name, "field": key, "previous": float(value)})
                setattr(module, key, 0.0)
    return {"disabled": True, "changes": changes}


def assert_qat_policy_active(model: Any, controller: Any) -> None:
    while hasattr(model, "module"):
        model = model.module
    if getattr(model, "_a2ui_qat_controller", None) is not controller or not controller._original_forwards:
        raise RuntimeError("QAT policy wrappers disappeared before GRPO generation/loss")
    for module, original in controller._original_forwards.items():
        if module.forward == original:
            raise RuntimeError("A QAT policy forward was restored during GRPO")
    if any(parameter.requires_grad and "lora_" not in name for name, parameter in model.named_parameters()):
        raise RuntimeError("QAT GRPO may update only LoRA A/B parameters")


@dataclass
class QATGRPOModel:
    config: dict[str, Any]
    model: Any
    tokenizer: Any
    adapter: Any
    metadata: dict[str, Any]
    generation_eos_ids: list[int]
    controller: Any = None


def load_qat_grpo_model(config: dict[str, Any], *, config_path: Path | None = None) -> QATGRPOModel:
    """Load the verified original base and its trained adapter, never merge it."""
    validate_qat_grpo_config(config)
    from peft import PeftModel
    from transformers import set_seed

    from ir_training.common.config import repo_root, resolve_path, training_root
    from ir_training.common.git import current_commit
    from ir_training.generation_policy import preserve_generation_eos
    from ir_training.models.registry import create_adapter
    from ir_training.qat.mobile_seed_architecture import (
        validate_mobile_seed_architecture,
    )
    from ir_training.qat.mobile_training_seed import (
        verify_configured_mobile_training_seed,
    )
    from ir_training.train import sft
    from ir_training.train.lora_config import (
        build_lora_config,
        load_retained_mobile_lora_qparams,
        resolve_lora_config_targets,
    )
    from ir_training.train.lora_targets import bind_retained_mobile_peft_targets
    from ir_training.train.qat_grpo_contract import verify_sft_adapter_lineage

    sft._stabilize_torch_runtime()
    model_cfg, training = dict(config["model"]), config["training"]
    sft._enforce_cuda_requirement(model_cfg, training)
    sft._enforce_ddp_launch_requirement(training)
    initialization_seed = sft._initialize_training_seed(training, config.get("run") or {}, seed_setter=set_seed)
    lineage = verify_sft_adapter_lineage(config)
    if lineage.get("verified") is not True:
        raise ValueError(f"SFT adapter provenance failed before loading: {lineage}")
    seed = verify_configured_mobile_training_seed(model_cfg, base=training_root(), require_materialized=True)
    if not seed.get("verified"):
        raise ValueError("QAT GRPO original mobile seed identity verification failed")
    architecture = {"required": bool(seed.get("required")), "verified": not bool(seed.get("required"))}
    if seed.get("required"):
        architecture = validate_mobile_seed_architecture(model_cfg, base=training_root(), verified_seed=seed)
        if not architecture.get("verified"):
            raise ValueError("QAT GRPO mobile seed architecture verification failed")
    model_cfg["dtype"] = sft._resolve_training_dtype(str(model_cfg.get("dtype", "bfloat16")))
    adapter = create_adapter(model_cfg)
    tokenizer, model = adapter.load_tokenizer(), adapter.load_model()
    sft._align_tokenizer_and_model(tokenizer, model)
    sft._assert_tokenizer_model_vocab_alignment(tokenizer, model, context="QAT GRPO original seed")
    eos_ids = preserve_generation_eos(model, tokenizer)
    lora_config = build_lora_config(adapter, config["lora"])
    qparams = load_retained_mobile_lora_qparams(model_cfg, config["qat"], verified_seed=seed)
    targets = resolve_lora_config_targets(lora_config, model, retained_mobile_qparams=qparams)
    checkpoint = resolve_path(config["grpo"]["sft_checkpoint"], training_root())
    sft._require_peft_resume_checkpoint(checkpoint)
    model = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=True)
    if qparams is not None:
        bind_retained_mobile_peft_targets(model, targets)
    source_adapter = sft._validate_resumed_lora_model(model, expected_config=lora_config, checkpoint=checkpoint)
    sft._disable_peft_vocab_probe(model)
    sft._disable_model_cache_for_training(model)
    sft._enable_input_grads_for_kbit_lora(model)
    sft._align_tokenizer_and_model(tokenizer, model)
    sft._assert_tokenizer_model_vocab_alignment(tokenizer, model, context="QAT GRPO trained SFT adapter")
    sft._place_model_for_training(model)
    tokenizer.padding_side = "left"
    metadata = {
        "training_metadata_version": 4, "checkpoint_kind": "lora_adapter", "model": model_cfg,
        "git_commit": current_commit(repo_root()),
        "training": dict(training), "lora": dict(config["lora"]), "initialization_seed": initialization_seed,
        "resolved_lora_targets": sorted(targets), "mobile_training_seed": seed,
        "mobile_seed_architecture": architecture, "source_adapter": source_adapter,
        "generation_eos_token_ids": eos_ids, "dropout": disable_policy_dropout(model),
        "grpo": {"schema_version": 1, "algorithm": "grpo", "family": config["grpo"]["family"],
                 "source_lineage": lineage, "beta": 0.0},
    }
    if config_path is not None:
        metadata.update(config_path=str(config_path), training_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
    return QATGRPOModel(config, model, tokenizer, adapter, metadata, eos_ids)


def _backward_probe(bundle: QATGRPOModel, row: dict[str, Any], *, max_seq_length: int) -> dict[str, Any]:
    """Finite QAT completion likelihood gradients; no optimizer or data mutation."""
    import torch

    from ir_training.train import sft
    model = bundle.model
    collator = sft._CausalLMDataCollator(bundle.tokenizer,
        input_vocab_size=sft._require_model_input_vocab_size(model),
        label_vocab_size=sft._require_model_label_vocab_size(model), max_position_embeddings=max_seq_length)
    batch = {key: value.to(sft._model_input_device(model)) for key, value in collator([row]).items()}
    labels = batch.pop("labels")
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    try:
        assert_qat_policy_active(model, bundle.controller)
        logits = sft._extract_logits(model(**batch))
        loss = sft._checked_shifted_causal_lm_loss(logits, labels)
        loss.backward()
        trainables = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        finite = all(bool(torch.isfinite(p).all()) and (p.grad is None or bool(torch.isfinite(p.grad).all())) for _, p in trainables)
        nonzero = sum(p.grad is not None and bool(torch.count_nonzero(p.grad)) for _, p in trainables)
        gradients = [p.grad.detach().float() for _, p in trainables if p.grad is not None]
        gradient_norm = math.sqrt(sum(float(g.square().sum()) for g in gradients))
        gradient_max = max((float(g.abs().max()) for g in gradients), default=0.0)
        if (not math.isfinite(float(loss.detach())) or not finite or nonzero < 1
                or not math.isfinite(gradient_norm) or not math.isfinite(gradient_max)):
            raise RuntimeError("QAT GRPO likelihood backward preflight has nonfinite or zero gradients")
        return {"status": "passed", "kind": "completion_log_likelihood_backward", "loss": float(loss.detach()),
                "finite_parameters_and_gradients": finite, "nonzero_gradient_tensors": nonzero,
                "gradient_l2_norm": gradient_norm, "gradient_max_abs": gradient_max,
                "trainable_tensors": len(trainables), "optimizer_steps": 0}
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)


def run_qat_grpo_preflight(bundle: QATGRPOModel, rows: Sequence[Mapping[str, Any]], *, max_seq_length: int) -> dict[str, Any]:
    from ir_training.qat.fake_quant import prepare_qat_model
    from ir_training.train import sft
    from ir_training.train.qat_grpo_contract import verify_sft_adapter_lineage
    cfg = bundle.config.get("preflight") or {}
    count = max(int(cfg.get("rows", 1)), int(cfg.get("greedy_probe_rows", 1)))
    if len(rows) == 0:
        raise ValueError("QAT GRPO numeric probes need non-empty training rows")
    text_rows, token_rows = [], []
    for index in range(min(len(rows), count)):
        row = rows[index]
        prompt, completion = str(row["prompt"]), str(row["completion"])
        text_rows.append({"prompt_text": prompt, "completion_text": completion, "text": prompt + completion})
        token_rows.append(sft._tokenize_completion_only_row(tokenizer=bundle.tokenizer,
            prompt_text=prompt, completion_text=completion, full_text=prompt + completion, max_seq_length=max_seq_length))
    model = bundle.model
    kwargs = {"model": model, "tokenizer": bundle.tokenizer,
        "input_vocab_size": sft._require_model_input_vocab_size(model),
        "label_vocab_size": sft._require_model_label_vocab_size(model), "max_position_embeddings": max_seq_length,
        "split": token_rows, "max_rows": int(cfg.get("rows", 1)), "logit_probe_tokens": int(cfg.get("logit_probe_tokens", 16))}
    greedy_kwargs = {"model": model, "tokenizer": bundle.tokenizer, "split": text_rows, "max_position_embeddings": max_seq_length,
        "max_rows": int(cfg.get("greedy_probe_rows", 1)), "max_new_tokens": int(cfg.get("greedy_probe_new_tokens", 32)),
        "min_new_tokens": int(cfg.get("min_greedy_tokens", 8))}
    baseline = sft._run_forward_numeric_gate(**kwargs, label="sft_adapter_qat_off")
    baseline_greedy = sft._run_deterministic_greedy_gate(**greedy_kwargs, repeats=1, label="sft_adapter_qat_off")
    bundle.controller = prepare_qat_model(model, bundle.config)
    try:
        sft._require_trainable_qat_scope(bundle.config["qat"], bundle.controller)
        assert_qat_policy_active(model, bundle.controller)
        qat_on = sft._run_forward_numeric_gate(**kwargs, label="sft_adapter_qat_on")
        greedy = sft._run_deterministic_greedy_gate(**greedy_kwargs, repeats=2, label="sft_adapter_qat_on")
        report = sft._compare_initial_numeric_reports(baseline, qat_on, preflight_cfg=cfg, qat_enabled=True)
        report.update(adapter_initialization_mode="sft_adapter", source_adapter=bundle.metadata["source_adapter"],
            zero_adapter_initialization={"required": False, "reason": "continued_from_trained_sft_adapter", "verified_zero_delta": False},
            greedy_generation=sft._compare_initial_greedy_reports(baseline_greedy, greedy, preflight_cfg=cfg, qat_enabled=True))
        report["backward"] = _backward_probe(bundle, token_rows[0], max_seq_length=max_seq_length)
        bundle.metadata["numeric_preflight"] = report
        bundle.metadata["qat"] = bundle.controller.summary()
        bundle.metadata["grpo"]["source_lineage"] = verify_sft_adapter_lineage(bundle.config, numeric=report)
        if bundle.metadata["grpo"]["source_lineage"].get("verified") is not True:
            raise ValueError("Materialized SFT adapter failed QAT GRPO provenance validation")
        return report
    except BaseException:
        bundle.controller.restore()
        raise


def build_qat_grpo_trainer(base_trainer: Any, bundle: QATGRPOModel) -> Any:
    """Keep the single QAT policy bound through TRL rollout and loss paths."""
    class QATGRPOTrainer(base_trainer):
        def _get_per_token_logps_and_entropies(self, model: Any, *args: Any, **kwargs: Any) -> Any:
            assert_qat_policy_active(model, bundle.controller)
            with self.accelerator.autocast():
                return super()._get_per_token_logps_and_entropies(model, *args, **kwargs)

        def _generate_and_score_completions(self, *args: Any, **kwargs: Any) -> Any:
            assert_qat_policy_active(self.model, bundle.controller)
            return super()._generate_and_score_completions(*args, **kwargs)

        def compute_loss(self, model: Any, *args: Any, **kwargs: Any) -> Any:
            assert_qat_policy_active(model, bundle.controller)
            return super().compute_loss(model, *args, **kwargs)

        def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> Any:
            # Same generation-config serialization compatibility as checked SFT.
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

    return QATGRPOTrainer


def _script_module(name: str) -> Any:
    import sys

    from ir_training.common.config import training_root
    path = training_root() / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"a2ui_shared_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _file_identity(path: Path) -> dict[str, Any]:
    return {"path": str(path), "present": True, "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _health_snapshot(health: Any, step: int, thresholds: Any) -> dict[str, Any]:
    from ir_training.train.grpo_runtime import GRPOHealthMonitor
    rows = list(health.monitor.rows)
    monitor = GRPOHealthMonitor(thresholds)
    result: dict[str, Any] = {}
    for offset, row in enumerate(rows, step - len(rows) + 1):
        result = monitor.observe(offset, row)
    return {"passed_window": bool(health.passed_window), "optimizer_steps": step,
            "window_updates": len(rows), "last_step": health.monitor.last_step,
            "window_metrics": rows, "thresholds": asdict(thresholds),
            "aggregates": result.get("aggregates", {}),
            "last_metrics": rows[-1] if rows else {}}


def _bind_selected_health(summary: dict[str, Any] | None, snapshots: dict[int, dict[str, Any]]) -> None:
    """Bind a selected checkpoint to its own window, including interrupted runs."""
    if not summary or not summary.get("checkpoint_dir"):
        return
    from ir_training.train.callbacks import _write_checkpoint_provenance
    selected = Path(summary["checkpoint_dir"])
    metadata_path = selected / "training_metadata.json"
    if not metadata_path.is_file():
        raise ValueError("Golden checkpoint lacks materialized provenance")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["grpo"]["health"] = snapshots.get(int(summary["step"]), {"passed_window": False})
    _write_checkpoint_provenance(selected, role="best_golden", payload=payload)


def verify_prepared_training_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Reuse the prepared-split hash and reserved-cohort checks before training."""
    from ir_training.common.config import resolve_path, training_root
    from ir_training.eval.prepared_contract import verify_reserved_train_validation
    base = training_root()
    dataset = resolve_path(config["run"]["dataset_dir"], base)
    golden = config.get("golden_eval") or {}
    split = resolve_path(golden["split_path"], base) if golden.get("split_path") else resolve_path(golden["dataset_dir"], base) / f"{golden.get('split', 'all')}.jsonl"
    final = config.get("final_evaluation_datasets") or {}
    reserved = {}
    for name in ("golden35", "bixby50"):
        value = final.get(name)
        if isinstance(value, dict):
            value = value.get("path") or value.get("split_path")
        if value:
            reserved[name] = resolve_path(value, base)
        elif (dataset / f"{name}.jsonl").is_file():
            reserved[name] = dataset / f"{name}.jsonl"
    result = _script_module("prepare_review_training").verify_prepared(dataset, split,
        max_sequence=int(config["training"]["max_seq_length"]),
        max_prompt=int(golden["max_input_tokens"]), **reserved)
    verify_reserved_train_validation(dataset, [split, *reserved.values()])
    return result


def make_qat_rollout(bundle: QATGRPOModel, output: str, *, audit_limit: int):
    from ir_training.train.grpo_runtime import make_express_rollout
    shared = make_express_rollout(output, bundle.generation_eos_ids, audit_limit=audit_limit)

    def rollout(prompts: list[str], trainer: Any) -> dict[str, Any]:
        assert_qat_policy_active(trainer.model, bundle.controller)
        with trainer.accelerator.autocast():
            return shared(prompts, trainer)

    return rollout


def train_qat_grpo(config: dict[str, Any], config_path: Path, *, dependency_preflight_only: bool = False) -> dict[str, Any]:
    """Run GRPO then write the same PEFT checkpoints consumed by shared export."""
    validate_qat_grpo_config(config)
    from ir_training.common.config import resolve_path, training_root
    from ir_training.generation_policy import preserve_generation_eos
    from ir_training.train import sft
    from ir_training.train.callbacks import build_checkpoint_provenance_callback
    from ir_training.train.grpo_runtime import (
        HealthThresholds,
        audited_reward,
        dependency_report,
        make_health_callback,
        validate_runtime_features,
    )
    from ir_training.train.prepared_binding import verify_tokenizer_binding
    training, grpo, run = config["training"], config["grpo"], config["run"]
    output = resolve_path(run["output_dir"], training_root())
    output.mkdir(parents=True, exist_ok=True)
    rank = max(0, int(os.environ.get("RANK", "0")))
    dependencies = dependency_report()
    dependency_path = output / f"grpo_dependencies.rank{rank}.json"
    dependency_path.write_text(json.dumps(dependencies, indent=2), encoding="utf-8")
    errors = [name for name, info in dependencies["packages"].items() if info.get("import_error")]
    if errors or dependencies["packages"]["trl"].get("outside_distribution_root"):
        raise RuntimeError(f"GRPO dependency/source preflight failed: {errors}; see {dependency_path}")
    import torch
    import trl
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    validate_runtime_features(GRPOConfig, GRPOTrainer, str(trl.__version__))
    if not hasattr(TrainerCallback, "on_pre_optimizer_step"):
        raise RuntimeError("GRPO health monitoring requires TrainerCallback.on_pre_optimizer_step")
    if dependency_preflight_only:
        return {"dependencies": dependencies, "training_executed": False}
    if not torch.cuda.is_available():
        raise RuntimeError("QAT GRPO training requires CUDA; use --dependency-preflight-only for environment validation")
    thresholds = HealthThresholds(**(grpo.get("health") or {}))
    limit = sft._training_limit_config(training)
    if limit["max_optimizer_steps"] is not None and limit["max_optimizer_steps"] < thresholds.window_steps:
        raise ValueError("GRPO max_steps must cover at least one complete health window")
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    generations = grpo.get("num_generations", 4)
    micro, accumulation = training["per_device_train_batch_size"], training["gradient_accumulation_steps"]
    effective_batch = micro * accumulation * world_size
    if effective_batch % generations:
        raise ValueError("GRPO global accumulated sample batch must be divisible by num_generations")
    prepared_contract = verify_prepared_training_contract(config)
    bundle = load_qat_grpo_model(config, config_path=config_path)
    try:
        dataset_dir = resolve_path(run["dataset_dir"], training_root())
        prepared_binding = verify_tokenizer_binding(dataset_dir, bundle.tokenizer, config["model"], required=True)
        shared_data = _script_module("train_grpo")
        datasets = {}
        for name in ("train", "val"):
            path = dataset_dir / f"{name}.jsonl"
            if path.is_file() and path.stat().st_size:
                datasets[name] = shared_data.load_training_dataset(str(path), None, tokenizer=bundle.tokenizer,
                    prompt_source="prepared", chat_template_kwargs=config["model"].get("chat_template_kwargs") or {})
        train_data, eval_data = datasets["train"], datasets.get("val")
        if len(train_data) < effective_batch // generations:
            raise ValueError("Prepared train split is too small for a complete GRPO generation batch")
        max_prompt = grpo.get("max_prompt_length", 4096)
        max_completion = grpo.get("max_completion_length", 2048)
        context = sft._model_position_limit(bundle.model) or int(config["model"]["max_context_tokens"])
        if max_prompt + max_completion + 1 > context:
            raise ValueError("GRPO prompt + completion + masked terminal sentinel exceeds model context")
        for name, split in datasets.items():
            for index, row in enumerate(split):
                if row["prompt_provenance"]["prompt_tokens"] > max_prompt:
                    raise ValueError(f"Prepared {name}[{index}] exceeds GRPO prompt budget; truncation is forbidden")
        initial_lineage = bundle.metadata["grpo"]["source_lineage"]
        numeric = run_qat_grpo_preflight(bundle, train_data, max_seq_length=min(context, int(training["max_seq_length"])))
        preflight_path = output / f"grpo_preflight.rank{rank}.json"
        preflight_path.write_text(json.dumps({"schema_version": 1,
            "training_config_sha256": bundle.metadata["training_config_sha256"], "numeric_preflight": numeric,
            "source_lineage": initial_lineage}, indent=2), encoding="utf-8")
        bundle.metadata["grpo"]["preflight"] = _file_identity(preflight_path)
        from pipeline.genui_quality import (
            ensure_v5_4_validation_ready,
            load_reward_config_v5_4,
        )

        from ir_training.train.qat_grpo_reward import (
            QAT_GRPO_REWARD_VERSION,
            make_qat_grpo_reward,
        )
        ensure_v5_4_validation_ready()
        reward_path = resolve_path(grpo.get("reward_config", "../dataset/configs/genui_metric_v5_4.yaml"), training_root())
        reward = make_qat_grpo_reward(load_reward_config_v5_4(reward_path), model_checkpoint=grpo["sft_checkpoint"])
        reward = audited_reward(reward, str(output), rank, audit_limit=int(grpo.get("audit_rollout_limit", 256)))
        bundle.metadata["grpo"]["reward"] = {"version": QAT_GRPO_REWARD_VERSION, "config": _file_identity(reward_path)}
        eval_batch = training.get("per_device_eval_batch_size") or generations // math.gcd(generations, world_size)
        if eval_data is not None and eval_batch * world_size % generations:
            raise ValueError("Global GRPO evaluation batch must be divisible by num_generations")
        kwargs = dict(output_dir=str(output), learning_rate=float(training.get("learning_rate", 5e-6)),
            num_train_epochs=limit["num_train_epochs"], max_steps=limit["max_optimizer_steps"] or -1,
            per_device_train_batch_size=micro, per_device_eval_batch_size=eval_batch,
            gradient_accumulation_steps=accumulation, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False}, ddp_broadcast_buffers=False,
            ddp_find_unused_parameters=False, dataloader_drop_last=True, logging_steps=1,
            logging_nan_inf_filter=False, save_strategy="steps", save_steps=int(training.get("save_steps", 100)),
            eval_strategy="steps" if eval_data is not None else "no", eval_steps=int(training.get("eval_steps", 100)),
            save_total_limit=int(training.get("save_total_limit", 2)), report_to=training.get("report_to", "none"),
            remove_unused_columns=False, num_generations=generations, max_completion_length=max_completion,
            disable_dropout=True, cast_lm_head_to_fp32=False, temperature=float(grpo.get("temperature", 1.0)),
            top_p=1.0, top_k=0, repetition_penalty=1.0,
            steps_per_generation=accumulation, num_iterations=1, generation_kwargs={"eos_token_id": bundle.generation_eos_ids},
            scale_rewards=False if grpo.get("scale_rewards", "batch") == "none" else grpo.get("scale_rewards", "batch"),
            loss_type=grpo.get("loss_type", "dapo"), beta=0.0, reward_weights=[1.0],
            mask_truncated_completions=True, use_vllm=False, seed=bundle.metadata["initialization_seed"],
            max_grad_norm=float(training.get("max_grad_norm", 1.0)), weight_decay=float(training.get("weight_decay", 0.0)),
            **sft._training_precision_flags(bundle.metadata["model"]["dtype"], training))
        parameters = inspect.signature(GRPOConfig.__init__).parameters
        sft._apply_training_data_seed(training, kwargs, parameters)
        # The prepared prompt scan above is authoritative on runtimes that have
        # removed max_prompt_length; no library truncation is permitted.
        if "max_prompt_length" in parameters:
            kwargs["max_prompt_length"] = max_prompt
        if "warmup_steps" in training:
            kwargs["warmup_steps"] = training["warmup_steps"]
        elif "warmup_ratio" in parameters:
            kwargs["warmup_ratio"] = training.get("warmup_ratio", 0.03)
        logging_dir = sft._resolve_training_tensorboard_dir(training, run_id=str(run.get("id") or output.name))
        if logging_dir is not None and "logging_dir" in parameters:
            kwargs["logging_dir"] = str(logging_dir)
        trainer_cls = build_qat_grpo_trainer(GRPOTrainer, bundle)
        with sft._training_tensorboard_environment(logging_dir):
            trainer = trainer_cls(model=bundle.model, args=GRPOConfig(**kwargs), reward_funcs=[reward],
                train_dataset=train_data, eval_dataset=eval_data, processing_class=bundle.tokenizer,
                rollout_func=make_qat_rollout(bundle, str(output),
                    audit_limit=int(grpo.get("audit_rollout_limit", 256))))
        eos_ids = preserve_generation_eos(trainer.model, bundle.tokenizer, extra_eos_ids=bundle.generation_eos_ids)
        trainer.generation_config.eos_token_id = list(eos_ids)
        trainer.generation_kwargs["eos_token_id"] = list(eos_ids)
        assert_qat_policy_active(trainer.model, bundle.controller)
        health = make_health_callback(trainer.accelerator, str(output), thresholds)
        trainer.add_callback(health)
        if bundle.controller.saturation_monitor is not None:
            from ir_training.qat.saturation import build_saturation_trainer_callback
            saturation = build_saturation_trainer_callback(bundle.controller.saturation_monitor,
                report_path=output / "saturation_telemetry.json", log_dir=logging_dir)
            if saturation is not None:
                trainer.add_callback(saturation)
        metadata = bundle.metadata
        metadata.update(run_id=run.get("id", output.name), dataset_dir=str(dataset_dir),
            prepared_training_contract=prepared_contract, prepared_dataset_binding=prepared_binding,
            golden_eval=config.get("golden_eval") or {}, effective_batch_size=effective_batch,
            trainable_parameter_names=[n for n, p in trainer.model.named_parameters() if p.requires_grad],
            trainable_parameter_counts=dict(zip(("trainable", "total"), sft._trainable_parameter_count(trainer.model))))
        snapshots: dict[int, dict[str, Any]] = {}

        class HealthProvenance(TrainerCallback):
            def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **unused: Any) -> Any:
                if logs and "reward" in logs and "eval_reward" not in logs:
                    snapshot = _health_snapshot(health, int(state.global_step), thresholds)
                    metadata["grpo"]["health"] = snapshot
                    snapshots[int(state.global_step)] = snapshot
                return control

        trainer.add_callback(HealthProvenance())
        golden = sft._build_optional_golden_callback(golden_eval_cfg=config.get("golden_eval") or {},
            base=training_root(), output_dir=output, adapter=bundle.adapter, tokenizer=bundle.tokenizer,
            model_cfg=config["model"], training_cfg=training, metric_logger=trainer.log,
            tensorboard_root=sft._resolve_training_tensorboard_root(training), tensorboard_run_id=str(run.get("id") or output.name))
        if golden is not None:
            trainer.add_callback(golden)
        trainer.add_callback(build_checkpoint_provenance_callback(output_dir=output, metadata=metadata,
            tokenizer=bundle.tokenizer, generation_eos_ids=eos_ids, config_path=config_path,
            golden_summary_provider=golden.summary if golden is not None else None))

        class SelectedHealthProvenance(TrainerCallback):
            def on_save(self, args: Any, state: Any, control: Any, **unused: Any) -> Any:
                if state.is_world_process_zero and golden is not None:
                    _bind_selected_health(golden.summary(), snapshots)
                return control

        trainer.add_callback(SelectedHealthProvenance())
        trainer.train()
        final_dir = output / "final_adapter"
        # SFT restores fake-quant forwards for final PEFT serialization as well.
        bundle.controller.restore()
        trainer.save_model(str(final_dir))
        if trainer.is_world_process_zero():
            bundle.tokenizer.save_pretrained(str(final_dir))
            metadata["final_adapter"] = str(final_dir)
            checkpoints = [("final", final_dir)]
            summary = golden.summary() if golden is not None else None
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
