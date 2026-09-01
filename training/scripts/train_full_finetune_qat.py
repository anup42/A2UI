"""Full-parameter Gemma 3 270M A2UI Express QAT training.

This runner deliberately does not import or attach PEFT/LoRA.  It reuses the
project's model adapter, Express formatter, completion-only tokenizer/collator,
and true-QAT controller, while allowing Hugging Face Trainer to use a supplied
DeepSpeed configuration (ZeRO-2 for the Space run).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.config import load_yaml, resolve_path, training_root
from ir_training.common.git import current_commit
from ir_training.models.registry import create_adapter
from ir_training.qat.fake_quant import QATSpec, prepare_qat_model, qat_numeric_contract
from ir_training.train.sft import (
    _CausalLMDataCollator,
    _assert_tokenizer_model_vocab_alignment,
    _disable_model_cache_for_training,
    _effective_max_seq_length,
    _materialize_sft_text_dataset,
    _model_position_limit,
    _model_output_vocab_size,
    _model_vocab_size,
    _require_model_input_vocab_size,
    _require_model_label_vocab_size,
    _tokenize_sft_text_dataset,
)
from ir_training.train.callbacks import (
    build_checkpoint_provenance_callback,
    build_golden_set_eval_callback,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rank() -> int:
    try:
        return int(os.environ.get("RANK", "0"))
    except ValueError:
        return 0


def _world_size() -> int:
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        return 1


def _is_zero() -> bool:
    return _rank() == 0


def _print(message: str) -> None:
    print(message, flush=True)


def _sanitize_generation_config_for_checkpoint_save(
    model: Any, seen: set[int] | None = None
) -> None:
    """Keep training cache disabled while making Transformers saves valid.

    Gemma 3's base generation config can carry ``cache_implementation=hybrid``
    from a newer Transformers runtime. During training the model correctly
    uses ``use_cache=False``; however, Transformers validates the inherited
    generation config while writing a checkpoint and rejects that combination.
    Removing only the generation-time cache implementation preserves the model
    weights and training behavior while allowing a portable checkpoint.
    """
    if model is None:
        return
    if seen is None:
        seen = set()
    model_id = id(model)
    if model_id in seen:
        return
    seen.add(model_id)
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        if hasattr(generation_config, "cache_implementation"):
            generation_config.cache_implementation = None
        if hasattr(generation_config, "cache_config"):
            generation_config.cache_config = None
    for nested_attr in ("base_model", "model", "language_model", "module", "wrapped_model"):
        _sanitize_generation_config_for_checkpoint_save(
            getattr(model, nested_attr, None), seen
        )


def _path_from_config(value: Any, base: Path) -> Path:
    return resolve_path(str(value), base)


def _require_under(path: Path, root: Path, label: str) -> None:
    path = path.resolve()
    root = root.resolve()
    if not path.is_relative_to(root):
        raise RuntimeError(f"{label} must stay under {root}; got {path}")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _model_file_identity(model_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(model_dir), "files": []}
    for path in sorted(model_dir.glob("*.safetensors")):
        result["files"].append(
            {"name": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        path = model_dir / name
        if path.is_file():
            result[name] = {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
    return result


def _training_args(config: dict[str, Any], *, output_dir: Path, logging_dir: Path, deepspeed: Path, has_eval: bool) -> Any:
    from transformers import TrainingArguments  # type: ignore

    training = config.get("training") if isinstance(config.get("training"), dict) else {}
    params = inspect.signature(TrainingArguments.__init__).parameters
    eval_name = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(training.get("epochs", 2)),
        "learning_rate": float(training.get("learning_rate", 2e-5)),
        "weight_decay": float(training.get("weight_decay", 0.01)),
        "per_device_train_batch_size": int(training.get("per_device_train_batch_size", 2)),
        "per_device_eval_batch_size": int(training.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", 8)),
        "logging_steps": int(training.get("logging_steps", 10)),
        "logging_strategy": "steps",
        "save_strategy": "steps",
        "save_steps": int(training.get("save_steps", 2000)),
        "save_total_limit": int(training.get("save_total_limit", 3)),
        "eval_steps": int(training.get("eval_steps", 5000)) if has_eval else None,
        eval_name: "steps" if has_eval else "no",
        "max_grad_norm": float(training.get("max_grad_norm", 1.0)),
        "seed": int(training.get("seed", 42)),
        "bf16": True,
        "fp16": False,
        "report_to": ["tensorboard"],
        "logging_dir": str(logging_dir),
        "remove_unused_columns": False,
        "dataloader_num_workers": int(training.get("dataloader_num_workers", 4)),
        "dataloader_pin_memory": bool(training.get("dataloader_pin_memory", True)),
        "ddp_find_unused_parameters": False,
        "deepspeed": str(deepspeed),
        "load_best_model_at_end": False,
        "save_safetensors": True,
        "prediction_loss_only": True,
    }
    optional = {
        "tf32": True,
        "optim": str(training.get("optim", "adamw_torch_fused")),
        "gradient_checkpointing": bool(training.get("gradient_checkpointing", False)),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "warmup_ratio": float(training.get("warmup_ratio", 0.03)),
        "logging_first_step": True,
    }
    for key, value in optional.items():
        if key in params:
            kwargs[key] = value
    if "deepspeed" not in params:
        raise RuntimeError("Installed Transformers Trainer does not expose TrainingArguments.deepspeed")
    # Keep the runner compatible with the installed Transformers minor
    # version.  In particular, recent releases removed save_safetensors from
    # TrainingArguments because safetensors is now the default serialization.
    kwargs = {key: value for key, value in kwargs.items() if value is not None and key in params}
    return TrainingArguments(**kwargs)


def _trainer(model: Any, tokenizer: Any, args: Any, tokenized: Any, collator: Any) -> Any:
    from transformers import Trainer  # type: ignore

    params = inspect.signature(Trainer.__init__).parameters
    kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized.get("validation"),
        "data_collator": collator,
    }
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    return Trainer(**kwargs)


def _build_golden_eval_callback(
    *,
    config: dict[str, Any],
    output_dir: Path,
    adapter: Any,
    tokenizer: Any,
    model_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    trainer: Any,
) -> Any | None:
    golden_cfg = config.get("golden_eval")
    if not isinstance(golden_cfg, dict) or not bool(golden_cfg.get("enabled", False)):
        return None
    split_path_value = golden_cfg.get("split_path")
    if not split_path_value:
        raise ValueError("golden_eval.split_path is required when Golden evaluation is enabled")
    base = training_root()
    split_path = _path_from_config(split_path_value, base)
    eval_output_dir = _path_from_config(
        golden_cfg.get("output_dir", output_dir / "golden_eval"), base
    )
    best_checkpoint_value = golden_cfg.get("best_checkpoint_dir")
    best_checkpoint_dir = (
        _path_from_config(best_checkpoint_value, base)
        if best_checkpoint_value
        else output_dir / "best_golden_checkpoint"
    )
    max_input_tokens = int(
        golden_cfg.get(
            "max_input_tokens",
            training_cfg.get("max_seq_length", model_cfg.get("max_context_tokens", 8192)),
        )
    )
    max_new_tokens = int(
        golden_cfg.get("max_new_tokens", model_cfg.get("max_output_tokens", 1024))
    )
    model_context_tokens = int(model_cfg.get("max_context_tokens", 0) or 0)
    if model_context_tokens > 0 and max_input_tokens + max_new_tokens > model_context_tokens:
        max_new_tokens = max(1, model_context_tokens - max_input_tokens)
    callback = build_golden_set_eval_callback(
        enabled=True,
        split_path=split_path,
        output_dir=eval_output_dir,
        adapter=adapter,
        tokenizer=tokenizer,
        max_rows=int(golden_cfg.get("max_rows", 100)),
        required_rows=int(golden_cfg.get("required_rows", 100)),
        require_exact_rows=bool(golden_cfg.get("require_exact_rows", True)),
        require_unique_rows=bool(golden_cfg.get("require_unique_rows", True)),
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
        weights_config_path=(
            _path_from_config(golden_cfg["weights_config"], base)
            if golden_cfg.get("weights_config")
            else None
        ),
        baseline_aggregate_path=(
            _path_from_config(golden_cfg["baseline_aggregate"], base)
            if golden_cfg.get("baseline_aggregate")
            else None
        ),
        metric_version=str(golden_cfg.get("metric_version", "dual")),
        trigger=str(golden_cfg.get("trigger", "evaluate")),
        interval=int(golden_cfg.get("interval", 1)),
        metric_for_best_model=str(golden_cfg.get("metric_for_best_model", "overall_score")),
        greater_is_better=bool(golden_cfg.get("greater_is_better", True)),
        save_best_checkpoint=bool(golden_cfg.get("save_best_checkpoint", True)),
        best_checkpoint_dir=best_checkpoint_dir,
        metric_logger=getattr(trainer, "log", None),
        metric_log_prefix=str(golden_cfg.get("metric_log_prefix", "golden")),
    )
    if _is_zero():
        _print(
            f"Golden-100 checkpoint evaluation: split={split_path} "
            f"rows={int(golden_cfg.get('required_rows', 100))} "
            f"trigger={golden_cfg.get('trigger', 'evaluate')}"
        )
    return callback


def _validate_config(config: dict[str, Any], config_path: Path, task_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, Path]:
    run = config.get("run") if isinstance(config.get("run"), dict) else {}
    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else {}
    training = config.get("training") if isinstance(config.get("training"), dict) else {}
    qat = config.get("qat") if isinstance(config.get("qat"), dict) else {}
    if str(training.get("method", "")).strip().lower() != "full_finetune_qat":
        raise ValueError("training.method must be full_finetune_qat")
    if qat.get("enabled") is not True:
        raise ValueError("qat.enabled must be true")
    if int(qat.get("weight_bits", 0)) != 8 or int(qat.get("activation_bits", 0)) < 16:
        raise ValueError("This Gemma 3 INT8 run requires W8 with floating-point activation edges")
    if str(qat.get("quantizer", "")).strip().lower() != "ste_ai_edge":
        raise ValueError("This run requires the public ste_ai_edge QAT contract")
    if not bool(qat.get("quantize_embeddings", False)):
        raise ValueError("This run requires embedding QAT for the Gemma 3 INT8 package")
    model_id = str(model_cfg.get("model_id") or "")
    if "gemma-3-270m" not in model_id.lower():
        raise ValueError(f"Expected Gemma 3 270M base model, got {model_id!r}")
    dataset_dir = _path_from_config(run.get("dataset_dir"), training_root())
    output_dir = _path_from_config(run.get("output_dir"), training_root())
    logging_dir = _path_from_config(training.get("logging_dir", str(task_root / "tensorboard")), training_root())
    deepspeed_path = _path_from_config(training.get("deepspeed"), training_root())
    for path, label in ((output_dir, "output_dir"), (logging_dir, "logging_dir"), (deepspeed_path, "deepspeed config")):
        _require_under(path, task_root, label)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Missing A2UI Express dataset directory: {dataset_dir}")
    if not (dataset_dir / "train.jsonl").is_file():
        raise FileNotFoundError(f"Missing train.jsonl: {dataset_dir / 'train.jsonl'}")
    if not (dataset_dir / "val.jsonl").is_file():
        raise FileNotFoundError(f"Missing val.jsonl: {dataset_dir / 'val.jsonl'}")
    if not deepspeed_path.is_file():
        raise FileNotFoundError(f"Missing DeepSpeed config: {deepspeed_path}")
    spec = QATSpec.from_config(config)
    contract = qat_numeric_contract(spec)
    if contract.get("public_ai_edge_numeric_contract") is not True:
        raise ValueError(f"QAT numeric contract failed: {contract}")
    model_cfg = dict(model_cfg)
    model_cfg["device_map"] = "none"
    return run, model_cfg, training, dataset_dir, output_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Load/tokenize the real model and dataset, prepare QAT, run one forward, and stop before training.",
    )
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)
    task_root = Path(os.environ.get("A2UI_TASK_ROOT", config_path.parent)).expanduser().resolve()
    task_root.mkdir(parents=True, exist_ok=True)
    run, model_cfg, training_cfg, dataset_dir, output_dir = _validate_config(config, config_path, task_root)
    logging_dir = _path_from_config(training_cfg.get("logging_dir", str(task_root / "tensorboard")), training_root())
    deepspeed_path = _path_from_config(training_cfg["deepspeed"], training_root())
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch  # type: ignore
    from datasets import load_dataset  # type: ignore
    from transformers import set_seed  # type: ignore

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this full-finetune run")
    if torch.cuda.device_count() < 2 and _world_size() < 2:
        raise RuntimeError("Expected the two-H100 DeepSpeed launch to expose at least two GPUs")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    set_seed(int(training_cfg.get("seed", 42)))
    if _is_zero():
        _print(f"Full-finetune QAT: rank={_rank()} world_size={_world_size()} local_rank={local_rank}")
        _print(f"DeepSpeed config: {deepspeed_path}")
        _print(f"A2UI Express data: {dataset_dir}")
        _print(f"Output: {output_dir}")

    adapter = create_adapter(model_cfg)
    tokenizer = adapter.load_tokenizer()
    model = adapter.load_model()
    _disable_model_cache_for_training(model)
    _sanitize_generation_config_for_checkpoint_save(model)
    _assert_tokenizer_model_vocab_alignment(tokenizer, model, context="full-finetune initial model load")
    max_position_embeddings = _model_position_limit(model)
    max_seq_length = _effective_max_seq_length(
        int(training_cfg.get("max_seq_length", 4096)), max_position_embeddings
    )
    input_vocab_size = _require_model_input_vocab_size(model)
    label_vocab_size = _require_model_label_vocab_size(model)
    for parameter in model.parameters():
        parameter.requires_grad = True
    if any("lora_" in name.lower() for name, _ in model.named_parameters()):
        raise RuntimeError("Full-finetune run unexpectedly contains LoRA parameters")

    qat_controller = prepare_qat_model(model, config)
    qat_summary = qat_controller.summary()
    if qat_controller.wrapped_count < 1:
        raise RuntimeError("QAT preparation wrapped no model modules")
    if _is_zero():
        _print(
            f"QAT active: wrapped={qat_controller.wrapped_count} "
            f"linear={qat_summary['wrapped_linear_count']} "
            f"embedding={qat_summary['wrapped_embedding_count']} "
            f"weight_bits={qat_summary['spec']['weight_bits']} "
            f"activation_bits={qat_summary['spec']['activation_bits']}"
        )

    data_files = {"train": str(dataset_dir / "train.jsonl"), "validation": str(dataset_dir / "val.jsonl")}
    dataset = load_dataset("json", data_files=data_files, keep_in_memory=False)
    if _is_zero():
        _print(f"Dataset rows: train={len(dataset['train'])} validation={len(dataset['validation'])}")
    if len(dataset["train"]) < 150000:
        raise RuntimeError("The configured dataset is smaller than the expected ~2-lakh Express corpus")

    def format_example(example: dict[str, Any]) -> str:
        return adapter.format_example(example, tokenizer=tokenizer, include_assistant=True)

    def format_prompt(example: dict[str, Any]) -> str:
        return adapter.format_example(example, tokenizer=tokenizer, include_assistant=False)

    text_dataset = _materialize_sft_text_dataset(dataset, format_example, format_prompt)
    tokenized = _tokenize_sft_text_dataset(text_dataset, tokenizer, max_seq_length)
    collator = _CausalLMDataCollator(
        tokenizer,
        input_vocab_size=input_vocab_size,
        label_vocab_size=label_vocab_size,
        max_position_embeddings=max_position_embeddings,
    )
    model.to(torch.device(f"cuda:{local_rank}"))
    model.eval()
    probe = tokenized["train"][0]
    probe_input_ids = list(probe["input_ids"])
    probe_all_labels = list(probe["labels"])
    valid_label_positions = [
        index for index, label in enumerate(probe_all_labels)
        if int(label) != -100
    ]
    if not valid_label_positions:
        raise RuntimeError("QAT forward preflight probe has no trainable completion labels")
    # The first tokens are usually prompt-masked.  Probing that prefix makes
    # Transformers' mean cross-entropy undefined (all labels are -100), so
    # select a 128-token window ending at a real completion label.
    probe_end = min(len(probe_input_ids), valid_label_positions[-1] + 1)
    probe_start = max(0, probe_end - 128)
    if not any(int(label) != -100 for label in probe_all_labels[probe_start + 1:probe_end]):
        probe_start = max(0, valid_label_positions[0] - 1)
        probe_end = min(len(probe_input_ids), probe_start + 128)
    probe_ids = torch.tensor(
        [probe_input_ids[probe_start:probe_end]],
        dtype=torch.long,
        device=torch.device(f"cuda:{local_rank}"),
    )
    probe_labels = torch.tensor(
        [probe_all_labels[probe_start:probe_end]],
        dtype=torch.long,
        device=torch.device(f"cuda:{local_rank}"),
    )
    if not any(int(label) != -100 for label in probe_labels[0, 1:].tolist()):
        raise RuntimeError("QAT forward preflight probe window has no shifted completion labels")
    with torch.no_grad():
        probe_output = model(input_ids=probe_ids, labels=probe_labels)
    probe_loss = float(probe_output.loss.detach().float().cpu())
    if not torch.isfinite(probe_output.loss):
        raise RuntimeError("QAT forward preflight produced a non-finite loss")
    model.train()
    if _is_zero():
        _print(f"QAT forward preflight passed: probe_loss={probe_loss:.6f}")

    args_obj = _training_args(
        config, output_dir=output_dir, logging_dir=logging_dir, deepspeed=deepspeed_path,
        has_eval=True,
    )
    trainer = _trainer(model, tokenizer, args_obj, tokenized, collator)
    if args.preflight_only:
        qat_controller.restore()
        if _is_zero():
            _print("Full-finetune QAT preflight passed; optimizer/training was not started.")
        return 0
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    start_metadata = {
        "status": "running",
        "run_id": run.get("id", output_dir.name),
        "training_method": "full_finetune_qat",
        "lora": False,
        "deep_speed": {"enabled": True, "zero_stage": 2, "config": str(deepspeed_path)},
        "world_size": _world_size(),
        "model": _model_file_identity(Path(str(model_cfg.get("model_id")))),
        "model_id": model_cfg.get("model_id"),
        "dataset": {
            "dir": str(dataset_dir),
            "format": run.get("dataset_format", "a2ui_express_v1"),
            "prompt_version": run.get("prompt_version"),
            "train_rows": len(dataset["train"]),
            "validation_rows": len(dataset["validation"]),
            "train_sha256": _sha256(dataset_dir / "train.jsonl"),
            "validation_sha256": _sha256(dataset_dir / "val.jsonl"),
        },
        "qat": qat_summary,
        "qat_numeric_contract": qat_numeric_contract(QATSpec.from_config(config)),
        "probe_loss": probe_loss,
        "parameters": {"trainable": trainable, "total": total, "all_trainable": trainable == total},
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "git_commit": current_commit(ROOT.parent),
        "golden_eval": config.get("golden_eval", {}),
    }
    if _is_zero():
        _write_json(task_root / "run_metadata.json", start_metadata)
        shutil.copy2(config_path, task_root / "training_config.yaml")

    golden_callback = _build_golden_eval_callback(
        config=config,
        output_dir=output_dir,
        adapter=adapter,
        tokenizer=tokenizer,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        trainer=trainer,
    )
    if golden_callback is not None:
        trainer.add_callback(golden_callback)
    trainer.add_callback(
        build_checkpoint_provenance_callback(
            output_dir=output_dir,
            metadata=start_metadata,
            config_path=config_path,
            golden_summary_provider=(
                golden_callback.summary
                if golden_callback is not None
                else None
            ),
        )
    )

    resume = args.resume_from_checkpoint or None
    try:
        result = trainer.train(resume_from_checkpoint=resume)
    finally:
        qat_controller.restore()
    trainer.save_state()
    if _is_zero():
        final_model = task_root / "final_model"
        trainer.save_model(str(final_model))
        tokenizer.save_pretrained(str(final_model))
        metrics = dict(getattr(result, "metrics", {}) or {})
        complete = {
            **start_metadata,
            "status": "completed",
            "final_model": str(final_model),
            "global_step": int(getattr(trainer.state, "global_step", 0)),
            "metrics": metrics,
            "final_model_files": _model_file_identity(final_model),
        }
        _write_json(task_root / "training_complete.json", complete)
        _print(f"Training completed: step={complete['global_step']} final_model={final_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
