from __future__ import annotations

import json
import inspect
import shutil
from pathlib import Path
from typing import Any

from ir_training.common.config import repo_root, resolve_path, training_root
from ir_training.common.git import current_commit
from ir_training.models.registry import create_adapter
from ir_training.train.callbacks import TrainingMetadataCallback
from ir_training.train.lora_config import build_lora_config


def train_sft(config: dict[str, Any], config_path: Path | None = None) -> dict[str, Any]:
    try:
        from datasets import load_dataset  # type: ignore
        from peft import get_peft_model, prepare_model_for_kbit_training  # type: ignore
        from transformers import TrainingArguments  # type: ignore
        from trl import SFTTrainer  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency failure path
        raise RuntimeError("Install training/requirements-training.txt before running SFT training.") from exc

    run_cfg = config.get("run") if isinstance(config.get("run"), dict) else {}
    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else {}
    training_cfg = config.get("training") if isinstance(config.get("training"), dict) else {}
    lora_cfg = config.get("lora") if isinstance(config.get("lora"), dict) else {}

    base = training_root()
    dataset_dir = resolve_path(run_cfg.get("dataset_dir", "outputs/datasets/dataset_v1_stage3"), base)
    output_dir = resolve_path(run_cfg.get("output_dir", "runs/gemma_e2b_ir_lora"), base)
    train_path = dataset_dir / "train.jsonl"
    val_path = dataset_dir / "val.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train split: {train_path}")

    adapter = create_adapter(model_cfg)
    tokenizer = adapter.load_tokenizer()
    model = adapter.load_model()
    if bool(model_cfg.get("load_in_4bit", False)):
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, build_lora_config(adapter, lora_cfg))

    data_files: dict[str, str] = {"train": str(train_path)}
    if val_path.exists():
        data_files["validation"] = str(val_path)
    dataset = load_dataset("json", data_files=data_files)

    def formatting_func(example: dict[str, Any]) -> str:
        return adapter.format_example(example, tokenizer=tokenizer, include_assistant=True)

    eval_strategy_name = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )
    training_args_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(training_cfg.get("epochs", 2)),
        "learning_rate": float(training_cfg.get("learning_rate", 2e-4)),
        "per_device_train_batch_size": int(training_cfg.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(training_cfg.get("gradient_accumulation_steps", 16)),
        "warmup_ratio": float(training_cfg.get("warmup_ratio", 0.03)),
        "logging_steps": int(training_cfg.get("logging_steps", 20)),
        "save_steps": int(training_cfg.get("save_steps", 500)),
        "eval_steps": int(training_cfg.get("eval_steps", 500)),
        eval_strategy_name: "steps" if "validation" in dataset else "no",
        "save_total_limit": 3,
        "bf16": str(model_cfg.get("dtype", "bfloat16")).lower() == "bfloat16",
        "report_to": "none",
    }
    args = TrainingArguments(**training_args_kwargs)

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset.get("validation"),
        "args": args,
        "formatting_func": formatting_func,
    }
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    if "max_seq_length" in trainer_params:
        trainer_kwargs["max_seq_length"] = int(training_cfg.get("max_seq_length", adapter.max_context()))
    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    final_adapter = output_dir / "final_adapter"
    trainer.model.save_pretrained(str(final_adapter))
    tokenizer.save_pretrained(str(final_adapter))

    metadata = {
        "run_id": run_cfg.get("id", output_dir.name),
        "model": model_cfg,
        "training": training_cfg,
        "lora": lora_cfg,
        "dataset_dir": str(dataset_dir),
        "final_adapter": str(final_adapter),
        "git_commit": current_commit(repo_root()),
    }
    if config_path is not None:
        metadata["config_path"] = str(config_path)
        shutil.copy2(config_path, output_dir / "config.yaml")
    TrainingMetadataCallback(output_dir, metadata).write()
    return metadata
