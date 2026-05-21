from __future__ import annotations

import json
import inspect
import shutil
from pathlib import Path
from typing import Any, Callable

from ir_training.common.config import repo_root, resolve_path, training_root
from ir_training.common.git import current_commit
from ir_training.models.registry import create_adapter
from ir_training.train.callbacks import TrainingMetadataCallback, build_golden_set_eval_callback
from ir_training.train.lora_config import build_lora_config


def train_sft(config: dict[str, Any], config_path: Path | None = None) -> dict[str, Any]:
    try:
        from datasets import load_dataset  # type: ignore
        from peft import get_peft_model, prepare_model_for_kbit_training  # type: ignore
        from transformers import TrainingArguments  # type: ignore
        from trl import SFTTrainer  # type: ignore
        try:
            from trl import SFTConfig  # type: ignore
        except Exception:  # pragma: no cover - older TRL versions
            SFTConfig = None
    except Exception as exc:  # pragma: no cover - dependency failure path
        raise RuntimeError("Install training/requirements-training.txt before running SFT training.") from exc

    run_cfg = config.get("run") if isinstance(config.get("run"), dict) else {}
    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else {}
    training_cfg = config.get("training") if isinstance(config.get("training"), dict) else {}
    lora_cfg = config.get("lora") if isinstance(config.get("lora"), dict) else {}
    golden_eval_cfg = config.get("golden_eval") if isinstance(config.get("golden_eval"), dict) else {}

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
    _align_tokenizer_and_model(tokenizer, model)
    if bool(model_cfg.get("load_in_4bit", False)):
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, build_lora_config(adapter, lora_cfg))

    data_files: dict[str, str] = {"train": str(train_path)}
    if val_path.exists():
        data_files["validation"] = str(val_path)
    dataset = load_dataset("json", data_files=data_files)
    _print_training_sample_summary(dataset)

    def formatting_func(example: dict[str, Any]) -> str:
        return adapter.format_example(example, tokenizer=tokenizer, include_assistant=True)

    args_cls = SFTConfig if SFTConfig is not None else TrainingArguments
    args_params = inspect.signature(args_cls.__init__).parameters
    eval_strategy_name = (
        "eval_strategy"
        if "eval_strategy" in args_params
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
    max_seq_length = int(training_cfg.get("max_seq_length", adapter.max_context()))
    if "completion_only_loss" in args_params:
        # formatting_func produces a full language-modeling text record.
        # TRL's completion-only loss is incompatible with that path.
        training_args_kwargs["completion_only_loss"] = bool(training_cfg.get("completion_only_loss", False))
    if "assistant_only_loss" in args_params:
        training_args_kwargs["assistant_only_loss"] = bool(training_cfg.get("assistant_only_loss", False))
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    supports_text_dataset = "dataset_text_field" in args_params or "dataset_text_field" in trainer_params
    if supports_text_dataset:
        dataset = _materialize_sft_text_dataset(dataset, formatting_func)
        _validate_sft_token_ids(
            dataset=dataset,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            vocab_size=_model_vocab_size(model),
            max_rows=int(training_cfg.get("preflight_token_check_rows", 0)),
        )
        if "dataset_text_field" in args_params:
            training_args_kwargs["dataset_text_field"] = "text"
        if "packing" in args_params:
            training_args_kwargs["packing"] = bool(training_cfg.get("packing", False))
    if "max_seq_length" in args_params:
        training_args_kwargs["max_seq_length"] = max_seq_length
    elif "max_length" in args_params:
        training_args_kwargs["max_length"] = max_seq_length
    args = args_cls(**training_args_kwargs)

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset.get("validation"),
        "args": args,
    }
    if supports_text_dataset:
        if "dataset_text_field" in trainer_params:
            trainer_kwargs["dataset_text_field"] = "text"
    else:
        trainer_kwargs["formatting_func"] = formatting_func
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    if "max_seq_length" in trainer_params and "max_seq_length" not in training_args_kwargs:
        trainer_kwargs["max_seq_length"] = max_seq_length
    trainer = SFTTrainer(**trainer_kwargs)

    golden_callback = _build_optional_golden_callback(
        golden_eval_cfg=golden_eval_cfg,
        base=base,
        output_dir=output_dir,
        adapter=adapter,
        tokenizer=tokenizer,
        model_cfg=model_cfg,
    )
    if golden_callback is not None:
        trainer.add_callback(golden_callback)

    trainer.train()
    final_adapter = output_dir / "final_adapter"
    trainer.model.save_pretrained(str(final_adapter))
    tokenizer.save_pretrained(str(final_adapter))

    metadata = {
        "run_id": run_cfg.get("id", output_dir.name),
        "model": model_cfg,
        "training": training_cfg,
        "lora": lora_cfg,
        "golden_eval": golden_eval_cfg,
        "dataset_dir": str(dataset_dir),
        "final_adapter": str(final_adapter),
        "git_commit": current_commit(repo_root()),
    }
    if config_path is not None:
        metadata["config_path"] = str(config_path)
        shutil.copy2(config_path, output_dir / "config.yaml")
    TrainingMetadataCallback(output_dir, metadata).write()
    return metadata


def _materialize_sft_text_dataset(dataset: Any, formatting_func: Callable[[dict[str, Any]], str]) -> Any:
    def add_text(example: dict[str, Any]) -> dict[str, str]:
        return {"text": formatting_func(example)}

    return dataset.map(add_text, desc="Formatting SFT text")


def _align_tokenizer_and_model(tokenizer: Any, model: Any) -> None:
    token_count = len(tokenizer) if hasattr(tokenizer, "__len__") else None
    vocab_size = _model_vocab_size(model)
    if token_count is not None and vocab_size is not None and token_count > vocab_size:
        model.resize_token_embeddings(token_count)
    for attr in ("pad_token_id", "bos_token_id", "eos_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is None:
            continue
        if hasattr(model, "config"):
            setattr(model.config, attr, value)
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            setattr(generation_config, attr, value)


def _model_vocab_size(model: Any) -> int | None:
    embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    return getattr(embeddings, "num_embeddings", None)


def _validate_sft_token_ids(
    *,
    dataset: Any,
    tokenizer: Any,
    max_seq_length: int,
    vocab_size: int | None,
    max_rows: int,
) -> None:
    if vocab_size is None:
        return
    split_names = [name for name in ("train", "validation") if name in dataset]
    for split_name in split_names:
        split = dataset[split_name]
        limit = len(split) if max_rows <= 0 else min(len(split), max_rows)
        for row_index in range(limit):
            text = str(split[row_index].get("text", ""))
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_seq_length,
                add_special_tokens=True,
            )
            input_ids = encoded.get("input_ids") or []
            if not input_ids:
                raise ValueError(f"SFT preflight failed: empty tokenization at {split_name}[{row_index}]")
            for token_id in input_ids:
                if not isinstance(token_id, int) or token_id < 0 or token_id >= vocab_size:
                    raise ValueError(
                        "SFT preflight failed: token id outside model vocabulary "
                        f"at {split_name}[{row_index}] token={token_id} vocab_size={vocab_size}. "
                        "Check tokenizer/model pairing before running CUDA training."
                    )


def _print_training_sample_summary(dataset: Any) -> None:
    summary = _summarize_training_sample_models(dataset)
    print("Training sample source summary:", flush=True)
    for split_name, split_summary in summary.items():
        print(f"  {split_name}: {split_summary['total']} samples", flush=True)
        _print_count_section("response models", split_summary["response_generation"])
        _print_count_section("IR models", split_summary["ir_generation"])
        _print_count_section("response -> IR model pairs", split_summary["response_to_ir_generation"])


def _print_count_section(title: str, counts: dict[str, int]) -> None:
    if not counts:
        print(f"    {title}: none", flush=True)
        return
    print(f"    {title}:", flush=True)
    for label, count in counts.items():
        print(f"      {label}: {count}", flush=True)


def _summarize_training_sample_models(dataset: Any) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    split_names = list(dataset.keys()) if hasattr(dataset, "keys") else []
    for split_name in split_names:
        split = dataset[split_name]
        split_summary: dict[str, Any] = {
            "total": len(split),
            "response_generation": {},
            "ir_generation": {},
            "response_to_ir_generation": {},
        }
        for row in split:
            metadata = row.get("metadata") if isinstance(row, dict) else {}
            if not isinstance(metadata, dict):
                metadata = {}
            response_label = _generation_label(metadata.get("response_generation"))
            ir_label = _generation_label(metadata.get("ir_generation") or metadata.get("source_generation"))
            pair_label = f"{response_label} -> {ir_label}"
            _increment_count(split_summary["response_generation"], response_label)
            _increment_count(split_summary["ir_generation"], ir_label)
            _increment_count(split_summary["response_to_ir_generation"], pair_label)
        for key in ("response_generation", "ir_generation", "response_to_ir_generation"):
            split_summary[key] = dict(sorted(split_summary[key].items()))
        summary[str(split_name)] = split_summary
    return summary


def _increment_count(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _generation_label(value: Any) -> str:
    if not isinstance(value, dict):
        return "unknown/unknown"
    provider = str(value.get("provider") or "unknown").strip() or "unknown"
    model = str(value.get("model") or "unknown").strip() or "unknown"
    return f"{provider}/{model}"


def _build_optional_golden_callback(
    *,
    golden_eval_cfg: dict[str, Any],
    base: Path,
    output_dir: Path,
    adapter: Any,
    tokenizer: Any,
    model_cfg: dict[str, Any],
) -> Any | None:
    if not bool(golden_eval_cfg.get("enabled", False)):
        return None
    split_path_value = golden_eval_cfg.get("split_path")
    if not split_path_value:
        dataset_dir = resolve_path(golden_eval_cfg.get("dataset_dir", "outputs/datasets/golden50_stage3_eval"), base)
        split_name = str(golden_eval_cfg.get("split", "all")).strip() or "all"
        split_path = dataset_dir / f"{split_name}.jsonl"
    else:
        split_path = resolve_path(split_path_value, base)
    eval_output_dir = resolve_path(
        golden_eval_cfg.get("output_dir", output_dir / "golden_eval"),
        base,
    )
    weights_config_path = golden_eval_cfg.get("weights_config")
    baseline_aggregate_path = golden_eval_cfg.get("baseline_aggregate")
    return build_golden_set_eval_callback(
        enabled=True,
        split_path=split_path,
        output_dir=eval_output_dir,
        adapter=adapter,
        tokenizer=tokenizer,
        max_rows=int(golden_eval_cfg.get("max_rows", 50)),
        max_new_tokens=int(golden_eval_cfg.get("max_new_tokens", model_cfg.get("max_output_tokens", 8192))),
        weights_config_path=resolve_path(weights_config_path, base) if weights_config_path else None,
        baseline_aggregate_path=resolve_path(baseline_aggregate_path, base) if baseline_aggregate_path else None,
    )
