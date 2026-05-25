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
        from transformers import Trainer, TrainingArguments  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency failure path
        raise RuntimeError("Install training/requirements-training.txt before running SFT training.") from exc

    run_cfg = config.get("run") if isinstance(config.get("run"), dict) else {}
    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else {}
    model_cfg = dict(model_cfg)
    training_cfg = config.get("training") if isinstance(config.get("training"), dict) else {}
    lora_cfg = config.get("lora") if isinstance(config.get("lora"), dict) else {}
    golden_eval_cfg = config.get("golden_eval") if isinstance(config.get("golden_eval"), dict) else {}
    _enforce_cuda_requirement(model_cfg, training_cfg)
    requested_dtype = str(model_cfg.get("dtype", "bfloat16")).lower()
    resolved_dtype = _resolve_training_dtype(requested_dtype)
    model_cfg["dtype"] = resolved_dtype

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
    _assert_tokenizer_model_vocab_alignment(tokenizer, model, context="initial model load")
    max_position_embeddings = _model_position_limit(model)
    if bool(model_cfg.get("load_in_4bit", False)):
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, build_lora_config(adapter, lora_cfg))
    _align_tokenizer_and_model(tokenizer, model)
    _assert_tokenizer_model_vocab_alignment(tokenizer, model, context="after LoRA wrapping")
    input_vocab_size = _require_model_input_vocab_size(model)
    label_vocab_size = _require_model_label_vocab_size(model)
    _print_tokenizer_model_alignment(tokenizer, model)
    _print_trainable_parameter_summary(model)

    data_files: dict[str, str] = {"train": str(train_path)}
    if val_path.exists():
        data_files["validation"] = str(val_path)
    dataset = load_dataset("json", data_files=data_files)
    _print_training_sample_summary(dataset)

    def formatting_func(example: dict[str, Any]) -> str:
        return adapter.format_example(example, tokenizer=tokenizer, include_assistant=True)

    def prompt_formatting_func(example: dict[str, Any]) -> str:
        return adapter.format_example(example, tokenizer=tokenizer, include_assistant=False)

    trainer_backend = str(training_cfg.get("trainer_backend", "hf")).strip().lower() or "hf"
    if trainer_backend not in {"hf", "trl"}:
        raise ValueError(f"Unsupported training.trainer_backend: {trainer_backend!r}. Use 'hf' or 'trl'.")

    SFTTrainer = None
    SFTConfig = None
    if trainer_backend == "trl":
        try:
            from trl import SFTTrainer as ImportedSFTTrainer  # type: ignore

            SFTTrainer = ImportedSFTTrainer
            try:
                from trl import SFTConfig as ImportedSFTConfig  # type: ignore

                SFTConfig = ImportedSFTConfig
            except Exception:  # pragma: no cover - older TRL versions
                SFTConfig = None
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("training.trainer_backend='trl' requires the trl package.") from exc

    args_cls = SFTConfig if trainer_backend == "trl" and SFTConfig is not None else TrainingArguments
    args_params = inspect.signature(args_cls.__init__).parameters
    eval_strategy_name = (
        "eval_strategy"
        if "eval_strategy" in args_params
        else "evaluation_strategy"
    )
    precision_flags = _training_precision_flags(resolved_dtype)
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
        **precision_flags,
        "report_to": "none",
    }
    max_seq_length = _effective_max_seq_length(
        configured=int(training_cfg.get("max_seq_length", adapter.max_context())),
        max_position_embeddings=max_position_embeddings,
    )
    if trainer_backend == "trl" and "completion_only_loss" in args_params:
        # formatting_func produces a full language-modeling text record.
        # TRL's completion-only loss is incompatible with that path.
        training_args_kwargs["completion_only_loss"] = bool(training_cfg.get("completion_only_loss", False))
    if trainer_backend == "trl" and "assistant_only_loss" in args_params:
        training_args_kwargs["assistant_only_loss"] = bool(training_cfg.get("assistant_only_loss", False))
    sft_text_dataset = _materialize_sft_text_dataset(dataset, formatting_func, prompt_formatting_func)
    _validate_sft_token_ids(
        dataset=sft_text_dataset,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        input_vocab_size=input_vocab_size,
        label_vocab_size=label_vocab_size,
        max_rows=int(training_cfg.get("preflight_token_check_rows", 0)),
        tokenizer_size=len(tokenizer) if hasattr(tokenizer, "__len__") else None,
    )

    if trainer_backend == "trl":
        if SFTTrainer is None:
            raise RuntimeError("TRL trainer backend was selected but SFTTrainer is unavailable.")
        trainer_params = inspect.signature(SFTTrainer.__init__).parameters
        supports_text_dataset = "dataset_text_field" in args_params or "dataset_text_field" in trainer_params
        if not supports_text_dataset:
            raise RuntimeError(
                "Configured TRL SFTTrainer does not support dataset_text_field. "
                "Use the default HF trainer backend or upgrade TRL."
            )
        dataset = sft_text_dataset
        if "dataset_text_field" in args_params:
            training_args_kwargs["dataset_text_field"] = "text"
        if "packing" in args_params:
            training_args_kwargs["packing"] = bool(training_cfg.get("packing", False))
        if "max_seq_length" in args_params:
            training_args_kwargs["max_seq_length"] = max_seq_length
        elif "max_length" in args_params:
            training_args_kwargs["max_length"] = max_seq_length
    args = args_cls(**training_args_kwargs)

    if trainer_backend == "trl":
        trainer_kwargs = {
            "model": model,
            "train_dataset": dataset["train"],
            "eval_dataset": dataset.get("validation"),
            "args": args,
        }
        if "dataset_text_field" in trainer_params:
            trainer_kwargs["dataset_text_field"] = "text"
        if "tokenizer" in trainer_params:
            trainer_kwargs["tokenizer"] = tokenizer
        elif "processing_class" in trainer_params:
            trainer_kwargs["processing_class"] = tokenizer
        if "max_seq_length" in trainer_params and "max_seq_length" not in training_args_kwargs:
            trainer_kwargs["max_seq_length"] = max_seq_length
        trainer = SFTTrainer(**trainer_kwargs)
    else:
        print("Using explicit HF Trainer causal-LM backend for SFT.", flush=True)
        tokenized_dataset = _tokenize_sft_text_dataset(sft_text_dataset, tokenizer, max_seq_length)
        _validate_tokenized_sft_dataset(
            dataset=tokenized_dataset,
            input_vocab_size=input_vocab_size,
            label_vocab_size=label_vocab_size,
            max_rows=int(training_cfg.get("preflight_token_check_rows", 0)),
        )
        _run_forward_smoke_check(
            model=model,
            split=tokenized_dataset["train"],
            tokenizer=tokenizer,
            input_vocab_size=input_vocab_size,
            label_vocab_size=label_vocab_size,
            max_position_embeddings=max_position_embeddings,
        )
        checked_trainer_cls = _build_checked_causal_lm_trainer(Trainer)
        trainer_params = inspect.signature(Trainer.__init__).parameters
        trainer_kwargs = {
            "model": model,
            "train_dataset": tokenized_dataset["train"],
            "eval_dataset": tokenized_dataset.get("validation"),
            "args": args,
            "data_collator": _CausalLMDataCollator(
                tokenizer,
                input_vocab_size=input_vocab_size,
                label_vocab_size=label_vocab_size,
                max_position_embeddings=max_position_embeddings,
            ),
        }
        if "tokenizer" in trainer_params:
            trainer_kwargs["tokenizer"] = tokenizer
        elif "processing_class" in trainer_params:
            trainer_kwargs["processing_class"] = tokenizer
        trainer = checked_trainer_cls(**trainer_kwargs)

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


def _materialize_sft_text_dataset(
    dataset: Any,
    formatting_func: Callable[[dict[str, Any]], str],
    prompt_formatting_func: Callable[[dict[str, Any]], str] | None = None,
) -> Any:
    def add_text(example: dict[str, Any]) -> dict[str, str]:
        text = formatting_func(example)
        prompt_text = prompt_formatting_func(example) if prompt_formatting_func is not None else ""
        completion_text = _completion_suffix_text(example, text, prompt_text)
        return {"text": text, "prompt_text": prompt_text, "completion_text": completion_text}

    return dataset.map(add_text, desc="Formatting SFT text")


def _tokenize_sft_text_dataset(dataset: Any, tokenizer: Any, max_seq_length: int) -> Any:
    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        rows = [
            _tokenize_completion_only_row(
                tokenizer=tokenizer,
                prompt_text=str(prompt_text or ""),
                completion_text=str(completion_text or ""),
                full_text=str(full_text or ""),
                max_seq_length=max_seq_length,
            )
            for prompt_text, completion_text, full_text in zip(
                batch.get("prompt_text", []),
                batch.get("completion_text", []),
                batch.get("text", []),
                strict=False,
            )
        ]
        return {
            "input_ids": [row["input_ids"] for row in rows],
            "attention_mask": [row["attention_mask"] for row in rows],
            "labels": [row["labels"] for row in rows],
        }

    tokenized_splits: dict[str, Any] = {}
    for split_name in dataset.keys():
        split = dataset[split_name]
        tokenized_splits[split_name] = split.map(
            tokenize_batch,
            batched=True,
            remove_columns=split.column_names,
            desc=f"Tokenizing {split_name} SFT text",
        )
    return tokenized_splits


def _completion_suffix_text(example: dict[str, Any], full_text: str, prompt_text: str) -> str:
    if prompt_text and full_text.startswith(prompt_text):
        suffix = full_text[len(prompt_text) :]
        if suffix:
            return suffix
    completion = example.get("completion")
    if isinstance(completion, str) and completion:
        return completion
    messages = example.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return str(message.get("content") or "")
    return full_text


def _tokenize_completion_only_row(
    *,
    tokenizer: Any,
    prompt_text: str,
    completion_text: str,
    full_text: str,
    max_seq_length: int,
) -> dict[str, list[int]]:
    prompt_ids = _tokenize_text(tokenizer, prompt_text)
    completion_ids = _tokenize_text(tokenizer, completion_text)
    if not completion_ids:
        fallback = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=True,
        )
        input_ids = list(fallback.get("input_ids") or [])
        if not input_ids:
            raise ValueError("SFT tokenization produced no input_ids.")
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": list(input_ids),
        }

    if len(completion_ids) >= max_seq_length:
        input_ids = completion_ids[:max_seq_length]
        labels = list(input_ids)
    else:
        prompt_budget = max_seq_length - len(completion_ids)
        prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []
        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + list(completion_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def _tokenize_text(tokenizer: Any, text: str) -> list[int]:
    if not text:
        return []
    encoded = tokenizer(text, add_special_tokens=False)
    return list(encoded.get("input_ids") or [])


def _build_checked_causal_lm_trainer(base_trainer_cls: Any) -> Any:
    class CheckedCausalLMTrainer(base_trainer_cls):  # type: ignore[misc, valid-type]
        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any | None = None,
        ) -> Any:
            del num_items_in_batch
            labels = inputs.get("labels")
            if labels is None:
                return super().compute_loss(model, inputs, return_outputs=return_outputs)
            model_inputs = dict(inputs)
            labels = model_inputs.pop("labels")
            outputs = model(**model_inputs)
            logits = _extract_logits(outputs)
            loss = _checked_shifted_causal_lm_loss(logits, labels)
            if not getattr(self, "_a2ui_checked_loss_logged", False):
                print(
                    "Checked causal-LM loss active: "
                    f"logits_shape={tuple(logits.shape)}, "
                    f"labels_shape={tuple(labels.shape)}, "
                    f"label_range={_tensor_label_range(labels)}",
                    flush=True,
                )
                setattr(self, "_a2ui_checked_loss_logged", True)
            if return_outputs:
                return loss, outputs
            return loss

    return CheckedCausalLMTrainer


def _run_forward_smoke_check(
    *,
    model: Any,
    split: Any,
    tokenizer: Any,
    input_vocab_size: int,
    label_vocab_size: int,
    max_position_embeddings: int | None,
) -> None:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - dependency failure path
        print(f"Skipping forward smoke check because torch import failed: {exc!r}", flush=True)
        return

    if len(split) <= 0:
        raise ValueError("Cannot run SFT forward smoke check: train split is empty.")
    row = split[0]
    features = [
        {
            "input_ids": row.get("input_ids") or [],
            "attention_mask": row.get("attention_mask") or [],
            "labels": row.get("labels") or [],
        }
    ]
    collator = _CausalLMDataCollator(
        tokenizer,
        input_vocab_size=input_vocab_size,
        label_vocab_size=label_vocab_size,
        max_position_embeddings=max_position_embeddings,
    )
    batch = collator(features)
    device = _model_input_device(model)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    labels = batch["labels"]
    print(
        "SFT forward smoke check: "
        f"input_shape={tuple(input_ids.shape)}, "
        f"attention_shape={tuple(attention_mask.shape) if attention_mask is not None else 'none'}, "
        f"input_range={_tensor_int_range(input_ids)}, "
        f"label_range={_tensor_label_range(labels)}, "
        f"input_vocab_size={input_vocab_size}, "
        f"label_vocab_size={label_vocab_size}, "
        f"model_device={device}, "
        f"hf_device_map={_summarize_device_map(getattr(model, 'hf_device_map', None))}",
        flush=True,
    )
    try:
        model.eval()
        with torch.no_grad():
            model_inputs = {"input_ids": input_ids}
            if attention_mask is not None:
                model_inputs["attention_mask"] = attention_mask
            outputs = model(**model_inputs)
            logits = _extract_logits(outputs)
            _validate_labels_against_logits_vocab(labels.to(logits.device), int(logits.shape[-1]))
            print(
                "SFT forward smoke check passed: "
                f"logits_shape={tuple(logits.shape)}, logits_device={logits.device}, "
                f"logits_vocab_size={int(logits.shape[-1])}",
                flush=True,
            )
    except Exception as exc:
        raise RuntimeError(
            "SFT forward smoke check failed before training. "
            "Token preflight passed, so this is likely a model-forward/CUDA environment issue, "
            "not a dataset tokenization issue. Compare torch/transformers/bitsandbytes/CUDA versions "
            "with the PC where the same code works; also verify the Hugging Face model cache is not stale. "
            f"Diagnostics: input_shape={tuple(input_ids.shape)}, input_range={_tensor_int_range(input_ids)}, "
            f"label_range={_tensor_label_range(labels)}, input_vocab_size={input_vocab_size}, "
            f"label_vocab_size={label_vocab_size}, model_device={device}, "
            f"hf_device_map={_summarize_device_map(getattr(model, 'hf_device_map', None))}, "
            f"error={exc!r}"
        ) from exc


def _model_input_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except Exception:
        try:
            import torch

            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        except Exception:
            return "cpu"


def _tensor_int_range(tensor: Any) -> str:
    try:
        if getattr(tensor, "numel", lambda: 0)() == 0:
            return "empty"
        return f"min={int(tensor.min().item())}, max={int(tensor.max().item())}, count={int(tensor.numel())}"
    except Exception as exc:
        return f"unavailable:{exc!r}"


def _summarize_device_map(device_map: Any) -> str:
    if not isinstance(device_map, dict):
        return "none"
    counts: dict[str, int] = {}
    for value in device_map.values():
        label = str(value)
        counts[label] = counts.get(label, 0) + 1
    return ",".join(f"{device}:{count}" for device, count in sorted(counts.items()))


def _extract_logits(outputs: Any) -> Any:
    if isinstance(outputs, dict):
        logits = outputs.get("logits")
    else:
        logits = getattr(outputs, "logits", None)
    if logits is None:
        raise ValueError("Model forward did not return logits; cannot compute checked causal-LM loss.")
    return logits


def _checked_shifted_causal_lm_loss(logits: Any, labels: Any) -> Any:
    import torch.nn.functional as F

    if getattr(logits, "dim", lambda: 0)() != 3:
        raise ValueError(f"Expected causal-LM logits with shape [batch, seq, vocab], got {tuple(logits.shape)}")
    if getattr(labels, "dim", lambda: 0)() != 2:
        raise ValueError(f"Expected causal-LM labels with shape [batch, seq], got {tuple(labels.shape)}")
    if logits.shape[:2] != labels.shape[:2]:
        raise ValueError(f"Logits/labels shape mismatch: logits={tuple(logits.shape)} labels={tuple(labels.shape)}")
    if logits.shape[1] < 2:
        raise ValueError(f"Sequence length is too short for shifted causal-LM loss: logits={tuple(logits.shape)}")

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    logits_vocab_size = int(shift_logits.shape[-1])
    _validate_labels_against_logits_vocab(shift_labels, logits_vocab_size)
    return F.cross_entropy(
        shift_logits.view(-1, logits_vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def _validate_labels_against_logits_vocab(labels: Any, logits_vocab_size: int) -> None:
    trainable = labels[labels != -100]
    if getattr(trainable, "numel", lambda: 0)() == 0:
        raise ValueError("Checked causal-LM loss found zero trainable labels after shift.")
    min_label = int(trainable.min().item())
    max_label = int(trainable.max().item())
    if min_label < 0 or max_label >= logits_vocab_size:
        raise ValueError(
            "Checked causal-LM loss blocked CUDA cross-entropy because labels exceed logits vocabulary. "
            f"logits_vocab_size={logits_vocab_size}, min_label={min_label}, max_label={max_label}, "
            f"label_range={_tensor_label_range(labels)}. "
            "This usually means the tokenizer has tokens not represented by the model output head."
        )


def _tensor_label_range(labels: Any) -> str:
    try:
        trainable = labels[labels != -100]
        if getattr(trainable, "numel", lambda: 0)() == 0:
            return "no_trainable_labels"
        return f"min={int(trainable.min().item())}, max={int(trainable.max().item())}, trainable={int(trainable.numel())}"
    except Exception as exc:
        return f"unavailable:{exc!r}"


class _CausalLMDataCollator:
    def __init__(
        self,
        tokenizer: Any,
        vocab_size: int | None = None,
        input_vocab_size: int | None = None,
        label_vocab_size: int | None = None,
        max_position_embeddings: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.input_vocab_size = input_vocab_size if input_vocab_size is not None else vocab_size
        self.label_vocab_size = label_vocab_size if label_vocab_size is not None else vocab_size
        self.max_position_embeddings = max_position_embeddings

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        input_features = [
            {key: value for key, value in feature.items() if key in {"input_ids", "attention_mask"}}
            for feature in features
        ]
        batch = self.tokenizer.pad(input_features, padding=True, return_tensors="pt")
        _validate_padded_input_id_tensor(batch["input_ids"], self.input_vocab_size, self.tokenizer)
        _validate_padded_sequence_length(batch["input_ids"], self.max_position_embeddings)
        provided_labels = [feature.get("labels") for feature in features]
        if all(labels is not None for labels in provided_labels):
            labels = torch.full_like(batch["input_ids"], -100)
            for row_index, row_labels in enumerate(provided_labels):
                values = torch.tensor(list(row_labels), dtype=labels.dtype)
                length = min(values.numel(), labels.shape[1])
                labels[row_index, :length] = values[:length]
        else:
            labels = batch["input_ids"].clone()
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                labels = labels.masked_fill(attention_mask == 0, -100)
            else:
                pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
                if pad_token_id is not None:
                    labels = labels.masked_fill(batch["input_ids"] == pad_token_id, -100)
        _validate_trainable_label_tensor(labels, self.label_vocab_size)
        batch["labels"] = labels
        return batch


def _align_tokenizer_and_model(tokenizer: Any, model: Any) -> None:
    token_count = len(tokenizer) if hasattr(tokenizer, "__len__") else None
    input_vocab_size = _model_vocab_size(model)
    output_vocab_size = _model_output_vocab_size(model)
    min_vocab_size = _min_known_vocab_size(input_vocab_size, output_vocab_size)
    if token_count is not None and min_vocab_size is not None and token_count > min_vocab_size:
        model.resize_token_embeddings(token_count)
        input_vocab_size = _model_vocab_size(model)
        output_vocab_size = _model_output_vocab_size(model)
    model_vocab_size = _min_known_vocab_size(input_vocab_size, output_vocab_size)
    _sync_model_config_vocab_size(model, output_vocab_size or input_vocab_size)
    _ensure_safe_pad_token(tokenizer, model_vocab_size)
    for attr in ("pad_token_id", "bos_token_id", "eos_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is None:
            continue
        if hasattr(model, "config"):
            setattr(model.config, attr, value)
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            setattr(generation_config, attr, value)


def _assert_tokenizer_model_vocab_alignment(tokenizer: Any, model: Any, *, context: str) -> None:
    token_count = len(tokenizer) if hasattr(tokenizer, "__len__") else None
    input_vocab_size = _require_model_input_vocab_size(model)
    output_vocab_size = _model_output_vocab_size(model)
    min_vocab_size = _min_known_vocab_size(input_vocab_size, output_vocab_size)
    if token_count is not None and min_vocab_size is not None and token_count > min_vocab_size:
        resizer = getattr(model, "resize_token_embeddings", None)
        if callable(resizer):
            resizer(token_count)
            input_vocab_size = _require_model_input_vocab_size(model)
            output_vocab_size = _model_output_vocab_size(model)
            min_vocab_size = _min_known_vocab_size(input_vocab_size, output_vocab_size)
    if token_count is not None and min_vocab_size is not None and token_count > min_vocab_size:
        raise ValueError(
            "Tokenizer/model vocabulary mismatch after attempted resize "
            f"({context}): tokenizer_size={token_count}, "
            f"input_vocab_size={input_vocab_size}, output_vocab_size={output_vocab_size}. "
            "Training would fail in the embedding or loss layer with a CUDA device-side assert."
        )
    _sync_model_config_vocab_size(model, output_vocab_size or input_vocab_size)
    _ensure_safe_pad_token(tokenizer, min_vocab_size)


def _require_model_input_vocab_size(model: Any) -> int:
    vocab_size = _model_vocab_size(model)
    if vocab_size is None:
        raise ValueError(
            "Unable to determine model input embedding vocabulary size. "
            "Refusing to start CUDA training because token IDs cannot be preflight-checked."
        )
    return vocab_size


def _require_model_label_vocab_size(model: Any) -> int:
    output_vocab_size = _model_output_vocab_size(model)
    if output_vocab_size is not None:
        return output_vocab_size
    return _require_model_input_vocab_size(model)


def _ensure_safe_pad_token(tokenizer: Any, vocab_size: int | None) -> None:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if _is_valid_token_id(pad_token_id, vocab_size):
        return

    for token_attr, id_attr in (
        ("eos_token", "eos_token_id"),
        ("bos_token", "bos_token_id"),
        ("unk_token", "unk_token_id"),
    ):
        candidate_id = getattr(tokenizer, id_attr, None)
        candidate_token = getattr(tokenizer, token_attr, None)
        if not _is_valid_token_id(candidate_id, vocab_size):
            continue
        if candidate_token is not None:
            try:
                tokenizer.pad_token = candidate_token
            except Exception:
                pass
        if not _is_valid_token_id(getattr(tokenizer, "pad_token_id", None), vocab_size):
            try:
                tokenizer.pad_token_id = candidate_id
            except Exception:
                pass
        if _is_valid_token_id(getattr(tokenizer, "pad_token_id", None), vocab_size):
            print(
                "Adjusted tokenizer pad token for safe training: "
                f"pad_token_id={getattr(tokenizer, 'pad_token_id', None)}",
                flush=True,
            )
            return

    if _is_valid_token_id(0, vocab_size):
        try:
            tokenizer.pad_token_id = 0
        except Exception:
            pass
        if _is_valid_token_id(getattr(tokenizer, "pad_token_id", None), vocab_size):
            print("Adjusted tokenizer pad token for safe training: pad_token_id=0", flush=True)
            return

    raise ValueError(
        "Tokenizer pad_token_id is missing or outside model vocabulary and no safe fallback token is available. "
        f"pad_token_id={pad_token_id} vocab_size={vocab_size}"
    )


def _is_valid_token_id(value: Any, vocab_size: int | None) -> bool:
    if not isinstance(value, int):
        return False
    if value < 0:
        return False
    return vocab_size is None or value < vocab_size


def _validate_padded_input_id_tensor(input_ids: Any, vocab_size: int | None, tokenizer: Any) -> None:
    if vocab_size is None:
        return
    if getattr(input_ids, "numel", lambda: 0)() == 0:
        raise ValueError("Padded SFT batch contains no input_ids.")
    min_token_id = int(input_ids.min().item())
    max_token_id = int(input_ids.max().item())
    if min_token_id < 0 or max_token_id >= vocab_size:
        raise ValueError(
            "Padded SFT batch has token IDs outside the model vocabulary before CUDA execution. "
            f"min_token_id={min_token_id} max_token_id={max_token_id} vocab_size={vocab_size} "
            f"pad_token_id={getattr(tokenizer, 'pad_token_id', None)}"
        )


def _validate_padded_sequence_length(input_ids: Any, max_position_embeddings: int | None) -> None:
    if max_position_embeddings is None:
        return
    shape = getattr(input_ids, "shape", None)
    if not shape:
        return
    sequence_length = int(shape[-1])
    if sequence_length > max_position_embeddings:
        raise ValueError(
            "Padded SFT batch sequence length exceeds model position limit before CUDA execution. "
            f"sequence_length={sequence_length} max_position_embeddings={max_position_embeddings}"
        )


def _validate_trainable_label_tensor(labels: Any, vocab_size: int | None) -> None:
    if getattr(labels, "numel", lambda: 0)() == 0:
        raise ValueError("SFT batch contains no labels.")
    trainable = int((labels != -100).sum().item())
    if trainable <= 0:
        raise ValueError(
            "SFT batch has zero trainable labels after prompt masking. "
            "Increase training.max_seq_length or reduce prompt size so completion tokens remain in the batch."
        )
    if vocab_size is not None:
        trainable_labels = labels[labels != -100]
        if getattr(trainable_labels, "numel", lambda: 0)() > 0:
            min_label_id = int(trainable_labels.min().item())
            max_label_id = int(trainable_labels.max().item())
            if min_label_id < 0 or max_label_id >= vocab_size:
                raise ValueError(
                    "SFT batch has trainable label IDs outside the model vocabulary before CUDA execution. "
                    f"min_label_id={min_label_id} max_label_id={max_label_id} vocab_size={vocab_size}"
                )


def _print_trainable_parameter_summary(model: Any) -> None:
    trainable = 0
    total = 0
    for parameter in getattr(model, "parameters", lambda: [])():
        count = int(parameter.numel()) if hasattr(parameter, "numel") else 0
        total += count
        if bool(getattr(parameter, "requires_grad", False)):
            trainable += count
    if trainable <= 0:
        raise ValueError(
            "SFT model has zero trainable parameters after applying LoRA. "
            "Check lora.target_modules against the loaded model module names."
        )
    pct = (trainable / total * 100.0) if total else 0.0
    print(
        "Trainable parameter summary: "
        f"trainable={trainable}, total={total}, trainable_pct={pct:.4f}",
        flush=True,
    )


def _print_tokenizer_model_alignment(tokenizer: Any, model: Any) -> None:
    input_vocab_size = _model_vocab_size(model)
    output_vocab_size = _model_output_vocab_size(model)
    position_limit = _model_position_limit(model)
    print(
        "Tokenizer/model alignment: "
        f"tokenizer_size={len(tokenizer) if hasattr(tokenizer, '__len__') else 'unknown'}, "
        f"input_vocab_size={input_vocab_size if input_vocab_size is not None else 'unknown'}, "
        f"output_vocab_size={output_vocab_size if output_vocab_size is not None else 'unknown'}, "
        f"config_vocab_size={_model_config_vocab_size(model) or 'unknown'}, "
        f"max_position_embeddings={position_limit if position_limit is not None else 'unknown'}, "
        f"pad_token_id={getattr(tokenizer, 'pad_token_id', None)}, "
        f"bos_token_id={getattr(tokenizer, 'bos_token_id', None)}, "
        f"eos_token_id={getattr(tokenizer, 'eos_token_id', None)}",
        flush=True,
    )


def _effective_max_seq_length(configured: int, max_position_embeddings: int | None) -> int:
    if max_position_embeddings is None or configured <= max_position_embeddings:
        print(
            "Effective SFT max sequence length: "
            f"{configured} (configured={configured}, model_position_limit={max_position_embeddings or 'unknown'})",
            flush=True,
        )
        return configured
    print(
        "Clamping SFT max sequence length to model position limit: "
        f"configured={configured}, model_position_limit={max_position_embeddings}",
        flush=True,
    )
    return max_position_embeddings


def _enforce_cuda_requirement(model_cfg: dict[str, Any], training_cfg: dict[str, Any]) -> None:
    allow_cpu = bool(model_cfg.get("allow_cpu", False) or training_cfg.get("allow_cpu", False))
    if allow_cpu:
        return
    if _cuda_available():
        return
    raise RuntimeError(
        "CUDA is not available to PyTorch, so SFT training would run on CPU and be extremely slow. "
        f"{_cuda_diagnostic_summary()} "
        "Fix the launch environment before training: use the CUDA-enabled torch wheel, run Singularity "
        "with GPU passthrough (`singularity exec --nv ...`), and submit Slurm jobs with a GPU allocation "
        "through `training/scripts/slurm_train_gemma4_e2b.sbatch` or equivalent `srun --gres=gpu:1`. "
        "If you intentionally want a CPU smoke run, set `training.allow_cpu: true` or `model.allow_cpu: true`."
    )


def _resolve_training_dtype(requested_dtype: str) -> str:
    dtype = (requested_dtype or "bfloat16").strip().lower()
    if dtype in {"bf16", "bfloat16"}:
        if _cuda_bf16_supported():
            print("Training precision: bfloat16", flush=True)
            return "bfloat16"
        if _cuda_available():
            print(
                "Training precision fallback: requested bfloat16, but this GPU/PyTorch setup "
                "does not support bf16. Using float16 instead.",
                flush=True,
            )
            return "float16"
        print(
            "Training precision fallback: requested bfloat16, but CUDA is unavailable. "
            "Using float32 so TrainingArguments does not fail before reporting the real device issue.",
            flush=True,
        )
        return "float32"
    if dtype in {"fp16", "float16", "half"}:
        if _cuda_available():
            print("Training precision: float16", flush=True)
            return "float16"
        print(
            "Training precision fallback: requested float16, but CUDA is unavailable. Using float32.",
            flush=True,
        )
        return "float32"
    if dtype in {"fp32", "float32", "full"}:
        print("Training precision: float32", flush=True)
        return "float32"
    print(f"Training precision: unknown dtype {requested_dtype!r}; using float32.", flush=True)
    return "float32"


def _training_precision_flags(dtype_name: str) -> dict[str, bool]:
    dtype = dtype_name.strip().lower()
    return {
        "bf16": dtype == "bfloat16",
        "fp16": dtype == "float16",
    }


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _cuda_bf16_supported() -> bool:
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return False
        checker = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(checker):
            return bool(checker())
        major, _minor = torch.cuda.get_device_capability(0)
        return int(major) >= 8
    except Exception:
        return False


def _cuda_diagnostic_summary() -> str:
    try:
        import os
        import torch  # type: ignore

        return (
            f"torch={getattr(torch, '__version__', 'unknown')}, "
            f"torch_cuda={getattr(getattr(torch, 'version', None), 'cuda', None)}, "
            f"cuda_available={torch.cuda.is_available()}, "
            f"device_count={torch.cuda.device_count() if hasattr(torch, 'cuda') else 'unknown'}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}."
        )
    except Exception as exc:
        return f"Unable to collect torch CUDA diagnostics: {exc!r}."


def _model_vocab_size(model: Any, seen: set[int] | None = None) -> int | None:
    if model is None:
        return None
    if seen is None:
        seen = set()
    model_id = id(model)
    if model_id in seen:
        return None
    seen.add(model_id)

    embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    vocab_size = _embedding_vocab_size(embeddings)
    if vocab_size is not None:
        return vocab_size
    for nested_attr in ("base_model", "model", "language_model", "module", "wrapped_model"):
        nested = _safe_getattr(model, nested_attr)
        nested_vocab_size = _model_vocab_size(nested, seen)
        if nested_vocab_size is not None:
            return nested_vocab_size
    return None


def _model_output_vocab_size(model: Any, seen: set[int] | None = None) -> int | None:
    if model is None:
        return None
    if seen is None:
        seen = set()
    model_id = id(model)
    if model_id in seen:
        return None
    seen.add(model_id)

    output_embeddings = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    output_vocab_size = _embedding_vocab_size(output_embeddings)
    if output_vocab_size is not None:
        return output_vocab_size
    lm_head = _safe_getattr(model, "lm_head")
    lm_head_vocab_size = _linear_output_vocab_size(lm_head)
    if lm_head_vocab_size is not None:
        return lm_head_vocab_size
    for nested_attr in ("base_model", "model", "language_model", "module", "wrapped_model"):
        nested = _safe_getattr(model, nested_attr)
        nested_vocab_size = _model_output_vocab_size(nested, seen)
        if nested_vocab_size is not None:
            return nested_vocab_size
    return None


def _linear_output_vocab_size(value: Any) -> int | None:
    if value is None:
        return None
    weight = _safe_getattr(value, "weight")
    shape = _safe_getattr(weight, "shape")
    if shape is not None:
        try:
            if len(shape) >= 1:
                return int(shape[0])
        except Exception:
            pass
    out_features = _safe_getattr(value, "out_features")
    if isinstance(out_features, int) and out_features > 0:
        return out_features
    return None


def _model_config_vocab_size(model: Any, seen: set[int] | None = None) -> int | None:
    if model is None:
        return None
    if seen is None:
        seen = set()
    model_id = id(model)
    if model_id in seen:
        return None
    seen.add(model_id)
    config = _safe_getattr(model, "config")
    vocab_size = _safe_getattr(config, "vocab_size")
    if isinstance(vocab_size, int) and vocab_size > 0:
        return vocab_size
    for nested_attr in ("base_model", "model", "language_model", "module", "wrapped_model"):
        nested = _safe_getattr(model, nested_attr)
        nested_vocab_size = _model_config_vocab_size(nested, seen)
        if nested_vocab_size is not None:
            return nested_vocab_size
    return None


def _sync_model_config_vocab_size(model: Any, vocab_size: int | None, seen: set[int] | None = None) -> None:
    if model is None or vocab_size is None:
        return
    if seen is None:
        seen = set()
    model_id = id(model)
    if model_id in seen:
        return
    seen.add(model_id)
    config = _safe_getattr(model, "config")
    if config is not None:
        _safe_setattr(config, "vocab_size", vocab_size)
    generation_config = _safe_getattr(model, "generation_config")
    if generation_config is not None:
        _safe_setattr(generation_config, "vocab_size", vocab_size)
    for nested_attr in ("base_model", "model", "language_model", "module", "wrapped_model"):
        nested = _safe_getattr(model, nested_attr)
        _sync_model_config_vocab_size(nested, vocab_size, seen)


def _min_known_vocab_size(*values: int | None) -> int | None:
    known = [value for value in values if isinstance(value, int) and value > 0]
    return min(known) if known else None


def _embedding_vocab_size(embeddings: Any) -> int | None:
    if embeddings is None:
        return None
    weight = _safe_getattr(embeddings, "weight")
    shape = _safe_getattr(weight, "shape")
    if shape is not None:
        try:
            if len(shape) >= 1:
                return int(shape[0])
        except Exception:
            pass
    vocab_size = _safe_getattr(embeddings, "num_embeddings")
    if isinstance(vocab_size, int) and vocab_size > 0:
        return vocab_size
    return None


def _model_position_limit(model: Any, seen: set[int] | None = None) -> int | None:
    if model is None:
        return None
    if seen is None:
        seen = set()
    model_id = id(model)
    if model_id in seen:
        return None
    seen.add(model_id)

    config = _safe_getattr(model, "config")
    candidates: list[int] = []
    _collect_position_limit_candidates(config, candidates)
    for nested_attr in ("base_model", "model", "language_model", "module", "wrapped_model"):
        nested = _safe_getattr(model, nested_attr)
        nested_limit = _model_position_limit(nested, seen)
        if nested_limit is not None:
            candidates.append(nested_limit)
    return min(candidates) if candidates else None


def _collect_position_limit_candidates(
    value: Any,
    candidates: list[int],
    seen: set[int] | None = None,
    depth: int = 0,
) -> None:
    if value is None:
        return
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen or depth > 4:
        return
    seen.add(value_id)

    mapping = _safe_mapping(value)
    for attr in ("max_position_embeddings", "max_sequence_length", "seq_length"):
        attr_value = mapping.get(attr) if attr in mapping else _safe_getattr(value, attr)
        if isinstance(attr_value, int) and attr_value > 0:
            candidates.append(attr_value)
    for nested_attr in ("text_config", "llm_config", "language_config"):
        nested = mapping.get(nested_attr) if nested_attr in mapping else _safe_getattr(value, nested_attr)
        if nested is not None and nested is not value:
            _collect_position_limit_candidates(nested, candidates, seen, depth + 1)


def _safe_getattr(value: Any, attr: str, default: Any = None) -> Any:
    try:
        return getattr(value, attr, default)
    except RecursionError:
        return default
    except Exception:
        return default


def _safe_setattr(value: Any, attr: str, attr_value: Any) -> None:
    try:
        setattr(value, attr, attr_value)
    except Exception:
        pass


def _safe_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        mapping = object.__getattribute__(value, "__dict__")
        return mapping if isinstance(mapping, dict) else {}
    except Exception:
        return {}


def _validate_sft_token_ids(
    *,
    dataset: Any,
    tokenizer: Any,
    max_seq_length: int,
    max_rows: int,
    vocab_size: int | None = None,
    input_vocab_size: int | None = None,
    label_vocab_size: int | None = None,
    tokenizer_size: int | None = None,
) -> None:
    input_limit = input_vocab_size if input_vocab_size is not None else vocab_size
    label_limit = label_vocab_size if label_vocab_size is not None else vocab_size
    if input_limit is None or label_limit is None:
        print("SFT token preflight skipped: model vocabulary size is unavailable.", flush=True)
        return
    split_names = [name for name in ("train", "validation") if name in dataset]
    total_rows_checked = 0
    max_seen_token_id = -1
    for split_name in split_names:
        split = dataset[split_name]
        limit = len(split) if max_rows <= 0 else min(len(split), max_rows)
        for row_index in range(limit):
            total_rows_checked += 1
            text = str(split[row_index].get("text", ""))
            completion_text = str(split[row_index].get("completion_text", ""))
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
                if isinstance(token_id, int):
                    max_seen_token_id = max(max_seen_token_id, token_id)
                if not isinstance(token_id, int) or token_id < 0 or token_id >= input_limit:
                    raise ValueError(
                        "SFT preflight failed: token id outside model vocabulary (input) "
                        f"at {split_name}[{row_index}] token={token_id} input_vocab_size={input_limit}. "
                        "Check tokenizer/model pairing before running CUDA training."
                    )
            completion_ids = _tokenize_text(tokenizer, completion_text) if completion_text else input_ids
            for label_id in completion_ids:
                if not isinstance(label_id, int) or label_id < 0 or label_id >= label_limit:
                    raise ValueError(
                        "SFT preflight failed: completion label id outside model vocabulary (output) "
                        f"at {split_name}[{row_index}] label={label_id} label_vocab_size={label_limit}. "
                        "The output head/config must be resized before CUDA training."
                    )
    print(
        "SFT token preflight passed: "
        f"rows_checked={total_rows_checked}, "
        f"tokenizer_size={tokenizer_size if tokenizer_size is not None else 'unknown'}, "
        f"input_vocab_size={input_limit}, "
        f"label_vocab_size={label_limit}, "
        f"max_token_id={max_seen_token_id}, "
        f"max_seq_length={max_seq_length}",
        flush=True,
    )


def _validate_tokenized_sft_dataset(
    *,
    dataset: Any,
    max_rows: int,
    vocab_size: int | None = None,
    input_vocab_size: int | None = None,
    label_vocab_size: int | None = None,
) -> None:
    input_limit = input_vocab_size if input_vocab_size is not None else vocab_size
    label_limit = label_vocab_size if label_vocab_size is not None else vocab_size
    if input_limit is None or label_limit is None:
        print("Tokenized SFT preflight skipped: model vocabulary size is unavailable.", flush=True)
        return
    split_names = list(dataset.keys()) if hasattr(dataset, "keys") else []
    total_rows_checked = 0
    max_seen_token_id = -1
    for split_name in split_names:
        split = dataset[split_name]
        limit = len(split) if max_rows <= 0 else min(len(split), max_rows)
        for row_index in range(limit):
            total_rows_checked += 1
            input_ids = split[row_index].get("input_ids") or []
            if not input_ids:
                raise ValueError(f"Tokenized SFT preflight failed: empty input_ids at {split_name}[{row_index}]")
            labels = split[row_index].get("labels") or []
            trainable_label_count = sum(1 for label in labels if isinstance(label, int) and label != -100)
            if trainable_label_count <= 0:
                raise ValueError(
                    "Tokenized SFT preflight failed: row has zero trainable labels after prompt masking "
                    f"at {split_name}[{row_index}]."
                )
            for label_id in labels:
                if label_id == -100:
                    continue
                if not isinstance(label_id, int) or label_id < 0 or label_id >= label_limit:
                    raise ValueError(
                        "Tokenized SFT preflight failed: trainable label id outside model vocabulary (output) "
                        f"at {split_name}[{row_index}] label={label_id} label_vocab_size={label_limit}."
                    )
            for token_id in input_ids:
                if isinstance(token_id, int):
                    max_seen_token_id = max(max_seen_token_id, token_id)
                if not isinstance(token_id, int) or token_id < 0 or token_id >= input_limit:
                    raise ValueError(
                        "Tokenized SFT preflight failed: token id outside model vocabulary (input) "
                        f"at {split_name}[{row_index}] token={token_id} input_vocab_size={input_limit}."
                    )
    print(
        "Tokenized SFT preflight passed: "
        f"rows_checked={total_rows_checked}, "
        f"input_vocab_size={input_limit}, "
        f"label_vocab_size={label_limit}, "
        f"max_token_id={max_seen_token_id}",
        flush=True,
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
