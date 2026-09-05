"""Static training recipe checks. These never load a tokenizer or model."""
from __future__ import annotations

import math
from typing import Any

LORA_METHODS = {"lora_sft", "sft_lora", "qat_lora_sft"}
FULL_METHODS = {"full_finetune_sft", "full_finetune_qat"}


def validate_sft_recipe(config: dict[str, Any]) -> str:
    training = config.get("training") or {}
    method = str(training.get("method", "lora_sft")).strip().lower()
    if method not in LORA_METHODS | FULL_METHODS:
        raise ValueError(f"Unsupported SFT training.method={method!r}")
    qat = config.get("qat") or {}
    golden = config.get("golden_eval") or {}
    if golden.get("enabled") and golden.get("metric_version", "legacy") not in {"legacy", "v5_4", "dual"}:
        raise ValueError("golden_eval.metric_version must be legacy, v5_4, or dual.")
    if method in {"qat_lora_sft", "full_finetune_qat"} and qat.get("enabled") is not True:
        raise ValueError(f"{method} requires qat.enabled: true")
    if method in FULL_METHODS:
        if config.get("lora"):
            raise ValueError("Full finetuning must omit lora settings; all model parameters are trained.")
        if (config.get("model") or {}).get("load_in_4bit"):
            raise ValueError("Full finetuning does not support a frozen 4-bit base model.")
        if qat.get("scale_mode") == "retained_mobile" or config.get("qat_mtp"):
            raise ValueError("Full finetuning is not supported by the retained mobile LoRA/MTP contract.")
    resume = training.get("resume_from_checkpoint") or config.get("resume_from_checkpoint")
    if resume and training.get("refuse_resume"):
        raise ValueError("training.refuse_resume forbids resume_from_checkpoint for this recipe.")
    if str(training.get("trainer_backend", "hf")).lower() != "hf":
        raise ValueError("Completion-masked SFT requires training.trainer_backend: hf.")
    if training.get("packing"):
        raise ValueError("Packing is unsupported by the checked completion-masked HF SFT path.")
    if training.get("deepspeed") or training.get("fsdp"):
        raise ValueError("Use the reviewed DDP path; DeepSpeed/FSDP require separate checkpoint and QAT validation.")
    if str(training.get("overflow_policy", "error")) != "error":
        raise ValueError("SFT overflow_policy must be error; quarantine overlength rows during preparation.")
    return method


def resolve_eval_strategy(training: dict[str, Any], golden: dict[str, Any], *, has_validation: bool) -> str:
    configured = training.get("eval_strategy", training.get("evaluation_strategy", "steps" if has_validation else "no"))
    # PyYAML's YAML 1.1 reader parses an unquoted `no` as False.
    strategy = "no" if configured is False else str(configured).lower()
    if strategy not in {"no", "steps", "epoch"}:
        raise ValueError(f"Unsupported evaluation strategy: {strategy!r}")
    if strategy != "no" and not has_validation:
        raise ValueError("Evaluation requires a nonempty val.jsonl split.")
    if golden.get("enabled") and str(golden.get("trigger", "epoch")) == "evaluate" and strategy == "no":
        raise ValueError("Golden trigger=evaluate requires enabled HF evaluation, or use trigger=epoch.")
    return strategy


def optimizer_steps(*, rows: int, world_size: int, microbatch: int, accumulation: int, epochs: float) -> int:
    if min(rows, world_size, microbatch, accumulation) <= 0 or epochs <= 0:
        raise ValueError("Dataset, world size, batch, accumulation and epochs must be positive.")
    batches = math.ceil(math.ceil(rows / world_size) / microbatch)
    return math.ceil(math.ceil(batches / accumulation) * epochs)


def validate_effective_batch(training: dict[str, Any], world_size: int) -> int:
    microbatch = int(training.get("per_device_train_batch_size", 1))
    accumulation = int(training.get("gradient_accumulation_steps", 16))
    if min(world_size, microbatch, accumulation) <= 0:
        raise ValueError("GPU count, microbatch and accumulation must be positive.")
    actual = world_size * microbatch * accumulation
    expected = training.get("expected_effective_batch_size")
    if expected is not None and int(expected) != actual:
        raise ValueError(f"Effective batch mismatch: expected {expected}, actual {world_size} * {microbatch} * {accumulation} = {actual}")
    return actual
