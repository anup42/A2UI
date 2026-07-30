#!/usr/bin/env python3
"""Reference GRPO + LoRA entry point for text -> GenUI FlatSpec.

Dataset columns
---------------
Required:
  response_text: upstream LLM text that the completion must represent as UI.
Recommended:
  expected_ui_contract: persisted source-side contract, independent of candidate.
  intent_bucket, assets, query_id/source_id/response_id.
Optional:
  genui_json: accepted reference output used only to estimate completion length.

The reward never judges the factual or writing quality of response_text.  It
scores the completion as a renderer-facing representation of that source.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_SRC = REPO_ROOT / "dataset" / "src"
if str(DATASET_SRC) not in sys.path:
    sys.path.insert(0, str(DATASET_SRC))

from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, set_seed
import trl
from trl import GRPOConfig, GRPOTrainer

from pipeline.genui_quality import (
    RewardInflationMonitor,
    ensure_v5_4_validation_ready,
    load_reward_config_v5_4,
    make_genui_grpo_reward_v5_4,
    metric_fingerprint_v5_4,
    reward_pipeline_fingerprint_v5_4,
)


MIN_TRL_VERSION = Version("0.29.1")


def validate_trl_version() -> None:
    installed = Version(str(getattr(trl, "__version__", "0")))
    if installed < MIN_TRL_VERSION:
        raise RuntimeError(
            f"TRL >= {MIN_TRL_VERSION} is required for GRPO reward diagnostics; found {installed}."
        )


def _to_render_asset_path(path: str) -> str:
    value = str(path or "").replace("\\", "/").strip()
    if not value:
        return ""
    if value.startswith("../assets/"):
        return value
    value = value.lstrip("/").lstrip("./")
    return "../" + value if value.startswith("assets/") else value


def _asset_kind(path: str, url: str) -> str:
    suffix = Path((path or url).split("?", 1)[0]).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        return "image"
    if suffix == ".svg" or "bootstrap-icons" in url.lower() or "/icons/" in url.lower():
        return "icon"
    if suffix in {".mp4", ".webm"}:
        return "video"
    if suffix in {".mp3", ".wav", ".m4a"}:
        return "audio"
    return "asset"


def build_asset_context(assets: Any) -> str:
    if not isinstance(assets, Sequence) or isinstance(assets, (str, bytes, bytearray)):
        return ""
    lines: list[str] = []
    for item in assets:
        if not isinstance(item, Mapping):
            continue
        url = str(item.get("url") or "").strip()
        local_path = _to_render_asset_path(str(item.get("path") or ""))
        if not local_path:
            continue
        kind = _asset_kind(local_path, url)
        lines.append(f"- [{kind}] {url} -> {local_path}" if url else f"- [{kind}] {local_path}")
    if not lines:
        return ""
    return (
        "Assets (use only mapped local paths on the right; do not invent paths):\n"
        + "\n".join(lines)
    )


def build_prompt(template: str, response_text: str, assets: Any) -> str:
    if "{response_text}" not in template:
        raise ValueError("Prompt template must contain {response_text}")
    has_assets = bool(assets) and isinstance(assets, Sequence) and not isinstance(
        assets, (str, bytes, bytearray)
    )
    if has_assets:
        asset_policy = (
            "Asset policy:\n"
            "- Use only local media paths from the supplied mapping.\n"
            "- Do not emit remote URLs for mapped images or icons.\n"
            "- Do not invent local placeholder paths."
        )
    else:
        asset_policy = (
            "Asset policy:\n"
            "- No local asset mapping is supplied.\n"
            "- Preserve required source media URLs exactly.\n"
            "- Do not invent local placeholder paths."
        )
    source = f"{response_text}\n\n{asset_policy}"
    context = build_asset_context(assets)
    if context:
        source += f"\n\n{context}"
    return template.replace("{response_text}", source)


def _stable_source_id(row: Mapping[str, Any]) -> str:
    for key in ("source_id", "query_id", "response_id", "ui_id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    source = str(row.get("response_text") or "")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:20]


def _source_model_family(row: Mapping[str, Any]) -> str:
    explicit = row.get("source_model_family")
    if explicit:
        return str(explicit)
    generation = row.get("gen")
    if isinstance(generation, Mapping):
        value = generation.get("model") or generation.get("provider")
        if value:
            return str(value)
    return "unknown"


def _source_timestamp(row: Mapping[str, Any]) -> str:
    value = row.get("source_created_at") or row.get("created_at")
    return str(value or "")


def load_training_dataset(path: str, template: str) -> Dataset:
    if Path(path).suffix.lower() not in {".json", ".jsonl"}:
        raise ValueError("The reference loader supports .json and .jsonl")
    dataset = load_dataset("json", data_files=path, split="train")
    if "response_text" not in dataset.column_names:
        raise ValueError("Dataset is missing required column: response_text")

    def prepare(row: dict[str, Any]) -> dict[str, Any]:
        assets = row.get("assets") or []
        return {
            "source_id": _stable_source_id(row),
            "prompt": build_prompt(template, str(row["response_text"]), assets),
            "response_text": str(row["response_text"]),
            "intent_bucket": row.get("intent_bucket"),
            "assets": assets,
            "expected_ui_contract": row.get("expected_ui_contract"),
            "source_model_family": _source_model_family(row),
            "source_created_at": _source_timestamp(row),
            # Used only for generation-length estimation, never as reward truth.
            "genui_json": row.get("genui_json"),
        }

    return dataset.map(prepare, remove_columns=dataset.column_names)


def split_by_source_model_time(dataset: Dataset, eval_fraction: float, seed: int) -> DatasetDict:
    """Leak-free source split, stratified by model family with latest sources held out."""
    if not (0.0 < eval_fraction < 1.0) or len(dataset) < 20:
        return DatasetDict({"train": dataset})

    source_groups: dict[str, list[int]] = defaultdict(list)
    source_metadata: dict[str, tuple[str, str]] = {}
    for index, row in enumerate(dataset):
        source_id = str(row["source_id"])
        source_groups[source_id].append(index)
        source_metadata.setdefault(
            source_id,
            (str(row.get("source_model_family") or "unknown"), str(row.get("source_created_at") or "")),
        )

    by_family: dict[str, list[str]] = defaultdict(list)
    for source_id, (family, _) in source_metadata.items():
        by_family[family].append(source_id)

    eval_sources: set[str] = set()
    for family, source_ids in by_family.items():
        if len(source_ids) < 2:
            continue
        source_ids.sort(
            key=lambda source_id: (
                source_metadata[source_id][1],
                hashlib.sha256(f"{seed}:{family}:{source_id}".encode("utf-8")).hexdigest(),
            )
        )
        eval_count = max(1, min(len(source_ids) - 1, math.ceil(len(source_ids) * eval_fraction)))
        eval_sources.update(source_ids[-eval_count:])

    if not eval_sources:
        ordered = sorted(
            source_groups,
            key=lambda source_id: hashlib.sha256(f"{seed}:{source_id}".encode("utf-8")).hexdigest(),
        )
        eval_sources.update(ordered[-max(1, math.ceil(len(ordered) * eval_fraction)):])

    eval_indices = [index for source_id in eval_sources for index in source_groups[source_id]]
    train_indices = [index for source_id, indices in source_groups.items() if source_id not in eval_sources for index in indices]
    eval_ds = dataset.select(sorted(eval_indices))
    train_ds = dataset.select(sorted(train_indices))
    if len(eval_ds) == 0 or len(train_ds) == 0:
        raise ValueError("Source-level split produced an empty partition; adjust eval_fraction")
    overlap = set(train_ds["source_id"]) & set(eval_ds["source_id"])
    if overlap:
        raise AssertionError("Source leakage detected between train and evaluation sets")
    return DatasetDict({"train": train_ds, "test": eval_ds})


def validate_sft_checkpoint(checkpoint: str) -> None:
    """Reject an ambiguous local base-model directory before reward optimization."""
    path = Path(checkpoint)
    if not path.exists():
        # Hub checkpoints cannot be proven locally; the explicit CLI name records the contract.
        return
    if not path.is_dir():
        raise ValueError(f"SFT checkpoint must be a directory or Hub ID: {checkpoint}")
    markers = (
        "adapter_config.json",
        "trainer_state.json",
        "sft_manifest.json",
        "training_args.bin",
    )
    if not any((path / marker).exists() for marker in markers):
        raise ValueError(
            "Local --sft-checkpoint has no SFT/adapter marker "
            f"({', '.join(markers)}); refusing to start GRPO from an unverified base checkpoint."
        )


def _percentile(values: Sequence[int], p: float) -> int:
    if not values:
        raise ValueError("Cannot calculate a percentile from no values")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1))
    return ordered[index]


def estimate_completion_length(dataset: Dataset, tokenizer: Any) -> int | None:
    if "genui_json" not in dataset.column_names:
        return None
    lengths: list[int] = []
    for spec in dataset["genui_json"]:
        if isinstance(spec, Mapping):
            text = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
            lengths.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
    if not lengths:
        return None
    p99 = _percentile(lengths, 0.99)
    return max(512, int(math.ceil((p99 * 1.20) / 128.0) * 128))


def make_grpo_config(**kwargs: Any) -> GRPOConfig:
    """Fail clearly when an installed TRL version lacks required controls."""
    parameters = inspect.signature(GRPOConfig.__init__).parameters
    if "eval_strategy" not in parameters and "evaluation_strategy" in parameters:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy", "no")
    unsupported = sorted(key for key in kwargs if key not in parameters)
    required = {
        "scale_rewards",
        "loss_type",
        "remove_unused_columns",
        "num_generations",
        "mask_truncated_completions",
    }
    missing = sorted(required.intersection(unsupported))
    if missing:
        raise RuntimeError(
            "Installed TRL lacks required GRPO controls: " + ", ".join(missing)
        )
    for key in unsupported:
        kwargs.pop(key)
    return GRPOConfig(**kwargs)


def parse_args() -> argparse.Namespace:
    default_reward_config = (
        REPO_ROOT / "dataset" / "configs" / "genui_metric_v5_4.yaml"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sft-checkpoint",
        dest="model",
        required=True,
        help="Validated SFT checkpoint, local path, or Hub ID (never an untrained base model)",
    )
    parser.add_argument("--dataset", required=True, help="Training JSON/JSONL")
    parser.add_argument("--prompt-template", required=True, help="Markdown containing {response_text}")
    parser.add_argument("--reward-config", default=str(default_reward_config))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=None,
        help="Default is the smallest per-device batch making the global eval batch divisible by generations",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-completion-length", type=int, default=None)
    parser.add_argument("--max-prompt-length", type=int, default=None)
    parser.add_argument("--beta", type=float, default=0.0, help="KL coefficient")
    parser.add_argument("--loss-type", choices=("dapo", "dr_grpo", "grpo"), default="dapo")
    parser.add_argument("--scale-rewards", choices=("group", "batch", "none"), default="batch")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--alert-component-growth", type=float, default=0.20)
    parser.add_argument("--alert-length-growth", type=float, default=0.20)
    parser.add_argument("--alert-min-fidelity-gain", type=float, default=0.01)
    parser.add_argument("--alert-ema-alpha", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_trl_version()
    if args.num_generations < 2:
        raise ValueError("--num-generations must be at least 2 for group-relative advantages")
    if args.per_device_batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("Training batch size and gradient accumulation must be positive")
    if args.per_device_eval_batch_size is not None and args.per_device_eval_batch_size < 1:
        raise ValueError("--per-device-eval-batch-size must be positive")
    validate_sft_checkpoint(args.model)
    set_seed(args.seed)
    template = Path(args.prompt_template).read_text(encoding="utf-8")
    dataset = load_training_dataset(args.dataset, template)
    split = split_by_source_model_time(dataset, args.eval_fraction, args.seed)
    train_dataset = split["train"]
    eval_dataset = split.get("test")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    max_completion_length = args.max_completion_length or estimate_completion_length(
        train_dataset, tokenizer
    )
    if max_completion_length is None:
        raise ValueError(
            "Pass --max-completion-length because no accepted genui_json values are available "
            "for p99 estimation. Do not guess a small limit that truncates valid FlatSpec."
        )

    prompt_lengths = [
        len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        for prompt in train_dataset["prompt"]
    ]
    p99_prompt = _percentile(prompt_lengths, 0.99)
    model_context = getattr(tokenizer, "model_max_length", None)
    if isinstance(model_context, int) and model_context < 1_000_000:
        if p99_prompt + max_completion_length > model_context:
            raise ValueError(
                f"p99 prompt ({p99_prompt}) + completion ({max_completion_length}) exceeds "
                f"model context ({model_context}); shorten prompts or use a longer-context model."
            )

    # Include world size in this product for distributed training.
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    effective_batch = args.per_device_batch_size * args.gradient_accumulation_steps * world_size
    if effective_batch % args.num_generations != 0:
        raise ValueError(
            "per_device_batch_size * gradient_accumulation_steps * WORLD_SIZE must be divisible by "
            "num_generations."
        )
    eval_per_device_batch = args.per_device_eval_batch_size or (
        args.num_generations // math.gcd(args.num_generations, world_size)
    )
    if eval_dataset is not None and (eval_per_device_batch * world_size) % args.num_generations != 0:
        raise ValueError(
            "per_device_eval_batch_size * WORLD_SIZE must be divisible by num_generations."
        )

    ensure_v5_4_validation_ready()
    reward_config = load_reward_config_v5_4(args.reward_config)
    reward_metric_fingerprint = metric_fingerprint_v5_4(reward_config)
    reward_pipeline_fingerprint = reward_pipeline_fingerprint_v5_4(
        reward_metric_fingerprint
    )
    inflation_monitor = RewardInflationMonitor(
        component_growth_threshold=args.alert_component_growth,
        length_growth_threshold=args.alert_length_growth,
        min_fidelity_gain=args.alert_min_fidelity_gain,
        ema_alpha=args.alert_ema_alpha,
    )
    reward_fn = make_genui_grpo_reward_v5_4(
        reward_config,
        model_checkpoint=args.model,
        inflation_monitor=inflation_monitor,
    )
    scale_rewards: str | bool = False if args.scale_rewards == "none" else args.scale_rewards

    grpo_args = make_grpo_config(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=eval_per_device_batch,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        bf16=args.bf16,
        logging_steps=1,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=100,
        report_to=args.report_to,
        remove_unused_columns=False,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=max_completion_length,
        scale_rewards=scale_rewards,
        loss_type=args.loss_type,
        beta=args.beta,
        reward_weights=[1.0],
        mask_truncated_completions=True,
        use_vllm=args.use_vllm,
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    trainer = GRPOTrainer(
        model=args.model,
        args=grpo_args,
        reward_funcs=[reward_fn],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    print(
        json.dumps(
            {
                "train_samples": len(train_dataset),
                "eval_samples": 0 if eval_dataset is None else len(eval_dataset),
                "p99_prompt_tokens": p99_prompt,
                "max_completion_length": max_completion_length,
                "reward_config": args.reward_config,
                "metric_version": "5.4.0",
                "metric_fingerprint": reward_metric_fingerprint,
                "reward_pipeline_fingerprint": (
                    reward_pipeline_fingerprint
                ),
                "effective_batch": effective_batch,
                "per_device_eval_batch_size": eval_per_device_batch,
                "world_size": world_size,
                "split_policy": "source_id + model_family + chronological holdout",
            },
            indent=2,
        )
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
