#!/usr/bin/env python3
"""Reference GRPO + LoRA entry point for text -> A2UI Express v1.

Dataset columns
---------------
Required:
  response_text: upstream LLM text that the completion must represent as UI.
Recommended:
  expected_ui_contract: persisted source-side contract, independent of candidate.
  intent_bucket, assets, query_id/source_id/response_id.
Required for active training:
  completion: strict A2UI Express v1 reference completion.

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
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_SRC = REPO_ROOT / "dataset" / "src"
if str(DATASET_SRC) not in sys.path:
    sys.path.insert(0, str(DATASET_SRC))
sys.path.insert(0, str(REPO_ROOT / "training" / "src"))

from ir_training.generation_policy import preserve_generation_eos
from ir_training.eval.generate import build_prediction_record
from ir_training.data.url_preprocess import restore_url_placeholders
from ir_training.models.hf_loading import load_hf_model
from ir_training.train.lora_config import resolve_lora_config_targets
from ir_training.train.grpo_runtime import (
    HealthThresholds, audited_reward, dependency_report, make_express_rollout, make_health_callback,
    prepared_prompt_messages, prompt_fingerprint, render_chat_prompt, validate_runtime_features,
)

from pipeline.genui_quality import (
    RewardInflationMonitor,
    ensure_v5_4_validation_ready,
    load_reward_config_v5_4,
    make_genui_grpo_reward_v5_4,
    metric_fingerprint_v5_4,
    reward_pipeline_fingerprint_v5_4,
)


def build_prompt(template: str, response_text: str, assets: Any) -> str:
    # SFT prepared rows already contain the exact masked source and asset policy.
    # Appending a GRPO-only policy changed even no-asset prompts by 145 characters.
    # Assets remain source-side reward context, never an implicit prompt mutation.
    del assets
    if "{response_text}" not in template:
        raise ValueError("Prompt template must contain {response_text}")
    return template.replace("{response_text}", response_text.strip())


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


def _express_completion_from_row(row: Mapping[str, Any]) -> str:
    """Read the sole training target and reject legacy graph targets."""
    target_format = str(row.get("target_format") or "a2ui_express_v1").strip().lower()
    if target_format != "a2ui_express_v1":
        raise ValueError(f"Active GRPO requires a2ui_express_v1, got {target_format!r}")
    completion = row.get("completion")
    if not isinstance(completion, str) or not completion.strip():
        targets = row.get("completion_targets")
        if isinstance(targets, Mapping):
            completion = targets.get("a2ui_express_v1")
    if not isinstance(completion, str) or not completion.strip():
        raise ValueError("Training row is missing a strict A2UI Express completion")
    # Validate the target at the data-loader boundary so a legacy JSON graph
    # can never silently enter GRPO as a completion.
    from pipeline.ir_formats import validate_express_completion, compile_express_to_wire

    result = validate_express_completion(completion)
    if not result.raw_valid:
        detail = result.errors[0] if result.errors else "invalid_express_completion"
        raise ValueError(f"Invalid A2UI Express training completion: {detail}")
    try:
        compile_express_to_wire(result.canonical_graph)
    except Exception as exc:
        raise ValueError(f"Production-invalid A2UI Express reference: {exc}") from exc
    return completion


def load_chat_scaffold(path: str | None) -> list[dict[str, str]]:
    """Fixed leading messages (system prompt, few-shot turns) prepended to every prompt.

    The JSON file holds a list of ``{"role": ..., "content": ...}`` entries. Use it
    to reproduce the exact conversation prefix the SFT checkpoint was trained on;
    a reward optimized under a different prefix does not transfer back to it.
    """
    if not path:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("--chat-scaffold must contain a JSON list of chat messages")
    scaffold: list[dict[str, str]] = []
    for entry in data:
        if not isinstance(entry, dict) or "role" not in entry or "content" not in entry:
            raise ValueError("Each --chat-scaffold entry needs a 'role' and a 'content'")
        scaffold.append({"role": str(entry["role"]), "content": str(entry["content"])})
    return scaffold


def load_training_dataset(
    path: str,
    template: str | None,
    *,
    tokenizer: Any | None = None,
    scaffold: Sequence[Mapping[str, str]] = (),
    prompt_source: str = "prepared",
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> Dataset:
    from datasets import load_dataset
    if Path(path).suffix.lower() not in {".json", ".jsonl"}:
        raise ValueError("The reference loader supports .json and .jsonl")
    dataset = load_dataset("json", data_files=path, split="train")
    if "response_text" not in dataset.column_names:
        raise ValueError("Dataset is missing required column: response_text")

    def prepare(row: dict[str, Any]) -> dict[str, Any]:
        completion = _express_completion_from_row(row)
        bound = build_prediction_record({**row, "completion": completion}, "")
        url_map = bound["url_map"]
        assets = restore_url_placeholders(bound["assets"], url_map)
        if prompt_source == "prepared":
            if tokenizer is None:
                raise ValueError("Prepared SFT messages require --chat-template")
            messages = prepared_prompt_messages({**row, "completion": completion})
            prompt = render_chat_prompt(tokenizer, messages, chat_template_kwargs)
        else:
            if not template:
                raise ValueError("Template mode requires --prompt-template")
            body = build_prompt(template, str(row["response_text"]), assets)
            messages = [dict(m) for m in scaffold] + [{"role": "user", "content": body}]
            prompt = render_chat_prompt(tokenizer, messages, chat_template_kwargs) if tokenizer else body
            if row.get("messages"):
                if tokenizer is None:
                    raise ValueError("Cannot compare saved chat messages to a raw-text GRPO prompt")
                expected = render_chat_prompt(tokenizer, prepared_prompt_messages({**row, "completion": completion}), chat_template_kwargs)
                if prompt != expected:
                    raise ValueError("Reconstructed GRPO prompt differs from saved SFT prompt; use --prompt-source prepared")
        fingerprint = prompt_fingerprint(tokenizer, prompt) if tokenizer else {}
        return {
            "source_id": _stable_source_id(bound),
            "id": bound["id"],
            "query_id": bound["query_id"],
            "response_id": bound["response_id"],
            "ui_id": bound["ui_id"],
            "prompt": prompt,
            "prompt_message_roles": [message["role"] for message in messages],
            "prompt_provenance": fingerprint,
            # The model sees the original prepared/masked prompt. Reward sees
            # the same restored response/assets/contracts as Golden evaluation.
            "response_text": bound["response_text"],
            "response_text_sha256": bound["response_text_sha256"],
            "source_context_sha256": bound["source_context_sha256"],
            "url_map": url_map,
            "intent_bucket": bound["intent_bucket"] or bound["intent"],
            "assets": assets,
            "expected_ui_contract": restore_url_placeholders(
                bound["expected_ui_contract_v5_4"] if bound["expected_ui_contract_v5_4"] is not None else bound["expected_ui_contract"], url_map),
            "expected_ui_contract_source": bound["expected_ui_contract_v5_4_source"] or bound["expected_ui_contract_source"],
            "source_model_family": _source_model_family(row),
            "source_created_at": _source_timestamp(row),
            "completion": completion,
            "target_format": "a2ui_express_v1",
        }

    return dataset.map(prepare, remove_columns=dataset.column_names)


def split_by_source_model_time(dataset: Dataset, eval_fraction: float, seed: int) -> DatasetDict:
    """Leak-free source split, stratified by model family with latest sources held out."""
    from datasets import DatasetDict
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


def validate_sft_checkpoint(checkpoint: str) -> dict[str, Any]:
    """Verify local weight provenance without claiming numerical merge parity."""
    path = Path(checkpoint)
    if not path.exists():
        return {"kind": "hub_reference", "local_identity_verified": False,
                "numerical_parity_verified": False, "checkpoint": checkpoint}
    if not path.is_dir():
        raise ValueError(f"SFT checkpoint must be a directory or Hub ID: {checkpoint}")
    if (path / "adapter_config.json").is_file():
        raise ValueError("GRPO requires a verified merged SFT seed, not an adapter-only checkpoint. "
                         "Merge with its original base and prove generation parity first.")
    weights = {item.name: item for pattern in ("model*.safetensors", "pytorch_model*.bin")
               for item in path.glob(pattern) if item.is_file()}
    if not (path / "config.json").is_file() or not weights:
        raise ValueError("GRPO seed is missing HF config.json or full model weight shards")
    digests: dict[str, str] = {}

    def files_match(items: Any) -> bool:
        if not isinstance(items, list):
            return False
        if any(not isinstance(item, dict) for item in items):
            return False
        recorded = {str(item.get("path")): item for item in items}
        if len(recorded) != len(items):
            return False
        recorded_weights = {name for name in recorded if name.endswith((".safetensors", ".bin"))
                            and name.startswith(("model", "pytorch_model"))}
        if recorded_weights != set(weights):
            return False
        for name, expected in recorded.items():
            # Repository manifests use artifact basenames. A copied run may
            # retain old absolute checkpoint paths, but file names stay portable.
            if Path(name).name != name or name in {"", ".", ".."}:
                return False
            file = path / name
            if not file.is_file():
                return False
            if expected.get("size") is not None and int(expected["size"]) != file.stat().st_size:
                return False
            if name not in digests:
                digest = hashlib.sha256()
                with file.open("rb") as handle:
                    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                digests[name] = digest.hexdigest()
            if expected.get("sha256") != digests[name]:
                return False
        return True

    merge_path = path / "qat_mtp_merge_metadata.json"
    metadata_path = path / "training_metadata.json"
    if merge_path.is_file():
        metadata = json.loads(merge_path.read_text(encoding="utf-8-sig"))
        if not isinstance(metadata, dict) or not metadata.get("adapter_files") or not files_match(metadata.get("merged_model_files")):
            raise ValueError("Merged seed provenance does not match the local full model weights")
        return {"kind": "merged_lora", "local_identity_verified": True,
                "numerical_parity_verified": False, "manifest": str(merge_path),
                "training_run_verified": bool((metadata.get("training_run_metadata") or {}).get("verified")),
                "model_sha256s": {name: digests[name] for name in weights}}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        method = str((metadata.get("training") or {}).get("method", ""))
        entries = metadata.get("adapter_checkpoints") or []
        if method not in {"full_finetune_sft", "full_finetune_qat"} or not any(
                item.get("checkpoint_kind") == "full_model" and files_match(item.get("files"))
                for item in entries if isinstance(item, dict)):
            raise ValueError("Full-SFT seed provenance does not match local full model weights")
        return {"kind": "full_finetune", "training_method": method, "local_identity_verified": True,
                "numerical_parity_verified": False, "manifest": str(metadata_path),
                "model_sha256s": {name: digests[name] for name in weights}}
    raise ValueError("Local GRPO seed requires matching qat_mtp_merge_metadata.json or full-SFT training_metadata.json; "
                     "a trainer_state marker alone does not bind these weights to training.")


def _percentile(values: Sequence[int], p: float) -> int:
    if not values:
        raise ValueError("Cannot calculate a percentile from no values")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1))
    return ordered[index]


def estimate_completion_length(dataset: Dataset, tokenizer: Any) -> int | None:
    if "completion" not in dataset.column_names:
        return None
    lengths: list[int] = []
    for completion in dataset["completion"]:
        if isinstance(completion, str) and completion.strip():
            lengths.append(len(tokenizer(completion, add_special_tokens=False)["input_ids"]))
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
    if unsupported:
        raise RuntimeError(
            "Installed TRL does not support requested GRPO controls: " + ", ".join(unsupported)
        )
    return GRPOConfig(**kwargs)


def load_grpo_model(args: Any, peft_config: Any) -> tuple[Any, dict[str, Any], list[str]]:
    """Called only on the GPU training path, before PEFT wraps projection modules."""
    loading_config = {
        "model_loader": args.model_loader,
        # Match TRL 0.29.1's default weight dtype; bf16 AMP is a separate setting.
        "dtype": args.model_dtype,
        "attn_implementation": args.attn_implementation or "",
        "device_map": None,
        "trust_remote_code": False,
        "require_exact_checkpoint_keys": True,
    }
    model = load_hf_model(args.model, loading_config)
    resolved = sorted(resolve_lora_config_targets(peft_config, model))
    if any(any(branch in name for branch in ("vision_model", "audio_tower", "audio_model", "vision_tower"))
           for name in resolved):
        raise ValueError("Text-only GRPO LoRA includes unused audio/vision modules; narrow --lora-target-modules")
    return model, loading_config, resolved


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
    parser.add_argument("--prompt-template", help="Explicit reconstruction only: Markdown containing {response_text}")
    parser.add_argument("--prompt-source", choices=("prepared", "template"), default="prepared",
                        help="Default consumes exact saved SFT messages; template mode checks parity when present")
    parser.add_argument("--chat-template-kwargs", help="JSON file with the exact SFT model.chat_template_kwargs")
    parser.add_argument("--dependency-preflight-only", action="store_true",
                        help="Record/check Python libraries and exit before tokenizer/model loading")
    parser.add_argument(
        "--chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Render prompts through the checkpoint's chat template so they match the "
            "conversation format the SFT checkpoint was trained on. Use --no-chat-template "
            "for a checkpoint trained on raw-text prompts."
        ),
    )
    parser.add_argument(
        "--chat-scaffold",
        default=None,
        help=(
            "JSON list of fixed leading chat messages (system prompt, few-shot turns) "
            "prepended to every prompt to reproduce the SFT conversation prefix."
        ),
    )
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
    parser.add_argument("--max-steps", type=int, default=-1, help="Use 20 for the first GPU-only learning smoke")
    parser.add_argument("--max-completion-length", type=int, default=None)
    parser.add_argument("--max-prompt-length", type=int, default=None)
    parser.add_argument("--beta", type=float, default=0.0, help="KL coefficient")
    parser.add_argument("--loss-type", choices=("dapo", "dr_grpo", "grpo"), default="dapo")
    parser.add_argument("--scale-rewards", choices=("group", "batch", "none"), default="batch")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", help="Required for training: comma-separated suffixes or regex:<expression>; inspect language-only matches")
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--ddp-broadcast-buffers", action=argparse.BooleanOptionalAction, default=False,
                        help="Default disables rank-dependent forward buffer broadcasts; GPU DDP smoke still required")
    parser.add_argument("--health-window-steps", type=int, default=20)
    parser.add_argument("--health-min-diverse-groups", type=float, default=0.25)
    parser.add_argument("--health-max-clipped", type=float, default=0.05)
    parser.add_argument("--health-min-nonzero-gradients", type=float, default=0.25)
    parser.add_argument("--health-min-nonzero-updates", type=float, default=0.25)
    parser.add_argument("--audit-rollout-limit", type=int, default=256)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-loader", choices=("auto_causal_lm", "auto_multimodal_lm"), default="auto_causal_lm",
                        help="Use the same HF loader as the verified SFT seed")
    parser.add_argument("--model-dtype", choices=("float32", "bfloat16", "float16"), default="float32",
                        help="Weight loading dtype; default preserves TRL's float32 load, independently of --bf16 AMP")
    parser.add_argument("--attn-implementation", default=None,
                        help="Optional HF attention backend; omitted preserves the model's default")
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--alert-component-growth", type=float, default=0.20)
    parser.add_argument("--alert-length-growth", type=float, default=0.20)
    parser.add_argument("--alert-min-fidelity-gain", type=float, default=0.01)
    parser.add_argument("--alert-ema-alpha", type=float, default=0.25)
    return parser.parse_args()


def _report_prompt_rendering(
    tokenizer: Any,
    dataset: Dataset,
    chat_template_applied: bool,
    scaffold: Sequence[Mapping[str, str]],
) -> None:
    """Print how prompts were rendered so a format mismatch is visible immediately.

    Also surfaces a duplicated leading BOS, which is the usual silent failure when
    a chat template that already emits BOS is tokenized with special tokens added.
    """
    sample = str(dataset["prompt"][0])
    head = tokenizer(sample, add_special_tokens=False)["input_ids"][:6]
    bos_id = getattr(tokenizer, "bos_token_id", None)
    doubled = bool(bos_id is not None and len(head) >= 2 and head[0] == bos_id and head[1] == bos_id)
    print(
        json.dumps(
            {
                "chat_template_applied": chat_template_applied,
                "prompt_message_roles": dataset[0].get("prompt_message_roles", [m.get("role") for m in scaffold]),
                "prompt_head_token_ids": [int(t) for t in head],
                "prompt_head_tokens": tokenizer.convert_ids_to_tokens(head),
                "prompt_preview": sample[:180],
                "duplicate_leading_bos": doubled,
            },
            indent=2,
        ),
        flush=True,
    )
    if doubled:
        raise ValueError(
            "Rendered prompt begins with two BOS tokens; the chat template already "
            "emits BOS. Fix the scaffold or disable --chat-template."
        )


def main() -> None:
    args = parse_args()
    # Importing this module and --help are safe without the GPU stack. This
    # explicit preflight imports libraries but never loads model/tokenizer data.
    global GRPOConfig
    report = dependency_report()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rank = max(0, int(os.environ.get("RANK", "0")))
    report_path = output_dir / f"grpo_dependencies.rank{rank}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    errors = [name for name, info in report["packages"].items() if info.get("import_error")]
    if errors:
        raise RuntimeError(f"Dependency preflight failed for {', '.join(errors)}; details: {report_path}")
    if report["packages"]["trl"].get("outside_distribution_root"):
        raise RuntimeError(f"TRL is shadowed outside its installed distribution. Remove stale PYTHONPATH overrides; {report_path}")
    import trl
    from trl import GRPOConfig, GRPOTrainer
    from peft import LoraConfig
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig, TrainerCallback, set_seed
    validate_runtime_features(GRPOConfig, GRPOTrainer, str(trl.__version__))
    if not hasattr(TrainerCallback, "on_pre_optimizer_step"):
        raise RuntimeError("Transformers lacks on_pre_optimizer_step required for health checks")
    if args.dependency_preflight_only:
        print(f"Dependency/source preflight passed; no model loaded. Report: {report_path}")
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GRPO training requires a CUDA GPU; use --dependency-preflight-only on this PC")
    if args.use_vllm:
        raise ValueError("The reviewed termination adapter supports HF only; vLLM requires a separately validated rollout")
    if not args.lora_target_modules:
        raise ValueError("Choose --lora-target-modules explicitly; all-linear may include unused multimodal branches")
    target_modules = (args.lora_target_modules[len("regex:"):] if args.lora_target_modules.startswith("regex:")
                      else [value.strip() for value in args.lora_target_modules.split(",") if value.strip()])
    if not target_modules:
        raise ValueError("--lora-target-modules must not be empty")
    thresholds = HealthThresholds(args.health_window_steps, args.health_min_diverse_groups,
                                  args.health_max_clipped, args.health_min_nonzero_gradients,
                                  args.health_min_nonzero_updates)
    if 0 < args.max_steps < thresholds.window_steps:
        raise ValueError("--max-steps must cover at least one complete health window")
    if args.num_generations < 2:
        raise ValueError("--num-generations must be at least 2 for group-relative advantages")
    if args.per_device_batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("Training batch size and gradient accumulation must be positive")
    if args.per_device_eval_batch_size is not None and args.per_device_eval_batch_size < 1:
        raise ValueError("--per-device-eval-batch-size must be positive")
    seed_provenance = validate_sft_checkpoint(args.model)
    set_seed(args.seed)

    # The tokenizer is needed before the dataset so prompts can be rendered
    # through the same chat template the SFT checkpoint was trained on.
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    native_config = AutoConfig.from_pretrained(args.model)
    try:
        native_generation = GenerationConfig.from_pretrained(args.model)
    except OSError:
        native_generation = GenerationConfig.from_model_config(native_config)
    native_eos_ids = preserve_generation_eos(
        SimpleNamespace(config=native_config, generation_config=native_generation), tokenizer)

    template = Path(args.prompt_template).read_text(encoding="utf-8") if args.prompt_template else None
    scaffold = load_chat_scaffold(args.chat_scaffold)
    template_kwargs = (json.loads(Path(args.chat_template_kwargs).read_text(encoding="utf-8"))
                       if args.chat_template_kwargs else {})
    if not isinstance(template_kwargs, dict):
        raise ValueError("--chat-template-kwargs must contain a JSON object")
    if args.prompt_source == "prepared" and (template or scaffold):
        raise ValueError("Prepared mode uses saved messages; remove --prompt-template/--chat-scaffold "
                         "or select template mode for an explicit parity check")
    dataset = load_training_dataset(
        args.dataset,
        template,
        tokenizer=tokenizer if args.chat_template else None,
        scaffold=scaffold,
        prompt_source=args.prompt_source,
        chat_template_kwargs=template_kwargs,
    )
    split = split_by_source_model_time(dataset, args.eval_fraction, args.seed)
    train_dataset = split["train"]
    eval_dataset = split.get("test")

    _report_prompt_rendering(tokenizer, train_dataset, bool(args.chat_template), scaffold)

    max_completion_length = args.max_completion_length or estimate_completion_length(
        train_dataset, tokenizer
    )
    if max_completion_length is None:
        raise ValueError(
            "Pass --max-completion-length because no accepted A2UI Express completions are available "
            "for p99 estimation. Do not guess a small limit that truncates valid Express output."
        )

    prompt_lengths = [
        len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        for prompt in dataset["prompt"]
    ]
    p99_prompt = _percentile(prompt_lengths, 0.99)
    max_prompt = max(prompt_lengths)
    if args.max_prompt_length is not None and max_prompt > args.max_prompt_length:
        raise ValueError(f"A prompt has {max_prompt} tokens, exceeding --max-prompt-length; "
                         "filter/rebuild whole rows instead of truncating chat prefixes")
    model_context = getattr(tokenizer, "model_max_length", None)
    if isinstance(model_context, int) and model_context < 1_000_000:
        # One additional loss-masked terminal sentinel may be used by TRL.
        if max_prompt + max_completion_length + 1 > model_context:
            raise ValueError(
                f"max prompt ({max_prompt}) + completion ({max_completion_length}) + terminal sentinel exceeds "
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
    if len(train_dataset) < effective_batch // args.num_generations:
        raise ValueError("Too few distinct dataset rows for one complete generated-sample batch")
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
    reward_fn = audited_reward(reward_fn, args.output_dir, rank, audit_limit=args.audit_rollout_limit)
    scale_rewards: str | bool = False if args.scale_rewards == "none" else args.scale_rewards

    grpo_args = make_grpo_config(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=eval_per_device_batch,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_broadcast_buffers=args.ddp_broadcast_buffers,
        ddp_find_unused_parameters=False,
        dataloader_drop_last=True,
        bf16=args.bf16,
        logging_steps=1,
        logging_nan_inf_filter=False,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=100,
        report_to=args.report_to,
        remove_unused_columns=False,
        num_generations=args.num_generations,
        max_completion_length=max_completion_length,
        steps_per_generation=args.gradient_accumulation_steps,
        num_iterations=1,
        generation_kwargs={"eos_token_id": native_eos_ids},
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
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model, loading_config, resolved_lora_targets = load_grpo_model(args, peft_config)
    native_eos_ids = preserve_generation_eos(model, tokenizer, extra_eos_ids=native_eos_ids)
    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        reward_funcs=[reward_fn],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        rollout_func=make_express_rollout(args.output_dir, native_eos_ids, audit_limit=args.audit_rollout_limit),
    )
    effective_eos = preserve_generation_eos(trainer.model, tokenizer, extra_eos_ids=native_eos_ids)
    trainer.generation_config.eos_token_id = list(effective_eos)
    trainer.generation_kwargs["eos_token_id"] = list(effective_eos)
    trainer.add_callback(make_health_callback(trainer.accelerator, args.output_dir, thresholds))
    matched_modules = [name for name, module in trainer.model.named_modules() if hasattr(module, "lora_A")]
    if any(any(branch in name for branch in ("vision_model", "audio_tower", "audio_model", "vision_tower"))
           for name in matched_modules):
        raise ValueError("Text-only GRPO LoRA includes unused audio/vision modules; narrow --lora-target-modules")
    manifest = {"seed_provenance": seed_provenance, "prompt_source": args.prompt_source, "chat_template_kwargs": template_kwargs,
                "prompt_fingerprints_sha256": hashlib.sha256(json.dumps(
                    list(dataset["prompt_provenance"]), sort_keys=True).encode("utf-8")).hexdigest(),
                "first_prompt": train_dataset[0]["prompt_provenance"], "effective_eos_ids": effective_eos,
                "stopping": "generated-only unquoted </a2ui> or native EOS",
                "trl_termination": "PAD terminal sentinel with env_mask=0; no synthetic token loss",
                "ddp_broadcast_buffers": args.ddp_broadcast_buffers,
                "gradient_checkpointing_kwargs": {"use_reentrant": False},
                "lora_target_modules": target_modules, "resolved_lora_targets": resolved_lora_targets,
                "matched_lora_modules": matched_modules, "model_loading": loading_config,
                "gpu_runtime_validated": False, "health_thresholds": vars(thresholds),
                "reward_weights": [1.0], "beta": args.beta}
    (output_dir / f"grpo_preflight.rank{rank}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
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
