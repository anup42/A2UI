"""Auditable GRPO runtime boundaries; imports do not load an ML runtime or model.

The rollout adapter uses TRL's documented ``rollout_func`` extension. Its masked
PAD sentinel represents a runtime stop to TRL 0.29.1, whose truncation detector
otherwise only recognizes scalar tokenizer EOS/PAD. It is never a sampled token
or a supervised target. Raw model tokens remain in the diagnostics.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ir_training.generation_policy import closing_sentinel_end, sha256_text, stop_express_completion

REVIEWED_TRL_VERSION = "0.29.1"


def validate_runtime_features(config_class: Any, trainer_class: Any, version: str) -> None:
    # This adapter relies on env_mask and scalar-EOS masking semantics, not just
    # the presence of a method name. Re-review a new TRL release before updating.
    if version != REVIEWED_TRL_VERSION:
        raise RuntimeError(
            f"GRPO stop/mask integration was source-reviewed for trl=={REVIEWED_TRL_VERSION}; "
            f"found {version}. Use the dedicated GRPO environment or review/test the new API."
        )
    required_config = {
        "ddp_broadcast_buffers", "gradient_checkpointing_kwargs", "generation_kwargs",
        "mask_truncated_completions", "scale_rewards", "loss_type", "steps_per_generation",
        "num_iterations", "dataloader_drop_last", "logging_nan_inf_filter",
    }
    missing = required_config - set(inspect.signature(config_class.__init__).parameters)
    if "rollout_func" not in inspect.signature(trainer_class.__init__).parameters:
        missing.add("GRPOTrainer.rollout_func")
    try:
        source = inspect.getsource(trainer_class)
    except (OSError, TypeError) as exc:
        raise RuntimeError("Cannot inspect the installed GRPOTrainer source") from exc
    if 'extra_fields.pop("env_mask", None)' not in source:
        missing.add("rollout env_mask support")
    if missing:
        raise RuntimeError("Installed GRPO runtime lacks required controls: " + ", ".join(sorted(missing)))


def dependency_report() -> dict[str, Any]:
    """Record actual imports, including PYTHONPATH overrides, without model IO."""
    result: dict[str, Any] = {"reviewed_trl_version": REVIEWED_TRL_VERSION, "packages": {}}
    for name in ("torch", "transformers", "accelerate", "peft", "trl", "datasets"):
        try:
            module = importlib.import_module(name)
        except ImportError as exc:
            result["packages"][name] = {"import_error": str(exc)}
            continue
        path = Path(module.__file__).resolve() if getattr(module, "__file__", None) else None
        try:
            distribution = importlib.metadata.distribution(name)
            dist_version = distribution.version
            direct_url = distribution.read_text("direct_url.json")
            distribution_root = Path(distribution.locate_file("")).resolve()
        except importlib.metadata.PackageNotFoundError:
            dist_version, direct_url, distribution_root = None, None, None
        result["packages"][name] = {
            "imported_version": str(getattr(module, "__version__", "unknown")),
            "distribution_version": dist_version,
            "imported_source": str(path) if path else None,
            "module_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path else None,
            "direct_url": json.loads(direct_url) if direct_url else None,
            "outside_distribution_root": bool(path and distribution_root and not path.is_relative_to(distribution_root)),
        }
    for name in ("trl.trainer.grpo_trainer", "trl.trainer.grpo_config", "trl.models.utils"):
        try:
            module = importlib.import_module(name)
        except ImportError as exc:
            result["packages"][name] = {"import_error": str(exc)}
            continue
        path = Path(module.__file__).resolve()
        result["packages"][name] = {
            "imported_source": str(path),
            "module_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return result


def prepared_prompt_messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Prepared GRPO input requires saved SFT messages; rebuild pairs or select template mode")
    result = []
    for entry in messages:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("content"), str):
            raise ValueError("GRPO supports text-only saved SFT messages")
        if entry.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("Invalid saved SFT message role")
        result.append({"role": str(entry["role"]), "content": entry["content"]})
    if result[-1]["role"] == "assistant":
        completion = row.get("completion")
        if isinstance(completion, str) and result[-1]["content"].strip() != completion.strip():
            raise ValueError("Saved SFT assistant differs from the validated completion")
        result.pop()
    if not result or result[-1]["role"] != "user":
        raise ValueError("Saved SFT prompt must end with the actual user request")
    from ir_training.eval.generate import _extract_user_text, _extract_url_map
    from ir_training.data.url_preprocess import restore_url_placeholders
    source = str(_extract_user_text(dict(row)) or "").strip()
    marker = "Create A2UI Express v1 GenUI IR for this response:\n\n"
    final_source = result[-1]["content"].split(marker, 1)[-1]
    url_map = _extract_url_map(dict(row))
    if not source or restore_url_placeholders(final_source, url_map).strip() != restore_url_placeholders(source, url_map).strip():
        raise ValueError("Saved final user request differs from this row's response_text")
    return result


def render_chat_prompt(tokenizer: Any, messages: Sequence[Mapping[str, str]],
                       template_kwargs: Mapping[str, Any] | None = None) -> str:
    kwargs = dict(template_kwargs or {})
    if {"tokenize", "add_generation_prompt"} & kwargs.keys():
        raise ValueError("Chat template kwargs cannot override prompt boundary controls")
    return tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True, **kwargs)


def prompt_fingerprint(tokenizer: Any, prompt: str) -> dict[str, Any]:
    ids = [int(value) for value in tokenizer(prompt, add_special_tokens=False)["input_ids"]]
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None and len(ids) > 1 and ids[:2] == [bos, bos]:
        raise ValueError("Rendered prompt contains duplicated leading BOS")
    return {"prompt_sha256": sha256_text(prompt), "prompt_token_ids_sha256": sha256_text(json.dumps(ids)),
            "prompt_tokens": len(ids), "add_special_tokens": False,
            "chat_template_sha256": sha256_text(str(getattr(tokenizer, "chat_template", "") or ""))}


def normalize_rollout_completion(tokenizer: Any, generated_ids: Sequence[int], *,
                                 eos_token_ids: Sequence[int], pad_token_id: int,
                                 max_new_tokens: int) -> dict[str, Any]:
    """Preserve sampled IDs; add only a loss-masked runtime terminal sentinel.

    Scan prefixes to remove HF batch padding after a close tag or native EOS.
    Never re-tokenize text: even a tag that ends inside a token retains that
    complete sampled token. The reward wrapper applies the same serving text
    boundary as evaluation while retaining the overshoot in raw evidence.
    """
    raw = [int(value) for value in generated_ids]
    ids: list[int] = []
    reason = "max_new_tokens" if len(raw) >= max_new_tokens else "unknown"
    for value in raw:
        ids.append(value)
        if value in eos_token_ids:
            reason = "eos_token"
            break
        if closing_sentinel_end(tokenizer.decode(ids, skip_special_tokens=True)) is not None:
            reason = "closing_sentinel"
            break
    if not ids:
        raise ValueError("Runtime returned an empty GRPO completion")
    terminated = reason in {"eos_token", "closing_sentinel"}
    sampled = list(ids)
    mask = [1] * len(ids)
    if terminated:
        ids.append(int(pad_token_id))
        mask.append(0)
    else:
        # PAD sampled without recognized EOS is not proof of termination. TRL's
        # scalar PAD heuristic would misclassify it, so reject that runtime case.
        if ids[-1] == pad_token_id:
            raise ValueError("Unterminated completion ends in PAD; refusing ambiguous TRL masking")
    return {"completion_ids": ids, "env_mask": mask, "sampled_token_ids": sampled,
            "stop_reason": reason, "terminated": terminated, "synthetic_terminal_tokens": int(terminated),
            "runtime_token_ids": raw, "generated_tokens": len(sampled)}


@dataclass(frozen=True)
class HealthThresholds:
    window_steps: int = 20
    min_diverse_group_fraction: float = 0.25
    max_clipped_fraction: float = 0.05
    min_nonzero_gradient_fraction: float = 0.25
    min_nonzero_update_fraction: float = 0.25
    zero_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if self.window_steps < 1:
            raise ValueError("health window_steps must be positive")
        for name in ("min_diverse_group_fraction", "max_clipped_fraction",
                     "min_nonzero_gradient_fraction", "min_nonzero_update_fraction"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.zero_tolerance < 0:
            raise ValueError("zero_tolerance must be nonnegative")


class GRPOHealthMonitor:
    """Gate rolling optimizer updates, using globally gathered training metrics."""
    def __init__(self, thresholds: HealthThresholds):
        self.thresholds = thresholds
        self.rows: deque[dict[str, float]] = deque(maxlen=thresholds.window_steps)
        self.last_step = -1

    def observe(self, step: int, metrics: Mapping[str, Any]) -> dict[str, Any]:
        keys = ("frac_reward_zero_std", "completions/clipped_ratio", "grad_norm",
                "sampled_parameter_max_delta", "nonfinite_parameters_or_gradients")
        missing = [key for key in keys if key not in metrics]
        issues = ["missing_metrics:" + ",".join(missing)] if missing else []
        row: dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if not math.isfinite(value):
                    issues.append("nonfinite:" + key)
                row[key] = float(value)
        if row.get("nonfinite_parameters_or_gradients", 0) > 0:
            issues.append("nonfinite_parameters_or_gradients")
        if step <= self.last_step:
            raise ValueError("Health updates must have strictly increasing optimizer steps")
        self.last_step = step
        self.rows.append(row)
        ready = len(self.rows) == self.thresholds.window_steps
        aggregates: dict[str, float] = {}
        if ready and not issues:
            if any(any(key not in item for key in keys) for item in self.rows):
                issues.append("missing_metrics_in_window")
            else:
                count = len(self.rows)
                aggregates = {
                    "diverse_group_fraction": sum(1 - item["frac_reward_zero_std"] for item in self.rows) / count,
                    "clipped_fraction": sum(item["completions/clipped_ratio"] for item in self.rows) / count,
                    "nonzero_gradient_fraction": sum(item["grad_norm"] > self.thresholds.zero_tolerance for item in self.rows) / count,
                    "nonzero_update_fraction": sum(item["sampled_parameter_max_delta"] > self.thresholds.zero_tolerance for item in self.rows) / count,
                }
                if aggregates["diverse_group_fraction"] <= self.thresholds.min_diverse_group_fraction:
                    issues.append("insufficient_reward_variance")
                if aggregates["clipped_fraction"] > self.thresholds.max_clipped_fraction:
                    issues.append("excessive_completion_clipping")
                if aggregates["nonzero_gradient_fraction"] <= self.thresholds.min_nonzero_gradient_fraction:
                    issues.append("insufficient_nonzero_gradients")
                if aggregates["nonzero_update_fraction"] <= self.thresholds.min_nonzero_update_fraction:
                    issues.append("insufficient_sampled_parameter_updates")
        return {"step": step, "status": "failed" if issues else "passed_window" if ready else "warming_up",
                "issues": issues, "window_updates": len(self.rows), "thresholds": asdict(self.thresholds),
                "aggregates": aggregates, "metrics": row}


def write_json_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Standard JSON, including when a non-finite health failure is being logged.
    def safe(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, Mapping):
            return {str(key): safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        return value
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(safe(record), ensure_ascii=False, allow_nan=False) + "\n")


def make_express_rollout(output_dir: str, eos_token_ids: Sequence[int], *, audit_limit: int = 256):
    """Build the supported HF-only text rollout; no model is loaded here."""
    from ir_training.generation_policy import build_stopping_criteria, generation_diagnostics
    audit_count = 0

    def rollout(prompts: list[str], trainer: Any) -> dict[str, Any]:
        import torch
        from trl.models import unwrap_model_for_generation
        nonlocal audit_count
        if trainer.use_vllm or trainer.is_fsdp_enabled or trainer.is_deepspeed_enabled:
            raise RuntimeError("Reviewed Express rollout supports HF single-device/DDP only")
        checkpoint_kwargs = dict(getattr(trainer.args, "gradient_checkpointing_kwargs", None) or {})
        if checkpoint_kwargs.get("use_reentrant") is not False:
            raise ValueError("Reviewed GRPO requires gradient_checkpointing_kwargs.use_reentrant=False")
        tokenizer = trainer.processing_class
        encoded = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(trainer.accelerator.device) for key, value in encoded.items()
                   if key in {"input_ids", "attention_mask"}}
        width = encoded["input_ids"].shape[1]
        kwargs = dict(trainer.generation_kwargs)
        kwargs.update(eos_token_id=list(eos_token_ids), pad_token_id=tokenizer.pad_token_id,
                      max_new_tokens=trainer.max_completion_length, stop_strings=None)
        was_checkpointing = bool(getattr(trainer.model, "is_gradient_checkpointing", False))
        try:
            with unwrap_model_for_generation(trainer.model_wrapped, trainer.accelerator,
                                              generation_kwargs=kwargs) as model, torch.no_grad():
                generated = model.generate(
                    **encoded, **kwargs, disable_compile=True, synced_gpus=False,
                    stopping_criteria=build_stopping_criteria(tokenizer, width),
                )
        finally:
            # TRL 0.29.1 restores checkpointing without passing kwargs. Older
            # Transformers defaults can silently restore use_reentrant=True.
            if was_checkpointing:
                trainer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpoint_kwargs)
        prompt_ids = [ids[mask.bool()].tolist() for ids, mask in
                      zip(encoded["input_ids"], encoded["attention_mask"], strict=True)]
        rows = [normalize_rollout_completion(tokenizer, ids.tolist(), eos_token_ids=eos_token_ids,
                pad_token_id=tokenizer.pad_token_id, max_new_tokens=trainer.max_completion_length)
                for ids in generated[:, width:]]
        mode = "train" if trainer.model.training else "eval"
        rank = trainer.accelerator.process_index
        for prompt, input_ids, row in zip(prompts, prompt_ids, rows, strict=True):
            if audit_count < audit_limit:
                record = {"step": int(trainer.state.global_step), "mode": mode, "rank": rank,
                          **row, "prompt_token_ids": input_ids,
                          "raw_completion": tokenizer.decode(row["sampled_token_ids"], skip_special_tokens=True),
                          **generation_diagnostics(tokenizer, row["sampled_token_ids"],
                              eos_token_ids=eos_token_ids, max_new_tokens=trainer.max_completion_length,
                              prompt_text=prompt, input_ids=input_ids)}
                write_json_record(Path(output_dir) / f"grpo_rollouts.rank{rank}.jsonl", record)
                audit_count += 1
        # Always provide env_mask on every rank, including all-truncated groups:
        # no rank conditionally skips the log-probability/loss path.
        return {"prompt_ids": prompt_ids, "completion_ids": [row["completion_ids"] for row in rows],
                "logprobs": None, "env_mask": [row["env_mask"] for row in rows],
                "termination_reason": [row["stop_reason"] for row in rows]}

    return rollout


def audited_reward(reward_fn: Any, output_dir: str, rank: int, *, audit_limit: int = 256):
    """Save candidate-group evidence while preserving the exact reward values."""
    count = 0

    def reward(*args: Any, **kwargs: Any):
        nonlocal count
        evidence: dict[str, Any] = {}
        original_log_extra = kwargs.get("log_extra")

        def capture(key: str, values: Any) -> None:
            evidence[key] = values
            if callable(original_log_extra):
                original_log_extra(key, values)

        from ir_training.data.url_preprocess import restore_url_placeholders
        raw_completions = kwargs.get("completions", args[0] if args else [])
        maps = kwargs.get("url_map") or [{}] * len(raw_completions)
        if isinstance(maps, Mapping):
            maps = [maps] * len(raw_completions)
        if len(maps) != len(raw_completions):
            raise ValueError("GRPO URL maps do not align with completion rows")
        # Apply the boundary before restoring URLs, matching Golden decoding.
        completions = [restore_url_placeholders(
            stop_express_completion(value) if isinstance(value, str) else value, mapping or {})
            for value, mapping in zip(raw_completions, maps, strict=True)]
        effective_kwargs = {**kwargs, "log_extra": capture}
        effective_args = args
        if "completions" in kwargs:
            effective_kwargs["completions"] = completions
        elif args:
            effective_args = (completions, *args[1:])
        values = reward_fn(*effective_args, **effective_kwargs)
        prompts = kwargs.get("prompts") or [""] * len(values)
        sources = kwargs.get("source_id") or [None] * len(values)
        trainer_state = kwargs.get("trainer_state")
        for index, value in enumerate(values):
            if count >= audit_limit:
                break
            write_json_record(Path(output_dir) / f"grpo_rewards.rank{rank}.jsonl", {
                "step": getattr(trainer_state, "global_step", None), "rank": rank,
                "source_id": sources[index], "prompt_sha256": sha256_text(str(prompts[index])),
                "completion": completions[index], "raw_completion": raw_completions[index], "reward": value,
                "breakdown": {key: item[index] if isinstance(item, (list, tuple)) and len(item) == len(values)
                              else item for key, item in evidence.items()},
            })
            count += 1
        return values

    reward.__name__ = getattr(reward_fn, "__name__", "genui_reward_v5_4")
    return reward


def make_health_callback(accelerator: Any, output_dir: str, thresholds: HealthThresholds):
    """All ranks audit the same optimizer-step boundaries and fail together.

    Parameter changes are sampled across trainable tensors, not a claim of full
    tensor equality. Gradients and parameter finiteness are checked in full.
    """
    import torch
    from transformers import TrainerCallback

    class HealthCallback(TrainerCallback):
        def __init__(self):
            self.monitor = GRPOHealthMonitor(thresholds)
            self.samples = []
            self.before = []
            self.grad = 0.0
            self.nonfinite = False
            self.step_metrics = {}
            self.passed_window = False
            self.model = None
            self.initial_buffer_digest = None

        def buffer_digest(self):
            digest = hashlib.sha256()
            for name, buffer in sorted(self.model.named_buffers()):
                digest.update(json.dumps([name, list(buffer.shape), str(buffer.dtype)]).encode("utf-8"))
                digest.update(buffer.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
            return digest.digest()

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            self.model = model
            if accelerator.num_processes > 1 and not args.ddp_broadcast_buffers:
                self.initial_buffer_digest = self.buffer_digest()
                digest = torch.tensor([list(self.initial_buffer_digest)], dtype=torch.uint8, device=accelerator.device)
                ranks = accelerator.gather(digest)
                if not bool((ranks == ranks[0]).all().item()):
                    raise RuntimeError("DDP buffers differ across ranks before GRPO; broadcast-disabled precondition failed")
            for name, param in model.named_parameters():
                if param.requires_grad and param.numel():
                    indices = torch.linspace(0, param.numel() - 1, min(32, param.numel()),
                                             device=param.device).long().unique()
                    self.samples.append((name, param, indices))
            if not self.samples:
                raise RuntimeError("GRPO model has no trainable parameters")
            self.before = [param.detach().reshape(-1)[indices].float().clone()
                           for _, param, indices in self.samples]

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            norm_sq = torch.zeros((), device=accelerator.device)
            finite = torch.ones((), dtype=torch.bool, device=accelerator.device)
            for _, param, _ in self.samples:
                finite &= torch.isfinite(param.detach()).all()
                if param.grad is not None:
                    finite &= torch.isfinite(param.grad).all()
                    norm_sq += param.grad.detach().float().square().sum()
            self.grad = float(norm_sq.sqrt().item())
            self.nonfinite = not bool(finite.item())

        def on_step_end(self, args, state, control, **kwargs):
            delta = torch.zeros((), device=accelerator.device)
            finite = torch.ones((), dtype=torch.bool, device=accelerator.device)
            for previous, (_, param, indices) in zip(self.before, self.samples, strict=True):
                current = param.detach().reshape(-1)[indices].float()
                finite &= torch.isfinite(param.detach()).all()
                delta = torch.maximum(delta, (current - previous).abs().max())
                previous.copy_(current)
            # Exactly one fixed-shape gather per optimizer step, on every rank.
            values = torch.tensor([[self.grad, float(delta.item()),
                                    float(self.nonfinite or not bool(finite.item()))]],
                                  dtype=torch.float64, device=accelerator.device)
            ranks = accelerator.gather(values)
            self.step_metrics = {"grad_norm": float(ranks[:, 0].min().item()),
                                 "sampled_parameter_max_delta": float(ranks[:, 1].min().item()),
                                 "nonfinite_parameters_or_gradients": float(ranks[:, 2].max().item())}
            if self.initial_buffer_digest is not None and state.global_step % thresholds.window_steps == 0:
                changed = torch.tensor([int(self.buffer_digest() != self.initial_buffer_digest)],
                                       device=accelerator.device)
                changed_ranks = accelerator.gather(changed)
                if bool(changed_ranks.any().item()):
                    write_json_record(Path(output_dir) / f"grpo_health.rank{accelerator.process_index}.jsonl",
                                      {"step": state.global_step, "status": "failed", "issues": ["mutable_ddp_buffers"]})
                    raise RuntimeError("GRPO model buffers changed with broadcasts disabled; review dynamic buffer behavior")

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "reward" not in logs or "eval_reward" in logs:
                return
            # TRL's reward/clipping metrics are already gathered across ranks.
            result = self.monitor.observe(int(state.global_step), {**logs, **self.step_metrics})
            write_json_record(Path(output_dir) / f"grpo_health.rank{accelerator.process_index}.jsonl", result)
            self.passed_window |= result["status"] == "passed_window"
            if result["issues"]:
                raise RuntimeError("GRPO health gate failed: " + ", ".join(result["issues"]))

        def on_train_end(self, args, state, control, **kwargs):
            if not self.passed_window:
                raise RuntimeError("GRPO ended before a complete health window passed; smoke is inconclusive")

    return HealthCallback()
