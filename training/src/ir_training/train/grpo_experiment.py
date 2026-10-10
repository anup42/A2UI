"""Bounded, source-preserving GRPO experiment controls (no ML imports at import).

These controls do not change rewards, sampling, likelihoods, or the GRPO loss.
Optimizer health is not a model-quality gate; promotion uses a fresh QAT-on
step-zero development baseline and never Golden35/Bixby50.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

SELECTOR = "unique_source_generation_reward_v5_4_avg"
GUARDS = ("unique_source_schema_valid_strict_rate", "unique_source_cap_v5_4_avg",
          "unique_source_fully_root_reachable_v5_4_avg")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_token_budgets(sequence: int, prompt: int, completion: int, context: int) -> None:
    for name, value in (("sequence", sequence), ("prompt", prompt),
                        ("completion", completion), ("context", context)):
        if type(value) is not int or value < 1:
            raise ValueError(f"Token budgets must be positive integers (GRPO {name})")
    # sequence is prompt + REFERENCE completion, not the rollout prompt length.
    if sequence > context:
        raise ValueError("Prepared reference sequence exceeds model context")
    if prompt + completion + 1 > context:
        raise ValueError("Prompt + completion + GRPO stop sentinel exceeds model context")


def check_preparation_retention(audit: Mapping[str, Any], minimum: float) -> dict[str, Any]:
    if isinstance(minimum, bool) or not isinstance(minimum, (int, float)) or not math.isfinite(minimum) or not 0 <= minimum <= 1:
        raise ValueError("min_retained_fraction must be finite and between 0 and 1")
    results = {}
    for split in ("train", "val"):
        counts = (audit.get("prepared_counts") or {}).get(split)
        if not isinstance(counts, Mapping):
            raise ValueError(f"Preparation is missing {split} retention evidence")
        total, kept = counts.get("input_rows"), counts.get("accepted_rows")
        if type(total) is not int or type(kept) is not int or total < 1 or not 0 < kept <= total:
            raise ValueError(f"Invalid or empty {split} preparation counts")
        fraction = kept / total
        results[split] = {"input_rows": total, "accepted_rows": kept,
                          "retained_fraction": fraction, "minimum": minimum,
                          "quarantine_reasons": counts.get("quarantine_reasons", {})}
        if fraction < minimum:
            raise ValueError(f"GRPO {split} retained only {kept}/{total} ({fraction:.2%}); "
                             f"minimum is {minimum:.2%}. Inspect data_audit.json and token lengths; "
                             "increase valid budgets/reprepare whole rows, never truncate source facts.")
    return results


def validation_subset_indices(rows: Sequence[Mapping[str, Any]], maximum: int, seed: int) -> list[int]:
    """Deterministic, source-deduplicated round-robin by intent and prompt length.

    Used only for inexpensive periodic RL validation. All original prepared
    validation rows remain on disk. maximum=0 retains the complete split.
    Selection consumes no global RNG and does not inspect scores or targets.
    """
    if type(maximum) is not int or maximum < 0:
        raise ValueError("validation_max_rows must be a nonnegative integer")
    if maximum == 0:
        return list(range(len(rows)))
    sources: dict[str, tuple[str, int]] = {}
    for index, row in enumerate(rows):
        source = row.get("source_id")
        if not isinstance(source, str) or not source:
            raise ValueError("Periodic validation requires source_id")
        prompt = str(row.get("prompt", ""))
        digest = hashlib.sha256(f"{seed}:{source}:{prompt}".encode()).hexdigest()
        previous = sources.get(source)
        if previous is None or (digest, index) < previous:
            sources[source] = (digest, index)
    buckets: dict[tuple[str, int], list[tuple[str, int]]] = defaultdict(list)
    for digest, index in sources.values():
        row = rows[index]
        length = int((row.get("prompt_provenance") or {}).get("prompt_tokens", 0))
        buckets[(str(row.get("intent_bucket") or "unknown"), length // 512)].append((digest, index))
    # Seeded bucket order avoids alphabetically favoring intents at the cap.
    order = sorted(buckets, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    queues = deque(deque(sorted(buckets[key])) for key in order)
    selected = []
    while queues and len(selected) < maximum:
        bucket = queues.popleft()
        selected.append(bucket.popleft()[1])
        if bucket:
            queues.append(bucket)
    return sorted(selected)


def quality_metrics(aggregate: Mapping[str, Any]) -> dict[str, float]:
    result = {}
    for key in (SELECTOR, *GUARDS):
        value = aggregate.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Missing/nonfinite development metric: {key}")
        upper = 100.0 if key == SELECTOR else 1.0
        if not 0 <= value <= upper:
            raise ValueError(f"Development metric is out of range: {key}")
        result[key] = float(value)
    return result


@dataclass(frozen=True)
class QualityPolicy:
    patience: int = 3
    min_delta: float = 0.5  # V5.4 points, not 0..1 reward units
    max_guard_drop: float = 0.0

    def __post_init__(self) -> None:
        if type(self.patience) is not int or self.patience < 0:
            raise ValueError("quality patience must be a nonnegative integer (0 disables stopping)")
        for name in ("min_delta", "max_guard_drop"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"quality {name} must be finite and nonnegative")
        if self.max_guard_drop > 1:
            raise ValueError("max_guard_drop must not exceed 1")


class QualityMonitor:
    def __init__(self, baseline: Mapping[str, Any], policy: QualityPolicy):
        self.baseline = quality_metrics(baseline)
        self.policy = policy
        self.best = self.baseline[SELECTOR]
        self.stale = 0
        self.last_step = 0

    def promotion(self, aggregate: Mapping[str, Any]) -> dict[str, Any]:
        metrics = quality_metrics(aggregate)
        issues = []
        gain = metrics[SELECTOR] - self.baseline[SELECTOR]
        if gain <= self.policy.min_delta:
            issues.append("no_meaningful_development_gain")
        for key in GUARDS:
            if metrics[key] + self.policy.max_guard_drop + 1e-12 < self.baseline[key]:
                issues.append("regressed:" + key)
        return {"passed": not issues, "issues": issues, "gain_points": gain,
                "baseline": self.baseline, "selected": metrics, "policy": asdict(self.policy)}

    def observe(self, step: int, aggregate: Mapping[str, Any]) -> dict[str, Any]:
        if type(step) is not int or step <= self.last_step:
            raise ValueError("Development observations require increasing positive optimizer steps")
        self.last_step = step
        decision = self.promotion(aggregate)
        score = decision["selected"][SELECTOR]
        if decision["passed"] and score > self.best + self.policy.min_delta:
            self.best, self.stale = score, 0
        else:
            self.stale += 1
        return {**decision, "step": step, "stale_evaluations": self.stale,
                "best_eligible_score": self.best,
                "should_stop": bool(self.policy.patience and self.stale >= self.policy.patience)}


def make_quality_callback(*, accelerator: Any, baseline_callback: Any, golden_callback: Any,
                          baseline_dir: Path, golden_dir: Path, output: Path,
                          config_path: Path, policy: QualityPolicy) -> Any:
    """All ranks run the same callback events; only rank zero reads/writes reports.

    Uses the shared Golden generator/scorer, not stochastic RL-validation reward.
    The separate step-zero callback has save_best_checkpoint=False. It can never
    be mislabeled as a trained GRPO checkpoint with a passed health window.
    """
    from transformers import TrainerCallback
    from ir_training.train.callbacks import _evaluation_label
    from ir_training.train.grpo_runtime import write_json_record

    def on_main(fn):
        payload = [None]
        if accelerator.is_main_process:
            try:
                payload[0] = {"value": fn()}
            except Exception as exc:
                payload[0] = {"error": f"{type(exc).__name__}: {exc}"}
        if accelerator.num_processes > 1:
            import torch.distributed as distributed
            distributed.broadcast_object_list(payload, src=0)
        if not isinstance(payload[0], dict) or "error" in payload[0]:
            raise RuntimeError(f"GRPO quality control failed: {payload[0]}")
        return payload[0]["value"]

    def read_metrics(path):
        aggregate = json.loads(path.read_text(encoding="utf-8"))
        return quality_metrics(aggregate)

    def check_cohort(before, after):
        left = json.loads(before.read_text(encoding="utf-8"))
        right = json.loads(after.read_text(encoding="utf-8"))
        for key in ("golden_set_sha256", "golden_set_rows", "unique_source_count"):
            if left.get(key) is None or left.get(key) != right.get(key):
                raise ValueError(f"GRPO baseline/selected cohort mismatch: {key}")

    class QualityCallback(TrainerCallback):
        monitor = None
        baseline_path = None

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            if int(state.global_step) != 0:
                raise ValueError("GRPO quality baseline requires a fresh optimizer at step zero")
            # Preserve the rank-specific training RNG across extra evaluation.
            import random
            import numpy as np
            import torch
            devices = [accelerator.device.index] if accelerator.device.type == "cuda" else []
            python_state, numpy_state = random.getstate(), np.random.get_state()
            try:
                with torch.random.fork_rng(devices=devices):
                    baseline_callback.on_evaluate(args, state, control, model=model)
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)
            self.baseline_path = baseline_dir / _evaluation_label("evaluate", state) / "aggregate_metrics.json"
            metrics = on_main(lambda: read_metrics(self.baseline_path))
            self.monitor = QualityMonitor(metrics, policy)
            return control

        def observe(self, state, control):
            step = int(state.global_step)
            if self.monitor is None:
                raise RuntimeError("No fresh GRPO quality baseline was recorded")
            if step == 0 or step == self.monitor.last_step:
                return control
            if golden_callback.last_completed_step != step:
                raise RuntimeError("GRPO quality control must run immediately after Golden evaluation")
            path = golden_dir / _evaluation_label("evaluate", state) / "aggregate_metrics.json"
            metrics = on_main(lambda: read_metrics(path))
            decision = self.monitor.observe(step, metrics)
            on_main(lambda: write_json_record(output / "grpo_quality_history.jsonl", decision))
            if decision["should_stop"]:
                control.should_training_stop = True
            return control

        def on_evaluate(self, args, state, control, **kwargs):
            return self.observe(state, control)

        def on_train_end(self, args, state, control, **kwargs):
            return self.observe(state, control)

        def finalize(self):
            def write_gate():
                selected = golden_callback.summary()
                if not selected or not selected.get("checkpoint_dir") or self.monitor is None:
                    raise ValueError("GRPO has no selected checkpoint or step-zero baseline")
                checkpoint = Path(selected["checkpoint_dir"])
                aggregate_path = checkpoint / "aggregate_metrics.json"
                adapter_path = checkpoint / "adapter_model.safetensors"
                check_cohort(self.baseline_path, aggregate_path)
                decision = self.monitor.promotion(read_metrics(aggregate_path))
                gate = {"schema_version": 1, **decision, "selected_step": int(selected["step"]),
                        "training_config_sha256": file_sha256(config_path),
                        "selected_adapter_sha256": file_sha256(adapter_path),
                        "selected_aggregate_sha256": file_sha256(aggregate_path),
                        "baseline_path": str(self.baseline_path),
                        "baseline_sha256": file_sha256(self.baseline_path),
                        "scope": "development_screen_only; independent holdout and native QA still required"}
                path = output / "grpo_quality_gate.json"
                path.write_text(json.dumps(gate, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                return gate
            return on_main(write_gate)

    return QualityCallback()


def verify_quality_gate(output: Path, checkpoint: Path, config_path: Path) -> list[Path]:
    path = output / "grpo_quality_gate.json"
    gate = json.loads(path.read_text(encoding="utf-8"))
    baseline = Path(gate["baseline_path"])
    aggregate = checkpoint / "aggregate_metrics.json"
    bindings = ((config_path, "training_config_sha256"),
                (checkpoint / "adapter_model.safetensors", "selected_adapter_sha256"),
                (aggregate, "selected_aggregate_sha256"), (baseline, "baseline_sha256"))
    if any(file_sha256(file) != gate.get(key) for file, key in bindings):
        raise ValueError("GRPO quality gate does not bind the current checkpoint/config/evidence")
    from ir_training.common.config import load_yaml
    configured = load_yaml(config_path).get("grpo", {}).get("quality_control") or {}
    policy = QualityPolicy(**(configured.get("policy") or {}))
    if configured.get("enabled") is not True or asdict(policy) != gate.get("policy"):
        raise ValueError("GRPO quality gate policy differs from the bound training config")
    monitor = QualityMonitor(json.loads(baseline.read_text()), policy)
    actual = monitor.promotion(json.loads(aggregate.read_text()))
    if gate.get("passed") is not True or not actual["passed"]:
        raise ValueError("GRPO quality promotion blocked: " + ", ".join(actual["issues"]))
    return [path, baseline, aggregate]


def periodic_reward_audit(reward_fn: Any, output: Path, rank: int, every_steps: int):
    """Capture a complete first local batch per selected step AND train/eval mode.

    Unlike the startup-only 256-candidate audit, this covers late training. All
    ranks capture the same event, preserving groups split over devices. No new
    collectives, scorer changes, resampling, or reward-dependent selection.
    """
    if type(every_steps) is not int or every_steps < 0:
        raise ValueError("audit_every_steps must be a nonnegative integer")
    if every_steps == 0:
        return reward_fn
    seen = set()

    def reward(*args, **kwargs):
        step = getattr(kwargs.get("trainer_state"), "global_step", None)
        modes = kwargs.get("grpo_rollout_mode")
        if not isinstance(modes, list) or not modes or len(set(modes)) != 1 or modes[0] not in {"train", "eval"}:
            raise ValueError("Periodic GRPO audit requires an aligned train/eval rollout mode")
        mode = modes[0]
        key = (step, mode)
        if type(step) is not int or step % every_steps or key in seen:
            return reward_fn(*args, **kwargs)
        evidence = {}
        original = kwargs.get("log_extra")

        def capture(name, values):
            evidence[name] = values
            if callable(original):
                original(name, values)

        values = reward_fn(*args, **{**kwargs, "log_extra": capture})
        from ir_training.train.grpo_runtime import write_json_record
        completions = kwargs.get("completions", args[0] if args else [])
        prompts = kwargs.get("prompts") or [""] * len(values)
        sources = kwargs.get("source_id") or [None] * len(values)
        tokens = kwargs.get("completion_ids") or [None] * len(values)
        reasons = kwargs.get("termination_reason") or [None] * len(values)
        if not all(len(items) == len(values) for items in (completions, prompts, sources, tokens, reasons, modes)):
            raise ValueError("Periodic GRPO audit columns do not align")
        for index, value in enumerate(values):
            write_json_record(output / f"grpo_periodic_rewards.rank{rank}.jsonl", {
                "step": step, "mode": mode, "rank": rank, "local_index": index,
                "source_id": sources[index], "prompt_sha256": hashlib.sha256(str(prompts[index]).encode()).hexdigest(),
                "raw_completion": completions[index], "completion_ids": tokens[index],
                "termination_reason": reasons[index], "reward": value,
                "breakdown": {name: item[index] if isinstance(item, (tuple, list)) and len(item) == len(values)
                              else item for name, item in evidence.items()},
                "scope": "complete_first_local_batch_for_step_and_mode; not a full-run census",
            })
        seen.add(key)
        return values

    reward.__name__ = getattr(reward_fn, "__name__", "qat_reward")
    return reward
