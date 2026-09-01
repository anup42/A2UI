"""
Shared utilities for the QAT training scripts in this folder.

Used by:
    train_qat_lora.py  -- QAT + LoRA (pure-PyTorch STE fake quant on frozen base layers)
    train_qat_only.py  -- QAT only  (torchao 8da4w, all weights trainable)

Contents:
    1. Pure-PyTorch fake-quantization primitives (STE, group-wise INT4/INT8 weights,
       per-token dynamic INT8 activations) and forward patching that keeps modules as
       nn.Linear so PEFT/LoRA still accepts them.
    2. A version-tolerant shim for torchao's Int8DynActInt4WeightQATQuantizer.
    3. Trainer callbacks: QAT warmup, QAT diagnostics, generation-based evaluation
       (distributed) and the external heuristic-pipeline evaluation.
    4. Argparse blocks, model/tokenizer loading and SFTConfig construction shared by
       both trainers.

Nothing here depends on mslk, bitsandbytes or any custom kernel. torchao is imported
lazily and only by train_qat_only.py.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig

from metrics import MetricsAggregator

# Linear layers that get fake-quantized (attention + MLP projections).
QAT_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)


# ---------------------------------------------------------------------------
# 1. Fake-quantization primitives
# ---------------------------------------------------------------------------

def _qrange(bits: int):
    if bits == 4:
        return -8, 7
    if bits == 8:
        return -128, 127
    raise ValueError(f"Unsupported bit width: {bits}. Use 4 or 8.")


def fake_quant_weight(weight: torch.Tensor, bits: int = 4, group_size: int = 32) -> torch.Tensor:
    """Group-wise symmetric signed fake quantization of a weight tensor.

    A unique scale is computed for every `group_size` input features (the scheme used
    by AWQ / GPTQ / QA-LoRA), which isolates outliers instead of letting one channel
    dominate a per-tensor scale. If in_features is not divisible by group_size the
    leftover columns form a final per-channel group.

    A straight-through estimator is applied, so gradients bypass round()/clamp() and
    flow to the original floating-point weight. `weight` itself is never modified.
    """
    qmin, qmax = _qrange(bits)

    if weight.dim() < 2:  # bias / 1-D parameter: nothing to group over
        return weight

    out_features, in_features = weight.shape
    group_size = max(1, min(group_size, in_features))
    num_groups = in_features // group_size
    leftover = in_features % group_size

    def _quantize(block: torch.Tensor, groups: int) -> torch.Tensor:
        # block: (out_features, groups * width) -> reshape to (out_features, groups, width)
        reshaped = block.reshape(out_features, groups, -1)
        max_abs = reshaped.detach().abs().amax(dim=-1, keepdim=True)
        # Guard zero / non-finite scales without a host synchronisation.
        max_abs = torch.where(
            torch.isfinite(max_abs) & (max_abs > 0), max_abs, torch.ones_like(max_abs)
        )
        scale = max_abs / float(qmax)
        q = torch.clamp(torch.round(reshaped / scale), qmin, qmax)
        return (q * scale).reshape(out_features, -1)

    parts = []
    if num_groups > 0:
        parts.append(_quantize(weight[:, : num_groups * group_size], num_groups))
    if leftover > 0:
        parts.append(_quantize(weight[:, num_groups * group_size:], 1))

    weight_fake = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
    # Straight-through estimator.
    return weight + (weight_fake - weight).detach()


def fake_quant_activation(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Per-token symmetric signed fake quantization of an activation tensor.

    The scale is computed per token (over the last dim) because LLM hidden states carry
    large outliers (attention sinks) that would collapse a per-tensor scale.

    `bits >= 32` means "no activation quantization" (weight-only mode) and returns x
    unchanged. Non-finite entries pass through untouched. All guards are tensor ops, so
    this never forces a device->host sync in the training loop.
    """
    if bits >= 32:
        return x

    qmin, qmax = _qrange(bits)

    max_abs = x.detach().abs().amax(dim=-1, keepdim=True)
    max_abs = torch.where(
        torch.isfinite(max_abs) & (max_abs > 0), max_abs, torch.ones_like(max_abs)
    )
    scale = max_abs / float(qmax)

    q = torch.clamp(torch.round(x / scale), qmin, qmax)
    x_fake = torch.where(torch.isfinite(x), q * scale, x)
    # Straight-through estimator.
    return x + (x_fake - x).detach()


def _fake_quant_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Replacement forward for a patched nn.Linear.

    Reads `_qat_enabled` off the module on every call, so the warmup callback can turn
    fake quantization on mid-training without re-patching anything.
    """
    if not getattr(self, "_qat_enabled", False):
        return self._original_forward(x)

    weight_fake = fake_quant_weight(
        self.weight,
        bits=self._fake_quant_weight_bits,
        group_size=self._fake_quant_group_size,
    )
    x_fake = fake_quant_activation(x, bits=self._fake_quant_activation_bits)
    return F.linear(x_fake, weight_fake, self.bias)


def _name_matches(name: str, targets) -> bool:
    """True if any dotted component of `name` is in `targets`."""
    return any(part in targets for part in name.split("."))


def _find_linear(module: nn.Module):
    """Resolve the real `nn.Linear` inside `module`, and the attribute path to it.

    Handles three cases:
      * `module` is itself an `nn.Linear` -- returns (module, "").
      * `module` is a PEFT-wrapped layer with `.base_layer` -- returns (base_layer, "base_layer").
      * `module` is an architecture-specific wrapper that holds the real Linear as a
        `.linear` child (e.g. Gemma's `Gemma4ClippableLinear`, used for weight
        clipping) -- returns (inner_linear, "linear").

    Returns (None, None) if `module` is none of the above. Callers that only need the
    tensor (`apply_fake_quant`, `export_int4_weights`) use the returned linear directly;
    `resolve_target_modules` uses the attribute path to build a dotted name PEFT can
    target -- handing PEFT a bare wrapper name (e.g. "q_proj" when q_proj is actually a
    Gemma4ClippableLinear) makes PEFT's own model-wide suffix match pick the wrapper
    itself and crash with "Target module ... is not supported".
    """
    if isinstance(module, nn.Linear):
        return module, ""
    base_layer = getattr(module, "base_layer", None)
    if isinstance(base_layer, nn.Linear):
        return base_layer, "base_layer"
    inner = getattr(module, "linear", None)
    if isinstance(inner, nn.Linear):
        return inner, "linear"
    return None, None


def apply_fake_quant(
    model: nn.Module,
    target_modules=QAT_TARGET_MODULES,
    weight_bits: int = 4,
    activation_bits: int = 8,
    group_size: int = 32,
) -> int:
    """Patch the forward of every target nn.Linear to apply fake quantization.

    Works for plain and PEFT models:
      * For a LoRA-wrapped layer the `base_layer` (the real nn.Linear) is patched, so
        fake quant runs on the frozen base weight while the LoRA A/B deltas stay in
        full precision -- the QA-LoRA setup.
      * Target linears with no adapter are patched directly, so the whole model is
        quantization-simulated even when LoRA only covers a subset of layers.

    Modules keep their nn.Linear type, so PEFT can still wrap them. Patching is
    idempotent (guarded by `_original_forward`), which matters because DDP and
    gradient checkpointing can walk the module tree more than once -- double patching
    would nest fake_quant(fake_quant(...)).

    Fake quant starts DISABLED; flip it on with `set_fake_quant_enabled` (the warmup
    callback does this). Returns the number of patched layers.
    """
    targets = {target_modules} if isinstance(target_modules, str) else set(target_modules)
    patched = 0

    for name, module in model.named_modules():
        # Resolves PEFT's `.base_layer` and architecture wrappers like Gemma's
        # Gemma4ClippableLinear (`.linear`) to the real nn.Linear underneath.
        linear, _ = _find_linear(module)
        if linear is None:
            continue
        if not _name_matches(name, targets):
            continue
        if hasattr(linear, "_original_forward"):  # already patched
            continue

        linear._original_forward = linear.forward
        linear.forward = _fake_quant_forward.__get__(linear, nn.Linear)
        linear._fake_quant_patched = True
        linear._fake_quant_weight_bits = weight_bits
        linear._fake_quant_activation_bits = activation_bits
        linear._fake_quant_group_size = group_size
        linear._qat_enabled = False
        patched += 1

    act = "off (weight-only)" if activation_bits >= 32 else f"INT{activation_bits}"
    print(
        f"  [QAT] Patched {patched} nn.Linear layers -> INT{weight_bits} weights "
        f"(group_size={group_size}) / activations {act}; initially DISABLED"
    )
    return patched


def set_fake_quant_enabled(model: nn.Module, enabled: bool) -> int:
    """Enable/disable fake quantization on all patched layers. Returns the count."""
    count = 0
    for module in model.modules():
        if getattr(module, "_fake_quant_patched", False):
            module._qat_enabled = enabled
            count += 1
    return count


def restore_original_forward(model: nn.Module) -> nn.Module:
    """Undo `apply_fake_quant` (useful before export or full-precision inference)."""
    for module in model.modules():
        if getattr(module, "_fake_quant_patched", False):
            module.forward = module._original_forward
            del module._original_forward, module._fake_quant_patched
    return model


@torch.no_grad()
def export_int4_weights(
    model: nn.Module,
    target_modules=QAT_TARGET_MODULES,
    group_size: int = 32,
    output_path: str | None = None,
) -> dict:
    """Quantize target layers to real INT4 tensors (int8 storage + per-group scales).

    Training always uses fake quantization; this is the post-training export step. The
    grouping is identical to `fake_quant_weight`, including the leftover per-channel
    group, so `q * scale` reproduces exactly what training simulated.

    Returns {module_name: {"q", "scale", "group_size", "shape"}}.
    """
    targets = {target_modules} if isinstance(target_modules, str) else set(target_modules)
    qmin, qmax = _qrange(4)
    exported: dict = {}

    for name, module in model.named_modules():
        linear, _ = _find_linear(module)
        if linear is None or not _name_matches(name, targets):
            continue
        if name in exported:
            continue

        w = linear.weight.detach().float()
        out_features, in_features = w.shape
        gs = max(1, min(group_size, in_features))
        num_groups, leftover = in_features // gs, in_features % gs

        q_parts, scale_parts = [], []
        for block, groups in (
            (w[:, : num_groups * gs], num_groups),
            (w[:, num_groups * gs:], 1 if leftover else 0),
        ):
            if groups == 0:
                continue
            reshaped = block.reshape(out_features, groups, -1)
            max_abs = reshaped.abs().amax(dim=-1, keepdim=True)
            max_abs = torch.where(
                torch.isfinite(max_abs) & (max_abs > 0), max_abs, torch.ones_like(max_abs)
            )
            scale = max_abs / float(qmax)
            q = torch.clamp(torch.round(reshaped / scale), qmin, qmax)
            q_parts.append(q.reshape(out_features, -1).to(torch.int8))
            scale_parts.append(scale.squeeze(-1))

        exported[name] = {
            "q": torch.cat(q_parts, dim=1).cpu(),
            "scale": torch.cat(scale_parts, dim=1).cpu(),
            "group_size": gs,
            "shape": tuple(linear.weight.shape),
        }

    print(f"  [QAT] Exported INT4 weights for {len(exported)} layers (group_size={group_size}).")
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        torch.save(exported, output_path)
        print(f"  [QAT] Saved INT4 weights to: {output_path}")
    return exported


# ---------------------------------------------------------------------------
# 2. torchao backend shim
# ---------------------------------------------------------------------------

def load_torchao_qat_quantizer(group_size: int = 32, precision=None):
    """Build torchao's Int8DynActInt4WeightQATQuantizer (8da4w).

    The import path moved between torchao releases, so both the current and the
    legacy prototype locations are probed. Constructor kwargs are also version
    dependent, so they are applied best-effort.
    """
    quantizer_cls = None
    errors = []
    for module_path in ("torchao.quantization.qat", "torchao.quantization.prototype.qat"):
        try:
            module = __import__(module_path, fromlist=["Int8DynActInt4WeightQATQuantizer"])
            quantizer_cls = getattr(module, "Int8DynActInt4WeightQATQuantizer")
            print(f"  [QAT] Using torchao quantizer from {module_path}")
            break
        except Exception as exc:  # ImportError or AttributeError
            errors.append(f"{module_path}: {exc}")

    if quantizer_cls is None:
        raise ImportError(
            "Could not import Int8DynActInt4WeightQATQuantizer from torchao.\n"
            "Install it with:  pip install torchao\n"
            "Probe results:\n  " + "\n  ".join(errors)
        )

    kwargs = {"groupsize": group_size, "padding_allowed": True}
    if precision is not None:
        kwargs["precision"] = precision
        kwargs["scales_precision"] = precision

    while True:
        try:
            return quantizer_cls(**kwargs)
        except TypeError as exc:
            # Drop unsupported kwargs one at a time rather than failing outright.
            dropped = None
            for key in ("scales_precision", "precision", "padding_allowed", "groupsize"):
                if key in kwargs:
                    dropped = key
                    break
            if dropped is None:
                raise
            print(f"  [QAT] torchao quantizer rejected kwarg '{dropped}' ({exc}); retrying without it")
            kwargs.pop(dropped)


def set_torchao_fake_quant_enabled(model: nn.Module, enabled: bool) -> int:
    """Best-effort toggle of torchao fake quantizers (for warmup).

    Newer torchao exposes `.enabled` on its fake-quantizer submodules; older releases
    do not. Returns how many toggles were applied so the caller can warn instead of
    silently pretending warmup worked.
    """
    count = 0
    for name, module in model.named_modules():
        if "fake_quantizer" in name.split(".")[-1] and hasattr(module, "enabled"):
            module.enabled = enabled
            count += 1
    return count


# ---------------------------------------------------------------------------
# 3. Callbacks
# ---------------------------------------------------------------------------

def _dist_info():
    """(is_distributed, world_size, rank)."""
    if dist.is_available() and dist.is_initialized():
        return True, dist.get_world_size(), dist.get_rank()
    return False, 1, 0


def _unwrap(model):
    """Strip DDP/FSDP wrappers."""
    return getattr(model, "module", model)


def _save_checkpoint(model, tokenizer, dest: str, info: dict | None = None,
                     info_name: str = "info.json", copy_files=()):
    """Save a model (PEFT adapter or full model) + tokenizer into `dest`."""
    os.makedirs(dest, exist_ok=True)
    _unwrap(model).save_pretrained(dest)
    tokenizer.save_pretrained(dest)
    for src in copy_files:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, os.path.basename(src)))
    if info is not None:
        with open(os.path.join(dest, info_name), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, default=str)
    return dest


class QATWarmupCallback(TrainerCallback):
    """Train in bf16 for `warmup_steps`, then switch fake quantization on.

    A short full-precision warmup lets the optimizer settle before the extra STE noise
    arrives, which is the usual recipe for stable low-bit QAT. `warmup_steps=0` means
    quantization is on from step 1.
    """

    def __init__(self, warmup_steps: int, backend: str = "torch", output_dir: str | None = None):
        self.warmup_steps = warmup_steps
        self.backend = backend
        self.output_dir = output_dir
        self.enabled = False

    def _set(self, model, enabled: bool) -> int:
        if self.backend == "torchao":
            return set_torchao_fake_quant_enabled(model, enabled)
        return set_fake_quant_enabled(model, enabled)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.warmup_steps <= 0:
            count = self._set(model, True)
            self.enabled = True
            print(f"\n[QAT] Fake quantization ACTIVE from step 0 ({count} layers).")
        else:
            self._set(model, False)
            print(f"\n[QAT] Full-precision warmup for {self.warmup_steps} steps, "
                  f"then fake quantization turns on.")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.enabled or state.global_step < self.warmup_steps:
            return
        count = self._set(model, True)
        self.enabled = True
        print(f"\n[QAT] Warmup finished at step {state.global_step}: "
              f"fake quantization ENABLED on {count} layers.")
        if self.output_dir and state.is_world_process_zero:
            writer = SummaryWriter(log_dir=self.output_dir)
            writer.add_scalar("qat/enabled", 1.0, state.global_step)
            writer.close()


class QATDiagnosticsCallback(TrainerCallback):
    """Periodic sanity checks that QAT is actually doing what it claims.

    Logs to TensorBoard every `interval` steps (rank 0 only):
      qat/fake_quant_enabled       -- 1 once quantization is live
      qat/probe_weight_changed     -- did the watched layer's weight move? Must stay 0
                                      in the LoRA path (PEFT freezes the base) and
                                      should become 1 in the full-finetune path.
      qat/trainable_weight_drift   -- L1 drift of all trainable params from step 0; if
                                      this stays flat, nothing is learning.
      qat/fake_quant_unique_values -- distinct levels in the quantized probe weight;
                                      <= 2 means it has collapsed. Pure-torch backend
                                      only (with torchao the quantizer owns the math).

    `expect_frozen_base=True` (the LoRA path) turns a moving probe weight into a warning.
    All reductions happen on-device and are read back with a single .item() per step, so
    the diagnostics do not add a host sync per parameter.
    """

    def __init__(self, interval: int = 50, output_dir: str | None = None,
                 backend: str = "torch", expect_frozen_base: bool = True):
        self.interval = max(1, interval)
        self.output_dir = output_dir
        self.backend = backend
        self.expect_frozen_base = expect_frozen_base
        self._probe_name = None
        self._probe_checksum = None
        self._trainable_checksum = None

    def _probe(self, model):
        """Pick one layer to watch: a fake-quant-patched one if there is any."""
        fallback = (None, None)
        for name, module in _unwrap(model).named_modules():
            if getattr(module, "_fake_quant_patched", False):
                return name, module
            if fallback[0] is None and isinstance(module, nn.Linear) \
                    and _name_matches(name, set(QAT_TARGET_MODULES)):
                fallback = (name, module)
        return fallback

    def _quant_active(self, model, probe) -> float:
        if self.backend == "torchao":
            for name, module in _unwrap(model).named_modules():
                if "fake_quantizer" in name.split(".")[-1] and hasattr(module, "enabled"):
                    return float(bool(module.enabled))
            return float("nan")  # this build offers no introspectable flag
        return float(getattr(probe, "_qat_enabled", False))

    @torch.no_grad()
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.interval != 0 or not state.is_world_process_zero:
            return

        name, probe = self._probe(model)
        if probe is None:
            return
        if self._probe_name is None:
            self._probe_name = name

        # One device-side reduction for the probe and one for all trainable params,
        # stacked so the whole diagnostic costs a single host sync.
        probe_sum = probe.weight.detach().float().abs().sum()
        trainable = [p.detach().float().abs().sum()
                     for p in _unwrap(model).parameters() if p.requires_grad]
        trainable_sum = (torch.stack(trainable).sum() if trainable
                         else torch.zeros((), device=probe_sum.device))
        probe_val, trainable_val = torch.stack([probe_sum, trainable_sum]).tolist()

        first_call = self._probe_checksum is None
        if first_call:
            self._probe_checksum = probe_val
            self._trainable_checksum = trainable_val
        probe_changed = float(abs(probe_val - self._probe_checksum) > 1e-6)
        drift = abs(trainable_val - self._trainable_checksum)

        enabled = self._quant_active(model, probe)
        unique = float("nan")
        if self.backend == "torch" and enabled == 1.0:
            w_fake = fake_quant_weight(
                probe.weight.detach(),
                bits=probe._fake_quant_weight_bits,
                group_size=probe._fake_quant_group_size,
            )
            unique = float(w_fake.unique().numel())

        print(f"\n[QAT diag step {state.global_step}] probe={self._probe_name} "
              f"quant={'on' if enabled == 1.0 else 'off' if enabled == 0.0 else 'unknown'} "
              f"probe_weight_changed={probe_changed:.0f} trainable_drift={drift:.4f} "
              f"unique_levels={'n/a' if unique != unique else f'{unique:.0f}'}")
        if unique == unique and unique <= 2:
            print("  WARNING: fake-quantized weight collapsed to <= 2 unique values.")
        if probe_changed and self.expect_frozen_base:
            print("  WARNING: the base weight moved, but PEFT should have frozen it.")
        if not probe_changed and not self.expect_frozen_base and not first_call:
            print("  WARNING: the probe weight has not moved in a full-finetune run.")

        if self.output_dir:
            writer = SummaryWriter(log_dir=self.output_dir)
            if enabled == enabled:
                writer.add_scalar("qat/fake_quant_enabled", enabled, state.global_step)
            writer.add_scalar("qat/probe_weight_changed", probe_changed, state.global_step)
            writer.add_scalar("qat/trainable_weight_drift", drift, state.global_step)
            if unique == unique:
                writer.add_scalar("qat/fake_quant_unique_values", unique, state.global_step)
            writer.close()


class EvalPredictionCallback(TrainerCallback):
    """Generation-based evaluation, matching train_gemma.py / train_qwen.py.

    On every evaluation it greedily generates predictions (sharded round-robin across
    ranks, gathered and reordered), scores them with MetricsAggregator, writes
    predictions/metrics JSON and TensorBoard scalars, and maintains two checkpoints:
      best_test_checkpoint/     -- best `metric_for_best_model` on the test set
      best_val_loss_checkpoint/ -- lowest eval_loss

    When `test_data` is given it is used in full; otherwise the eval split is sampled
    up to `max_prediction_samples`.
    """

    def __init__(self, tokenizer, input_field, output_field, max_new_tokens=512,
                 max_length=1024, prediction_interval=1, max_prediction_samples=50,
                 test_data=None, metric_for_best_model="mean_direct_match_score",
                 output_dir=None, save_predictions=True, save_predictions_interval=1,
                 save_predictions_limit=0):
        self.tokenizer = tokenizer
        self.input_field = input_field
        self.output_field = output_field
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        self.prediction_interval = max(1, prediction_interval)
        self.max_prediction_samples = max_prediction_samples
        self.test_data = test_data
        self.metric_for_best_model = metric_for_best_model
        self.output_dir = output_dir
        self.metrics_aggregator = MetricsAggregator()

        # Step-archived prediction history. The rolling {prefix}_predictions.json is
        # always written (the heuristic callback reads it); these are additional
        # per-step copies under predictions/ so earlier evals are not lost.
        self.save_predictions = save_predictions
        self.save_predictions_interval = max(1, save_predictions_interval)
        self.save_predictions_limit = save_predictions_limit
        self._archived_predictions = []  # oldest-first, for --save_predictions_limit
        self._eval_counter = 0

        self.best_metric_value = -float("inf")
        self.best_metric_step = -1
        self.best_model_saved = False
        self.best_val_loss = float("inf")
        self.best_val_loss_step = -1
        self.best_val_loss_model_saved = False

        # Set post-construction (trainer = SFTTrainer(...); callback.trainer = trainer)
        # so on_evaluate can force a checkpoint save BEFORE running generation. See
        # on_evaluate for why: HF's Trainer._maybe_log_save_evaluate always evaluates
        # before saving, so if generation OOMs, that step's checkpoint never happens.
        self.trainer = None

    def _archive_predictions(self, prefix, predictions_data, batch, step):
        """Write a per-step copy of the predictions so eval history is kept.

        The rolling `{prefix}_predictions.json` is overwritten on every eval, so without
        this only the latest predictions survive a run. Files land in
        `output_dir/predictions/{prefix}_predictions_step{N}.json` and each carries its
        own metrics header, making a single file self-describing.

        Rank-0 only -- the caller already guards on is_world_process_zero.
        """
        if not self.save_predictions:
            return

        self._eval_counter += 1
        if self._eval_counter % self.save_predictions_interval != 0:
            return

        pred_dir = os.path.join(self.output_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        path = os.path.join(pred_dir, f"{prefix}_predictions_step{step}.json")

        payload = {
            "step": step,
            "source": prefix,
            "num_samples": batch["num_samples"],
            "metrics": {
                "exact_match_accuracy": batch["exact_match_accuracy"],
                "valid_prediction_json_ratio": batch["valid_prediction_json_ratio"],
                "mean_direct_match_score": batch["mean_direct_match_score"],
                "mean_value_only_match_score": batch["mean_value_only_match_score"],
            },
            "predictions": predictions_data,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            # Never let an archive write kill a training run.
            print(f"WARNING: could not archive predictions to {path}: {exc}")
            return

        print(f"  Predictions archived: {path}")
        self._archived_predictions.append(path)
        self._prune_archived_predictions()

    def _prune_archived_predictions(self):
        """Keep only the newest `save_predictions_limit` archives (0 = keep all).

        Prediction files hold full generated JSON for up to max_prediction_samples rows,
        so a long run with frequent evals can add up; this bounds the footprint.
        """
        limit = self.save_predictions_limit
        if limit <= 0:
            return
        while len(self._archived_predictions) > limit:
            stale = self._archived_predictions.pop(0)
            try:
                os.remove(stale)
            except OSError:
                pass

    def _generate_local(self, model, data_source, num_samples, world_size, rank):
        """Generate on this rank's shard: indices rank, rank+ws, rank+2*ws, ..."""
        local_indices = list(range(rank, num_samples, world_size))
        results = []

        if rank == 0:
            print(f"\nGenerating predictions on {num_samples} samples "
                  f"across {world_size} process(es)...")

        for local_idx, global_idx in enumerate(local_indices):
            example = data_source[global_idx]
            input_text = example.get(self.input_field, "")
            reference = example.get(self.output_field, "")
            if not input_text:
                continue

            messages = [{"role": "user", "content": input_text}]
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt = f"<|im_start|>user\n{input_text}\n<|im_start|>assistant\n"

            # Gemma-4's processor is multi-modal: __call__(images=None, text=None, ...),
            # with `images` first positionally. Passing prompt positionally lands it in
            # `images`, leaving text=None -> "'NoneType' object is not subscriptable" when
            # the processor does text[0]. text= as a keyword is unambiguous for both plain
            # tokenizers (gemma-3) and multi-modal processors (gemma-4).
            encoded = self.tokenizer(
                text=prompt, return_tensors="pt", truncation=True, max_length=self.max_length,
            ).to(model.device)

            with torch.no_grad():
                output = model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            input_len = int(encoded["attention_mask"].sum().item())
            prediction = self.tokenizer.decode(
                output[0][input_len:], skip_special_tokens=True
            ).strip()

            results.append({
                "input": input_text,
                "reference": reference,
                "prediction": prediction,
                "global_idx": global_idx,
            })

            if (local_idx + 1) % 5 == 0 or (local_idx + 1) == len(local_indices):
                print(f"  Rank {rank}: {local_idx + 1}/{len(local_indices)} samples")

        return results

    def _gather(self, local_results, device, is_distributed, world_size):
        """Gather per-rank results, reorder by original index, drop the index."""
        if not is_distributed:
            gathered = list(local_results)
        else:
            payload = [json.dumps(p) for p in local_results]
            local_count = torch.tensor([len(payload)], device=device, dtype=torch.long)
            counts = [torch.zeros_like(local_count) for _ in range(world_size)]
            dist.all_gather(counts, local_count)
            max_count = max(int(c.item()) for c in counts)
            padded = payload + [""] * (max_count - len(payload))

            all_padded = [None] * world_size
            dist.all_gather_object(all_padded, padded)

            gathered = [
                json.loads(item)
                for proc_list in all_padded
                for item in (proc_list or [])
                if isinstance(item, str) and item
            ]

        gathered.sort(key=lambda p: p["global_idx"])
        for p in gathered:
            p.pop("global_idx", None)
        return gathered

    def _score_and_save(self, state, model, data_source, num_samples, prefix,
                        is_distributed, world_size, rank):
        local = self._generate_local(model, data_source, num_samples, world_size, rank)
        predictions_data = self._gather(local, model.device, is_distributed, world_size)

        if rank != 0:
            return None

        references = [p["reference"] for p in predictions_data]
        predictions = [p["prediction"] for p in predictions_data]
        batch = self.metrics_aggregator.compute_batch_metrics(references, predictions)

        print(f"\n{prefix.capitalize()} prediction metrics (step {state.global_step}):")
        print(f"  Exact Match:      {batch['exact_match_accuracy']:.4f}")
        print(f"  Valid JSON:       {batch['valid_prediction_json_ratio']:.4f}")
        print(f"  Direct Match:     {batch['mean_direct_match_score']:.4f}")
        print(f"  Value-Only Match: {batch['mean_value_only_match_score']:.4f}")

        with open(os.path.join(self.output_dir, f"{prefix}_predictions.json"), "w",
                  encoding="utf-8") as f:
            json.dump(predictions_data, f, indent=2, ensure_ascii=False)

        self._archive_predictions(prefix, predictions_data, batch, state.global_step)

        summary = {
            "step": state.global_step,
            "num_samples": batch["num_samples"],
            "exact_match_accuracy": batch["exact_match_accuracy"],
            "valid_prediction_json_ratio": batch["valid_prediction_json_ratio"],
            "mean_direct_match_score": batch["mean_direct_match_score"],
            "std_direct_match_score": batch["std_direct_match_score"],
            "mean_value_only_match_score": batch["mean_value_only_match_score"],
            "std_value_only_match_score": batch["std_value_only_match_score"],
        }
        with open(os.path.join(self.output_dir, f"{prefix}_metrics.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        writer = SummaryWriter(log_dir=self.output_dir)
        for tag, key in (
            ("exact_match_accuracy", "exact_match_accuracy"),
            ("valid_json_ratio", "valid_prediction_json_ratio"),
            ("direct_match_score", "mean_direct_match_score"),
            ("value_only_match_score", "mean_value_only_match_score"),
            ("direct_match_std", "std_direct_match_score"),
            ("value_only_match_std", "std_value_only_match_score"),
        ):
            writer.add_scalar(f"{prefix}/{tag}", batch[key], state.global_step)
        writer.close()

        return summary

    def on_evaluate(self, args, state, control, model=None, eval_dataloader=None,
                    metrics=None, **kwargs):
        is_distributed, world_size, rank = _dist_info()

        # ---- save checkpoint FIRST, before the generation loop below ----
        # HF's Trainer._maybe_log_save_evaluate calls evaluate() (which fires this
        # on_evaluate) BEFORE checking control.should_save and calling
        # _save_checkpoint -- unconditionally, hardcoded in transformers' Trainer.
        # But control.should_save is already computed by this point (set at the
        # step boundary, earlier in the training loop). If this step would save
        # anyway, do it NOW, before the OOM-prone generation loop below, so a
        # generation-time OOM doesn't lose this step's checkpoint. Then suppress
        # HF's own later save to avoid a redundant duplicate write.
        if control.should_save and self.trainer is not None:
            if rank == 0:
                print(f"\n  [checkpoint-first] Saving checkpoint for step {state.global_step} "
                      f"before generation-based eval (avoids losing it if generation OOMs)")
            self.trainer._save_checkpoint(model, trial=None)
            control.should_save = False

        # ---- best checkpoint by validation loss ----
        if metrics and "eval_loss" in metrics and rank == 0:
            val_loss = metrics["eval_loss"]
            if val_loss < self.best_val_loss:
                self.best_val_loss, self.best_val_loss_step = val_loss, state.global_step
                print(f"\n  *** New best eval_loss: {val_loss:.4f} at step {state.global_step} ***")
                _save_checkpoint(
                    model, self.tokenizer,
                    os.path.join(self.output_dir, "best_val_loss_checkpoint"),
                    info={"best_val_loss": val_loss, "best_step": state.global_step,
                          "all_metrics": metrics},
                    info_name="best_val_loss_info.json",
                )
                self.best_val_loss_model_saved = True
            else:
                print(f"\n  eval_loss: {val_loss:.4f} "
                      f"(best {self.best_val_loss:.4f} @ step {self.best_val_loss_step})")

        # ---- generation-based metrics, every `prediction_interval` evaluations ----
        eval_index = state.global_step // max(1, args.eval_steps or 1)
        if eval_index % self.prediction_interval != 0:
            if is_distributed:
                dist.barrier()
            return

        if self.test_data:
            data_source, num_samples, prefix = self.test_data, len(self.test_data), "test"
        elif eval_dataloader is not None:
            data_source = eval_dataloader.dataset
            num_samples = min(len(data_source), self.max_prediction_samples)
            prefix = "eval"
        else:
            if is_distributed:
                dist.barrier()
            return

        was_training = model.training
        model.eval()
        summary = self._score_and_save(
            state, model, data_source, num_samples, prefix,
            is_distributed, world_size, rank,
        )
        if was_training:
            model.train()

        # ---- best checkpoint by generation metric (test mode only) ----
        if prefix == "test" and summary and rank == 0:
            value = summary.get(self.metric_for_best_model, 0.0)
            if value > self.best_metric_value:
                self.best_metric_value, self.best_metric_step = value, state.global_step
                print(f"  *** New best {self.metric_for_best_model}: {value:.4f} "
                      f"at step {state.global_step} ***")
                _save_checkpoint(
                    model, self.tokenizer,
                    os.path.join(self.output_dir, "best_test_checkpoint"),
                    info={"best_metric": self.metric_for_best_model,
                          "best_metric_value": value, "best_step": state.global_step,
                          "all_metrics": summary},
                    info_name="best_metric_info.json",
                    copy_files=[os.path.join(self.output_dir, "test_predictions.json"),
                                os.path.join(self.output_dir, "test_metrics.json")],
                )
                self.best_model_saved = True
            else:
                print(f"  {self.metric_for_best_model}: {value:.4f} "
                      f"(best {self.best_metric_value:.4f} @ step {self.best_metric_step})")

        if is_distributed:
            dist.barrier()


class HeuristicEvaluationCallback(TrainerCallback):
    """Run the external GenUI heuristic pipeline on the generated predictions.

    Rank 0 only, and deliberately WITHOUT a barrier: the pipeline is a slow subprocess
    and the other ranks must keep training rather than sit in a collective until NCCL
    times out. Saves best_heuristic_checkpoint/ when overall_score improves.

    Self-disables when the util folder or pipeline script is missing.
    """

    UTIL_FILES = ("queries.jsonl", "responses.jsonl", "golden50_url_mapping.jsonl")

    def __init__(self, tokenizer, util_folder=None, config_path=None, heuristic_interval=1,
                 output_dir=None, script_dir=None):
        self.tokenizer = tokenizer
        self.util_folder = Path(util_folder) if util_folder else None
        self.config_path = Path(config_path) if config_path else None
        self.heuristic_interval = max(1, heuristic_interval)
        self.output_dir = Path(output_dir) if output_dir else None
        self.script_dir = Path(script_dir) if script_dir else None

        self.eval_count = 0
        self.best_heuristic_score = -float("inf")
        self.best_heuristic_step = -1
        self.best_heuristic_model_saved = False

        if self.util_folder and not self.util_folder.exists():
            print(f"WARNING: heuristic util folder not found, disabling: {self.util_folder}")
            self.util_folder = None
        if self.script_dir and not (self.script_dir / "run_heuristic_pipeline.py").exists():
            print(f"WARNING: run_heuristic_pipeline.py not found in {self.script_dir}, "
                  f"disabling heuristic evaluation")
            self.script_dir = None

    @property
    def is_enabled(self) -> bool:
        return bool(self.util_folder and self.script_dir and self.output_dir)

    def _run_pipeline(self, work_dir: Path, step: int):
        cmd = [
            sys.executable, str(self.script_dir / "run_heuristic_pipeline.py"),
            "--input", str((work_dir / "predictions.jsonl").resolve()),
            "--util-folder", str(self.util_folder),
        ]
        if self.config_path and self.config_path.exists():
            cmd += ["--config", str(self.config_path)]

        print(f"\n{'=' * 60}\nRunning heuristic pipeline at step {step}\n"
              f"  {' '.join(cmd)}\n{'=' * 60}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    cwd=str(self.script_dir))
        except Exception as exc:
            print(f"ERROR: could not launch heuristic pipeline: {exc}")
            return None

        if result.returncode != 0:
            print(f"ERROR: heuristic pipeline exited {result.returncode}")
            print(f"STDOUT:\n{result.stdout[-4000:]}")
            print(f"STDERR:\n{result.stderr[-4000:]}")
            return None

        aggregates_path = work_dir / "aggregates.json"
        if not aggregates_path.exists():
            print(f"WARNING: aggregates.json not produced at {aggregates_path}")
            return None

        with open(aggregates_path, "r", encoding="utf-8") as f:
            aggregates = json.load(f)
        shutil.copy2(aggregates_path, self.output_dir / f"aggregate_step_{step}.json")
        return aggregates

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if not self.is_enabled:
            return
        _, _, rank = _dist_info()
        if rank != 0:
            return

        eval_index = state.global_step // max(1, args.eval_steps or 1)
        if eval_index % self.heuristic_interval != 0:
            return

        predictions_file = self.output_dir / "test_predictions.json"
        if not predictions_file.exists():
            predictions_file = self.output_dir / "eval_predictions.json"
        if not predictions_file.exists():
            print(f"WARNING: no predictions file in {self.output_dir}, skipping heuristic eval")
            return

        self.eval_count += 1
        print(f"\n{'#' * 60}\n# Heuristic evaluation #{self.eval_count} at step "
              f"{state.global_step} (rank 0 only, other ranks keep training)\n{'#' * 60}")

        with open(predictions_file, "r", encoding="utf-8") as f:
            predictions = json.load(f)

        work_dir = self.output_dir / f"heuristic_eval_step_{state.global_step}"
        work_dir.mkdir(parents=True, exist_ok=True)
        with open(work_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps({"prediction": pred.get("prediction", "")},
                                   ensure_ascii=False) + "\n")
        for filename in self.UTIL_FILES:
            source = self.util_folder / filename
            if source.exists():
                shutil.copy2(source, work_dir / filename)
            else:
                print(f"  WARNING: util file missing: {source}")

        aggregates = self._run_pipeline(work_dir, state.global_step)
        shutil.rmtree(work_dir, ignore_errors=True)

        if not aggregates:
            print(f"WARNING: heuristic evaluation failed at step {state.global_step}")
            return

        overall = aggregates.get("overall_score", 0.0)
        media = aggregates.get("media_score", 0.0)
        schema = aggregates.get("schema_valid_strict_rate", 0.0)
        coverage = aggregates.get("content_coverage_avg", 0.0)

        writer = SummaryWriter(log_dir=str(self.output_dir))
        writer.add_scalar("heuristic/overall_score", overall, state.global_step)
        writer.add_scalar("heuristic/media_score", media, state.global_step)
        writer.add_scalar("heuristic/schema_valid_strict_rate", schema, state.global_step)
        writer.add_scalar("heuristic/content_coverage_avg", coverage, state.global_step)
        writer.close()

        print(f"\nHeuristic metrics (step {state.global_step}):")
        print(f"  Overall Score:        {overall:.4f}")
        print(f"  Media Score:          {media:.4f}")
        print(f"  Schema Valid Rate:    {schema:.4f}")
        print(f"  Content Coverage Avg: {coverage:.4f}")

        if overall > self.best_heuristic_score:
            self.best_heuristic_score, self.best_heuristic_step = overall, state.global_step
            print(f"  *** New best overall_score: {overall:.4f} ***")
            source = self.output_dir / f"aggregate_step_{state.global_step}.json"
            if source.exists():
                shutil.copy2(source, self.output_dir / "best_heuristic_aggregates.json")
            _save_checkpoint(
                model, self.tokenizer,
                str(self.output_dir / "best_heuristic_checkpoint"),
                info={"best_overall_score": overall, "best_step": state.global_step,
                      "aggregates": aggregates},
                info_name="best_heuristic_info.json",
            )
            self.best_heuristic_model_saved = True


# ---------------------------------------------------------------------------
# 4. Shared plumbing
# ---------------------------------------------------------------------------

def select_gpus(argv=None) -> str:
    """Apply `--gpus a,b` to CUDA_VISIBLE_DEVICES.

    MUST be called before torch is imported, so each trainer script inlines a copy of
    this logic at module top. Kept here for reuse/testing.

    Skipped when a launcher (accelerate / torchrun) already owns device assignment --
    overriding CUDA_VISIBLE_DEVICES then breaks the rank -> device mapping.
    """
    argv = sys.argv if argv is None else argv
    gpus = ""
    for i, token in enumerate(argv):
        if token == "--gpus" and i + 1 < len(argv):
            gpus = argv[i + 1]
        elif token.startswith("--gpus="):
            gpus = token.split("=", 1)[1]
    if not gpus:
        return ""
    if "LOCAL_RANK" in os.environ or "CUDA_VISIBLE_DEVICES" in os.environ:
        print(f"[*] Ignoring --gpus '{gpus}': the launcher already assigned devices "
              f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}). "
              f"Set CUDA_VISIBLE_DEVICES before `accelerate launch` instead.")
        return ""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    print(f"[*] Set CUDA_VISIBLE_DEVICES to '{gpus}'")
    return gpus


def get_model_class(model_name: str):
    """Pick the model class from config.json. Returns (class, is_vl_model)."""
    config_path = os.path.join(model_name, "config.json")
    model_type = ""
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                model_type = json.load(f).get("model_type", "")
        except Exception as exc:
            print(f"WARNING: could not read {config_path}: {exc}")

    if model_type in {"qwen3_vl", "qwen2_vl", "qwen2_5_vl"}:
        from transformers import Qwen3VLForConditionalGeneration
        print(f"  Detected vision-language model (model_type={model_type})")
        return Qwen3VLForConditionalGeneration, True

    print(f"  Using AutoModelForCausalLM (model_type={model_type or 'unknown'})")
    return AutoModelForCausalLM, False


def load_tokenizer(model_name: str, allow_missing_chat_template: bool = False):
    """Load the tokenizer, mirroring train_gemma.py's Gemma special case.

    Gemma checkpoints are loaded with PreTrainedTokenizerFast to avoid the
    sentencepiece dependency. A missing chat template is a hard error by default: the
    dataloader silently falls back to a "User: ...\\nAssistant: ..." string, which no
    longer matches the generation-time prompt and quietly destroys the metrics.
    """
    if "gemma" in model_name.lower():
        from transformers import PreTrainedTokenizerFast
        print("  Gemma checkpoint detected, using PreTrainedTokenizerFast")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if not getattr(tokenizer, "chat_template", None):
        message = (
            f"Tokenizer for '{model_name}' has no chat_template. dataloader.format_example "
            f"would silently fall back to a plain 'User:/Assistant:' string that does not "
            f"match the prompt used at generation time. Add a chat_template to "
            f"tokenizer_config.json, or pass --allow_missing_chat_template to proceed anyway."
        )
        if not allow_missing_chat_template:
            raise ValueError(message)
        print(f"WARNING: {message}")

    return tokenizer


def load_model(model_name: str):
    """Load the model in bf16 (fp32 on CPU) with the cache disabled for training."""
    model_class, _ = get_model_class(model_name)
    model = model_class.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.config.use_cache = False
    return model


def resolve_target_modules(model, layers: str = "all", target_names=QAT_TARGET_MODULES):
    """Resolve `--lora_layers` into concrete module paths found on this model.

    `layers` is 'all', 'early', 'late', 'middle', or comma-separated indices ("28,29").
    Names are read off the model rather than guessed from a hardcoded path template, so
    this works for both:
      * plain nn.Linear architectures (Llama/Qwen-style `model.layers.0.self_attn.q_proj`)
      * wrapped-linear architectures (Gemma-style `layers.0.self_attn.q_proj`, which is a
        `Gemma4ClippableLinear` holding the real nn.Linear as a `.linear` child)

    Always returns FULL dotted paths (e.g. "...q_proj" or "...q_proj.linear"), never a
    bare suffix like "q_proj". PEFT matches target_modules by name suffix across the
    WHOLE model, ignoring type -- handing it a bare "q_proj" on a Gemma-style model
    matches the wrapper module itself (not the nn.Linear inside it) and PEFT then raises
    "Target module ... is not supported". Full paths route PEFT to the exact nn.Linear
    (or its wrapping `.linear` attribute) every time.
    """
    targets = set(target_names)
    found = []  # (config_path, layer_idx or None)

    for name, module in model.named_modules():
        if name.split(".")[-1] not in targets:
            continue
        linear, attr_path = _find_linear(module)
        if linear is None:
            continue
        config_path = name if not attr_path else f"{name}.{attr_path}"
        layer_match = re.search(r"layers\.(\d+)\.", name)
        found.append((config_path, int(layer_match.group(1)) if layer_match else None))

    if not found:
        raise ValueError(
            f"No linear modules matching {sorted(targets)} found in the model "
            f"(checked plain nn.Linear and wrapped '.linear' children)."
        )

    indexed = [(path, idx) for path, idx in found if idx is not None]
    if not indexed:
        # No layer index in the names (unusual architecture): adapt every match found.
        paths = sorted({path for path, _ in found})
        print(f"  Target modules (no layer index found, adapting all matches): {len(paths)}")
        return paths

    num_layers = max(idx for _, idx in indexed) + 1

    if layers == "all":
        chosen = set(range(num_layers))
    elif layers == "early":
        chosen = set(range(0, max(1, num_layers // 4)))
    elif layers == "late":
        chosen = set(range(num_layers - max(1, num_layers // 4), num_layers))
    elif layers == "middle":
        start = num_layers // 4
        chosen = set(range(start, num_layers - start))
    else:
        try:
            chosen = {int(part) for part in layers.split(",") if part.strip()}
        except ValueError:
            raise ValueError(
                f"--lora_layers must be 'all', 'early', 'late', 'middle', or comma-separated "
                f"layer indices; got '{layers}'"
            )
        unknown = {i for i in chosen if i >= num_layers}
        if unknown:
            raise ValueError(f"Layer indices {sorted(unknown)} out of range (0..{num_layers - 1})")

    selected = sorted({path for path, idx in indexed if idx in chosen})
    if not selected:
        raise ValueError(f"--lora_layers '{layers}' selected no modules")
    wrapped = any(path.endswith(".linear") for path in selected)
    print(f"  Target modules: {len(selected)} modules on layers {sorted(chosen)}"
          f"{' (wrapped .linear children)' if wrapped else ''}")
    return selected


def resolve_target_module_suffixes(model, target_names=QAT_TARGET_MODULES):
    """Resolve target modules as bare NAME SUFFIXES, for Unsloth's get_peft_model.

    Unsloth validates target_modules against its own list of adaptable layer names and
    rejects full dotted paths with "No layers to finetune?", so `resolve_target_modules`
    (which deliberately returns full paths for raw PEFT) cannot be used there.

    Returns short names like "q_proj", or "q_proj.linear" when the architecture wraps the
    real nn.Linear in a container (Gemma4ClippableLinear). PEFT's suffix match accepts
    both forms, so one code path covers every Gemma generation:
      * gemma-3 style: plain nn.Linear at ...self_attn.q_proj  -> "q_proj"
      * gemma-4 style: Gemma4ClippableLinear wrapper           -> "q_proj.linear"

    A model mixing both forms for the same name yields both suffixes, which is correct --
    each matches only the layers actually shaped that way.
    """
    targets = set(target_names)
    suffixes = set()

    for name, module in model.named_modules():
        short = name.split(".")[-1]
        if short not in targets:
            continue
        linear, attr_path = _find_linear(module)
        if linear is None:
            continue
        # attr_path is "" for a plain nn.Linear, "linear" for a wrapped one. "base_layer"
        # only appears on an already-PEFT-wrapped model, which is not the case here.
        suffixes.add(short if not attr_path else f"{short}.{attr_path}")

    if not suffixes:
        raise ValueError(
            f"No linear modules matching {sorted(targets)} found in the model "
            f"(checked plain nn.Linear and wrapped '.linear' children)."
        )

    resolved = sorted(suffixes)
    wrapped = any(s.endswith(".linear") for s in resolved)
    print(f"  Target modules (suffix form{', wrapped .linear children' if wrapped else ''}): "
          f"{resolved}")
    return resolved


def add_common_args(parser):
    """Arguments shared with train_gemma.py / train_qwen.py, plus --gpus."""
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model path or HF id")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Training JSON/JSONL file or folder")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpus", type=str, default="",
                        help="Comma separated GPU ids, e.g. '0,1'. Applied to "
                             "CUDA_VISIBLE_DEVICES before torch is imported. Ignored "
                             "when launched via accelerate/torchrun.")

    parser.add_argument("--input_field", type=str, default="response_text")
    parser.add_argument("--output_field", type=str, default="genui_json")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--train_samples", type=int, default=-1)
    parser.add_argument("--eval_samples", type=int, default=200)
    parser.add_argument("--eval_split_ratio", type=float, default=0.05)
    parser.add_argument("--test_data_path", type=str, default=None,
                        help="Held-out set for generation-based evaluation")
    parser.add_argument("--test_samples", type=int, default=-1)

    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                        action="store_false")

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--report_to", type=str, nargs="+", default=["tensorboard"])

    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--prediction_interval", type=int, default=1,
                        help="Run generation-based eval every N evaluations")
    parser.add_argument("--max_prediction_samples", type=int, default=50,
                        help="Cap on eval-split samples when --test_data_path is unset")
    parser.add_argument("--metric_for_best_model", type=str,
                        default="mean_direct_match_score")

    # Prediction archiving. {test,eval}_predictions.json is always rewritten each eval;
    # these control the per-step copies kept under output_dir/predictions/.
    parser.add_argument("--no_save_predictions", action="store_true",
                        help="Don't keep per-step prediction files (only the rolling latest)")
    parser.add_argument("--save_predictions_interval", type=int, default=1,
                        help="Archive predictions every Nth evaluation (1 = every eval)")
    parser.add_argument("--save_predictions_limit", type=int, default=0,
                        help="Keep only the newest N archived prediction files (0 = keep all)")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_missing_chat_template", action="store_true",
                        help="Proceed even if the tokenizer has no chat template")

    default_dataset = Path(__file__).resolve().parent.parent.parent / "dataset"
    parser.add_argument("--heuristic_util_folder", type=str,
                        default=str(default_dataset / "data" / "runs" / "util"))
    parser.add_argument("--heuristic_config_path", type=str,
                        default=str(default_dataset / "configs" / "run.yaml"))
    parser.add_argument("--heuristic_script_dir", type=str,
                        default=str(default_dataset / "scripts"))
    parser.add_argument("--heuristic_interval", type=int, default=1)
    parser.add_argument("--no_heuristic_eval", action="store_true",
                        help="Skip the external heuristic pipeline entirely")
    return parser


def add_qat_args(parser):
    """QAT arguments shared by both trainers."""
    parser.add_argument("--qat_scheme", type=str, default="int4", choices=["int4", "int8"],
                        help="Weight bit width for fake quantization")
    parser.add_argument("--qat_group_size", type=int, default=32,
                        help="Weight quantization group size (32 = QA-LoRA strict, 128 standard)")
    parser.add_argument("--qat_warmup_steps", type=int, default=0,
                        help="Train in full precision for N steps before enabling fake quant")
    parser.add_argument("--qat_weight_only", action="store_true",
                        help="Quantize weights only, leave activations in bf16")
    parser.add_argument("--qat_target_modules", type=str, nargs="+",
                        default=list(QAT_TARGET_MODULES),
                        help="Linear module names to fake-quantize")
    parser.add_argument("--qat_diag_interval", type=int, default=50,
                        help="Log QAT diagnostics every N steps (0 disables)")
    return parser


def build_sft_config(args, has_eval: bool) -> SFTConfig:
    """SFTConfig matching train_gemma.py, with the no-eval case handled."""
    return SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=args.eval_steps if has_eval else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=args.gradient_checkpointing,
        # Non-reentrant checkpointing is required when most parameters are frozen
        # (LoRA), and is the recommended default in current PyTorch.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=list(args.report_to),
        remove_unused_columns=False,
        # Reduce dataloader workers from the default (64 on multi-core servers)
        # to prevent OOM and excessive memory overhead during data loading.
        dataloader_num_workers=4,
        dataloader_pin_memory=False,
        # load_best_model_at_end needs an eval loop to have something to rank.

        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
        max_grad_norm=1.0,
        seed=args.seed,
    )


def build_callbacks(args, tokenizer, test_data, backend: str, expect_frozen_base: bool = True):
    """QAT warmup + diagnostics + generation eval + heuristic eval."""
    callbacks = [
        QATWarmupCallback(args.qat_warmup_steps, backend=backend, output_dir=args.output_dir),
    ]
    if args.qat_diag_interval > 0:
        callbacks.append(QATDiagnosticsCallback(
            args.qat_diag_interval, args.output_dir,
            backend=backend, expect_frozen_base=expect_frozen_base,
        ))

    eval_callback = EvalPredictionCallback(
        tokenizer=tokenizer,
        input_field=args.input_field,
        output_field=args.output_field,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        prediction_interval=args.prediction_interval,
        max_prediction_samples=args.max_prediction_samples,
        test_data=test_data,
        metric_for_best_model=args.metric_for_best_model,
        output_dir=args.output_dir,
        save_predictions=not args.no_save_predictions,
        save_predictions_interval=args.save_predictions_interval,
        save_predictions_limit=args.save_predictions_limit,
    )
    callbacks.append(eval_callback)

    heuristic_callback = None
    if args.no_heuristic_eval:
        print("Heuristic evaluation disabled (--no_heuristic_eval)")
    else:
        heuristic_callback = HeuristicEvaluationCallback(
            tokenizer=tokenizer,
            util_folder=args.heuristic_util_folder,
            config_path=args.heuristic_config_path,
            heuristic_interval=args.heuristic_interval,
            output_dir=args.output_dir,
            script_dir=args.heuristic_script_dir,
        )
        if heuristic_callback.is_enabled:
            print(f"Heuristic evaluation enabled every {args.heuristic_interval} eval(s)")
            print(f"  Util folder: {heuristic_callback.util_folder}")
            print(f"  Scripts:     {heuristic_callback.script_dir}")
            callbacks.append(heuristic_callback)
        else:
            print("Heuristic evaluation unavailable (missing util folder or pipeline script)")
            heuristic_callback = None

    return callbacks, eval_callback, heuristic_callback


def print_best_checkpoints(args, eval_callback, heuristic_callback=None):
    """Final summary of the checkpoints each selection criterion produced."""
    print("=" * 60)
    print("Training completed!")
    print(f"Output: {args.output_dir}")
    if eval_callback.best_model_saved:
        print(f"  Best {eval_callback.metric_for_best_model}: "
              f"{eval_callback.best_metric_value:.4f} @ step {eval_callback.best_metric_step}"
              f" -> {os.path.join(args.output_dir, 'best_test_checkpoint')}")
    if eval_callback.best_val_loss_model_saved:
        print(f"  Best eval_loss: {eval_callback.best_val_loss:.4f} "
              f"@ step {eval_callback.best_val_loss_step}"
              f" -> {os.path.join(args.output_dir, 'best_val_loss_checkpoint')}")
    if heuristic_callback and heuristic_callback.best_heuristic_model_saved:
        print(f"  Best overall_score: {heuristic_callback.best_heuristic_score:.4f} "
              f"@ step {heuristic_callback.best_heuristic_step}"
              f" -> {os.path.join(args.output_dir, 'best_heuristic_checkpoint')}")
    print(f"\nTensorBoard:  tensorboard --logdir {args.output_dir}")
    print("=" * 60)
