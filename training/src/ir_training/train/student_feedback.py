"""Experimental matched A/B/C QAT-LoRA trials; never auto-export a new objective.

A: completion SFT on a deterministic ordinary-source schedule.
B: the same schedule with bounded hard-example replacements.
C: the B schedule, adding DPO only to explicitly paired replay occurrences.

Uses shared model loading, QAT/numeric gates and Golden evaluation, NOT GRPO
optimization. Checkpoints intentionally lack production training_metadata.json:
this new objective needs a separately reviewed export/provenance integration.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import random
from typing import Any

from ir_training.data.student_feedback import exposure_schedule, fingerprint, text_hash
from ir_training.pipeline.student_feedback import (
    IndexedRows, check_inventory, inventory, load_feedback, loading_config,
    new_output, offline_hf, sha_file, write_json,
)
from ir_training.train.feedback_objective import (
    FeedbackCollator, completion_logps, encode_completion, feedback_loss,
)


@contextmanager
def preserve_rng():
    import torch
    import numpy as np
    py, numpy = random.getstate(), np.random.get_state()
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(py)
        np.random.set_state(numpy)


class FeedbackDataset:
    def __init__(self, rows, indices, hard_mask, encode, pairs=None, references=None):
        if len(indices) != len(hard_mask) or any(type(x) is not bool for x in hard_mask):
            raise ValueError("Exposure indices and explicit hard masks must align")
        self.rows, self.indices, self.hard_mask, self.encode = rows, indices, hard_mask, encode
        self.pairs, self.references = pairs or {}, references if references is not None else {}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        row = self.rows[index]
        item = {"chosen": self.encode(row)}
        pair = self.pairs.get(index) if self.hard_mask[position] else None
        if pair is not None:
            if fingerprint(row) != pair["row_sha256"] or row["completion"] != pair["chosen"]:
                raise ValueError("Preference source/positive changed")
            if text_hash(item["chosen"]["prompt"]) != pair["prompt_sha256"]:
                raise ValueError("Preference prompt differs from the probed prompt")
            item["rejected"] = self.encode(row, completion=pair["rejected"])
            if index not in self.references:
                raise ValueError("Frozen QAT SFT reference has not been computed")
            item["reference"] = self.references[index]
        return item


def trainer_class(base, bundle, *, beta, sft_weight):
    from ir_training.train.qat_grpo import assert_qat_policy_active

    class FeedbackTrainer(base):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            assert_qat_policy_active(model, bundle.controller)
            with self.accelerator.autocast():
                result = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False)
            sums, counts = completion_logps(result.logits, inputs["labels"])
            loss, metrics = feedback_loss(sums, counts, inputs["num_chosen"], inputs["pair_positions"],
                                          inputs["reference"], beta=beta, sft_weight=sft_weight)
            if model.training:
                self.feedback_metrics = {key: float(value) for key, value in metrics.items()}
                current = [int(metrics["chosen_tokens"]), int(metrics["rejected_tokens"]),
                           int(inputs["attention_mask"].sum()), inputs["num_chosen"], inputs["pair_positions"].numel()]
                prior = getattr(self, "feedback_token_counts", [0] * len(current))
                self.feedback_token_counts = [a + b for a, b in zip(prior, current, strict=True)]
            # Gradient accumulation is handled once by Trainer; see the explicit
            # model_accepts_loss_kwargs=False setting at construction below.
            return (loss, {"loss": loss}) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            import torch
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad(), self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            return loss.detach(), None, None

        def save_model(self, output_dir=None, _internal_call=False):
            generation = getattr(self.model, "generation_config", None)
            previous = getattr(generation, "cache_implementation", None)
            if generation is not None:
                generation.cache_implementation = None
            try:
                return super().save_model(output_dir, _internal_call=_internal_call)
            finally:
                if generation is not None:
                    generation.cache_implementation = previous
    return FeedbackTrainer


def health_callback(trainer, output: Path):
    import torch
    from transformers import TrainerCallback
    from ir_training.train.grpo_runtime import write_json_record
    accelerator = trainer.accelerator

    class FiniteUpdates(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            self.parameters = [p for p in model.parameters() if p.requires_grad and p.numel()]
            if not self.parameters:
                raise RuntimeError("No trainable feedback policy parameters")
            self.samples = [(p, torch.linspace(0, p.numel()-1, min(32, p.numel()), device=p.device).long().unique())
                            for p in self.parameters]
            self.before = [p.detach().reshape(-1)[i].float().clone() for p, i in self.samples]
            self.nonzero_steps = 0
            self.grad_norm = 0.0

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            finite = torch.ones((), dtype=torch.bool, device=accelerator.device)
            norm = torch.zeros((), device=accelerator.device)
            for p in self.parameters:
                finite &= torch.isfinite(p).all()
                if p.grad is not None:
                    finite &= torch.isfinite(p.grad).all()
                    norm += p.grad.detach().float().square().sum()
            values = accelerator.gather(torch.stack((finite.float(), norm.sqrt())).reshape(1, 2))
            if not bool(torch.isfinite(values).all()) or not bool(values[:, 0].all()):
                raise FloatingPointError("Nonfinite feedback gradient/parameter before optimizer update")
            self.grad_norm = float(values[:, 1].min())

        def on_step_end(self, args, state, control, **kwargs):
            delta = torch.zeros((), device=accelerator.device)
            for old, (p, indices) in zip(self.before, self.samples, strict=True):
                now = p.detach().reshape(-1)[indices].float()
                delta = torch.maximum(delta, (now-old).abs().max())
                old.copy_(now)
            ranks = accelerator.gather(delta.reshape(1))
            if not bool(torch.isfinite(ranks).all()):
                raise FloatingPointError("Nonfinite feedback optimizer update")
            actual = float(ranks.min())
            self.nonzero_steps += int(actual > 0 and self.grad_norm > 0)
            write_json_record(output / f"feedback_health.rank{accelerator.process_index}.jsonl",
                {"step": state.global_step, "gradient_norm_rank_min": self.grad_norm,
                 "sampled_parameter_delta_rank_min": actual,
                 "last_microbatch_metrics": getattr(trainer, "feedback_metrics", {})})

        def on_train_end(self, args, state, control, **kwargs):
            if not self.nonzero_steps:
                raise RuntimeError("Feedback training produced no measured nonzero gradient/update")
    return FiniteUpdates()


@offline_hf()
def train(feedback_dir: Path, output: Path, *, arm: str, max_steps=100,
          accumulation=16, learning_rate=5e-7, seed=42, beta=.1, sft_weight=1.,
          eval_steps=25, max_hard_exposures=3, execute=False):
    if arm not in {"sft", "replay", "dpo_sft"}:
        raise ValueError("Choose sft, replay or dpo_sft")
    for value in (max_steps, accumulation, eval_steps):
        if type(value) is not int or value < 1:
            raise ValueError("Step/accumulation/evaluation budgets must be positive integers")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
           for v in (learning_rate, beta, sft_weight)):
        raise ValueError("Learning rate and loss coefficients must be finite and positive")
    rows, families, replay, preferences, binding, manifest = load_feedback(feedback_dir)
    source = Path(binding["source_dir"])
    probe = binding["probe_plan"]
    source_config, checkpoint = Path(probe["source_config"]), Path(probe["checkpoint"])
    check_inventory(checkpoint, probe["checkpoint_files"])
    if sha_file(source_config) != probe["source_config_sha256"]:
        raise ValueError("Source SFT configuration changed")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < 1:
        raise ValueError("WORLD_SIZE must be positive")
    schedule = exposure_schedule(rows, families, replay,
        total_examples=max_steps * accumulation * world, seed=seed,
        max_hard_token_fraction=binding["options"]["max_extra_token_fraction"],
        max_hard_exposures_per_family=max_hard_exposures)
    if arm != "sft" and not schedule["report"]["hard_examples"]:
        raise ValueError("No hard examples fit this pilot; inspect the feedback report")
    if arm == "dpo_sft" and not preferences:
        raise ValueError("No trustworthy preference pairs; use replay SFT, not manufactured negatives")
    plan = {"version": "student_feedback_v1", "arm": arm, "max_steps": max_steps,
            "learning_rate": learning_rate, "seed": seed, "beta": beta, "sft_weight": sft_weight,
            "gradient_accumulation_steps": accumulation, "per_device_train_batch_size": 1,
            "world_size": world, "effective_batch_size": accumulation * world,
            "eval_steps": eval_steps, "schedule": schedule["report"],
            "feedback_manifest_sha256": sha_file(feedback_dir / "manifest.json"),
            "source_checkpoint": str(checkpoint), "source_config_sha256": probe["source_config_sha256"],
            "production_promotion_allowed": False, "native_quality_validated": False,
            "objective": "mean completion NLL + beta-scaled sequence-sum DPO on paired replay rows",
            "reference": "initial QAT SFT policy; precomputed before any optimizer step"}
    if not execute:
        return {"status": "plan_only", **plan}
    import torch
    from transformers import Trainer, TrainingArguments, TrainerCallback
    from ir_training.train import sft
    from ir_training.train.qat_grpo import load_qat_grpo_model, run_qat_grpo_preflight, assert_qat_policy_active
    from ir_training.train.prepared_binding import verify_tokenizer_binding
    from ir_training.eval.prepared_contract import verify_golden_preparation
    from ir_training.pipeline.official_mobile import SELECTOR
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Feedback training v1 requires a BF16-capable CUDA device")
    if "processing_class" not in inspect.signature(Trainer.__init__).parameters or not hasattr(TrainerCallback, "on_pre_optimizer_step"):
        raise RuntimeError("Installed Trainer lacks the required public callback/processing interface")
    # All ranks validate the same artifacts. Only rank zero creates output;
    # the existing loader/trainer establishes the process group afterwards.
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        new_output(output, feedback_dir, source, checkpoint, source_config)
        write_json(output / "feedback_plan.json", plan)
        write_json(output / "exposure_schedule.json", schedule)
    cfg = loading_config(source_config, checkpoint, source, output, seed=seed, accumulation=accumulation)
    seq = manifest["tokenizer"]["max_seq_length"]
    cfg["training"]["max_seq_length"] = seq
    cfg["training"]["seed"] = seed
    golden_path = source / "golden32.jsonl"
    verify_golden_preparation(source, golden_path, required_rows=32, max_sequence=seq,
                              max_prompt=manifest["tokenizer"]["max_input_tokens"])
    from ir_training.train.cuda_runtime import sdpa_policy
    with sdpa_policy(cfg):
        bundle = load_qat_grpo_model(cfg, config_path=source_config)
        try:
            tokenizer = bundle.tokenizer
            verify_tokenizer_binding(source, tokenizer, cfg["model"], required=True)
            template_kwargs = cfg["model"].get("chat_template_kwargs") or {}
            def encode(row, **kwargs):
                return encode_completion(row, tokenizer, template_kwargs=template_kwargs, max_length=seq, **kwargs)
            probe_rows = [{"prompt": encode(rows[i])["prompt"], "completion": rows[i]["completion"]}
                          for i in schedule["baseline_indices"][:max(2, cfg.get("preflight", {}).get("rows", 1))]]
            numeric = run_qat_grpo_preflight(bundle, probe_rows, max_seq_length=seq)
            pairs, excluded = {}, []
            for pair in preferences if arm == "dpo_sft" else []:
                try:
                    encode(rows[pair["train_index"]], completion=pair["rejected"])
                except ValueError as exc:
                    # Only the explicit length case is a quarantine. Boundary/data
                    # mismatches indicate a broken pipeline and must stop training.
                    if "exceeds budget" not in str(exc):
                        raise
                    excluded.append({"pair_id": pair["pair_id"], "reason": "complete_pair_exceeds_sequence_budget"})
                    continue
                pairs[pair["train_index"]] = pair
            indices = schedule["baseline_indices"] if arm == "sft" else schedule["replay_indices"]
            hard_mask = [False] * len(indices) if arm == "sft" else schedule["hard_mask"]
            active_pairs = {i for i, hard in zip(indices, hard_mask, strict=True) if hard and i in pairs}
            if arm == "dpo_sft" and not active_pairs:
                raise ValueError("No complete pairs fit the actual pilot schedule; DPO trial was not started")
            references = {}
            dataset = FeedbackDataset(rows, indices, hard_mask, encode, pairs=pairs, references=references)
            val_rows = IndexedRows(source / "val.jsonl")
            validation = FeedbackDataset(val_rows, list(range(min(32, len(val_rows)))), [False] * min(32, len(val_rows)), encode)
            args = TrainingArguments(output_dir=str(output), max_steps=max_steps,
                learning_rate=learning_rate, per_device_train_batch_size=1, per_device_eval_batch_size=1,
                gradient_accumulation_steps=accumulation, gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False}, ddp_broadcast_buffers=False,
                ddp_find_unused_parameters=False, bf16=True, fp16=False, logging_steps=1,
                logging_nan_inf_filter=False, save_strategy="steps", save_steps=eval_steps,
                save_total_limit=2, eval_strategy="steps", eval_steps=eval_steps, seed=seed, data_seed=seed,
                dataloader_drop_last=True, dataloader_num_workers=0, remove_unused_columns=False,
                report_to="none", weight_decay=0., max_grad_norm=1., warmup_steps=0, push_to_hub=False)
            cls = trainer_class(Trainer, bundle, beta=beta, sft_weight=sft_weight)
            collator = FeedbackCollator(tokenizer.pad_token_id)
            trainer = cls(model=bundle.model, args=args, train_dataset=dataset, eval_dataset=validation,
                          data_collator=collator, processing_class=tokenizer)
            trainer.model_accepts_loss_kwargs = False
            if trainer.is_fsdp_enabled or trainer.is_deepspeed_enabled:
                raise ValueError("Feedback v1 supports single-device or DDP, not sharded training")
            trainer.accelerator.wait_for_everyone()
            reference_evidence = []
            with preserve_rng(), torch.no_grad():
                was_training = bundle.model.training
                bundle.model.eval()
                try:
                    for index in sorted(active_pairs):
                        feature = {"chosen": encode(rows[index]), "rejected": encode(rows[index], completion=pairs[index]["rejected"]),
                                   "reference": (0., 0.)}  # Shape-only placeholders, never used for optimization.
                        batch = trainer._prepare_inputs(collator([feature]))
                        assert_qat_policy_active(bundle.model, bundle.controller)
                        with trainer.accelerator.autocast():
                            logits = bundle.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
                        values, _ = completion_logps(logits, batch["labels"])
                        references[index] = (float(values[0]), float(values[1]))
                        reference_evidence.append({"pair_id": pairs[index]["pair_id"], "logps": references[index],
                            "tokenization_sha256": fingerprint([feature["chosen"], feature["rejected"]])})
                        del logits, values, batch
                finally:
                    bundle.model.train(was_training)
            check_inventory(checkpoint, probe["checkpoint_files"])
            metadata = {**plan, "numeric_preflight": numeric, "loader_source_lineage": bundle.metadata["grpo"]["source_lineage"],
                        "qat": bundle.controller.summary(), "reference_evidence": reference_evidence,
                        "pair_quarantine": excluded, "training_executed": False,
                        "source_binding": binding, "load_only_contract": cfg,
                        "training_arguments": trainer.args.to_dict(),
                        "experiment_code_sha256": {"runner": sha_file(Path(__file__)),
                            "objective": sha_file(Path(__file__).with_name("feedback_objective.py"))},
                        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate")}}
            write_json(output / f"feedback_preflight.rank{rank}.json", metadata)
            golden_cfg = {"enabled": True, "split_path": str(golden_path), "max_rows": 32, "required_rows": 32,
                "require_exact_rows": True, "require_unique_rows": True, "trigger": "evaluate", "interval": 1,
                "max_input_tokens": manifest["tokenizer"]["max_input_tokens"], "max_new_tokens": probe["max_new_tokens"],
                "metric_version": "v5_4", "metric_for_best_model": SELECTOR, "greater_is_better": True,
                "save_best_checkpoint": True, "log_to_trainer": False, "evaluate_at_end": True}
            def golden(settings):
                return sft._build_optional_golden_callback(golden_eval_cfg=settings, base=source,
                    output_dir=output, adapter=bundle.adapter, tokenizer=tokenizer, model_cfg=cfg["model"],
                    training_cfg=cfg["training"], clear_generation_cache_on_save=True)
            baseline = golden({**golden_cfg, "output_dir": str(output / "baseline_golden32"), "save_best_checkpoint": False})
            with preserve_rng():
                baseline._run_if_due(state=trainer.state, control=trainer.control, model=trainer.model, force=True)
            selector = golden({**golden_cfg, "output_dir": str(output / "golden32")})
            trainer.add_callback(health_callback(trainer, output))
            trainer.add_callback(selector)
            trainer.train()
            counts = torch.tensor([getattr(trainer, "feedback_token_counts", [0]*5)],
                                  dtype=torch.long, device=trainer.accelerator.device)
            totals = trainer.accelerator.gather(counts).sum(0).tolist()
            metadata["observed_training_counts"] = dict(zip(
                ("chosen_completion_tokens", "rejected_completion_tokens", "processed_sequence_tokens",
                 "chosen_examples", "paired_examples"), totals, strict=True))
            check_inventory(source, binding["source_files"])
            bundle.controller.restore()
            trainer.save_model(str(output / "final_adapter"))
            trainer.accelerator.wait_for_everyone()
            if trainer.is_world_process_zero():
                metadata.update(training_executed=True, completed_optimizer_steps=int(trainer.state.global_step),
                    best_golden=selector.summary(), final_adapter_files=inventory(output / "final_adapter"))
                # Deliberately distinct from production SFT/GRPO export admission.
                write_json(output / "feedback_training_metadata.json", metadata)
            return {"status": "experimental_training_complete", "output": str(output),
                    "production_promotion_allowed": False, "native_quality_validated": False}
        finally:
            if bundle.controller is not None:
                bundle.controller.restore()
