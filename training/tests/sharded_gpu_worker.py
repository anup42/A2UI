"""Torchrun worker for the opt-in real ZeRO-2/3 integration test."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


class _Collator:
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        return {
            key: torch.tensor([feature[key] for feature in features], dtype=torch.long)
            for key in ("input_ids", "attention_mask", "labels")
        }


def _training_config(zero_stage: int) -> dict[str, Any]:
    config = {
        "distributed_backend": "sharded",
        "full_parameter_training": True,
        "optim": "adamw_torch",
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "max_grad_norm": 0.0,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }
    if zero_stage == 3:
        config["zero_stage"] = 3
    return config


def _qat_config() -> dict[str, Any]:
    return {
        "qat": {
            "enabled": True,
            "quantizer": "ste_ai_edge",
            "weight_bits": 8,
            "activation_bits": 16,
            "weight_symmetric": True,
            "activation_symmetric": True,
            "weight_per_channel": True,
            "weight_axis": 0,
            "quantize_embeddings": True,
            "exclude_modules": [],
            "only_base_layers": True,
        }
    }


def _new_model(*, tie_word_embeddings: bool) -> tuple[Any, Any]:
    import torch
    from ir_training.qat.fake_quant import prepare_qat_model
    from transformers import LlamaConfig, LlamaForCausalLM

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.manual_seed(713)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
            tie_word_embeddings=tie_word_embeddings,
            use_cache=False,
        )
    ).float().to(torch.device(f"cuda:{local_rank}"))
    model._a2ui_full_parameter_amp = True
    model.register_buffer("sharded_test_scale", torch.tensor([1.25], dtype=torch.float32), persistent=True)
    for parameter in model.parameters():
        parameter.requires_grad = True
    controller = prepare_qat_model(model, _qat_config())
    if controller.wrapped_count < 1:
        raise RuntimeError("Tiny HF model did not activate QAT forward wrappers")
    return model, controller


def _rows() -> list[dict[str, list[int]]]:
    rows = []
    for offset in range(8):
        ids = [1, 3 + offset % 8, 12, 7, 2, 5]
        rows.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": list(ids)})
    return rows


def _trainer_components(
    output_dir: Path, *, max_steps: int, zero_stage: int, tie_word_embeddings: bool
) -> tuple[Any, dict[str, Any], Any, Any]:
    from ir_training.train.sft import _build_checked_causal_lm_trainer
    from ir_training.train.sharded_contract import build_deepspeed_config
    from transformers import Trainer, TrainingArguments

    cfg = _training_config(zero_stage)
    world_size = int(os.environ["WORLD_SIZE"])
    # Preserve the production path: ordinary model construction precedes
    # TrainingArguments/DeepSpeed setup; this test does not introduce ZeRO.Init.
    model, controller = _new_model(tie_word_embeddings=tie_word_embeddings)
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        max_steps=max_steps,
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        max_grad_norm=cfg["max_grad_norm"],
        optim=cfg["optim"],
        bf16=False,
        fp16=False,
        deepspeed=build_deepspeed_config(cfg, world_size),
        remove_unused_columns=False,
        save_strategy="no",
        eval_strategy="no",
        logging_strategy="no",
        report_to=[],
        disable_tqdm=True,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs=cfg["gradient_checkpointing_kwargs"],
    )
    checked = _build_checked_causal_lm_trainer(Trainer, cfg)
    kwargs = {
        "model": model,
        "args": args,
        "train_dataset": _rows(),
        "data_collator": _Collator(),
    }
    return checked, kwargs, model, controller


def _load_safetensor_directory(path: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    tensors: dict[str, Any] = {}
    for shard in sorted(path.glob("*.safetensors")):
        tensors.update(load_file(str(shard), device="cpu"))
    if not tensors:
        raise RuntimeError(f"No safetensors model weights were saved under {path}")
    return tensors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase", choices=("preflight", "train"), required=True)
    parser.add_argument("--zero-stage", type=int, choices=(2, 3), default=2)
    parser.add_argument("--tie-word-embeddings", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.distributed as dist
    from ir_training.train.sharded_contract import (
        build_deepspeed_config,
        validate_sharded_runtime,
    )
    from ir_training.train.sharded_preflight import (
        run_sharded_trainer_preflight,
        validate_sharded_probe,
    )

    runtime = validate_sharded_runtime()
    cfg = _training_config(args.zero_stage)
    if args.phase == "preflight":
        checked, trainer_kwargs, _preflight_model, controller = _trainer_components(
            output_dir / "preflight", max_steps=2, zero_stage=args.zero_stage,
            tie_word_embeddings=args.tie_word_embeddings,
        )
        receipt = run_sharded_trainer_preflight(
            checked, trainer_kwargs, training_cfg=cfg, longest_row=_rows()[0]
        )
        controller.restore()
        if receipt.get("passed") is not True:
            raise RuntimeError(f"Sharded production preflight did not pass: {receipt}")
        if dist.get_rank() == 0:
            (output_dir / "preflight_receipt.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        dist.barrier()
        return 0

    preflight_path = output_dir / "preflight_receipt.json"
    if not preflight_path.is_file():
        raise RuntimeError("Training phase requires a completed disposable preflight receipt")
    receipt = json.loads(preflight_path.read_text(encoding="utf-8"))
    validate_sharded_probe(receipt, accumulation_steps=cfg["gradient_accumulation_steps"],
                           world_size=int(os.environ["WORLD_SIZE"]), local_rank=0,
                           zero_stage=args.zero_stage)
    if receipt["sharded_probe"]["config"] != build_deepspeed_config(cfg, int(os.environ["WORLD_SIZE"])):
        raise RuntimeError("Disposable receipt differs from the training configuration")

    checked, trainer_kwargs, live_model, controller = _trainer_components(
        output_dir / "training", max_steps=2, zero_stage=args.zero_stage,
        tie_word_embeddings=args.tie_word_embeddings,
    )
    trainer = checked(**trainer_kwargs)
    trainer.train()
    if trainer.state.global_step != 2:
        raise RuntimeError(f"Expected two optimizer steps, got {trainer.state.global_step}")
    from ir_training.qat.full_model_contract import full_parameter_autocast

    live_model.eval()
    prompt = torch.tensor([[1, 3, 12]], dtype=torch.long, device=next(live_model.parameters()).device)
    with torch.no_grad(), full_parameter_autocast(live_model):
        generated = live_model.generate(
            prompt, min_new_tokens=2, max_new_tokens=2, do_sample=False,
            synced_gpus=args.zero_stage == 3,
        )
    if generated.shape != (1, 5) or not bool(((generated >= 0) & (generated < 32)).all().item()):
        raise RuntimeError(f"Sharded trained replica produced invalid bounded generation: {generated}")
    controller.restore()
    saved = output_dir / "saved_model"
    if args.zero_stage == 3:
        # Exercise the production Trainer override on every rank.
        trainer.save_model(str(saved))
    else:
        trainer.save_model(str(saved))  # Called by every rank; Trainer controls the writer.
    dist.barrier()
    final_saved = output_dir / "final_model"
    if args.zero_stage == 3:
        from ir_training.train.zero3_checkpoint import save_zero3_checkpoint

        save_zero3_checkpoint(trainer, final_saved, tokenizer=None)
    elif dist.get_rank() == 0:
        live_model.save_pretrained(final_saved, safe_serialization=True)
    dist.barrier()

    # All ZeRO-3 ranks must participate in full-state gathering.
    raw_live_state = (
        trainer.accelerator.get_state_dict(trainer.deepspeed)
        if args.zero_stage == 3
        else None
    )
    if dist.get_rank() == 0:
        if args.zero_stage == 3:
            if not isinstance(raw_live_state, dict):
                raise RuntimeError("Rank 0 did not receive the consolidated ZeRO-3 state")
            live_state = {
                name: value.detach().cpu() for name, value in raw_live_state.items()
            }
        else:
            live_state = {name: value.detach().cpu() for name, value in live_model.state_dict().items()}
        saved_states = {"trainer": _load_safetensor_directory(saved),
                        "final": _load_safetensor_directory(final_saved)}
        parameter_names = {name for name, _ in live_model.named_parameters()}
        buffer_names = {name for name, _ in live_model.named_buffers() if name in live_state}
        for role, saved_state in saved_states.items():
            if args.tie_word_embeddings:
                for alias, canonical in (
                    ("lm_head.weight", "model.embed_tokens.weight"),
                    ("model.embed_tokens.weight", "lm_head.weight"),
                ):
                    if alias in live_state and alias not in saved_state and canonical in saved_state:
                        saved_state[alias] = saved_state[canonical]
            if set(saved_state) != set(live_state):
                raise RuntimeError(f"{role} saved state keys differ from live graph")
            non_fp32 = [name for name in parameter_names if saved_state[name].dtype != torch.float32]
            if non_fp32:
                raise RuntimeError(f"{role} save contains non-FP32 weights: {non_fp32[:5]}")
            unequal = [name for name in saved_state if not torch.equal(saved_state[name], live_state[name])]
            if unequal:
                raise RuntimeError(f"{role} tensors differ from trained live graph: {unequal[:5]}")
            if "sharded_test_scale" not in buffer_names or "sharded_test_scale" not in saved_state:
                raise RuntimeError(f"{role} save omitted persistent graph buffer")
        result = {
            "passed": True,
            "world_size": dist.get_world_size(),
            "zero_stage": args.zero_stage,
            "tie_word_embeddings": args.tie_word_embeddings,
            "optimizer_steps": trainer.state.global_step,
            "runtime": runtime,
            "preflight": receipt,
            "saved_tensor_count": len(saved_states["trainer"]),
            "saved_parameter_count": len(parameter_names),
            "saved_persistent_buffer_count": len(buffer_names),
            "all_saved_tensors_fp32": True,
            "saved_matches_live_graph": True,
            "bounded_generation_verified": True,
        }
        (output_dir / "gpu_test_receipt.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    dist.barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
