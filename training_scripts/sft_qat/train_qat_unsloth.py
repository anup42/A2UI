"""
QAT training via Unsloth's native quantizer (int4 by default).

Single GPU only -- Unsloth does not officially support multi-GPU/DDP.

Usage:
    python train_qat_unsloth.py --model_name MODEL --data_path DATA --output_dir OUT
"""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path

from unsloth import FastModel
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer

from dataloader import load_dataset, load_test_data  # noqa: E402
from qat_utils import (  # noqa: E402
    QAT_TARGET_MODULES,
    EvalPredictionCallback,
    HeuristicEvaluationCallback,
    build_sft_config,
    print_best_checkpoints,
    resolve_target_module_suffixes,
)


def parse_args():
    p = argparse.ArgumentParser(description="QAT training via Unsloth's native quantizer")
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--test_data_path", type=str, default=None)
    p.add_argument("--test_samples", type=int, default=-1)

    p.add_argument("--input_field", type=str, default="response_text")
    p.add_argument("--output_field", type=str, default="genui_json")
    p.add_argument("--max_length", type=int, default=6144)
    p.add_argument("--train_samples", type=int, default=-1)
    p.add_argument("--eval_samples", type=int, default=100)
    p.add_argument("--eval_split_ratio", type=float, default=0.05)

    p.add_argument("--epochs", type=float, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)

    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--report_to", type=str, nargs="+", default=["tensorboard"])

    p.add_argument("--max_new_tokens", type=int, default=6144)
    p.add_argument("--prediction_interval", type=int, default=1)
    p.add_argument("--max_prediction_samples", type=int, default=50)
    p.add_argument("--metric_for_best_model", type=str, default="mean_direct_match_score")

    # Prediction archiving. test_predictions.json is always rewritten each eval; these
    # control the per-step copies kept under output_dir/predictions/.
    p.add_argument("--no_save_predictions", action="store_true",
                    help="Don't keep per-step prediction files (only the rolling latest)")
    p.add_argument("--save_predictions_interval", type=int, default=1,
                    help="Archive predictions every Nth evaluation (1 = every eval)")
    p.add_argument("--save_predictions_limit", type=int, default=0,
                    help="Keep only the newest N archived prediction files (0 = keep all)")

    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--qat_scheme", type=str, default="int8-int4",
                    choices=["int4", "int8-int4", "fp8-int4", "fp8-fp8"])

    # Same repo-relative defaults as qat_utils.add_common_args / train_gemma.py, so
    # heuristic eval works out of the box without passing all three paths explicitly.
    # HeuristicEvaluationCallback self-disables if a path doesn't actually exist.
    default_dataset = Path(__file__).resolve().parent.parent.parent / "dataset"
    p.add_argument("--heuristic_util_folder", type=str,
                    default=str(default_dataset / "data" / "runs" / "golden100_gt"))
    p.add_argument("--heuristic_config_path", type=str,
                    default=str(default_dataset / "configs" / "run.yaml"))
    p.add_argument("--heuristic_script_dir", type=str,
                    default=str(default_dataset / "scripts"))
    p.add_argument("--heuristic_interval", type=int, default=1)
    p.add_argument("--no_heuristic_eval", action="store_true",
                    help="Skip the external heuristic pipeline entirely")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                    help="Path to a checkpoint-N directory to resume from, or 'auto' to "
                         "resume from the latest checkpoint under --output_dir. Omit to "
                         "start fresh.")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    return args


def _get_peft_model_checked(model, **kwargs):
    """FastModel.get_peft_model, asserting Unsloth actually applied QAT.

    Unsloth prints "Applying QAT to mitigate quantization degradation" only from
    inside the `if qat_scheme is not None:` branch of FastBaseModel.get_peft_model.
    Capturing that marker is the only implementation-agnostic proof QAT was really
    wired in: static module/dtype inspection cannot tell the difference, because
    fake quant recomputes its quantize/dequantize round-trip from the bf16 weight
    on every forward and leaves no distinguishing state behind.

    This guard exists because FastLanguageModel.get_peft_model silently dropped
    qat_scheme on the new-model path -- training looked fine and produced
    plain-bf16 checkpoints for months. Fail loudly instead.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model = FastModel.get_peft_model(model, **kwargs)
    captured = buf.getvalue()
    if captured:
        print(captured, end="")

    if kwargs.get("qat_scheme") is not None and "Applying QAT" not in captured:
        raise RuntimeError(
            f"qat_scheme={kwargs.get('qat_scheme')!r} was passed, but Unsloth never "
            f"printed its 'Applying QAT to mitigate quantization degradation' "
            f"confirmation. QAT was NOT applied -- this run would train in plain "
            f"bf16 and every checkpoint would be unquantizable. Refusing to start. "
            f"Verify with test_fake_quant_active.py --use_fastmodel."
        )
    if kwargs.get("qat_scheme") is not None:
        print(f"  [QAT verified] Unsloth confirmed QAT applied "
              f"(scheme={kwargs.get('qat_scheme')!r})")
    return model


def _resolve_resume_checkpoint(output_dir: str, requested: str | None):
    """Turn --resume_from_checkpoint into a concrete path or None.

    'auto' resolves via HF's own checkpoint-N naming convention (get_last_checkpoint),
    so it stays correct regardless of --save_steps or how many checkpoints exist.
    """
    if not requested:
        return None
    if requested == "auto":
        last_ckpt = get_last_checkpoint(output_dir)
        if last_ckpt is None:
            print(f"WARNING: --resume_from_checkpoint auto requested but no checkpoint "
                  f"found under {output_dir}; starting fresh.")
            return None
        print(f"Resuming from latest checkpoint: {last_ckpt}")
        return last_ckpt
    if not os.path.isdir(requested):
        raise ValueError(f"--resume_from_checkpoint path does not exist: {requested}")
    print(f"Resuming from checkpoint: {requested}")
    return requested


def _restore_best_tracking(output_dir: str, eval_callback, heuristic_callback):
    """Re-seed each callback's best-metric bookkeeping from its info JSON on disk.

    HF's own resume restores model/optimizer/scheduler/RNG state, but callback
    attributes like eval_callback.best_val_loss are plain Python fields that reset to
    their constructor defaults on a fresh process. Without this, the first eval after
    a resume looks like an unconditional "new best" and overwrites an already-better
    best_val_loss_checkpoint/ / best_test_checkpoint/ with a worse one.
    """
    def _load(name):
        path = os.path.join(output_dir, name)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    info = _load("best_val_loss_info.json")
    if info:
        eval_callback.best_val_loss = info.get("best_val_loss", eval_callback.best_val_loss)
        eval_callback.best_val_loss_step = info.get("best_step", eval_callback.best_val_loss_step)
        eval_callback.best_val_loss_model_saved = True
        print(f"  Restored best eval_loss tracking: {eval_callback.best_val_loss:.4f} "
              f"@ step {eval_callback.best_val_loss_step}")

    info = _load("best_metric_info.json")
    if info:
        eval_callback.best_metric_value = info.get("best_metric_value", eval_callback.best_metric_value)
        eval_callback.best_metric_step = info.get("best_step", eval_callback.best_metric_step)
        eval_callback.best_model_saved = True
        print(f"  Restored best {eval_callback.metric_for_best_model} tracking: "
              f"{eval_callback.best_metric_value:.4f} @ step {eval_callback.best_metric_step}")

    if heuristic_callback is not None:
        info = _load("best_heuristic_info.json")
        if info:
            heuristic_callback.best_heuristic_score = info.get(
                "best_overall_score", heuristic_callback.best_heuristic_score)
            heuristic_callback.best_heuristic_step = info.get(
                "best_step", heuristic_callback.best_heuristic_step)
            heuristic_callback.best_heuristic_model_saved = True
            print(f"  Restored best heuristic overall_score tracking: "
                  f"{heuristic_callback.best_heuristic_score:.4f} "
                  f"@ step {heuristic_callback.best_heuristic_step}")


def main():
    args = parse_args()
    set_seed(args.seed)

    # Convert args to JSON-safe dict (handle Path objects and other non-serializable types)
    args_dict = {}
    for k, v in vars(args).items():
        if v is None:
            args_dict[k] = None
        elif isinstance(v, (str, int, float, bool, list, tuple)):
            args_dict[k] = v
        else:
            # Convert Path objects and other types to string
            args_dict[k] = str(v)

    with open(os.path.join(args.output_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=2)

    print("Loading model...")
    # QAT needs a 16-bit base: fake quant is simulated during training, so the stored
    # weights must stay bf16. load_in_4bit defaults to True, so switch it off explicitly
    # or Unsloth rejects the combination.
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_length,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Unsloth wants bare suffixes, not full dotted paths (it rejects paths with
    # "No layers to finetune?"). The resolver reads the actual module types off the model,
    # so this covers plain-nn.Linear Gemmas (gemma-3) and wrapper-based ones
    # (gemma-4's Gemma4ClippableLinear -> "q_proj.linear") from the same code path.
    target_modules = resolve_target_module_suffixes(model, QAT_TARGET_MODULES)

    print("Applying LoRA + QAT...")
    # MUST be FastModel, not FastLanguageModel. FastLanguageModel.get_peft_model
    # (llama.py) SILENTLY DROPS qat_scheme when it forwards to the new-model path
    # (UNSLOTH_USE_NEW_MODEL=1, which is what Gemma-3/Gemma-4 load under), so QAT
    # is never applied -- verified behaviorally: logits were bit-identical with
    # and without qat_scheme. FastModel routes to FastBaseModel.get_peft_model
    # (vision.py), which honors qat_scheme in Unsloth's intended order
    # (_get_peft_model -> _prepare_model_for_qat -> fix_lora_auto_mapping ->
    # post_patch_model). See test_fake_quant_active.py for the verification.
    #
    # finetune_* filters are passed EXPLICITLY: FastModel defaults
    # finetune_vision_layers=True, which would attach LoRA to Gemma-4's vision/
    # audio towers. This is a text->JSON task, so keep adapters language-only.
    model = _get_peft_model_checked(
        model,
        r=args.lora_r,
        target_modules=target_modules,
        lora_alpha=args.lora_alpha,
        qat_scheme=args.qat_scheme,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        finetune_vision_layers=False,
        finetune_audio_layers=False,
    )

    print("Loading dataset...")
    train_dataset, eval_dataset = load_dataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        input_field=args.input_field,
        output_field=args.output_field,
        max_length=args.max_length,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        eval_split_ratio=args.eval_split_ratio,
        seed=args.seed,
    )

    test_data = None
    if args.test_data_path:
        if args.test_samples == -1:
            print(f"WARNING: --test_samples not set (default -1 = ALL rows in "
                  f"--test_data_path will be used for predictions/heuristic eval "
                  f"every eval_steps). Pass --test_samples N to cap this -- note "
                  f"this is a SEPARATE argument from --eval_samples, which only "
                  f"controls the validation split size carved from --data_path.")
        test_data = load_test_data(
            test_path=args.test_data_path,
            input_field=args.input_field,
            output_field=args.output_field,
            num_samples=args.test_samples,
        ) or None
        if test_data:
            print(f"  Loaded {len(test_data)} test sample(s) from {args.test_data_path} "
                  f"(--test_samples={args.test_samples})")

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
    callbacks = [eval_callback]
    if not args.no_save_predictions:
        print(f"Prediction archiving: every {args.save_predictions_interval} eval(s) -> "
              f"{os.path.join(args.output_dir, 'predictions')}"
              + (f" (keeping newest {args.save_predictions_limit})"
                 if args.save_predictions_limit > 0 else ""))

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

    trainer = SFTTrainer(
        model=model,
        args=build_sft_config(args, has_eval=eval_dataset is not None),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    # Lets on_evaluate force a checkpoint save before running generation, so a
    # generation-time OOM doesn't lose a step whose checkpoint was due.
    eval_callback.trainer = trainer

    resume_from_checkpoint = _resolve_resume_checkpoint(
        args.output_dir, args.resume_from_checkpoint)

    # Verify checkpoint will be loaded properly
    if resume_from_checkpoint:
        # Save a fresh checkpoint BEFORE eval runs, so eval uses the latest state
        print(f"\n  Saving checkpoint before eval (so eval uses fresh checkpoint)...")
        trainer.save_model()
        trainer.state.save_to_file(os.path.join(args.output_dir, "trainer_state.json"))
        print(f"  ✓ Checkpoint saved")
        print()
        print(f"\n" + "=" * 60)
        print("CHECKPOINT RESUME CONFIGURED")
        print("=" * 60)
        print(f"  Checkpoint path: {resume_from_checkpoint}")

        # Check what's in the checkpoint
        import torch
        from pathlib import Path
        ckpt_file = Path(resume_from_checkpoint) / "pytorch_model.bin"
        if not ckpt_file.exists():
            ckpt_file = Path(resume_from_checkpoint) / "model.safetensors"

        if ckpt_file.exists():
            print(f"  Model weights: {ckpt_file.name}")

        optimizer_file = Path(resume_from_checkpoint) / "optimizer.pt"
        if optimizer_file.exists():
            print(f"  Optimizer state: {optimizer_file.name} (WILL BE LOADED)")
        else:
            print(f"  WARNING: optimizer.pt not found - optimizer will start fresh!")

        scheduler_file = Path(resume_from_checkpoint) / "scheduler.pt"
        if scheduler_file.exists():
            print(f"  Scheduler state: {scheduler_file.name} (WILL BE LOADED)")
        else:
            print(f"  WARNING: scheduler.pt not found - scheduler will start fresh!")

        trainer_state_file = Path(resume_from_checkpoint) / "trainer_state.json"
        if trainer_state_file.exists():
            with open(trainer_state_file, "r") as f:
                trainer_state = json.load(f)
            print(f"  Training state: epoch={trainer_state.get('epoch')}, "
                  f"global_step={trainer_state.get('global_step')}, "
                  f"max_steps={trainer_state.get('max_steps')}")
        else:
            print(f"  WARNING: trainer_state.json not found!")

        print("=" * 60 + "\n")

        _restore_best_tracking(args.output_dir, eval_callback, heuristic_callback)

    # Log eval config for consistency: convert_qat_to_quantized.py's --eval_samples
    # maps to THIS script's --test_samples (predictions on --test_data_path), NOT
    # --eval_samples (which here means the validation split carved from
    # --data_path -- a different, unrelated knob). Record the actual count
    # test_data resolved to, since --test_samples -1 means "unlimited" and
    # convert-time needs a concrete number to match.
    resolved_test_samples = len(test_data) if test_data else args.test_samples
    eval_config = {
        "eval_samples": resolved_test_samples,
        "eval_batch_size": args.eval_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "qat_scheme": args.qat_scheme,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "note": f"During conversion, use: --eval_samples {resolved_test_samples} --seed {args.seed} "
               f"to evaluate on the same {resolved_test_samples} samples deterministically. "
               f"(this maps to this training run's --test_samples, not its --eval_samples, "
               f"which is an unrelated validation-split-size argument)"
    }
    eval_config_path = os.path.join(args.output_dir, "eval_config.json")
    with open(eval_config_path, "w") as f:
        json.dump(eval_config, f, indent=2)
    print(f"Eval config saved to: {eval_config_path}")
    print(f"  For later conversion consistency:")
    print(f"    eval_samples: {resolved_test_samples}")
    print(f"    seed: {args.seed}")

    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if trainer.is_world_process_zero():
        adapter_dir = os.path.join(args.output_dir, "adapter")
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        print(f"Adapter saved to: {adapter_dir}")

        # QAT-specific export: convert the fake-quantized (STE) weights into the real
        # quantized int4 checkpoint. Per Unsloth's QAT docs:
        # https://unsloth.ai/docs/blog/quantization-aware-training-qat
        #
        # NO try/except: this silently swallowed the export failure on every
        # prior run (save_pretrained_torchao's signature is
        # (self, save_directory, tokenizer=None, torchao_config=None, ...) --
        # passing `trainer.model` as save_directory and an explicit
        # torchao_config for a QAT model, which asserts `not has_qat_config`,
        # both fail immediately). Every run reported success while writing
        # nothing real.
        from torchao.quantization import quantize_
        from torchao.quantization.qat import QATConfig

        quantize_(trainer.model, QATConfig(step="convert"))
        torchao_dir = os.path.join(args.output_dir, "model_torchao")
        os.makedirs(torchao_dir, exist_ok=True)
        # torchao_config omitted: auto-read from trainer.model._torchao_config
        # for QAT models; passing it explicitly asserts `not has_qat_config`.
        trainer.model.save_pretrained_torchao(torchao_dir, tokenizer)
        print(f"Real int4 (torchao) checkpoint saved to: {torchao_dir}")

        print_best_checkpoints(args, eval_callback, heuristic_callback)


if __name__ == "__main__":
    main()
