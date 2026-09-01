"""QAT + LoRA via Unsloth's native quantizer, with DDP multi-GPU and fast eval.

Differences from train_qat_unsloth.py (which this does not modify):
  * Multi-GPU via DDP. Unsloth documents torchrun as supported and auto-enables
    DDP at >1 GPU: https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth/ddp
  * Generation metrics on BOTH the validation split and the test file, reported
    separately as eval/* and test/*.
  * FastEvalCallback instead of EvalPredictionCallback: KV cache enabled for
    generation, batched left-padded decode, stop-token early exit.

Usage:
    torchrun --nproc_per_node=2 train_qat_unsloth_ddp.py \
        --model_name MODEL --data_path DATA --output_dir OUT
"""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path

from unsloth import FastModel                  # must precede transformers/trl
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer

from dataloader import load_dataset, load_test_data
from fast_eval import DEFAULT_STOP_STRINGS, IR_PREFIX, FastEvalCallback
from qat_utils import (
    QAT_TARGET_MODULES,
    HeuristicEvaluationCallback,
    build_sft_config,
    print_best_checkpoints,
    resolve_target_module_suffixes,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="QAT+LoRA via Unsloth with DDP and fast batched eval")
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
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                   action="store_false")

    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--report_to", type=str, nargs="+", default=["tensorboard"])

    p.add_argument("--prediction_interval", type=int, default=1)
    p.add_argument("--metric_for_best_model", type=str, default="mean_direct_match_score")

    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--qat_scheme", type=str, default="int8-int4",
                   choices=["int4", "int8-int4", "fp8-int4", "fp8-fp8"])

    # ---- fast eval ----
    p.add_argument("--fast_eval_sources", type=str, default="both",
                   choices=["both", "test", "eval"],
                   help="Which sources get generation-scored each eval")
    p.add_argument("--fast_eval_batch_size", type=int, default=8,
                   help="Generation batch size (the old callback was hardcoded to 1)")
    p.add_argument("--fast_eval_min_batch_size", type=int, default=1)
    p.add_argument("--fast_eval_token_budget", type=int, default=0,
                   help="Cap batch size by longest prompt: bs = budget // longest. 0=off")
    p.add_argument("--fast_eval_max_new_tokens", type=int, default=0,
                   help="<=0 measures p99 of reference token lengths from the data")
    p.add_argument("--fast_eval_max_prompt_tokens", type=int, default=0,
                   help="0 uses --max_length")
    p.add_argument("--fast_eval_val_samples", type=int, default=50,
                   help="Cap on validation rows that get generation-scored")
    p.add_argument("--fast_eval_truncation_side", type=str, default="left",
                   choices=["left", "right"],
                   help="'left' keeps the generation cue; 'right' reproduces the old behavior")
    p.add_argument("--fast_eval_no_sort_by_length", action="store_true",
                   help="Disable length-sorted batching (increases padding waste)")
    p.add_argument("--fast_eval_no_extra_stop_tokens", action="store_true",
                   help="Use only the tokenizer's own eos_token_id, no <end_of_turn>")
    p.add_argument("--fast_eval_best_source", type=str, default="auto",
                   choices=["auto", "test", "eval"])
    p.add_argument("--fast_eval_no_ir_prefix", action="store_true",
                   help="Don't add IR_PREFIX to validation rows (old behavior, for A/B)")
    p.add_argument("--fast_eval_unsloth_inference", action="store_true",
                   help="Use FastLanguageModel.for_inference during eval. OFF by default: "
                        "its fused kernels may bypass the torchao fake quantizer, which "
                        "would report bf16 quality as if it were quantized.")
    p.add_argument("--fast_eval_no_empty_cache_after", action="store_true")
    p.add_argument("--fast_eval_selftest", action="store_true",
                   help="On the first eval, verify batched output matches batch_size=1")

    p.add_argument("--no_save_predictions", action="store_true")
    p.add_argument("--save_predictions_interval", type=int, default=1)
    p.add_argument("--save_predictions_limit", type=int, default=0)

    default_dataset = Path(__file__).resolve().parent.parent.parent / "dataset"
    p.add_argument("--heuristic_util_folder", type=str,
                   default=str(default_dataset / "data" / "runs" / "golden100_gt"))
    p.add_argument("--heuristic_config_path", type=str,
                   default=str(default_dataset / "configs" / "run.yaml"))
    p.add_argument("--heuristic_script_dir", type=str,
                   default=str(default_dataset / "scripts"))
    p.add_argument("--heuristic_interval", type=int, default=1)
    p.add_argument("--no_heuristic_eval", action="store_true")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset_num_proc", type=int, default=1,
                   help="Number of workers for parallel tokenization (reduce if OOM during data loading)")
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
    plain-bf16 checkpoints. Fail loudly instead. Raises on every rank, so a DDP
    run dies uniformly rather than hanging on a mismatched collective.
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


def _resolve_resume_checkpoint(output_dir: str, requested):
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
        eval_callback.best_metric_value = info.get(
            "best_metric_value", eval_callback.best_metric_value)
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


def normalize_rows(rows, input_field, output_field, add_ir_prefix=True, cap=-1):
    """Coerce any row source into the flat list[dict] the callback expects.

    Doing this in the caller rather than the callback is what keeps the eval plan
    rank-invariant, and it sidesteps a real hazard: TRL's _prepare_dataset may
    drop the raw string columns despite remove_unused_columns=False, in which
    case reading them at eval time yields empty strings and produces zero
    predictions silently.

    It is also where the IR_PREFIX asymmetry gets fixed. format_example bakes the
    prefix into the `text` column only (dataloader.py:79), leaving response_text
    raw, while load_test_data pre-prefixes its input (dataloader.py:210). So
    validation prompts were missing the prefix the model trained on. The
    startswith guard makes this idempotent for already-prefixed test rows.
    """
    out = []
    for r in rows or ():
        inp = r.get(input_field) or ""
        ref = r.get(output_field) or ""
        if isinstance(ref, (dict, list)):
            ref = json.dumps(ref, ensure_ascii=False)
        if not inp or not ref:
            continue
        if add_ir_prefix and not inp.startswith(IR_PREFIX):
            inp = IR_PREFIX + inp
        out.append({input_field: inp, output_field: ref})
        if cap > 0 and len(out) >= cap:
            break
    return out


def dataset_to_rows(dataset, input_field, output_field, cap=-1):
    """Materialize the needed columns off an HF Dataset into plain dicts."""
    if dataset is None:
        return []
    cols = set(getattr(dataset, "column_names", []) or [])
    if input_field not in cols or output_field not in cols:
        print(f"WARNING: eval split lacks '{input_field}'/'{output_field}' columns "
              f"(has {sorted(cols)}); validation generation disabled.")
        return []
    n = len(dataset) if cap <= 0 else min(len(dataset), cap)
    subset = dataset.select(range(n))
    return [{input_field: r.get(input_field, ""), output_field: r.get(output_field, "")}
            for r in subset]


def main():
    print("[DEBUG] main() started", flush=True)
    args = parse_args()
    print(f"[DEBUG] args parsed: {args.model_name}", flush=True)
    set_seed(args.seed)
    print(f"[DEBUG] seed set to {args.seed}", flush=True)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main = local_rank == 0

    if is_main:
        with open(os.path.join(args.output_dir, "train_config.json"), "w",
                  encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, default=str)
        if world_size > 1:
            print(f"DDP: {world_size} processes. Unsloth documents torchrun DDP as "
                  f"supported, but calls multi-GPU pre-release and documents nothing "
                  f"about the QAT combination -- watch the first eval closely.")

    print("Loading model...")
    # QAT needs a 16-bit base: fake quant is simulated during training, so stored
    # weights stay bf16. load_in_4bit defaults to True, hence the explicit False.
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_length,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"     # training side; eval pads left itself

    # Bare suffixes, not full paths -- Unsloth rejects paths with "No layers to
    # finetune?". Reads module types off the model, so plain-nn.Linear Gemmas and
    # wrapper-based ones (Gemma4ClippableLinear -> "q_proj.linear") both work.
    target_modules = resolve_target_module_suffixes(model, QAT_TARGET_MODULES)

    print("Applying LoRA + QAT...")
    # MUST be FastModel, not FastLanguageModel -- the latter silently drops
    # qat_scheme on the new-model path (UNSLOTH_USE_NEW_MODEL=1, which is what
    # Gemma-3/Gemma-4 load under), so QAT never applies and training runs in
    # plain bf16. Verified behaviorally: logits bit-identical with/without
    # qat_scheme. See test_fake_quant_active.py.
    #
    # finetune_* filters passed explicitly: FastModel defaults
    # finetune_vision_layers=True, which would attach LoRA to Gemma-4's vision/
    # audio towers. Text->JSON task, so keep adapters language-only.
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

    raw_test = None
    if args.test_data_path:
        if args.test_samples == -1 and is_main:
            print(f"WARNING: --test_samples not set (default -1 = ALL rows in "
                  f"--test_data_path will be used). This is a SEPARATE argument "
                  f"from --eval_samples, which only controls the validation "
                  f"split size carved from --data_path. --fast_eval_val_samples "
                  f"separately caps the validation-source predictions.")
        raw_test = load_test_data(
            test_path=args.test_data_path,
            input_field=args.input_field,
            output_field=args.output_field,
            num_samples=args.test_samples,
        ) or None

    # ---- normalize both generation sources up front ----
    add_prefix = not args.fast_eval_no_ir_prefix
    want = args.fast_eval_sources

    val_rows = None
    if want in ("both", "eval"):
        val_rows = normalize_rows(
            dataset_to_rows(eval_dataset, args.input_field, args.output_field,
                            cap=args.fast_eval_val_samples),
            args.input_field, args.output_field,
            add_ir_prefix=add_prefix, cap=args.fast_eval_val_samples,
        ) or None

    test_rows = None
    if want in ("both", "test"):
        test_rows = normalize_rows(
            raw_test, args.input_field, args.output_field,
            add_ir_prefix=add_prefix,
        ) or None

    if is_main:
        print(f"Generation sources: "
              f"eval={len(val_rows) if val_rows else 0} rows, "
              f"test={len(test_rows) if test_rows else 0} rows")
        if not val_rows and not test_rows:
            print("WARNING: no generation source available; only eval_loss will be tracked.")

    eval_callback = FastEvalCallback(
        tokenizer=tokenizer,
        input_field=args.input_field,
        output_field=args.output_field,
        val_data=val_rows,
        test_data=test_rows,
        max_length=args.max_length,
        max_prompt_tokens=args.fast_eval_max_prompt_tokens,
        max_new_tokens=args.fast_eval_max_new_tokens,
        batch_size=args.fast_eval_batch_size,
        min_batch_size=args.fast_eval_min_batch_size,
        token_budget=args.fast_eval_token_budget,
        sort_by_length=not args.fast_eval_no_sort_by_length,
        truncation_side=args.fast_eval_truncation_side,
        stop_strings=() if args.fast_eval_no_extra_stop_tokens else DEFAULT_STOP_STRINGS,
        prediction_interval=args.prediction_interval,
        metric_for_best_model=args.metric_for_best_model,
        best_source=args.fast_eval_best_source,
        output_dir=args.output_dir,
        save_predictions=not args.no_save_predictions,
        save_predictions_interval=args.save_predictions_interval,
        save_predictions_limit=args.save_predictions_limit,
        use_unsloth_inference=args.fast_eval_unsloth_inference,
        empty_cache_after=not args.fast_eval_no_empty_cache_after,
        selftest=args.fast_eval_selftest,
    )
    callbacks = [eval_callback]

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
            # Appended AFTER eval_callback on purpose: HF fires callbacks in
            # registration order and the pipeline reads the rolling
            # test_predictions.json that eval_callback just wrote.
            callbacks.append(heuristic_callback)
            if args.heuristic_interval != args.prediction_interval:
                print(f"WARNING: heuristic_interval ({args.heuristic_interval}) != "
                      f"prediction_interval ({args.prediction_interval}); the heuristic "
                      f"pipeline may re-score stale predictions.")
        else:
            print("Heuristic evaluation unavailable (missing util folder or pipeline script)")
            heuristic_callback = None

    sft_config = build_sft_config(args, has_eval=eval_dataset is not None)
    sft_config.dataset_num_proc = args.dataset_num_proc
    if world_size > 1:
        # With LoRA, only adapters receive gradients, not frozen base parameters.
        # Tell DDP to allow unused parameters to avoid gradient sync errors.
        sft_config.ddp_find_unused_parameters = True

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
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
    if resume_from_checkpoint:
        _restore_best_tracking(args.output_dir, eval_callback, heuristic_callback)

    # Log eval config for consistency: convert_qat_to_quantized.py should use
    # the same eval_samples and seed to evaluate on identical samples.
    # DDP rank 0 only (multiple ranks would write the same file simultaneously).
    if trainer.is_world_process_zero():
        eval_config = {
            "eval_samples": args.fast_eval_val_samples,
            "eval_batch_size": args.fast_eval_batch_size,
            "max_new_tokens": args.fast_eval_max_new_tokens if args.fast_eval_max_new_tokens > 0 else None,
            "seed": args.seed,
            "qat_scheme": args.qat_scheme,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "note": f"During conversion, use: --eval_samples {args.fast_eval_val_samples} --seed {args.seed} "
                   f"to evaluate on the same {args.fast_eval_val_samples} samples deterministically."
        }
        eval_config_path = os.path.join(args.output_dir, "eval_config.json")
        with open(eval_config_path, "w") as f:
            json.dump(eval_config, f, indent=2)
        print(f"Eval config saved to: {eval_config_path}")
        print(f"  For later conversion consistency:")
        print(f"    eval_samples: {args.fast_eval_val_samples}")
        print(f"    seed: {args.seed}")

    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if trainer.is_world_process_zero():
        adapter_dir = os.path.join(args.output_dir, "adapter")
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        print(f"Adapter saved to: {adapter_dir}")

        # Convert the fake-quantized (STE) weights into a real quantized
        # checkpoint. https://unsloth.ai/docs/blog/quantization-aware-training-qat
        try:
            from torchao.quantization import quantize_
            from torchao.quantization.qat import QATConfig

            quantize_(trainer.model, QATConfig(step="convert"))
            trainer.model.save_pretrained_torchao(
                trainer.model, tokenizer,
                torchao_config=trainer.model._torchao_config.base_config,
            )
            print("Real quantized (torchao) checkpoint saved.")
        except Exception as exc:
            print(f"WARNING: torchao QAT convert/export failed ({exc}); "
                  f"the bf16 LoRA adapter above is still valid.")

        print_best_checkpoints(args, eval_callback, heuristic_callback)


if __name__ == "__main__":
    print("[DEBUG] Script entry point reached", flush=True)
    try:
        main()
    except Exception as e:
        print(f"[DEBUG] Exception in main(): {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise
