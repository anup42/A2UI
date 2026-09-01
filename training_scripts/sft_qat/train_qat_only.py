"""
QAT only -- full-parameter quantization-aware training with torchao 8da4w.

Every weight trains (no LoRA) while torchao's Int8DynActInt4WeightQATQuantizer inserts
int4 weight / int8 dynamic-activation fake quantizers, so the whole network adapts to the
quantization grid. After training, `quantizer.convert()` produces a real quantized model.

This keeps TRL's SFTTrainer (rather than a hand-rolled loop) so it shares the accelerate
/ DDP setup, the distributed generation-based evaluation, the heuristic pipeline and the
TensorBoard layout with train_qat_lora.py and train_gemma.py -- the metrics are directly
comparable.

Requires:  pip install torchao

Usage:
    Single GPU:  python train_qat_only.py --gpus 0 --model_name MODEL --data_path DATA \
                     --output_dir OUT --test_data_path TEST
    Multi-GPU:   CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 \
                     --mixed_precision bf16 train_qat_only.py --model_name MODEL ...
"""

import argparse
import json
import os
import sys

# --- must run before torch is imported so the mask actually takes effect ---
_gpus = ""
for _i, _tok in enumerate(sys.argv):
    if _tok == "--gpus" and _i + 1 < len(sys.argv):
        _gpus = sys.argv[_i + 1]
    elif _tok.startswith("--gpus="):
        _gpus = _tok.split("=", 1)[1]
if _gpus:
    if "LOCAL_RANK" in os.environ or "CUDA_VISIBLE_DEVICES" in os.environ:
        print(f"[*] Ignoring --gpus '{_gpus}': the launcher already assigned devices.")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = _gpus
        print(f"[*] Set CUDA_VISIBLE_DEVICES to '{_gpus}'")

import torch  # noqa: E402
from transformers import set_seed  # noqa: E402
from trl import SFTTrainer  # noqa: E402

from dataloader import load_dataset, load_test_data  # noqa: E402
from qat_utils import (  # noqa: E402
    add_common_args,
    add_qat_args,
    build_callbacks,
    build_sft_config,
    load_model,
    load_tokenizer,
    load_torchao_qat_quantizer,
    print_best_checkpoints,
    set_torchao_fake_quant_enabled,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full-parameter QAT training with torchao 8da4w"
    )
    add_common_args(parser)
    add_qat_args(parser)
    parser.add_argument("--no_int4_export", action="store_true",
                        help="Skip the final quantizer.convert() export")
    parser.set_defaults(lr=2e-5)  # full finetuning wants a lower LR than LoRA
    args = parser.parse_args()

    # torchao's 8da4w is int8-dynamic-activation by construction; silently ignoring
    # these would misreport what was actually trained.
    if args.qat_weight_only:
        parser.error(
            "--qat_weight_only is not supported by the torchao 8da4w quantizer "
            "(activations are int8-dynamic by design). Use train_qat_lora.py "
            "--qat_weight_only for weight-only QAT."
        )
    if args.qat_scheme != "int4":
        parser.error(
            f"--qat_scheme {args.qat_scheme} is not supported here; the torchao 8da4w "
            f"quantizer is int4-weight only. Use train_qat_lora.py --qat_scheme int8."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    return args


def main():
    args = parse_args()

    print("=" * 60)
    print("QAT Training (torchao 8da4w, full-parameter)")
    print("=" * 60)
    print(f"Model:       {args.model_name}")
    print(f"Data:        {args.data_path}")
    print(f"Output:      {args.output_dir}")
    print(f"GPUs:        {torch.cuda.device_count()} visible")
    print(f"QAT scheme:  int8 dynamic activations / int4 weights "
          f"(group_size={args.qat_group_size})")
    print(f"QAT warmup:  {args.qat_warmup_steps} steps")
    print("=" * 60)

    set_seed(args.seed)
    with open(os.path.join(args.output_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model_name, args.allow_missing_chat_template)

    print("Loading model...")
    model = load_model(args.model_name)

    # Prepare BEFORE anything captures parameter references (gradient checkpointing is
    # enabled by the Trainer, and the optimizer is built inside trainer.train()).
    # torchao swaps each nn.Linear for a fake-quantized subclass, so preparing later
    # would leave stale parameter references behind.
    print("Preparing model for QAT (torchao 8da4w)...")
    quantizer = load_torchao_qat_quantizer(
        group_size=args.qat_group_size,
        precision=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model = quantizer.prepare(model)
    model.config.use_cache = False

    if args.qat_warmup_steps > 0:
        toggled = set_torchao_fake_quant_enabled(model, False)
        if toggled == 0:
            print(f"WARNING: this torchao build exposes no toggleable fake quantizers, so "
                  f"--qat_warmup_steps {args.qat_warmup_steps} cannot be honoured. "
                  f"Quantization is active from step 0.")
            args.qat_warmup_steps = 0
        else:
            print(f"  Disabled {toggled} fake quantizers for warmup")

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
        test_data = load_test_data(
            test_path=args.test_data_path,
            input_field=args.input_field,
            output_field=args.output_field,
            num_samples=args.test_samples,
        ) or None
        if test_data:
            print(f"Test data: {len(test_data)} samples; "
                  f"best model selected by {args.metric_for_best_model}")
        else:
            print("WARNING: no test data loaded, falling back to the eval split.")

    callbacks, eval_callback, heuristic_callback = build_callbacks(
        args, tokenizer, test_data, backend="torchao", expect_frozen_base=False
    )

    trainer = SFTTrainer(
        model=model,
        args=build_sft_config(args, has_eval=eval_dataset is not None),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    print("Starting training...")
    train_result = trainer.train()

    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    if not trainer.is_world_process_zero():
        return

    # 1. Fake-quantized model, still in the prepared (trainable) form. Reloading it
    #    requires re-running quantizer.prepare() on the base architecture first --
    #    recorded in qat_config.json next to it.
    model_dir = os.path.join(args.output_dir, "model")
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Fake-quantized model saved to: {model_dir}")

    with open(os.path.join(args.output_dir, "qat_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "backend": "torchao Int8DynActInt4WeightQATQuantizer (8da4w)",
            "weight_bits": 4,
            "activation_bits": 8,
            "activation_granularity": "per-token dynamic",
            "group_size": args.qat_group_size,
            "warmup_steps": args.qat_warmup_steps,
            "model_dir": "model (prepared/fake-quantized; call quantizer.prepare() on the "
                         "base architecture before load_state_dict)",
            "converted_dir": None if args.no_int4_export else "model_int4/qat_8da4w.pt",
        }, f, indent=2)

    # 2. Real quantized export. Guarded so a convert failure cannot discard an
    #    otherwise-complete training run.
    if not args.no_int4_export:
        print("Converting to a real quantized model...")
        try:
            export_model = trainer.model
            export_model.eval()
            with torch.no_grad():
                converted = quantizer.convert(export_model)
            int4_dir = os.path.join(args.output_dir, "model_int4")
            os.makedirs(int4_dir, exist_ok=True)
            torch.save(converted.state_dict(), os.path.join(int4_dir, "qat_8da4w.pt"))
            tokenizer.save_pretrained(int4_dir)
            print(f"Converted int4 model saved to: {os.path.join(int4_dir, 'qat_8da4w.pt')}")
        except Exception as exc:
            print(f"ERROR: quantizer.convert() failed: {exc}")
            print(f"       Training results and {model_dir} are intact; you can re-run "
                  f"convert offline on that checkpoint.")

    print_best_checkpoints(args, eval_callback, heuristic_callback)


if __name__ == "__main__":
    main()
