"""
QAT + LoRA training (QA-LoRA style).

The frozen base weights are fake-quantized on every forward pass with a straight-through
estimator (group-wise symmetric INT4 by default, plus per-token dynamic INT8
activations), while only the LoRA adapters train in bf16. The adapters therefore learn
to compensate for the quantization error they will actually face at inference time --
which is what a PTQ-then-LoRA pipeline cannot do.

Backend: pure PyTorch (see qat_utils.py). No torchao, bitsandbytes or custom kernels.

Usage:
    Single GPU:  python train_qat_lora.py --gpus 0 --model_name MODEL --data_path DATA \
                     --output_dir OUT --test_data_path TEST
    Multi-GPU:   CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 \
                     --mixed_precision bf16 train_qat_lora.py --model_name MODEL ...
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
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import set_seed  # noqa: E402
from trl import SFTTrainer  # noqa: E402

from dataloader import load_dataset, load_test_data  # noqa: E402
from qat_utils import (  # noqa: E402
    add_common_args,
    add_qat_args,
    apply_fake_quant,
    build_callbacks,
    build_sft_config,
    export_int4_weights,
    load_model,
    load_tokenizer,
    print_best_checkpoints,
    resolve_target_modules,
    set_fake_quant_enabled,
)


def parse_args():
    parser = argparse.ArgumentParser(description="QAT + LoRA SFT training")
    add_common_args(parser)
    add_qat_args(parser)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_layers", type=str, default="all",
                        help="'all', 'early', 'late', 'middle', or comma-separated indices")
    parser.add_argument("--merge_before_export", action="store_true",
                        help="Merge the LoRA adapter into the base weights before the "
                             "INT4 export, so the exported tensors contain the tuned "
                             "weights instead of the untouched pretrained ones")
    parser.add_argument("--no_int4_export", action="store_true",
                        help="Skip writing int4_weights.pt")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    return args


def main():
    args = parse_args()

    print("=" * 60)
    print("QAT + LoRA Training (pure-PyTorch fake quant)")
    print("=" * 60)
    print(f"Model:       {args.model_name}")
    print(f"Data:        {args.data_path}")
    print(f"Output:      {args.output_dir}")
    print(f"GPUs:        {torch.cuda.device_count()} visible")
    print(f"QAT scheme:  {args.qat_scheme} weights (group_size={args.qat_group_size}), "
          f"activations {'bf16' if args.qat_weight_only else 'int8 per-token'}")
    print(f"QAT warmup:  {args.qat_warmup_steps} steps")
    print("=" * 60)

    set_seed(args.seed)
    with open(os.path.join(args.output_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model_name, args.allow_missing_chat_template)

    print("Loading model...")
    model = load_model(args.model_name)

    # 1. LoRA first, so the fake-quant patch lands on `base_layer` and PEFT's frozen
    #    base-weight guarantee holds.
    print("Applying LoRA...")
    target_modules = resolve_target_modules(model, args.lora_layers, args.qat_target_modules)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    ))
    model.print_trainable_parameters()

    # 2. Patch the frozen base linears with fake quantization.
    print("Applying fake quantization...")
    weight_bits = 4 if args.qat_scheme == "int4" else 8
    activation_bits = 32 if args.qat_weight_only else 8
    patched = apply_fake_quant(
        model,
        target_modules=args.qat_target_modules,
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        group_size=args.qat_group_size,
    )
    if patched == 0:
        raise RuntimeError(
            f"No layers were fake-quantized. Check --qat_target_modules "
            f"({args.qat_target_modules}) against the model's linear layer names."
        )
    # QATWarmupCallback owns the on/off switch from here; set the initial state so a
    # crash before the first step_end can't leave it ambiguous.
    set_fake_quant_enabled(model, args.qat_warmup_steps <= 0)

    # Gradient checkpointing with a mostly-frozen model needs input grads enabled,
    # otherwise the checkpointed blocks have no grad path back to the adapters.
    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

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
        args, tokenizer, test_data, backend="torch"
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

    if trainer.is_world_process_zero():
        adapter_dir = os.path.join(args.output_dir, "adapter")
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        print(f"Adapter saved to: {adapter_dir}")

        if not args.no_int4_export:
            export_model = trainer.model
            if args.merge_before_export:
                print("Merging LoRA into the base weights before INT4 export...")
                export_model = export_model.merge_and_unload()
            else:
                print("NOTE: exporting INT4 weights WITHOUT merging the adapter. The base "
                      "weights were frozen during training, so int4_weights.pt is a "
                      "quantization of the original pretrained weights and the learned "
                      "adaptation lives only in adapter/. Pass --merge_before_export to "
                      "fold the adapter in first.")
            export_int4_weights(
                export_model,
                target_modules=args.qat_target_modules,
                group_size=args.qat_group_size,
                output_path=os.path.join(args.output_dir, "int4_weights.pt"),
            )
            with open(os.path.join(args.output_dir, "qat_config.json"), "w",
                      encoding="utf-8") as f:
                json.dump({
                    "backend": "pure-pytorch fake quant (STE)",
                    "weight_bits": weight_bits,
                    "activation_bits": activation_bits,
                    "group_size": args.qat_group_size,
                    "symmetric": True,
                    "target_modules": list(args.qat_target_modules),
                    "warmup_steps": args.qat_warmup_steps,
                    "patched_layers": patched,
                    "adapter_merged_into_export": bool(args.merge_before_export),
                    "int4_weights_layout": "q int8 in [-8,7], scale (out_features, num_groups); "
                                           "dequantize as q.reshape(out,-1,group_size) * "
                                           "scale.unsqueeze(-1)",
                }, f, indent=2)

        print_best_checkpoints(args, eval_callback, heuristic_callback)


if __name__ == "__main__":
    main()
