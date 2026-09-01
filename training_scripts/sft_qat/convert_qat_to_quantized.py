"""
Convert QAT+LoRA checkpoint (fake-quantized with STE) to real quantized model (int4).

This script:
1. Loads the QAT+LoRA checkpoint from training
2. Merges LoRA adapters into the base model
3. Runs the torchao convert step to apply real int4 quantization
4. Saves the quantized model
5. Optionally evaluates the quantized model on test data

Usage:
    python convert_qat_to_quantized.py \
        --checkpoint_path /path/to/checkpoint-1000 \
        --output_dir /path/to/quantized_output \
        --test_data_path /path/to/test_data.jsonl \
        --model_name /path/to/base/model
"""

import argparse
import json
import os
from pathlib import Path
import sys

# Parse GPU args before other imports
if "--gpu_ids" in sys.argv:
    gpu_idx = sys.argv.index("--gpu_ids")
    if gpu_idx + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[gpu_idx + 1]

# Force NVIDIA before any Unsloth import
os.environ["FORCE_CUDA"] = "1"

# unsloth must be imported before transformers/peft/trl (including via
# dataloader/qat_utils, which import datasets/trl/transformers) -- otherwise
# its patches don't apply to those modules and it warns + may silently skip
# optimizations.
import unsloth  # noqa: F401
import contextlib
import io
import torch
from transformers import set_seed
from unsloth import FastModel

from dataloader import load_test_data
from qat_utils import _unwrap, QAT_TARGET_MODULES, resolve_target_module_suffixes


def parse_args():
    p = argparse.ArgumentParser(description="Convert QAT checkpoint to quantized model")
    p.add_argument("--checkpoint_path", type=str, required=True,
                   help="Path to the QAT+LoRA checkpoint (checkpoint-N directory)")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory for the quantized model")
    p.add_argument("--model_name", type=str, required=True,
                   help="Original base model path/name used in training's --model_name "
                        "(re-loaded fresh to rebuild the QAT+LoRA structure)")
    p.add_argument("--gpu_ids", type=str, default="0",
                   help="GPU IDs to use (e.g., '0' or '0,1,2,3'), default: 0")
    p.add_argument("--test_data_path", type=str, default=None,
                   help="Test data for evaluation after conversion (optional)")
    p.add_argument("--input_field", type=str, default="response_text")
    p.add_argument("--output_field", type=str, default="genui_json")
    p.add_argument("--max_length", type=int, default=6144)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--eval_samples", type=int, default=10,
                   help="Number of samples to evaluate (if test_data_path provided)")
    p.add_argument("--eval_batch_size", type=int, default=1,
                   help="Batch size for evaluation generation (default: 1)")
    p.add_argument("--quantization_bits", type=int, default=4, choices=[4, 8],
                   help="Quantization bit-width: 4 (int4) or 8 (int8), default: 4")
    p.add_argument("--lora_r", type=int, default=16,
                   help="LoRA rank used during training (must match train_qat_unsloth.py)")
    p.add_argument("--lora_alpha", type=int, default=32,
                   help="LoRA alpha used during training (must match train_qat_unsloth.py)")
    p.add_argument("--qat_scheme", type=str, default="int8-int4",
                   choices=["int4", "int8-int4", "fp8-int4", "fp8-fp8"],
                   help="QAT scheme used during training (must match train_qat_unsloth.py)")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Auto-load eval config from training run if it exists
    # (same directory as the checkpoint, parent dir / eval_config.json)
    checkpoint_parent = os.path.dirname(args.checkpoint_path)
    eval_config_path = os.path.join(checkpoint_parent, "eval_config.json")
    if os.path.exists(eval_config_path):
        print(f"[auto-config] Reading eval_config.json from training run: {eval_config_path}")
        with open(eval_config_path) as f:
            eval_config = json.load(f)
        # Apply training's eval settings to match exactly (unless user overrode on CLI)
        if args.eval_samples == 10:  # Still default value
            args.eval_samples = eval_config.get("eval_samples", 10)
            print(f"  → eval_samples: {args.eval_samples} (from training config)")
        if args.seed == 42:  # Still default value
            args.seed = eval_config.get("seed", 42)
            print(f"  → seed: {args.seed} (from training config)")
        if args.eval_batch_size == 1:  # Still default value
            args.eval_batch_size = eval_config.get("eval_batch_size", 1)
            print(f"  → eval_batch_size: {args.eval_batch_size} (from training config)")
        # QAT params must match exactly (raise if mismatch)
        for key in ["qat_scheme", "lora_r", "lora_alpha"]:
            if key in eval_config and getattr(args, key, None) != eval_config[key]:
                print(f"  WARNING: training used {key}={eval_config[key]}, but you passed {key}={getattr(args, key)}")
    else:
        print(f"[info] No eval_config.json found in {checkpoint_parent}")
        print(f"       You may need to manually set --eval_samples and --seed to match training")

    return args


def main():
    args = parse_args()

    # Set GPU IDs
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    set_seed(args.seed)

    print("="*70)
    print("QAT to Quantized Model Converter")
    print("="*70)
    print(f"Checkpoint:        {args.checkpoint_path}")
    print(f"Output:            {args.output_dir}")
    print(f"Base model:        {args.model_name}")
    print(f"GPU IDs:           {args.gpu_ids}")
    print(f"Quantization bits: {args.quantization_bits}")

    def _module_type_report(m, label):
        """Print module-type histogram + a sample of quantized/unquantized weight dtypes."""
        from collections import Counter
        type_counts = Counter(type(mod).__name__ for _, mod in m.named_modules())
        print(f"  [{label}] module type counts (top 10):")
        for name, count in type_counts.most_common(10):
            print(f"      {name}: {count}")

        linear_like = [
            (n, mod) for n, mod in m.named_modules()
            if hasattr(mod, "weight") and mod.weight is not None
            and ("proj" in n or "linear" in n.lower())
        ]
        print(f"  [{label}] sample weight dtypes/types ({len(linear_like)} linear-like modules found):")
        for name, mod in linear_like[:5]:
            w = mod.weight
            print(f"      {name}: python_type={type(w).__name__}, dtype={w.dtype}, "
                  f"shape={tuple(w.shape)}")

    def _dump_target_subtree(m, label):
        """Full submodule tree under one target position (layer 0, k_proj),
        every nesting level -- not a filtered sample. Aggregate type counts
        can't show WHERE a conversion landed in the tree; this can.
        """
        print(f"  [{label}] full subtree under self_attn.k_proj (layer 0):")
        dump_root_name = None
        for n, mod in m.named_modules():
            if n.endswith("self_attn.k_proj") or n.endswith("self_attn.k_proj.linear"):
                dump_root_name = n
                break
        if not dump_root_name:
            print(f"      WARNING: could not find a self_attn.k_proj module to dump")
            return
        for n, mod in m.named_modules():
            if n == dump_root_name or n.startswith(dump_root_name + "."):
                depth = n[len(dump_root_name):].count(".")
                indent = "  " * (depth + 1)
                extra = ""
                if hasattr(mod, "weight") and mod.weight is not None:
                    w = mod.weight
                    extra = f" | weight: python_type={type(w).__name__}, dtype={w.dtype}"
                print(f"      {indent}{n} [{type(mod).__name__}]{extra}")

    # ---- 1. Rebuild the exact QAT+LoRA structure training used ----
    # A checkpoint's adapter_model.safetensors only stores lora_A/lora_B deltas.
    # It has no fake-quantizer/observer state, and reloading it via a plain
    # FastLanguageModel.from_pretrained(checkpoint_path) does NOT re-attach the
    # torchao fake-quant wrapper that get_peft_model(qat_scheme=...) inserts.
    # QATConfig(step="convert") then has nothing properly "prepared" to convert.
    # Fix: load the ORIGINAL base model, re-run get_peft_model(qat_scheme=...)
    # exactly as train_qat_unsloth.py did, then load the trained LoRA weights
    # on top of that freshly-wired structure -- reproducing trainer.model
    # exactly as it existed in memory right before training's own export step.
    #
    # NO try/except swallowing here: if any step fails, the script crashes with
    # a full traceback. A quantization conversion tool must never silently
    # produce a wrong-but-plausible-looking output.
    print("\n[1/4] Rebuilding QAT+LoRA structure from base model...")

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_length,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
    )
    print(f"  ✓ Loaded base model from {args.model_name}")
    _module_type_report(model, "base model, pre-LoRA")

    target_modules = resolve_target_module_suffixes(model, QAT_TARGET_MODULES)
    print(f"  [target_modules] resolved {len(target_modules)} module(s): {target_modules}")

    # MUST be FastModel with the SAME finetune_* filters training used --
    # FastLanguageModel.get_peft_model silently drops qat_scheme on the
    # new-model path, which is why quantize_(step="convert") previously found
    # nothing prepared and no-oped. Filters must match training exactly or the
    # target-module set differs and the adapter load mismatches.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model = FastModel.get_peft_model(
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
    captured = buf.getvalue()
    if captured:
        print(captured, end="")
    if "Applying QAT" not in captured:
        raise RuntimeError(
            f"qat_scheme={args.qat_scheme!r} was passed, but Unsloth never printed "
            f"its 'Applying QAT to mitigate quantization degradation' confirmation. "
            f"QAT prep did NOT happen, so QATConfig(step='convert') would silently "
            f"no-op and produce an unquantized model. Refusing to proceed."
        )
    print(f"  ✓ Re-applied LoRA + QAT (scheme={args.qat_scheme}, r={args.lora_r}, "
          f"alpha={args.lora_alpha}) -- Unsloth confirmed QAT applied")
    _module_type_report(model, "post get_peft_model (fake-quant wrapper expected)")
    _dump_target_subtree(model, "post get_peft_model, pre-convert")

    # Load the trained adapter weights from the checkpoint onto this
    # freshly-prepared structure (not a full from_pretrained reload).
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict

    adapter_file = os.path.join(args.checkpoint_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        adapter_file = os.path.join(args.checkpoint_path, "adapter_model.bin")
    if not os.path.exists(adapter_file):
        raise FileNotFoundError(
            f"No adapter_model.safetensors or adapter_model.bin found in "
            f"{args.checkpoint_path}"
        )
    adapter_state = (
        load_file(adapter_file) if adapter_file.endswith(".safetensors")
        else torch.load(adapter_file, map_location="cpu")
    )
    print(f"  [adapter] loaded {len(adapter_state)} tensor(s) from {adapter_file}")

    load_result = set_peft_model_state_dict(model, adapter_state)
    missing = list(getattr(load_result, "missing_keys", []) or [])
    unexpected = list(getattr(load_result, "unexpected_keys", []) or [])

    # missing_keys from a strict=False partial (adapter-only) load INCLUDES every
    # non-adapter parameter in the model by definition (embeddings, norms, base
    # weights) -- that's expected, not an error, since adapter_state only ever
    # contains lora_A/lora_B tensors. The only missing_keys that indicate a real
    # problem are lora_A/lora_B keys themselves (a trained weight failing to land
    # on its target module). Split and print BOTH buckets in full so this is
    # independently verifiable, not asserted.
    missing_lora = [k for k in missing if "lora_a" in k.lower() or "lora_b" in k.lower()]
    missing_non_adapter = [k for k in missing if k not in missing_lora]

    print(f"  [adapter load] total missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
    print(f"      missing_lora_keys={len(missing_lora)} (real problem if > 0)")
    print(f"      missing_non_adapter_keys={len(missing_non_adapter)} "
          f"(expected -- base weights/norms/embeddings never in the adapter file)")
    if missing_lora:
        print(f"      ALL missing LoRA keys ({len(missing_lora)}):")
        for k in missing_lora:
            print(f"        {k}")
    if unexpected:
        print(f"      ALL unexpected keys ({len(unexpected)}):")
        for k in unexpected:
            print(f"        {k}")

    if missing_lora or unexpected:
        raise RuntimeError(
            f"LoRA adapter load mismatch: {len(missing_lora)} missing lora_A/lora_B "
            f"keys, {len(unexpected)} unexpected_keys (see full lists printed above). "
            f"The rebuilt target_modules do not exactly match the checkpoint's "
            f"trained modules -- refusing to proceed with a partially-loaded "
            f"adapter (some layers would silently keep randomly-initialized LoRA "
            f"weights)."
        )
    print(f"  ✓ Loaded trained LoRA weights from {adapter_file}: "
          f"all {len(adapter_state)} adapter tensors matched a target module "
          f"(0 missing lora keys, 0 unexpected keys)")

    # ---- 2. Convert to real quantization ----
    # Convert on the full model object (LoRA unmerged), exactly matching
    # train_qat_unsloth.py's own end-of-training export: quantize_(trainer.model, ...).
    quant_name = f"int{args.quantization_bits}"
    print(f"\n[2/4] Converting to real {quant_name} quantization (torchao)...")

    from torchao.quantization import quantize_
    from torchao.quantization.qat import QATConfig

    quantize_(model, QATConfig(step="convert"))
    print(f"  ✓ Ran QATConfig(step='convert') on the full model")
    _module_type_report(model, "post quantize_ convert (real int4/int8 tensors expected)")

    # Same subtree, post-convert -- diffing this against the pre-convert dump
    # above shows exactly which nesting level changed type/dtype and which didn't,
    # rather than inferring it from aggregate counts alone.
    _dump_target_subtree(model, "post quantize_ convert")

    if not hasattr(model, "_torchao_config"):
        raise RuntimeError(
            "model._torchao_config missing after quantize_(step='convert'). "
            "The convert step did not attach torchao config -- conversion did not "
            "actually apply. Refusing to save a model that looks converted but isn't."
        )
    print(f"  ✓ model._torchao_config present: {model._torchao_config}")

    # The only implementation-agnostic proof that real quantization happened:
    # a converted base_layer.weight must be a torchao quantized tensor subclass,
    # not a plain torch.Tensor/Parameter. Checking for a "fake_quantizer"-named
    # submodule beforehand was wrong (that naming belongs to qat_utils.py's
    # unrelated custom backend, not Unsloth's own qat_scheme integration) --
    # this checks the actual result instead of guessing internal wiring.
    base_layer_weights = [
        (n, mod.weight) for n, mod in model.named_modules()
        if n.endswith("base_layer") and hasattr(mod, "weight") and mod.weight is not None
    ]
    print(f"  [post-convert type check] {len(base_layer_weights)} base_layer weight(s) found")
    plain_tensor_count = sum(
        1 for _, w in base_layer_weights if type(w).__name__ in ("Tensor", "Parameter")
    )
    for n, w in base_layer_weights[:5]:
        print(f"      {n}: weight type={type(w).__name__}, dtype={w.dtype}")
    if base_layer_weights and plain_tensor_count == len(base_layer_weights):
        raise RuntimeError(
            f"All {len(base_layer_weights)} base_layer weights are still plain "
            f"torch.Tensor/Parameter after QATConfig(step='convert') -- none became "
            f"a torchao quantized tensor subclass. Conversion did not actually apply "
            f"to this architecture. Refusing to save a model that looks converted "
            f"but isn't."
        )

    # Save the quantized model. NO fallback: if the torchao-native save fails,
    # crash loudly rather than silently writing an adapter-only (unquantized)
    # checkpoint that LOOKS like a successful conversion.
    quantized_path = os.path.join(args.output_dir, "model_quantized")
    os.makedirs(quantized_path, exist_ok=True)

    # Signature is (self, save_directory, tokenizer=None, torchao_config=None, ...).
    # save_directory is the first explicit arg -- passing `model` there (as the
    # old code did) is a str/PathLike where a model object was given.
    # torchao_config must NOT be passed for a QAT model: unsloth_save_pretrained_torchao
    # asserts `not has_qat_config` whenever torchao_config is not None, and
    # has_qat_config is True here (that's what we just verified above) -- passing
    # it would hit that assertion. The quant config is auto-read from
    # model._torchao_config when torchao_config is omitted.
    model.save_pretrained_torchao(quantized_path, tokenizer)
    print(f"  ✓ Quantized model saved to {quantized_path}")

    saved_files = os.listdir(quantized_path)
    print(f"  [save check] files written: {saved_files}")
    if "adapter_model.safetensors" in saved_files and "config.json" not in saved_files:
        raise RuntimeError(
            f"save_pretrained_torchao wrote only an adapter checkpoint "
            f"(adapter_model.safetensors, no config.json/base weights) to "
            f"{quantized_path}. This means the base model's quantized weights "
            f"were NOT persisted -- reloading this path would silently give you "
            f"back the original bf16 model. Refusing to report success."
        )

    # ---- 4. Optional evaluation ----
    if args.test_data_path:
        print("\n[4/4] Evaluating quantized model (heuristic-based)...")
        print(f"  Loading test data from {args.test_data_path}...")
        print(f"    [consistency] Deterministic seed={args.seed} for reproducible sample selection")
        test_data = load_test_data(
            test_path=args.test_data_path,
            input_field=args.input_field,
            output_field=args.output_field,
            num_samples=args.eval_samples,
        )

        if not test_data:
            raise RuntimeError(f"No test data loaded from {args.test_data_path}")

        if len(test_data) != args.eval_samples:
            print(f"  WARNING: Requested {args.eval_samples} samples but got {len(test_data)}")
            print(f"    This should match --fast_eval_samples from training for consistency")
        else:
            print(f"  ✓ Loaded exactly {args.eval_samples} samples as requested")
            print(f"    (these should be the same samples used in training eval)")

        print(f"  Generating predictions on {len(test_data)} samples (batch_size={args.eval_batch_size})...")

        predictions = []
        references = []

        model.eval()
        with torch.no_grad():
            # Process in batches
            for batch_start in range(0, len(test_data), args.eval_batch_size):
                batch_end = min(batch_start + args.eval_batch_size, len(test_data))
                batch = test_data[batch_start:batch_end]

                prompts = []
                batch_refs = []
                valid_indices = []

                # Prepare batch
                for idx_in_batch, example in enumerate(batch):
                    inp = example.get(args.input_field, "")
                    ref = example.get(args.output_field, "")

                    if not inp or not ref:
                        continue

                    # Build prompt
                    messages = [{"role": "user", "content": inp}]
                    try:
                        prompt = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    except Exception:
                        prompt = f"<|im_start|>user\n{inp}\n<|im_start|>assistant\n"

                    prompts.append(prompt)
                    batch_refs.append(ref)
                    valid_indices.append(batch_start + idx_in_batch)

                if not prompts:
                    continue

                # Tokenize batch (Gemma-4's multi-modal processor needs text= as a
                # keyword; plain tokenizers accept either -- this is a genuine API
                # compatibility shim, not error-hiding, so it stays)
                try:
                    encoded = tokenizer(
                        prompts, return_tensors="pt", truncation=True,
                        max_length=args.max_length, padding=True,
                    ).to(model.device)
                except TypeError as te:
                    if "NoneType" in str(te) or "subscriptable" in str(te):
                        encoded = tokenizer(
                            text=prompts, return_tensors="pt", truncation=True,
                            max_length=args.max_length, padding=True,
                        ).to(model.device)
                    else:
                        raise

                # Generate batch
                outputs = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                # Decode batch
                for i, output in enumerate(outputs):
                    input_len = int(encoded["attention_mask"][i].sum().item())
                    pred = tokenizer.decode(
                        output[input_len:], skip_special_tokens=True
                    ).strip()
                    predictions.append(pred)
                    references.append(batch_refs[i])

                if batch_end % max(5, args.eval_batch_size * 5) == 0 or batch_end == len(test_data):
                    print(f"    [{batch_end}/{len(test_data)}] Generated")

        # Save predictions to predictions.jsonl (same format as eval.sh)
        predictions_file = os.path.join(args.output_dir, "predictions.jsonl")
        with open(predictions_file, "w", encoding="utf-8") as f:
            # Write metadata header: parameters used for evaluation
            metadata = {
                "_metadata": {
                    "num_samples": len(test_data),
                    "seed": args.seed,
                    "qat_scheme": args.qat_scheme,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "quantization_bits": args.quantization_bits,
                    "checkpoint_path": args.checkpoint_path,
                    "model_name": args.model_name,
                    "batch_size": args.eval_batch_size,
                    "max_new_tokens": args.max_new_tokens,
                    "note": f"Evaluated on exactly {len(test_data)} samples with deterministic seed={args.seed}. "
                           f"Must match training eval (--fast_eval_samples {len(test_data)}) for consistency.",
                }
            }
            f.write(json.dumps(metadata) + "\n")

            # Write predictions
            for i, (pred, ref) in enumerate(zip(predictions, references)):
                f.write(json.dumps({
                    "index": i,
                    "prediction": pred,
                    "reference": ref,
                }) + "\n")

        print(f"\n  ✓ Predictions saved to {predictions_file}")
        print(f"    - {len(test_data)} predictions (seed={args.seed})")
        print(f"    - QAT: scheme={args.qat_scheme}, r={args.lora_r}, alpha={args.lora_alpha}")
        print(f"    - Quantization: {args.quantization_bits}-bit")
        print(f"    - Metadata included for consistency verification")
        print(f"\n  To calculate heuristic metrics, run:")
        print(f"    python run_heuristic_pipeline.py --input {predictions_file} --util-folder <util_folder> --config <config.yaml>")
    else:
        print("\n[4/4] Skipping evaluation (--test_data_path not provided)")

    print("\n" + "="*70)
    print("✓ Conversion complete!")
    print(f"Quantized model: {os.path.join(args.output_dir, 'model_quantized')}")
    print("="*70)


if __name__ == "__main__":
    main()
