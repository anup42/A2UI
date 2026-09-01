#!/usr/bin/env python3
"""Evaluate a TorchAO quantized checkpoint (.pt) on test data.

This script loads a quantized model produced by either:
  - convert_to_torchao.py  (quantized.pt)
  - merge_and_quantize.py  (merged_quantized.pt)

Key features:
  - Loads torchao .pt checkpoint (reads quantization.json for metadata)
  - KV cache enabled for faster decoding
  - NO tree-based metrics (no MetricsAggregator)
  - Multi-GPU support via Accelerate
  - Optional heuristic pipeline

Usage:
    Single GPU:
      python eval_quantized.py \
          --model_name /path/to/base_model \
          --checkpoint /path/to/quantized.pt \
          --data_path test.jsonl \
          --output_dir ./eval_output

    Multi-GPU:
      accelerate launch eval_quantized.py \
          --model_name /path/to/base_model \
          --checkpoint /path/to/merged_quantized.pt \
          --data_path test.jsonl \
          --output_dir ./eval_output
"""

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import (
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    quantize_,
)
from tqdm import tqdm
from accelerate import Accelerator


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a TorchAO quantized checkpoint (.pt)"
    )

    # Model
    parser.add_argument("--model_name", type=str, required=True,
                        help="Original base model name or path (for architecture + tokenizer)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to quantized .pt checkpoint (quantized.pt or merged_quantized.pt)")

    # Data
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to test JSON/JSONL file")
    parser.add_argument("--input_field", type=str, default="response_text",
                        help="Field name for input text in the test data")
    parser.add_argument("--output_field", type=str, default="genui_json",
                        help="Field name for reference output in the test data")
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="Number of samples to evaluate (-1 for all)")

    # Generation
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Max input tokens for tokenization")
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Max new tokens to generate")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for generation")

    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for predictions")

    # Heuristic pipeline
    parser.add_argument("--run_heuristic", action="store_true",
                        help="Run heuristic pipeline after evaluation")
    parser.add_argument("--heuristic_util_folder", type=str, default=None,
                        help="Path to folder containing queries.jsonl, responses.jsonl, etc.")
    parser.add_argument("--heuristic_config_path", type=str, default=None,
                        help="Path to run.yaml config file for heuristic pipeline")
    parser.add_argument("--heuristic_script_dir", type=str, default="",
                        help="Directory containing run_heuristic_pipeline.py")

    # Other
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    return args


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_path, input_field, output_field, num_samples=-1):
    """Load test data from JSON/JSONL file."""
    with open(data_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("["):
            data = json.loads(content)
        else:
            data = [json.loads(line) for line in content.split("\n") if line.strip()]

    # Filter valid (must have both input and output)
    data = [ex for ex in data if ex.get(input_field) and ex.get(output_field)]

    if num_samples > 0:
        data = data[:num_samples]

    return data


# ---------------------------------------------------------------------------
# Quantization config (must match the checkpoint's config)
# ---------------------------------------------------------------------------

def make_quant_config(bits, group_size, int4_algorithm="hqq", int4_packing="tile_packed_to_4d"):
    """Recreate the same TorchAO config that was used to create the checkpoint."""
    if bits == 4:
        return Int4WeightOnlyConfig(
            group_size=group_size,
            int4_choose_qparams_algorithm=int4_algorithm,
            int4_packing_format=int4_packing,
            set_inductor_config=False,
        )
    elif bits == 8:
        return Int8DynamicActivationInt8WeightConfig()
    else:
        raise ValueError(f"Unsupported bits: {bits}")


def is_linear(module, name):
    """Quantize Linear layers, but leave lm_head in floating point."""
    return isinstance(module, nn.Linear) and not name.endswith("lm_head")


# ---------------------------------------------------------------------------
# Model loading: load base + apply quant config + load state dict
# ---------------------------------------------------------------------------

def load_quantized_model(args, device):
    """
    Load a TorchAO quantized checkpoint.

    Steps:
      1. Read quantization.json from checkpoint dir (bits, group_size, etc.)
      2. Load base model architecture (bfloat16)
      3. Apply same torchao quantization config (sets up tensor subclasses)
      4. torch.load(checkpoint, weights_only=False) -> load_state_dict()
      5. Enable KV cache, set eval mode
    """
    checkpoint_path = Path(args.checkpoint)
    checkpoint_dir = checkpoint_path.parent

    # Step 1: Read quantization.json metadata
    quant_meta_path = checkpoint_dir / "quantization.json"
    if quant_meta_path.exists():
        with open(quant_meta_path, "r", encoding="utf-8") as f:
            quant_meta = json.load(f)
        bits = int(quant_meta.get("bits", 4))
        group_size = int(quant_meta.get("group_size", 128))
        int4_algorithm = quant_meta.get("int4_algorithm", "hqq")
        int4_packing = quant_meta.get("int4_packing_format", "tile_packed_to_4d")
        print(f"  Quantization metadata: bits={bits}, group_size={group_size}, "
              f"algorithm={int4_algorithm}, packing={int4_packing}")
    else:
        # Fallback: try to read from checkpoint itself
        print(f"  WARNING: quantization.json not found at {quant_meta_path}")
        print(f"  Attempting to read metadata from checkpoint...")
        # Peek at checkpoint metadata (load without state_dict)
        peek = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        if isinstance(peek, dict):
            bits = int(peek.get("bits", 4))
            group_size = int(peek.get("group_size", 128))
            int4_algorithm = peek.get("int4_algorithm", "hqq")
            int4_packing = peek.get("int4_packing_format", "tile_packed_to_4d")
        else:
            bits, group_size = 4, 128
            int4_algorithm, int4_packing = "hqq", "tile_packed_to_4d"
        del peek
        print(f"  Checkpoint metadata: bits={bits}, group_size={group_size}")

    # Step 2: Load base model architecture (bfloat16)
    print(f"\n  Loading base model from {args.model_name} (bfloat16)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=args.trust_remote_code,
    )

    # Move to GPU BEFORE quantizing — tile_packed_to_4d uses CUDA-only kernels
    # (aten::_convert_weight_to_int4pack) that don't exist on CPU.
    print(f"  Moving model to {device}...")
    model = model.to(device)

    # Step 3: Apply same quantization config (sets up tensor subclasses)
    print(f"  Applying torchao quantization config (bits={bits})...")
    config = make_quant_config(bits, group_size, int4_algorithm, int4_packing)
    quantize_(model, config, filter_fn=is_linear)

    # Step 4: Load quantized state dict
    print(f"  Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(
        str(checkpoint_path),
        map_location=device,
        weights_only=False,  # required for torchao tensor subclasses
    )
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    print(f"  ✓ Checkpoint loaded")

    # Step 5: Move to device, enable KV cache, eval mode
    model = model.to(device)
    model.config.use_cache = True  # Enable KV cache for faster decoding
    model.eval()
    print(f"  ✓ Model ready (KV cache enabled)")

    return model


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_predictions(model, tokenizer, data, args, device):
    """Generate predictions for all test samples."""
    print("\n" + "=" * 60)
    print("GENERATING PREDICTIONS")
    print("=" * 60)

    inputs = [ex[args.input_field] for ex in data]
    references = [ex[args.output_field] for ex in data]

    predictions = []

    for i in tqdm(range(0, len(inputs), args.batch_size), desc="Generating"):
        batch_inputs = inputs[i:i + args.batch_size]
        batch_refs = references[i:i + args.batch_size]

        # Format with chat template
        prompts = []
        for inp in batch_inputs:
            messages = [{"role": "user", "content": inp}]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt = f"User: {inp}\nAssistant: "
            prompts.append(prompt)

        # Tokenize
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,  # Explicitly enable KV cache
            )

        # Decode only new tokens
        for j, output in enumerate(outputs):
            input_len = int(encoded["attention_mask"][j].sum().item())
            pred = tokenizer.decode(
                output[input_len:], skip_special_tokens=True
            ).strip()
            predictions.append(pred)

        if (i // args.batch_size + 1) % 5 == 0 or i + args.batch_size >= len(inputs):
            print(f"  Generated {min(i + args.batch_size, len(inputs))}/{len(inputs)}")

    return predictions, references


# ---------------------------------------------------------------------------
# Heuristic pipeline
# ---------------------------------------------------------------------------

def run_heuristic_pipeline(args, predictions_path):
    """Run the heuristic pipeline on predictions.jsonl."""
    print("\n" + "=" * 60)
    print("RUNNING HEURISTIC PIPELINE")
    print("=" * 60)

    # Auto-detect script dir if not provided
    script_dir = args.heuristic_script_dir
    if not script_dir:
        script_dir = str(
            Path(__file__).resolve().parents[4] / "dataset" / "scripts"
        )
        print(f"  Auto-detected heuristic script dir: {script_dir}")

    pipeline_script = os.path.join(script_dir, "run_heuristic_pipeline.py")
    if not os.path.exists(pipeline_script):
        print(f"  ERROR: run_heuristic_pipeline.py not found at {pipeline_script}")
        print(f"  Pass --heuristic_script_dir to specify the correct path")
        return None

    if not args.heuristic_util_folder:
        print(f"  ERROR: --heuristic_util_folder not provided")
        print(f"  Skipping heuristic pipeline")
        return None

    cmd = [
        sys.executable,
        pipeline_script,
        "--input", str(predictions_path),
        "--util-folder", str(args.heuristic_util_folder),
    ]
    if args.heuristic_config_path:
        cmd.extend(["--config", str(args.heuristic_config_path)])

    print(f"  Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"  ERROR: Heuristic pipeline failed!")
        return None

    # Read aggregates.json
    aggregates_path = os.path.join(args.output_dir, "aggregates.json")
    if os.path.exists(aggregates_path):
        with open(aggregates_path, "r", encoding="utf-8") as f:
            aggregates = json.load(f)
        print(f"\n  ✓ Aggregates loaded from {aggregates_path}")
        return aggregates
    else:
        print(f"  WARNING: aggregates.json not found at {aggregates_path}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Initialize accelerator for multi-GPU support
    accelerator = Accelerator()

    print("\n" + "=" * 60)
    print("EVAL: TORCHAO QUANTIZED MODEL")
    print("=" * 60)
    print(f"  Base model:       {args.model_name}")
    print(f"  Checkpoint:       {args.checkpoint}")
    print(f"  Test data:        {args.data_path}")
    print(f"  Output dir:       {args.output_dir}")
    print(f"  Max length:       {args.max_length}")
    print(f"  Max new tokens:   {args.max_new_tokens}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Device:           {accelerator.device}")
    print(f"  Num processes:    {accelerator.num_processes}")
    print(f"  Heuristic:        {'ENABLED' if args.run_heuristic else 'SKIP'}")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True,
                                               trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"  ✓ Tokenizer loaded")

    # Load quantized model
    print("\nLoading quantized model...")
    model = load_quantized_model(args, accelerator.device)

    # Load test data
    print("\nLoading test data...")
    data = load_data(args.data_path, args.input_field, args.output_field, args.num_samples)
    print(f"  Test samples: {len(data)}")

    # Split data across GPUs for parallel processing
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index

    inputs = [ex[args.input_field] for ex in data]
    references = [ex[args.output_field] for ex in data]

    local_inputs = [inputs[i] for i in range(len(inputs)) if i % num_processes == process_index]
    local_references = [references[i] for i in range(len(references)) if i % num_processes == process_index]

    # Create local data subset for this GPU
    local_data = [data[i] for i in range(len(data)) if i % num_processes == process_index]

    if accelerator.is_main_process:
        print(f"  Processing {len(local_inputs)}/{len(inputs)} samples on this GPU")

    # Generate predictions for this process's subset (pass local_data, not full data!)
    local_predictions_path = os.path.join(args.output_dir, f"predictions_p{process_index}.jsonl")
    local_predictions, local_refs = generate_predictions(
        model, tokenizer, local_data, args, accelerator.device
    )

    # Write local predictions
    with open(local_predictions_path, "w", encoding="utf-8") as f:
        for i, (pred, ref) in enumerate(zip(local_predictions, local_refs)):
            f.write(json.dumps({
                "index": i,
                "prediction": pred,
                "reference": ref,
            }, ensure_ascii=False) + "\n")

    # Wait for all processes
    accelerator.wait_for_everyone()

    # Free model memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Gather all predictions on main process
    if accelerator.is_main_process:
        print("\nCombining predictions from all GPUs...")

        all_predictions_data = []
        for p in range(num_processes):
            local_path = os.path.join(args.output_dir, f"predictions_p{p}.jsonl")
            if os.path.exists(local_path):
                with open(local_path, "r", encoding="utf-8") as f:
                    for line in f:
                        all_predictions_data.append(json.loads(line))
                os.remove(local_path)

        # Sort by index
        all_predictions_data.sort(key=lambda x: x["index"])

        # Filter out entries without required fields (e.g., metadata headers, empty entries)
        all_predictions_data = [
            item for item in all_predictions_data
            if "index" in item and "prediction" in item and "reference" in item
        ]

        # Write combined predictions
        predictions_path = os.path.join(args.output_dir, "predictions.jsonl")
        with open(predictions_path, "w", encoding="utf-8") as f:
            for item in all_predictions_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Save metadata separately (not in predictions.jsonl)
        metadata = {
            "model_name": args.model_name,
            "checkpoint": args.checkpoint,
            "num_samples": len(all_predictions_data),
            "max_length": args.max_length,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "note": "Predictions from TorchAO quantized checkpoint",
        }
        metadata_path = os.path.join(args.output_dir, "eval_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Metadata saved to: {metadata_path}")

        print(f"\n✓ Predictions saved to: {predictions_path}")

        predictions = [p["prediction"] for p in all_predictions_data]
        references = [p["reference"] for p in all_predictions_data]

        # Run heuristic pipeline
        if args.run_heuristic:
            aggregates = run_heuristic_pipeline(args, predictions_path)

            if aggregates:
                print("\n" + "=" * 60)
                print("HEURISTIC METRICS")
                print("=" * 60)
                print(f"  overall_score:           {aggregates.get('overall_score', 'N/A')}")
                print(f"  media_score:              {aggregates.get('media_score', 'N/A')}")
                print(f"  schema_valid_strict_rate: {aggregates.get('schema_valid_strict_rate', 'N/A')}")
                print(f"  content_coverage_avg:     {aggregates.get('content_coverage_avg', 'N/A')}")
                print(f"  intent_score_avg:         {aggregates.get('intent_score_avg', 'N/A')}")
                print(f"  action_coverage_avg:      {aggregates.get('action_coverage_avg', 'N/A')}")
                print(f"  lint_score_avg:            {aggregates.get('lint_score_avg', 'N/A')}")
                print(f"  dup_rate_avg:             {aggregates.get('dup_rate_avg', 'N/A')}")
                print("=" * 60)

                # Save eval summary
                summary = {
                    "model_name": args.model_name,
                    "checkpoint": args.checkpoint,
                    "num_samples": len(all_predictions_data),
                    "overall_score": aggregates.get("overall_score"),
                    "media_score": aggregates.get("media_score"),
                    "schema_valid_strict_rate": aggregates.get("schema_valid_strict_rate"),
                    "content_coverage_avg": aggregates.get("content_coverage_avg"),
                    "intent_score_avg": aggregates.get("intent_score_avg"),
                    "action_coverage_avg": aggregates.get("action_coverage_avg"),
                    "lint_score_avg": aggregates.get("lint_score_avg"),
                    "dup_rate_avg": aggregates.get("dup_rate_avg"),
                }
                summary_path = os.path.join(args.output_dir, "eval_summary.json")
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
                print(f"\n  Summary saved to: {summary_path}")
        else:
            print(f"\n  Heuristic pipeline skipped (--run_heuristic not set)")

        # Print sample
        if predictions:
            print("\n" + "-" * 60)
            print("SAMPLE OUTPUT")
            print("-" * 60)
            print(f"Input:      {inputs[0][:200]}...")
            print(f"Reference:  {references[0][:200]}...")
            print(f"Prediction: {predictions[0][:200]}...")
            print("-" * 60)

        print("\n" + "=" * 60)
        print("✓ EVAL COMPLETE")
        print("=" * 60)
    else:
        print(f"Process {process_index} finished.")


if __name__ == "__main__":
    main()
