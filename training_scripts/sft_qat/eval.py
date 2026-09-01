"""
SFT Evaluation Script with Multi-GPU Support via Accelerate.

Input: agent response string
Output: JSON-like structured string

Usage:
    Single GPU:  python3 eval.py --model_name ... --adapter_dir ... --data_path ... --output_dir ...
    Multi-GPU:   accelerate launch eval.py --model_name ... --adapter_dir ... --data_path ... --output_dir ...
"""

import argparse
import json
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Qwen3VLForConditionalGeneration
from peft import PeftModel
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator

# Import metrics module
from metrics import MetricsAggregator


# Mapping from config model_type to model class
VL_MODEL_TYPES = {"qwen3_vl", "qwen2_vl", "qwen2_5_vl"}


def get_model_class(model_name: str):
    """Auto-detect the model class based on the model's config.json.

    Reads the model_type from the config and returns the appropriate class:
    - VL models (qwen3_vl, qwen2_vl, etc.) -> Qwen3VLForConditionalGeneration
    - Text-only LLMs (qwen3, qwen2, llama, etc.) -> AutoModelForCausalLM

    Returns:
        tuple: (model_class, is_vl_model)
    """
    config_path = os.path.join(model_name, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        model_type = config.get("model_type", "")
    else:
        # For HuggingFace hub models, use AutoConfig
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        model_type = getattr(config, "model_type", "")

    is_vl = model_type in VL_MODEL_TYPES

    if is_vl:
        print(f"Detected VL model (model_type={model_type}), using Qwen3VLForConditionalGeneration")
        return Qwen3VLForConditionalGeneration, True
    else:
        print(f"Detected text-only LLM (model_type={model_type}), using AutoModelForCausalLM")
        return AutoModelForCausalLM, False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Base model name or path")
    parser.add_argument("--adapter_dir", type=str, default=None, help="Path to LoRA adapter (if use_lora was used)")
    parser.add_argument("--quantization_bits", type=int, choices=[4, 8], default=None, help="Quantize the merged model before evaluation")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test JSON/JSONL file")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--input_field", type=str, default="response_text")
    parser.add_argument("--output_field", type=str, default="genui_json")
    parser.add_argument("--max_length", type=int, default=4096, help="Max input tokens for tokenization")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to evaluate (-1 for all)")
    parser.add_argument("--log_to_tensorboard", action="store_true", help="Log metrics to TensorBoard")
    parser.add_argument("--run_heuristic", action="store_true", help="Run heuristic pipeline after evaluation")
    parser.add_argument("--heuristic_util_folder", type=str, default="/home/c_kularni/Storage_gpu/AdvancedResearch_ST/c_kulkarni/mdc/GenUI-LM/dataset/data/runs/util")
    parser.add_argument("--heuristic_config_path", type=str, default="/home/c_kularni/Storage_gpu/AdvancedResearch_ST/c_kulkarni/mdc/GenUI-LM/dataset/configs/run.yaml")
    parser.add_argument("--heuristic_script_dir", type=str, default="")
    return parser.parse_args()


def load_data(data_path, input_field, output_field, num_samples=-1):
    """Load test data from JSON/JSONL file."""
    with open(data_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("["):
            data = json.loads(content)
        else:
            data = [json.loads(line) for line in content.split("\n") if line.strip()]

    # Filter valid
    data = [ex for ex in data if ex.get(input_field) and ex.get(output_field)]

    if num_samples > 0:
        data = data[:num_samples]

    return data


def generate_outputs(model, tokenizer, inputs, max_new_tokens=512, max_length=4096, batch_size=4, output_file=None, references=None, metrics_aggregator=None):
    """Generate outputs for a list of input strings.

    If output_file is provided, writes each prediction to JSONL immediately.
    If references is provided along with output_file, includes reference and metrics.
    Returns list of predictions.
    """
    all_outputs = []

    for i in tqdm(range(0, len(inputs), batch_size)):
        batch_inputs = inputs[i:i + batch_size]
        batch_refs = references[i:i + batch_size] if references else None

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
            padding_side="left",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only new tokens and write to file immediately
        for j, output in enumerate(outputs):
            input_len = encoded["attention_mask"][j].sum().item()
            new_tokens = output[input_len:]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
            prediction = decoded.strip()
            all_outputs.append(prediction)

            # Write to JSONL file immediately if provided
            if output_file:
                result = {
                    "index": i + j,
                    "input": batch_inputs[j],
                    "prediction": prediction,
                }
                if batch_refs:
                    result["reference"] = batch_refs[j]

                    # Compute metrics using MetricsAggregator
                    if metrics_aggregator:
                        metrics = metrics_aggregator.compute_all_metrics(batch_refs[j], prediction)
                        result["exact_match"] = metrics["exact_match"]
                        result["valid_json"] = metrics["valid_prediction_json"]
                        result["direct_match_score"] = metrics["direct_match_score"]
                        result["value_only_match_score"] = metrics["value_only_match_score"]
                    else:
                        # Fallback to basic metrics
                        result["exact_match"] = batch_refs[j].strip() == prediction.strip()
                        result["valid_json"] = False
                        try:
                            json.loads(prediction)
                            result["valid_json"] = True
                        except (json.JSONDecodeError, TypeError):
                            pass

                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_file.flush()  # Ensure it's written immediately

        print(f"  Generated {min(i + batch_size, len(inputs))}/{len(inputs)}")

    return all_outputs


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize accelerator for multi-GPU support
    accelerator = Accelerator()

    print("=" * 60)
    print("SFT Evaluation")
    print("=" * 60)
    print(f"Device: {accelerator.device}")
    print(f"Num processes: {accelerator.num_processes}")
    print(f"Is main process: {accelerator.is_main_process}")
    print("=" * 60)

    # Initialize metrics aggregator
    metrics_aggregator = MetricsAggregator()

    # Initialize TensorBoard writer if requested (only on main process)
    writer = None
    if args.log_to_tensorboard and accelerator.is_main_process:
        tensorboard_dir = os.path.join(args.output_dir, "tensorboard")
        writer = SummaryWriter(log_dir=tensorboard_dir)
        print(f"TensorBoard logging enabled: {tensorboard_dir}")

    # Load tokenizer
    if accelerator.is_main_process:
        print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model — auto-detect VL vs text-only LLM from config
    if accelerator.is_main_process:
        print("Loading model...")
    model_class, is_vl_model = get_model_class(args.model_name)

    if args.adapter_dir and args.quantization_bits:
        if accelerator.is_main_process:
            print(f"Loading base model to merge and quantize to {args.quantization_bits}-bit...")
        base_model = model_class.from_pretrained(
            args.model_name,
            device_map="cpu" if accelerator.num_processes == 1 else accelerator.device,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        model = PeftModel.from_pretrained(base_model, args.adapter_dir)
        if accelerator.is_main_process:
            print("Merging LoRA adapter into base weights...")
        model = model.merge_and_unload()

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            if accelerator.is_main_process:
                model.save_pretrained(tmpdir, safe_serialization=True)
            accelerator.wait_for_everyone()
            del model
            del base_model
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            if accelerator.is_main_process:
                print(f"Reloading merged model with {args.quantization_bits}-bit quantization...")
            from transformers import BitsAndBytesConfig
            if args.quantization_bits == 4:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                )
            model = model_class.from_pretrained(
                tmpdir,
                device_map=accelerator.device,
                quantization_config=bnb_config,
            )
    else:
        model = model_class.from_pretrained(
            args.model_name,
            device_map=accelerator.device,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

        # Load adapter if provided
        if args.adapter_dir:
            if accelerator.is_main_process:
                print(f"Loading adapter from: {args.adapter_dir}")
            model = PeftModel.from_pretrained(model, args.adapter_dir)

    model.eval()

    # Load data (same on all processes)
    if accelerator.is_main_process:
        print("Loading test data...")
    data = load_data(args.data_path, args.input_field, args.output_field, args.num_samples)
    if accelerator.is_main_process:
        print(f"Test samples: {len(data)}")

    inputs = [ex[args.input_field] for ex in data]
    references = [ex[args.output_field] for ex in data]

    # Split data across GPUs for parallel processing
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index

    # Split inputs and references for this process
    local_inputs = [inputs[i] for i in range(len(inputs)) if i % num_processes == process_index]
    local_references = [references[i] for i in range(len(references)) if i % num_processes == process_index]

    if accelerator.is_main_process:
        print(f"Processing {len(local_inputs)}/{len(inputs)} samples on this GPU")
        print("Generating outputs...")

    # Generate predictions for this process's subset
    local_predictions_path = os.path.join(args.output_dir, f"predictions_p{process_index}.jsonl")
    with open(local_predictions_path, "w", encoding="utf-8") as f:
        local_predictions = generate_outputs(
            model, tokenizer, local_inputs,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
            batch_size=args.batch_size,
            output_file=f,
            references=local_references,
            metrics_aggregator=metrics_aggregator,
        )

    # Wait for all processes to finish
    accelerator.wait_for_everyone()

    # Gather all predictions on main process
    if accelerator.is_main_process:
        print("\nCombining predictions from all GPUs...")

        # Read all local predictions and combine
        all_predictions_data = []
        for p in range(num_processes):
            local_path = os.path.join(args.output_dir, f"predictions_p{p}.jsonl")
            if os.path.exists(local_path):
                with open(local_path, "r", encoding="utf-8") as f:
                    for line in f:
                        all_predictions_data.append(json.loads(line))
                os.remove(local_path)  # Clean up temp file

        # Sort by index to restore original order
        all_predictions_data.sort(key=lambda x: x["index"])

        # Write combined predictions
        predictions_path = os.path.join(args.output_dir, "predictions.jsonl")
        with open(predictions_path, "w", encoding="utf-8") as f:
            for item in all_predictions_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Predictions saved to: {predictions_path}")

        # Extract predictions list for metrics
        predictions = [p["prediction"] for p in all_predictions_data]
        references = [p["reference"] for p in all_predictions_data]

        # Compute batch metrics
        print("Computing metrics...")
        batch_metrics = metrics_aggregator.compute_batch_metrics(references, predictions)

        # Print metrics
        print("\n" + "=" * 60)
        print("EVALUATION METRICS")
        print("=" * 60)
        print(f"Num Samples:              {batch_metrics['num_samples']}")
        print(f"Exact Match Accuracy:     {batch_metrics['exact_match_accuracy']:.4f}")
        print(f"Valid Reference JSON:     {batch_metrics['valid_reference_json_ratio']:.4f}")
        print(f"Valid Prediction JSON:    {batch_metrics['valid_prediction_json_ratio']:.4f}")
        print(f"Direct Match Score:       {batch_metrics['mean_direct_match_score']:.4f} ± {batch_metrics['std_direct_match_score']:.4f}")
        print(f"Value-Only Match Score:   {batch_metrics['mean_value_only_match_score']:.4f} ± {batch_metrics['std_value_only_match_score']:.4f}")
        print("=" * 60)

        # Log to TensorBoard
        if writer is not None:
            print("\nLogging to TensorBoard...")
            metrics_aggregator.log_to_tensorboard(
                writer,
                step=0,
                metrics=batch_metrics,
                prefix="eval"
            )
            writer.close()
            print(f"TensorBoard logs saved to: {tensorboard_dir}")
            print(f"View with: tensorboard --logdir {tensorboard_dir}")

        # Save summary
        summary = {
            "num_samples": batch_metrics['num_samples'],
            "exact_match_accuracy": batch_metrics['exact_match_accuracy'],
            "valid_reference_json_ratio": batch_metrics['valid_reference_json_ratio'],
            "valid_prediction_json_ratio": batch_metrics['valid_prediction_json_ratio'],
            "mean_direct_match_score": batch_metrics['mean_direct_match_score'],
            "std_direct_match_score": batch_metrics['std_direct_match_score'],
            "mean_value_only_match_score": batch_metrics['mean_value_only_match_score'],
            "std_value_only_match_score": batch_metrics['std_value_only_match_score'],
        }
        summary_path = os.path.join(args.output_dir, "eval_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Print sample
        if predictions:
            print("\n" + "=" * 60)
            print("SAMPLE OUTPUT")
            print("=" * 60)
            print(f"Input:      {inputs[0][:200]}")
            print(f"Reference:  {references[0][:200]}")
            print(f"Prediction: {predictions[0][:200]}")
            print("=" * 60)

        # Run heuristic pipeline
        if args.run_heuristic:
            print("\n" + "=" * 60)
            print("RUNNING HEURISTIC PIPELINE")
            print("=" * 60)

            import sys
            from pathlib import Path
            script_dir = args.heuristic_script_dir
            if not script_dir:
                script_dir = str(Path(__file__).resolve().parents[3] / "dataset" / "scripts")

            pipeline_script = os.path.join(script_dir, "run_heuristic_pipeline.py")
            if not os.path.exists(pipeline_script):
                print(f"ERROR: Heuristic script not found: {pipeline_script}")
            else:
                cmd = [
                    sys.executable,
                    pipeline_script,
                    "--input", predictions_path,
                    "--util-folder", args.heuristic_util_folder,
                    "--config", args.heuristic_config_path
                ]
                print(f"Executing: {' '.join(cmd)}")
                import subprocess
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)

                if result.returncode != 0:
                    print(f"Heuristic pipeline failed!\n{result.stdout}\n{result.stderr}")
                else:
                    print("Heuristic pipeline completed successfully.")
                    aggregates_path = os.path.join(args.output_dir, "aggregates.json")
                    if os.path.exists(aggregates_path):
                        print(f"Aggregates saved to {aggregates_path}")

        print(f"\nResults saved to: {args.output_dir}")
    else:
        # Non-main processes just finish their work
        print(f"Process {process_index} finished.")


if __name__ == "__main__":
    main()