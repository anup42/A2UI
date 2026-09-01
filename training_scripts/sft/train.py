"""
SFT Training Script with Multi-GPU Support via Accelerate.
With Memory Cleanup for Long Sequence Training (max_length=8192).

Input: agent response string
Output: JSON-like structured string

Usage:
    Single GPU:   accelerate launch train.py --model_name MODEL --data_path DATA --output_dir OUTPUT
    Multi-GPU:    accelerate launch train.py ... (configure with `accelerate config`)

Memory Optimization:
- Sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to prevent fragmentation
- Periodically calls torch.cuda.empty_cache() to free unused memory
"""

# Set memory allocation config BEFORE any torch imports
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Auto-detect CUDA_HOME for DeepSpeed compatibility
if not os.environ.get("CUDA_HOME"):
    try:
        _cuda_home = getattr(torch.utils.cpp_extension, "CUDA_HOME", None)
        if _cuda_home and os.path.isdir(_cuda_home):
            os.environ["CUDA_HOME"] = _cuda_home
    except Exception:
        pass
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, TrainerCallback
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

from dataloader import load_dataset, load_test_data
from metrics import MetricsAggregator


# Fixed LoRA target modules (attention only - common for Qwen, Gemma, Llama, etc.)
# For Gemma4, these match the inner linear layers (e.g., layers.0.attn.q_proj.linear)
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
# For Gemma4 with wrapped linear layers, use the .linear suffix
LORA_TARGET_MODULES_GEMMA4 = ["q_proj.linear", "k_proj.linear", "v_proj.linear", "o_proj.linear"]


def get_lora_layer_modules(model, num_layers: int, layers: str = "all"):
    """Generate target module names for LoRA based on layer selection.

    Args:
        model: The model to analyze (used to detect module path format)
        num_layers: Total number of transformer layers
        layers: Layer selection ('all', 'early', 'late', 'middle', or comma-separated indices)

    Returns:
        List of full module paths to target for LoRA, or "all-linear" for auto-detection
    """
    # For Gemma4 and similar models, use "all-linear" to auto-detect all linear layers
    # This avoids issues with incorrect module name matching
    model_type = getattr(model.config, "model_type", "")
    is_gemma4 = "gemma" in model_type.lower()

    if layers == "all" and is_gemma4:
        # Use "all-linear" for automatic detection of all linear layers
        # This is more robust than manually specifying module names
        print("  Using 'all-linear' target modules for automatic detection")
        return "all-linear"

    # For non-Gemma models or specific layer selection, use manual module names
    target_modules_base = LORA_TARGET_MODULES_GEMMA4 if is_gemma4 else LORA_TARGET_MODULES

    layer_indices = []
    if layers == "early":
        # First 25% of layers
        num_target = max(1, num_layers // 4)
        layer_indices = list(range(num_target))
    elif layers == "late":
        # Last 25% of layers
        num_target = max(1, num_layers // 4)
        layer_indices = list(range(num_layers - num_target, num_layers))
    elif layers == "middle":
        # Middle 50% of layers
        start = num_layers // 4
        end = start + (num_layers // 2)
        layer_indices = list(range(start, end))
    else:
        # Parse comma-separated indices
        try:
            layer_indices = [int(x.strip()) for x in layers.split(",")]
            # Validate indices
            layer_indices = [i for i in layer_indices if 0 <= i < num_layers]
            if not layer_indices:
                print(f"WARNING: No valid layer indices found, using all layers")
                return LORA_TARGET_MODULES
        except ValueError:
            print(f"WARNING: Invalid layer specification '{layers}', using all layers")
            return LORA_TARGET_MODULES

    # Detect module path format from model structure
    # Check if model uses 'model.layers.{idx}.self_attn.{module}' (Qwen-style)
    # or 'layers.{idx}.{module}' (Gemma-style)
    model_type = getattr(model.config, "model_type", "")

    # Gemma models use: layers.{idx}.{module}
    # Qwen/Llama models use: model.layers.{idx}.self_attn.{module}
    if "gemma" in model_type.lower():
        layer_prefix = "layers"
        attn_prefix = ""  # Gemma has modules directly under layer
    else:
        # Default to Qwen/Llama style
        layer_prefix = "model.layers"
        attn_prefix = "self_attn."

    # Build full module paths for selected layers
    target_modules = []
    for layer_idx in layer_indices:
        for module in LORA_TARGET_MODULES:
            if attn_prefix:
                target_modules.append(f"{layer_prefix}.{layer_idx}.{attn_prefix}{module}")
            else:
                target_modules.append(f"{layer_prefix}.{layer_idx}.{module}")

    print(f"  LoRA target layers: {layer_indices} out of {num_layers} total layers")
    print(f"  LoRA target modules per layer: {LORA_TARGET_MODULES}")
    print(f"  Total LoRA modules: {len(target_modules)}")

    return target_modules


def get_model_class(model_name: str):
    """Auto-detect the model class based on the model's config.json.

    Uses AutoModelForCausalLM for all models - works with Qwen, Gemma, Llama, etc.

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

    print(f"Loading model (model_type={model_type}), using AutoModelForCausalLM")
    return AutoModelForCausalLM, False


def parse_args():
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True, help="Path to JSON/JSONL file")
    parser.add_argument("--output_dir", type=str, required=True)

    # Data
    parser.add_argument("--input_field", type=str, default="response_text")
    parser.add_argument("--output_field", type=str, default="genui_json")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--train_samples", type=int, default=-1)
    parser.add_argument("--eval_samples", type=int, default=200)
    parser.add_argument("--eval_split_ratio", type=float, default=0.05)

    # Training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)

    # Logging
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)

    # LoRA
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_layers", type=str, default="all",
                        help="Which layers to apply LoRA to. Options: 'all', 'early', 'late', 'middle', "
                             "or comma-separated layer indices (e.g., '0,1,2' or '28,29,30,31' for last 4 layers). "
                             "'early': first 25% of layers, 'late': last 25% of layers, 'middle': middle 50% of layers. "
                             "LoRA is applied to q_proj, k_proj, v_proj, o_proj modules only.")

    # Monitoring
    parser.add_argument("--report_to", type=str, nargs="+", default=["tensorboard"])
    parser.add_argument("--wandb_project", type=str, default="sft-training")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)

    # Test data for prediction-based evaluation
    parser.add_argument("--test_data_path", type=str, default=None,
                        help="Path to test JSON/JSONL file for prediction-based evaluation during training. "
                             "If provided, predictions are generated on this data instead of the eval split, "
                             "and the best model is selected based on test prediction metrics.")
    parser.add_argument("--test_samples", type=int, default=-1,
                        help="Number of test samples to use for prediction (-1 for all)")
    parser.add_argument("--metric_for_best_model", type=str, default="mean_direct_match_score",
                        help="Metric to use for best model selection when test_data_path is provided. "
                             "Options: mean_direct_match_score, mean_value_only_match_score, "
                             "exact_match_accuracy, valid_prediction_json_ratio")

    # Prediction generation
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Max new tokens for eval predictions")

    # Heuristic evaluation (always enabled with defaults, same as train_qwen.py)
    parser.add_argument("--heuristic_interval", type=int, default=1,
                        help="Run heuristic evaluation every N prediction intervals (1 = every eval_steps)")
    parser.add_argument("--heuristic_util_folder", type=str,
                        default="/home/h_sagri/GenUI-LM/dataset/data/runs/golden100_v3",
                        help="Path to folder containing queries.jsonl, responses.jsonl, golden50_url_mapping.jsonl")
    parser.add_argument("--heuristic_config_path", type=str,
                        default="/home/h_sagri/GenUI-LM/dataset/configs/run.yaml",
                        help="Path to run.yaml config file for heuristic pipeline")
    parser.add_argument("--heuristic_url_unmask", action="store_true", default=True,
                        help="Whether to unmask URL placeholders in heuristic evaluation")
    parser.add_argument("--heuristic_script_dir", type=str, default="",
                        help="Directory containing run_heuristic_pipeline.py (auto-detected if not set)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    return args


class EvalPredictionCallback(TrainerCallback):
    """Callback to generate predictions and compute metrics during training with multi-GPU support.

    When test_data is provided, predictions are generated on the test data instead
    of the eval split, and the best model checkpoint is tracked based on the
    configured metric (default: mean_direct_match_score).

    When test_data is NOT provided, falls back to the original behavior of
    generating predictions on the eval split from training data.

    Multi-GPU Support:
    - Split test/eval data across GPUs using round-robin distribution
    - Each GPU processes its subset of samples in parallel
    - Gather all predictions using dist.all_gather()
    - Only the main process saves results and logs metrics

    This provides ~Nx speedup on N GPUs for evaluation.
    """

    def __init__(self, tokenizer, input_field, output_field, max_new_tokens=512, max_length=1024,
                 prediction_interval=1, max_prediction_samples=10, test_data=None,
                 metric_for_best_model="mean_direct_match_score", output_dir=None):
        self.tokenizer = tokenizer
        self.input_field = input_field
        self.output_field = output_field
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        self.prediction_interval = prediction_interval
        self.max_prediction_samples = max_prediction_samples
        self.metrics_aggregator = MetricsAggregator()

        # Test data for prediction-based evaluation
        self.test_data = test_data
        self.metric_for_best_model = metric_for_best_model
        self.output_dir = output_dir

        # Track best model based on test prediction metrics
        self.best_metric_value = -float('inf')
        self.best_metric_step = -1
        self.best_model_saved = False

        # Track best model based on validation loss
        self.best_val_loss = float('inf')
        self.best_val_loss_step = -1
        self.best_val_loss_model_saved = False

    def _generate_predictions_distributed(self, model, data_source, num_samples, is_distributed, world_size, rank):
        """Generate predictions on the given data source with multi-GPU support.

        Args:
            model: The model to use for generation
            data_source: Either a list of dicts (test_data) or a HuggingFace Dataset (eval_dataset)
            num_samples: Max number of samples to process
            is_distributed: Whether running in distributed mode
            world_size: Number of GPUs/processes
            rank: Current process rank

        Returns:
            List of dicts with input, reference, prediction, and global_idx keys
        """
        local_predictions_data = []

        # Split data across GPUs using round-robin distribution
        # Each GPU processes samples at indices: rank, rank+world_size, rank+2*world_size, ...
        local_indices = list(range(rank, num_samples, world_size))
        local_num_samples = len(local_indices)

        if rank == 0:
            print(f"\nGenerating predictions with {world_size} GPU(s)...")
            print(f"  Total samples: {num_samples}, Local samples (rank {rank}): {local_num_samples}")
        else:
            print(f"  Rank {rank}: Processing {local_num_samples} samples...")

        # Each GPU processes its subset of samples
        for local_idx, global_idx in enumerate(local_indices):
            # Get example - either from list or dataset
            if isinstance(data_source, list):
                example = data_source[global_idx]
            else:
                example = data_source[global_idx]

            input_text = example.get(self.input_field, "")
            reference = example.get(self.output_field, "")

            if not input_text:
                continue

            # Format input for generation (just the user message)
            messages = [{"role": "user", "content": input_text}]
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt = f"<|im_start|>user\n{input_text}\n<|im_start|>assistant\n"

            # Tokenize
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(model.device)

            # Generate
            with torch.no_grad():
                output = model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode only new tokens
            input_len = encoded["attention_mask"].sum().item()
            new_tokens = output[0][input_len:]
            prediction = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            local_predictions_data.append({
                "input": input_text,
                "reference": reference,
                "prediction": prediction,
                "global_idx": global_idx,  # Track original index for reordering
            })

            if (local_idx + 1) % 5 == 0 or (local_idx + 1) == local_num_samples:
                print(f"  Rank {rank}: Processed {local_idx + 1}/{local_num_samples} samples...")

        return local_predictions_data

    def _gather_predictions(self, local_predictions_data, model, is_distributed, world_size):
        """Gather predictions from all GPUs and reorder by original index.

        Args:
            local_predictions_data: List of predictions from this GPU
            model: The model (for device access)
            is_distributed: Whether running in distributed mode
            world_size: Number of GPUs/processes

        Returns:
            Combined and reordered list of all predictions
        """
        if not is_distributed:
            return local_predictions_data

        # Prepare data for gathering - convert to JSON strings for easy gathering
        local_json_strings = [json.dumps(p) for p in local_predictions_data]

        # Gather all JSON strings from all processes
        # First, we need to know how many items each process has
        local_count = torch.tensor([len(local_json_strings)], device=model.device, dtype=torch.long)
        counts_list = [torch.zeros_like(local_count) for _ in range(world_size)]
        dist.all_gather(counts_list, local_count)

        # Pad to max length for all_gather
        max_count = max(c.item() for c in counts_list)
        padded_local = local_json_strings + [""] * (max_count - len(local_json_strings))

        # Gather all padded lists
        # all_gather_object gathers the entire list as one object per process
        # So all_gathered[proc_idx] is the list from process proc_idx
        all_gathered = [None] * world_size
        dist.all_gather_object(all_gathered, padded_local)

        # Reconstruct the full list, filtering out padding
        all_predictions_data = []
        for proc_idx in range(world_size):
            proc_list = all_gathered[proc_idx]  # Get the list from this process
            for item in proc_list:
                if isinstance(item, str) and item:  # It's a non-empty JSON string
                    all_predictions_data.append(json.loads(item))
                elif isinstance(item, dict):  # Already a dict (fallback)
                    all_predictions_data.append(item)

        # Sort by global_idx to maintain consistent ordering
        all_predictions_data.sort(key=lambda x: x["global_idx"])

        # Remove global_idx from final output
        for p in all_predictions_data:
            del p["global_idx"]

        return all_predictions_data

    def _generate_and_save_predictions(self, args, state, control, model, output_dir, is_distributed, world_size, rank, data_source, num_samples, predictions_file, metrics_file, log_prefix, checkpoint_dir=None):
        """Helper method to generate predictions, compute metrics, and save them.

        Args:
            checkpoint_dir: Optional directory to save predictions (defaults to output_dir)
        """
        save_dir = checkpoint_dir if checkpoint_dir else output_dir

        # Generate predictions (distributed across GPUs)
        local_predictions_data = self._generate_predictions_distributed(
            model, data_source, num_samples, is_distributed, world_size, rank
        )

        # Gather predictions from all GPUs
        all_predictions_data = self._gather_predictions(
            local_predictions_data, model, is_distributed, world_size
        )

        # Only save and log metrics from main process
        if rank == 0:
            # Compute metrics
            references = [p["reference"] for p in all_predictions_data]
            predictions = [p["prediction"] for p in all_predictions_data]
            batch_metrics = self.metrics_aggregator.compute_batch_metrics(references, predictions)

            # Print metrics
            print(f"\n{log_prefix.capitalize()} Prediction Metrics (step {state.global_step}):")
            print(f"  Exact Match:        {batch_metrics['exact_match_accuracy']:.4f}")
            print(f"  Valid JSON:         {batch_metrics['valid_prediction_json_ratio']:.4f}")
            print(f"  Direct Match:       {batch_metrics['mean_direct_match_score']:.4f}")
            print(f"  Value-Only Match:   {batch_metrics['mean_value_only_match_score']:.4f}")

            # Save predictions
            predictions_path = os.path.join(save_dir, predictions_file)
            with open(predictions_path, "w", encoding="utf-8") as f:
                json.dump(all_predictions_data, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(all_predictions_data)} predictions to: {predictions_path}")

            # Save metrics summary
            metrics_summary = {
                "step": state.global_step,
                "exact_match_accuracy": batch_metrics['exact_match_accuracy'],
                "valid_prediction_json_ratio": batch_metrics['valid_prediction_json_ratio'],
                "mean_direct_match_score": batch_metrics['mean_direct_match_score'],
                "std_direct_match_score": batch_metrics['std_direct_match_score'],
                "mean_value_only_match_score": batch_metrics['mean_value_only_match_score'],
                "std_value_only_match_score": batch_metrics['std_value_only_match_score'],
            }
            metrics_path = os.path.join(save_dir, metrics_file)
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics_summary, f, indent=2)

            # Log to TensorBoard
            writer = SummaryWriter(log_dir=output_dir)
            writer.add_scalar(f"{log_prefix}/exact_match_accuracy", batch_metrics['exact_match_accuracy'], state.global_step)
            writer.add_scalar(f"{log_prefix}/valid_json_ratio", batch_metrics['valid_prediction_json_ratio'], state.global_step)
            writer.add_scalar(f"{log_prefix}/direct_match_score", batch_metrics['mean_direct_match_score'], state.global_step)
            writer.add_scalar(f"{log_prefix}/value_only_match_score", batch_metrics['mean_value_only_match_score'], state.global_step)
            writer.add_scalar(f"{log_prefix}/direct_match_std", batch_metrics['std_direct_match_score'], state.global_step)
            writer.add_scalar(f"{log_prefix}/value_only_match_std", batch_metrics['std_value_only_match_score'], state.global_step)
            writer.close()
            print(f"{log_prefix.capitalize()} metrics logged to TensorBoard at step {state.global_step}")

            return batch_metrics, metrics_summary
        return None, None

    def on_save(self, args, state, control, model, **kwargs):
        """Generate and save test predictions inside the checkpoint directory on every save."""
        # Only generate predictions if test_data is available
        if not self.test_data:
            return

        # Get the checkpoint path being saved
        checkpoint_path = kwargs.get('checkpoint_path')
        if not checkpoint_path:
            return

        # Determine output directory
        output_dir = args.output_dir

        # Check if we're in distributed environment
        is_distributed = dist.is_initialized()
        world_size = dist.get_world_size() if is_distributed else 1
        rank = dist.get_rank() if is_distributed else 0

        # Only generate predictions from main process (but all processes must sync)
        if rank == 0:
            print(f"\nGenerating test predictions for checkpoint: {checkpoint_path}")

        # Use ALL test samples for checkpoint predictions
        num_samples = len(self.test_data)

        # Generate and save predictions inside the checkpoint directory
        model.eval()
        self._generate_and_save_predictions(
            args=args,
            state=state,
            control=control,
            model=model,
            output_dir=output_dir,
            is_distributed=is_distributed,
            world_size=world_size,
            rank=rank,
            data_source=self.test_data,
            num_samples=num_samples,
            predictions_file="test_predictions.json",
            metrics_file="test_metrics.json",
            log_prefix="test",
            checkpoint_dir=checkpoint_path,  # Save inside checkpoint directory
        )
        model.train()

        # Synchronize all processes before continuing
        if is_distributed:
            dist.barrier()

    def on_evaluate(self, args, state, control, model, eval_dataloader, metrics=None, **kwargs):
        """Generate and save predictions after each evaluation with multi-GPU support.

        Also tracks and saves the best model based on validation loss.
        """
        # Determine output directory
        output_dir = args.output_dir

        # Check if we're in distributed environment
        is_distributed = dist.is_initialized()
        world_size = dist.get_world_size() if is_distributed else 1
        rank = dist.get_rank() if is_distributed else 0

        # ---- Track best model based on validation loss (on EVERY evaluation) ----
        # metrics is a dict containing 'eval_loss' and other eval metrics
        if metrics is not None and 'eval_loss' in metrics:
            current_val_loss = metrics['eval_loss']

            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                self.best_val_loss_step = state.global_step
                print(f"\n  *** New best validation loss: {current_val_loss:.4f} at step {state.global_step} ***")

                # Save best validation loss model checkpoint
                best_val_checkpoint_dir = os.path.join(output_dir, "best_val_loss_checkpoint")
                os.makedirs(best_val_checkpoint_dir, exist_ok=True)

                # Save the model
                unwrapped_model = model
                try:
                    unwrapped_model = model.module  # Unwrap DDP/FSDP
                except AttributeError:
                    pass

                # Check if it's a PeftModel
                from peft import PeftModel
                if isinstance(unwrapped_model, PeftModel):
                    unwrapped_model.save_pretrained(best_val_checkpoint_dir)
                else:
                    unwrapped_model.save_pretrained(best_val_checkpoint_dir)

                self.tokenizer.save_pretrained(best_val_checkpoint_dir)

                # Save best val loss info
                best_val_info = {
                    "best_val_loss": self.best_val_loss,
                    "best_val_loss_step": self.best_val_loss_step,
                    "all_metrics": metrics,
                }
                with open(os.path.join(best_val_checkpoint_dir, "best_val_loss_info.json"), "w") as f:
                    json.dump(best_val_info, f, indent=2)

                print(f"  Best validation loss model checkpoint saved to: {best_val_checkpoint_dir}")
                self.best_val_loss_model_saved = True
            else:
                print(f"\n  Current val_loss: {current_val_loss:.4f} (best: {self.best_val_loss:.4f} at step {self.best_val_loss_step})")

        # Only run prediction generation at specified intervals
        current_eval = state.global_step // args.eval_steps
        if current_eval % self.prediction_interval != 0:
            # Synchronize all processes before continuing
            if is_distributed:
                dist.barrier()
            return

        model.eval()

        # Determine data source and number of samples
        if self.test_data:
            data_source = self.test_data
            # Use ALL test samples (not limited)
            num_samples = len(self.test_data)
            prefix = "test"
            predictions_file = "test_predictions.json"
            metrics_file = "test_metrics.json"
            log_prefix = "test"
        else:
            data_source = eval_dataloader.dataset
            max_samples = self.max_prediction_samples
            num_samples = min(len(data_source), max_samples)
            prefix = "eval"
            predictions_file = "eval_predictions.json"
            metrics_file = "eval_metrics.json"
            log_prefix = "eval"

        # Generate and save predictions (to output_dir root for eval predictions)
        _, metrics_summary = self._generate_and_save_predictions(
            args=args,
            state=state,
            control=control,
            model=model,
            output_dir=output_dir,
            is_distributed=is_distributed,
            world_size=world_size,
            rank=rank,
            data_source=data_source,
            num_samples=num_samples,
            predictions_file=predictions_file,
            metrics_file=metrics_file,
            log_prefix=log_prefix,
            checkpoint_dir=None,  # Save to output_dir root
        )

        # ---- Track best model based on test prediction metric (only for test data mode) ----
        if self.test_data and metrics_summary:
            current_metric_value = metrics_summary.get(self.metric_for_best_model, 0.0)
            is_best = current_metric_value > self.best_metric_value

            if is_best:
                self.best_metric_value = current_metric_value
                self.best_metric_step = state.global_step
                print(f"  *** New best {self.metric_for_best_model}: {current_metric_value:.4f} at step {state.global_step} ***")

                # Save best model checkpoint
                best_checkpoint_dir = os.path.join(output_dir, "best_test_checkpoint")
                os.makedirs(best_checkpoint_dir, exist_ok=True)

                # Save the model
                unwrapped_model = model
                try:
                    unwrapped_model = model.module  # Unwrap DDP/FSDP
                except AttributeError:
                    pass

                # Check if it's a PeftModel
                from peft import PeftModel
                if isinstance(unwrapped_model, PeftModel):
                    unwrapped_model.save_pretrained(best_checkpoint_dir)
                else:
                    unwrapped_model.save_pretrained(best_checkpoint_dir)

                self.tokenizer.save_pretrained(best_checkpoint_dir)

                # Copy test predictions to best_test_checkpoint
                test_predictions_path = os.path.join(output_dir, "test_predictions.json")
                if os.path.exists(test_predictions_path):
                    import shutil
                    shutil.copy(test_predictions_path, os.path.join(best_checkpoint_dir, "test_predictions.json"))
                    shutil.copy(os.path.join(output_dir, "test_metrics.json"), os.path.join(best_checkpoint_dir, "test_metrics.json"))

                # Save best metric info
                best_info = {
                    "best_metric": self.metric_for_best_model,
                    "best_metric_value": self.best_metric_value,
                    "best_step": self.best_metric_step,
                    "all_metrics": metrics_summary,
                }
                with open(os.path.join(best_checkpoint_dir, "best_metric_info.json"), "w") as f:
                    json.dump(best_info, f, indent=2)

                print(f"  Best model checkpoint saved to: {best_checkpoint_dir}")
                self.best_model_saved = True
            else:
                print(f"  Current {self.metric_for_best_model}: {current_metric_value:.4f} "
                      f"(best: {self.best_metric_value:.4f} at step {self.best_metric_step})")

        # Synchronize all processes before continuing (critical for distributed training)
        if is_distributed:
            dist.barrier()


class MemoryCleanupCallback(TrainerCallback):
    """Callback to periodically free unused CUDA memory during training.

    This helps prevent OOM errors when training with long sequences (max_length=8192)
    by calling torch.cuda.empty_cache() at regular intervals.
    """

    def __init__(self, cleanup_interval: int = 25):
        """
        Args:
            cleanup_interval: Call empty_cache() every N training steps
        """
        self.cleanup_interval = cleanup_interval

    def on_step_end(self, args, state, control, **kwargs):
        """Free unused CUDA memory at regular intervals."""
        if state.global_step > 0 and state.global_step % self.cleanup_interval == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"[Memory Cleanup] Step {state.global_step}: Freed unused CUDA memory")


class HeuristicEvaluationCallback(TrainerCallback):
    """Callback to run heuristic evaluation pipeline during training.

    After each evaluation, this callback:
    1. Reads the predictions generated by EvalPredictionCallback
    2. Converts them to JSONL format expected by the heuristic pipeline
    3. Copies utility files (queries.jsonl, responses.jsonl, golden50_url_mapping.jsonl)
    4. Runs the heuristic pipeline (steps 0-5) on the predictions
    5. Saves aggregates as aggregates_step_<N>.json
    6. Saves best_heuristic_checkpoint when overall_score improves

    Multi-GPU Support:
    - Only runs on the main process (rank 0)
    - Synchronizes with other processes after completion
    """

    def __init__(self, tokenizer, util_folder: str, config_path: str = None,
                 heuristic_interval: int = 1, url_unmask: bool = False,
                 output_dir: str = None, script_dir: str = None):
        """
        Args:
            tokenizer: Tokenizer to save with best heuristic checkpoint
            util_folder: Path to folder containing queries.jsonl, responses.jsonl, golden50_url_mapping.jsonl
            config_path: Path to run.yaml config file (optional)
            heuristic_interval: Run heuristic eval every N prediction intervals
            url_unmask: Whether to unmask URL placeholders
            output_dir: Output directory for aggregates
            script_dir: Directory containing the heuristic pipeline scripts
        """
        self.tokenizer = tokenizer
        self.util_folder = Path(util_folder) if util_folder else None
        self.config_path = Path(config_path) if config_path else None
        self.heuristic_interval = heuristic_interval
        self.url_unmask = url_unmask
        self.output_dir = Path(output_dir) if output_dir else None
        self.script_dir = Path(script_dir) if script_dir else None

        # Track heuristic evaluation count
        self.heuristic_eval_count = 0
        self.best_heuristic_score = -float('inf')
        self.best_heuristic_step = -1
        self.best_heuristic_model_saved = False

        # Model will be set during training
        self.model = None

        # Validate paths
        if self.util_folder and not self.util_folder.exists():
            print(f"WARNING: Util folder not found: {self.util_folder}")
            self.util_folder = None

    def _convert_predictions_to_jsonl(self, predictions: list, output_path: Path):
        """Convert predictions from JSON format to JSONL format expected by the pipeline.

        The pipeline expects predictions.jsonl with 'prediction' field.
        Our predictions are in format: {"input": ..., "reference": ..., "prediction": ...}

        We need to create entries with 'prediction' field that the pipeline can process.
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for pred in predictions:
                # Create entry with just the prediction field (pipeline expects this)
                entry = {"prediction": pred.get("prediction", "")}
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def _copy_util_files(self, dest_dir: Path):
        """Copy utility files to the destination directory."""
        if not self.util_folder:
            print("WARNING: No util folder specified, skipping util file copy")
            return False

        files_to_copy = ["queries.jsonl", "responses.jsonl", "golden50_url_mapping.jsonl"]
        copied = 0

        for filename in files_to_copy:
            source = self.util_folder / filename
            dest = dest_dir / filename

            if source.exists():
                shutil.copy2(source, dest)
                print(f"  Copied: {filename}")
                copied += 1
            else:
                print(f"  WARNING: Source file not found: {source}")

        return copied == len(files_to_copy)

    def _run_heuristic_pipeline(self, run_dir: Path, step: int, output_dir: Path) -> dict:
        """Run the complete heuristic pipeline (steps 0-5) in a temporary directory.

        Only saves aggregate_step_<step>.json to output_dir, cleaning up intermediate files.

        Returns:
            Dictionary with aggregate metrics or None if failed
        """
        if not self.script_dir:
            print("ERROR: Script directory not specified")
            return None

        pipeline_script = self.script_dir / "run_heuristic_pipeline.py"
        if not pipeline_script.exists():
            print(f"ERROR: Pipeline script not found: {pipeline_script}")
            return None

        # Build command
        predictions_jsonl = run_dir / "predictions.jsonl"

        cmd = [
            sys.executable,
            str(pipeline_script),
            "--input", str(predictions_jsonl),
            "--util-folder", str(self.util_folder),
        ]

        if self.config_path and self.config_path.exists():
            cmd.extend(["--config", str(self.config_path)])

        # Note: --url_unmask is not a flag for run_heuristic_pipeline.py
        # URL unmasking is handled internally by step4_fix_genui_predictions_v2.py
        # when the golden50_url_mapping.jsonl file is present in the util folder

        print(f"\n{'='*60}")
        print(f"Running Heuristic Pipeline at step {step}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}\n")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.script_dir))

            if result.returncode != 0:
                print(f"ERROR: Heuristic pipeline failed!")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return None

            print(f"SUCCESS: Heuristic pipeline completed!")

            # Read aggregates from temporary directory
            aggregates_path = run_dir / "aggregates.json"
            if aggregates_path.exists():
                with open(aggregates_path, 'r', encoding='utf-8') as f:
                    aggregates = json.load(f)

                # Copy only aggregates to output directory (not the whole folder)
                output_aggregates_path = output_dir / f"aggregate_step_{step}.json"
                shutil.copy2(str(aggregates_path), str(output_aggregates_path))
                print(f"  Aggregates saved to: {output_aggregates_path}")

                # Clean up temporary working directory
                try:
                    shutil.rmtree(run_dir)
                    print(f"  Cleaned up temporary directory: {run_dir}")
                except Exception as cleanup_err:
                    print(f"  WARNING: Could not clean up temp directory: {cleanup_err}")

                return aggregates
            else:
                print(f"WARNING: Aggregates file not found at {aggregates_path}")
                return None

        except Exception as e:
            print(f"ERROR: Failed to run heuristic pipeline: {e}")
            return None

    def on_evaluate(self, args, state, control, model, eval_dataloader, **kwargs):
        """Run heuristic evaluation after predictions are generated.

        IMPORTANT: This runs ONLY on rank 0 to avoid NCCL timeout issues.
        The heuristic pipeline can take a long time, so other ranks should
        NOT wait at a barrier. Training continues normally on other ranks.
        """
        # Check if heuristic evaluation is enabled
        if not self.util_folder:
            return

        # Check if we're in distributed environment
        is_distributed = dist.is_initialized()
        rank = dist.get_rank() if is_distributed else 0

        # Only run on main process - other ranks continue without waiting
        if rank != 0:
            return

        # Check interval
        current_eval = state.global_step // args.eval_steps
        if current_eval % self.heuristic_interval != 0:
            return

        # Increment evaluation count
        self.heuristic_eval_count += 1

        # Determine predictions file
        predictions_file = self.output_dir / "test_predictions.json"
        if not predictions_file.exists():
            predictions_file = self.output_dir / "eval_predictions.json"

        if not predictions_file.exists():
            print(f"WARNING: Predictions file not found: {predictions_file}")
            return

        print(f"\n{'#'*60}")
        print(f"# Heuristic Evaluation #{self.heuristic_eval_count} at step {state.global_step}")
        print(f"# NOTE: Running on rank 0 only, other ranks continue training")
        print(f"{'#'*60}")

        # Load predictions
        with open(predictions_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)

        print(f"Loaded {len(predictions)} predictions")

        # Create working directory for this evaluation
        work_dir = self.output_dir / f"heuristic_eval_step_{state.global_step}"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Convert predictions to JSONL format
        predictions_jsonl = work_dir / "predictions.jsonl"
        print(f"Converting predictions to JSONL format...")
        self._convert_predictions_to_jsonl(predictions, predictions_jsonl)

        # Copy util files
        print(f"Copying utility files...")
        if not self._copy_util_files(work_dir):
            print("WARNING: Some util files could not be copied")

        # Run heuristic pipeline (uses temp dir, only saves aggregate_step_<x>.json to output_dir)
        print(f"Running heuristic pipeline...")
        print(f"NOTE: This may take a while. Training continues on other GPUs.")
        aggregates = self._run_heuristic_pipeline(work_dir, state.global_step, self.output_dir)

        if aggregates:
            # Log to TensorBoard
            writer = SummaryWriter(log_dir=str(self.output_dir))

            # Log key metrics
            overall_score = aggregates.get('overall_score', 0)
            media_score = aggregates.get('media_score', 0)
            schema_valid_rate = aggregates.get('schema_valid_strict_rate', 0)
            content_coverage_avg = aggregates.get('content_coverage_avg', 0)

            writer.add_scalar("heuristic/overall_score", overall_score, state.global_step)
            writer.add_scalar("heuristic/media_score", media_score, state.global_step)
            writer.add_scalar("heuristic/schema_valid_strict_rate", schema_valid_rate, state.global_step)
            writer.add_scalar("heuristic/content_coverage_avg", content_coverage_avg, state.global_step)
            writer.close()

            print(f"\nHeuristic Metrics (step {state.global_step}):")
            print(f"  Overall Score:        {overall_score:.4f}")
            print(f"  Media Score:          {media_score:.4f}")
            print(f"  Schema Valid Rate:    {schema_valid_rate:.4f}")
            print(f"  Content Coverage Avg: {content_coverage_avg:.4f}")

            # Track best heuristic score and save checkpoint
            if overall_score > self.best_heuristic_score:
                self.best_heuristic_score = overall_score
                self.best_heuristic_step = state.global_step
                print(f"  *** New best overall_score: {overall_score:.4f} ***")

                # Copy best aggregates to a separate file (already saved to output_dir by _run_heuristic_pipeline)
                best_aggregates_path = self.output_dir / "best_heuristic_aggregates.json"
                source_aggregates_path = self.output_dir / f"aggregate_step_{state.global_step}.json"
                if source_aggregates_path.exists():
                    shutil.copy2(str(source_aggregates_path), str(best_aggregates_path))

                # Save best model checkpoint based on heuristic score
                best_checkpoint_dir = os.path.join(self.output_dir, "best_heuristic_checkpoint")
                os.makedirs(best_checkpoint_dir, exist_ok=True)

                # Save the model
                unwrapped_model = model
                try:
                    unwrapped_model = model.module  # Unwrap DDP/FSDP
                except AttributeError:
                    pass

                # Check if it's a PeftModel
                from peft import PeftModel
                if isinstance(unwrapped_model, PeftModel):
                    unwrapped_model.save_pretrained(best_checkpoint_dir)
                else:
                    unwrapped_model.save_pretrained(best_checkpoint_dir)

                # Save tokenizer
                self.tokenizer.save_pretrained(best_checkpoint_dir)

                # Save best score info
                best_info = {
                    "best_overall_score": self.best_heuristic_score,
                    "best_step": self.best_heuristic_step,
                    "aggregates_file": f"aggregates_step_{state.global_step}.json",
                    "checkpoint_dir": "best_heuristic_checkpoint",
                }
                with open(self.output_dir / "best_heuristic_info.json", 'w') as f:
                    json.dump(best_info, f, indent=2)

                print(f"  Best heuristic checkpoint saved to: {best_checkpoint_dir}")
                self.best_heuristic_model_saved = True
        else:
            print(f"WARNING: Heuristic evaluation failed at step {state.global_step}")

        # NO barrier here - other ranks should NOT wait for heuristic evaluation
        # Training continues normally on other ranks
        print(f"\nHeuristic evaluation complete. Training continues on all ranks.")


def main():
    args = parse_args()

    print("=" * 60)
    print("SFT Training")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print(f"GPUs available: {torch.cuda.device_count()}")
    print("=" * 60)

    set_seed(args.seed)

    # Save config
    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # Load tokenizer
    print("Loading tokenizer...")

    # Detect model type for tokenizer loading
    config_path = os.path.join(args.model_name, "config.json")
    model_type = ""
    if "gemma" in args.model_name:
        # with open(config_path, "r") as f:
        #     config = json.load(f)
        # model_type = config.get("model_type", "")
        model_type = "gemma4"

    # For Gemma4, use PreTrainedTokenizerFast directly to avoid sentencepiece issues
    print(">>>>>>>>>>>>>>>>>>>>>>>", model_type)
    if "gemma" in model_type.lower():
        print(f"Detected Gemma4 model (model_type={model_type}), using PreTrainedTokenizerFast")
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_name)
    else:
        print(f"Loading tokenizer for model_type={model_type}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load data
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

    # Load test data if provided
    test_data = None
    if args.test_data_path:
        print("Loading test data for prediction-based evaluation...")
        test_data = load_test_data(
            test_path=args.test_data_path,
            input_field=args.input_field,
            output_field=args.output_field,
            num_samples=args.test_samples,
        )
        if test_data:
            print(f"Test data loaded: {len(test_data)} samples")
            print(f"Best model will be selected by: {args.metric_for_best_model}")
        else:
            print("WARNING: No test data loaded, falling back to eval split for predictions.")
            test_data = None

    # Load model — auto-detect VL vs text-only LLM from config
    print("Loading model...")
    model_class, is_vl_model = get_model_class(args.model_name)
    model = model_class.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.config.use_cache = False

    # Apply LoRA if requested
    if args.use_lora:
        print("Applying LoRA...")

        # Get number of layers for layer selection
        # For Gemma4 and similar models, num_hidden_layers is in text_config
        num_layers = getattr(model.config, "num_hidden_layers", 0)
        if num_layers == 0 and hasattr(model.config, "text_config"):
            num_layers = getattr(model.config.text_config, "num_hidden_layers", 0)
        print(f"  Model has {num_layers} transformer layers")

        # Get target modules based on layer selection
        target_modules = get_lora_layer_modules(model, num_layers, args.lora_layers)

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Training arguments
    training_args = SFTConfig(
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
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps if eval_dataset else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to if isinstance(args.report_to, list) else [args.report_to],
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_grad_norm=1.0,
    )

    # Create prediction callback
    eval_prediction_callback = EvalPredictionCallback(
        tokenizer=tokenizer,
        input_field=args.input_field,
        output_field=args.output_field,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        test_data=test_data,
        metric_for_best_model=args.metric_for_best_model,
        output_dir=args.output_dir,
    )

    # Default paths for heuristic evaluation (same as train_qwen.py)
    DEFAULT_UTIL_FOLDER = "/home/h_sagri/GenUI-LM/dataset/data/runs/golden100_v3"
    DEFAULT_CONFIG_PATH = "/home/h_sagri/GenUI-LM/dataset/configs/run.yaml"

    # Determine script directory (same directory as train.py)
    genui_dataset_scripts = Path(__file__).parent.parent.parent / "dataset" / "scripts"

    if not genui_dataset_scripts.exists():
        print(f"WARNING: GenUI dataset scripts directory not found at {genui_dataset_scripts}")
        genui_dataset_scripts = None

    # Create heuristic evaluation callback (always enabled with defaults)
    heuristic_callback = HeuristicEvaluationCallback(
        tokenizer=tokenizer,
        util_folder=args.heuristic_util_folder or DEFAULT_UTIL_FOLDER,
        config_path=args.heuristic_config_path or DEFAULT_CONFIG_PATH,
        heuristic_interval=args.heuristic_interval,
        url_unmask=True,  # Always enable URL unmasking
        output_dir=args.output_dir,
        script_dir=args.heuristic_script_dir or genui_dataset_scripts,
    )

    # Create memory cleanup callback (every 25 steps)
    memory_cleanup_callback = MemoryCleanupCallback(cleanup_interval=25)

    # Build callbacks list — heuristic always included, memory cleanup added
    callbacks = [eval_prediction_callback, heuristic_callback, memory_cleanup_callback]

    print(f"Heuristic evaluation enabled (by default):")
    print(f"  Util folder: {args.heuristic_util_folder or DEFAULT_UTIL_FOLDER}")
    print(f"  Config: {args.heuristic_config_path or DEFAULT_CONFIG_PATH}")
    print(f"  Interval: every {args.heuristic_interval} evaluation(s)")
    print(f"  URL unmask: True")

    # Trainer — accelerate handles multi-GPU automatically
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # Train
    print("Starting training...")
    train_result = trainer.train()

    # Save
    print("Saving model...")
    if args.use_lora:
        adapter_dir = os.path.join(args.output_dir, "adapter")
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        print(f"Adapter saved to: {adapter_dir}")
    else:
        model_dir = os.path.join(args.output_dir, "model")
        trainer.save_model(model_dir)
        tokenizer.save_pretrained(model_dir)
        print(f"Model saved to: {model_dir}")

    trainer.save_state()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Print best test model info
    if eval_prediction_callback.best_model_saved:
        print("=" * 60)
        print("Best Test Model Checkpoint")
        print("=" * 60)
        print(f"  Metric:            {eval_prediction_callback.metric_for_best_model}")
        print(f"  Best value:        {eval_prediction_callback.best_metric_value:.4f}")
        print(f"  Best step:         {eval_prediction_callback.best_metric_step}")
        print(f"  Checkpoint:        {os.path.join(args.output_dir, 'best_test_checkpoint')}")
        print("=" * 60)

    # Print best validation loss model info
    if eval_prediction_callback.best_val_loss_model_saved:
        print("=" * 60)
        print("Best Validation Loss Model Checkpoint")
        print("=" * 60)
        print(f"  Best val_loss:     {eval_prediction_callback.best_val_loss:.4f}")
        print(f"  Best step:         {eval_prediction_callback.best_val_loss_step}")
        print(f"  Checkpoint:        {os.path.join(args.output_dir, 'best_val_loss_checkpoint')}")
        print("=" * 60)

    # Print best heuristic model info
    if heuristic_callback.best_heuristic_model_saved:
        print("=" * 60)
        print("Best Heuristic Model Checkpoint")
        print("=" * 60)
        print(f"  Best overall_score: {heuristic_callback.best_heuristic_score:.4f}")
        print(f"  Best step:          {heuristic_callback.best_heuristic_step}")
        print(f"  Checkpoint:         {os.path.join(args.output_dir, 'best_heuristic_checkpoint')}")
        print(f"  Aggregates:         {os.path.join(args.output_dir, 'best_heuristic_aggregates.json')}")
        print("=" * 60)

    print("=" * 60)
    print("Training completed!")
    print(f"Output: {args.output_dir}")
    if eval_prediction_callback.best_model_saved:
        print(f"Best test model: {os.path.join(args.output_dir, 'best_test_checkpoint')}")
    if eval_prediction_callback.best_val_loss_model_saved:
        print(f"Best val_loss model: {os.path.join(args.output_dir, 'best_val_loss_checkpoint')}")
    if heuristic_callback.best_heuristic_model_saved:
        print(f"Best heuristic model: {os.path.join(args.output_dir, 'best_heuristic_checkpoint')}")
    print("=" * 60)


if __name__ == "__main__":
    main()