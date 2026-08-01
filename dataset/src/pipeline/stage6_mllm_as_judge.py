# -*- coding: utf-8 -*-
import os
import json
import re
import argparse
import statistics
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Default model path (can be overridden with --model argument)
MODEL_PATH = r"/home/adarsh_ag/models/Gemma-4-26B-A4B-it"

import torch
print(f"CUDA devices visible to PyTorch: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")



# ------------------------------------------------------------
# Generic multimodal prompt builder
# ------------------------------------------------------------
def build_ui_judge_messages(image_path: str, user_question: str):
    judge_prompt = f"""
You are a visual critic. Evaluate this UI screenshot.
Question: {user_question}

Score each dimension on a scale of 1 to 5:
- very bad: 1
- bad: 2
- average: 3
- above average/good: 4
- very good: 5

ONLY use whole numbers for each dimension.
Do NOT use decimals.

Dimensions to evaluate:
visual_appeal, visual_hierarchy, readability, color_harmony.

Return ONLY JSON and nothing else:
{{
  "verdict": "",
  "strengths": [],
  "weaknesses": [],
  "reason": ""
  "overall_score": 0,
  "dimension_scores": {{
    "visual_appeal": 0,
    "visual_hierarchy": 0,
    "readability": 0,
    "color_harmony": 0
  }}

}}

Do not output anything else.
""".strip()

    # Standard multimodal message format
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": judge_prompt},
            ],
        }
    ]


# ------------------------------------------------------------
# Helper: convert generic chat messages -> vLLM input
# ------------------------------------------------------------
def prepare_inputs_for_vllm(
    image_path: str,
    messages: List[Dict[str, Any]],
    processor,
):
    # Let HF processor render the model-specific chat string
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image = Image.open(image_path).convert("RGB")

    return {
        "prompt": prompt,
        "multi_modal_data": {
            "image": image
        },
    }


# ------------------------------------------------------------
# JSON extraction helper
# ------------------------------------------------------------
def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())

    raise ValueError("JSON not found in model output")


# ------------------------------------------------------------
# Statistical calculation helpers
# ------------------------------------------------------------
def calculate_statistics(values: List[float]) -> Dict[str, Any]:
    """Calculate mean, variance, and std_dev for a list of values."""
    if not values:
        return {
            "mean": None,
            "variance": None,
            "std_dev": None,
            "values": []
        }

    mean_val = round(statistics.mean(values), 4)
    if len(values) < 2:
        variance_val = None
        std_dev_val = None
    else:
        variance_val = round(statistics.variance(values), 4)
        std_dev_val = round(statistics.stdev(values), 4)

    return {
        "mean": mean_val,
        "variance": variance_val,
        "std_dev": std_dev_val,
        "values": values
    }


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


# ------------------------------------------------------------
# File helpers
# ------------------------------------------------------------
def get_pngs_from_folder(folder_path: str) -> List[str]:
    if not os.path.isdir(folder_path):
        return []

    return [
        os.path.join(folder_path, f)
        for f in sorted(os.listdir(folder_path))
        if f.lower().endswith(".png")
    ]


def compute_total_dimension_score(dimension_scores: Dict[str, Any]) -> float:
    if not isinstance(dimension_scores, dict):
        return 0.0
    return round(sum(safe_float(v) for v in dimension_scores.values()), 4)


# ------------------------------------------------------------
# Process single image with n iterations
# ------------------------------------------------------------
def process_image_with_iterations(
    image_path: str,
    user_prompt: str,
    processor,
    llm,
    sampling_params,
    n_iterations: int,
) -> Dict[str, Any]:
    """Process a single image n times and aggregate results."""

    messages = build_ui_judge_messages(image_path, user_prompt)
    vllm_input = prepare_inputs_for_vllm(
        image_path=image_path,
        messages=messages,
        processor=processor,
    )

    iterations = []

    for iteration in range(1, n_iterations + 1):
        print(f"  Iteration {iteration}/{n_iterations}...")
        print(vllm_input)

        try:
            outputs = llm.generate([vllm_input], sampling_params=sampling_params)
            generated_text = outputs[0].outputs[0].text

            try:
                parsed = extract_json(generated_text)
                parsed["iteration"] = iteration
                parsed["raw_output"] = generated_text
                print(generated_text)
                parsed["total_dimension_score"] = compute_total_dimension_score(
                    parsed.get("dimension_scores", {})
                )
                iterations.append(parsed)

                print(f"    Overall score: {parsed.get('overall_score', 'N/A')}")

            except Exception as e:
                print(f"    Failed to parse JSON: {e}")
                iterations.append({
                    "iteration": iteration,
                    "error": str(e),
                    "raw_output": generated_text
                })

        except Exception as e:
            print(f"    Model inference failed: {e}")
            iterations.append({
                "iteration": iteration,
                "error": str(e),
                "raw_output": ""
            })

    # Calculate per-image statistics
    valid_iterations = [it for it in iterations if "error" not in it]

    if not valid_iterations:
        return {
            "image_name": os.path.basename(image_path),
            "image_path": image_path,
            "iterations": iterations,
            "statistics": {
                "mean_overall_score": None,
                "variance_overall_score": None,
                "std_dev_overall_score": None,
                "dimension_statistics": {}
            },
            "errors": len(iterations)
        }

    # Calculate overall statistics
    overall_scores = [it.get("overall_score", 0) for it in valid_iterations]
    overall_stats = calculate_statistics(overall_scores)

    # Calculate dimension statistics
    dimension_keys = [
        "visual_appeal",
        "visual_hierarchy",
        "readability",
        "color_harmony",
    ]

    dimension_statistics = {}
    for key in dimension_keys:
        dim_values = []
        for it in valid_iterations:
            dim_scores = it.get("dimension_scores", {})
            dim_values.append(safe_float(dim_scores.get(key, 0)))
        dimension_statistics[key] = calculate_statistics(dim_values)

    return {
        "image_name": os.path.basename(image_path),
        "image_path": image_path,
        "iterations": iterations,
        "statistics": {
            "mean_overall_score": overall_stats["mean"],
            "variance_overall_score": overall_stats["variance"],
            "std_dev_overall_score": overall_stats["std_dev"],
            "dimension_statistics": dimension_statistics
        },
        "errors": len(iterations) - len(valid_iterations)
    }


# ------------------------------------------------------------
# Aggregate folder-level statistics
# ------------------------------------------------------------
def aggregate_folder_statistics(image_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate statistics across all images and iterations."""

    # Collect overall scores (means from each image)
    overall_scores = []

    # Collect variances from each image for overall score and each dimension
    overall_variances = []
    dimension_variances_map = {
        "visual_appeal": [],
        "visual_hierarchy": [],
        "readability": [],
        "color_harmony": [],
    }

    # Collect mean scores for each dimension
    dimension_means_map = {
        "visual_appeal": [],
        "visual_hierarchy": [],
        "readability": [],
        "color_harmony": [],
    }

    for img_result in image_results:
        if "statistics" not in img_result or img_result["errors"] > 0:
            continue

        # Get overall score mean
        if img_result["statistics"]["mean_overall_score"] is not None:
            overall_scores.append(img_result["statistics"]["mean_overall_score"])

        # Get overall score variance
        if img_result["statistics"]["variance_overall_score"] is not None:
            overall_variances.append(img_result["statistics"]["variance_overall_score"])

        # Collect dimension means and variances
        dim_stats = img_result["statistics"]["dimension_statistics"]
        for key in dimension_means_map:
            if dim_stats.get(key, {}).get("mean") is not None:
                dimension_means_map[key].append(dim_stats[key]["mean"])
            if dim_stats.get(key, {}).get("variance") is not None:
                dimension_variances_map[key].append(dim_stats[key]["variance"])

    # Calculate overall statistics (mean of per-image means)
    overall_stats = calculate_statistics(overall_scores)

    # Calculate overall variance as average of per-image variances
    overall_variance_avg = round(statistics.mean(overall_variances), 4) if overall_variances else None
    overall_std_dev_avg = round(overall_variance_avg ** 0.5, 4) if overall_variance_avg is not None else None

    # Calculate dimension statistics (mean of per-image means)
    folder_dimension_statistics = {}
    for key in dimension_means_map:
        folder_dimension_statistics[key] = calculate_statistics(dimension_means_map[key])

    # Calculate dimension variance as average of per-image variances
    for key in dimension_variances_map:
        if dimension_variances_map[key]:
            variance_avg = round(statistics.mean(dimension_variances_map[key]), 4)
            folder_dimension_statistics[key]["variance"] = variance_avg
            # Recalculate std_dev from the average variance
            folder_dimension_statistics[key]["std_dev"] = round(variance_avg ** 0.5, 4)

    return {
        "mean_overall_score": overall_stats["mean"],
        "variance_overall_score": overall_variance_avg,
        "std_dev_overall_score": overall_std_dev_avg,
        "dimension_statistics": folder_dimension_statistics
    }


# ------------------------------------------------------------
# Save per-image statistics
# ------------------------------------------------------------
def save_per_image_stats(
    folder_path: str,
    image_results: List[Dict[str, Any]],
    n_iterations: int,
    model_path: str,
    user_prompt: str,
    output_dir: str,
    temperature: float = 0.7,
    top_k: int = 25,
):
    folder_name = os.path.basename(os.path.normpath(folder_path))
    model_name = os.path.basename(os.path.normpath(model_path))

    per_image_stats = {
        "folder_name": folder_name,
        "folder_path": folder_path,
        "model_path": model_path,
        "n_iterations": n_iterations,
        "prompt": user_prompt,
        "temperature": temperature,
        "top_k": top_k,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "images": image_results
    }

    output_path = os.path.join(output_dir, f"{folder_name}_per_image_stats_{model_name}_temp{temperature}_topk{top_k}.json")
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(per_image_stats, f, indent=2, ensure_ascii=False)

    return output_path


# ------------------------------------------------------------
# Save overall statistics
# ------------------------------------------------------------
def save_overall_stats(
    folder_path: str,
    image_results: List[Dict[str, Any]],
    folder_statistics: Dict[str, Any],
    n_iterations: int,
    model_path: str,
    user_prompt: str,
    output_dir: str,
    temperature: float = 0.7,
    top_k: int = 25,
):
    folder_name = os.path.basename(os.path.normpath(folder_path))
    model_name = os.path.basename(os.path.normpath(model_path))

    num_images = len(image_results)
    total_iterations_run = num_images * n_iterations
    successful_iterations = sum(len(r.get("iterations", [])) - r.get("errors", 0) for r in image_results)

    overall_stats = {
        "folder_name": folder_name,
        "folder_path": folder_path,
        "model_path": model_path,
        "n_iterations": n_iterations,
        "prompt": user_prompt,
        "temperature": temperature,
        "top_k": top_k,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "num_images": num_images,
        "total_iterations_run": total_iterations_run,
        "successful_iterations": successful_iterations,
        "failed_iterations": total_iterations_run - successful_iterations,
        "statistics": folder_statistics
    }

    output_path = os.path.join(output_dir, f"{folder_name}_overall_stats_{model_name}_temp{temperature}_topk{top_k}.json")
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(overall_stats, f, indent=2, ensure_ascii=False)

    return output_path


# ------------------------------------------------------------
# Main folder processor
# ------------------------------------------------------------
def process_folder(
    folder_path: str,
    user_prompt: str,
    processor,
    llm,
    sampling_params,
    n_iterations: int,
    output_dir: str,
    model_path: str,
    temperature: float = 0.7,
    top_k: int = 25,
):
    image_list = get_pngs_from_folder(folder_path)

    if not image_list:
        print(f"\nNo PNG files found in folder: {folder_path}")
        return

    print("\n" + "=" * 100)
    print(f"Processing folder: {folder_path}")
    print(f"Found {len(image_list)} PNG files")
    print(f"Running {n_iterations} iterations per image")

    image_results: List[Dict[str, Any]] = []

    for img_idx, img in enumerate(image_list, start=1):
        print(f"\n[{img_idx}/{len(image_list)}] Processing {os.path.basename(img)}")
        result = process_image_with_iterations(
            image_path=img,
            user_prompt=user_prompt,
            processor=processor,
            llm=llm,
            sampling_params=sampling_params,
            n_iterations=n_iterations,
        )
        image_results.append(result)

    # Calculate folder-level statistics
    folder_statistics = aggregate_folder_statistics(image_results)

    # Save outputs
    per_image_path = save_per_image_stats(
        folder_path=folder_path,
        image_results=image_results,
        n_iterations=n_iterations,
        model_path=model_path,
        user_prompt=user_prompt,
        output_dir=output_dir,
        temperature=temperature,
        top_k=top_k,
    )
    print(f"\nSaved per-image statistics to: {per_image_path}")

    overall_path = save_overall_stats(
        folder_path=folder_path,
        image_results=image_results,
        folder_statistics=folder_statistics,
        n_iterations=n_iterations,
        model_path=model_path,
        user_prompt=user_prompt,
        output_dir=output_dir,
        temperature=temperature,
        top_k=top_k,
    )
    print(f"Saved overall statistics to: {overall_path}")

    # Print summary
    print(f"\nFolder Summary:")
    print(f"  Total images: {len(image_list)}")
    print(f"  Iterations per image: {n_iterations}")
    print(f"  Total iterations: {len(image_list) * n_iterations}")
    print(f"  Successful iterations: {image_results[0]['iterations'][0]['iteration'] if image_results else 0}")
    print(f"  Overall mean score: {folder_statistics['mean_overall_score']}")
    print(f"  Overall std dev: {folder_statistics['std_dev_overall_score']}")


# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "folders",
        nargs="+",
        help="One or more folder paths containing PNG images",
    )
    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help="HF model path or model id for a VLM supported by vLLM. Uses MODEL_PATH from script if not specified.",
    )
    parser.add_argument(
        "--prompt",
        default="Is this UI professional and aesthetically pleasing?",
        help="Prompt used for judging all images",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory where outputs will be stored. Defaults to each folder itself.",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of iterations to run the model for each image (default: 1)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=25,
        help="Top-k sampling parameter (default: 25)",
    )

    args = parser.parse_args()

    # Validate model path
    if not args.model:
        raise SystemExit("Error: MODEL_PATH not set in script and --model argument not provided. Please either set MODEL_PATH at the top of the script or provide --model argument.")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    llm = LLM(
        model=args.model,
        runner="generate",
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization="fp8"
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=4096,
        top_k=args.top_k,
        seed=None
    )

    for folder_path in args.folders:
        out_dir = args.output_dir or folder_path
        process_folder(
            folder_path=folder_path,
            user_prompt=args.prompt,
            processor=processor,
            llm=llm,
            sampling_params=sampling_params,
            n_iterations=args.n,
            output_dir=out_dir,
            model_path=args.model,
            temperature=args.temperature,
            top_k=args.top_k,
        )
