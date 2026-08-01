#!/usr/bin/env python3
"""
Visualization script for UI evaluation statistics - Version 5.
Compares overall scores across different Qwen models for a given temperature and top_k.

For each temperature + top_k combination, generates:
- One chart showing overall_score comparison
- X-axis: Image names
- Different colored bars for each Qwen model

This helps analyze how different Qwen model versions compare in scoring.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from pathlib import Path
from collections import defaultdict

# =========================================================
# CONFIGURATION
# =========================================================
# Path to the reports folder containing overall and per_image stats
REPORTS_DIR = "/home/c.kulkarni/llm_as_judge/reports"

# Output directory for generated charts
OUTPUT_DIR = "/home/c.kulkarni/llm_as_judge/reports/charts_v5"

# Qwen models to compare (in order of size)
QWEN_MODELS = [
    "Qwen2.5-VL-3B-Instruct",
    "Qwen3-VL-2B-Instruct",
    "Qwen3-VL-4B-Instruct",
    "Qwen3-VL-8B-Instruct",
]

# Colors for each Qwen model
MODEL_COLORS = {
    "Qwen2.5-VL-3B-Instruct": "#1f77b4",   # blue
    "Qwen3-VL-2B-Instruct": "#ff7f0e",     # orange
    "Qwen3-VL-4B-Instruct": "#2ca02c",     # green
    "Qwen3-VL-8B-Instruct": "#d62728",     # red
}

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def parse_filename(filename):
    """
    Extract model name, temperature, and top_k from filename.

    Returns: (model_name, temperature, top_k) or None if parsing fails
    """
    temp_match = re.search(r'temp([\d.]+)', filename)
    topk_match = re.search(r'topk(\d+)', filename)
    model_match = re.search(r'stats_(.+?)_temp', filename)

    if temp_match and topk_match and model_match:
        model_name = model_match.group(1)
        temperature = temp_match.group(1)
        top_k = topk_match.group(1)
        return model_name, temperature, top_k

    return None


def get_data_by_temp_topk(reports_dir):
    """
    Organize per_image stats files by (temperature, top_k) -> model_name -> file_path.
    Only includes Qwen models.

    Returns: Dict of {(temperature, top_k): {model_name: per_image_file_path}}
    """
    files = os.listdir(reports_dir)
    per_image_files = [f for f in files if 'per_image_stats' in f and f.endswith('.json')]

    data = defaultdict(lambda: {})

    for per_image_file in per_image_files:
        parsed = parse_filename(per_image_file)
        if not parsed:
            continue

        model_name, temp, top_k = parsed

        # Only include Qwen models
        if "Qwen" in model_name:
            data[(temp, top_k)][model_name] = os.path.join(reports_dir, per_image_file)

    return data


def load_per_image_data(file_path):
    """Load per_image stats from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def generate_model_comparison_chart(temp, top_k, model_data, output_dir):
    """
    Generate model comparison chart for a given temperature and top_k.

    Creates one chart showing overall_score comparison across Qwen models.

    Args:
        temp: Temperature value
        top_k: Top-k value
        model_data: Dict of {model_name: per_image_data}
        output_dir: Output directory for charts
    """
    # Get all unique image names across all models
    all_image_names = set()
    for model, data in model_data.items():
        if data and "images" in data:
            for img in data["images"]:
                all_image_names.add(img["image_name"].replace('.png', ''))

    # Sort image names (bad_*, good_*, u_*)
    image_names = sorted(list(all_image_names))

    if not image_names:
        print(f"  Warning: No images found for temp={temp} top_k={top_k}")
        return

    # Get available models (in defined order)
    available_models = [m for m in QWEN_MODELS if m in model_data and model_data[m] is not None]

    if not available_models:
        print(f"  Warning: No valid Qwen model data for temp={temp} top_k={top_k}")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(18, 8))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')

    # Number of images and models
    n_images = len(image_names)
    n_models = len(available_models)

    # Bar width calculations
    bar_width = 0.18
    group_width = n_models * bar_width
    x = np.arange(n_images) * (group_width + 0.3)  # Add spacing between image groups

    # Collect data for each model
    model_scores = {}
    for model in available_models:
        scores = []
        data = model_data.get(model)

        if data and "images" in data:
            # Create lookup by image name
            img_lookup = {img["image_name"].replace('.png', ''): img
                         for img in data["images"]}

            for img_name in image_names:
                img = img_lookup.get(img_name)
                if img:
                    score = img["statistics"].get("mean_overall_score", 0)
                    scores.append(score if score is not None else 0)
                else:
                    scores.append(0)
        else:
            scores = [0] * n_images

        model_scores[model] = scores

    # Plot bars for each model
    for idx, model in enumerate(available_models):
        color = MODEL_COLORS.get(model, "#333333")
        bar_positions = x + idx * bar_width

        # Shorter label for legend
        short_label = model.replace("-Instruct", "").replace("Qwen", "Qwen ")

        bars = ax.bar(bar_positions, model_scores[model], bar_width,
                      label=short_label,
                      color=color, alpha=0.85,
                      edgecolor='white', linewidth=0.5)

    # Styling
    ax.set_xlabel('Image', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Overall Score (1-5)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title('Overall Score - Qwen Model Comparison',
                 fontsize=15, fontweight='bold', pad=15)

    # X-axis ticks (center of each group)
    ax.set_xticks(x + group_width / 2 - bar_width / 2)
    ax.set_xticklabels(image_names, rotation=45, ha='right', fontsize=10)

    ax.set_ylim(0, 5.5)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.grid(axis='y', alpha=0.3, linestyle='--', color='#cccccc')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

    # Add legend
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10,
              facecolor='white', edgecolor='#cccccc',
              title='Model', title_fontsize=10)

    # Add main title with parameters
    fig.suptitle(f'Qwen Model Comparison (temp={temp}, top_k={top_k})\n'
                 f'Overall Score per Image',
                 fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save the figure
    output_file = os.path.join(output_dir,
                               f"qwen_model_comparison_temp{temp}_topk{top_k}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
    print(f"  Saved: {output_file}")
    plt.close()


# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    print("="*70)
    print("VISUALIZATION SCRIPT - VERSION 5")
    print("(Qwen Model Comparison for Fixed Temperature + Top-k)")
    print("="*70)
    print(f"Reports directory: {REPORTS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Models to compare: {QWEN_MODELS}")
    print()

    # Get data organized by (temp, top_k)
    data_by_temp_topk = get_data_by_temp_topk(REPORTS_DIR)

    if not data_by_temp_topk:
        print("ERROR: No per_image stats files found for Qwen models!")
        exit(1)

    print(f"Found {len(data_by_temp_topk)} temperature + top_k combinations.\n")

    # Process each (temp, top_k) combination
    for idx, ((temp, top_k), model_files) in enumerate(sorted(data_by_temp_topk.items()), 1):
        print(f"\n[{idx}/{len(data_by_temp_topk)}] Processing: temp={temp} | top_k={top_k}")
        print(f"  Models available: {list(model_files.keys())}")
        print("-" * 60)

        # Load all model data
        model_data = {}
        for model, file_path in model_files.items():
            try:
                model_data[model] = load_per_image_data(file_path)
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
                model_data[model] = None

        # Generate comparison chart
        try:
            generate_model_comparison_chart(temp, top_k, model_data, OUTPUT_DIR)
        except Exception as e:
            print(f"  Error generating charts: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*70)
    print(f"COMPLETED! Charts saved in: {OUTPUT_DIR}")
    print("="*70)