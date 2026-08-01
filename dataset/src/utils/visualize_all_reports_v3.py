#!/usr/bin/env python3
"""
Visualization script for UI evaluation statistics - Version 3.
Compares scores across different temperatures for a given model and top_k.

For each model + top_k combination, generates:
- 5 charts (one per dimension): visual_appeal, visual_hierarchy, readability, color_harmony, overall_score
- X-axis: Image names
- Different colored bars for each temperature value

This helps analyze how temperature affects scoring across different images.
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
OUTPUT_DIR = "/home/c.kulkarni/llm_as_judge/reports/charts_v3"

# Dimensions to plot
DIMENSIONS = ["visual_appeal", "visual_hierarchy", "readability", "color_harmony", "overall_score"]

# Temperature values to compare (sorted)
TEMPERATURES = ["0.0", "0.2", "0.4", "0.5", "0.7", "0.8", "1.0"]

# Colors for each temperature
TEMP_COLORS = {
    "0.0": "#1f77b4",    # blue
    "0.2": "#ff7f0e",    # orange
    "0.4": "#2ca02c",    # green
    "0.5": "#d62728",    # red
    "0.7": "#9467bd",    # purple
    "0.8": "#8c564b",    # brown
    "1.0": "#e377c2",    # pink
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


def get_data_by_model_topk(reports_dir):
    """
    Organize per_image stats files by (model_name, top_k) -> temperature -> file_path.

    Returns: Dict of {(model_name, top_k): {temperature: per_image_file_path}}
    """
    files = os.listdir(reports_dir)
    per_image_files = [f for f in files if 'per_image_stats' in f and f.endswith('.json')]

    data = defaultdict(lambda: {})

    for per_image_file in per_image_files:
        parsed = parse_filename(per_image_file)
        if not parsed:
            continue

        model_name, temp, top_k = parsed
        data[(model_name, top_k)][temp] = os.path.join(reports_dir, per_image_file)

    return data


def load_per_image_data(file_path):
    """Load per_image stats from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def generate_temperature_comparison_charts(model_name, top_k, temp_data, output_dir):
    """
    Generate temperature comparison charts for a given model and top_k.

    Creates 5 charts (one per dimension) showing how scores vary across
    images and temperatures.

    Args:
        model_name: Name of the model
        top_k: Top-k value
        temp_data: Dict of {temperature: per_image_data}
        output_dir: Output directory for charts
    """
    # Get all unique image names across all temperatures
    all_image_names = set()
    for temp, data in temp_data.items():
        if data and "images" in data:
            for img in data["images"]:
                all_image_names.add(img["image_name"].replace('.png', ''))

    # Sort image names (bad_*, good_*, u_*)
    image_names = sorted(list(all_image_names))

    if not image_names:
        print(f"  Warning: No images found for {model_name} top_k={top_k}")
        return

    # Get available temperatures (sorted numerically)
    available_temps = sorted([t for t in temp_data.keys() if temp_data[t] is not None],
                             key=lambda x: float(x))

    if not available_temps:
        print(f"  Warning: No valid temperature data for {model_name} top_k={top_k}")
        return

    # For each dimension, create a chart
    for dimension in DIMENSIONS:
        generate_single_dimension_chart(
            model_name, top_k, dimension, image_names,
            available_temps, temp_data, output_dir
        )


def generate_single_dimension_chart(model_name, top_k, dimension, image_names,
                                     temperatures, temp_data, output_dir):
    """
    Generate a single chart for one dimension comparing temperatures.

    X-axis: Image names
    Grouped bars: Different temperatures
    Y-axis: Score for the given dimension
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 8))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')

    # Number of images and temperatures
    n_images = len(image_names)
    n_temps = len(temperatures)

    # Bar width calculations
    bar_width = 0.1
    group_width = n_temps * bar_width
    x = np.arange(n_images) * (group_width + 0.3)  # Add spacing between image groups

    # Collect data for each temperature
    temp_scores = {}
    for temp in temperatures:
        scores = []
        data = temp_data.get(temp)

        if data and "images" in data:
            # Create lookup by image name
            img_lookup = {img["image_name"].replace('.png', ''): img
                         for img in data["images"]}

            for img_name in image_names:
                img = img_lookup.get(img_name)
                if img:
                    if dimension == "overall_score":
                        score = img["statistics"].get("mean_overall_score", 0)
                    else:
                        dim_stats = img["statistics"].get("dimension_statistics", {})
                        score = dim_stats.get(dimension, {}).get("mean", 0)
                    scores.append(score if score is not None else 0)
                else:
                    scores.append(0)
        else:
            scores = [0] * n_images

        temp_scores[temp] = scores

    # Plot bars for each temperature
    for idx, temp in enumerate(temperatures):
        color = TEMP_COLORS.get(temp, "#333333")
        bar_positions = x + idx * bar_width

        bars = ax.bar(bar_positions, temp_scores[temp], bar_width,
                      label=f'temp={temp}',
                      color=color, alpha=0.85,
                      edgecolor='white', linewidth=0.5)

    # Styling
    ax.set_xlabel('Image', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Score (1-5)', fontsize=13, fontweight='bold', labelpad=10)

    # Format dimension name for title
    dim_title = dimension.replace('_', ' ').title()
    ax.set_title(f'{dim_title} - Temperature Comparison',
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
              title='Temperature', title_fontsize=10)

    # Add main title with parameters
    fig.suptitle(f'Temperature Comparison for {model_name} (top_k={top_k})\n'
                 f'Dimension: {dim_title}',
                 fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save the figure
    output_file = os.path.join(output_dir,
                               f"temp_comparison_{model_name}_topk{top_k}_{dimension}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
    print(f"  Saved: {dimension} -> {output_file}")
    plt.close()


# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    print("="*70)
    print("VISUALIZATION SCRIPT - VERSION 3")
    print("(Temperature Comparison per Model + Top_k)")
    print("="*70)
    print(f"Reports directory: {REPORTS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Get data organized by (model, top_k)
    data_by_model_topk = get_data_by_model_topk(REPORTS_DIR)

    if not data_by_model_topk:
        print("ERROR: No per_image stats files found!")
        exit(1)

    print(f"Found {len(data_by_model_topk)} model + top_k combinations.\n")

    # Process each (model, top_k) combination
    for idx, ((model_name, top_k), temp_files) in enumerate(sorted(data_by_model_topk.items()), 1):
        print(f"\n[{idx}/{len(data_by_model_topk)}] Processing: {model_name} | top_k={top_k}")
        print(f"  Temperatures available: {sorted(temp_files.keys(), key=lambda x: float(x))}")
        print("-" * 60)

        # Load all temperature data
        temp_data = {}
        for temp, file_path in temp_files.items():
            try:
                temp_data[temp] = load_per_image_data(file_path)
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
                temp_data[temp] = None

        # Generate comparison charts
        try:
            generate_temperature_comparison_charts(model_name, top_k, temp_data, OUTPUT_DIR)
        except Exception as e:
            print(f"  Error generating charts: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*70)
    print(f"COMPLETED! Charts saved in: {OUTPUT_DIR}")
    print("="*70)