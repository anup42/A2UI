#!/usr/bin/env python3
"""
Generate bar chart visualizations for UI scoring model comparisons.
Creates bar charts showing mean ± standard deviation for each dimension.
X-axis: Image names (not model names)
Different models: grouped bar charts with error bars.
No shaded error regions - only error bars on top of bars.

Updated to process output from llm_as_judge_vllm_2.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# =========================================================
# CONFIG
# =========================================================
# Input: list of JSONL files from different models (output from llm_as_judge_vllm_2.py)
INPUT_JSONL_FILES = [
    "testset_ui/testset_ui_image_scores_qwen2.5_3b.jsonl",
    "testset_ui/testset_ui_image_scores_qwen2.5_7b.jsonl",
    "testset_ui/testset_ui_image_scores_qwen3_2b.jsonl",
    "testset_ui/testset_ui_image_scores_qwen3_4b.jsonl",
    "testset_ui/testset_ui_image_scores_internvl3_5_2b.jsonl",
    "testset_ui/testset_ui_image_scores_internvl3_5_8b.jsonl",
]

OUTPUT_DIR = "model_comparison_charts"

# Define dimensions to plot
DIMENSIONS = [
    "visual_appeal",
    "visual_hierarchy",
    "readability",
    "color_harmony",
]

# Define model colors
MODEL_COLORS = {
    "Qwen2.5-VL-3B-Instruct": "#1f77b4",  # blue
    "Qwen2.5-VL-7B-Instruct": "#ff4444",  # red
    "Qwen3-VL-2B-Instruct": "#9467bd",  # purple
    "Qwen3-VL-4B-Instruct": "#44aa20",  # green
    "internvl3_5-2B-Instruct": "#ff8c00",  # orange
    "internvl3_5-8B-Instruct": "#17becf",  # cyan
}

# Define ground truth/actual scores for each image
ACTUAL_SCORES = {
    "bad_1.png": 1,
    "bad_2.png": 1,
    "bad_3.png": 1,
    "bad_4.png": 1,
    "u_000031_01.png": 3,
    "u_000242_01.png": 3,
    "u_000402_01.png": 3,
    "u_000682_01.png": 3,
    "u_000884_01.png": 3,
    "good_1.png": 5,
    "good_2.png": 4,
    "good_3.png": 5,
    "good_4.png": 5,
}

# Model name mapping (from file name to display name)
MODEL_NAME_MAPPING = {
    "qwen2.5_3b": "Qwen2.5-VL-3B-Instruct",
    "qwen2.5_7b": "Qwen2.5-VL-7B-Instruct",
    "qwen3_2b": "Qwen3-VL-2B-Instruct",
    "qwen3_4b": "Qwen3-VL-4B-Instruct",
    "internvl3_5_2b": "internvl3_5-2B-Instruct",
    "internvl3_5_8b": "internvl3_5-8B-Instruct",
}

# Create output directory
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# =========================================================
# LOAD DATA FROM JSONL FILES
# =========================================================
def extract_model_name_from_filename(filename: str) -> str:
    """Extract model name from JSONL filename."""
    basename = Path(filename).stem  # e.g., "testset_ui_image_scores_qwen3_4b"
    parts = basename.split("_")

    # Find the model part (last 2 or 3 parts)
    for key in MODEL_NAME_MAPPING.keys():
        if key in basename:
            return MODEL_NAME_MAPPING[key]

    # Fallback: use last part
    return parts[-1] if parts else "unknown"

def load_jsonl_data(filepath: str) -> list:
    """Load data from a JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

all_data = []

for jsonl_file in INPUT_JSONL_FILES:
    if not Path(jsonl_file).exists():
        print(f"Warning: File not found: {jsonl_file}")
        continue

    model_name = extract_model_name_from_filename(jsonl_file)
    file_data = load_jsonl_data(jsonl_file)

    print(f"\nLoaded data from: {jsonl_file}")
    print(f"Model: {model_name}")
    print(f"Number of images: {len(file_data)}")

    for img_data in file_data:
        if 'error' not in img_data:  # Skip entries with errors
            dimension_scores = img_data.get('dimension_scores', {})
            all_data.append({
                'model': model_name,
                'image': img_data.get('image', ''),
                'dimension_scores': dimension_scores,
                'overall_score': img_data.get('overall_score', 0),
                'verdict': img_data.get('verdict', ''),
                'reason': img_data.get('reason', ''),
            })

print(f"\nTotal data points: {len(all_data)}")

if not all_data:
    print("ERROR: No valid data loaded.")
    exit(1)

# Get sorted list of unique image names and models
image_names = sorted(list(set(item['image'] for item in all_data)))
sorted_models = sorted(list(set(item['model'] for item in all_data)))

print(f"\nFound {len(sorted_models)} unique models: {sorted_models}")
print(f"Found {len(image_names)} unique images: {image_names}")

# =========================================================
# FUNCTION TO PLOT ONE DIMENSION (BAR CHART)
# =========================================================
def plot_dimension_bar_chart(dimension, data, output_dir, model_colors):
    """Create a grouped bar chart showing model comparison for one dimension.

    X-axis: Image names
    Y-axis: Score (1-5)
    Different models: grouped bar chart
    """

    plt.figure(figsize=(20, 8))

    # Prepare data for grouped bar chart
    n_images = len(image_names)
    n_models = len(sorted_models)
    bar_width = 0.12
    x = np.arange(n_images) * 1.5  # Add spacing between image groups

    for idx, model in enumerate(sorted_models):
        model_color = model_colors.get(model, "#000000")

        # Extract scores in image order
        image_scores = []

        for img in image_names:
            # Find data for this image
            img_data = next((item for item in data if item['model'] == model and item['image'] == img), None)
            if img_data:
                score = img_data['dimension_scores'].get(dimension, 0)
                image_scores.append(score)
            else:
                image_scores.append(0)

        # Calculate bar positions for this model
        bar_positions = x + (idx - n_models/2 + 0.5) * bar_width

        # Plot bars
        bars = plt.bar(
            bar_positions,
            image_scores,
            bar_width,
            label=model,
            color=model_color,
            alpha=0.8,
            edgecolor='black',
            linewidth=1.0,
        )

    # Customize chart
    plt.xticks(x, image_names, rotation=45, ha='right')
    plt.ylim(0, 5.5)
    plt.xlabel('Image', fontsize=12, fontweight='bold')
    plt.ylabel('Score (1-5)', fontsize=12, fontweight='bold')
    plt.title(f"{dimension.replace('_', ' ').title()}",
             fontsize=15, fontweight='bold', pad=20)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.tight_layout()

    # Save chart
    output_path = Path(output_dir) / f"{dimension}_bar_chart.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")

    return output_path

# =========================================================
# FUNCTION TO PLOT OVERALL SCORE (BAR CHART)
# =========================================================
def plot_overall_score_bar_chart(data, output_dir, model_colors, actual_scores):
    """Create a grouped bar chart showing overall score comparison across images.

    X-axis: Image names
    Y-axis: Overall Score (1-5)
    Different models: grouped bar chart
    Includes ground truth/actual scores as horizontal lines
    MSE shown in legend for each model
    """

    plt.figure(figsize=(20, 8))

    # Prepare data for grouped bar chart
    n_images = len(image_names)
    n_models = len(sorted_models)
    bar_width = 0.12
    x = np.arange(n_images) * 1.5  # Add spacing between image groups

    # Calculate MSE for each model
    model_mse = {}
    for model in sorted_models:
        squared_errors = []
        for img in image_names:
            img_data = next((item for item in data if item['model'] == model and item['image'] == img), None)
            if img_data:
                predicted = img_data['overall_score']
                actual = actual_scores.get(img, 0)
                squared_errors.append((predicted - actual) ** 2)
        model_mse[model] = sum(squared_errors) / len(squared_errors) if squared_errors else 0

    for idx, model in enumerate(sorted_models):
        model_color = model_colors.get(model, "#000000")

        # Extract overall scores in image order
        image_scores = []

        for img in image_names:
            # Find data for this image
            img_data = next((item for item in data if item['model'] == model and item['image'] == img), None)
            if img_data:
                image_scores.append(img_data['overall_score'])
            else:
                image_scores.append(0)

        # Calculate bar positions for this model
        bar_positions = x + (idx - n_models/2 + 0.5) * bar_width

        # Plot bars - include MSE in label
        mse = model_mse.get(model, 0)
        label_with_mse = f"{model} (MSE: {mse:.2f})"
        bars = plt.bar(
            bar_positions,
            image_scores,
            bar_width,
            label=label_with_mse,
            color=model_color,
            alpha=0.8,
            edgecolor='black',
            linewidth=1.0,
        )

    # Add ground truth/actual scores as horizontal lines with labels
    actual_scores_list = [actual_scores.get(img, 0) for img in image_names]

    # Plot actual scores as horizontal dashed lines with markers
    plt.plot(x, actual_scores_list, 'o--', color='black', linewidth=2, markersize=8, label='Actual Score', alpha=0.7, zorder=10)

    # Add value labels on actual score points
    for idx, (x_pos, score) in enumerate(zip(x, actual_scores_list)):
        plt.text(x_pos, score + 0.1, f'{score}',
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')

    # Customize chart
    plt.xticks(x, image_names, rotation=45, ha='right')
    plt.ylim(0, 5.5)
    plt.xlabel('Image', fontsize=12, fontweight='bold')
    plt.ylabel('Overall Score (1-5)', fontsize=12, fontweight='bold')
    plt.title('Overall Score Comparison\nFormula: 0.25 × (visual_appeal + visual_hierarchy + readability + color_harmony)',
             fontsize=15, fontweight='bold', pad=20)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.tight_layout()

    # Save chart
    output_path = Path(output_dir) / "overall_score_bar_chart.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")

    return output_path

# =========================================================
# GENERATE ALL DIMENSION BAR CHARTS
# =========================================================
print("\n" + "="*60)
print("GENERATING DIMENSION-WISE BAR CHARTS")
print("="*60)

for dimension in DIMENSIONS:
    plot_dimension_bar_chart(dimension, all_data, OUTPUT_DIR, MODEL_COLORS)

print("\n" + "="*60)
print("GENERATING OVERALL SCORE BAR CHART")
print("="*60)

plot_overall_score_bar_chart(all_data, OUTPUT_DIR, MODEL_COLORS, ACTUAL_SCORES)

print("\n" + "="*60)
print(f"All bar chart plots generated successfully in: {OUTPUT_DIR}/")
print("="*60)