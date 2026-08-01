#!/usr/bin/env python3
"""
Visualization script for UI evaluation statistics - Version 6.
Creates a grid visualization comparing Qwen models across temperature and top_k.

Grid structure:
- Y-axis: Temperature values (0, 0.2, 0.5, 0.7, 1.0)
- X-axis: Top-k values (1, 25, 50, 75, 100)
- Each cell contains 5 colored circles (one per model) with overall score inside
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
# Path to the reports folder
REPORTS_DIR = "/home/c.kulkarni/llm_as_judge/reports"

# Output directory
OUTPUT_DIR = "/home/c.kulkarni/llm_as_judge/reports/charts_v6"

# Qwen models to compare (in order)
QWEN_MODELS = [
    "Qwen2.5-VL-3B-Instruct",
    "Qwen3-VL-2B-Instruct",
    "Qwen3-VL-4B-Instruct",
    "Qwen3-VL-8B-Instruct",
]

# Temperature values (Y-axis)
TEMPERATURES = ["0.0", "0.2", "0.5", "0.7", "1.0"]

# Top-k values (X-axis)
TOPK_VALUES = ["1", "25", "50", "75", "100"]

# Colors for each model
MODEL_COLORS = {
    "Qwen2.5-VL-3B-Instruct": "#1f77b4",   # blue
    "Qwen3-VL-2B-Instruct": "#ff7f0e",     # orange
    "Qwen3-VL-4B-Instruct": "#2ca02c",     # green
    "Qwen3-VL-8B-Instruct": "#d62728",     # red
}

# Short names for legend
MODEL_SHORT_NAMES = {
    "Qwen2.5-VL-3B-Instruct": "Qwen2.5-3B",
    "Qwen3-VL-2B-Instruct": "Qwen3-2B",
    "Qwen3-VL-4B-Instruct": "Qwen3-4B",
    "Qwen3-VL-8B-Instruct": "Qwen3-8B",
}

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def parse_filename(filename):
    """Extract model name, temperature, and top_k from filename."""
    temp_match = re.search(r'temp([\d.]+)', filename)
    topk_match = re.search(r'topk(\d+)', filename)
    model_match = re.search(r'stats_(.+?)_temp', filename)

    if temp_match and topk_match and model_match:
        model_name = model_match.group(1)
        temperature = temp_match.group(1)
        top_k = topk_match.group(1)
        return model_name, temperature, top_k

    return None


def load_overall_stats(reports_dir):
    """
    Load overall stats from all JSON files.
    Returns: Dict of {(model, temp, top_k): overall_score}
    """
    files = os.listdir(reports_dir)
    overall_files = [f for f in files if 'overall_stats' in f and f.endswith('.json')]

    data = {}

    for filename in overall_files:
        parsed = parse_filename(filename)
        if not parsed:
            continue

        model, temp, top_k = parsed

        # Only include Qwen models
        if model not in QWEN_MODELS:
            continue

        filepath = os.path.join(reports_dir, filename)
        try:
            with open(filepath, 'r') as f:
                content = json.load(f)

            overall_score = content.get('statistics', {}).get('mean_overall_score')
            data[(model, temp, top_k)] = overall_score
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    return data


def create_grid_visualization(data, output_dir):
    """
    Create a grid visualization with colored squares showing overall scores.

    Each cell in the grid contains 4 colored squares (one per model) with the score inside.
    Thin borders between squares within a cell, thick borders between cells.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')

    n_temps = len(TEMPERATURES)
    n_topk = len(TOPK_VALUES)
    n_models = len(QWEN_MODELS)

    # Grid dimensions
    cell_width = 1.0
    cell_height = 1.0

    # Sub-cell dimensions (each cell is divided into 2x2 grid)
    sub_width = cell_width / 2
    sub_height = cell_height / 2

    # Positions for each model within a cell (2x2 grid)
    # Top-left, Top-right, Bottom-left, Bottom-right
    sub_positions = [
        (0, 0.5),       # Top-left (model 0)
        (0.5, 0.5),     # Top-right (model 1)
        (0, 0),         # Bottom-left (model 2)
        (0.5, 0),       # Bottom-right (model 3)
    ]

    # Draw thick black cell borders (between different temp/topk cells)
    for i in range(n_topk + 1):
        ax.axvline(x=i * cell_width, color='black', linewidth=4, zorder=1)

    for j in range(n_temps + 1):
        ax.axhline(y=j * cell_height, color='black', linewidth=4, zorder=1)

    # Fill cells with colored squares
    for i, temp in enumerate(TEMPERATURES):
        for j, topk in enumerate(TOPK_VALUES):
            # Cell bottom-left corner
            cell_x = j * cell_width
            cell_y = (n_temps - 1 - i) * cell_height

            # Draw sub-squares for each model
            for k, model in enumerate(QWEN_MODELS):
                score = data.get((model, temp, topk), None)

                # Sub-square position
                sub_x = sub_positions[k][0] * cell_width
                sub_y = sub_positions[k][1] * cell_height

                # Draw colored rectangle
                rect_x = cell_x + sub_x
                rect_y = cell_y + sub_y

                if score is not None:
                    color = MODEL_COLORS[model]

                    # Draw rectangle
                    rect = plt.Rectangle(
                        (rect_x, rect_y),
                        sub_width,
                        sub_height,
                        facecolor=color,
                        edgecolor='white',
                        linewidth=1,  # Thin border between sub-cells
                        alpha=0.85,
                        zorder=2
                    )
                    ax.add_patch(rect)

                    # Add score text
                    text_x = rect_x + sub_width / 2
                    text_y = rect_y + sub_height / 2
                    ax.text(text_x, text_y, f'{score:.1f}',
                           ha='center', va='center',
                           fontsize=10, fontweight='bold',
                           color='white', zorder=3)
                else:
                    # No data - draw empty rectangle
                    rect = plt.Rectangle(
                        (rect_x, rect_y),
                        sub_width,
                        sub_height,
                        facecolor='#e0e0e0',
                        edgecolor='white',
                        linewidth=1,
                        alpha=0.5,
                        zorder=2
                    )
                    ax.add_patch(rect)
                    ax.text(rect_x + sub_width/2, rect_y + sub_height/2, 'N/A',
                           ha='center', va='center',
                           fontsize=8, color='#666666', zorder=3)

    # Set axis labels
    ax.set_xlabel('Top-k', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Temperature', fontsize=14, fontweight='bold', labelpad=10)

    # Set tick positions and labels (center of each cell)
    ax.set_xticks([i * cell_width + cell_width / 2 for i in range(n_topk)])
    ax.set_xticklabels(TOPK_VALUES, fontsize=12)

    ax.set_yticks([i * cell_height + cell_height / 2 for i in range(n_temps)])
    ax.set_yticklabels(TEMPERATURES[::-1], fontsize=12)

    # Set axis limits
    ax.set_xlim(0, n_topk * cell_width)
    ax.set_ylim(0, n_temps * cell_height)

    # Remove default spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Title
    ax.set_title('Qwen Model Comparison: Overall Score by Temperature & Top-k\n'
                 '(Each colored square shows the overall score for a model)',
                 fontsize=14, fontweight='bold', pad=20)

    # Create legend with colored squares
    legend_elements = []
    for model in QWEN_MODELS:
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1,
                         facecolor=MODEL_COLORS[model],
                         edgecolor='white',
                         linewidth=1,
                         label=MODEL_SHORT_NAMES[model])
        )

    ax.legend(handles=legend_elements,
              loc='upper left',
              bbox_to_anchor=(1.02, 1),
              fontsize=11,
              title='Model',
              title_fontsize=12,
              framealpha=0.9,
              facecolor='white',
              edgecolor='#cccccc')

    plt.tight_layout()

    # Save
    output_file = os.path.join(output_dir, 'qwen_model_comparison_grid.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
    print(f"Saved: {output_file}")
    plt.close()

    return output_file


# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    print("="*70)
    print("VISUALIZATION SCRIPT - VERSION 6")
    print("(Qwen Model Comparison Grid: Temperature vs Top-k)")
    print("="*70)
    print(f"Reports directory: {REPORTS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Load data
    data = load_overall_stats(REPORTS_DIR)

    print(f"Loaded {len(data)} data points.")
    print(f"Models: {QWEN_MODELS}")
    print(f"Temperatures: {TEMPERATURES}")
    print(f"Top-k values: {TOPK_VALUES}")
    print()

    # Create visualization
    create_grid_visualization(data, OUTPUT_DIR)

    print("\n" + "="*70)
    print(f"COMPLETED! Chart saved in: {OUTPUT_DIR}")
    print("="*70)