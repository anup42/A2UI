#!/usr/bin/env python3
"""
Visualization script for UI evaluation statistics - Version 2.
Processes all model reports in the reports folder and generates:
1. Overall statistics (average across all images)
2. Per-image statistics with IMAGE THUMBNAILS on x-axis instead of text labels

The script automatically detects model name, temperature, and top_k from filenames.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from pathlib import Path
from PIL import Image

# =========================================================
# CONFIGURATION
# =========================================================
# Path to the reports folder containing overall and per_image stats
REPORTS_DIR = "/home/c.kulkarni/llm_as_judge/reports"

# Path to the folder containing UI images
IMAGES_DIR = "/home/c.kulkarni/llm_as_judge/testset_ui"

# Output directory for generated charts
OUTPUT_DIR = "/home/c.kulkarni/llm_as_judge/reports/charts_v2"

# Thumbnail size for images on x-axis (width, height in inches)
THUMBNAIL_SIZE = (0.6, 0.6)

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def parse_filename(filename):
    """
    Extract model name, temperature, and top_k from filename.

    Examples:
    - testset_ui_overall_stats_Qwen2.5-VL-3B-Instruct_temp0.5_topk25.json
    - testset_ui_per_image_stats_internvl3_5-2B-Instruct_temp0.0_topk1.json

    Returns: (model_name, temperature, top_k) or None if parsing fails
    """
    # Pattern to match: temp{number}_topk{number}
    temp_match = re.search(r'temp([\d.]+)', filename)
    topk_match = re.search(r'topk(\d+)', filename)

    # Extract model name - between "stats_" and "_temp"
    # Handle both overall and per_image files
    model_match = re.search(r'stats_(.+?)_temp', filename)

    if temp_match and topk_match and model_match:
        model_name = model_match.group(1)
        temperature = temp_match.group(1)
        top_k = topk_match.group(1)
        return model_name, temperature, top_k

    return None


def get_model_report_pairs(reports_dir):
    """
    Find all matching pairs of overall and per_image stats files.

    Returns: List of tuples (overall_file, per_image_file, model_name, temp, top_k)
    """
    files = os.listdir(reports_dir)

    # Separate overall and per_image files
    overall_files = [f for f in files if 'overall_stats' in f and f.endswith('.json')]
    per_image_files = [f for f in files if 'per_image_stats' in f and f.endswith('.json')]

    pairs = []

    for overall_file in overall_files:
        parsed = parse_filename(overall_file)
        if not parsed:
            print(f"Warning: Could not parse {overall_file}")
            continue

        model_name, temp, top_k = parsed

        # Find matching per_image file
        # Construct the expected per_image filename
        per_image_file = overall_file.replace('overall_stats', 'per_image_stats')

        if per_image_file in per_image_files:
            pairs.append((
                os.path.join(reports_dir, overall_file),
                os.path.join(reports_dir, per_image_file),
                model_name,
                temp,
                top_k
            ))
        else:
            print(f"Warning: No matching per_image file for {overall_file}")

    return pairs


def load_and_resize_image(image_path, thumbnail_size=(100, 100)):
    """
    Load an image and resize it to thumbnail size.

    Args:
        image_path: Path to the image file
        thumbnail_size: Tuple of (width, height) in pixels

    Returns:
        PIL Image object or None if loading fails
    """
    try:
        img = Image.open(image_path)
        # Convert to RGBA if necessary (handle different image modes)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        # Resize maintaining aspect ratio
        img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        print(f"Warning: Could not load image {image_path}: {e}")
        return None


def generate_overall_chart(overall_data, model_name, temp, top_k, output_dir):
    """
    Generate overall statistics chart (average across all images).
    Same design as original script.
    """
    dim_stats = overall_data["statistics"]["dimension_statistics"]

    # Prepare data - filter out dimensions with None values
    dimensions = []
    means = []
    std_devs = []
    for dim in dim_stats.keys():
        if dim_stats[dim]["mean"] is not None and dim_stats[dim]["std_dev"] is not None:
            dimensions.append(dim)
            means.append(dim_stats[dim]["mean"] if dim_stats[dim]["mean"] is not None else 0)
            std_devs.append(dim_stats[dim]["std_dev"] if dim_stats[dim]["std_dev"] is not None else 0)

    # Add overall score to the data (check for None values)
    overall_mean = overall_data["statistics"].get("mean_overall_score")
    overall_std = overall_data["statistics"].get("std_dev_overall_score")

    dimensions_with_overall = dimensions + ['overall_score']
    means_with_overall = means + [overall_mean if overall_mean is not None else 0]
    std_devs_with_overall = std_devs + [overall_std if overall_std is not None else 0]

    # Format dimension names for display
    dimension_labels = [dim.replace('_', ' ').title() for dim in dimensions_with_overall]

    # Create figure with better aesthetics
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')

    # Color palette - professional colors (added one more for overall)
    colors = ['#4ECDC4', '#FF6B6B', '#95E1D3', '#F38181', '#6C5CE7']

    # Plot with narrower bars
    bar_width = 0.5
    x_pos = np.arange(len(dimensions_with_overall))
    bars = ax.bar(x_pos, means_with_overall, bar_width, yerr=std_devs_with_overall, capsize=6,
                   color=colors, alpha=0.9, edgecolor='white', linewidth=1.5,
                   error_kw={'elinewidth': 2, 'ecolor': '#333333', 'capthick': 2})

    # Styling
    ax.set_xlabel('Evaluation Dimensions', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Score', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title('Mean Scores with Standard Deviation', fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(dimension_labels, rotation=0, ha='center', fontsize=11)
    ax.set_ylim(0, 5)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.grid(axis='y', alpha=0.3, linestyle='--', color='#cccccc')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

    # Add value labels on bars
    for bar, mean, std in zip(bars, means_with_overall, std_devs_with_overall):
        height = bar.get_height()
        ax.annotate(f'{mean:.2f} ± {std:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height + std),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    color='#333333')

    # Add main title with parameters
    fig.suptitle(f'UI Evaluation Statistics - Overall Average\n'
                 f'Model: {model_name} | '
                 f'Prompt: "{overall_data.get("prompt", "N/A")}"\n'
                 f'Parameters: top_k={top_k}, temp={temp} | '
                 f'Images: {overall_data.get("num_images", "N/A")} | Iterations: {overall_data.get("n_iterations", "N/A")}',
                 fontsize=11, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save the figure with model name, temp, top_k in filename
    output_file = os.path.join(output_dir, f"overall_stats_{model_name}_temp{temp}_topk{top_k}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
    print(f"Overall statistics saved to: {output_file}")
    plt.close()

    return output_file


def generate_per_image_chart_with_thumbnails(per_image_data, model_name, temp, top_k, output_dir, images_dir):
    """
    Generate per-image statistics chart with IMAGE THUMBNAILS on x-axis.

    Instead of text labels for image names, this version displays the actual
    resized images as thumbnails on the x-axis.
    """
    images = per_image_data["images"]

    # Check if images list is empty
    if not images:
        raise ValueError("No images found in per_image_data")

    # Extract image names and their dimension scores
    image_names = []
    all_scores = {
        "visual_appeal": [],
        "visual_hierarchy": [],
        "readability": [],
        "color_harmony": [],
        "overall_score": []
    }

    # Get dimensions from the first image (assuming all have same dimensions)
    dimensions = list(images[0]["statistics"]["dimension_statistics"].keys()) if images else []

    for img in images:
        image_names.append(img["image_name"].replace('.png', ''))
        dim_stats_img = img["statistics"]["dimension_statistics"]
        for dim in all_scores.keys():
            if dim == "overall_score":
                overall_score = img["statistics"].get("mean_overall_score")
                all_scores[dim].append(overall_score if overall_score is not None else 0)
            else:
                mean_val = dim_stats_img.get(dim, {}).get("mean")
                all_scores[dim].append(mean_val if mean_val is not None else 0)

    # Create figure with more height to accommodate thumbnails
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')

    # X positions for images
    x = np.arange(len(image_names))
    bar_width = 0.15

    # Colors for each dimension
    dim_colors = {
        "visual_appeal": '#4ECDC4',
        "visual_hierarchy": '#FF6B6B',
        "readability": '#95E1D3',
        "color_harmony": '#F38181',
        "overall_score": '#6C5CE7'
    }

    # Plot bars for each dimension
    bars_list = []
    for i, (dim, scores) in enumerate(all_scores.items()):
        bars = ax.bar(x + i * bar_width, scores, bar_width,
                       label=dim.replace('_', ' ').title(),
                       color=dim_colors[dim], alpha=0.9,
                       edgecolor='white', linewidth=1)
        bars_list.append(bars)

    # Styling
    ax.set_ylabel('Mean Score', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title('Per-Image Dimension Scores', fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x + bar_width * 2)
    ax.set_ylim(0, 5)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.grid(axis='y', alpha=0.3, linestyle='--', color='#cccccc')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

    # Add legend
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10,
              facecolor='white', edgecolor='#cccccc')

    # =========================================================
    # ADD IMAGE THUMBNAILS ON X-AXIS
    # =========================================================
    # Get the current axis position
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Calculate thumbnail position (below the x-axis)
    thumbnail_y_pos = -0.6  # Position below the chart
    thumbnail_height = 0.5  # Height in axis coordinates

    # Create a new inset axes for each image thumbnail
    for i, img_name in enumerate(image_names):
        # Calculate x position (center of the image group)
        x_pos = x[i] + bar_width * 2

        # Load the image
        img_path = os.path.join(images_dir, f"{img_name}.png")
        pil_img = load_and_resize_image(img_path, thumbnail_size=(80, 80))

        if pil_img is not None:
            # Convert PIL image to numpy array
            img_array = np.array(pil_img)

            # Create inset axes for the thumbnail
            # Position: [left, bottom, width, height] in figure coordinates
            # We need to transform axis coordinates to figure coordinates

            # Get the figure position of the main axis
            ax_pos = ax.get_position()

            # Calculate thumbnail position in figure coordinates
            # Map x_pos from axis coordinates to figure coordinates
            x_fig = ax_pos.x0 + (x_pos - xlim[0]) / (xlim[1] - xlim[0]) * ax_pos.width

            # Width of each thumbnail in figure coordinates
            thumb_width_fig = (bar_width * 5) / (xlim[1] - xlim[0]) * ax_pos.width * 1.5
            thumb_height_fig = thumb_width_fig  # Square thumbnails

            # Y position (below the x-axis)
            y_fig = ax_pos.y0 - thumb_height_fig - 0.02

            # Create inset axes
            inset_ax = fig.add_axes([x_fig - thumb_width_fig/2, y_fig, thumb_width_fig, thumb_height_fig])
            inset_ax.imshow(img_array)
            inset_ax.axis('off')
            inset_ax.set_frame_on(True)
            for spine in inset_ax.spines.values():
                spine.set_color('#cccccc')
                spine.set_linewidth(1)

    # Adjust the main plot to leave room for thumbnails
    plt.subplots_adjust(bottom=0.2)

    # Add main title with parameters
    fig.suptitle(f'UI Evaluation Statistics - Per Image\n'
                 f'Model: {model_name} | '
                 f'Prompt: "{per_image_data.get("prompt", "N/A")}"\n'
                 f'Parameters: top_k={top_k}, temp={temp} | '
                 f'Iterations per image: {per_image_data.get("n_iterations", "N/A")}',
                 fontsize=11, fontweight='bold', y=1.02)

    # Save the figure with model name, temp, top_k in filename
    output_file = os.path.join(output_dir, f"per_image_stats_{model_name}_temp{temp}_topk{top_k}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
    print(f"Per-image statistics saved to: {output_file}")
    plt.close()

    return output_file


# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    print("="*70)
    print("VISUALIZATION SCRIPT FOR ALL MODEL REPORTS - VERSION 2")
    print("(With Image Thumbnails on X-axis for Per-Image Charts)")
    print("="*70)
    print(f"Reports directory: {REPORTS_DIR}")
    print(f"Images directory: {IMAGES_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Get all model report pairs
    pairs = get_model_report_pairs(REPORTS_DIR)

    if not pairs:
        print("ERROR: No matching report pairs found!")
        exit(1)

    print(f"Found {len(pairs)} model report pairs to process.\n")

    # Track success/failure
    successful = 0
    failed = 0

    # Process each pair
    for idx, (overall_file, per_image_file, model_name, temp, top_k) in enumerate(pairs, 1):
        print(f"\n[{idx}/{len(pairs)}] Processing: {model_name} | temp={temp} | top_k={top_k}")
        print("-" * 60)

        # Load data
        try:
            with open(overall_file, 'r') as f:
                overall_data = json.load(f)

            with open(per_image_file, 'r') as f:
                per_image_data = json.load(f)
        except Exception as e:
            print(f"Error loading files: {e}")
            failed += 1
            continue

        # Generate charts
        try:
            generate_overall_chart(overall_data, model_name, temp, top_k, OUTPUT_DIR)
            generate_per_image_chart_with_thumbnails(per_image_data, model_name, temp, top_k, OUTPUT_DIR, IMAGES_DIR)
            successful += 1
        except Exception as e:
            print(f"Error generating charts: {e}")
            failed += 1
            continue

    print("\n" + "="*70)
    print(f"COMPLETED!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*70)