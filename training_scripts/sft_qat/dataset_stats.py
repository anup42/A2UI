"""
Dataset Statistics Script for SFT Training Data.

Analyzes JSONL datasets and generates intent distribution visualizations.

Usage:
    python dataset_stats.py --data_paths file1.jsonl file2.jsonl
    python dataset_stats.py --data_paths "data/*.jsonl" --output_dir ./stats
"""

import argparse
import json
import os
import glob
from collections import Counter
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np

# Use seaborn's default color palettes - they're designed to be visually appealing


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num} in {file_path}: {e}")
    return data


def load_json(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSON file (array of objects)."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return [data]
    return []


def load_data_file(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSON or JSONL file."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.jsonl':
        return load_jsonl(file_path)
    elif ext == '.json':
        return load_json(file_path)
    else:
        try:
            return load_jsonl(file_path)
        except:
            return load_json(file_path)


def analyze_intents(data: List[Dict[str, Any]]) -> Dict[str, int]:
    """Extract and count all intents."""
    intents = [item.get("intent", "UNKNOWN") for item in data]
    return dict(Counter(intents).most_common())


def create_visualizations(intent_dist: Dict[str, int], total_samples: int, output_dir: str):
    """Create and save visualization plots."""
    os.makedirs(output_dir, exist_ok=True)

    num_intents = len(intent_dist)

    # 1. Full Intent Distribution Bar Chart (ALL intents)
    fig_height = max(12, num_intents * 0.4)  # Dynamic height based on number of intents
    fig, ax = plt.subplots(figsize=(16, fig_height))

    intents = list(intent_dist.keys())
    counts = list(intent_dist.values())
    # Use seaborn's husl palette - generates visually distinct colors for any number of items
    colors = sns.husl_palette(len(intents))

    bars = ax.barh(range(len(intents)), counts, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(intents)))
    ax.set_yticklabels(intents, fontsize=9)
    ax.set_xlabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(f'Intent Distribution (All {num_intents} Intents)\nTotal Samples: {total_samples:,}',
                 fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()

    # Add count labels on bars
    max_count = max(counts)
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(count + max_count * 0.01, i, f'{count}', va='center', fontsize=8)

    # Add grid
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "intent_distribution_full.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Top 20 Intent Distribution
    fig, ax = plt.subplots(figsize=(14, 10))

    top_n = min(20, num_intents)
    top_intents = list(intent_dist.keys())[:top_n]
    top_counts = [intent_dist[i] for i in top_intents]
    colors = sns.color_palette("husl", len(top_intents))

    bars = ax.barh(range(len(top_intents)), top_counts, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(top_intents)))
    ax.set_yticklabels(top_intents, fontsize=11)
    ax.set_xlabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Intent Distribution', fontsize=14, fontweight='bold', pad=15)
    ax.invert_yaxis()

    for i, (bar, count) in enumerate(zip(bars, top_counts)):
        ax.text(count + max(top_counts) * 0.01, i, f'{count}', va='center', fontsize=10, fontweight='bold')

    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "intent_distribution_top20.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Intent Frequency Histogram (distribution of counts)
    fig, ax = plt.subplots(figsize=(14, 8))

    counts_list = list(intent_dist.values())
    bins = min(50, len(set(counts_list)))

    n, bins_edges, patches = ax.hist(counts_list, bins=bins,
                                      edgecolor='white', alpha=0.85)

    # Color each bar with seaborn husl palette
    hist_colors = sns.color_palette("husl", len(patches))
    for i, patch in enumerate(patches):
        patch.set_facecolor(hist_colors[i])

    ax.set_xlabel('Intent Count', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Intents', fontsize=12, fontweight='bold')
    ax.set_title('Intent Frequency Distribution\n(How many intents have each count range)',
                 fontsize=14, fontweight='bold', pad=15)

    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "intent_frequency_histogram.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Summary Dashboard
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Summary text
    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')
    summary_text = f"""
    DATASET SUMMARY
    {'='*40}
    📊 Total Samples: {total_samples:,}
    🎯 Unique Intents: {num_intents}

    Top 5 Intents:
    {chr(10).join([f'  {i+1}. {k}: {v}' for i, (k, v) in enumerate(list(intent_dist.items())[:5])])}

    Bottom 5 Intents:
    {chr(10).join([f'  {i+1}. {k}: {v}' for i, (k, v) in enumerate(list(intent_dist.items())[-5:])])}
    """
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', edgecolor='#2196F3', linewidth=2))

    # Top 10 pie chart
    ax = fig.add_subplot(gs[0, 1])
    top_10 = dict(list(intent_dist.items())[:10])
    other_count = sum(v for k, v in intent_dist.items() if k not in top_10)
    if other_count > 0:
        top_10['Other'] = other_count

    colors = sns.color_palette("husl", len(top_10))
    wedges, texts, autotexts = ax.pie(top_10.values(), labels=top_10.keys(),
                                       autopct='%1.1f%%', colors=colors,
                                       startangle=90, pctdistance=0.75)
    ax.set_title('Top 10 Intents', fontsize=12, fontweight='bold')

    # Top 15 bar chart
    ax = fig.add_subplot(gs[1, 0])
    top_15 = dict(list(intent_dist.items())[:15])
    colors = sns.color_palette("husl", len(top_15))
    ax.barh(range(len(top_15)), list(top_15.values()), color=colors, edgecolor='white')
    ax.set_yticks(range(len(top_15)))
    ax.set_yticklabels(top_15.keys(), fontsize=9)
    ax.invert_yaxis()
    ax.set_title('Top 15 Intents', fontsize=12, fontweight='bold')
    ax.set_xlabel('Count')

    # Frequency histogram
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(counts_list, bins=30, color=sns.color_palette("husl")[0], edgecolor='white', alpha=0.85)
    ax.set_xlabel('Intent Count')
    ax.set_ylabel('Number of Intents')
    ax.set_title('Intent Frequency Distribution', fontsize=12, fontweight='bold')

    plt.suptitle('Dataset Statistics Dashboard', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dashboard.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n📊 Visualizations saved to: {output_dir}")
    print(f"   - intent_distribution_full.png (all {num_intents} intents)")
    print("   - intent_distribution_top20.png")
    print("   - intent_frequency_histogram.png")
    print("   - dashboard.png")


def print_summary(intent_dist: Dict[str, int], total_samples: int):
    """Print brief summary to console."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"📊 Total Samples: {total_samples:,}")
    print(f"🎯 Unique Intents: {len(intent_dist)}")
    print("\nTop 10 Intents:")
    for i, (intent, count) in enumerate(list(intent_dist.items())[:10]):
        pct = (count / total_samples) * 100
        print(f"  {i+1:2d}. {intent}: {count} ({pct:.1f}%)")
    print("=" * 60)


def expand_paths(paths: List[str]) -> List[str]:
    """Expand glob patterns and return list of file paths."""
    expanded = []
    for p in paths:
        if '*' in p:
            matches = glob.glob(p)
            expanded.extend(sorted(matches))
        else:
            expanded.append(p)
    return expanded


def main():
    parser = argparse.ArgumentParser(description="Analyze SFT dataset intent distribution")
    parser.add_argument("--data_paths", type=str, nargs='+', required=True,
                        help="Paths to JSON/JSONL data files (supports glob patterns)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save plots (default: same as first data file)")
    args = parser.parse_args()

    # Expand glob patterns
    file_paths = expand_paths(args.data_paths)

    if not file_paths:
        print("Error: No files found matching the provided paths")
        return

    print(f"Found {len(file_paths)} file(s):")
    for fp in file_paths:
        print(f"  - {fp}")

    # Load all data
    all_data = []
    for fp in file_paths:
        print(f"Loading: {fp}")
        data = load_data_file(fp)
        print(f"  Loaded {len(data)} samples")
        all_data.extend(data)

    total_samples = len(all_data)
    print(f"\nTotal samples across all files: {total_samples}")

    # Analyze intents
    intent_dist = analyze_intents(all_data)

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(file_paths[0]) if file_paths else "."

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save intent distribution to JSON
    stats_path = os.path.join(output_dir, "intent_distribution.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": total_samples,
            "unique_intents": len(intent_dist),
            "intent_distribution": intent_dist
        }, f, indent=2, ensure_ascii=False)
    print(f"Intent distribution saved to: {stats_path}")

    # Create visualizations
    create_visualizations(intent_dist, total_samples, output_dir)

    # Print summary
    print_summary(intent_dist, total_samples)


if __name__ == "__main__":
    main()