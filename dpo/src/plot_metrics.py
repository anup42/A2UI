import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import sys

def plot_metrics_from_csv(csv_path):
    """
    Plot graphs for each metric in the training data CSV file.

    Args:
        csv_path (str): Path to the CSV file containing training metrics
    """
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found.")
        sys.exit(1)

    # Load data
    try:
        # Read CSV with error handling for rows with inconsistent column counts
        df = pd.read_csv(csv_path, error_bad_lines=False, warn_bad_lines=True)
        print(f"Loaded data from {csv_path}")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
    except TypeError:
        # For newer pandas versions where error_bad_lines is deprecated
        try:
            df = pd.read_csv(csv_path, on_bad_lines='skip')
            print(f"Loaded data from {csv_path}")
            print(f"Data shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    # Get list of columns (metrics)
    metrics = df.columns.tolist()
    print(f"Found {len(metrics)} metrics: {metrics}")

    # Remove 'step' and 'epoch' from metrics to plot (they are x-axes)
    x_metrics = ['step', 'epoch']
    y_metrics = [col for col in metrics if col not in x_metrics]

    # Create directory for plots
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Saving plots to '{plots_dir}' directory")

    # Plot each metric
    for metric in y_metrics:
        plt.figure(figsize=(12, 5))

        # Check if step column exists
        if 'step' in df.columns:
            x_values = df['step']
            x_label = 'Step'
        else:
            x_values = range(len(df))
            x_label = 'Index'

        # Plot against step/epoch
        plt.subplot(1, 2, 1)
        plt.plot(x_values, df[metric], marker='o', linestyle='-', markersize=4)
        plt.xlabel(x_label)
        plt.ylabel(metric)
        plt.title(f'{metric} vs {x_label}')
        plt.grid(True, alpha=0.3)

        # Plot against epoch if epoch data is available and different from step
        plt.subplot(1, 2, 2)
        if 'epoch' in df.columns and not df['epoch'].equals(x_values):
            plt.plot(df['epoch'], df[metric], marker='o', linestyle='-', markersize=4, color='orange')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.title(f'{metric} vs Epoch')
        else:
            # If epoch is not available or same as step, just plot against index
            plt.plot(range(len(df)), df[metric], marker='o', linestyle='-', markersize=4, color='orange')
            plt.xlabel('Index')
            plt.ylabel(metric)
            plt.title(f'{metric}')

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot with sanitized filename
        filename = metric.replace("/", "_").replace("\\", "_").replace(" ", "_")
        plt.savefig(f'{plots_dir}/{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved plot for {metric}")

    # Create combined plots for some key metrics
    key_metrics = ['loss', 'grad_norm', 'learning_rate', 'entropy', 'mean_token_accuracy']
    available_key_metrics = [m for m in key_metrics if m in df.columns]

    if available_key_metrics:
        n_metrics = len(available_key_metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        plt.figure(figsize=(5*n_cols, 4*n_rows))

        for i, metric in enumerate(available_key_metrics, 1):
            plt.subplot(n_rows, n_cols, i)

            if 'step' in df.columns:
                x_values = df['step']
                x_label = 'Step'
            else:
                x_values = range(len(df))
                x_label = 'Index'

            plt.plot(x_values, df[metric], marker='o', linestyle='-', markersize=3)
            plt.xlabel(x_label)
            plt.ylabel(metric)
            plt.title(f'{metric} vs {x_label}')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{plots_dir}/key_metrics_combined.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved combined plot for key metrics")

    print(f"All plots saved in '{plots_dir}' directory")

def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from CSV file')
    parser.add_argument('csv_path', help='Path to the CSV file containing training metrics')

    args = parser.parse_args()

    plot_metrics_from_csv(args.csv_path)

if __name__ == "__main__":
    main()
