#!/usr/bin/env python3
"""
Visualization script for recalculated metrics.

Usage:
    uv run python visualize_metrics.py metrics_file.csv
    uv run python visualize_metrics.py metrics_file.csv --save plots/
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def plot_metrics_trends(df, save_dir=None, run_id=None):
    """
    Create comprehensive plots for metrics trends over epochs.
    
    Args:
        df: DataFrame with metrics data
        save_dir: Directory to save plots (optional)
        run_id: Run ID for plot titles (optional)
    """
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Get available metric columns (exclude metadata columns)
    metadata_cols = ['epoch', 'n_samples', 'samples_file']
    metric_cols = [col for col in df.columns if col not in metadata_cols]
    
    if not metric_cols:
        print("❌ No metric columns found in the data")
        return
    
    # Create figure with subplots
    n_metrics = len(metric_cols)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # Ensure axes is always a flat array for consistent indexing
    if n_metrics == 1:
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()
    
    # Remove any extra subplots
    for i in range(n_metrics, n_rows * n_cols):
        fig.delaxes(axes[i])
    
    # Plot each metric
    for idx, metric in enumerate(metric_cols):
        ax = axes[idx]
        
        # Filter out NaN values
        valid_data = df[['epoch', metric]].dropna()
        
        if len(valid_data) == 0:
            ax.text(0.5, 0.5, f'No valid data\nfor {metric}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(metric)
            continue
        
        # Plot the trend
        ax.plot(valid_data['epoch'], valid_data[metric], 'o-', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} over Epochs')
        ax.grid(True, alpha=0.3)
        
        # Add trend information
        if len(valid_data) > 1:
            first_val = valid_data[metric].iloc[0]
            last_val = valid_data[metric].iloc[-1]
            change = last_val - first_val
            direction = "↗️" if change > 0 else "↘️" if change < 0 else "→"
            
            # Add text box with trend info
            textstr = f'{direction} {first_val:.3f} → {last_val:.3f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=props)
    
    # Main title
    main_title = f'Metrics Trends - {run_id}' if run_id else 'Metrics Trends'
    fig.suptitle(main_title, fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save or show
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        filename = f'metrics_trends_{run_id}.png' if run_id else 'metrics_trends.png'
        plt.savefig(save_path / filename, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to {save_path / filename}")
    else:
        plt.show()

def plot_metrics_correlation(df, save_dir=None, run_id=None):
    """
    Create correlation matrix for metrics.
    
    Args:
        df: DataFrame with metrics data
        save_dir: Directory to save plots (optional)
        run_id: Run ID for plot titles (optional)
    """
    # Get metric columns only
    metadata_cols = ['epoch', 'n_samples', 'samples_file']
    metric_cols = [col for col in df.columns if col not in metadata_cols]
    
    if len(metric_cols) < 2:
        print("⚠️  Need at least 2 metrics for correlation analysis")
        return
    
    # Calculate correlation matrix
    corr_matrix = df[metric_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    
    title = f'Metrics Correlation Matrix - {run_id}' if run_id else 'Metrics Correlation Matrix'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        filename = f'metrics_correlation_{run_id}.png' if run_id else 'metrics_correlation.png'
        plt.savefig(save_path / filename, dpi=300, bbox_inches='tight')
        print(f"📊 Correlation plot saved to {save_path / filename}")
    else:
        plt.show()

def plot_metrics_distribution(df, save_dir=None, run_id=None):
    """
    Create distribution plots for final epoch metrics.
    
    Args:
        df: DataFrame with metrics data
        save_dir: Directory to save plots (optional) 
        run_id: Run ID for plot titles (optional)
    """
    # Get final epoch data
    final_epoch = df.iloc[-1]
    
    # Get metric columns
    metadata_cols = ['epoch', 'n_samples', 'samples_file']
    metric_cols = [col for col in df.columns if col not in metadata_cols]
    
    # Create bar plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    metric_values = []
    metric_names = []
    
    for metric in metric_cols:
        value = final_epoch[metric]
        if not pd.isna(value):
            metric_values.append(value)
            metric_names.append(metric)
    
    if not metric_values:
        ax.text(0.5, 0.5, 'No valid final epoch data', ha='center', va='center', 
               transform=ax.transAxes)
        return
    
    # Create bar plot
    bars = ax.bar(metric_names, metric_values)
    
    # Color bars based on values (assuming lower is better for most metrics)
    colors = ['green' if 'IsPositive' in name else 'red' if val > 0 else 'blue' 
              for name, val in zip(metric_names, metric_values)]
    
    for bar, color in zip(bars, colors):
        bar.set_color(color)
        bar.set_alpha(0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
               f'{value:.4f}', ha='center', va='bottom')
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Value')
    title = f'Final Epoch Metrics - {run_id}' if run_id else 'Final Epoch Metrics'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        filename = f'final_metrics_{run_id}.png' if run_id else 'final_metrics.png'
        plt.savefig(save_path / filename, dpi=300, bbox_inches='tight')
        print(f"📊 Final metrics plot saved to {save_path / filename}")
    else:
        plt.show()

def generate_summary_stats(df):
    """Generate summary statistics for metrics."""
    
    print("\n📈 Metrics Summary Statistics")
    print("=" * 50)
    
    # Get metric columns
    metadata_cols = ['epoch', 'n_samples', 'samples_file']
    metric_cols = [col for col in df.columns if col not in metadata_cols]
    
    for metric in metric_cols:
        valid_data = df[metric].dropna()
        
        if len(valid_data) == 0:
            print(f"\n{metric}: No valid data")
            continue
        
        print(f"\n{metric}:")
        print(f"  📊 Count: {len(valid_data)}")
        print(f"  📈 Mean: {valid_data.mean():.6f}")
        print(f"  📐 Std: {valid_data.std():.6f}")
        print(f"  📉 Min: {valid_data.min():.6f}")
        print(f"  📊 Median: {valid_data.median():.6f}")
        print(f"  📈 Max: {valid_data.max():.6f}")
        
        if len(valid_data) > 1:
            first_val = valid_data.iloc[0]
            last_val = valid_data.iloc[-1]
            change = last_val - first_val
            pct_change = (change / abs(first_val)) * 100 if first_val != 0 else float('inf')
            direction = "↗️ Improved" if change > 0 else "↘️ Degraded" if change < 0 else "→ Stable"
            print(f"  🔄 Trend: {direction} ({change:+.6f}, {pct_change:+.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Visualize recalculated metrics")
    parser.add_argument("csv_file", help="Path to metrics CSV file")
    parser.add_argument("--save", "-s", help="Directory to save plots")
    parser.add_argument("--no-show", action="store_true", help="Don't show plots interactively")
    parser.add_argument("--correlation", "-c", action="store_true", help="Generate correlation plot")
    parser.add_argument("--distribution", "-d", action="store_true", help="Generate distribution plot")
    parser.add_argument("--all", "-a", action="store_true", help="Generate all plots")
    
    args = parser.parse_args()
    
    # Load data
    try:
        df = pd.read_csv(args.csv_file)
        print(f"✅ Loaded {len(df)} epochs from {args.csv_file}")
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return
    
    # Extract run ID from filename
    run_id = Path(args.csv_file).stem.replace("metrics_", "")
    
    # Generate summary statistics
    generate_summary_stats(df)
    
    # Determine what plots to generate
    generate_trends = True  # Always generate trends
    generate_correlation = args.correlation or args.all
    generate_distribution = args.distribution or args.all
    
    # Generate plots
    save_dir = args.save if not args.no_show else args.save
    
    if generate_trends:
        plot_metrics_trends(df, save_dir, run_id)
    
    if generate_correlation:
        plot_metrics_correlation(df, save_dir, run_id)
    
    if generate_distribution:
        plot_metrics_distribution(df, save_dir, run_id)
    
    if not args.no_show and not args.save:
        plt.show()
    
    print(f"\n✅ Visualization completed for run {run_id}")

if __name__ == "__main__":
    main()