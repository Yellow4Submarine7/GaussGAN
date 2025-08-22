#!/usr/bin/env python3
"""
Quick example of how to use the metrics recalculation system.

This script demonstrates the most common usage patterns for recalculating
metrics from saved CSV files.
"""

import sys
sys.path.append('..')

from recalculate_metrics import MetricsRecalculator
from pathlib import Path
import pandas as pd

def example_single_run():
    """Example: Process a single run with fast metrics."""
    
    print("🎯 Example 1: Processing single run with fast metrics")
    
    # Find the first available run
    mlruns_path = Path("../mlruns")
    
    # Look for a run with CSV files
    run_path = None
    for exp_dir in mlruns_path.iterdir():
        if not exp_dir.is_dir() or exp_dir.name == "0":
            continue
        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue
            artifacts_dir = run_dir / "artifacts"
            if artifacts_dir.exists() and list(artifacts_dir.glob("gaussian_generated_epoch_*.csv")):
                run_path = run_dir
                break
        if run_path:
            break
    
    if not run_path:
        print("❌ No runs with CSV files found in mlruns/")
        return
    
    print(f"📁 Using run: {run_path}")
    
    # Initialize recalculator
    recalculator = MetricsRecalculator("../data/normal.pickle")
    
    # Process with fast mode and selected metrics
    results_df = recalculator.process_experiment(
        str(run_path),
        output_file="example_metrics.csv",
        selected_metrics=["IsPositive", "LogLikelihood", "WassersteinDistance"],
        fast_mode=True
    )
    
    print(f"✅ Processed {len(results_df)} epochs")
    print("\n📊 Sample results:")
    print(results_df[['epoch', 'IsPositive', 'LogLikelihood', 'WassersteinDistance']].head())
    
    return results_df

def example_metric_comparison():
    """Example: Compare different metrics for the same data."""
    
    print("\n🎯 Example 2: Comparing all available metrics")
    
    # Find a run as before
    mlruns_path = Path("../mlruns")
    run_path = None
    for exp_dir in mlruns_path.iterdir():
        if not exp_dir.is_dir() or exp_dir.name == "0":
            continue
        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue
            artifacts_dir = run_dir / "artifacts"
            if artifacts_dir.exists() and list(artifacts_dir.glob("gaussian_generated_epoch_*.csv")):
                run_path = run_dir
                break
        if run_path:
            break
    
    if not run_path:
        print("❌ No runs found")
        return
    
    recalculator = MetricsRecalculator("../data/normal.pickle")
    
    # Calculate all metrics (except KL divergence for speed)
    all_metrics = ["IsPositive", "LogLikelihood", "WassersteinDistance", "MMDDistance", "MMDivergenceFromGMM"]
    
    results_df = recalculator.process_experiment(
        str(run_path),
        selected_metrics=all_metrics,
        fast_mode=True  # Use fast mode for demo
    )
    
    print(f"✅ Calculated {len(all_metrics)} metrics for {len(results_df)} epochs")
    
    # Show final epoch results
    final_epoch = results_df.iloc[-1]
    print(f"\n📈 Final epoch ({final_epoch['epoch']}) metrics:")
    for metric in all_metrics:
        value = final_epoch[metric]
        print(f"   - {metric}: {value:.4f}")
    
    # Show metric trends
    print(f"\n📊 Metric trends (first -> last epoch):")
    for metric in all_metrics:
        first_val = results_df[metric].iloc[0]
        last_val = results_df[metric].iloc[-1]
        change = last_val - first_val
        direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
        print(f"   {direction} {metric}: {first_val:.4f} -> {last_val:.4f} (Δ{change:+.4f})")
    
    return results_df

def example_batch_processing():
    """Example: Process multiple runs and generate summary."""
    
    print("\n🎯 Example 3: Batch processing multiple runs")
    
    recalculator = MetricsRecalculator("../data/normal.pickle")
    
    # Process all runs with fast mode
    all_results = recalculator.process_all_experiments(
        mlruns_dir="../mlruns",
        output_dir="batch_results"
    )
    
    if all_results:
        print(f"✅ Processed {len(all_results)} runs")
        
        # Generate summary
        summary_df = recalculator.generate_summary_report(
            all_results, 
            "batch_results/summary.csv"
        )
        
        if not summary_df.empty:
            print("\n📋 Summary statistics:")
            print(f"   - Total runs: {len(summary_df)}")
            print(f"   - Average epochs per run: {summary_df['total_epochs'].mean():.1f}")
            
            # Show best performing runs for different metrics
            for metric in ['LogLikelihood', 'WassersteinDistance']:
                best_col = f"{metric}_best"
                if best_col in summary_df.columns:
                    best_run = summary_df.loc[summary_df[best_col].idxmax()]
                    print(f"   - Best {metric}: {best_run[best_col]:.4f} (run: {best_run['run_id'][:8]}...)")
    
    return all_results

def main():
    """Run all examples."""
    
    print("🚀 Metrics Recalculation Examples")
    print("=" * 50)
    
    try:
        # Example 1: Single run
        results1 = example_single_run()
        
        # Example 2: Metric comparison  
        results2 = example_metric_comparison()
        
        # Example 3: Batch processing
        # Uncomment the line below if you want to process all runs
        # results3 = example_batch_processing()
        
        print("\n✅ All examples completed successfully!")
        print("\n💡 Tips:")
        print("   - Use --fast mode for quick analysis")
        print("   - Avoid KLDivergence for large datasets (very slow)")
        print("   - Save results to CSV for further analysis")
        print("   - Use selected metrics to focus on specific aspects")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()