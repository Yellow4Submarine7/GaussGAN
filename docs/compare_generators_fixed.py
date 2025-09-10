#!/usr/bin/env python
"""
Quantum vs Classical Generator Performance Comparison Script (FIXED VERSION)
For comparing performance metrics of different generator types

Fixed issues:
1. Visualization array indexing errors
2. Added MLflow error handling
3. Added metrics validation
4. Improved error handling and logging
"""

import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import warnings
import logging
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_experiment_runs(experiment_name: str, max_runs: int = 1000) -> pd.DataFrame:
    """Get all run records for an experiment
    
    Args:
        experiment_name: Experiment name
        max_runs: Maximum number of runs to return, to prevent memory issues
    
    Returns:
        DataFrame containing run data
    """
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            logger.warning(f"Experiment '{experiment_name}' does not exist")
            return pd.DataFrame()
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attribute.start_time desc"],
            max_results=max_runs
        )
        
        if not runs:
            logger.warning(f"No runs found in experiment '{experiment_name}'")
            return pd.DataFrame()
        
        # Extract run data
        data = []
        for run in runs:
            run_data = {
                'run_id': run.info.run_id,
                'generator_type': run.data.params.get('generator_type', 'unknown'),
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
                'duration_seconds': (run.info.end_time - run.info.start_time) / 1000 if run.info.end_time else None
            }
            
            # Add key metrics
            metrics_to_track = [
                'ValidationStep_FakeData_KLDivergence',
                'ValidationStep_FakeData_LogLikelihood', 
                'ValidationStep_FakeData_IsPositive',
                'ValidationStep_FakeData_WassersteinDistance',
                'ValidationStep_FakeData_MMDDistance',
                'train_loss_step',
                'd_loss',
                'g_loss'
            ]
            
            for metric in metrics_to_track:
                run_data[metric] = run.data.metrics.get(metric, None)
            
            data.append(run_data)
        
        df = pd.DataFrame(data)
        logger.info(f"Successfully retrieved {len(df)} run records")
        return df
        
    except Exception as e:
        logger.error(f"Failed to get experiment run records: {e}")
        return pd.DataFrame()

def validate_metrics(gen_runs: pd.DataFrame, gen_type: str) -> bool:
    """Validate that key metrics exist and are valid
    
    Args:
        gen_runs: Generator run data
        gen_type: Generator type
    
    Returns:
        Whether validation passes
    """
    required_metrics = [
        'ValidationStep_FakeData_KLDivergence',
        'duration_seconds'
    ]
    
    issues = []
    for metric in required_metrics:
        if metric not in gen_runs.columns:
            issues.append(f"Missing metric column {metric}")
        elif gen_runs[metric].isna().all():
            issues.append(f"Metric {metric} is all null values")
        elif gen_runs[metric].notna().sum() == 0:
            issues.append(f"Metric {metric} has no valid data")
    
    if issues:
        logger.warning(f"Generator {gen_type} data validation failed: {'; '.join(issues)}")
        return False
    
    return True

def analyze_convergence(client, run_id: str) -> Dict:
    """Analyze convergence characteristics of a single run
    
    Args:
        client: MLflow client
        run_id: Run ID
    
    Returns:
        Convergence analysis result dictionary
    """
    try:
        # Get historical metrics
        metric_history = client.get_metric_history(run_id, "ValidationStep_FakeData_KLDivergence")
        
        if not metric_history:
            logger.warning(f"Run {run_id} has no KL divergence history data")
            return {}
        
        epochs = [m.step for m in metric_history]
        values = [m.value for m in metric_history]
        
        if not values:
            return {}
        
        # Calculate convergence metrics
        convergence_info = {
            'final_value': values[-1],
            'best_value': min(values),
            'epochs_to_best': epochs[np.argmin(values)],
            'improvement_rate': (values[0] - values[-1]) / len(values) if len(values) > 1 else 0,
            'stability': np.std(values[-5:]) if len(values) >= 5 else None
        }
        
        return convergence_info
        
    except Exception as e:
        logger.error(f"Failed to analyze convergence for run {run_id}: {e}")
        return {}

def safe_calculate_percentage_diff(val1: float, val2: float) -> Optional[float]:
    """Safely calculate percentage difference
    
    Args:
        val1: Baseline value
        val2: Comparison value
    
    Returns:
        Percentage difference, returns None if calculation fails
    """
    try:
        if pd.isna(val1) or pd.isna(val2):
            return None
        if val1 == 0:
            return float('inf') if val2 != 0 else 0
        return ((val2 - val1) / val1 * 100)
    except Exception:
        return None

def compare_generators(experiment_name: str = "quantum_vs_classical_comparison", 
                      output_dir: str = ".") -> Optional[pd.DataFrame]:
    """Main comparison function
    
    Args:
        experiment_name: Experiment name
        output_dir: Output directory
    
    Returns:
        Comparison results DataFrame, returns None on failure
    """
    print("=" * 80)
    print("Quantum vs Classical Generator Performance Comparison Analysis")
    print("=" * 80)
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get run data
    df = get_experiment_runs(experiment_name)
    
    if df.empty:
        logger.error("No experiment data found, please run experiments first")
        return None
    
    # Group by generator type
    generator_types = df['generator_type'].unique()
    print(f"\nFound {len(generator_types)} generator types: {list(generator_types)}")
    print(f"Total {len(df)} runs\n")
    
    # Create comparison table
    comparison_results = []
    client = mlflow.tracking.MlflowClient()
    
    for gen_type in generator_types:
        gen_runs = df[df['generator_type'] == gen_type]
        
        if gen_runs.empty:
            continue
            
        try:
            # Validate data quality
            if not validate_metrics(gen_runs, gen_type):
                logger.warning(f"Skipping generator type {gen_type} due to data quality issues")
                continue
            
            # Calculate average metrics (using only valid data)
            result = {
                'Generator Type': gen_type,
                'Run Count': len(gen_runs),
                'Avg Training Time (s)': gen_runs['duration_seconds'].mean(),
                'KL Divergence (avg)': gen_runs['ValidationStep_FakeData_KLDivergence'].mean(),
                'KL Divergence (best)': gen_runs['ValidationStep_FakeData_KLDivergence'].min(),
                'Log Likelihood (avg)': gen_runs['ValidationStep_FakeData_LogLikelihood'].mean(),
                'Wasserstein Distance': gen_runs['ValidationStep_FakeData_WassersteinDistance'].mean(),
                'MMD Distance': gen_runs['ValidationStep_FakeData_MMDDistance'].mean(),
            }
            
            # Get convergence info for best run
            best_run_id = gen_runs.nsmallest(1, 'ValidationStep_FakeData_KLDivergence')['run_id'].values[0]
            convergence = analyze_convergence(client, best_run_id)
            
            if convergence:
                result.update({
                    'Convergence Epochs': convergence.get('epochs_to_best', 'N/A'),
                    'Improvement Rate': convergence.get('improvement_rate', 'N/A'),
                    'Final Stability': convergence.get('stability', 'N/A')
                })
            else:
                result.update({
                    'Convergence Epochs': 'N/A',
                    'Improvement Rate': 'N/A',
                    'Final Stability': 'N/A'
                })
            
            comparison_results.append(result)
            
        except Exception as e:
            logger.error(f"Generator comparison analysis failed: {e}")
            continue
    
    if not comparison_results:
        logger.error("No valid comparison results")
        return None
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_results)
    
    # Print detailed comparison
    print("-" * 80)
    print("Performance Comparison Results")
    print("-" * 80)
    
    # Training efficiency comparison
    print("\n📊 Training Efficiency Comparison:")
    print("-" * 40)
    for _, row in comparison_df.iterrows():
        duration = row['Avg Training Time (s)']
        if pd.notna(duration):
            print(f"{row['Generator Type']:20s}: {duration:.2f} seconds")
        else:
            print(f"{row['Generator Type']:20s}: N/A")
    
    # Generation quality comparison
    print("\n📈 Generation Quality Comparison (lower is better):")
    print("-" * 40)
    print(f"{'Metric':<20} {'Classical Generator':<15} {'Quantum Generator':<15} {'Difference':<15}")
    print("-" * 65)
    
    metrics_to_compare = ['KL Divergence (best)', 'Wasserstein Distance', 'MMD Distance']
    
    for metric in metrics_to_compare:
        classical_val = comparison_df[comparison_df['Generator Type'].str.contains('classical', na=False)][metric].values
        quantum_val = comparison_df[comparison_df['Generator Type'].str.contains('quantum', na=False)][metric].values
        
        if len(classical_val) > 0 and len(quantum_val) > 0:
            c_val = classical_val[0]
            q_val = quantum_val[0]
            diff = safe_calculate_percentage_diff(c_val, q_val)
            
            if pd.notna(c_val) and pd.notna(q_val):
                diff_str = f"{diff:.1f}%" if diff is not None else "N/A"
                print(f"{metric:<20} {c_val:<15.4f} {q_val:<15.4f} {diff_str:<15}")
    
    # Convergence speed comparison
    print("\n⚡ Convergence Speed Comparison:")
    print("-" * 40)
    for _, row in comparison_df.iterrows():
        print(f"{row['Generator Type']:20s}: {row['Convergence Epochs']} epochs")
    
    # Performance ratio calculation
    print("\n" + "=" * 80)
    print("Performance Ratio Analysis")
    print("=" * 80)
    
    classical_time = comparison_df[comparison_df['Generator Type'].str.contains('classical', na=False)]['Avg Training Time (s)'].values
    quantum_time = comparison_df[comparison_df['Generator Type'].str.contains('quantum', na=False)]['Avg Training Time (s)'].values
    
    if len(classical_time) > 0 and len(quantum_time) > 0:
        c_time = classical_time[0]
        q_time = quantum_time[0]
        if pd.notna(c_time) and pd.notna(q_time) and c_time > 0:
            time_ratio = q_time / c_time
            print(f"\n⏱️  Time Ratio: Quantum generator is {time_ratio:.1f}x slower than classical generator")
        else:
            print("\n⏱️  Time Ratio: Cannot calculate (incomplete data)")
    
    # Create visualization
    try:
        create_comparison_charts(comparison_df, output_path)
    except Exception as e:
        logger.error(f"Visualization creation failed: {e}")
    
    # Save results
    csv_path = output_path / "generator_comparison_results.csv"
    try:
        comparison_df.to_csv(csv_path, index=False)
        print(f"\n💾 Results saved to: {csv_path}")
    except Exception as e:
        logger.error(f"Failed to save CSV file: {e}")
    
    return comparison_df

def create_comparison_charts(comparison_df: pd.DataFrame, output_path: Path):
    """Create comparison visualization charts
    
    Args:
        comparison_df: Comparison data DataFrame
        output_path: Output path
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Training time comparison - Fixed array indexing
        ax = axes[0][0]  # Fix: Changed from axes[0, 0] to axes[0][0]
        valid_data = comparison_df.dropna(subset=['Avg Training Time (s)'])
        if not valid_data.empty:
            ax.bar(valid_data['Generator Type'], valid_data['Avg Training Time (s)'])
        ax.set_title('Training Time Comparison')
        ax.set_ylabel('Time (seconds)')
        ax.set_xlabel('Generator Type')
        ax.tick_params(axis='x', rotation=45)
        
        # KL divergence comparison - Fixed array indexing
        ax = axes[0][1]  # Fix: Changed from axes[0, 1] to axes[0][1]
        valid_data = comparison_df.dropna(subset=['KL Divergence (best)'])
        if not valid_data.empty:
            ax.bar(valid_data['Generator Type'], valid_data['KL Divergence (best)'])
        ax.set_title('KL Divergence Comparison (lower is better)')
        ax.set_ylabel('KL Divergence')
        ax.set_xlabel('Generator Type')
        ax.tick_params(axis='x', rotation=45)
        
        # Wasserstein distance comparison - Fixed array indexing
        ax = axes[1][0]  # Fix: Changed from axes[1, 0] to axes[1][0]
        valid_data = comparison_df.dropna(subset=['Wasserstein Distance'])
        if not valid_data.empty:
            ax.bar(valid_data['Generator Type'], valid_data['Wasserstein Distance'])
        ax.set_title('Wasserstein Distance Comparison (lower is better)')
        ax.set_ylabel('Wasserstein Distance')
        ax.set_xlabel('Generator Type')
        ax.tick_params(axis='x', rotation=45)
        
        # MMD distance comparison - Fixed array indexing
        ax = axes[1][1]  # Fix: Changed from axes[1, 1] to axes[1][1]
        valid_data = comparison_df.dropna(subset=['MMD Distance'])
        if not valid_data.empty:
            ax.bar(valid_data['Generator Type'], valid_data['MMD Distance'])
        ax.set_title('MMD Distance Comparison (lower is better)')
        ax.set_ylabel('MMD Distance')
        ax.set_xlabel('Generator Type')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save image
        png_path = output_path / "generator_comparison_charts.png"
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"📊 Visualization charts saved to: {png_path}")
        
        # Clean up matplotlib resources
        plt.close()
        
    except Exception as e:
        logger.error(f"Failed to create visualization charts: {e}")


if __name__ == "__main__":
    # Run comparison analysis
    try:
        results = compare_generators()
        
        if results is not None and not results.empty:
            print("\n" + "=" * 80)
            print("🎯 Key Findings:")
            print("=" * 80)
            
            # Calculate key metrics
            classical_rows = results[results['Generator Type'].str.contains('classical', na=False)]
            quantum_rows = results[results['Generator Type'].str.contains('quantum', na=False)]
            
            if not classical_rows.empty and not quantum_rows.empty:
                c_kl = classical_rows['KL Divergence (best)'].values[0]
                q_kl = quantum_rows['KL Divergence (best)'].values[0]
                c_time = classical_rows['Avg Training Time (s)'].values[0]
                q_time = quantum_rows['Avg Training Time (s)'].values[0]
                
                if pd.notna(c_time) and pd.notna(q_time) and c_time > 0:
                    print(f"\n1. Quantum generator training time is {q_time/c_time:.1f}x that of classical generator")
                
                if pd.notna(c_kl) and pd.notna(q_kl):
                    if q_kl < c_kl:
                        print(f"2. Quantum generator's KL divergence is {(c_kl-q_kl)/c_kl*100:.1f}% lower than classical (better)")
                    else:
                        print(f"2. Quantum generator's KL divergence is {(q_kl-c_kl)/c_kl*100:.1f}% higher than classical (worse)")
                
                print("\nThese quantitative results directly answer Ale's questions:")
                print("✅ We can now precisely measure the performance difference between classical and quantum generators")
                print("✅ Not only do we have visual comparisons, but also specific numerical metrics")
            else:
                print("Insufficient classical and quantum generator data found for comparison")
        else:
            print("Comparison analysis failed or no valid data")
            
    except Exception as e:
        logger.error(f"Program execution failed: {e}")
        print("An error occurred during program execution, please check the logs")