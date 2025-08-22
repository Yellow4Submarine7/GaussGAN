"""
Statistical Analysis Demonstration Script
========================================

This script demonstrates the complete statistical analysis workflow for comparing
quantum vs classical generators in the GaussGAN project. It shows how to:

1. Load or simulate experimental data
2. Perform comprehensive statistical analysis
3. Generate detailed reports and visualizations
4. Interpret results for scientific publication

This serves as both a working example and a template for actual analysis.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
from datetime import datetime, timedelta
import logging

# Add the docs directory to the path to import our modules
sys.path.append(str(Path(__file__).parent))

from statistical_analysis_framework import (
    ExperimentRun, StatisticalAnalyzer, ConvergenceAnalyzer, 
    StabilityAnalyzer, ResultsAggregator, MultiRunExperimentManager
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def simulate_realistic_experimental_data(n_runs_per_generator: int = 20) -> list:
    """
    Simulate realistic experimental data based on expected GaussGAN performance.
    
    This function creates synthetic but realistic experimental results that mirror
    the expected behavior of different generator types in the GaussGAN project.
    
    Args:
        n_runs_per_generator: Number of experimental runs per generator type
        
    Returns:
        List of ExperimentRun objects with simulated data
    """
    logger.info(f"Simulating experimental data with {n_runs_per_generator} runs per generator")
    
    generator_types = ['classical_normal', 'classical_uniform', 'quantum_samples', 'quantum_shadows']
    
    # Define realistic performance characteristics for each generator
    performance_profiles = {
        'classical_normal': {
            'KLDivergence': {'mean': 0.12, 'std': 0.03, 'convergence_rate': 0.95},
            'LogLikelihood': {'mean': -2.3, 'std': 0.2, 'convergence_rate': 0.95},
            'WassersteinDistance': {'mean': 0.08, 'std': 0.02, 'convergence_rate': 0.90},
            'MMDDistance': {'mean': 0.06, 'std': 0.015, 'convergence_rate': 0.85},
            'training_time': {'mean': 120, 'std': 20}  # seconds
        },
        'classical_uniform': {
            'KLDivergence': {'mean': 0.15, 'std': 0.04, 'convergence_rate': 0.88},
            'LogLikelihood': {'mean': -2.6, 'std': 0.25, 'convergence_rate': 0.88},
            'WassersteinDistance': {'mean': 0.10, 'std': 0.025, 'convergence_rate': 0.85},
            'MMDDistance': {'mean': 0.08, 'std': 0.02, 'convergence_rate': 0.80},
            'training_time': {'mean': 110, 'std': 18}
        },
        'quantum_samples': {
            'KLDivergence': {'mean': 0.22, 'std': 0.06, 'convergence_rate': 0.75},
            'LogLikelihood': {'mean': -3.1, 'std': 0.4, 'convergence_rate': 0.75},
            'WassersteinDistance': {'mean': 0.14, 'std': 0.04, 'convergence_rate': 0.70},
            'MMDDistance': {'mean': 0.12, 'std': 0.035, 'convergence_rate': 0.65},
            'training_time': {'mean': 450, 'std': 80}  # Much slower due to quantum simulation
        },
        'quantum_shadows': {
            'KLDivergence': {'mean': 0.18, 'std': 0.05, 'convergence_rate': 0.80},
            'LogLikelihood': {'mean': -2.9, 'std': 0.35, 'convergence_rate': 0.80},
            'WassersteinDistance': {'mean': 0.12, 'std': 0.035, 'convergence_rate': 0.75},
            'MMDDistance': {'mean': 0.10, 'std': 0.03, 'convergence_rate': 0.70},
            'training_time': {'mean': 380, 'std': 70}  # Faster than quantum_samples due to shadows
        }
    }
    
    runs = []
    base_time = datetime.now() - timedelta(days=7)  # Simulate experiments from a week ago
    
    for gen_type in generator_types:
        profile = performance_profiles[gen_type]
        
        for run_idx in range(n_runs_per_generator):
            # Generate run metadata
            seed = 42 + len(runs)
            run_id = f"{gen_type}_run_{run_idx:03d}_seed_{seed}"
            
            # Simulate convergence
            converged = np.random.random() < profile['KLDivergence']['convergence_rate']
            convergence_epoch = np.random.randint(15, 45) if converged else None
            
            # Simulate training duration
            training_duration = np.random.normal(
                profile['training_time']['mean'],
                profile['training_time']['std']
            )
            training_duration = max(30, training_duration)  # Minimum 30 seconds
            
            # Generate final metrics
            final_metrics = {}
            for metric_name, metric_params in profile.items():
                if metric_name == 'training_time':
                    continue
                    
                if converged:
                    # Better performance if converged
                    value = np.random.normal(metric_params['mean'], metric_params['std'] * 0.7)
                else:
                    # Worse performance if not converged
                    if metric_name in ['KLDivergence', 'WassersteinDistance', 'MMDDistance']:
                        value = np.random.normal(metric_params['mean'] * 1.5, metric_params['std'] * 1.3)
                    else:  # LogLikelihood
                        value = np.random.normal(metric_params['mean'] * 1.2, metric_params['std'] * 1.3)
                
                # Add some realistic bounds
                if metric_name in ['KLDivergence', 'WassersteinDistance', 'MMDDistance']:
                    value = max(0.01, value)  # Cannot be negative or zero
                elif metric_name == 'LogLikelihood':
                    value = min(-1.0, value)  # Cannot be too positive
                
                final_metrics[metric_name] = value
            
            # Add IsPositive metric (proportion of points with x > 0)
            final_metrics['IsPositive'] = np.random.beta(8, 3)  # Typically high, around 0.7-0.9
            
            # Generate metric history (convergence over epochs)
            max_epochs = convergence_epoch + 10 if converged else 50
            metric_history = {}
            
            for metric_name in ['KLDivergence', 'LogLikelihood', 'WassersteinDistance', 'MMDDistance']:
                if metric_name in ['KLDivergence', 'WassersteinDistance', 'MMDDistance']:
                    # These should decrease over time
                    start_value = final_metrics[metric_name] * 3
                    end_value = final_metrics[metric_name]
                    trend = -1
                else:
                    # LogLikelihood should increase (become less negative)
                    start_value = final_metrics[metric_name] * 1.5
                    end_value = final_metrics[metric_name]
                    trend = 1
                
                history = []
                for epoch in range(max_epochs):
                    # Exponential convergence with noise
                    progress = 1 - np.exp(-epoch / 15)
                    value = start_value + (end_value - start_value) * progress
                    value += np.random.normal(0, abs(end_value) * 0.05)  # Add noise
                    history.append(value)
                
                metric_history[metric_name] = history
            
            # Generate loss history
            loss_history = {
                'g_loss': [np.random.normal(-2.0 + epoch * 0.01, 0.3) for epoch in range(max_epochs)],
                'd_loss': [np.random.normal(0.0, 0.2) for epoch in range(max_epochs)]
            }
            
            # Create experiment run
            start_time = base_time + timedelta(hours=len(runs))
            end_time = start_time + timedelta(seconds=training_duration)
            
            run = ExperimentRun(
                generator_type=gen_type,
                seed=seed,
                run_id=run_id,
                start_time=start_time,
                end_time=end_time,
                convergence_epoch=convergence_epoch,
                final_metrics=final_metrics,
                metric_history=metric_history,
                loss_history=loss_history,
                hyperparameters={
                    'max_epochs': max_epochs,
                    'batch_size': 256,
                    'learning_rate': 0.001,
                    'generator_type': gen_type
                },
                training_duration=training_duration,
                converged=converged
            )
            
            runs.append(run)
    
    logger.info(f"Generated {len(runs)} experimental runs")
    return runs


def perform_comprehensive_analysis(runs: list) -> dict:
    """
    Perform comprehensive statistical analysis on experimental data.
    
    Args:
        runs: List of ExperimentRun objects
        
    Returns:
        Dictionary containing all analysis results
    """
    logger.info("Starting comprehensive statistical analysis")
    
    # Initialize analyzers
    statistical_analyzer = StatisticalAnalyzer(alpha=0.05, correction_method='bonferroni')
    convergence_analyzer = ConvergenceAnalyzer(
        convergence_thresholds={
            'KLDivergence': 0.25,
            'LogLikelihood': -2.8,
            'WassersteinDistance': 0.12,
            'MMDDistance': 0.10
        }
    )
    stability_analyzer = StabilityAnalyzer(outlier_method='iqr')
    results_aggregator = ResultsAggregator(output_dir="docs/statistical_analysis")
    
    # Generate comprehensive report
    report = results_aggregator.generate_comprehensive_report(
        runs=runs,
        statistical_analyzer=statistical_analyzer,
        convergence_analyzer=convergence_analyzer,
        stability_analyzer=stability_analyzer
    )
    
    return report


def create_detailed_visualizations(runs: list, report: dict):
    """
    Create detailed visualizations for the analysis results.
    
    Args:
        runs: List of ExperimentRun objects
        report: Analysis report dictionary
    """
    logger.info("Creating detailed visualizations")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    output_dir = Path("docs/statistical_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Performance Distribution Plots
    create_performance_distributions(runs, output_dir)
    
    # 2. Convergence Analysis Plots
    create_convergence_analysis_plots(runs, report, output_dir)
    
    # 3. Statistical Significance Matrix
    create_significance_matrix(report, output_dir)
    
    # 4. Stability Analysis
    create_stability_analysis_plots(runs, report, output_dir)
    
    # 5. Training Time Analysis
    create_training_time_analysis(runs, output_dir)
    
    logger.info(f"Visualizations saved to {output_dir}")


def create_performance_distributions(runs: list, output_dir: Path):
    """Create box plots showing performance distributions."""
    metrics = ['KLDivergence', 'LogLikelihood', 'WassersteinDistance', 'MMDDistance']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Prepare data
        data = []
        labels = []
        generator_types = ['classical_normal', 'classical_uniform', 'quantum_samples', 'quantum_shadows']
        
        for gen_type in generator_types:
            values = []
            for run in runs:
                if (run.generator_type == gen_type and 
                    run.final_metrics and 
                    metric in run.final_metrics):
                    values.append(run.final_metrics[metric])
            
            if values:
                data.append(values)
                labels.append(gen_type.replace('_', '\n'))
        
        # Create box plot
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title(f'{metric} Distribution by Generator Type', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        
        # Add statistical annotations
        classical_values = data[0] + data[1]  # Combine classical generators
        quantum_values = data[2] + data[3]    # Combine quantum generators
        
        if classical_values and quantum_values:
            from scipy.stats import mannwhitneyu
            statistic, p_value = mannwhitneyu(classical_values, quantum_values)
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            ax.text(0.02, 0.98, f'Classical vs Quantum: {significance}\n(p={p_value:.3f})', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_convergence_analysis_plots(runs: list, report: dict, output_dir: Path):
    """Create convergence analysis plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Convergence Rates
    ax = axes[0, 0]
    conv_data = report['convergence_analysis']
    
    generator_types = ['classical_normal', 'classical_uniform', 'quantum_samples', 'quantum_shadows']
    metrics = ['KLDivergence', 'LogLikelihood']
    
    x = np.arange(len(generator_types))
    width = 0.35
    
    for i, metric in enumerate(metrics):
        if metric in conv_data:
            rates = [conv_data[metric]['convergence_rates'].get(gen, 0) for gen in generator_types]
            ax.bar(x + i * width, rates, width, label=metric, alpha=0.8)
    
    ax.set_xlabel('Generator Type')
    ax.set_ylabel('Convergence Rate')
    ax.set_title('Convergence Rates by Generator Type')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([gen.replace('_', '\n') for gen in generator_types])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Mean Epochs to Convergence
    ax = axes[0, 1]
    for i, metric in enumerate(metrics):
        if metric in conv_data:
            epochs = [conv_data[metric]['mean_epochs_to_threshold'].get(gen, np.nan) for gen in generator_types]
            epochs = [e if not np.isnan(e) else 0 for e in epochs]
            ax.bar(x + i * width, epochs, width, label=metric, alpha=0.8)
    
    ax.set_xlabel('Generator Type')
    ax.set_ylabel('Mean Epochs to Convergence')
    ax.set_title('Speed of Convergence')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([gen.replace('_', '\n') for gen in generator_types])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Example Convergence Curves
    ax = axes[1, 0]
    colors = ['blue', 'green', 'red', 'orange']
    
    for i, gen_type in enumerate(generator_types):
        # Find a representative run for each generator
        gen_runs = [run for run in runs if run.generator_type == gen_type and run.converged]
        if gen_runs:
            run = gen_runs[0]  # Take first converged run
            if 'KLDivergence' in run.metric_history:
                history = run.metric_history['KLDivergence']
                epochs = range(len(history))
                ax.plot(epochs, history, color=colors[i], label=gen_type.replace('_', ' '), 
                       linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('KL Divergence')
    ax.set_title('Example Convergence Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Training Duration Distribution
    ax = axes[1, 1]
    durations_by_type = {}
    for gen_type in generator_types:
        durations = [run.training_duration for run in runs 
                    if run.generator_type == gen_type and run.training_duration]
        durations_by_type[gen_type] = durations
    
    ax.boxplot([durations_by_type[gen] for gen in generator_types],
               labels=[gen.replace('_', '\n') for gen in generator_types])
    ax.set_ylabel('Training Duration (seconds)')
    ax.set_title('Training Duration by Generator Type')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_significance_matrix(report: dict, output_dir: Path):
    """Create statistical significance heatmap."""
    generator_types = ['classical_normal', 'classical_uniform', 'quantum_samples', 'quantum_shadows']
    metrics = ['KLDivergence', 'LogLikelihood', 'WassersteinDistance', 'MMDDistance']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        
        # Create p-value matrix
        n_gen = len(generator_types)
        p_matrix = np.ones((n_gen, n_gen))
        
        if metric in report['statistical_tests']:
            comparisons = report['statistical_tests'][metric]['pairwise_comparisons']
            
            for i, gen1 in enumerate(generator_types):
                for j, gen2 in enumerate(generator_types):
                    if gen1 in comparisons and gen2 in comparisons[gen1]:
                        p_value = comparisons[gen1][gen2]['p_value']
                        p_matrix[i, j] = p_value
                        p_matrix[j, i] = p_value
        
        # Create heatmap
        mask = np.triu(np.ones_like(p_matrix, dtype=bool))  # Show only lower triangle
        
        im = ax.imshow(p_matrix, cmap='viridis_r', vmin=0, vmax=0.05)
        
        # Add text annotations
        for i in range(n_gen):
            for j in range(n_gen):
                if not mask[i, j]:
                    p_val = p_matrix[i, j]
                    if p_val < 0.001:
                        text = '***'
                    elif p_val < 0.01:
                        text = '**'
                    elif p_val < 0.05:
                        text = '*'
                    else:
                        text = f'{p_val:.3f}'
                    
                    ax.text(j, i, text, ha="center", va="center", 
                           color="white" if p_val < 0.025 else "black")
        
        ax.set_xticks(range(n_gen))
        ax.set_yticks(range(n_gen))
        ax.set_xticklabels([gen.replace('_', '\n') for gen in generator_types])
        ax.set_yticklabels([gen.replace('_', '\n') for gen in generator_types])
        ax.set_title(f'{metric} - Statistical Significance (p-values)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('p-value')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'significance_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_stability_analysis_plots(runs: list, report: dict, output_dir: Path):
    """Create stability analysis plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    generator_types = ['classical_normal', 'classical_uniform', 'quantum_samples', 'quantum_shadows']
    
    # Plot 1: Coefficient of Variation
    ax = axes[0, 0]
    metrics = ['KLDivergence', 'LogLikelihood', 'WassersteinDistance', 'MMDDistance']
    
    cv_data = []
    for gen_type in generator_types:
        cv_values = []
        for metric in metrics:
            if (metric in report['stability_analysis'] and 
                gen_type in report['stability_analysis'][metric]['final_value_stability']):
                cv = report['stability_analysis'][metric]['final_value_stability'][gen_type].get('coefficient_of_variation', np.nan)
                if not np.isnan(cv):
                    cv_values.append(cv)
        cv_data.append(cv_values)
    
    ax.boxplot(cv_data, labels=[gen.replace('_', '\n') for gen in generator_types])
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('Performance Stability (Lower is Better)')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Outlier Rates
    ax = axes[0, 1]
    outlier_rates = []
    for gen_type in generator_types:
        rates = []
        for metric in metrics:
            if (metric in report['stability_analysis'] and 
                gen_type in report['stability_analysis'][metric]['final_value_stability']):
                rate = report['stability_analysis'][metric]['final_value_stability'][gen_type].get('outlier_rate', 0)
                rates.append(rate)
        outlier_rates.append(np.mean(rates) if rates else 0)
    
    bars = ax.bar(range(len(generator_types)), outlier_rates, 
                  color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'])
    ax.set_xlabel('Generator Type')
    ax.set_ylabel('Average Outlier Rate')
    ax.set_title('Outlier Rates by Generator Type')
    ax.set_xticks(range(len(generator_types)))
    ax.set_xticklabels([gen.replace('_', '\n') for gen in generator_types])
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, rate in zip(bars, outlier_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
               f'{rate:.3f}', ha='center', va='bottom')
    
    # Plot 3: Range Ratios
    ax = axes[1, 0]
    range_ratios = []
    for gen_type in generator_types:
        ratios = []
        for metric in metrics:
            if (metric in report['stability_analysis'] and 
                gen_type in report['stability_analysis'][metric]['final_value_stability']):
                ratio = report['stability_analysis'][metric]['final_value_stability'][gen_type].get('range_ratio', np.nan)
                if not np.isnan(ratio):
                    ratios.append(ratio)
        range_ratios.append(ratios)
    
    ax.boxplot(range_ratios, labels=[gen.replace('_', '\n') for gen in generator_types])
    ax.set_ylabel('Range Ratio (Range/Median)')
    ax.set_title('Performance Range Analysis')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Success Rates
    ax = axes[1, 1]
    success_rates = []
    for gen_type in generator_types:
        total_runs = len([run for run in runs if run.generator_type == gen_type])
        successful_runs = len([run for run in runs if run.generator_type == gen_type and run.converged])
        success_rate = successful_runs / total_runs if total_runs > 0 else 0
        success_rates.append(success_rate)
    
    bars = ax.bar(range(len(generator_types)), success_rates,
                  color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'])
    ax.set_xlabel('Generator Type')
    ax.set_ylabel('Success Rate')
    ax.set_title('Training Success Rates')
    ax.set_xticks(range(len(generator_types)))
    ax.set_xticklabels([gen.replace('_', '\n') for gen in generator_types])
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{rate:.2%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'stability_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_training_time_analysis(runs: list, output_dir: Path):
    """Create training time analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    generator_types = ['classical_normal', 'classical_uniform', 'quantum_samples', 'quantum_shadows']
    
    # Plot 1: Training Duration Distribution
    ax = axes[0]
    durations_by_type = []
    for gen_type in generator_types:
        durations = [run.training_duration for run in runs 
                    if run.generator_type == gen_type and run.training_duration]
        durations_by_type.append([d/60 for d in durations])  # Convert to minutes
    
    bp = ax.boxplot(durations_by_type, labels=[gen.replace('_', '\n') for gen in generator_types],
                    patch_artist=True)
    
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Training Duration (minutes)')
    ax.set_title('Training Duration Distribution')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Performance vs Training Time
    ax = axes[1]
    colors_map = {'classical_normal': 'blue', 'classical_uniform': 'green', 
                  'quantum_samples': 'red', 'quantum_shadows': 'orange'}
    
    for gen_type in generator_types:
        durations = []
        kl_divergences = []
        
        for run in runs:
            if (run.generator_type == gen_type and 
                run.training_duration and 
                run.final_metrics and 
                'KLDivergence' in run.final_metrics):
                durations.append(run.training_duration / 60)  # minutes
                kl_divergences.append(run.final_metrics['KLDivergence'])
        
        if durations and kl_divergences:
            ax.scatter(durations, kl_divergences, 
                      label=gen_type.replace('_', ' '), 
                      color=colors_map[gen_type], alpha=0.7, s=50)
    
    ax.set_xlabel('Training Duration (minutes)')
    ax.set_ylabel('Final KL Divergence')
    ax.set_title('Performance vs Training Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_time_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_summary_report(report: dict, output_dir: Path):
    """Generate a comprehensive summary report."""
    logger.info("Generating summary report")
    
    summary_lines = []
    summary_lines.append("# GaussGAN Statistical Analysis Report")
    summary_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("")
    
    # Metadata
    metadata = report['metadata']
    summary_lines.append("## Experiment Overview")
    summary_lines.append(f"- Total experimental runs: {metadata['total_runs']}")
    summary_lines.append(f"- Generator types tested: {', '.join(metadata['generator_types'])}")
    summary_lines.append(f"- Metrics analyzed: {', '.join(metadata['metrics_analyzed'])}")
    summary_lines.append("")
    
    # Best performing generators
    insights = report['summary_insights']
    summary_lines.append("## Key Findings")
    summary_lines.append("")
    summary_lines.append("### Best Performing Generators")
    for metric, best_info in insights['best_performing_generators'].items():
        summary_lines.append(f"- **{metric}**: {best_info['generator']} "
                            f"(value: {best_info['value']:.4f})")
    summary_lines.append("")
    
    summary_lines.append("### Most Stable Generators")
    for metric, stable_info in insights['most_stable_generators'].items():
        summary_lines.append(f"- **{metric}**: {stable_info['generator']} "
                            f"(stability score: {stable_info['stability_score']:.4f})")
    summary_lines.append("")
    
    summary_lines.append("### Fastest Converging Generators")
    for metric, fast_info in insights['fastest_converging_generators'].items():
        summary_lines.append(f"- **{metric}**: {fast_info['generator']} "
                            f"({fast_info['epochs']:.1f} epochs)")
    summary_lines.append("")
    
    # Statistical significance
    summary_lines.append("### Significant Differences Found")
    significant_diffs = insights['significant_differences']
    if significant_diffs:
        for diff in significant_diffs:
            effect_size_text = f" (effect size: {diff['effect_size']:.3f})" if diff['effect_size'] else ""
            summary_lines.append(f"- **{diff['metric']}**: {diff['generator1']} vs {diff['generator2']} "
                                f"(p={diff['p_value']:.4f}){effect_size_text}")
    else:
        summary_lines.append("- No statistically significant differences found")
    summary_lines.append("")
    
    # Recommendations
    summary_lines.append("## Recommendations")
    summary_lines.append("")
    
    # Classical vs Quantum comparison
    classical_gens = ['classical_normal', 'classical_uniform']
    quantum_gens = ['quantum_samples', 'quantum_shadows']
    
    # Find overall best performers
    kl_best = insights['best_performing_generators'].get('KLDivergence', {}).get('generator', '')
    stability_best = insights['most_stable_generators'].get('KLDivergence', {}).get('generator', '')
    speed_best = insights['fastest_converging_generators'].get('KLDivergence', {}).get('generator', '')
    
    if kl_best in classical_gens:
        summary_lines.append("1. **Classical generators show superior performance** in terms of final KL divergence")
    elif kl_best in quantum_gens:
        summary_lines.append("1. **Quantum generators show competitive performance** despite higher computational cost")
    
    if stability_best in classical_gens:
        summary_lines.append("2. **Classical generators are more stable** across multiple runs")
    elif stability_best in quantum_gens:
        summary_lines.append("2. **Quantum generators demonstrate surprising stability** given their stochastic nature")
    
    if speed_best in classical_gens:
        summary_lines.append("3. **Classical generators converge faster** requiring fewer training epochs")
    elif speed_best in quantum_gens:
        summary_lines.append("3. **Quantum generators show competitive convergence speed** despite complexity")
    
    summary_lines.append("")
    summary_lines.append("## Files Generated")
    summary_lines.append("- `comprehensive_analysis_report.json`: Detailed analysis results")
    summary_lines.append("- `performance_distributions.png`: Performance comparison plots")
    summary_lines.append("- `convergence_analysis.png`: Convergence analysis visualizations")
    summary_lines.append("- `significance_matrix.png`: Statistical significance heatmaps")
    summary_lines.append("- `stability_analysis.png`: Stability analysis plots")
    summary_lines.append("- `training_time_analysis.png`: Training duration analysis")
    
    # Save report
    report_path = output_dir / 'analysis_summary.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    logger.info(f"Summary report saved to {report_path}")


def main():
    """Main demonstration function."""
    print("=" * 60)
    print("GaussGAN Statistical Analysis Framework Demonstration")
    print("=" * 60)
    print()
    
    # Step 1: Generate or load experimental data
    print("Step 1: Generating realistic experimental data...")
    runs = simulate_realistic_experimental_data(n_runs_per_generator=20)
    print(f"✓ Generated {len(runs)} experimental runs")
    print()
    
    # Step 2: Perform comprehensive analysis
    print("Step 2: Performing comprehensive statistical analysis...")
    report = perform_comprehensive_analysis(runs)
    print("✓ Statistical analysis completed")
    print()
    
    # Step 3: Create visualizations
    print("Step 3: Creating detailed visualizations...")
    create_detailed_visualizations(runs, report)
    print("✓ Visualizations created")
    print()
    
    # Step 4: Generate summary report
    print("Step 4: Generating summary report...")
    generate_summary_report(report, Path("docs/statistical_analysis"))
    print("✓ Summary report generated")
    print()
    
    # Display key insights
    print("=" * 60)
    print("KEY INSIGHTS FROM ANALYSIS")
    print("=" * 60)
    
    insights = report['summary_insights']
    
    print("\n🏆 BEST PERFORMING GENERATORS:")
    for metric, best_info in insights['best_performing_generators'].items():
        print(f"  • {metric}: {best_info['generator']} (value: {best_info['value']:.4f})")
    
    print("\n🎯 MOST STABLE GENERATORS:")
    for metric, stable_info in insights['most_stable_generators'].items():
        print(f"  • {metric}: {stable_info['generator']} (stability: {stable_info['stability_score']:.4f})")
    
    print("\n⚡ FASTEST CONVERGING GENERATORS:")
    for metric, fast_info in insights['fastest_converging_generators'].items():
        print(f"  • {metric}: {fast_info['generator']} ({fast_info['epochs']:.1f} epochs)")
    
    print(f"\n📊 SIGNIFICANT DIFFERENCES: {len(insights['significant_differences'])} found")
    
    print("\n" + "=" * 60)
    print("Analysis complete! Check docs/statistical_analysis/ for detailed results.")
    print("=" * 60)


if __name__ == "__main__":
    main()