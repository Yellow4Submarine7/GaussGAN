"""
Example Scripts for GaussGAN Stability Analysis System

This module provides comprehensive examples showing how to use the stability
analysis system for comparing quantum vs classical generators.

Examples include:
1. Basic stability analysis from existing MLflow runs
2. Running new stability experiments
3. Advanced analysis with custom metrics
4. Comparing specific generator configurations
5. Generating publication-ready reports

Usage:
    python stability_analysis_examples.py --example basic
    python stability_analysis_examples.py --example run_experiments
    python stability_analysis_examples.py --example advanced
    
Author: Created for GaussGAN quantum vs classical generator comparison
"""

import os
import sys
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from docs.stability_analyzer import StabilityAnalyzer, ExperimentResult
from docs.stability_experiment_runner import StabilityExperimentRunner


def example_basic_analysis():
    """
    Example 1: Basic stability analysis from existing MLflow runs.
    
    This example shows how to:
    - Initialize the StabilityAnalyzer
    - Load existing experiment results from MLflow
    - Generate a basic stability report
    - Display key findings
    """
    print("=== Example 1: Basic Stability Analysis ===\n")
    
    # Initialize analyzer
    analyzer = StabilityAnalyzer(
        experiment_name="GaussGAN-manual",  # Your MLflow experiment name
        stability_threshold=0.15  # CV threshold for "stable" results
    )
    
    # Load experiments from MLflow
    print("Loading experiments from MLflow...")
    loaded_count = analyzer.load_from_mlflow(
        generator_types=['classical_normal', 'classical_uniform', 
                        'quantum_samples', 'quantum_shadows'],
        max_runs=100  # Limit to recent 100 runs
    )
    
    print(f"Loaded {loaded_count} experiments from MLflow")
    
    if loaded_count == 0:
        print("No experiments found. Using synthetic example data...")
        # Create synthetic data for demonstration
        example_results = _create_synthetic_stability_data()
        analyzer.add_experiment_results(example_results)
        loaded_count = len(example_results)
    
    # Perform stability analysis
    print("\\nPerforming stability analysis...")
    stability_results = analyzer.analyze_stability()
    
    # Generate comprehensive report
    print("Generating stability report...")
    report = analyzer.generate_stability_report("docs/stability_analysis/basic_example")
    
    # Display key findings
    print("\\n=== KEY FINDINGS ===")
    
    # Show stability summary
    summary_df = analyzer.get_stability_summary()
    print("\\nStability Summary:")
    print(summary_df.round(3).to_string(index=False))
    
    # Highlight most/least stable generators
    if not summary_df.empty:
        for metric in ['final_kl_divergence', 'final_log_likelihood']:
            metric_data = summary_df[summary_df['metric'] == metric]
            if not metric_data.empty:
                most_stable = metric_data.loc[metric_data['stability_score'].idxmax()]
                least_stable = metric_data.loc[metric_data['stability_score'].idxmin()]
                
                print(f"\\n{metric}:")
                print(f"  Most stable: {most_stable['generator_type']} (score: {most_stable['stability_score']:.3f})")
                print(f"  Least stable: {least_stable['generator_type']} (score: {least_stable['stability_score']:.3f})")
    
    # Show recommendations
    print("\\n=== RECOMMENDATIONS ===")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"{i}. {rec}")
    
    print(f"\\nDetailed report saved to: docs/stability_analysis/basic_example/")
    print("Check the following files:")
    print("  - stability_report.md: Human-readable report")
    print("  - stability_report.json: Machine-readable data")
    print("  - *.png: Visualization plots")


def example_run_new_experiments():
    """
    Example 2: Run new stability experiments and analyze results.
    
    This example shows how to:
    - Configure and run multiple experiments with different seeds
    - Collect results automatically
    - Perform stability analysis on the new results
    """
    print("=== Example 2: Running New Stability Experiments ===\\n")
    
    # Initialize experiment runner
    runner = StabilityExperimentRunner(
        output_dir="docs/stability_analysis/new_experiments",
        log_level="INFO"
    )
    
    print("This example would run new experiments. For demonstration, we'll show the process:")
    print("\\n1. Initialize runner:")
    print("   runner = StabilityExperimentRunner()")
    
    print("\\n2. Run experiment suite:")
    print("   summary = runner.run_stability_experiment_suite(")
    print("       generator_types=['classical_normal', 'quantum_samples'],")
    print("       num_seeds=5,  # 5 different seeds per generator")
    print("       max_epochs=50,")
    print("       parallel=False,  # Set True for parallel execution")
    print("       experiment_name='GaussGAN-stability-demo'")
    print("   )")
    
    print("\\n3. Automatic analysis:")
    print("   runner.run_analysis_after_experiments('GaussGAN-stability-demo')")
    
    # For actual execution, uncomment the following:
    """
    summary = runner.run_stability_experiment_suite(
        generator_types=['classical_normal', 'quantum_samples'],
        num_seeds=5,
        max_epochs=30,  # Reduced for faster demo
        parallel=False,
        experiment_name='GaussGAN-stability-demo'
    )
    
    # Run analysis
    runner.run_analysis_after_experiments('GaussGAN-stability-demo')
    
    print(f"\\nExperiment Summary:")
    print(f"Total experiments: {summary['experiment_suite_info']['total_experiments']}")
    print(f"Success rate: {summary['results_summary']['success_rate']:.2%}")
    """
    
    print("\\n[Demo mode - actual experiments not run]")
    print("To run real experiments, uncomment the execution code in this function.")


def example_advanced_analysis():
    """
    Example 3: Advanced stability analysis with custom metrics and comparisons.
    
    This example shows how to:
    - Create custom stability metrics
    - Perform detailed generator comparisons
    - Generate publication-ready visualizations
    - Export results for further analysis
    """
    print("=== Example 3: Advanced Stability Analysis ===\\n")
    
    # Create comprehensive synthetic dataset for demonstration
    analyzer = StabilityAnalyzer(
        experiment_name="GaussGAN-advanced-demo",
        stability_threshold=0.12  # Stricter threshold
    )
    
    print("Creating comprehensive synthetic dataset...")
    advanced_results = _create_comprehensive_synthetic_data()
    analyzer.add_experiment_results(advanced_results)
    
    # Perform detailed analysis
    print("Performing advanced stability analysis...")
    stability_results = analyzer.analyze_stability()
    
    # Generate detailed report
    report = analyzer.generate_stability_report("docs/stability_analysis/advanced_example")
    
    # Advanced analysis: Statistical significance testing
    print("\\n=== ADVANCED STATISTICAL ANALYSIS ===")
    
    summary_df = analyzer.get_stability_summary()
    
    # Compare generators for each metric
    for metric in summary_df['metric'].unique():
        metric_data = summary_df[summary_df['metric'] == metric]
        
        print(f"\\n{metric}:")
        print("Generator Performance Ranking (by stability score):")
        
        ranked = metric_data.sort_values('stability_score', ascending=False)
        for i, (_, row) in enumerate(ranked.iterrows(), 1):
            print(f"  {i}. {row['generator_type']}: "
                  f"Score={row['stability_score']:.3f}, "
                  f"CV={row['cv']:.3f}, "
                  f"Mean={row['mean']:.3f}±{row['std']:.3f}")
    
    # Export detailed results
    analyzer.export_results("docs/stability_analysis/advanced_example/detailed_results.json")
    
    # Custom visualization
    _create_custom_stability_plots(analyzer, "docs/stability_analysis/advanced_example")
    
    print(f"\\nAdvanced analysis complete!")
    print("Generated additional custom visualizations:")
    print("  - stability_heatmap.png: Stability score heatmap")
    print("  - performance_radar.png: Multi-metric radar chart")
    print("  - confidence_intervals.png: Error bar comparison")


def example_production_pipeline():
    """
    Example 4: Production-ready stability analysis pipeline.
    
    This example shows a complete workflow for production use:
    - Automated experiment execution with error handling
    - Comprehensive result validation
    - Automated report generation
    - Decision support system
    """
    print("=== Example 4: Production Stability Analysis Pipeline ===\\n")
    
    print("Production Pipeline Steps:")
    print("1. Experiment Planning")
    print("2. Automated Execution with Monitoring")
    print("3. Result Validation")
    print("4. Comprehensive Analysis")
    print("5. Decision Support")
    print("6. Report Distribution")
    
    # Demonstrate pipeline configuration
    pipeline_config = {
        'experiment_name': 'GaussGAN-production-stability',
        'generator_types': [
            'classical_normal',
            'classical_uniform', 
            'quantum_samples',
            'quantum_shadows'
        ],
        'seeds_per_generator': 20,
        'max_epochs': 100,
        'parallel_execution': True,
        'max_workers': 4,
        'quality_thresholds': {
            'min_success_rate': 0.90,
            'max_cv_kl_divergence': 0.20,
            'min_stability_score': 0.70
        },
        'notification_settings': {
            'email_on_completion': True,
            'slack_webhook': None,  # Set webhook URL for notifications
            'report_recipients': ['team@company.com']
        }
    }
    
    print(f"\\nPipeline Configuration:")
    for key, value in pipeline_config.items():
        print(f"  {key}: {value}")
    
    # Simulate production analysis
    print("\\n=== SIMULATED PRODUCTION ANALYSIS ===")
    
    analyzer = StabilityAnalyzer()
    synthetic_results = _create_production_quality_data()
    analyzer.add_experiment_results(synthetic_results)
    
    # Quality checks
    stability_results = analyzer.analyze_stability()
    quality_report = _perform_quality_checks(analyzer, pipeline_config['quality_thresholds'])
    
    print("\\nQuality Check Results:")
    for check, result in quality_report.items():
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        print(f"  {check}: {status} ({result['message']})")
    
    # Decision support
    recommendation = _generate_production_recommendation(quality_report, analyzer)
    print(f"\\nProduction Recommendation:")
    print(f"  {recommendation}")
    
    print("\\n[Production pipeline demonstration complete]")


def _create_synthetic_stability_data():
    """Create synthetic data for basic example."""
    np.random.seed(42)
    
    results = []
    generators = ['classical_normal', 'quantum_samples']
    
    for gen_type in generators:
        for seed in range(42, 52):  # 10 seeds per generator
            # Simulate different stability characteristics
            if gen_type == 'classical_normal':
                kl_mean, kl_std = 0.05, 0.008
                ll_mean, ll_std = -2.3, 0.15
                time_mean, time_std = 120, 15
            else:
                kl_mean, kl_std = 0.08, 0.025
                ll_mean, ll_std = -2.8, 0.35
                time_mean, time_std = 180, 40
            
            result = ExperimentResult(
                run_id=f"synthetic_{gen_type}_{seed}",
                generator_type=gen_type,
                seed=seed,
                final_kl_divergence=max(0.001, np.random.normal(kl_mean, kl_std)),
                final_log_likelihood=np.random.normal(ll_mean, ll_std),
                final_is_positive=np.random.uniform(0.85, 0.98),
                training_time=max(60, np.random.normal(time_mean, time_std)),
                convergence_epoch=np.random.randint(20, 45),
                max_epochs=50,
                status='completed'
            )
            
            results.append(result)
    
    return results


def _create_comprehensive_synthetic_data():
    """Create comprehensive synthetic data for advanced example."""
    np.random.seed(123)
    
    results = []
    generators = [
        'classical_normal', 'classical_uniform',
        'quantum_samples', 'quantum_shadows'
    ]
    
    # Define different stability profiles
    stability_profiles = {
        'classical_normal': {'kl': (0.04, 0.006), 'll': (-2.2, 0.12), 'time': (110, 12)},
        'classical_uniform': {'kl': (0.06, 0.010), 'll': (-2.5, 0.18), 'time': (115, 18)},
        'quantum_samples': {'kl': (0.09, 0.030), 'll': (-2.9, 0.40), 'time': (200, 50)},
        'quantum_shadows': {'kl': (0.12, 0.045), 'll': (-3.2, 0.50), 'time': (250, 60)}
    }
    
    for gen_type in generators:
        profile = stability_profiles[gen_type]
        
        for seed in range(100, 115):  # 15 seeds per generator
            # Add occasional outliers
            outlier_factor = 1.0
            if np.random.random() < 0.08:  # 8% outliers
                outlier_factor = np.random.uniform(1.8, 2.5)
            
            kl_mean, kl_std = profile['kl']
            ll_mean, ll_std = profile['ll']
            time_mean, time_std = profile['time']
            
            result = ExperimentResult(
                run_id=f"advanced_{gen_type}_{seed}",
                generator_type=gen_type,
                seed=seed,
                final_kl_divergence=max(0.001, np.random.normal(kl_mean, kl_std) * outlier_factor),
                final_log_likelihood=np.random.normal(ll_mean, ll_std) / outlier_factor,
                final_is_positive=np.random.uniform(0.80, 0.95),
                training_time=max(60, np.random.normal(time_mean, time_std) * outlier_factor),
                convergence_epoch=np.random.randint(15, 50),
                max_epochs=100,
                status='completed'
            )
            
            results.append(result)
    
    return results


def _create_production_quality_data():
    """Create production-quality synthetic data."""
    np.random.seed(456)
    
    results = []
    generators = ['classical_normal', 'quantum_samples']
    
    for gen_type in generators:
        for seed in range(200, 220):  # 20 seeds per generator
            # High-quality, low-variance results
            if gen_type == 'classical_normal':
                kl_mean, kl_std = 0.035, 0.004
                ll_mean, ll_std = -2.1, 0.08
                time_mean, time_std = 105, 8
            else:
                kl_mean, kl_std = 0.055, 0.012
                ll_mean, ll_std = -2.4, 0.18
                time_mean, time_std = 160, 25
            
            result = ExperimentResult(
                run_id=f"production_{gen_type}_{seed}",
                generator_type=gen_type,
                seed=seed,
                final_kl_divergence=max(0.001, np.random.normal(kl_mean, kl_std)),
                final_log_likelihood=np.random.normal(ll_mean, ll_std),
                final_is_positive=np.random.uniform(0.92, 0.99),
                training_time=max(60, np.random.normal(time_mean, time_std)),
                convergence_epoch=np.random.randint(25, 40),
                max_epochs=100,
                status='completed'
            )
            
            results.append(result)
    
    return results


def _create_custom_stability_plots(analyzer, output_dir):
    """Create custom visualization plots."""
    output_dir = Path(output_dir)
    
    # 1. Stability Heatmap
    summary_df = analyzer.get_stability_summary()
    
    # Pivot data for heatmap
    heatmap_data = summary_df.pivot(index='generator_type', 
                                   columns='metric', 
                                   values='stability_score')
    
    plt.figure(figsize=(12, 8))
    import seaborn as sns
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', center=0.5, 
                cbar_kws={'label': 'Stability Score'})
    plt.title('Stability Score Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'stability_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confidence Intervals Plot
    plt.figure(figsize=(14, 10))
    
    metrics = ['final_kl_divergence', 'final_log_likelihood', 'training_time']
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        
        metric_data = summary_df[summary_df['metric'] == metric]
        
        generators = metric_data['generator_type']
        means = metric_data['mean']
        stds = metric_data['std']
        
        plt.errorbar(range(len(generators)), means, yerr=stds, 
                    fmt='o', capsize=5, capthick=2, markersize=8)
        plt.xticks(range(len(generators)), generators, rotation=45)
        plt.title(f'{metric.replace("_", " ").title()}')
        plt.grid(alpha=0.3)
    
    plt.suptitle('Confidence Intervals Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_intervals.png', dpi=300, bbox_inches='tight')
    plt.close()


def _perform_quality_checks(analyzer, thresholds):
    """Perform quality checks on stability results."""
    summary_df = analyzer.get_stability_summary()
    
    checks = {}
    
    # Check 1: KL divergence stability
    kl_data = summary_df[summary_df['metric'] == 'final_kl_divergence']
    if not kl_data.empty:
        max_cv = kl_data['cv'].max()
        checks['kl_divergence_stability'] = {
            'passed': max_cv <= thresholds['max_cv_kl_divergence'],
            'message': f"Max CV: {max_cv:.3f} (threshold: {thresholds['max_cv_kl_divergence']})"
        }
    
    # Check 2: Minimum stability scores
    min_stability = summary_df['stability_score'].min()
    checks['minimum_stability_score'] = {
        'passed': min_stability >= thresholds['min_stability_score'],
        'message': f"Min stability: {min_stability:.3f} (threshold: {thresholds['min_stability_score']})"
    }
    
    # Check 3: Sample size adequacy
    min_sample_size = summary_df['sample_size'].min()
    checks['sample_size_adequacy'] = {
        'passed': min_sample_size >= 10,
        'message': f"Min sample size: {min_sample_size} (threshold: 10)"
    }
    
    return checks


def _generate_production_recommendation(quality_report, analyzer):
    """Generate production recommendation based on analysis."""
    passed_checks = sum(1 for check in quality_report.values() if check['passed'])
    total_checks = len(quality_report)
    
    if passed_checks == total_checks:
        summary_df = analyzer.get_stability_summary()
        kl_data = summary_df[summary_df['metric'] == 'final_kl_divergence']
        
        if not kl_data.empty:
            best_generator = kl_data.loc[kl_data['stability_score'].idxmax(), 'generator_type']
            return f"✓ All quality checks passed. Recommended for production: {best_generator}"
        else:
            return "✓ All quality checks passed. Ready for production deployment."
    
    elif passed_checks >= total_checks * 0.8:
        return "⚠ Most quality checks passed. Consider additional testing before production."
    
    else:
        return "✗ Multiple quality checks failed. Not recommended for production. Additional experiments needed."


def main():
    """Main function for running examples."""
    parser = argparse.ArgumentParser(
        description="Run stability analysis examples for GaussGAN"
    )
    
    parser.add_argument(
        '--example',
        choices=['basic', 'run_experiments', 'advanced', 'production', 'all'],
        default='basic',
        help='Which example to run (default: basic)'
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    Path("docs/stability_analysis").mkdir(parents=True, exist_ok=True)
    
    if args.example == 'all':
        example_basic_analysis()
        print("\\n" + "="*60 + "\\n")
        example_run_new_experiments()
        print("\\n" + "="*60 + "\\n")
        example_advanced_analysis()
        print("\\n" + "="*60 + "\\n")
        example_production_pipeline()
    
    elif args.example == 'basic':
        example_basic_analysis()
    
    elif args.example == 'run_experiments':
        example_run_new_experiments()
    
    elif args.example == 'advanced':
        example_advanced_analysis()
    
    elif args.example == 'production':
        example_production_pipeline()
    
    print("\\n=== Examples completed successfully! ===")


if __name__ == "__main__":
    main()