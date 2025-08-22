"""
Complete Usage Example for GaussGAN Statistical Analysis Framework
================================================================

This script demonstrates the complete workflow for comparing quantum vs classical
generators using the statistical analysis framework. It serves as both a tutorial
and a practical template for real-world analysis.

Run this script to see the complete analysis pipeline in action:
    python docs/complete_usage_example.py

The script will:
1. Set up the analysis environment
2. Generate or load experimental data
3. Perform comprehensive statistical analysis
4. Generate visualizations and reports
5. Provide actionable recommendations
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings

# Add the docs directory to Python path
sys.path.append(str(Path(__file__).parent))

# Import framework components
from statistical_analysis_framework import (
    MultiRunExperimentManager, StatisticalAnalyzer, 
    ExperimentRun
)
from convergence_analysis_tools import AdvancedConvergenceAnalyzer
from stability_variance_analyzer import StabilityAnalyzer
from comprehensive_reporting_system import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def setup_analysis_environment():
    """Set up the analysis environment and directories."""
    logger.info("Setting up analysis environment...")
    
    # Create output directories
    output_dirs = [
        'docs/complete_analysis',
        'docs/complete_analysis/experiments',
        'docs/complete_analysis/statistical_analysis',
        'docs/complete_analysis/convergence_analysis',
        'docs/complete_analysis/stability_analysis',
        'docs/complete_analysis/reports'
    ]
    
    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("✓ Analysis environment ready")
    return Path('docs/complete_analysis')


def generate_comprehensive_experimental_data(n_runs_per_generator=25):
    """
    Generate comprehensive experimental data for demonstration.
    
    This function creates realistic experimental data that represents
    what you would get from actual GaussGAN training runs.
    
    Args:
        n_runs_per_generator: Number of runs per generator type
        
    Returns:
        List of ExperimentRun objects
    """
    logger.info(f"Generating experimental data ({n_runs_per_generator} runs per generator)...")
    
    generator_types = [
        'classical_normal', 
        'classical_uniform', 
        'quantum_samples', 
        'quantum_shadows'
    ]
    
    # Realistic performance profiles based on expected behavior
    performance_profiles = {
        'classical_normal': {
            'KLDivergence': {'mean': 0.10, 'std': 0.025, 'convergence_rate': 0.96},
            'LogLikelihood': {'mean': -2.1, 'std': 0.15, 'convergence_rate': 0.96},
            'WassersteinDistance': {'mean': 0.06, 'std': 0.015, 'convergence_rate': 0.92},
            'MMDDistance': {'mean': 0.05, 'std': 0.012, 'convergence_rate': 0.88},
            'IsPositive': {'mean': 0.85, 'std': 0.08, 'convergence_rate': 0.95},
            'training_time': {'mean': 145, 'std': 25},
            'pattern_weights': {'smooth_exponential': 0.7, 'oscillatory': 0.2, 'stepped': 0.1}
        },
        'classical_uniform': {
            'KLDivergence': {'mean': 0.14, 'std': 0.035, 'convergence_rate': 0.90},
            'LogLikelihood': {'mean': -2.4, 'std': 0.20, 'convergence_rate': 0.90},
            'WassersteinDistance': {'mean': 0.09, 'std': 0.020, 'convergence_rate': 0.86},
            'MMDDistance': {'mean': 0.07, 'std': 0.018, 'convergence_rate': 0.82},
            'IsPositive': {'mean': 0.82, 'std': 0.10, 'convergence_rate': 0.88},
            'training_time': {'mean': 130, 'std': 22},
            'pattern_weights': {'smooth_exponential': 0.6, 'oscillatory': 0.25, 'stepped': 0.15}
        },
        'quantum_samples': {
            'KLDivergence': {'mean': 0.24, 'std': 0.065, 'convergence_rate': 0.76},
            'LogLikelihood': {'mean': -3.2, 'std': 0.45, 'convergence_rate': 0.76},
            'WassersteinDistance': {'mean': 0.15, 'std': 0.040, 'convergence_rate': 0.72},
            'MMDDistance': {'mean': 0.13, 'std': 0.035, 'convergence_rate': 0.68},
            'IsPositive': {'mean': 0.75, 'std': 0.12, 'convergence_rate': 0.74},
            'training_time': {'mean': 520, 'std': 95},
            'pattern_weights': {'oscillatory': 0.4, 'late_convergence': 0.3, 'smooth_exponential': 0.2, 'stepped': 0.1}
        },
        'quantum_shadows': {
            'KLDivergence': {'mean': 0.19, 'std': 0.050, 'convergence_rate': 0.84},
            'LogLikelihood': {'mean': -2.8, 'std': 0.32, 'convergence_rate': 0.84},
            'WassersteinDistance': {'mean': 0.12, 'std': 0.032, 'convergence_rate': 0.80},
            'MMDDistance': {'mean': 0.10, 'std': 0.025, 'convergence_rate': 0.76},
            'IsPositive': {'mean': 0.78, 'std': 0.11, 'convergence_rate': 0.82},
            'training_time': {'mean': 410, 'std': 75},
            'pattern_weights': {'smooth_exponential': 0.5, 'oscillatory': 0.3, 'late_convergence': 0.2}
        }
    }
    
    runs = []
    base_time = datetime.now()
    
    for gen_type in generator_types:
        profile = performance_profiles[gen_type]
        logger.info(f"  Generating {n_runs_per_generator} runs for {gen_type}...")
        
        for run_idx in range(n_runs_per_generator):
            seed = 42 + len(runs)
            run_id = f"{gen_type}_run_{run_idx:03d}_seed_{seed}"
            
            # Determine if this run converges
            converged = np.random.random() < profile['KLDivergence']['convergence_rate']
            
            # Generate training duration
            training_duration = max(30, np.random.normal(
                profile['training_time']['mean'],
                profile['training_time']['std']
            ))
            
            # Generate convergence epoch
            if converged:
                # Earlier convergence for better performing generators
                base_convergence_epoch = 20 if 'classical' in gen_type else 35
                convergence_epoch = max(10, int(np.random.normal(base_convergence_epoch, 8)))
            else:
                convergence_epoch = None
            
            max_epochs = convergence_epoch + 15 if converged else 50
            
            # Generate final metrics
            final_metrics = {}
            for metric_name, metric_params in profile.items():
                if metric_name in ['training_time', 'pattern_weights']:
                    continue
                
                base_mean = metric_params['mean']
                base_std = metric_params['std']
                
                if converged:
                    # Better performance if converged
                    value = np.random.normal(base_mean, base_std * 0.8)
                else:
                    # Worse performance if not converged
                    if metric_name in ['KLDivergence', 'WassersteinDistance', 'MMDDistance']:
                        # Higher is worse for these metrics
                        value = np.random.normal(base_mean * 1.4, base_std * 1.2)
                    elif metric_name == 'LogLikelihood':
                        # More negative is worse
                        value = np.random.normal(base_mean * 1.3, base_std * 1.2)
                    else:  # IsPositive
                        # Lower is worse
                        value = np.random.normal(base_mean * 0.8, base_std * 1.2)
                
                # Apply realistic bounds
                if metric_name in ['KLDivergence', 'WassersteinDistance', 'MMDDistance']:
                    value = max(0.01, value)
                elif metric_name == 'LogLikelihood':
                    value = min(-0.5, value)
                elif metric_name == 'IsPositive':
                    value = np.clip(value, 0.1, 0.99)
                
                final_metrics[metric_name] = value
            
            # Generate metric history
            metric_history = {}
            for metric_name in ['KLDivergence', 'LogLikelihood', 'WassersteinDistance', 'MMDDistance']:
                final_value = final_metrics[metric_name]
                
                if metric_name in ['KLDivergence', 'WassersteinDistance', 'MMDDistance']:
                    # These should decrease
                    start_value = final_value * np.random.uniform(2.5, 4.0)
                    improving = True
                else:
                    # LogLikelihood should increase (become less negative)
                    start_value = final_value * np.random.uniform(1.3, 2.0)
                    improving = False
                
                # Generate convergence pattern
                pattern_weights = profile.get('pattern_weights', {'smooth_exponential': 1.0})
                pattern = np.random.choice(
                    list(pattern_weights.keys()),
                    p=list(pattern_weights.values())
                )
                
                history = generate_metric_history(
                    start_value, final_value, max_epochs, pattern, improving, converged
                )
                metric_history[metric_name] = history
            
            # Generate loss history
            loss_history = {
                'g_loss': generate_loss_history(max_epochs, 'generator'),
                'd_loss': generate_loss_history(max_epochs, 'discriminator')
            }
            
            # Create experiment run
            start_time = base_time
            end_time = base_time
            
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
                    'batch_size': np.random.choice([256, 512]),
                    'learning_rate': np.random.choice([0.001, 0.0005]),
                    'generator_type': gen_type,
                    'quantum_qubits': 6 if 'quantum' in gen_type else None,
                    'quantum_layers': 2 if 'quantum' in gen_type else None
                },
                training_duration=training_duration,
                converged=converged
            )
            
            runs.append(run)
    
    logger.info(f"✓ Generated {len(runs)} experimental runs")
    return runs


def generate_metric_history(start_value, final_value, max_epochs, pattern, improving, converged):
    """Generate realistic metric history based on convergence pattern."""
    history = []
    epochs = np.arange(max_epochs)
    
    if pattern == 'smooth_exponential':
        # Smooth exponential convergence
        decay_rate = 0.08 if converged else 0.03
        for epoch in epochs:
            progress = 1 - np.exp(-epoch * decay_rate)
            value = start_value + (final_value - start_value) * progress
            noise = np.random.normal(0, abs(final_value) * 0.02)
            history.append(value + noise)
    
    elif pattern == 'oscillatory':
        # Oscillatory convergence
        for epoch in epochs:
            base_progress = 1 - np.exp(-epoch * 0.05)
            oscillation = 0.1 * np.sin(epoch * 0.3) * np.exp(-epoch * 0.02)
            progress = base_progress + oscillation
            progress = max(0, min(1, progress))
            value = start_value + (final_value - start_value) * progress
            history.append(value)
    
    elif pattern == 'stepped':
        # Stepped convergence with plateaus
        for epoch in epochs:
            # Create steps at specific epochs
            step_epochs = [10, 25, 40]
            step_progress = 0
            for step_epoch in step_epochs:
                if epoch >= step_epoch:
                    step_progress += 0.3
            step_progress = min(1, step_progress)
            value = start_value + (final_value - start_value) * step_progress
            noise = np.random.normal(0, abs(final_value) * 0.01)
            history.append(value + noise)
    
    elif pattern == 'late_convergence':
        # Slow start, then rapid convergence
        for epoch in epochs:
            if epoch < max_epochs * 0.6:
                progress = 0.1 * (epoch / (max_epochs * 0.6))
            else:
                remaining_progress = 0.9
                remaining_epochs = max_epochs - max_epochs * 0.6
                epoch_in_phase = epoch - max_epochs * 0.6
                progress = 0.1 + remaining_progress * (1 - np.exp(-epoch_in_phase * 0.15))
            
            value = start_value + (final_value - start_value) * progress
            noise = np.random.normal(0, abs(final_value) * 0.03)
            history.append(value + noise)
    
    else:
        # Default: linear convergence
        for epoch in epochs:
            progress = epoch / max_epochs if max_epochs > 0 else 0
            value = start_value + (final_value - start_value) * progress
            noise = np.random.normal(0, abs(final_value) * 0.02)
            history.append(value + noise)
    
    return history


def generate_loss_history(max_epochs, loss_type):
    """Generate realistic loss history."""
    history = []
    
    if loss_type == 'generator':
        # Generator loss: starts high, decreases with oscillations
        for epoch in range(max_epochs):
            base_loss = 2.0 * np.exp(-epoch * 0.03) + 0.5
            oscillation = 0.3 * np.sin(epoch * 0.2)
            noise = np.random.normal(0, 0.1)
            history.append(base_loss + oscillation + noise)
    
    else:  # discriminator
        # Discriminator loss: oscillates around zero
        for epoch in range(max_epochs):
            base_loss = np.random.normal(0, 0.2)
            trend = -0.01 * epoch  # Slight downward trend
            history.append(base_loss + trend)
    
    return history


def perform_statistical_analysis(experimental_runs, output_dir):
    """Perform comprehensive statistical analysis."""
    logger.info("Performing statistical analysis...")
    
    # Initialize statistical analyzer
    analyzer = StatisticalAnalyzer(
        alpha=0.05,
        correction_method='bonferroni'
    )
    
    # Metrics to analyze
    metrics_to_analyze = ['KLDivergence', 'LogLikelihood', 'WassersteinDistance', 'MMDDistance', 'IsPositive']
    
    statistical_results = {
        'descriptive_statistics': {},
        'pairwise_comparisons': {},
        'summary_insights': {
            'best_performing_generators': {},
            'most_stable_generators': {},
            'significant_differences': []
        }
    }
    
    # Group runs by generator type
    groups = analyzer.group_runs_by_generator(experimental_runs)
    logger.info(f"  Analyzing {len(groups)} generator types across {len(metrics_to_analyze)} metrics")
    
    # Descriptive statistics
    for metric in metrics_to_analyze:
        logger.info(f"  Computing descriptive statistics for {metric}...")
        statistical_results['descriptive_statistics'][metric] = {}
        
        for gen_type, gen_runs in groups.items():
            stats = analyzer.compute_descriptive_statistics(gen_runs, metric)
            statistical_results['descriptive_statistics'][metric][gen_type] = stats
    
    # Pairwise comparisons
    for metric in metrics_to_analyze:
        logger.info(f"  Performing pairwise comparisons for {metric}...")
        comparisons = analyzer.compare_all_generators(experimental_runs, metric)
        statistical_results['pairwise_comparisons'][metric] = comparisons
        
        # Extract significant differences
        for gen1, gen1_tests in comparisons.items():
            for gen2, test_result in gen1_tests.items():
                if test_result.significant:
                    statistical_results['summary_insights']['significant_differences'].append({
                        'metric': metric,
                        'generator1': gen1,
                        'generator2': gen2,
                        'p_value': test_result.p_value,
                        'effect_size': test_result.effect_size,
                        'test_name': test_result.test_name
                    })
    
    # Find best performing generators
    for metric in metrics_to_analyze:
        desc_stats = statistical_results['descriptive_statistics'][metric]
        
        # Determine if lower or higher is better
        lower_is_better = metric in ['KLDivergence', 'WassersteinDistance', 'MMDDistance']
        
        best_gen = None
        best_value = float('inf') if lower_is_better else float('-inf')
        
        for gen_type, stats in desc_stats.items():
            if stats['count'] > 0:
                mean_value = stats['mean']
                if lower_is_better and mean_value < best_value:
                    best_value = mean_value
                    best_gen = gen_type
                elif not lower_is_better and mean_value > best_value:
                    best_value = mean_value
                    best_gen = gen_type
        
        if best_gen:
            statistical_results['summary_insights']['best_performing_generators'][metric] = {
                'generator': best_gen,
                'value': best_value
            }
    
    logger.info("✓ Statistical analysis completed")
    return statistical_results


def perform_convergence_analysis(experimental_runs, output_dir):
    """Perform convergence analysis."""
    logger.info("Performing convergence analysis...")
    
    # Initialize convergence analyzer
    conv_analyzer = AdvancedConvergenceAnalyzer(output_dir / 'convergence_analysis')
    
    # Set convergence thresholds
    conv_analyzer.thresholds = {
        'KLDivergence': 0.25,
        'LogLikelihood': -2.8,
        'WassersteinDistance': 0.12,
        'MMDDistance': 0.10
    }
    
    # Perform analysis
    convergence_results = conv_analyzer.analyze_convergence_events(
        experimental_runs=experimental_runs,
        metrics_to_analyze=['KLDivergence', 'LogLikelihood', 'WassersteinDistance', 'MMDDistance']
    )
    
    # Create visualizations
    try:
        conv_analyzer.create_convergence_visualizations(convergence_results)
        logger.info("  ✓ Convergence visualizations created")
    except Exception as e:
        logger.warning(f"  Could not create convergence visualizations: {e}")
    
    logger.info("✓ Convergence analysis completed")
    return convergence_results


def perform_stability_analysis(experimental_runs, output_dir):
    """Perform stability analysis."""
    logger.info("Performing stability analysis...")
    
    # Initialize stability analyzer
    stability_analyzer = StabilityAnalyzer(
        output_dir=output_dir / 'stability_analysis',
        confidence_level=0.95
    )
    
    # Perform analysis
    stability_results = stability_analyzer.analyze_stability(
        experimental_runs=experimental_runs,
        metrics_to_analyze=['KLDivergence', 'LogLikelihood', 'WassersteinDistance', 'MMDDistance']
    )
    
    # Create visualizations
    try:
        stability_analyzer.create_stability_visualizations(stability_results)
        logger.info("  ✓ Stability visualizations created")
    except Exception as e:
        logger.warning(f"  Could not create stability visualizations: {e}")
    
    logger.info("✓ Stability analysis completed")
    return stability_results


def generate_comprehensive_reports(statistical_results, convergence_results, stability_results, output_dir):
    """Generate comprehensive reports."""
    logger.info("Generating comprehensive reports...")
    
    # Initialize report generator
    report_gen = ReportGenerator(
        output_dir=output_dir / 'reports',
        report_title="GaussGAN Quantum vs Classical Generator Analysis",
        author="Statistical Analysis Framework",
        institution="GaussGAN Research Project"
    )
    
    # Generate reports
    try:
        generated_files = report_gen.generate_comprehensive_report(
            statistical_results=statistical_results,
            convergence_results=convergence_results,
            stability_results=stability_results,
            experiment_metadata={
                'analysis_type': 'Complete demonstration analysis',
                'total_runs': len(statistical_results.get('descriptive_statistics', {}).get('KLDivergence', {})) * 25,
                'generator_types': ['classical_normal', 'classical_uniform', 'quantum_samples', 'quantum_shadows'],
                'metrics_analyzed': ['KLDivergence', 'LogLikelihood', 'WassersteinDistance', 'MMDDistance', 'IsPositive'],
                'analysis_framework_version': '1.0.0',
                'timestamp': datetime.now().isoformat()
            }
        )
        
        logger.info("✓ Comprehensive reports generated:")
        for report_type, file_path in generated_files.items():
            if file_path:
                logger.info(f"    {report_type}: {file_path}")
        
        return generated_files
    
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return {}


def analyze_and_summarize_results(statistical_results, convergence_results, stability_results):
    """Analyze and summarize the key findings."""
    logger.info("Analyzing and summarizing results...")
    
    print("\n" + "="*80)
    print("GAUSSGAN STATISTICAL ANALYSIS RESULTS")
    print("="*80)
    
    # Statistical findings
    print("\n🔬 STATISTICAL ANALYSIS")
    print("-" * 40)
    
    best_performers = statistical_results['summary_insights']['best_performing_generators']
    if best_performers:
        print("Best performing generators by metric:")
        for metric, info in best_performers.items():
            print(f"  • {metric}: {info['generator']} (value: {info['value']:.4f})")
        
        # Find overall best performer
        generator_counts = {}
        for info in best_performers.values():
            gen = info['generator']
            generator_counts[gen] = generator_counts.get(gen, 0) + 1
        
        overall_best = max(generator_counts.keys(), key=lambda x: generator_counts[x])
        print(f"\n🏆 Overall best performer: {overall_best}")
        print(f"   (Best in {generator_counts[overall_best]} out of {len(best_performers)} metrics)")
    
    # Significant differences
    sig_diffs = statistical_results['summary_insights']['significant_differences']
    print(f"\n📊 Statistically significant differences found: {len(sig_diffs)}")
    
    if sig_diffs:
        print("Key significant comparisons:")
        for diff in sig_diffs[:5]:  # Show top 5
            print(f"  • {diff['metric']}: {diff['generator1']} vs {diff['generator2']} "
                  f"(p={diff['p_value']:.4f}, effect size={diff.get('effect_size', 'N/A')})")
    
    # Convergence findings
    print("\n⏱️ CONVERGENCE ANALYSIS")
    print("-" * 40)
    
    if 'speed_comparison' in convergence_results:
        speed_comp = convergence_results['speed_comparison']
        
        # Average convergence rates
        avg_convergence_rates = {}
        for metric, comparisons in speed_comp.items():
            for gen_type, stats in comparisons.items():
                if gen_type not in avg_convergence_rates:
                    avg_convergence_rates[gen_type] = []
                avg_convergence_rates[gen_type].append(stats.get('convergence_rate', 0))
        
        for gen_type in avg_convergence_rates:
            avg_convergence_rates[gen_type] = np.mean(avg_convergence_rates[gen_type])
        
        print("Average convergence rates:")
        for gen_type, rate in sorted(avg_convergence_rates.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {gen_type}: {rate:.1%}")
        
        fastest_converter = max(avg_convergence_rates.keys(), key=lambda x: avg_convergence_rates[x])
        print(f"\n⚡ Fastest converging generator: {fastest_converter}")
    
    # Pattern analysis
    if 'pattern_analysis' in convergence_results:
        pattern_freq = convergence_results['pattern_analysis']['pattern_frequency']
        print(f"\nConvergence patterns observed:")
        for pattern, freq in sorted(pattern_freq.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {pattern}: {freq} occurrences")
    
    # Stability findings
    print("\n🛡️ STABILITY ANALYSIS")
    print("-" * 40)
    
    if 'risk_assessment' in stability_results:
        risk_assessment = stability_results['risk_assessment']
        
        print("Risk classifications:")
        for gen_type, assessment in risk_assessment.items():
            risk_class = assessment['risk_classification']
            risk_score = assessment['overall_risk_score']
            print(f"  • {gen_type}: {risk_class} (score: {risk_score:.3f})")
        
        # Find safest generator
        safest_gen = min(risk_assessment.keys(), 
                        key=lambda x: risk_assessment[x]['overall_risk_score'])
        print(f"\n🔒 Safest generator: {safest_gen}")
    
    # Final recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 40)
    
    recommendations = []
    
    # Performance recommendation
    if best_performers:
        performance_rec = max(generator_counts.keys(), key=lambda x: generator_counts[x])
        recommendations.append(f"For best performance: Use {performance_rec}")
    
    # Reliability recommendation
    if 'speed_comparison' in convergence_results and avg_convergence_rates:
        reliability_rec = max(avg_convergence_rates.keys(), key=lambda x: avg_convergence_rates[x])
        recommendations.append(f"For reliability: Use {reliability_rec}")
    
    # Safety recommendation
    if 'risk_assessment' in stability_results:
        safety_rec = min(risk_assessment.keys(), 
                        key=lambda x: risk_assessment[x]['overall_risk_score'])
        recommendations.append(f"For safety: Use {safety_rec}")
    
    # Classical vs Quantum comparison
    classical_gens = [g for g in generator_counts.keys() if 'classical' in g]
    quantum_gens = [g for g in generator_counts.keys() if 'quantum' in g]
    
    if classical_gens and quantum_gens:
        classical_performance = sum(generator_counts.get(g, 0) for g in classical_gens)
        quantum_performance = sum(generator_counts.get(g, 0) for g in quantum_gens)
        
        if classical_performance > quantum_performance:
            recommendations.append("Classical generators show superior overall performance")
        elif quantum_performance > classical_performance:
            recommendations.append("Quantum generators show competitive performance")
        else:
            recommendations.append("Classical and quantum generators show comparable performance")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    print("\n" + "="*80)
    print("Analysis complete! Check the generated reports for detailed results.")
    print("="*80)


def main():
    """Main function that orchestrates the complete analysis workflow."""
    print("GaussGAN Statistical Analysis Framework - Complete Usage Example")
    print("=" * 70)
    print()
    
    try:
        # Step 1: Setup
        output_dir = setup_analysis_environment()
        
        # Step 2: Generate experimental data
        experimental_runs = generate_comprehensive_experimental_data(n_runs_per_generator=25)
        
        # Step 3: Statistical analysis
        statistical_results = perform_statistical_analysis(experimental_runs, output_dir)
        
        # Step 4: Convergence analysis
        convergence_results = perform_convergence_analysis(experimental_runs, output_dir)
        
        # Step 5: Stability analysis
        stability_results = perform_stability_analysis(experimental_runs, output_dir)
        
        # Step 6: Generate reports
        generated_files = generate_comprehensive_reports(
            statistical_results, convergence_results, stability_results, output_dir
        )
        
        # Step 7: Analyze and summarize
        analyze_and_summarize_results(statistical_results, convergence_results, stability_results)
        
        # Success message
        print(f"\n✅ Complete analysis finished successfully!")
        print(f"📁 All results saved to: {output_dir}")
        print(f"📊 {len(generated_files)} report types generated")
        print(f"🔬 {len(experimental_runs)} experimental runs analyzed")
        
        return {
            'statistical_results': statistical_results,
            'convergence_results': convergence_results,
            'stability_results': stability_results,
            'generated_files': generated_files,
            'output_directory': output_dir
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the complete analysis
    results = main()
    
    if results:
        print("\n🎉 Example completed successfully!")
        print("\nThis demonstrates the complete workflow for:")
        print("  • Multi-run experiment management")
        print("  • Comprehensive statistical analysis") 
        print("  • Convergence pattern analysis")
        print("  • Stability and risk assessment")
        print("  • Automated report generation")
        print("  • Publication-ready visualizations")
        
        print(f"\n📂 Check {results['output_directory']} for all generated files and reports.")
    else:
        print("\n❌ Example failed. Check the logs for details.")