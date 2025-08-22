#!/usr/bin/env python3
"""
Convergence Speed Experiment Runner for GaussGAN

This script runs comparative experiments to measure convergence speed
between different generator types (classical vs quantum).

Usage:
    python convergence_experiment.py --generators classical_normal quantum_samples --epochs 100
    python convergence_experiment.py --comparative-analysis  # After experiments are complete
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any

import yaml

from source.training_integration import ConvergenceTrainingManager


def run_single_experiment(generator_type: str, max_epochs: int = 50, config_overrides: Dict = None) -> bool:
    """
    Run a single convergence experiment for a specific generator type.
    
    Args:
        generator_type: Type of generator to test
        max_epochs: Number of epochs to train
        config_overrides: Additional configuration overrides
        
    Returns:
        True if experiment completed successfully
    """
    print(f"\n{'='*60}")
    print(f"Running convergence experiment: {generator_type}")
    print(f"Max epochs: {max_epochs}")
    print(f"{'='*60}")
    
    # Prepare command arguments
    cmd_args = [
        "uv", "run", "python", "main.py",
        "--generator_type", generator_type,
        "--max_epochs", str(max_epochs),
        "--stage", "train"
    ]
    
    # Add any additional overrides as command line arguments
    if config_overrides:
        for key, value in config_overrides.items():
            if key not in ['generator_type', 'max_epochs', 'stage']:
                cmd_args.extend([f"--{key}", str(value)])
    
    start_time = time.time()
    
    try:
        # Run the training experiment
        result = subprocess.run(cmd_args, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            duration = time.time() - start_time
            print(f"✅ Experiment completed successfully in {duration:.1f}s")
            print("Training output (last 20 lines):")
            output_lines = result.stdout.split('\n')
            for line in output_lines[-20:]:
                if line.strip():
                    print(f"  {line}")
            return True
        else:
            print(f"❌ Experiment failed with return code {result.returncode}")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ Experiment timed out after 1 hour")
        return False
    except Exception as e:
        print(f"❌ Experiment failed with exception: {e}")
        return False


def run_comparative_experiment(generator_types: List[str], max_epochs: int = 50, 
                             config_overrides: Dict = None) -> Dict[str, bool]:
    """
    Run comparative experiments across multiple generator types.
    
    Args:
        generator_types: List of generator types to test
        max_epochs: Number of epochs for each experiment
        config_overrides: Additional configuration overrides
        
    Returns:
        Dictionary mapping generator types to success status
    """
    print(f"\n🚀 Starting Comparative Convergence Experiment")
    print(f"Generator Types: {generator_types}")
    print(f"Training Epochs: {max_epochs}")
    print(f"Expected Duration: ~{len(generator_types) * max_epochs * 2 // 60} minutes")
    
    results = {}
    
    for i, gen_type in enumerate(generator_types, 1):
        print(f"\n📊 Experiment {i}/{len(generator_types)}")
        success = run_single_experiment(gen_type, max_epochs, config_overrides)
        results[gen_type] = success
        
        if not success:
            print(f"⚠️  Experiment for {gen_type} failed, but continuing with others...")
    
    # Summary
    successful = [gen for gen, success in results.items() if success]
    failed = [gen for gen, success in results.items() if not success]
    
    print(f"\n📋 Experiment Summary:")
    print(f"✅ Successful: {successful}")
    if failed:
        print(f"❌ Failed: {failed}")
    
    return results


def analyze_completed_experiments(results_dir: str = "docs/convergence_analysis") -> Dict[str, Any]:
    """
    Analyze results from completed convergence experiments.
    
    Args:
        results_dir: Directory containing experiment results
        
    Returns:
        Analysis results dictionary
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return {}
    
    # Find all convergence analysis files
    analysis_files = list(results_path.glob("convergence_analysis_*.json"))
    
    if not analysis_files:
        print(f"❌ No convergence analysis files found in {results_dir}")
        return {}
    
    print(f"📊 Found {len(analysis_files)} analysis files")
    
    # Load and organize results by generator type
    results_by_generator = {}
    
    for file_path in analysis_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            gen_type = data.get('generator_type', 'unknown')
            timestamp = data.get('analysis_timestamp', 'unknown')
            
            if gen_type not in results_by_generator:
                results_by_generator[gen_type] = []
            
            results_by_generator[gen_type].append({
                'file_path': str(file_path),
                'timestamp': timestamp,
                'data': data
            })
            
        except Exception as e:
            print(f"⚠️  Error loading {file_path}: {e}")
    
    # Use most recent results for each generator
    latest_results = {}
    for gen_type, runs in results_by_generator.items():
        # Sort by timestamp and take most recent
        latest_run = sorted(runs, key=lambda x: x['timestamp'])[-1]
        latest_results[gen_type] = latest_run['data']
    
    print(f"📈 Analyzing results for: {list(latest_results.keys())}")
    
    # Generate comparative analysis
    if len(latest_results) < 2:
        print("⚠️  Need at least 2 generator types for comparison")
        return latest_results
    
    # Create comparison report
    comparison_report = generate_comparison_report(latest_results)
    
    # Save comprehensive report
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = results_path / f"comparative_analysis_{timestamp}.json"
    
    comprehensive_results = {
        "individual_results": latest_results,
        "comparative_analysis": comparison_report,
        "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(report_path, 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    
    print(f"💾 Comprehensive analysis saved to: {report_path}")
    
    return comprehensive_results


def generate_comparison_report(results: Dict[str, Dict]) -> Dict[str, Any]:
    """Generate a detailed comparison report from experiment results."""
    
    report = {
        "summary": {},
        "metric_comparisons": {},
        "convergence_speed_ranking": [],
        "efficiency_analysis": {}
    }
    
    # Extract key metrics for comparison
    generator_metrics = {}
    
    for gen_type, data in results.items():
        analysis = data.get('convergence_analysis', {})
        metrics = analysis.get('metrics', {})
        efficiency = analysis.get('training_efficiency', {})
        
        generator_metrics[gen_type] = {
            'total_epochs': efficiency.get('total_epochs', 0),
            'convergence_rate': analysis.get('summary', {}).get('overall_convergence_rate', 0),
            'avg_epoch_time': efficiency.get('average_epoch_time', 0),
            'total_training_time': efficiency.get('total_training_time', 0),
            'metrics_converged': analysis.get('summary', {}).get('metrics_converged', 0),
            'convergence_details': {}
        }
        
        # Extract convergence details for each metric
        for metric_name, metric_data in metrics.items():
            if 'ValidationStep_FakeData_' in metric_name:
                clean_name = metric_name.replace('ValidationStep_FakeData_', '')
                generator_metrics[gen_type]['convergence_details'][clean_name] = {
                    'converged': metric_data.get('converged', False),
                    'epochs_to_convergence': metric_data.get('epochs_to_convergence'),
                    'improvement_rate': metric_data.get('improvement_rate', 0),
                    'final_value': metric_data.get('final_value'),
                    'convergence_pattern': metric_data.get('convergence_pattern', 'unknown')
                }
    
    # Create rankings
    
    # 1. Overall convergence rate ranking
    convergence_ranking = sorted(
        generator_metrics.items(),
        key=lambda x: x[1]['convergence_rate'],
        reverse=True
    )
    
    # 2. Speed ranking (epochs to convergence, weighted by number of converged metrics)
    speed_ranking = []
    for gen_type, metrics in generator_metrics.items():
        # Calculate average epochs to convergence for converged metrics only
        converged_epochs = [
            detail['epochs_to_convergence'] 
            for detail in metrics['convergence_details'].values()
            if detail['converged'] and detail['epochs_to_convergence'] is not None
        ]
        
        if converged_epochs:
            avg_convergence_epochs = sum(converged_epochs) / len(converged_epochs)
            # Weight by proportion of metrics that converged
            weighted_score = avg_convergence_epochs * (1 / (metrics['convergence_rate'] + 0.01))
        else:
            weighted_score = float('inf')  # No convergence
        
        speed_ranking.append((gen_type, weighted_score, avg_convergence_epochs if converged_epochs else None))
    
    speed_ranking.sort(key=lambda x: x[1])
    
    # 3. Efficiency ranking (convergence per unit time)
    efficiency_ranking = []
    for gen_type, metrics in generator_metrics.items():
        if metrics['total_training_time'] > 0:
            efficiency_score = metrics['convergence_rate'] / (metrics['total_training_time'] / 3600)  # per hour
        else:
            efficiency_score = 0
        efficiency_ranking.append((gen_type, efficiency_score))
    
    efficiency_ranking.sort(key=lambda x: x[1], reverse=True)
    
    # Fill report
    report["convergence_speed_ranking"] = [
        {
            "rank": i + 1,
            "generator_type": gen_type,
            "weighted_score": score,
            "avg_epochs_to_convergence": avg_epochs
        }
        for i, (gen_type, score, avg_epochs) in enumerate(speed_ranking)
    ]
    
    report["efficiency_analysis"] = {
        "convergence_rate_ranking": [
            {"rank": i + 1, "generator_type": gen_type, "convergence_rate": metrics['convergence_rate']}
            for i, (gen_type, metrics) in enumerate(convergence_ranking)
        ],
        "time_efficiency_ranking": [
            {"rank": i + 1, "generator_type": gen_type, "efficiency_score": score}
            for i, (gen_type, score) in enumerate(efficiency_ranking)
        ]
    }
    
    # Detailed metric comparisons
    all_metric_names = set()
    for gen_type, metrics in generator_metrics.items():
        all_metric_names.update(metrics['convergence_details'].keys())
    
    for metric_name in all_metric_names:
        metric_comparison = {}
        for gen_type, metrics in generator_metrics.items():
            if metric_name in metrics['convergence_details']:
                metric_comparison[gen_type] = metrics['convergence_details'][metric_name]
        
        # Determine winner for this metric
        converged_generators = [gen for gen, data in metric_comparison.items() if data['converged']]
        if converged_generators:
            # Winner is fastest to converge
            winner = min(converged_generators, 
                        key=lambda gen: metric_comparison[gen]['epochs_to_convergence'] or float('inf'))
        else:
            # No convergence, compare final values
            if metric_name.lower() == 'kldivergence':
                # Lower is better for KL divergence
                winner = min(metric_comparison.keys(), 
                           key=lambda gen: metric_comparison[gen]['final_value'] or float('inf'))
            else:
                # Higher is better for others
                winner = max(metric_comparison.keys(), 
                           key=lambda gen: metric_comparison[gen]['final_value'] or float('-inf'))
        
        metric_comparison['winner'] = winner
        report["metric_comparisons"][metric_name] = metric_comparison
    
    # Overall summary
    report["summary"] = {
        "total_generators_tested": len(generator_metrics),
        "best_overall_convergence": convergence_ranking[0][0] if convergence_ranking else None,
        "fastest_convergence": speed_ranking[0][0] if speed_ranking[0][1] != float('inf') else None,
        "most_time_efficient": efficiency_ranking[0][0] if efficiency_ranking else None
    }
    
    return report


def print_analysis_summary(analysis_results: Dict[str, Any]):
    """Print a human-readable summary of the analysis results."""
    
    if not analysis_results:
        print("❌ No analysis results to display")
        return
    
    comparative = analysis_results.get('comparative_analysis', {})
    if not comparative:
        print("⚠️  No comparative analysis available")
        return
    
    summary = comparative.get('summary', {})
    speed_ranking = comparative.get('convergence_speed_ranking', [])
    efficiency = comparative.get('efficiency_analysis', {})
    metric_comparisons = comparative.get('metric_comparisons', {})
    
    print(f"\n🏆 CONVERGENCE EXPERIMENT RESULTS")
    print(f"{'='*60}")
    
    # Overall winners
    print(f"\n🥇 Overall Winners:")
    if summary.get('best_overall_convergence'):
        print(f"  Best Convergence Rate: {summary['best_overall_convergence']}")
    if summary.get('fastest_convergence'):
        print(f"  Fastest Convergence: {summary['fastest_convergence']}")
    if summary.get('most_time_efficient'):
        print(f"  Most Time Efficient: {summary['most_time_efficient']}")
    
    # Speed ranking
    print(f"\n⚡ Convergence Speed Ranking:")
    for entry in speed_ranking[:3]:  # Top 3
        gen_type = entry['generator_type']
        avg_epochs = entry['avg_epochs_to_convergence']
        if avg_epochs:
            print(f"  {entry['rank']}. {gen_type}: {avg_epochs:.1f} epochs average")
        else:
            print(f"  {entry['rank']}. {gen_type}: No convergence achieved")
    
    # Metric-by-metric winners
    print(f"\n📊 Metric-by-Metric Winners:")
    for metric_name, comparison in metric_comparisons.items():
        winner = comparison.get('winner', 'unknown')
        print(f"  {metric_name}: {winner}")
    
    # Efficiency insights
    conv_ranking = efficiency.get('convergence_rate_ranking', [])
    if conv_ranking:
        print(f"\n💯 Convergence Rate Ranking:")
        for entry in conv_ranking:
            rate = entry['convergence_rate']
            print(f"  {entry['rank']}. {entry['generator_type']}: {rate:.1%}")
    
    print(f"\n{'='*60}")


def main():
    """Main entry point for convergence experiments."""
    parser = argparse.ArgumentParser(description="Run GaussGAN convergence speed experiments")
    
    parser.add_argument(
        '--generators', 
        nargs='+', 
        default=['classical_normal', 'quantum_samples'],
        help='List of generator types to test'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=50,
        help='Number of training epochs per experiment'
    )
    
    parser.add_argument(
        '--comparative-analysis',
        action='store_true',
        help='Run analysis on completed experiments (no new training)'
    )
    
    parser.add_argument(
        '--results-dir',
        default='docs/convergence_analysis',
        help='Directory containing experiment results'
    )
    
    parser.add_argument(
        '--config-overrides',
        default='{}',
        help='JSON string of configuration overrides'
    )
    
    args = parser.parse_args()
    
    # Parse config overrides
    try:
        config_overrides = json.loads(args.config_overrides)
    except json.JSONDecodeError:
        print("❌ Invalid JSON in config-overrides")
        return 1
    
    if args.comparative_analysis:
        # Only run analysis, no new experiments
        print("🔍 Running comparative analysis on existing results...")
        results = analyze_completed_experiments(args.results_dir)
        print_analysis_summary(results)
    else:
        # Run experiments
        print(f"🚀 Starting convergence experiments")
        experiment_results = run_comparative_experiment(
            args.generators, 
            args.epochs,
            config_overrides
        )
        
        # Check if any experiments succeeded
        successful_experiments = [gen for gen, success in experiment_results.items() if success]
        
        if successful_experiments:
            print(f"\n🔍 Running analysis on completed experiments...")
            time.sleep(2)  # Give time for files to be written
            
            analysis_results = analyze_completed_experiments(args.results_dir)
            print_analysis_summary(analysis_results)
        else:
            print("❌ No experiments completed successfully")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())