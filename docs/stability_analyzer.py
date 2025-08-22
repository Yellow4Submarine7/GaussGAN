"""
Generation Stability Analysis System for GaussGAN

This module provides comprehensive stability analysis tools for comparing
quantum vs classical generators across multiple training runs.

Key Features:
- Multi-run stability tracking with different random seeds
- Statistical analysis of key metrics (mean, std, CV, IQR)
- Outlier detection and robustness assessment
- Generator type comparison
- Stability scoring and reliability metrics
- Confidence interval calculation
- Missing data handling

Usage:
    analyzer = StabilityAnalyzer()
    analyzer.add_experiment_results(results_dict)
    stability_report = analyzer.generate_stability_report()
    
Author: Created for GaussGAN quantum vs classical generator comparison
"""

import os
import json
import pickle
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import iqr, zscore
import mlflow
from mlflow.tracking import MlflowClient


@dataclass
class ExperimentResult:
    """Data structure for storing individual experiment results."""
    run_id: str
    generator_type: str
    seed: int
    final_kl_divergence: Optional[float] = None
    final_log_likelihood: Optional[float] = None
    final_is_positive: Optional[float] = None
    training_time: Optional[float] = None
    convergence_epoch: Optional[int] = None
    max_epochs: Optional[int] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    metrics_history: Optional[Dict[str, List[float]]] = None
    error_message: Optional[str] = None
    status: str = "completed"  # completed, failed, incomplete


@dataclass  
class StabilityMetrics:
    """Statistical metrics for stability analysis."""
    mean: float
    std: float
    median: float
    min_val: float
    max_val: float
    q25: float
    q75: float
    iqr: float
    coefficient_of_variation: float
    outlier_count: int
    outlier_indices: List[int]
    confidence_interval_95: Tuple[float, float]
    stability_score: float  # Custom composite score [0-1]
    sample_size: int


class StabilityAnalyzer:
    """
    Comprehensive stability analysis system for GaussGAN experiments.
    
    This class aggregates results from multiple training runs with different
    random seeds and provides detailed stability analysis comparing quantum
    vs classical generators.
    """
    
    def __init__(self, experiment_name: str = "GaussGAN-manual", 
                 stability_threshold: float = 0.15):
        """
        Initialize the StabilityAnalyzer.
        
        Args:
            experiment_name: Name of the MLflow experiment to analyze
            stability_threshold: CV threshold for considering results "stable"
        """
        self.experiment_name = experiment_name
        self.stability_threshold = stability_threshold
        self.experiments: List[ExperimentResult] = []
        self.stability_results: Dict[str, Dict[str, StabilityMetrics]] = {}
        
        # Initialize MLflow client
        self.client = MlflowClient()
        
    def add_experiment_result(self, result: ExperimentResult) -> None:
        """Add a single experiment result to the analyzer."""
        self.experiments.append(result)
        
    def add_experiment_results(self, results: List[ExperimentResult]) -> None:
        """Add multiple experiment results to the analyzer."""
        self.experiments.extend(results)
        
    def load_from_mlflow(self, generator_types: Optional[List[str]] = None,
                        max_runs: Optional[int] = None) -> int:
        """
        Load experiment results from MLflow tracking.
        
        Args:
            generator_types: Filter by specific generator types
            max_runs: Maximum number of runs to load
            
        Returns:
            Number of experiments loaded
        """
        try:
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            if not experiment:
                warnings.warn(f"Experiment '{self.experiment_name}' not found")
                return 0
                
            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=max_runs or 1000,
                order_by=["start_time DESC"]
            )
            
            loaded_count = 0
            for run in runs:
                try:
                    result = self._parse_mlflow_run(run)
                    if result and (not generator_types or 
                                 result.generator_type in generator_types):
                        self.add_experiment_result(result)
                        loaded_count += 1
                except Exception as e:
                    warnings.warn(f"Failed to parse run {run.info.run_id}: {e}")
                    
            return loaded_count
            
        except Exception as e:
            warnings.warn(f"Failed to load from MLflow: {e}")
            return 0
            
    def _parse_mlflow_run(self, run) -> Optional[ExperimentResult]:
        """Parse MLflow run data into ExperimentResult."""
        try:
            params = run.data.params
            metrics = run.data.metrics
            
            # Extract basic information
            generator_type = params.get('generator_type', 'unknown')
            seed = int(params.get('seed', 0))
            run_id = run.info.run_id
            
            # Extract final metrics (latest values)
            final_kl = metrics.get('ValidationStep_FakeData_KLDivergence')
            final_ll = metrics.get('ValidationStep_FakeData_LogLikelihood')
            final_pos = metrics.get('ValidationStep_FakeData_IsPositive')
            
            # Calculate training time
            start_time = run.info.start_time
            end_time = run.info.end_time
            training_time = None
            if start_time and end_time:
                training_time = (end_time - start_time) / 1000.0  # Convert to seconds
            
            # Find convergence epoch (when KL divergence stabilized)
            convergence_epoch = None
            max_epochs = int(params.get('max_epochs', 0))
            
            # Status check
            status = run.info.status.lower()
            if status != 'finished':
                status = 'failed' if status == 'failed' else 'incomplete'
            else:
                status = 'completed'
            
            return ExperimentResult(
                run_id=run_id,
                generator_type=generator_type,
                seed=seed,
                final_kl_divergence=final_kl,
                final_log_likelihood=final_ll,
                final_is_positive=final_pos,
                training_time=training_time,
                convergence_epoch=convergence_epoch,
                max_epochs=max_epochs,
                hyperparameters=dict(params),
                status=status
            )
            
        except Exception as e:
            warnings.warn(f"Error parsing MLflow run: {e}")
            return None
            
    def calculate_stability_metrics(self, values: List[float], 
                                  metric_name: str = "") -> StabilityMetrics:
        """
        Calculate comprehensive stability metrics for a set of values.
        
        Args:
            values: List of metric values from different runs
            metric_name: Name of the metric for context
            
        Returns:
            StabilityMetrics object with all statistical measures
        """
        # Filter out None values and convert to numpy array
        valid_values = np.array([v for v in values if v is not None and not np.isnan(v)])
        
        if len(valid_values) == 0:
            # Return empty metrics if no valid values
            return StabilityMetrics(
                mean=np.nan, std=np.nan, median=np.nan,
                min_val=np.nan, max_val=np.nan, q25=np.nan, q75=np.nan,
                iqr=np.nan, coefficient_of_variation=np.nan,
                outlier_count=0, outlier_indices=[],
                confidence_interval_95=(np.nan, np.nan),
                stability_score=0.0, sample_size=0
            )
        
        # Basic statistics
        mean_val = np.mean(valid_values)
        std_val = np.std(valid_values, ddof=1) if len(valid_values) > 1 else 0.0
        median_val = np.median(valid_values)
        min_val = np.min(valid_values)
        max_val = np.max(valid_values)
        q25 = np.percentile(valid_values, 25)
        q75 = np.percentile(valid_values, 75)
        iqr_val = q75 - q25
        
        # Coefficient of variation (relative variability)
        cv = abs(std_val / mean_val) if mean_val != 0 else np.inf
        
        # Outlier detection using IQR method
        outlier_threshold = 1.5
        lower_bound = q25 - outlier_threshold * iqr_val
        upper_bound = q75 + outlier_threshold * iqr_val
        outlier_mask = (valid_values < lower_bound) | (valid_values > upper_bound)
        outlier_indices = np.where(outlier_mask)[0].tolist()
        outlier_count = len(outlier_indices)
        
        # Confidence interval (95%)
        if len(valid_values) > 1:
            sem = stats.sem(valid_values)  # Standard error of mean
            ci_95 = stats.t.interval(0.95, len(valid_values)-1, 
                                   loc=mean_val, scale=sem)
        else:
            ci_95 = (mean_val, mean_val)
        
        # Custom stability score (0-1, higher is more stable)
        stability_score = self._calculate_stability_score(
            cv, outlier_count, len(valid_values), metric_name
        )
        
        return StabilityMetrics(
            mean=mean_val,
            std=std_val,
            median=median_val,
            min_val=min_val,
            max_val=max_val,
            q25=q25,
            q75=q75,
            iqr=iqr_val,
            coefficient_of_variation=cv,
            outlier_count=outlier_count,
            outlier_indices=outlier_indices,
            confidence_interval_95=ci_95,
            stability_score=stability_score,
            sample_size=len(valid_values)
        )
    
    def _calculate_stability_score(self, cv: float, outlier_count: int, 
                                 sample_size: int, metric_name: str) -> float:
        """
        Calculate a composite stability score (0-1, higher = more stable).
        
        This combines multiple factors:
        - Coefficient of variation (lower is better)
        - Outlier rate (lower is better)  
        - Sample size (higher is better, with diminishing returns)
        """
        if sample_size == 0:
            return 0.0
        
        # CV component (0-1, higher is more stable)
        if np.isinf(cv) or np.isnan(cv):
            cv_score = 0.0
        else:
            # Use sigmoid-like function: stable if CV < threshold
            cv_score = 1.0 / (1.0 + np.exp(5.0 * (cv - self.stability_threshold)))
        
        # Outlier rate component (0-1, higher is more stable)
        outlier_rate = outlier_count / sample_size
        outlier_score = 1.0 / (1.0 + np.exp(10.0 * (outlier_rate - 0.1)))
        
        # Sample size component (0-1, higher is more stable)
        # Sigmoid function with inflection at 10 samples
        size_score = 2.0 / (1.0 + np.exp(-0.5 * (sample_size - 10))) - 1.0
        size_score = max(0.0, min(1.0, size_score))
        
        # Weighted combination
        stability_score = 0.5 * cv_score + 0.3 * outlier_score + 0.2 * size_score
        
        return stability_score
    
    def analyze_stability(self) -> Dict[str, Dict[str, StabilityMetrics]]:
        """
        Perform comprehensive stability analysis across all experiments.
        
        Returns:
            Dictionary with generator types as keys and metric stability as values
        """
        if not self.experiments:
            warnings.warn("No experiments loaded for analysis")
            return {}
        
        # Group experiments by generator type
        generator_groups = {}
        for exp in self.experiments:
            if exp.generator_type not in generator_groups:
                generator_groups[exp.generator_type] = []
            generator_groups[exp.generator_type].append(exp)
        
        # Analyze stability for each generator type
        self.stability_results = {}
        
        for generator_type, experiments in generator_groups.items():
            print(f"Analyzing {len(experiments)} experiments for {generator_type}")
            
            # Extract metrics for stability analysis
            metrics_data = {
                'final_kl_divergence': [exp.final_kl_divergence for exp in experiments],
                'final_log_likelihood': [exp.final_log_likelihood for exp in experiments],
                'final_is_positive': [exp.final_is_positive for exp in experiments],
                'training_time': [exp.training_time for exp in experiments],
                'convergence_epoch': [exp.convergence_epoch for exp in experiments]
            }
            
            # Calculate stability metrics for each metric
            generator_stability = {}
            for metric_name, values in metrics_data.items():
                generator_stability[metric_name] = self.calculate_stability_metrics(
                    values, metric_name
                )
            
            self.stability_results[generator_type] = generator_stability
        
        return self.stability_results
    
    def generate_stability_report(self, output_dir: str = "docs/stability_analysis") -> Dict:
        """
        Generate comprehensive stability analysis report.
        
        Args:
            output_dir: Directory to save the report and visualizations
            
        Returns:
            Dictionary containing the complete stability analysis
        """
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Perform analysis if not done yet
        if not self.stability_results:
            self.analyze_stability()
        
        # Generate summary statistics
        report = {
            'experiment_summary': self._generate_experiment_summary(),
            'stability_analysis': self._convert_stability_metrics_to_dict(),
            'generator_comparison': self._compare_generators(),
            'recommendations': self._generate_recommendations(),
            'metadata': {
                'analysis_date': pd.Timestamp.now().isoformat(),
                'total_experiments': len(self.experiments),
                'stability_threshold': self.stability_threshold,
                'analyzer_version': '1.0.0'
            }
        }
        
        # Save report as JSON
        report_file = Path(output_dir) / 'stability_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate visualizations
        self.create_stability_plots(output_dir)
        
        # Generate markdown report
        self._generate_markdown_report(report, output_dir)
        
        print(f"Stability analysis report saved to {output_dir}")
        
        return report
    
    def _generate_experiment_summary(self) -> Dict:
        """Generate summary of all experiments."""
        if not self.experiments:
            return {}
        
        # Count by generator type and status
        summary = {
            'total_experiments': len(self.experiments),
            'by_generator_type': {},
            'by_status': {},
            'seed_range': [],
            'time_range': []
        }
        
        generator_counts = {}
        status_counts = {}
        seeds = []
        
        for exp in self.experiments:
            # Generator type counts
            if exp.generator_type not in generator_counts:
                generator_counts[exp.generator_type] = 0
            generator_counts[exp.generator_type] += 1
            
            # Status counts
            if exp.status not in status_counts:
                status_counts[exp.status] = 0
            status_counts[exp.status] += 1
            
            # Seeds
            if exp.seed is not None:
                seeds.append(exp.seed)
        
        summary['by_generator_type'] = generator_counts
        summary['by_status'] = status_counts
        
        if seeds:
            summary['seed_range'] = [min(seeds), max(seeds)]
        
        return summary
    
    def _convert_stability_metrics_to_dict(self) -> Dict:
        """Convert stability metrics to JSON-serializable format."""
        result = {}
        for generator_type, metrics in self.stability_results.items():
            result[generator_type] = {}
            for metric_name, stability_metrics in metrics.items():
                result[generator_type][metric_name] = asdict(stability_metrics)
        return result
    
    def _compare_generators(self) -> Dict:
        """Generate comparison between different generator types."""
        if len(self.stability_results) < 2:
            return {"warning": "Need at least 2 generator types for comparison"}
        
        comparison = {}
        
        # Compare stability scores for each metric
        for metric_name in ['final_kl_divergence', 'final_log_likelihood', 
                           'final_is_positive', 'training_time']:
            metric_comparison = {}
            
            for generator_type in self.stability_results.keys():
                if metric_name in self.stability_results[generator_type]:
                    stability = self.stability_results[generator_type][metric_name]
                    metric_comparison[generator_type] = {
                        'stability_score': stability.stability_score,
                        'cv': stability.coefficient_of_variation,
                        'mean': stability.mean,
                        'sample_size': stability.sample_size
                    }
            
            if len(metric_comparison) > 1:
                # Find most stable generator for this metric
                most_stable = max(metric_comparison.keys(), 
                                key=lambda x: metric_comparison[x]['stability_score'])
                
                metric_comparison['most_stable'] = most_stable
                comparison[metric_name] = metric_comparison
        
        return comparison
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on stability analysis."""
        recommendations = []
        
        if not self.stability_results:
            return ["Run stability analysis first"]
        
        # Check sample sizes
        min_sample_size = 10
        for generator_type, metrics in self.stability_results.items():
            for metric_name, stability in metrics.items():
                if stability.sample_size < min_sample_size:
                    recommendations.append(
                        f"Increase sample size for {generator_type} "
                        f"({metric_name}): currently {stability.sample_size}, "
                        f"recommend at least {min_sample_size}"
                    )
        
        # Check stability scores
        for generator_type, metrics in self.stability_results.items():
            for metric_name, stability in metrics.items():
                if stability.stability_score < 0.7:
                    recommendations.append(
                        f"{generator_type} shows low stability for {metric_name} "
                        f"(score: {stability.stability_score:.2f}). Consider "
                        f"hyperparameter tuning or more training runs."
                    )
        
        # Compare generators
        comparison = self._compare_generators()
        if 'final_kl_divergence' in comparison:
            kl_comparison = comparison['final_kl_divergence']
            most_stable_kl = kl_comparison.get('most_stable')
            if most_stable_kl:
                recommendations.append(
                    f"For KL divergence stability, {most_stable_kl} performs best. "
                    f"Consider using this generator type for production."
                )
        
        if not recommendations:
            recommendations.append("All generators show good stability.")
        
        return recommendations
    
    def create_stability_plots(self, output_dir: str) -> None:
        """Create comprehensive stability visualization plots."""
        if not self.stability_results:
            return
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Stability Score Comparison
        self._plot_stability_scores(output_dir)
        
        # 2. Distribution plots for key metrics  
        self._plot_metric_distributions(output_dir)
        
        # 3. Coefficient of Variation comparison
        self._plot_cv_comparison(output_dir)
        
        # 4. Box plots for outlier visualization
        self._plot_boxplots(output_dir)
        
        # 5. Time series of individual runs (if applicable)
        self._plot_individual_runs(output_dir)
        
        print(f"Stability plots saved to {output_dir}")
    
    def _plot_stability_scores(self, output_dir: str) -> None:
        """Plot stability scores comparison across generators and metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Generator Stability Score Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['final_kl_divergence', 'final_log_likelihood', 
                  'final_is_positive', 'training_time']
        
        for idx, metric_name in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            generator_types = []
            stability_scores = []
            cvs = []
            
            for generator_type, metrics_dict in self.stability_results.items():
                if metric_name in metrics_dict:
                    stability = metrics_dict[metric_name]
                    generator_types.append(generator_type)
                    stability_scores.append(stability.stability_score)
                    cvs.append(stability.coefficient_of_variation)
            
            if generator_types:
                # Create bar plot with color based on stability score
                bars = ax.bar(generator_types, stability_scores, 
                            color=plt.cm.RdYlGn([s for s in stability_scores]))
                
                ax.set_title(f'{metric_name.replace("_", " ").title()}')
                ax.set_ylabel('Stability Score')
                ax.set_ylim(0, 1.1)
                ax.grid(axis='y', alpha=0.3)
                
                # Add CV values as text on bars
                for bar, cv in zip(bars, cvs):
                    height = bar.get_height()
                    if not np.isnan(cv) and not np.isinf(cv):
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                               f'CV: {cv:.3f}', ha='center', va='bottom', fontsize=8)
                
                # Add stability threshold line
                ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, 
                          label='Good Stability (0.7)')
                ax.legend()
            
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'stability_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metric_distributions(self, output_dir: str) -> None:
        """Plot distributions of metric values across generator types."""
        metrics = ['final_kl_divergence', 'final_log_likelihood', 'final_is_positive']
        
        for metric_name in metrics:
            plt.figure(figsize=(12, 8))
            
            data_for_plot = []
            labels = []
            
            for generator_type in self.stability_results.keys():
                # Collect raw values for this generator and metric
                generator_experiments = [exp for exp in self.experiments 
                                       if exp.generator_type == generator_type]
                
                values = []
                for exp in generator_experiments:
                    value = getattr(exp, metric_name)
                    if value is not None and not np.isnan(value):
                        values.append(value)
                
                if values:
                    data_for_plot.append(values)
                    labels.append(f'{generator_type}\n(n={len(values)})')
            
            if data_for_plot:
                # Create violin plot
                plt.violinplot(data_for_plot, positions=range(len(labels)), 
                             showmeans=True, showmedians=True)
                
                plt.xticks(range(len(labels)), labels)
                plt.ylabel(metric_name.replace('_', ' ').title())
                plt.title(f'Distribution of {metric_name.replace("_", " ").title()} Across Generator Types')
                plt.grid(axis='y', alpha=0.3)
                
                # Add statistical annotations
                for i, values in enumerate(data_for_plot):
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    plt.text(i, plt.ylim()[1] * 0.95, 
                            f'μ={mean_val:.3f}\nσ={std_val:.3f}', 
                            ha='center', va='top', fontsize=9, 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(Path(output_dir) / f'{metric_name}_distributions.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_cv_comparison(self, output_dir: str) -> None:
        """Plot coefficient of variation comparison."""
        plt.figure(figsize=(12, 8))
        
        metrics = ['final_kl_divergence', 'final_log_likelihood', 
                  'final_is_positive', 'training_time']
        generator_types = list(self.stability_results.keys())
        
        x = np.arange(len(metrics))
        width = 0.35 if len(generator_types) <= 2 else 0.25
        
        for i, generator_type in enumerate(generator_types):
            cvs = []
            for metric_name in metrics:
                if metric_name in self.stability_results[generator_type]:
                    cv = self.stability_results[generator_type][metric_name].coefficient_of_variation
                    cvs.append(cv if not (np.isnan(cv) or np.isinf(cv)) else 0)
                else:
                    cvs.append(0)
            
            offset = (i - len(generator_types)/2 + 0.5) * width
            plt.bar(x + offset, cvs, width, label=generator_type, alpha=0.8)
        
        plt.xlabel('Metrics')
        plt.ylabel('Coefficient of Variation')
        plt.title('Coefficient of Variation Comparison (Lower = More Stable)')
        plt.xticks(x, [m.replace('_', ' ').title() for m in metrics], rotation=45)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Add stability threshold line
        plt.axhline(y=self.stability_threshold, color='red', linestyle='--', 
                   label=f'Stability Threshold ({self.stability_threshold})')
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'cv_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_boxplots(self, output_dir: str) -> None:
        """Create box plots to visualize outliers."""
        metrics = ['final_kl_divergence', 'final_log_likelihood', 'final_is_positive']
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 6))
        fig.suptitle('Outlier Analysis (Box Plots)', fontsize=16, fontweight='bold')
        
        for idx, metric_name in enumerate(metrics):
            ax = axes[idx] if len(metrics) > 1 else axes
            
            data_for_boxplot = []
            labels = []
            
            for generator_type in self.stability_results.keys():
                # Collect values for box plot
                generator_experiments = [exp for exp in self.experiments 
                                       if exp.generator_type == generator_type]
                values = []
                for exp in generator_experiments:
                    value = getattr(exp, metric_name)
                    if value is not None and not np.isnan(value):
                        values.append(value)
                
                if values:
                    data_for_boxplot.append(values)
                    
                    # Get outlier info from stability metrics
                    stability = self.stability_results[generator_type][metric_name]
                    labels.append(f'{generator_type}\n({stability.outlier_count} outliers)')
            
            if data_for_boxplot:
                box_plot = ax.boxplot(data_for_boxplot, labels=labels, patch_artist=True)
                
                # Color boxes based on outlier rate
                colors = plt.cm.RdYlGn([1 - len(outliers)/len(data) 
                                      for data, outliers in zip(data_for_boxplot, 
                                      [self.stability_results[gen][metric_name].outlier_indices 
                                       for gen in self.stability_results.keys()])])
                
                for patch, color in zip(box_plot['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            
            ax.set_title(metric_name.replace('_', ' ').title())
            ax.grid(axis='y', alpha=0.3)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'outlier_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_individual_runs(self, output_dir: str) -> None:
        """Plot individual run results to show variability."""
        metrics = ['final_kl_divergence', 'final_log_likelihood']
        
        for metric_name in metrics:
            plt.figure(figsize=(14, 8))
            
            for i, generator_type in enumerate(self.stability_results.keys()):
                generator_experiments = [exp for exp in self.experiments 
                                       if exp.generator_type == generator_type]
                
                seeds = []
                values = []
                
                for exp in generator_experiments:
                    value = getattr(exp, metric_name)
                    if value is not None and not np.isnan(value):
                        seeds.append(exp.seed)
                        values.append(value)
                
                if seeds and values:
                    # Sort by seed for better visualization
                    sorted_data = sorted(zip(seeds, values))
                    seeds, values = zip(*sorted_data)
                    
                    plt.scatter([s + i*0.1 for s in seeds], values, 
                              label=generator_type, alpha=0.7, s=50)
                    
                    # Add mean line
                    mean_val = np.mean(values)
                    plt.axhline(y=mean_val, color=plt.gca().lines[-1].get_color(), 
                              linestyle='--', alpha=0.5)
            
            plt.xlabel('Random Seed')
            plt.ylabel(metric_name.replace('_', ' ').title())
            plt.title(f'Individual Run Results: {metric_name.replace("_", " ").title()}')
            plt.legend()
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(Path(output_dir) / f'{metric_name}_individual_runs.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_markdown_report(self, report: Dict, output_dir: str) -> None:
        """Generate a markdown report file."""
        md_content = f"""# GaussGAN Stability Analysis Report

Generated on: {report['metadata']['analysis_date']}

## Executive Summary

Total Experiments Analyzed: {report['experiment_summary']['total_experiments']}
Stability Threshold: {report['metadata']['stability_threshold']}

### Generator Types Analyzed:
"""
        
        for gen_type, count in report['experiment_summary']['by_generator_type'].items():
            md_content += f"- **{gen_type}**: {count} experiments\n"
        
        md_content += "\n## Stability Analysis Results\n\n"
        
        for generator_type, metrics in report['stability_analysis'].items():
            md_content += f"### {generator_type}\n\n"
            
            for metric_name, stability_data in metrics.items():
                md_content += f"#### {metric_name.replace('_', ' ').title()}\n\n"
                md_content += f"- **Stability Score**: {stability_data['stability_score']:.3f}\n"
                md_content += f"- **Mean**: {stability_data['mean']:.3f}\n"
                md_content += f"- **Standard Deviation**: {stability_data['std']:.3f}\n"
                md_content += f"- **Coefficient of Variation**: {stability_data['coefficient_of_variation']:.3f}\n"
                md_content += f"- **Outliers**: {stability_data['outlier_count']}/{stability_data['sample_size']}\n"
                md_content += f"- **95% Confidence Interval**: ({stability_data['confidence_interval_95'][0]:.3f}, {stability_data['confidence_interval_95'][1]:.3f})\n\n"
        
        md_content += "## Generator Comparison\n\n"
        
        if 'generator_comparison' in report:
            for metric_name, comparison in report['generator_comparison'].items():
                if 'most_stable' in comparison:
                    md_content += f"**{metric_name.replace('_', ' ').title()}**: Most stable generator is **{comparison['most_stable']}**\n\n"
        
        md_content += "## Recommendations\n\n"
        
        for i, rec in enumerate(report['recommendations'], 1):
            md_content += f"{i}. {rec}\n"
        
        md_content += "\n## Visualizations\n\n"
        md_content += "The following plots have been generated:\n\n"
        md_content += "- `stability_scores.png`: Stability score comparison\n"
        md_content += "- `cv_comparison.png`: Coefficient of variation comparison\n" 
        md_content += "- `outlier_analysis.png`: Box plots showing outliers\n"
        md_content += "- `*_distributions.png`: Distribution plots for key metrics\n"
        md_content += "- `*_individual_runs.png`: Individual run scatter plots\n"
        
        # Save markdown report
        with open(Path(output_dir) / 'stability_report.md', 'w') as f:
            f.write(md_content)
    
    def export_results(self, output_file: str = "docs/stability_analysis/stability_results.json") -> None:
        """Export all results to JSON file for further analysis."""
        export_data = {
            'experiments': [asdict(exp) for exp in self.experiments],
            'stability_results': self._convert_stability_metrics_to_dict(),
            'metadata': {
                'export_date': pd.Timestamp.now().isoformat(),
                'total_experiments': len(self.experiments),
                'stability_threshold': self.stability_threshold
            }
        }
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Results exported to {output_file}")
    
    def get_stability_summary(self) -> pd.DataFrame:
        """Get a concise DataFrame summary of stability results."""
        if not self.stability_results:
            self.analyze_stability()
        
        rows = []
        for generator_type, metrics in self.stability_results.items():
            for metric_name, stability in metrics.items():
                rows.append({
                    'generator_type': generator_type,
                    'metric': metric_name,
                    'stability_score': stability.stability_score,
                    'mean': stability.mean,
                    'std': stability.std,
                    'cv': stability.coefficient_of_variation,
                    'outlier_rate': stability.outlier_count / stability.sample_size if stability.sample_size > 0 else 0,
                    'sample_size': stability.sample_size
                })
        
        return pd.DataFrame(rows)


def run_stability_analysis_example():
    """Example function showing how to use the StabilityAnalyzer."""
    
    print("=== GaussGAN Stability Analysis Example ===")
    
    # Initialize analyzer
    analyzer = StabilityAnalyzer(
        experiment_name="GaussGAN-manual",
        stability_threshold=0.15
    )
    
    # Load from MLflow (if available)
    print("Loading experiments from MLflow...")
    loaded_count = analyzer.load_from_mlflow(
        generator_types=['classical_normal', 'classical_uniform', 
                        'quantum_samples', 'quantum_shadows']
    )
    print(f"Loaded {loaded_count} experiments")
    
    if loaded_count == 0:
        print("No experiments found. Creating synthetic example data...")
        # Create some example data for demonstration
        example_results = _create_example_data()
        analyzer.add_experiment_results(example_results)
    
    # Generate comprehensive stability report
    print("Generating stability analysis report...")
    report = analyzer.generate_stability_report("docs/stability_analysis")
    
    # Display summary
    summary_df = analyzer.get_stability_summary()
    print("\nStability Summary:")
    print(summary_df.to_string(index=False))
    
    print("\nAnalysis complete! Check docs/stability_analysis/ for detailed results.")


def _create_example_data() -> List[ExperimentResult]:
    """Create synthetic example data for demonstration purposes."""
    
    np.random.seed(42)
    
    example_results = []
    generator_types = ['classical_normal', 'quantum_samples']
    
    for gen_type in generator_types:
        # Generate 15 example runs per generator type
        for i in range(15):
            seed = 40 + i
            
            # Simulate different stability characteristics
            if gen_type == 'classical_normal':
                # More stable classical generator
                kl_base = 0.05
                kl_noise = 0.01
                ll_base = -2.5
                ll_noise = 0.2
                training_time_base = 120
                training_time_noise = 20
            else:
                # Less stable quantum generator
                kl_base = 0.08
                kl_noise = 0.03
                ll_base = -2.8
                ll_noise = 0.4
                training_time_base = 180
                training_time_noise = 40
            
            # Add some outliers occasionally
            outlier_factor = 1.0
            if np.random.random() < 0.1:  # 10% chance of outlier
                outlier_factor = np.random.uniform(1.5, 3.0)
            
            result = ExperimentResult(
                run_id=f"example_{gen_type}_{i}",
                generator_type=gen_type,
                seed=seed,
                final_kl_divergence=max(0, np.random.normal(kl_base, kl_noise) * outlier_factor),
                final_log_likelihood=np.random.normal(ll_base, ll_noise) / outlier_factor,
                final_is_positive=np.random.uniform(0.7, 1.0),
                training_time=max(60, np.random.normal(training_time_base, training_time_noise)),
                convergence_epoch=np.random.randint(15, 45),
                max_epochs=50,
                status='completed'
            )
            
            example_results.append(result)
    
    return example_results


if __name__ == "__main__":
    run_stability_analysis_example()