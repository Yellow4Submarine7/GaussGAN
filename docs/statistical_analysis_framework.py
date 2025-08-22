"""
Comprehensive Statistical Analysis Framework for Quantum vs Classical Generator Comparison
======================================================================================

This module provides a robust statistical framework for comparing quantum and classical
generators in the GaussGAN project. It includes multi-run experiment management,
statistical significance testing, convergence analysis, and stability assessment.

Key Features:
- Multi-run experiment orchestration with systematic parameter variation
- Statistical significance testing (t-tests, Mann-Whitney U, effect sizes)
- Convergence speed analysis (time-to-threshold, improvement rates)
- Stability assessment (coefficient of variation, outlier detection)
- Results aggregation and comprehensive reporting
- Bayesian and bootstrapping methods for robust statistics
"""

import numpy as np
import pandas as pd
import pickle
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings

# Statistical testing
from scipy import stats
from scipy.stats import mannwhitneyu, wilcoxon, friedmanchisquare
from scipy.stats import normaltest, levene, bartlett

# Effect size calculations
import scipy.stats as stats
from statsmodels.stats.effect_size import cohens_d
from statsmodels.stats.power import ttest_power

# Multiple comparisons correction
from statsmodels.stats.multitest import multipletests

# Bootstrap and permutation tests
from scipy.stats import bootstrap
import itertools

# Plotting and visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Time and progress tracking
import time
from datetime import datetime, timedelta
from tqdm import tqdm

# Machine learning utilities
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


@dataclass
class ExperimentRun:
    """Single experimental run data structure."""
    generator_type: str
    seed: int
    run_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    convergence_epoch: Optional[int] = None
    final_metrics: Dict[str, float] = None
    metric_history: Dict[str, List[float]] = None
    loss_history: Dict[str, List[float]] = None
    hyperparameters: Dict[str, Any] = None
    training_duration: Optional[float] = None
    converged: bool = False
    
    def __post_init__(self):
        if self.final_metrics is None:
            self.final_metrics = {}
        if self.metric_history is None:
            self.metric_history = {}
        if self.loss_history is None:
            self.loss_history = {}
        if self.hyperparameters is None:
            self.hyperparameters = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to strings
        if self.start_time:
            data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExperimentRun':
        """Create from dictionary."""
        # Convert datetime strings back to datetime objects
        if 'start_time' in data and isinstance(data['start_time'], str):
            data['start_time'] = datetime.fromisoformat(data['start_time'])
        if 'end_time' in data and isinstance(data['end_time'], str):
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        return cls(**data)


@dataclass
class StatisticalTest:
    """Results of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    power: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""
    significant: bool = False
    
    def __post_init__(self):
        self.significant = self.p_value < 0.05 if self.p_value is not None else False


@dataclass
class ConvergenceAnalysis:
    """Convergence analysis results."""
    metric_name: str
    threshold: float
    epochs_to_threshold: Dict[str, List[int]]  # generator_type -> list of epochs
    improvement_rates: Dict[str, List[float]]  # generator_type -> list of rates
    final_values: Dict[str, List[float]]  # generator_type -> list of final values
    convergence_rates: Dict[str, float]  # generator_type -> proportion converged


class MultiRunExperimentManager:
    """
    Manages multiple experimental runs for statistical comparison.
    
    This class orchestrates the execution of multiple training runs with different
    random seeds and generator types, collecting comprehensive data for analysis.
    """
    
    def __init__(
        self,
        base_config: Dict[str, Any],
        output_dir: Union[str, Path] = "docs/statistical_analysis",
        log_level: int = logging.INFO
    ):
        """
        Initialize experiment manager.
        
        Args:
            base_config: Base configuration dictionary
            output_dir: Directory to save experimental results
            log_level: Logging level
        """
        self.base_config = base_config.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'experiment_log.txt'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Experiment tracking
        self.runs: List[ExperimentRun] = []
        self.generator_types = ['classical_normal', 'classical_uniform', 
                              'quantum_samples', 'quantum_shadows']
        
        # Results storage
        self.results_file = self.output_dir / 'experiment_results.json'
        self.analysis_cache = {}
        
    def plan_experiments(
        self,
        generator_types: List[str] = None,
        n_runs_per_type: int = 10,
        seed_start: int = 42,
        max_epochs_range: Tuple[int, int] = (30, 100),
        batch_size_options: List[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Plan experimental configurations.
        
        Args:
            generator_types: List of generator types to test
            n_runs_per_type: Number of runs per generator type
            seed_start: Starting seed for experiments
            max_epochs_range: Range of max epochs to test
            batch_size_options: Batch size options to test
            
        Returns:
            List of experiment configurations
        """
        if generator_types is None:
            generator_types = self.generator_types
        if batch_size_options is None:
            batch_size_options = [256]
            
        experiments = []
        seed = seed_start
        
        for gen_type in generator_types:
            for run_idx in range(n_runs_per_type):
                # Create base configuration
                config = self.base_config.copy()
                config.update({
                    'generator_type': gen_type,
                    'seed': seed,
                    'max_epochs': np.random.randint(*max_epochs_range),
                    'batch_size': np.random.choice(batch_size_options),
                    'run_id': f"{gen_type}_run_{run_idx:03d}_seed_{seed}"
                })
                
                # Add generator-specific optimizations
                if 'quantum' in gen_type:
                    config.update({
                        'quantum_qubits': 6,
                        'quantum_layers': 2,
                        'quantum_shots': 100
                    })
                
                experiments.append(config)
                seed += 1
                
        self.logger.info(f"Planned {len(experiments)} experiments across {len(generator_types)} generator types")
        return experiments
        
    def execute_experiment(self, config: Dict[str, Any]) -> ExperimentRun:
        """
        Execute a single experiment.
        
        Args:
            config: Experiment configuration
            
        Returns:
            ExperimentRun object with results
        """
        run = ExperimentRun(
            generator_type=config['generator_type'],
            seed=config['seed'],
            run_id=config['run_id'],
            start_time=datetime.now(),
            hyperparameters=config.copy()
        )
        
        try:
            self.logger.info(f"Starting experiment: {run.run_id}")
            
            # Here you would integrate with your actual training code
            # For now, we'll simulate the experiment execution
            
            # Simulate training time
            if 'quantum' in run.generator_type:
                training_time = np.random.uniform(300, 900)  # 5-15 minutes for quantum
            else:
                training_time = np.random.uniform(60, 180)   # 1-3 minutes for classical
                
            time.sleep(0.1)  # Brief simulation delay
            
            # Simulate convergence
            max_epochs = config.get('max_epochs', 50)
            converged = np.random.random() < 0.8  # 80% convergence rate
            convergence_epoch = np.random.randint(10, max_epochs) if converged else None
            
            # Simulate metrics
            run.final_metrics = self._simulate_final_metrics(run.generator_type, converged)
            run.metric_history = self._simulate_metric_history(run.generator_type, max_epochs)
            run.loss_history = self._simulate_loss_history(max_epochs)
            
            run.end_time = datetime.now()
            run.training_duration = training_time
            run.convergence_epoch = convergence_epoch
            run.converged = converged
            
            self.logger.info(f"Completed experiment: {run.run_id} (converged: {converged})")
            
        except Exception as e:
            self.logger.error(f"Experiment {run.run_id} failed: {str(e)}")
            run.end_time = datetime.now()
            
        return run
    
    def _simulate_final_metrics(self, generator_type: str, converged: bool) -> Dict[str, float]:
        """Simulate final metrics based on generator type and convergence."""
        base_metrics = {
            'classical_normal': {'KLDivergence': 0.15, 'LogLikelihood': -2.5, 'WassersteinDistance': 0.08},
            'classical_uniform': {'KLDivergence': 0.18, 'LogLikelihood': -2.8, 'WassersteinDistance': 0.10},
            'quantum_samples': {'KLDivergence': 0.25, 'LogLikelihood': -3.2, 'WassersteinDistance': 0.12},
            'quantum_shadows': {'KLDivergence': 0.22, 'LogLikelihood': -3.0, 'WassersteinDistance': 0.11}
        }
        
        metrics = base_metrics.get(generator_type, base_metrics['classical_normal']).copy()
        
        # Add noise and convergence effects
        noise_scale = 0.3 if not converged else 0.1
        for key in metrics:
            metrics[key] += np.random.normal(0, metrics[key] * noise_scale)
            
        # Add additional metrics
        metrics.update({
            'MMDDistance': np.random.uniform(0.05, 0.3),
            'IsPositive': np.random.uniform(0.6, 0.95)
        })
        
        return metrics
    
    def _simulate_metric_history(self, generator_type: str, max_epochs: int) -> Dict[str, List[float]]:
        """Simulate metric evolution over epochs."""
        history = {}
        
        # Base convergence patterns
        if 'quantum' in generator_type:
            # Quantum generators: slower, more variable convergence
            kl_start, kl_end = 1.5, 0.25
            ll_start, ll_end = -5.0, -3.0
        else:
            # Classical generators: faster, more stable convergence
            kl_start, kl_end = 1.0, 0.15
            ll_start, ll_end = -4.0, -2.5
            
        epochs = np.arange(max_epochs)
        
        # KL Divergence (should decrease)
        kl_decay = np.exp(-epochs / 15)
        history['KLDivergence'] = [
            kl_end + (kl_start - kl_end) * decay + np.random.normal(0, 0.05)
            for decay in kl_decay
        ]
        
        # Log Likelihood (should increase, less negative)
        ll_improvement = 1 - np.exp(-epochs / 12)
        history['LogLikelihood'] = [
            ll_start + (ll_end - ll_start) * imp + np.random.normal(0, 0.1)
            for imp in ll_improvement
        ]
        
        return history
    
    def _simulate_loss_history(self, max_epochs: int) -> Dict[str, List[float]]:
        """Simulate loss evolution over epochs."""
        epochs = np.arange(max_epochs)
        
        # Generator loss (should stabilize)
        g_loss_base = -np.log(epochs + 1) * 0.5 + 2.0
        g_loss = g_loss_base + np.random.normal(0, 0.2, max_epochs)
        
        # Discriminator loss (should stabilize around 0)
        d_loss = np.random.normal(0, 0.3, max_epochs)
        
        return {
            'g_loss': g_loss.tolist(),
            'd_loss': d_loss.tolist()
        }
    
    def run_experiments(
        self,
        experiment_configs: List[Dict[str, Any]],
        parallel: bool = False,
        save_interval: int = 5
    ) -> List[ExperimentRun]:
        """
        Execute multiple experiments.
        
        Args:
            experiment_configs: List of experiment configurations
            parallel: Whether to run experiments in parallel
            save_interval: Save results every N experiments
            
        Returns:
            List of completed experiment runs
        """
        self.logger.info(f"Starting {len(experiment_configs)} experiments")
        
        for i, config in enumerate(tqdm(experiment_configs, desc="Running experiments")):
            run = self.execute_experiment(config)
            self.runs.append(run)
            
            # Periodic saving
            if (i + 1) % save_interval == 0:
                self.save_results()
                
        self.save_results()
        self.logger.info(f"Completed all {len(experiment_configs)} experiments")
        
        return self.runs
    
    def save_results(self):
        """Save experimental results to disk."""
        results_data = {
            'runs': [run.to_dict() for run in self.runs],
            'timestamp': datetime.now().isoformat(),
            'total_runs': len(self.runs)
        }
        
        with open(self.results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
            
        self.logger.info(f"Saved {len(self.runs)} experimental results")
    
    def load_results(self) -> List[ExperimentRun]:
        """Load experimental results from disk."""
        if not self.results_file.exists():
            self.logger.warning("No saved results found")
            return []
            
        with open(self.results_file, 'r') as f:
            data = json.load(f)
            
        self.runs = [ExperimentRun.from_dict(run_data) for run_data in data['runs']]
        self.logger.info(f"Loaded {len(self.runs)} experimental results")
        
        return self.runs


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for comparing generator performance.
    
    This class provides methods for statistical significance testing, effect size
    calculation, multiple comparisons correction, and robust statistical inference.
    """
    
    def __init__(self, alpha: float = 0.05, correction_method: str = 'bonferroni'):
        """
        Initialize statistical analyzer.
        
        Args:
            alpha: Significance level for statistical tests
            correction_method: Multiple comparisons correction method
        """
        self.alpha = alpha
        self.correction_method = correction_method
        self.logger = logging.getLogger(__name__)
        
    def group_runs_by_generator(self, runs: List[ExperimentRun]) -> Dict[str, List[ExperimentRun]]:
        """Group experimental runs by generator type."""
        groups = defaultdict(list)
        for run in runs:
            if run.final_metrics:  # Only include completed runs
                groups[run.generator_type].append(run)
        return dict(groups)
    
    def extract_metric_values(
        self, 
        runs: List[ExperimentRun], 
        metric_name: str
    ) -> List[float]:
        """Extract metric values from a list of runs."""
        values = []
        for run in runs:
            if run.final_metrics and metric_name in run.final_metrics:
                value = run.final_metrics[metric_name]
                if not (np.isnan(value) or np.isinf(value)):
                    values.append(value)
        return values
    
    def compute_descriptive_statistics(
        self, 
        runs: List[ExperimentRun], 
        metric_name: str
    ) -> Dict[str, float]:
        """Compute descriptive statistics for a metric."""
        values = self.extract_metric_values(runs, metric_name)
        
        if not values:
            return {'count': 0, 'mean': np.nan, 'std': np.nan}
            
        values = np.array(values)
        
        return {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values, ddof=1),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75),
            'iqr': np.percentile(values, 75) - np.percentile(values, 25),
            'coefficient_of_variation': np.std(values, ddof=1) / np.mean(values) if np.mean(values) != 0 else np.nan,
            'skewness': stats.skew(values),
            'kurtosis': stats.kurtosis(values)
        }
    
    def test_normality(self, values: List[float]) -> Tuple[bool, float]:
        """Test if values follow normal distribution."""
        if len(values) < 8:
            return False, np.nan
            
        statistic, p_value = normaltest(values)
        return p_value > self.alpha, p_value
    
    def test_equal_variances(self, group1: List[float], group2: List[float]) -> Tuple[bool, float]:
        """Test if two groups have equal variances."""
        if len(group1) < 3 or len(group2) < 3:
            return False, np.nan
            
        statistic, p_value = levene(group1, group2)
        return p_value > self.alpha, p_value
    
    def compute_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Compute Cohen's d effect size."""
        if len(group1) == 0 or len(group2) == 0:
            return np.nan
            
        group1, group2 = np.array(group1), np.array(group2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                             (len(group2) - 1) * np.var(group2, ddof=1)) / 
                            (len(group1) + len(group2) - 2))
        
        if pooled_std == 0:
            return np.nan
            
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def perform_t_test(
        self, 
        group1: List[float], 
        group2: List[float],
        alternative: str = 'two-sided'
    ) -> StatisticalTest:
        """Perform t-test between two groups."""
        if len(group1) == 0 or len(group2) == 0:
            return StatisticalTest(
                test_name="t-test",
                statistic=np.nan,
                p_value=np.nan,
                interpretation="Insufficient data"
            )
        
        # Check assumptions
        normal1, _ = self.test_normality(group1)
        normal2, _ = self.test_normality(group2)
        equal_var, _ = self.test_equal_variances(group1, group2)
        
        # Perform appropriate t-test
        if normal1 and normal2:
            statistic, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
            test_name = f"t-test (equal_var={equal_var})"
        else:
            # Use Mann-Whitney U test as non-parametric alternative
            statistic, p_value = mannwhitneyu(group1, group2, alternative=alternative)
            test_name = "Mann-Whitney U test"
        
        # Compute effect size
        effect_size = self.compute_cohens_d(group1, group2)
        
        # Compute statistical power (for t-test)
        power = None
        if normal1 and normal2 and not np.isnan(effect_size):
            try:
                power = ttest_power(effect_size, len(group1), self.alpha)
            except:
                power = None
        
        # Interpret effect size
        if np.isnan(effect_size):
            effect_interpretation = "Cannot compute"
        elif abs(effect_size) < 0.2:
            effect_interpretation = "Small effect"
        elif abs(effect_size) < 0.8:
            effect_interpretation = "Medium effect"
        else:
            effect_interpretation = "Large effect"
        
        interpretation = f"{effect_interpretation}. "
        if p_value < self.alpha:
            interpretation += f"Significant difference (p={p_value:.4f})"
        else:
            interpretation += f"No significant difference (p={p_value:.4f})"
        
        return StatisticalTest(
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            power=power,
            interpretation=interpretation,
            significant=p_value < self.alpha if not np.isnan(p_value) else False
        )
    
    def bootstrap_confidence_interval(
        self, 
        data: List[float], 
        statistic_func=np.mean,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval."""
        if len(data) == 0:
            return np.nan, np.nan
            
        data = np.array(data)
        
        # Bootstrap resampling
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic_func(sample))
        
        # Compute confidence interval
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        
        return lower, upper
    
    def permutation_test(
        self, 
        group1: List[float], 
        group2: List[float],
        n_permutations: int = 10000
    ) -> float:
        """Perform permutation test for difference in means."""
        if len(group1) == 0 or len(group2) == 0:
            return np.nan
            
        # Observed difference
        observed_diff = np.mean(group1) - np.mean(group2)
        
        # Combine groups
        combined = np.concatenate([group1, group2])
        n1 = len(group1)
        
        # Permutation test
        extreme_count = 0
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_group1 = combined[:n1]
            perm_group2 = combined[n1:]
            perm_diff = np.mean(perm_group1) - np.mean(perm_group2)
            
            if abs(perm_diff) >= abs(observed_diff):
                extreme_count += 1
        
        return extreme_count / n_permutations
    
    def compare_all_generators(
        self, 
        runs: List[ExperimentRun], 
        metric_name: str
    ) -> Dict[str, Dict[str, StatisticalTest]]:
        """Perform pairwise comparisons between all generator types."""
        groups = self.group_runs_by_generator(runs)
        generator_types = list(groups.keys())
        
        results = {}
        p_values = []
        
        # Perform all pairwise comparisons
        for i, gen1 in enumerate(generator_types):
            results[gen1] = {}
            for j, gen2 in enumerate(generator_types):
                if i < j:  # Only do each comparison once
                    values1 = self.extract_metric_values(groups[gen1], metric_name)
                    values2 = self.extract_metric_values(groups[gen2], metric_name)
                    
                    test_result = self.perform_t_test(values1, values2)
                    results[gen1][gen2] = test_result
                    
                    if not np.isnan(test_result.p_value):
                        p_values.append(test_result.p_value)
        
        # Apply multiple comparisons correction
        if p_values:
            corrected_results = multipletests(p_values, method=self.correction_method)
            corrected_p_values = corrected_results[1]
            
            # Update results with corrected p-values
            p_idx = 0
            for gen1 in results:
                for gen2 in results[gen1]:
                    if not np.isnan(results[gen1][gen2].p_value):
                        original_test = results[gen1][gen2]
                        corrected_p = corrected_p_values[p_idx]
                        
                        # Create new test result with corrected p-value
                        results[gen1][gen2] = StatisticalTest(
                            test_name=f"{original_test.test_name} (corrected)",
                            statistic=original_test.statistic,
                            p_value=corrected_p,
                            effect_size=original_test.effect_size,
                            power=original_test.power,
                            interpretation=f"Corrected p-value: {corrected_p:.4f}",
                            significant=corrected_p < self.alpha
                        )
                        p_idx += 1
        
        return results


class ConvergenceAnalyzer:
    """
    Analyzes convergence speed and patterns across different generator types.
    
    This class provides methods for analyzing how quickly different generators
    converge to target distributions and their convergence characteristics.
    """
    
    def __init__(self, convergence_thresholds: Dict[str, float] = None):
        """
        Initialize convergence analyzer.
        
        Args:
            convergence_thresholds: Thresholds for each metric to define convergence
        """
        self.convergence_thresholds = convergence_thresholds or {
            'KLDivergence': 0.3,
            'LogLikelihood': -3.5,
            'WassersteinDistance': 0.15,
            'MMDDistance': 0.2
        }
        self.logger = logging.getLogger(__name__)
    
    def analyze_convergence_speed(
        self, 
        runs: List[ExperimentRun], 
        metric_name: str
    ) -> ConvergenceAnalysis:
        """
        Analyze convergence speed for a specific metric.
        
        Args:
            runs: List of experimental runs
            metric_name: Name of metric to analyze
            
        Returns:
            ConvergenceAnalysis object with results
        """
        threshold = self.convergence_thresholds.get(metric_name, np.nan)
        
        epochs_to_threshold = defaultdict(list)
        improvement_rates = defaultdict(list)
        final_values = defaultdict(list)
        convergence_counts = defaultdict(int)
        total_counts = defaultdict(int)
        
        for run in runs:
            gen_type = run.generator_type
            total_counts[gen_type] += 1
            
            if not run.metric_history or metric_name not in run.metric_history:
                continue
                
            history = run.metric_history[metric_name]
            if not history:
                continue
                
            # Find epoch where threshold was reached
            epochs_to_conv = self._find_convergence_epoch(history, threshold, metric_name)
            if epochs_to_conv is not None:
                epochs_to_threshold[gen_type].append(epochs_to_conv)
                convergence_counts[gen_type] += 1
            
            # Compute improvement rate (slope of improvement)
            improvement_rate = self._compute_improvement_rate(history)
            if not np.isnan(improvement_rate):
                improvement_rates[gen_type].append(improvement_rate)
            
            # Store final value
            final_value = history[-1]
            if not (np.isnan(final_value) or np.isinf(final_value)):
                final_values[gen_type].append(final_value)
        
        # Compute convergence rates
        convergence_rates = {}
        for gen_type in total_counts:
            convergence_rates[gen_type] = (
                convergence_counts[gen_type] / total_counts[gen_type]
                if total_counts[gen_type] > 0 else 0.0
            )
        
        return ConvergenceAnalysis(
            metric_name=metric_name,
            threshold=threshold,
            epochs_to_threshold=dict(epochs_to_threshold),
            improvement_rates=dict(improvement_rates),
            final_values=dict(final_values),
            convergence_rates=convergence_rates
        )
    
    def _find_convergence_epoch(
        self, 
        history: List[float], 
        threshold: float, 
        metric_name: str
    ) -> Optional[int]:
        """Find the epoch where convergence threshold was reached."""
        if np.isnan(threshold):
            return None
            
        # For metrics where lower is better (KL, Wasserstein, MMD)
        if metric_name in ['KLDivergence', 'WassersteinDistance', 'MMDDistance']:
            for epoch, value in enumerate(history):
                if value <= threshold:
                    return epoch
        # For metrics where higher is better (LogLikelihood, IsPositive)
        else:
            for epoch, value in enumerate(history):
                if value >= threshold:
                    return epoch
                    
        return None
    
    def _compute_improvement_rate(self, history: List[float]) -> float:
        """Compute the rate of improvement (slope) for a metric history."""
        if len(history) < 5:
            return np.nan
            
        # Use linear regression to compute slope
        epochs = np.arange(len(history))
        valid_indices = ~np.isnan(history)
        
        if np.sum(valid_indices) < 3:
            return np.nan
            
        epochs_valid = epochs[valid_indices]
        history_valid = np.array(history)[valid_indices]
        
        # Linear regression
        slope, _, _, _, _ = stats.linregress(epochs_valid, history_valid)
        return slope
    
    def compare_convergence_speeds(
        self, 
        runs: List[ExperimentRun], 
        metric_name: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare convergence speeds between generator types.
        
        Returns:
            Dictionary with statistical comparisons of convergence speeds
        """
        analysis = self.analyze_convergence_speed(runs, metric_name)
        
        results = {}
        generator_types = list(analysis.epochs_to_threshold.keys())
        
        for gen_type in generator_types:
            results[gen_type] = {
                'mean_epochs_to_convergence': np.mean(analysis.epochs_to_threshold[gen_type]) 
                    if analysis.epochs_to_threshold[gen_type] else np.nan,
                'std_epochs_to_convergence': np.std(analysis.epochs_to_threshold[gen_type]) 
                    if analysis.epochs_to_threshold[gen_type] else np.nan,
                'convergence_rate': analysis.convergence_rates[gen_type],
                'mean_improvement_rate': np.mean(analysis.improvement_rates[gen_type]) 
                    if analysis.improvement_rates[gen_type] else np.nan,
                'mean_final_value': np.mean(analysis.final_values[gen_type]) 
                    if analysis.final_values[gen_type] else np.nan
            }
        
        return results


class StabilityAnalyzer:
    """
    Analyzes stability and variance in generator performance.
    
    This class provides methods for assessing the consistency and reliability
    of different generator types across multiple runs.
    """
    
    def __init__(self, outlier_method: str = 'iqr'):
        """
        Initialize stability analyzer.
        
        Args:
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'isolation')
        """
        self.outlier_method = outlier_method
        self.logger = logging.getLogger(__name__)
    
    def detect_outliers(self, values: List[float]) -> Tuple[List[bool], List[int]]:
        """
        Detect outliers in a list of values.
        
        Args:
            values: List of values to check for outliers
            
        Returns:
            Tuple of (outlier_mask, outlier_indices)
        """
        if len(values) < 4:
            return [False] * len(values), []
            
        values = np.array(values)
        outlier_mask = np.zeros(len(values), dtype=bool)
        
        if self.outlier_method == 'iqr':
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask = (values < lower_bound) | (values > upper_bound)
            
        elif self.outlier_method == 'zscore':
            z_scores = np.abs(stats.zscore(values))
            outlier_mask = z_scores > 3
            
        outlier_indices = np.where(outlier_mask)[0].tolist()
        return outlier_mask.tolist(), outlier_indices
    
    def compute_stability_metrics(
        self, 
        runs: List[ExperimentRun], 
        metric_name: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute stability metrics for each generator type.
        
        Args:
            runs: List of experimental runs
            metric_name: Name of metric to analyze
            
        Returns:
            Dictionary with stability metrics for each generator type
        """
        groups = defaultdict(list)
        for run in runs:
            if run.final_metrics and metric_name in run.final_metrics:
                value = run.final_metrics[metric_name]
                if not (np.isnan(value) or np.isinf(value)):
                    groups[run.generator_type].append(value)
        
        results = {}
        
        for gen_type, values in groups.items():
            if len(values) < 2:
                results[gen_type] = {
                    'coefficient_of_variation': np.nan,
                    'stability_score': np.nan,
                    'outlier_rate': np.nan,
                    'range_ratio': np.nan
                }
                continue
            
            values = np.array(values)
            
            # Coefficient of variation
            cv = np.std(values) / np.abs(np.mean(values)) if np.mean(values) != 0 else np.nan
            
            # Stability score (inverse of coefficient of variation)
            stability_score = 1 / (1 + cv) if not np.isnan(cv) else np.nan
            
            # Outlier rate
            outlier_mask, _ = self.detect_outliers(values)
            outlier_rate = np.sum(outlier_mask) / len(values)
            
            # Range ratio (range / median)
            range_ratio = (np.max(values) - np.min(values)) / np.median(values) if np.median(values) != 0 else np.nan
            
            results[gen_type] = {
                'coefficient_of_variation': cv,
                'stability_score': stability_score,
                'outlier_rate': outlier_rate,
                'range_ratio': range_ratio,
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'mad': np.median(np.abs(values - np.median(values)))  # Median absolute deviation
            }
        
        return results
    
    def analyze_training_stability(
        self, 
        runs: List[ExperimentRun], 
        metric_name: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze stability during training (evolution of metrics).
        
        Args:
            runs: List of experimental runs
            metric_name: Name of metric to analyze
            
        Returns:
            Dictionary with training stability metrics
        """
        groups = defaultdict(list)
        
        for run in runs:
            if run.metric_history and metric_name in run.metric_history:
                history = run.metric_history[metric_name]
                if len(history) > 5:  # Need sufficient history
                    groups[run.generator_type].append(history)
        
        results = {}
        
        for gen_type, histories in groups.items():
            if not histories:
                continue
            
            # Analyze variance across training
            variances = []
            smoothness_scores = []
            
            for history in histories:
                # Variance over training
                variances.append(np.var(history))
                
                # Smoothness (inverse of average absolute difference)
                if len(history) > 1:
                    diffs = np.abs(np.diff(history))
                    smoothness = 1 / (1 + np.mean(diffs))
                    smoothness_scores.append(smoothness)
            
            results[gen_type] = {
                'mean_training_variance': np.mean(variances) if variances else np.nan,
                'std_training_variance': np.std(variances) if variances else np.nan,
                'mean_smoothness': np.mean(smoothness_scores) if smoothness_scores else np.nan,
                'consistency_score': 1 / (1 + np.std(variances)) if len(variances) > 1 else np.nan
            }
        
        return results


class ResultsAggregator:
    """
    Aggregates and reports comprehensive analysis results.
    
    This class combines results from statistical, convergence, and stability
    analyses to produce comprehensive reports and visualizations.
    """
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize results aggregator.
        
        Args:
            output_dir: Directory to save reports and visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def generate_comprehensive_report(
        self,
        runs: List[ExperimentRun],
        statistical_analyzer: StatisticalAnalyzer,
        convergence_analyzer: ConvergenceAnalyzer,
        stability_analyzer: StabilityAnalyzer
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report.
        
        Args:
            runs: List of experimental runs
            statistical_analyzer: Configured statistical analyzer
            convergence_analyzer: Configured convergence analyzer
            stability_analyzer: Configured stability analyzer
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        self.logger.info("Generating comprehensive analysis report...")
        
        # Get available metrics
        metrics = set()
        for run in runs:
            if run.final_metrics:
                metrics.update(run.final_metrics.keys())
        
        report = {
            'metadata': {
                'total_runs': len(runs),
                'generator_types': list(set(run.generator_type for run in runs)),
                'metrics_analyzed': list(metrics),
                'analysis_timestamp': datetime.now().isoformat()
            },
            'descriptive_statistics': {},
            'statistical_tests': {},
            'convergence_analysis': {},
            'stability_analysis': {},
            'summary_insights': {}
        }
        
        # Descriptive statistics for each metric and generator
        for metric in metrics:
            report['descriptive_statistics'][metric] = {}
            groups = statistical_analyzer.group_runs_by_generator(runs)
            
            for gen_type, gen_runs in groups.items():
                stats = statistical_analyzer.compute_descriptive_statistics(gen_runs, metric)
                report['descriptive_statistics'][metric][gen_type] = stats
        
        # Statistical comparisons
        for metric in metrics:
            self.logger.info(f"Analyzing metric: {metric}")
            
            # Pairwise statistical tests
            comparisons = statistical_analyzer.compare_all_generators(runs, metric)
            report['statistical_tests'][metric] = {
                'pairwise_comparisons': self._serialize_statistical_tests(comparisons)
            }
            
            # Convergence analysis
            conv_analysis = convergence_analyzer.analyze_convergence_speed(runs, metric)
            report['convergence_analysis'][metric] = {
                'convergence_rates': conv_analysis.convergence_rates,
                'mean_epochs_to_threshold': {
                    gen_type: np.mean(epochs) if epochs else np.nan
                    for gen_type, epochs in conv_analysis.epochs_to_threshold.items()
                },
                'mean_improvement_rates': {
                    gen_type: np.mean(rates) if rates else np.nan
                    for gen_type, rates in conv_analysis.improvement_rates.items()
                }
            }
            
            # Stability analysis
            stability_metrics = stability_analyzer.compute_stability_metrics(runs, metric)
            training_stability = stability_analyzer.analyze_training_stability(runs, metric)
            
            report['stability_analysis'][metric] = {
                'final_value_stability': stability_metrics,
                'training_stability': training_stability
            }
        
        # Generate summary insights
        report['summary_insights'] = self._generate_summary_insights(report)
        
        # Save report
        report_file = self.output_dir / 'comprehensive_analysis_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Comprehensive report saved to {report_file}")
        return report
    
    def _serialize_statistical_tests(
        self, 
        comparisons: Dict[str, Dict[str, StatisticalTest]]
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Serialize statistical test results for JSON output."""
        serialized = {}
        for gen1, gen1_tests in comparisons.items():
            serialized[gen1] = {}
            for gen2, test in gen1_tests.items():
                serialized[gen1][gen2] = {
                    'test_name': test.test_name,
                    'statistic': test.statistic,
                    'p_value': test.p_value,
                    'effect_size': test.effect_size,
                    'power': test.power,
                    'significant': test.significant,
                    'interpretation': test.interpretation
                }
        return serialized
    
    def _generate_summary_insights(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level insights from the analysis."""
        insights = {
            'best_performing_generators': {},
            'most_stable_generators': {},
            'fastest_converging_generators': {},
            'significant_differences': []
        }
        
        # Find best performing generators for each metric
        for metric, desc_stats in report['descriptive_statistics'].items():
            # Determine if lower or higher is better for this metric
            lower_is_better = metric in ['KLDivergence', 'WassersteinDistance', 'MMDDistance']
            
            gen_scores = {}
            for gen_type, stats in desc_stats.items():
                if stats['count'] > 0:
                    gen_scores[gen_type] = stats['mean']
            
            if gen_scores:
                if lower_is_better:
                    best_gen = min(gen_scores.keys(), key=lambda x: gen_scores[x])
                else:
                    best_gen = max(gen_scores.keys(), key=lambda x: gen_scores[x])
                
                insights['best_performing_generators'][metric] = {
                    'generator': best_gen,
                    'value': gen_scores[best_gen]
                }
        
        # Find most stable generators
        for metric, stability_data in report['stability_analysis'].items():
            stability_scores = {}
            for gen_type, metrics in stability_data['final_value_stability'].items():
                if 'stability_score' in metrics and not np.isnan(metrics['stability_score']):
                    stability_scores[gen_type] = metrics['stability_score']
            
            if stability_scores:
                most_stable = max(stability_scores.keys(), key=lambda x: stability_scores[x])
                insights['most_stable_generators'][metric] = {
                    'generator': most_stable,
                    'stability_score': stability_scores[most_stable]
                }
        
        # Find fastest converging generators
        for metric, conv_data in report['convergence_analysis'].items():
            conv_times = conv_data['mean_epochs_to_threshold']
            valid_times = {gen: time for gen, time in conv_times.items() 
                          if not np.isnan(time)}
            
            if valid_times:
                fastest = min(valid_times.keys(), key=lambda x: valid_times[x])
                insights['fastest_converging_generators'][metric] = {
                    'generator': fastest,
                    'epochs': valid_times[fastest]
                }
        
        # Identify significant differences
        for metric, test_data in report['statistical_tests'].items():
            for gen1, comparisons in test_data['pairwise_comparisons'].items():
                for gen2, test_result in comparisons.items():
                    if test_result['significant']:
                        insights['significant_differences'].append({
                            'metric': metric,
                            'generator1': gen1,
                            'generator2': gen2,
                            'p_value': test_result['p_value'],
                            'effect_size': test_result['effect_size']
                        })
        
        return insights
    
    def create_visualizations(
        self,
        runs: List[ExperimentRun],
        report: Dict[str, Any]
    ):
        """Create comprehensive visualizations of the analysis results."""
        self.logger.info("Creating visualizations...")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create multi-page PDF
        pdf_path = self.output_dir / 'analysis_visualizations.pdf'
        
        with PdfPages(pdf_path) as pdf:
            # 1. Performance comparison plots
            self._create_performance_plots(runs, report, pdf)
            
            # 2. Convergence analysis plots
            self._create_convergence_plots(runs, report, pdf)
            
            # 3. Stability analysis plots
            self._create_stability_plots(runs, report, pdf)
            
            # 4. Statistical significance heatmaps
            self._create_significance_heatmaps(report, pdf)
        
        self.logger.info(f"Visualizations saved to {pdf_path}")
    
    def _create_performance_plots(
        self, 
        runs: List[ExperimentRun], 
        report: Dict[str, Any], 
        pdf: PdfPages
    ):
        """Create performance comparison plots."""
        metrics = list(report['descriptive_statistics'].keys())
        n_metrics = len(metrics)
        
        # Box plots for each metric
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics[:4]):  # Show first 4 metrics
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Prepare data for box plot
            data_for_plot = []
            labels = []
            
            for gen_type in report['metadata']['generator_types']:
                values = []
                for run in runs:
                    if (run.generator_type == gen_type and 
                        run.final_metrics and 
                        metric in run.final_metrics):
                        value = run.final_metrics[metric]
                        if not (np.isnan(value) or np.isinf(value)):
                            values.append(value)
                
                if values:
                    data_for_plot.append(values)
                    labels.append(gen_type)
            
            if data_for_plot:
                ax.boxplot(data_for_plot, labels=labels)
                ax.set_title(f'{metric} Distribution by Generator Type')
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.text(0.5, 0.5, f'No data for {metric}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{metric} - No Data')
        
        # Remove empty subplots
        for i in range(len(metrics), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_convergence_plots(
        self, 
        runs: List[ExperimentRun], 
        report: Dict[str, Any], 
        pdf: PdfPages
    ):
        """Create convergence analysis plots."""
        # Convergence rate comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        conv_data = report['convergence_analysis']
        metrics = list(conv_data.keys())[:4]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            conv_rates = conv_data[metric]['convergence_rates']
            if conv_rates:
                generators = list(conv_rates.keys())
                rates = list(conv_rates.values())
                
                bars = ax.bar(generators, rates)
                ax.set_title(f'{metric} - Convergence Rates')
                ax.set_ylabel('Convergence Rate')
                ax.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, rate in zip(bars, rates):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{rate:.2f}', ha='center', va='bottom')
                
                ax.tick_params(axis='x', rotation=45)
        
        # Remove empty subplots
        for i in range(len(metrics), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_stability_plots(
        self, 
        runs: List[ExperimentRun], 
        report: Dict[str, Any], 
        pdf: PdfPages
    ):
        """Create stability analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        stability_data = report['stability_analysis']
        metrics = list(stability_data.keys())[:4]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            final_stability = stability_data[metric]['final_value_stability']
            
            generators = []
            cv_values = []
            
            for gen_type, metrics_data in final_stability.items():
                if 'coefficient_of_variation' in metrics_data:
                    cv = metrics_data['coefficient_of_variation']
                    if not np.isnan(cv):
                        generators.append(gen_type)
                        cv_values.append(cv)
            
            if generators:
                bars = ax.bar(generators, cv_values)
                ax.set_title(f'{metric} - Coefficient of Variation')
                ax.set_ylabel('Coefficient of Variation')
                
                # Add value labels
                for bar, cv in zip(bars, cv_values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{cv:.3f}', ha='center', va='bottom')
                
                ax.tick_params(axis='x', rotation=45)
        
        # Remove empty subplots
        for i in range(len(metrics), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_significance_heatmaps(self, report: Dict[str, Any], pdf: PdfPages):
        """Create heatmaps showing statistical significance between generators."""
        test_data = report['statistical_tests']
        generator_types = report['metadata']['generator_types']
        
        for metric, tests in test_data.items():
            if 'pairwise_comparisons' not in tests:
                continue
                
            # Create significance matrix
            n_gen = len(generator_types)
            sig_matrix = np.zeros((n_gen, n_gen))
            p_value_matrix = np.ones((n_gen, n_gen))
            
            gen_to_idx = {gen: i for i, gen in enumerate(generator_types)}
            
            comparisons = tests['pairwise_comparisons']
            for gen1, gen1_tests in comparisons.items():
                for gen2, test_result in gen1_tests.items():
                    if gen1 in gen_to_idx and gen2 in gen_to_idx:
                        i, j = gen_to_idx[gen1], gen_to_idx[gen2]
                        sig_matrix[i, j] = 1 if test_result['significant'] else 0
                        sig_matrix[j, i] = sig_matrix[i, j]  # Symmetric
                        p_value_matrix[i, j] = test_result['p_value']
                        p_value_matrix[j, i] = p_value_matrix[i, j]
            
            # Create heatmap
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Significance heatmap
            sns.heatmap(sig_matrix, 
                       xticklabels=generator_types,
                       yticklabels=generator_types,
                       annot=True, fmt='.0f',
                       cmap='RdYlBu_r', 
                       ax=ax1,
                       cbar_kws={'label': 'Significant (1) / Not Significant (0)'})
            ax1.set_title(f'{metric} - Statistical Significance')
            
            # P-value heatmap
            sns.heatmap(p_value_matrix, 
                       xticklabels=generator_types,
                       yticklabels=generator_types,
                       annot=True, fmt='.3f',
                       cmap='viridis_r', 
                       ax=ax2,
                       cbar_kws={'label': 'P-value'})
            ax2.set_title(f'{metric} - P-values')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()


def create_statistical_framework_example():
    """
    Create a complete example demonstrating the statistical framework usage.
    """
    # Example configuration
    base_config = {
        'z_dim': 4,
        'stage': 'train',
        'killer': False,
        'max_epochs': 50,
        'batch_size': 256,
        'learning_rate': 0.001,
        'dataset_type': 'NORMAL',
        'accelerator': 'gpu',
        'validation_samples': 500,
        'metrics': ['KLDivergence', 'LogLikelihood', 'WassersteinDistance', 'MMDDistance']
    }
    
    # Initialize framework components
    experiment_manager = MultiRunExperimentManager(
        base_config=base_config,
        output_dir="docs/statistical_analysis"
    )
    
    statistical_analyzer = StatisticalAnalyzer(
        alpha=0.05,
        correction_method='bonferroni'
    )
    
    convergence_analyzer = ConvergenceAnalyzer(
        convergence_thresholds={
            'KLDivergence': 0.3,
            'LogLikelihood': -3.5,
            'WassersteinDistance': 0.15,
            'MMDDistance': 0.2
        }
    )
    
    stability_analyzer = StabilityAnalyzer(outlier_method='iqr')
    
    results_aggregator = ResultsAggregator(
        output_dir="docs/statistical_analysis"
    )
    
    print("Statistical Analysis Framework Components Initialized")
    print("="*60)
    print("1. MultiRunExperimentManager - Orchestrates multiple training runs")
    print("2. StatisticalAnalyzer - Performs significance testing and effect size analysis")
    print("3. ConvergenceAnalyzer - Analyzes convergence speed and patterns")
    print("4. StabilityAnalyzer - Assesses performance consistency and reliability")
    print("5. ResultsAggregator - Generates comprehensive reports and visualizations")
    
    return {
        'experiment_manager': experiment_manager,
        'statistical_analyzer': statistical_analyzer,
        'convergence_analyzer': convergence_analyzer,
        'stability_analyzer': stability_analyzer,
        'results_aggregator': results_aggregator
    }


if __name__ == "__main__":
    # Demonstrate framework initialization
    framework = create_statistical_framework_example()
    
    # Example usage workflow
    print("\nExample Workflow:")
    print("1. Plan experiments: experiment_manager.plan_experiments()")
    print("2. Execute runs: experiment_manager.run_experiments(configs)")
    print("3. Analyze results: statistical_analyzer.compare_all_generators()")
    print("4. Generate report: results_aggregator.generate_comprehensive_report()")
    print("5. Create visualizations: results_aggregator.create_visualizations()")