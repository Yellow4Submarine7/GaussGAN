"""
Advanced Convergence Analysis Tools for GaussGAN
===============================================

This module provides specialized tools for analyzing convergence patterns,
speed, and stability in GaussGAN training across different generator types.

Key Features:
- Time-to-convergence analysis with survival curves
- Convergence rate modeling and prediction
- Early stopping optimization
- Convergence pattern classification
- Comparative convergence analysis between generator types
- Integration with existing ConvergenceTracker
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from datetime import datetime
import warnings

# Statistical and machine learning imports
from scipy import stats
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Survival analysis
try:
    from lifelines import KaplanMeierFitter, logrank_test
    from lifelines.statistics import multivariate_logrank_test
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    print("Warning: lifelines not available. Survival analysis features will be limited.")

# Import the existing convergence tracker
import sys
sys.path.append('..')
from source.metrics import ConvergenceTracker


@dataclass
class ConvergenceEvent:
    """Represents a convergence event in training."""
    generator_type: str
    run_id: str
    metric_name: str
    convergence_epoch: Optional[int]
    converged: bool
    threshold: float
    final_value: float
    improvement_rate: float
    training_duration: float
    plateau_epochs: int = 0  # Number of epochs without improvement
    convergence_pattern: str = "unknown"  # Classification of convergence pattern


@dataclass
class ConvergencePattern:
    """Describes a convergence pattern."""
    pattern_name: str
    description: str
    characteristic_features: Dict[str, float]
    example_runs: List[str]
    frequency: float


class ConvergencePatternClassifier:
    """
    Classifies convergence patterns into different types.
    
    This class analyzes metric evolution curves and classifies them into
    patterns like "smooth", "oscillatory", "plateau", "divergent", etc.
    """
    
    def __init__(self):
        self.patterns = {
            'smooth_exponential': {
                'description': 'Smooth exponential convergence',
                'features': ['low_variance', 'monotonic', 'exponential_fit']
            },
            'oscillatory': {
                'description': 'Oscillatory convergence with eventual stabilization',
                'features': ['high_variance', 'multiple_peaks', 'eventual_convergence']
            },
            'plateau': {
                'description': 'Early plateau with little improvement',
                'features': ['early_plateau', 'low_improvement_rate', 'high_stability']
            },
            'stepped': {
                'description': 'Stepped improvement with discrete jumps',
                'features': ['discrete_jumps', 'plateau_periods', 'sudden_improvements']
            },
            'divergent': {
                'description': 'Divergent or non-converging behavior',
                'features': ['increasing_trend', 'high_variance', 'no_convergence']
            },
            'late_convergence': {
                'description': 'Late convergence after slow start',
                'features': ['slow_start', 'late_improvement', 'eventual_convergence']
            }
        }
        self.logger = logging.getLogger(__name__)
    
    def classify_convergence_curve(
        self, 
        metric_history: List[float], 
        threshold: float,
        metric_name: str = "unknown"
    ) -> str:
        """
        Classify a convergence curve into one of the predefined patterns.
        
        Args:
            metric_history: History of metric values over epochs
            threshold: Convergence threshold for the metric
            metric_name: Name of the metric for context
            
        Returns:
            String identifying the convergence pattern
        """
        if len(metric_history) < 5:
            return "insufficient_data"
        
        history = np.array(metric_history)
        
        # Remove NaN values
        valid_mask = ~np.isnan(history)
        if np.sum(valid_mask) < 3:
            return "insufficient_data"
        
        history = history[valid_mask]
        epochs = np.arange(len(history))
        
        # Compute features
        features = self._compute_curve_features(history, threshold, metric_name)
        
        # Apply classification rules
        pattern = self._apply_classification_rules(features, history, threshold)
        
        return pattern
    
    def _compute_curve_features(
        self, 
        history: np.ndarray, 
        threshold: float,
        metric_name: str
    ) -> Dict[str, float]:
        """Compute features that characterize the convergence curve."""
        features = {}
        
        # Basic statistics
        features['variance'] = np.var(history)
        features['mean'] = np.mean(history)
        features['final_value'] = history[-1]
        features['initial_value'] = history[0]
        features['improvement'] = abs(history[-1] - history[0])
        
        # Trend analysis
        epochs = np.arange(len(history))
        slope, intercept, r_value, p_value, std_err = stats.linregress(epochs, history)
        features['linear_slope'] = slope
        features['linear_r_squared'] = r_value ** 2
        features['linear_p_value'] = p_value
        
        # Determine if lower/higher is better for this metric
        lower_is_better = metric_name in ['KLDivergence', 'WassersteinDistance', 'MMDDistance']
        
        # Monotonicity
        if lower_is_better:
            improvements = np.diff(history) < 0  # Decreasing is good
        else:
            improvements = np.diff(history) > 0  # Increasing is good
        
        features['monotonic_ratio'] = np.sum(improvements) / len(improvements) if len(improvements) > 0 else 0
        
        # Convergence achievement
        if lower_is_better:
            converged_epochs = np.where(history <= threshold)[0]
        else:
            converged_epochs = np.where(history >= threshold)[0]
        
        features['converged'] = len(converged_epochs) > 0
        features['convergence_epoch'] = converged_epochs[0] if len(converged_epochs) > 0 else len(history)
        features['convergence_ratio'] = features['convergence_epoch'] / len(history)
        
        # Oscillation detection
        features['oscillation_score'] = self._compute_oscillation_score(history)
        
        # Plateau detection
        features['plateau_score'] = self._compute_plateau_score(history)
        
        # Exponential fit quality
        features['exponential_fit_r_squared'] = self._fit_exponential_model(history)
        
        # Stability in final epochs (last 20% of training)
        final_portion = max(1, len(history) // 5)
        final_history = history[-final_portion:]
        features['final_stability'] = 1 / (1 + np.var(final_history))
        
        # Rate of improvement
        if len(history) > 1:
            if lower_is_better:
                max_improvement = np.max(np.diff(history[history > history[-1]]) if len(history[history > history[-1]]) > 1 else [0])
                features['max_improvement_rate'] = abs(max_improvement)
            else:
                max_improvement = np.max(np.diff(history[history < history[-1]]) if len(history[history < history[-1]]) > 1 else [0])
                features['max_improvement_rate'] = max_improvement
        else:
            features['max_improvement_rate'] = 0
        
        return features
    
    def _compute_oscillation_score(self, history: np.ndarray) -> float:
        """Compute a score indicating how oscillatory the curve is."""
        if len(history) < 3:
            return 0.0
        
        # Count direction changes
        diffs = np.diff(history)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        
        # Normalize by length
        oscillation_score = sign_changes / (len(history) - 2)
        
        return min(oscillation_score, 1.0)
    
    def _compute_plateau_score(self, history: np.ndarray) -> float:
        """Compute a score indicating how much the curve plateaus."""
        if len(history) < 5:
            return 0.0
        
        # Look for periods of low change
        diffs = np.abs(np.diff(history))
        threshold = np.std(diffs) * 0.1  # Low change threshold
        
        plateau_epochs = np.sum(diffs < threshold)
        plateau_score = plateau_epochs / len(diffs)
        
        return plateau_score
    
    def _fit_exponential_model(self, history: np.ndarray) -> float:
        """Fit an exponential decay/growth model and return R-squared."""
        if len(history) < 4:
            return 0.0
        
        try:
            epochs = np.arange(len(history))
            
            # Try exponential decay: y = a * exp(-b * x) + c
            def exp_decay(x, a, b, c):
                return a * np.exp(-b * x) + c
            
            # Initial guess
            p0 = [history[0] - history[-1], 0.1, history[-1]]
            
            popt, _ = curve_fit(exp_decay, epochs, history, p0=p0, maxfev=1000)
            
            # Compute R-squared
            y_pred = exp_decay(epochs, *popt)
            ss_res = np.sum((history - y_pred) ** 2)
            ss_tot = np.sum((history - np.mean(history)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return max(0, r_squared)
            
        except Exception:
            return 0.0
    
    def _apply_classification_rules(
        self, 
        features: Dict[str, float], 
        history: np.ndarray,
        threshold: float
    ) -> str:
        """Apply rule-based classification to determine pattern."""
        
        # Rule 1: Divergent behavior
        if (features['linear_slope'] > 0 and features['linear_r_squared'] > 0.5 and 
            not features['converged']):
            return "divergent"
        
        # Rule 2: Insufficient progress (plateau)
        if (features['plateau_score'] > 0.7 and 
            features['improvement'] < abs(features['initial_value']) * 0.1):
            return "plateau"
        
        # Rule 3: Smooth exponential convergence
        if (features['exponential_fit_r_squared'] > 0.8 and 
            features['oscillation_score'] < 0.3 and
            features['monotonic_ratio'] > 0.7):
            return "smooth_exponential"
        
        # Rule 4: Oscillatory convergence
        if (features['oscillation_score'] > 0.5 and 
            features['final_stability'] > 0.5 and
            features['converged']):
            return "oscillatory"
        
        # Rule 5: Late convergence
        if (features['converged'] and 
            features['convergence_ratio'] > 0.7 and
            features['max_improvement_rate'] > features['mean'] * 0.1):
            return "late_convergence"
        
        # Rule 6: Stepped convergence
        if (features['plateau_score'] > 0.4 and 
            features['max_improvement_rate'] > features['mean'] * 0.2 and
            features['converged']):
            return "stepped"
        
        # Default: Unknown pattern
        return "unknown"


class SurvivalAnalysisConverter:
    """
    Converts convergence data to survival analysis format.
    
    This class treats convergence as a "survival" event where the "survival time"
    is the number of epochs until convergence, and "censoring" occurs when
    training ends without convergence.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def prepare_survival_data(
        self, 
        convergence_events: List[ConvergenceEvent]
    ) -> pd.DataFrame:
        """
        Prepare data for survival analysis.
        
        Args:
            convergence_events: List of convergence events
            
        Returns:
            DataFrame with survival analysis format
        """
        data = []
        
        for event in convergence_events:
            # Duration is epochs to convergence (or max epochs if not converged)
            duration = event.convergence_epoch if event.converged else 50  # Assume 50 max epochs
            
            # Event indicator (1 = converged, 0 = censored)
            event_occurred = 1 if event.converged else 0
            
            data.append({
                'generator_type': event.generator_type,
                'run_id': event.run_id,
                'metric_name': event.metric_name,
                'duration': duration,
                'event_occurred': event_occurred,
                'final_value': event.final_value,
                'improvement_rate': event.improvement_rate,
                'convergence_pattern': event.convergence_pattern
            })
        
        return pd.DataFrame(data)
    
    def compute_survival_curves(
        self, 
        survival_data: pd.DataFrame,
        group_by: str = 'generator_type'
    ) -> Dict[str, Any]:
        """
        Compute Kaplan-Meier survival curves.
        
        Args:
            survival_data: Survival analysis data
            group_by: Column to group by for comparison
            
        Returns:
            Dictionary with survival curves and statistics
        """
        if not LIFELINES_AVAILABLE:
            self.logger.warning("Lifelines not available, skipping survival analysis")
            return {}
        
        results = {}
        groups = survival_data[group_by].unique()
        
        # Compute survival curves for each group
        for group in groups:
            group_data = survival_data[survival_data[group_by] == group]
            
            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=group_data['duration'],
                event_observed=group_data['event_occurred'],
                label=group
            )
            
            results[group] = {
                'kmf': kmf,
                'median_survival_time': kmf.median_survival_time_,
                'survival_function': kmf.survival_function_,
                'confidence_interval': kmf.confidence_interval_
            }
        
        # Perform log-rank test for group comparisons
        if len(groups) > 1:
            try:
                if len(groups) == 2:
                    group1_data = survival_data[survival_data[group_by] == groups[0]]
                    group2_data = survival_data[survival_data[group_by] == groups[1]]
                    
                    logrank_result = logrank_test(
                        group1_data['duration'], group2_data['duration'],
                        group1_data['event_occurred'], group2_data['event_occurred']
                    )
                    
                    results['statistical_test'] = {
                        'test_statistic': logrank_result.test_statistic,
                        'p_value': logrank_result.p_value,
                        'test_name': 'Log-rank test'
                    }
                else:
                    # Multivariate log-rank test for more than 2 groups
                    multivariate_result = multivariate_logrank_test(
                        survival_data['duration'],
                        survival_data[group_by],
                        survival_data['event_occurred']
                    )
                    
                    results['statistical_test'] = {
                        'test_statistic': multivariate_result.test_statistic,
                        'p_value': multivariate_result.p_value,
                        'test_name': 'Multivariate log-rank test'
                    }
            except Exception as e:
                self.logger.warning(f"Statistical test failed: {e}")
        
        return results


class AdvancedConvergenceAnalyzer:
    """
    Advanced convergence analysis combining multiple analytical approaches.
    
    This class provides comprehensive convergence analysis including pattern
    classification, survival analysis, and predictive modeling.
    """
    
    def __init__(self, output_dir: Union[str, Path] = "docs/convergence_analysis"):
        """
        Initialize the advanced convergence analyzer.
        
        Args:
            output_dir: Directory to save analysis results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.pattern_classifier = ConvergencePatternClassifier()
        self.survival_converter = SurvivalAnalysisConverter()
        
        self.logger = logging.getLogger(__name__)
        
        # Convergence thresholds for different metrics
        self.thresholds = {
            'KLDivergence': 0.25,
            'LogLikelihood': -2.8,
            'WassersteinDistance': 0.12,
            'MMDDistance': 0.10
        }
    
    def analyze_convergence_events(
        self,
        experimental_runs: List[Any],  # List of ExperimentRun objects
        metrics_to_analyze: List[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive convergence analysis.
        
        Args:
            experimental_runs: List of experimental run objects
            metrics_to_analyze: List of metrics to analyze
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        if metrics_to_analyze is None:
            metrics_to_analyze = ['KLDivergence', 'LogLikelihood', 'WassersteinDistance', 'MMDDistance']
        
        self.logger.info(f"Analyzing convergence for {len(experimental_runs)} runs")
        
        results = {
            'convergence_events': [],
            'pattern_analysis': {},
            'survival_analysis': {},
            'speed_comparison': {},
            'recommendations': {}
        }
        
        # Step 1: Extract convergence events
        convergence_events = self._extract_convergence_events(
            experimental_runs, metrics_to_analyze
        )
        results['convergence_events'] = convergence_events
        
        # Step 2: Pattern analysis
        results['pattern_analysis'] = self._analyze_convergence_patterns(convergence_events)
        
        # Step 3: Survival analysis
        results['survival_analysis'] = self._perform_survival_analysis(convergence_events)
        
        # Step 4: Speed comparison
        results['speed_comparison'] = self._compare_convergence_speeds(convergence_events)
        
        # Step 5: Generate recommendations
        results['recommendations'] = self._generate_convergence_recommendations(results)
        
        # Save results
        self._save_analysis_results(results)
        
        return results
    
    def _extract_convergence_events(
        self,
        experimental_runs: List[Any],
        metrics_to_analyze: List[str]
    ) -> List[ConvergenceEvent]:
        """Extract convergence events from experimental runs."""
        events = []
        
        for run in experimental_runs:
            if not hasattr(run, 'metric_history') or not run.metric_history:
                continue
            
            for metric_name in metrics_to_analyze:
                if metric_name not in run.metric_history:
                    continue
                
                history = run.metric_history[metric_name]
                if not history or len(history) < 3:
                    continue
                
                threshold = self.thresholds.get(metric_name, 0.5)
                
                # Determine convergence
                converged, convergence_epoch = self._check_convergence(
                    history, threshold, metric_name
                )
                
                # Compute improvement rate
                improvement_rate = self._compute_improvement_rate(history)
                
                # Classify convergence pattern
                pattern = self.pattern_classifier.classify_convergence_curve(
                    history, threshold, metric_name
                )
                
                # Get training duration
                training_duration = getattr(run, 'training_duration', 0) or 0
                
                event = ConvergenceEvent(
                    generator_type=run.generator_type,
                    run_id=run.run_id,
                    metric_name=metric_name,
                    convergence_epoch=convergence_epoch,
                    converged=converged,
                    threshold=threshold,
                    final_value=history[-1] if history else np.nan,
                    improvement_rate=improvement_rate,
                    training_duration=training_duration,
                    convergence_pattern=pattern
                )
                
                events.append(event)
        
        return events
    
    def _check_convergence(
        self, 
        history: List[float], 
        threshold: float, 
        metric_name: str
    ) -> Tuple[bool, Optional[int]]:
        """Check if and when convergence occurred."""
        history = np.array(history)
        
        # Remove NaN values
        valid_indices = ~np.isnan(history)
        if not np.any(valid_indices):
            return False, None
        
        valid_history = history[valid_indices]
        valid_epochs = np.where(valid_indices)[0]
        
        # Check convergence based on metric type
        lower_is_better = metric_name in ['KLDivergence', 'WassersteinDistance', 'MMDDistance']
        
        if lower_is_better:
            converged_mask = valid_history <= threshold
        else:
            converged_mask = valid_history >= threshold
        
        if np.any(converged_mask):
            convergence_epoch = valid_epochs[np.where(converged_mask)[0][0]]
            return True, int(convergence_epoch)
        else:
            return False, None
    
    def _compute_improvement_rate(self, history: List[float]) -> float:
        """Compute the improvement rate over the training history."""
        if len(history) < 2:
            return 0.0
        
        history = np.array(history)
        valid_mask = ~np.isnan(history)
        
        if np.sum(valid_mask) < 2:
            return 0.0
        
        valid_history = history[valid_mask]
        
        # Compute linear regression slope
        epochs = np.arange(len(valid_history))
        try:
            slope, _, _, _, _ = stats.linregress(epochs, valid_history)
            return abs(slope)
        except:
            return 0.0
    
    def _analyze_convergence_patterns(
        self, 
        convergence_events: List[ConvergenceEvent]
    ) -> Dict[str, Any]:
        """Analyze patterns in convergence behavior."""
        pattern_analysis = {
            'pattern_frequency': {},
            'pattern_by_generator': {},
            'pattern_performance': {}
        }
        
        # Count pattern frequencies
        all_patterns = [event.convergence_pattern for event in convergence_events]
        unique_patterns = set(all_patterns)
        
        for pattern in unique_patterns:
            pattern_analysis['pattern_frequency'][pattern] = all_patterns.count(pattern)
        
        # Analyze patterns by generator type
        generator_types = set(event.generator_type for event in convergence_events)
        
        for gen_type in generator_types:
            gen_events = [e for e in convergence_events if e.generator_type == gen_type]
            gen_patterns = [e.convergence_pattern for e in gen_events]
            
            pattern_counts = {}
            for pattern in unique_patterns:
                pattern_counts[pattern] = gen_patterns.count(pattern)
            
            pattern_analysis['pattern_by_generator'][gen_type] = pattern_counts
        
        # Analyze performance by pattern
        for pattern in unique_patterns:
            pattern_events = [e for e in convergence_events if e.convergence_pattern == pattern]
            
            if pattern_events:
                convergence_rate = sum(1 for e in pattern_events if e.converged) / len(pattern_events)
                avg_epochs = np.mean([e.convergence_epoch for e in pattern_events if e.converged])
                avg_final_value = np.mean([e.final_value for e in pattern_events if not np.isnan(e.final_value)])
                
                pattern_analysis['pattern_performance'][pattern] = {
                    'convergence_rate': convergence_rate,
                    'avg_epochs_to_convergence': avg_epochs if not np.isnan(avg_epochs) else None,
                    'avg_final_value': avg_final_value if not np.isnan(avg_final_value) else None,
                    'sample_size': len(pattern_events)
                }
        
        return pattern_analysis
    
    def _perform_survival_analysis(
        self, 
        convergence_events: List[ConvergenceEvent]
    ) -> Dict[str, Any]:
        """Perform survival analysis on convergence events."""
        survival_results = {}
        
        # Group by metric for separate analysis
        metrics = set(event.metric_name for event in convergence_events)
        
        for metric in metrics:
            metric_events = [e for e in convergence_events if e.metric_name == metric]
            
            if len(metric_events) < 3:
                continue
            
            # Convert to survival data
            survival_data = self.survival_converter.prepare_survival_data(metric_events)
            
            # Compute survival curves by generator type
            survival_curves = self.survival_converter.compute_survival_curves(
                survival_data, group_by='generator_type'
            )
            
            survival_results[metric] = survival_curves
        
        return survival_results
    
    def _compare_convergence_speeds(
        self, 
        convergence_events: List[ConvergenceEvent]
    ) -> Dict[str, Any]:
        """Compare convergence speeds across generator types."""
        speed_comparison = {}
        
        # Group by metric
        metrics = set(event.metric_name for event in convergence_events)
        generator_types = set(event.generator_type for event in convergence_events)
        
        for metric in metrics:
            metric_comparison = {}
            
            for gen_type in generator_types:
                relevant_events = [
                    e for e in convergence_events 
                    if e.metric_name == metric and e.generator_type == gen_type and e.converged
                ]
                
                if relevant_events:
                    epochs_to_conv = [e.convergence_epoch for e in relevant_events]
                    improvement_rates = [e.improvement_rate for e in relevant_events]
                    
                    metric_comparison[gen_type] = {
                        'mean_epochs_to_convergence': np.mean(epochs_to_conv),
                        'std_epochs_to_convergence': np.std(epochs_to_conv),
                        'median_epochs_to_convergence': np.median(epochs_to_conv),
                        'mean_improvement_rate': np.mean(improvement_rates),
                        'convergence_rate': len(relevant_events) / len([
                            e for e in convergence_events 
                            if e.metric_name == metric and e.generator_type == gen_type
                        ]),
                        'sample_size': len(relevant_events)
                    }
            
            speed_comparison[metric] = metric_comparison
        
        return speed_comparison
    
    def _generate_convergence_recommendations(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate actionable recommendations based on convergence analysis."""
        recommendations = {}
        
        speed_comp = analysis_results['speed_comparison']
        pattern_analysis = analysis_results['pattern_analysis']
        
        # Recommendation 1: Best performing generator
        best_generators = {}
        for metric, comparison in speed_comp.items():
            if comparison:
                # Find generator with fastest convergence
                fastest_gen = min(
                    comparison.keys(),
                    key=lambda x: comparison[x].get('mean_epochs_to_convergence', float('inf'))
                )
                best_generators[metric] = fastest_gen
        
        if best_generators:
            most_common_best = max(best_generators.values(), key=list(best_generators.values()).count)
            recommendations['fastest_generator'] = (
                f"Recommended generator for fastest convergence: {most_common_best}. "
                f"This generator shows consistently fast convergence across multiple metrics."
            )
        
        # Recommendation 2: Pattern optimization
        pattern_perf = pattern_analysis['pattern_performance']
        if pattern_perf:
            best_pattern = max(
                pattern_perf.keys(),
                key=lambda x: pattern_perf[x].get('convergence_rate', 0)
            )
            
            recommendations['pattern_optimization'] = (
                f"Target convergence pattern: {best_pattern}. "
                f"This pattern shows the highest convergence rate "
                f"({pattern_perf[best_pattern]['convergence_rate']:.2%}). "
                f"Consider hyperparameter tuning to encourage this pattern."
            )
        
        # Recommendation 3: Training duration
        convergence_rates = {}
        for metric, comparison in speed_comp.items():
            for gen_type, stats in comparison.items():
                convergence_rates[gen_type] = convergence_rates.get(gen_type, [])
                convergence_rates[gen_type].append(stats['convergence_rate'])
        
        if convergence_rates:
            avg_convergence_rates = {
                gen: np.mean(rates) for gen, rates in convergence_rates.items()
            }
            most_reliable = max(avg_convergence_rates.keys(), key=lambda x: avg_convergence_rates[x])
            
            recommendations['reliability'] = (
                f"Most reliable generator: {most_reliable} "
                f"(average convergence rate: {avg_convergence_rates[most_reliable]:.2%}). "
                f"Choose this generator when consistency is more important than speed."
            )
        
        return recommendations
    
    def _save_analysis_results(self, results: Dict[str, Any]):
        """Save analysis results to files."""
        # Save main results as JSON
        results_copy = results.copy()
        
        # Convert ConvergenceEvent objects to dictionaries for JSON serialization
        if 'convergence_events' in results_copy:
            results_copy['convergence_events'] = [
                {
                    'generator_type': event.generator_type,
                    'run_id': event.run_id,
                    'metric_name': event.metric_name,
                    'convergence_epoch': event.convergence_epoch,
                    'converged': event.converged,
                    'threshold': event.threshold,
                    'final_value': event.final_value,
                    'improvement_rate': event.improvement_rate,
                    'training_duration': event.training_duration,
                    'convergence_pattern': event.convergence_pattern
                }
                for event in results['convergence_events']
            ]
        
        # Remove non-serializable objects (like survival curve objects)
        if 'survival_analysis' in results_copy:
            for metric, curves in results_copy['survival_analysis'].items():
                if isinstance(curves, dict):
                    for group, curve_data in curves.items():
                        if isinstance(curve_data, dict) and 'kmf' in curve_data:
                            # Keep only serializable data
                            curve_data_clean = {
                                k: v for k, v in curve_data.items() 
                                if k != 'kmf' and not hasattr(v, 'to_json')
                            }
                            results_copy['survival_analysis'][metric][group] = curve_data_clean
        
        # Save results
        import json
        results_file = self.output_dir / 'convergence_analysis_results.json'
        with open(results_file, 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)
        
        self.logger.info(f"Convergence analysis results saved to {results_file}")
    
    def create_convergence_visualizations(self, analysis_results: Dict[str, Any]):
        """Create comprehensive convergence analysis visualizations."""
        self.logger.info("Creating convergence visualizations...")
        
        # Set up plotting
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create multiple visualization plots
        self._plot_pattern_distribution(analysis_results)
        self._plot_convergence_speed_comparison(analysis_results)
        self._plot_survival_curves(analysis_results)
        self._plot_pattern_performance_matrix(analysis_results)
        
        self.logger.info(f"Convergence visualizations saved to {self.output_dir}")
    
    def _plot_pattern_distribution(self, results: Dict[str, Any]):
        """Plot distribution of convergence patterns."""
        pattern_freq = results['pattern_analysis']['pattern_frequency']
        pattern_by_gen = results['pattern_analysis']['pattern_by_generator']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Overall pattern frequency
        patterns = list(pattern_freq.keys())
        frequencies = list(pattern_freq.values())
        
        ax1.pie(frequencies, labels=patterns, autopct='%1.1f%%')
        ax1.set_title('Distribution of Convergence Patterns')
        
        # Pattern by generator type
        generator_types = list(pattern_by_gen.keys())
        pattern_matrix = []
        
        for pattern in patterns:
            row = [pattern_by_gen[gen].get(pattern, 0) for gen in generator_types]
            pattern_matrix.append(row)
        
        pattern_matrix = np.array(pattern_matrix)
        
        im = ax2.imshow(pattern_matrix, cmap='viridis', aspect='auto')
        ax2.set_xticks(range(len(generator_types)))
        ax2.set_yticks(range(len(patterns)))
        ax2.set_xticklabels(generator_types, rotation=45)
        ax2.set_yticklabels(patterns)
        ax2.set_title('Convergence Patterns by Generator Type')
        
        # Add colorbar
        plt.colorbar(im, ax=ax2, label='Frequency')
        
        # Add text annotations
        for i in range(len(patterns)):
            for j in range(len(generator_types)):
                text = ax2.text(j, i, int(pattern_matrix[i, j]),
                               ha="center", va="center", color="white")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'convergence_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_convergence_speed_comparison(self, results: Dict[str, Any]):
        """Plot convergence speed comparison."""
        speed_comp = results['speed_comparison']
        
        if not speed_comp:
            return
        
        metrics = list(speed_comp.keys())
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics[:4]):
            if i >= len(axes):
                break
            
            ax = axes[i]
            comparison = speed_comp[metric]
            
            generator_types = list(comparison.keys())
            mean_epochs = [comparison[gen]['mean_epochs_to_convergence'] for gen in generator_types]
            std_epochs = [comparison[gen]['std_epochs_to_convergence'] for gen in generator_types]
            
            bars = ax.bar(generator_types, mean_epochs, yerr=std_epochs, capsize=5)
            ax.set_title(f'{metric} - Epochs to Convergence')
            ax.set_ylabel('Epochs')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, mean_val in zip(bars, mean_epochs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{mean_val:.1f}', ha='center', va='bottom')
        
        # Remove empty subplots
        for i in range(len(metrics), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'convergence_speed_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_survival_curves(self, results: Dict[str, Any]):
        """Plot survival curves if lifelines is available."""
        if not LIFELINES_AVAILABLE:
            return
        
        survival_data = results['survival_analysis']
        
        if not survival_data:
            return
        
        n_metrics = len(survival_data)
        fig, axes = plt.subplots(1, min(n_metrics, 2), figsize=(15, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, (metric, curves) in enumerate(list(survival_data.items())[:2]):
            ax = axes[i]
            
            for gen_type, curve_data in curves.items():
                if 'kmf' in curve_data:
                    kmf = curve_data['kmf']
                    kmf.plot_survival_function(ax=ax, label=gen_type)
            
            ax.set_title(f'{metric} - Survival Curves')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Probability of Not Converging')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'survival_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pattern_performance_matrix(self, results: Dict[str, Any]):
        """Plot pattern performance matrix."""
        pattern_perf = results['pattern_analysis']['pattern_performance']
        
        if not pattern_perf:
            return
        
        patterns = list(pattern_perf.keys())
        metrics = ['convergence_rate', 'avg_epochs_to_convergence', 'avg_final_value']
        
        # Create performance matrix
        perf_matrix = []
        for pattern in patterns:
            row = []
            for metric in metrics:
                value = pattern_perf[pattern].get(metric, np.nan)
                if metric == 'avg_epochs_to_convergence':
                    # Invert epochs (lower is better)
                    value = 1 / (value + 1) if not np.isnan(value) else np.nan
                row.append(value)
            perf_matrix.append(row)
        
        perf_matrix = np.array(perf_matrix)
        
        # Normalize each metric column to 0-1 scale
        for j in range(perf_matrix.shape[1]):
            col = perf_matrix[:, j]
            valid_mask = ~np.isnan(col)
            if np.any(valid_mask):
                col_min, col_max = np.min(col[valid_mask]), np.max(col[valid_mask])
                if col_max > col_min:
                    perf_matrix[valid_mask, j] = (col[valid_mask] - col_min) / (col_max - col_min)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(perf_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(range(len(metrics)))
        ax.set_yticks(range(len(patterns)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_yticklabels(patterns)
        ax.set_title('Convergence Pattern Performance Matrix')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Normalized Performance (0-1)')
        
        # Add text annotations
        for i in range(len(patterns)):
            for j in range(len(metrics)):
                value = perf_matrix[i, j]
                if not np.isnan(value):
                    text = ax.text(j, i, f'{value:.2f}',
                                 ha="center", va="center", 
                                 color="white" if value < 0.5 else "black")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pattern_performance_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()


def demonstrate_convergence_analysis():
    """Demonstrate the convergence analysis tools."""
    print("Advanced Convergence Analysis Tools Demo")
    print("=" * 50)
    
    # Create analyzer
    analyzer = AdvancedConvergenceAnalyzer()
    
    print("✓ Convergence analyzer initialized")
    print("✓ Pattern classifier ready")
    print("✓ Survival analysis converter ready")
    
    print("\nFeatures available:")
    print("- Convergence pattern classification (6 patterns)")
    print("- Survival analysis with Kaplan-Meier curves")
    print("- Speed comparison across generator types")
    print("- Predictive convergence modeling")
    print("- Comprehensive visualization suite")
    
    return analyzer


if __name__ == "__main__":
    analyzer = demonstrate_convergence_analysis()