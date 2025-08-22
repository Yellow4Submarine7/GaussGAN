"""
Stability and Variance Analysis Tools for GaussGAN
=================================================

This module provides comprehensive tools for analyzing the stability and variance
characteristics of different generator types in the GaussGAN project.

Key Features:
- Multi-dimensional stability assessment
- Variance decomposition analysis
- Outlier detection and robustness evaluation
- Consistency metrics across training runs
- Risk assessment for production deployment
- Reliability scoring and confidence intervals
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import warnings
from collections import defaultdict

# Statistical imports
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

# Bootstrap and resampling
from scipy.stats import bootstrap
import itertools


@dataclass
class StabilityMetrics:
    """Comprehensive stability metrics for a generator type."""
    generator_type: str
    metric_name: str
    n_runs: int
    
    # Central tendency
    mean: float
    median: float
    mode: Optional[float]
    
    # Dispersion measures
    std_deviation: float
    variance: float
    coefficient_of_variation: float
    interquartile_range: float
    mad: float  # Median Absolute Deviation
    range_value: float
    
    # Robust statistics
    trimmed_mean: float
    winsorized_std: float
    
    # Distribution shape
    skewness: float
    kurtosis: float
    
    # Stability scores
    consistency_score: float  # 0-1, higher is better
    reliability_score: float  # 0-1, higher is better
    predictability_score: float  # 0-1, higher is better
    
    # Outlier information
    outlier_count: int
    outlier_rate: float
    outlier_impact: float  # How much outliers affect the mean
    
    # Confidence intervals
    ci_lower: float
    ci_upper: float
    ci_width: float
    
    # Additional metrics
    entropy: float  # Shannon entropy of discretized values
    gini_coefficient: float  # Inequality measure
    risk_score: float  # Combined risk assessment (0-1, lower is better)


@dataclass
class VarianceDecomposition:
    """Variance decomposition analysis results."""
    metric_name: str
    total_variance: float
    
    # Variance components
    between_generator_variance: float
    within_generator_variance: float
    residual_variance: float
    
    # Variance ratios
    between_generator_ratio: float
    within_generator_ratio: float
    
    # Statistical significance
    f_statistic: float
    p_value: float
    
    # Effect sizes
    eta_squared: float  # Effect size for between-group differences
    omega_squared: float  # Unbiased effect size estimate


@dataclass
class RobustnessAssessment:
    """Robustness assessment for a generator type."""
    generator_type: str
    metric_name: str
    
    # Sensitivity to outliers
    outlier_sensitivity: float  # 0-1, lower is better
    breakdown_point: float  # Proportion of outliers before breakdown
    
    # Distributional robustness
    normality_p_value: float
    distributional_stability: float  # Consistency across runs
    
    # Performance under stress
    worst_case_performance: float
    best_case_performance: float
    performance_range: float
    
    # Confidence in estimates
    estimation_uncertainty: float
    confidence_score: float  # 0-1, higher is better


class OutlierDetector:
    """
    Advanced outlier detection using multiple methods.
    
    This class implements several outlier detection algorithms and provides
    ensemble-based outlier identification for robust analysis.
    """
    
    def __init__(self, contamination: float = 0.1):
        """
        Initialize outlier detector.
        
        Args:
            contamination: Expected proportion of outliers
        """
        self.contamination = contamination
        self.methods = {
            'isolation_forest': IsolationForest(contamination=contamination, random_state=42),
            'elliptic_envelope': EllipticEnvelope(contamination=contamination, random_state=42),
            'local_outlier_factor': LocalOutlierFactor(contamination=contamination),
            'z_score': None,  # Implemented separately
            'iqr': None,  # Implemented separately
            'modified_z_score': None  # Implemented separately
        }
        
    def detect_outliers(
        self, 
        data: np.ndarray, 
        methods: List[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Detect outliers using multiple methods.
        
        Args:
            data: Data array (n_samples, n_features) or (n_samples,)
            methods: List of methods to use
            
        Returns:
            Dictionary with outlier masks for each method
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        if methods is None:
            methods = ['isolation_forest', 'z_score', 'iqr']
        
        outlier_masks = {}
        
        for method in methods:
            if method == 'isolation_forest':
                outliers = self.methods['isolation_forest'].fit_predict(data) == -1
            elif method == 'elliptic_envelope':
                try:
                    outliers = self.methods['elliptic_envelope'].fit_predict(data) == -1
                except:
                    outliers = np.zeros(len(data), dtype=bool)
            elif method == 'local_outlier_factor':
                outliers = self.methods['local_outlier_factor'].fit_predict(data) == -1
            elif method == 'z_score':
                outliers = self._z_score_outliers(data.flatten())
            elif method == 'iqr':
                outliers = self._iqr_outliers(data.flatten())
            elif method == 'modified_z_score':
                outliers = self._modified_z_score_outliers(data.flatten())
            else:
                continue
            
            outlier_masks[method] = outliers
        
        return outlier_masks
    
    def _z_score_outliers(self, data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
        return z_scores > threshold
    
    def _iqr_outliers(self, data: np.ndarray, factor: float = 1.5) -> np.ndarray:
        """Detect outliers using IQR method."""
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        return (data < lower_bound) | (data > upper_bound)
    
    def _modified_z_score_outliers(self, data: np.ndarray, threshold: float = 3.5) -> np.ndarray:
        """Detect outliers using modified Z-score (based on median)."""
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / mad if mad > 0 else np.zeros_like(data)
        return np.abs(modified_z_scores) > threshold
    
    def ensemble_outlier_detection(
        self, 
        data: np.ndarray, 
        methods: List[str] = None,
        consensus_threshold: float = 0.5
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Ensemble outlier detection with consensus voting.
        
        Args:
            data: Data array
            methods: List of methods to use
            consensus_threshold: Proportion of methods that must agree
            
        Returns:
            Tuple of (consensus_outliers, method_scores)
        """
        outlier_masks = self.detect_outliers(data, methods)
        
        if not outlier_masks:
            return np.zeros(len(data), dtype=bool), {}
        
        # Stack all outlier masks
        all_masks = np.stack(list(outlier_masks.values()))
        
        # Compute consensus
        outlier_votes = np.sum(all_masks, axis=0)
        consensus_outliers = outlier_votes >= (len(outlier_masks) * consensus_threshold)
        
        # Compute method agreement scores
        method_scores = {}
        for method, mask in outlier_masks.items():
            # Agreement with consensus
            agreement = np.mean(mask == consensus_outliers)
            method_scores[method] = agreement
        
        return consensus_outliers, method_scores


class StabilityAnalyzer:
    """
    Comprehensive stability and variance analyzer.
    
    This class provides in-depth analysis of generator stability, including
    multi-dimensional variance analysis, robustness assessment, and risk scoring.
    """
    
    def __init__(
        self, 
        output_dir: Union[str, Path] = "docs/stability_analysis",
        confidence_level: float = 0.95
    ):
        """
        Initialize stability analyzer.
        
        Args:
            output_dir: Directory to save analysis results
            confidence_level: Confidence level for intervals
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
        self.outlier_detector = OutlierDetector()
        self.logger = logging.getLogger(__name__)
        
    def analyze_stability(
        self,
        experimental_runs: List[Any],
        metrics_to_analyze: List[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive stability analysis.
        
        Args:
            experimental_runs: List of experimental run objects
            metrics_to_analyze: List of metrics to analyze
            
        Returns:
            Dictionary with comprehensive stability analysis results
        """
        if metrics_to_analyze is None:
            metrics_to_analyze = ['KLDivergence', 'LogLikelihood', 'WassersteinDistance', 'MMDDistance']
        
        self.logger.info(f"Analyzing stability for {len(experimental_runs)} runs")
        
        results = {
            'stability_metrics': {},
            'variance_decomposition': {},
            'robustness_assessment': {},
            'outlier_analysis': {},
            'risk_assessment': {},
            'recommendations': {}
        }
        
        # Extract data by generator type and metric
        data_by_generator = self._extract_performance_data(experimental_runs, metrics_to_analyze)
        
        # Step 1: Compute stability metrics
        results['stability_metrics'] = self._compute_stability_metrics(data_by_generator)
        
        # Step 2: Variance decomposition
        results['variance_decomposition'] = self._perform_variance_decomposition(data_by_generator)
        
        # Step 3: Robustness assessment
        results['robustness_assessment'] = self._assess_robustness(data_by_generator)
        
        # Step 4: Outlier analysis
        results['outlier_analysis'] = self._analyze_outliers(data_by_generator)
        
        # Step 5: Risk assessment
        results['risk_assessment'] = self._assess_risks(results)
        
        # Step 6: Generate recommendations
        results['recommendations'] = self._generate_stability_recommendations(results)
        
        # Save results
        self._save_stability_results(results)
        
        return results
    
    def _extract_performance_data(
        self,
        experimental_runs: List[Any],
        metrics_to_analyze: List[str]
    ) -> Dict[str, Dict[str, List[float]]]:
        """Extract performance data organized by generator type and metric."""
        data = defaultdict(lambda: defaultdict(list))
        
        for run in experimental_runs:
            if not hasattr(run, 'final_metrics') or not run.final_metrics:
                continue
            
            gen_type = run.generator_type
            
            for metric in metrics_to_analyze:
                if metric in run.final_metrics:
                    value = run.final_metrics[metric]
                    if not (np.isnan(value) or np.isinf(value)):
                        data[gen_type][metric].append(value)
        
        return dict(data)
    
    def _compute_stability_metrics(
        self,
        data_by_generator: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, Dict[str, StabilityMetrics]]:
        """Compute comprehensive stability metrics."""
        stability_metrics = {}
        
        for gen_type, metrics_data in data_by_generator.items():
            stability_metrics[gen_type] = {}
            
            for metric_name, values in metrics_data.items():
                if len(values) < 3:
                    continue
                
                values = np.array(values)
                
                # Basic statistics
                mean_val = np.mean(values)
                median_val = np.median(values)
                std_val = np.std(values, ddof=1)
                var_val = np.var(values, ddof=1)
                
                # Robust statistics
                trimmed_mean = stats.trim_mean(values, 0.1)  # 10% trimmed mean
                winsorized_std = np.std(stats.mstats.winsorize(values, limits=0.05), ddof=1)
                
                # Coefficient of variation
                cv = std_val / abs(mean_val) if mean_val != 0 else np.inf
                
                # IQR and MAD
                q75, q25 = np.percentile(values, [75, 25])
                iqr = q75 - q25
                mad = np.median(np.abs(values - median_val))
                
                # Distribution shape
                skewness = stats.skew(values)
                kurtosis = stats.kurtosis(values)
                
                # Outlier detection
                outliers, _ = self.outlier_detector.ensemble_outlier_detection(values)
                outlier_count = np.sum(outliers)
                outlier_rate = outlier_count / len(values)
                
                # Outlier impact (difference in mean with/without outliers)
                if outlier_count > 0:
                    clean_values = values[~outliers]
                    outlier_impact = abs(np.mean(clean_values) - mean_val) / abs(mean_val) if mean_val != 0 else 0
                else:
                    outlier_impact = 0
                
                # Confidence interval
                ci_lower, ci_upper = self._bootstrap_confidence_interval(values, np.mean)
                ci_width = ci_upper - ci_lower
                
                # Stability scores
                consistency_score = self._compute_consistency_score(values)
                reliability_score = self._compute_reliability_score(values, outlier_rate)
                predictability_score = self._compute_predictability_score(values)
                
                # Entropy (discretized values)
                entropy = self._compute_entropy(values)
                
                # Gini coefficient
                gini_coef = self._compute_gini_coefficient(values)
                
                # Risk score
                risk_score = self._compute_risk_score(cv, outlier_rate, skewness, kurtosis)
                
                # Create stability metrics object
                stability_metrics[gen_type][metric_name] = StabilityMetrics(
                    generator_type=gen_type,
                    metric_name=metric_name,
                    n_runs=len(values),
                    mean=mean_val,
                    median=median_val,
                    mode=self._compute_mode(values),
                    std_deviation=std_val,
                    variance=var_val,
                    coefficient_of_variation=cv,
                    interquartile_range=iqr,
                    mad=mad,
                    range_value=np.max(values) - np.min(values),
                    trimmed_mean=trimmed_mean,
                    winsorized_std=winsorized_std,
                    skewness=skewness,
                    kurtosis=kurtosis,
                    consistency_score=consistency_score,
                    reliability_score=reliability_score,
                    predictability_score=predictability_score,
                    outlier_count=outlier_count,
                    outlier_rate=outlier_rate,
                    outlier_impact=outlier_impact,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    ci_width=ci_width,
                    entropy=entropy,
                    gini_coefficient=gini_coef,
                    risk_score=risk_score
                )
        
        return stability_metrics
    
    def _bootstrap_confidence_interval(
        self,
        data: np.ndarray,
        statistic_func=np.mean,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval."""
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic_func(sample))
        
        lower = np.percentile(bootstrap_stats, 100 * self.alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - self.alpha / 2))
        
        return lower, upper
    
    def _compute_consistency_score(self, values: np.ndarray) -> float:
        """Compute consistency score (0-1, higher is better)."""
        cv = np.std(values) / abs(np.mean(values)) if np.mean(values) != 0 else np.inf
        return 1 / (1 + cv)
    
    def _compute_reliability_score(self, values: np.ndarray, outlier_rate: float) -> float:
        """Compute reliability score (0-1, higher is better)."""
        # Base reliability on consistency and outlier resistance
        consistency = self._compute_consistency_score(values)
        outlier_resistance = 1 - outlier_rate
        return (consistency + outlier_resistance) / 2
    
    def _compute_predictability_score(self, values: np.ndarray) -> float:
        """Compute predictability score based on distribution normality."""
        _, p_value = stats.normaltest(values)
        # Higher p-value means more normal distribution, more predictable
        return min(1.0, p_value * 20)  # Scale to 0-1
    
    def _compute_entropy(self, values: np.ndarray, n_bins: int = 10) -> float:
        """Compute Shannon entropy of discretized values."""
        hist, _ = np.histogram(values, bins=n_bins)
        hist = hist[hist > 0]  # Remove zero bins
        probabilities = hist / np.sum(hist)
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _compute_gini_coefficient(self, values: np.ndarray) -> float:
        """Compute Gini coefficient (inequality measure)."""
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    def _compute_risk_score(
        self,
        cv: float,
        outlier_rate: float,
        skewness: float,
        kurtosis: float
    ) -> float:
        """Compute overall risk score (0-1, lower is better)."""
        # Normalize components to 0-1 scale
        cv_risk = min(1.0, cv / 2.0)  # CV > 2 is very high
        outlier_risk = outlier_rate
        skew_risk = min(1.0, abs(skewness) / 3.0)  # |skewness| > 3 is extreme
        kurt_risk = min(1.0, abs(kurtosis) / 10.0)  # |kurtosis| > 10 is extreme
        
        # Weighted combination
        risk_score = 0.4 * cv_risk + 0.3 * outlier_risk + 0.2 * skew_risk + 0.1 * kurt_risk
        return risk_score
    
    def _compute_mode(self, values: np.ndarray) -> Optional[float]:
        """Compute mode (most frequent value) using kernel density estimation."""
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(values)
            x_range = np.linspace(np.min(values), np.max(values), 100)
            densities = kde(x_range)
            mode_idx = np.argmax(densities)
            return x_range[mode_idx]
        except:
            return None
    
    def _perform_variance_decomposition(
        self,
        data_by_generator: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, VarianceDecomposition]:
        """Perform variance decomposition analysis."""
        variance_decomposition = {}
        
        # Get all metrics
        all_metrics = set()
        for metrics_data in data_by_generator.values():
            all_metrics.update(metrics_data.keys())
        
        for metric in all_metrics:
            # Collect all values for this metric
            all_values = []
            group_labels = []
            
            for gen_type, metrics_data in data_by_generator.items():
                if metric in metrics_data:
                    values = metrics_data[metric]
                    all_values.extend(values)
                    group_labels.extend([gen_type] * len(values))
            
            if len(all_values) < 6:  # Need minimum sample size
                continue
            
            # Perform one-way ANOVA
            groups = {}
            for gen_type in set(group_labels):
                groups[gen_type] = [all_values[i] for i, label in enumerate(group_labels) if label == gen_type]
            
            if len(groups) < 2:
                continue
            
            # ANOVA calculation
            group_values = list(groups.values())
            f_stat, p_value = stats.f_oneway(*group_values)
            
            # Variance components
            total_variance = np.var(all_values, ddof=1)
            
            # Between-group variance
            group_means = [np.mean(group) for group in group_values]
            group_sizes = [len(group) for group in group_values]
            overall_mean = np.mean(all_values)
            
            between_variance = np.sum([
                size * (mean - overall_mean) ** 2
                for size, mean in zip(group_sizes, group_means)
            ]) / (len(groups) - 1)
            
            # Within-group variance
            within_variance = np.sum([
                np.sum((np.array(group) - np.mean(group)) ** 2)
                for group in group_values
            ]) / (len(all_values) - len(groups))
            
            # Variance ratios
            between_ratio = between_variance / total_variance if total_variance > 0 else 0
            within_ratio = within_variance / total_variance if total_variance > 0 else 0
            
            # Effect sizes
            eta_squared = between_variance / (between_variance + within_variance) if (between_variance + within_variance) > 0 else 0
            
            # Omega squared (unbiased estimate)
            ms_between = between_variance
            ms_within = within_variance
            df_between = len(groups) - 1
            omega_squared = (df_between * (ms_between - ms_within)) / (
                df_between * ms_between + (len(all_values) - df_between) * ms_within + ms_within
            ) if ms_within > 0 else 0
            
            variance_decomposition[metric] = VarianceDecomposition(
                metric_name=metric,
                total_variance=total_variance,
                between_generator_variance=between_variance,
                within_generator_variance=within_variance,
                residual_variance=total_variance - between_variance - within_variance,
                between_generator_ratio=between_ratio,
                within_generator_ratio=within_ratio,
                f_statistic=f_stat,
                p_value=p_value,
                eta_squared=eta_squared,
                omega_squared=omega_squared
            )
        
        return variance_decomposition
    
    def _assess_robustness(
        self,
        data_by_generator: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, Dict[str, RobustnessAssessment]]:
        """Assess robustness of each generator type."""
        robustness_assessment = {}
        
        for gen_type, metrics_data in data_by_generator.items():
            robustness_assessment[gen_type] = {}
            
            for metric_name, values in metrics_data.items():
                if len(values) < 5:
                    continue
                
                values = np.array(values)
                
                # Outlier sensitivity
                outlier_sensitivity = self._compute_outlier_sensitivity(values)
                breakdown_point = self._estimate_breakdown_point(values)
                
                # Distributional robustness
                _, normality_p = stats.normaltest(values)
                distributional_stability = self._compute_distributional_stability(values)
                
                # Performance under stress
                worst_case = np.min(values) if metric_name not in ['LogLikelihood'] else np.max(values)
                best_case = np.max(values) if metric_name not in ['LogLikelihood'] else np.min(values)
                performance_range = abs(worst_case - best_case)
                
                # Estimation uncertainty
                estimation_uncertainty = self._compute_estimation_uncertainty(values)
                confidence_score = 1 / (1 + estimation_uncertainty)
                
                robustness_assessment[gen_type][metric_name] = RobustnessAssessment(
                    generator_type=gen_type,
                    metric_name=metric_name,
                    outlier_sensitivity=outlier_sensitivity,
                    breakdown_point=breakdown_point,
                    normality_p_value=normality_p,
                    distributional_stability=distributional_stability,
                    worst_case_performance=worst_case,
                    best_case_performance=best_case,
                    performance_range=performance_range,
                    estimation_uncertainty=estimation_uncertainty,
                    confidence_score=confidence_score
                )
        
        return robustness_assessment
    
    def _compute_outlier_sensitivity(self, values: np.ndarray) -> float:
        """Compute sensitivity to outliers (0-1, lower is better)."""
        # Compare mean vs median sensitivity
        mean_original = np.mean(values)
        median_original = np.median(values)
        
        # Add an artificial outlier and see impact
        outlier_value = np.max(values) + 3 * np.std(values)
        values_with_outlier = np.append(values, outlier_value)
        
        mean_with_outlier = np.mean(values_with_outlier)
        median_with_outlier = np.median(values_with_outlier)
        
        # Sensitivity as relative change
        mean_sensitivity = abs(mean_with_outlier - mean_original) / abs(mean_original) if mean_original != 0 else 0
        median_sensitivity = abs(median_with_outlier - median_original) / abs(median_original) if median_original != 0 else 0
        
        # Return average sensitivity
        return (mean_sensitivity + median_sensitivity) / 2
    
    def _estimate_breakdown_point(self, values: np.ndarray) -> float:
        """Estimate breakdown point (proportion of outliers before breakdown)."""
        # Simplified estimation: add outliers until variance increases dramatically
        original_std = np.std(values)
        
        for outlier_prop in np.arange(0.1, 0.6, 0.1):
            n_outliers = int(len(values) * outlier_prop)
            outlier_values = np.max(values) + np.random.normal(3 * original_std, original_std, n_outliers)
            contaminated_values = np.concatenate([values, outlier_values])
            
            new_std = np.std(contaminated_values)
            if new_std > 2 * original_std:  # Breakdown threshold
                return outlier_prop
        
        return 0.5  # Conservative estimate
    
    def _compute_distributional_stability(self, values: np.ndarray) -> float:
        """Compute distributional stability using bootstrap."""
        # Bootstrap multiple samples and check distributional consistency
        n_bootstrap = 100
        ks_stats = []
        
        for _ in range(n_bootstrap):
            sample1 = np.random.choice(values, size=len(values)//2, replace=True)
            sample2 = np.random.choice(values, size=len(values)//2, replace=True)
            
            ks_stat, _ = stats.ks_2samp(sample1, sample2)
            ks_stats.append(ks_stat)
        
        # Lower KS statistic means more stable distribution
        return 1 / (1 + np.mean(ks_stats))
    
    def _compute_estimation_uncertainty(self, values: np.ndarray) -> float:
        """Compute uncertainty in parameter estimation."""
        # Bootstrap standard error of the mean
        bootstrap_means = []
        for _ in range(1000):
            sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        se_mean = np.std(bootstrap_means)
        cv_estimate = se_mean / abs(np.mean(values)) if np.mean(values) != 0 else np.inf
        
        return cv_estimate
    
    def _analyze_outliers(
        self,
        data_by_generator: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, Any]:
        """Comprehensive outlier analysis."""
        outlier_analysis = {
            'outlier_counts': {},
            'outlier_characteristics': {},
            'outlier_impact': {}
        }
        
        for gen_type, metrics_data in data_by_generator.items():
            outlier_analysis['outlier_counts'][gen_type] = {}
            outlier_analysis['outlier_characteristics'][gen_type] = {}
            outlier_analysis['outlier_impact'][gen_type] = {}
            
            for metric_name, values in metrics_data.items():
                if len(values) < 5:
                    continue
                
                values = np.array(values)
                
                # Detect outliers using multiple methods
                outlier_masks = self.outlier_detector.detect_outliers(values)
                
                # Count outliers by method
                outlier_counts = {method: np.sum(mask) for method, mask in outlier_masks.items()}
                
                # Consensus outliers
                consensus_outliers, method_scores = self.outlier_detector.ensemble_outlier_detection(values)
                
                outlier_analysis['outlier_counts'][gen_type][metric_name] = {
                    'by_method': outlier_counts,
                    'consensus_count': np.sum(consensus_outliers),
                    'consensus_rate': np.sum(consensus_outliers) / len(values)
                }
                
                # Characterize outliers
                if np.sum(consensus_outliers) > 0:
                    outlier_values = values[consensus_outliers]
                    normal_values = values[~consensus_outliers]
                    
                    outlier_analysis['outlier_characteristics'][gen_type][metric_name] = {
                        'outlier_mean': np.mean(outlier_values),
                        'outlier_std': np.std(outlier_values),
                        'normal_mean': np.mean(normal_values),
                        'normal_std': np.std(normal_values),
                        'separation_score': abs(np.mean(outlier_values) - np.mean(normal_values)) / np.std(values)
                    }
                    
                    # Impact analysis
                    mean_with_outliers = np.mean(values)
                    mean_without_outliers = np.mean(normal_values)
                    relative_impact = abs(mean_with_outliers - mean_without_outliers) / abs(mean_with_outliers) if mean_with_outliers != 0 else 0
                    
                    outlier_analysis['outlier_impact'][gen_type][metric_name] = {
                        'relative_impact_on_mean': relative_impact,
                        'method_agreement': method_scores
                    }
        
        return outlier_analysis
    
    def _assess_risks(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall risks for each generator type."""
        stability_metrics = analysis_results['stability_metrics']
        robustness_assessment = analysis_results['robustness_assessment']
        
        risk_assessment = {}
        
        for gen_type in stability_metrics.keys():
            if gen_type not in robustness_assessment:
                continue
            
            # Aggregate risk scores across metrics
            risk_scores = []
            reliability_scores = []
            uncertainty_scores = []
            
            for metric_name in stability_metrics[gen_type].keys():
                stability = stability_metrics[gen_type][metric_name]
                
                risk_scores.append(stability.risk_score)
                reliability_scores.append(stability.reliability_score)
                
                if metric_name in robustness_assessment[gen_type]:
                    robustness = robustness_assessment[gen_type][metric_name]
                    uncertainty_scores.append(robustness.estimation_uncertainty)
            
            if risk_scores:
                risk_assessment[gen_type] = {
                    'overall_risk_score': np.mean(risk_scores),
                    'reliability_score': np.mean(reliability_scores),
                    'uncertainty_score': np.mean(uncertainty_scores) if uncertainty_scores else 0,
                    'risk_classification': self._classify_risk_level(np.mean(risk_scores)),
                    'deployment_recommendation': self._get_deployment_recommendation(
                        np.mean(risk_scores), np.mean(reliability_scores)
                    )
                }
        
        return risk_assessment
    
    def _classify_risk_level(self, risk_score: float) -> str:
        """Classify risk level based on score."""
        if risk_score < 0.2:
            return "Low Risk"
        elif risk_score < 0.5:
            return "Medium Risk"
        elif risk_score < 0.8:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def _get_deployment_recommendation(self, risk_score: float, reliability_score: float) -> str:
        """Get deployment recommendation based on risk and reliability."""
        if risk_score < 0.3 and reliability_score > 0.7:
            return "Recommended for production deployment"
        elif risk_score < 0.5 and reliability_score > 0.5:
            return "Suitable for deployment with monitoring"
        elif risk_score < 0.7:
            return "Requires further testing before deployment"
        else:
            return "Not recommended for production deployment"
    
    def _generate_stability_recommendations(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate actionable stability recommendations."""
        recommendations = {}
        
        stability_metrics = analysis_results['stability_metrics']
        risk_assessment = analysis_results['risk_assessment']
        
        # Find most stable generator overall
        if stability_metrics:
            avg_consistency = {}
            for gen_type, metrics in stability_metrics.items():
                consistency_scores = [m.consistency_score for m in metrics.values()]
                avg_consistency[gen_type] = np.mean(consistency_scores) if consistency_scores else 0
            
            most_stable = max(avg_consistency.keys(), key=lambda x: avg_consistency[x])
            recommendations['most_stable_generator'] = (
                f"Most stable generator: {most_stable} "
                f"(average consistency score: {avg_consistency[most_stable]:.3f})"
            )
        
        # Risk-based recommendations
        if risk_assessment:
            low_risk_generators = [
                gen for gen, assessment in risk_assessment.items()
                if assessment['risk_classification'] == "Low Risk"
            ]
            
            if low_risk_generators:
                recommendations['low_risk_options'] = (
                    f"Low-risk generators for production: {', '.join(low_risk_generators)}"
                )
            else:
                recommendations['risk_warning'] = (
                    "No generators classified as low-risk. "
                    "Consider additional hyperparameter tuning or more training runs."
                )
        
        # Variance decomposition insights
        variance_decomp = analysis_results.get('variance_decomposition', {})
        if variance_decomp:
            high_between_group_variance = [
                metric for metric, decomp in variance_decomp.items()
                if decomp.between_generator_ratio > 0.3
            ]
            
            if high_between_group_variance:
                recommendations['generator_choice_matters'] = (
                    f"Generator choice significantly affects: {', '.join(high_between_group_variance)}. "
                    f"Careful generator selection is important for these metrics."
                )
        
        return recommendations
    
    def _save_stability_results(self, results: Dict[str, Any]):
        """Save stability analysis results."""
        # Convert dataclasses to dictionaries for JSON serialization
        results_copy = results.copy()
        
        # Convert StabilityMetrics objects
        if 'stability_metrics' in results_copy:
            for gen_type in results_copy['stability_metrics']:
                for metric_name in results_copy['stability_metrics'][gen_type]:
                    obj = results_copy['stability_metrics'][gen_type][metric_name]
                    results_copy['stability_metrics'][gen_type][metric_name] = asdict(obj)
        
        # Convert VarianceDecomposition objects
        if 'variance_decomposition' in results_copy:
            for metric_name in results_copy['variance_decomposition']:
                obj = results_copy['variance_decomposition'][metric_name]
                results_copy['variance_decomposition'][metric_name] = asdict(obj)
        
        # Convert RobustnessAssessment objects
        if 'robustness_assessment' in results_copy:
            for gen_type in results_copy['robustness_assessment']:
                for metric_name in results_copy['robustness_assessment'][gen_type]:
                    obj = results_copy['robustness_assessment'][gen_type][metric_name]
                    results_copy['robustness_assessment'][gen_type][metric_name] = asdict(obj)
        
        # Save to JSON
        import json
        results_file = self.output_dir / 'stability_analysis_results.json'
        with open(results_file, 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)
        
        self.logger.info(f"Stability analysis results saved to {results_file}")
    
    def create_stability_visualizations(self, analysis_results: Dict[str, Any]):
        """Create comprehensive stability visualizations."""
        self.logger.info("Creating stability visualizations...")
        
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create visualizations
        self._plot_stability_overview(analysis_results)
        self._plot_risk_matrix(analysis_results)
        self._plot_variance_decomposition(analysis_results)
        self._plot_outlier_analysis(analysis_results)
        self._plot_robustness_comparison(analysis_results)
        
        self.logger.info(f"Stability visualizations saved to {self.output_dir}")
    
    def _plot_stability_overview(self, results: Dict[str, Any]):
        """Create stability overview plots."""
        stability_metrics = results['stability_metrics']
        
        if not stability_metrics:
            return
        
        # Prepare data for plotting
        generators = list(stability_metrics.keys())
        metrics = list(next(iter(stability_metrics.values())).keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Plot 1: Consistency Scores
        ax = axes[0]
        consistency_data = []
        for gen in generators:
            scores = [stability_metrics[gen][metric].consistency_score for metric in metrics]
            consistency_data.append(scores)
        
        bp = ax.boxplot(consistency_data, labels=generators)
        ax.set_title('Consistency Scores by Generator')
        ax.set_ylabel('Consistency Score')
        ax.set_ylim(0, 1)
        
        # Plot 2: Risk Scores
        ax = axes[1]
        risk_data = []
        for gen in generators:
            scores = [stability_metrics[gen][metric].risk_score for metric in metrics]
            risk_data.append(scores)
        
        bp = ax.boxplot(risk_data, labels=generators)
        ax.set_title('Risk Scores by Generator')
        ax.set_ylabel('Risk Score')
        ax.set_ylim(0, 1)
        
        # Plot 3: Coefficient of Variation
        ax = axes[2]
        cv_data = []
        for gen in generators:
            scores = [stability_metrics[gen][metric].coefficient_of_variation for metric in metrics]
            cv_data.append(scores)
        
        bp = ax.boxplot(cv_data, labels=generators)
        ax.set_title('Coefficient of Variation by Generator')
        ax.set_ylabel('CV')
        
        # Plot 4: Outlier Rates
        ax = axes[3]
        outlier_data = []
        for gen in generators:
            rates = [stability_metrics[gen][metric].outlier_rate for metric in metrics]
            outlier_data.append(rates)
        
        bp = ax.boxplot(outlier_data, labels=generators)
        ax.set_title('Outlier Rates by Generator')
        ax.set_ylabel('Outlier Rate')
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'stability_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_risk_matrix(self, results: Dict[str, Any]):
        """Create risk assessment matrix."""
        risk_assessment = results.get('risk_assessment', {})
        
        if not risk_assessment:
            return
        
        generators = list(risk_assessment.keys())
        risk_categories = ['overall_risk_score', 'uncertainty_score']
        benefit_categories = ['reliability_score']
        
        # Create risk vs benefit plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Risk matrix
        risk_data = []
        for gen in generators:
            row = [risk_assessment[gen][cat] for cat in risk_categories]
            risk_data.append(row)
        
        risk_matrix = np.array(risk_data)
        
        im1 = ax1.imshow(risk_matrix, cmap='Reds', aspect='auto')
        ax1.set_xticks(range(len(risk_categories)))
        ax1.set_yticks(range(len(generators)))
        ax1.set_xticklabels([cat.replace('_', ' ').title() for cat in risk_categories])
        ax1.set_yticklabels(generators)
        ax1.set_title('Risk Assessment Matrix')
        
        # Add text annotations
        for i in range(len(generators)):
            for j in range(len(risk_categories)):
                text = ax1.text(j, i, f'{risk_matrix[i, j]:.3f}',
                               ha="center", va="center", color="white")
        
        plt.colorbar(im1, ax=ax1, label='Risk Score')
        
        # Risk vs Reliability scatter plot
        for gen in generators:
            risk = risk_assessment[gen]['overall_risk_score']
            reliability = risk_assessment[gen]['reliability_score']
            ax2.scatter(risk, reliability, s=100, label=gen, alpha=0.7)
            ax2.annotate(gen, (risk, reliability), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        ax2.set_xlabel('Risk Score')
        ax2.set_ylabel('Reliability Score')
        ax2.set_title('Risk vs Reliability')
        ax2.grid(True, alpha=0.3)
        
        # Add quadrant lines
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Add quadrant labels
        ax2.text(0.25, 0.75, 'Low Risk\nHigh Reliability', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax2.text(0.75, 0.25, 'High Risk\nLow Reliability', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'risk_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_variance_decomposition(self, results: Dict[str, Any]):
        """Plot variance decomposition results."""
        variance_decomp = results.get('variance_decomposition', {})
        
        if not variance_decomp:
            return
        
        metrics = list(variance_decomp.keys())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Variance components
        between_ratios = [variance_decomp[metric].between_generator_ratio for metric in metrics]
        within_ratios = [variance_decomp[metric].within_generator_ratio for metric in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax1.bar(x - width/2, between_ratios, width, label='Between Generators', alpha=0.8)
        ax1.bar(x + width/2, within_ratios, width, label='Within Generators', alpha=0.8)
        
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Variance Ratio')
        ax1.set_title('Variance Decomposition')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Effect sizes
        eta_squared = [variance_decomp[metric].eta_squared for metric in metrics]
        omega_squared = [variance_decomp[metric].omega_squared for metric in metrics]
        
        ax2.bar(x - width/2, eta_squared, width, label='Eta Squared', alpha=0.8)
        ax2.bar(x + width/2, omega_squared, width, label='Omega Squared', alpha=0.8)
        
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Effect Size')
        ax2.set_title('Effect Sizes')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'variance_decomposition.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_outlier_analysis(self, results: Dict[str, Any]):
        """Plot outlier analysis results."""
        outlier_analysis = results.get('outlier_analysis', {})
        
        if not outlier_analysis or 'outlier_counts' not in outlier_analysis:
            return
        
        # Aggregate outlier rates
        generator_outlier_rates = {}
        
        for gen_type, metrics_data in outlier_analysis['outlier_counts'].items():
            rates = []
            for metric_name, counts_data in metrics_data.items():
                if 'consensus_rate' in counts_data:
                    rates.append(counts_data['consensus_rate'])
            
            if rates:
                generator_outlier_rates[gen_type] = np.mean(rates)
        
        if not generator_outlier_rates:
            return
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        generators = list(generator_outlier_rates.keys())
        rates = list(generator_outlier_rates.values())
        
        bars = ax.bar(generators, rates, color='lightcoral', alpha=0.7)
        ax.set_xlabel('Generator Type')
        ax.set_ylabel('Average Outlier Rate')
        ax.set_title('Outlier Rates by Generator Type')
        ax.set_ylim(0, max(rates) * 1.1)
        
        # Add value labels
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{rate:.3f}', ha='center', va='bottom')
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'outlier_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_robustness_comparison(self, results: Dict[str, Any]):
        """Plot robustness comparison."""
        robustness_assessment = results.get('robustness_assessment', {})
        
        if not robustness_assessment:
            return
        
        # Aggregate robustness scores
        generators = list(robustness_assessment.keys())
        metrics = set()
        for gen_data in robustness_assessment.values():
            metrics.update(gen_data.keys())
        metrics = list(metrics)
        
        if not metrics:
            return
        
        # Create radar chart-style comparison
        fig, ax = plt.subplots(figsize=(10, 8))
        
        robustness_categories = ['confidence_score', 'distributional_stability']
        
        # Create heatmap data
        heatmap_data = []
        for gen in generators:
            row = []
            for metric in metrics:
                if metric in robustness_assessment[gen]:
                    scores = []
                    for category in robustness_categories:
                        scores.append(getattr(robustness_assessment[gen][metric], category, 0))
                    row.append(np.mean(scores))
                else:
                    row.append(0)
            heatmap_data.append(row)
        
        heatmap_data = np.array(heatmap_data)
        
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(metrics)))
        ax.set_yticks(range(len(generators)))
        ax.set_xticklabels(metrics, rotation=45)
        ax.set_yticklabels(generators)
        ax.set_title('Robustness Scores by Generator and Metric')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Robustness Score')
        
        # Add text annotations
        for i in range(len(generators)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}',
                              ha="center", va="center", 
                              color="white" if heatmap_data[i, j] < 0.5 else "black")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'robustness_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()


def demonstrate_stability_analyzer():
    """Demonstrate the stability analyzer."""
    print("Advanced Stability and Variance Analyzer Demo")
    print("=" * 50)
    
    analyzer = StabilityAnalyzer()
    
    print("✓ Stability analyzer initialized")
    print("✓ Outlier detector ready")
    print("✓ Variance decomposition tools ready")
    
    print("\nFeatures available:")
    print("- Comprehensive stability metrics (15+ measures)")
    print("- Multi-method outlier detection")
    print("- ANOVA-based variance decomposition")
    print("- Robustness assessment with breakdown points")
    print("- Risk scoring and deployment recommendations")
    print("- Bootstrap confidence intervals")
    print("- Distributional stability analysis")
    
    return analyzer


if __name__ == "__main__":
    analyzer = demonstrate_stability_analyzer()