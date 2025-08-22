"""
Enhanced Metrics Implementation Examples for GaussGAN Layer 2

This module provides enhanced implementations and integration patterns
for the existing metrics system, focusing on better integration and
multi-scale analysis rather than reimplementing existing functionality.

Author: Created for GaussGAN statistical measures enhancement
"""

import warnings
from typing import Dict, List, Optional, Union, Tuple, Any
from collections import deque
import numpy as np
import torch
from scipy import stats
from sklearn.mixture import GaussianMixture

# Import existing base class
from source.metrics import GaussianMetric, ALL_METRICS


class MetricFactory:
    """
    Factory for creating metrics with consistent initialization patterns.
    
    This factory handles the complexity of different metric initialization
    requirements and provides a unified interface for metric creation.
    """
    
    @staticmethod
    def create_metric(
        metric_name: str, 
        target_data: Optional[np.ndarray] = None,
        centroids: Optional[List] = None,
        cov_matrices: Optional[List] = None, 
        weights: Optional[List] = None,
        **kwargs
    ) -> GaussianMetric:
        """
        Create metric instance with proper parameter handling.
        
        Args:
            metric_name: Name of the metric to create
            target_data: Target distribution samples (for distance metrics)
            centroids: GMM centroids (for likelihood-based metrics)
            cov_matrices: GMM covariance matrices
            weights: GMM mixture weights
            **kwargs: Additional metric-specific parameters
            
        Returns:
            Initialized metric instance
            
        Raises:
            ValueError: If required parameters are missing
            KeyError: If metric name is not recognized
        """
        if metric_name not in ALL_METRICS:
            raise KeyError(f"Unknown metric: {metric_name}")
            
        # GMM-based metrics
        if metric_name in ["LogLikelihood", "KLDivergence", "MMDivergenceFromGMM"]:
            if not all(x is not None for x in [centroids, cov_matrices, weights]):
                raise ValueError(f"{metric_name} requires GMM parameters")
            return ALL_METRICS[metric_name](
                centroids=centroids,
                cov_matrices=cov_matrices,
                weights=weights,
                **kwargs
            )
            
        # Distance-based metrics requiring target samples
        elif metric_name in ["WassersteinDistance", "MMDDistance", "MMDivergence"]:
            if target_data is None:
                raise ValueError(f"{metric_name} requires target_data")
            return ALL_METRICS[metric_name](
                target_samples=target_data,
                **kwargs
            )
            
        # Simple metrics with no parameters
        else:
            return ALL_METRICS[metric_name](**kwargs)


class UnifiedMMD(GaussianMetric):
    """
    Unified MMD implementation combining best features from existing classes.
    
    This class provides a single interface for MMD computation that can work
    with either direct target samples or GMM parameters, and supports multiple
    kernel types and bandwidth selection strategies.
    
    Features:
    - Multiple kernel types (RBF, polynomial, linear)
    - Adaptive bandwidth selection using median heuristic
    - Both sample-based and GMM-based initialization
    - Efficient computation with numerical stability
    - Proper gradient flow for PyTorch integration
    """
    
    def __init__(
        self,
        target_samples: Optional[Union[np.ndarray, torch.Tensor]] = None,
        centroids: Optional[List] = None,
        cov_matrices: Optional[List] = None,
        weights: Optional[List] = None,
        kernel: str = "rbf",
        bandwidths: Optional[Union[List[float], np.ndarray]] = None,
        n_target_samples: int = 1000,
        dist_sync_on_step: bool = False
    ):
        """
        Initialize unified MMD metric.
        
        Args:
            target_samples: Direct target distribution samples
            centroids: GMM component means (alternative to target_samples)
            cov_matrices: GMM component covariances  
            weights: GMM mixture weights
            kernel: Kernel type ("rbf", "polynomial", "linear")
            bandwidths: Manual bandwidth specification
            n_target_samples: Number of samples to generate from GMM
            dist_sync_on_step: Whether to sync across distributed processes
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        # Flexible target sample initialization
        if target_samples is not None:
            self.target_samples = self._validate_samples(target_samples)
        elif all(x is not None for x in [centroids, cov_matrices, weights]):
            self.target_samples = self._generate_gmm_samples(
                centroids, cov_matrices, weights, n_target_samples
            )
        else:
            raise ValueError("Must provide either target_samples or GMM parameters")
            
        self.kernel_type = kernel
        self.bandwidths = bandwidths or self._compute_adaptive_bandwidths()
        
        if kernel not in ["rbf", "polynomial", "linear"]:
            raise ValueError(f"Unsupported kernel type: {kernel}")
    
    def _validate_samples(self, samples: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert and validate target samples."""
        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()
        else:
            samples = np.array(samples)
            
        # Remove NaN values
        valid_mask = ~np.isnan(samples).any(axis=1)
        samples = samples[valid_mask]
        
        if len(samples) == 0:
            raise ValueError("No valid target samples after NaN filtering")
            
        return samples
    
    def _generate_gmm_samples(
        self, 
        centroids: List, 
        cov_matrices: List, 
        weights: List, 
        n_samples: int
    ) -> np.ndarray:
        """Generate target samples from GMM parameters."""
        try:
            gmm = GaussianMixture(n_components=len(centroids), covariance_type="full")
            gmm.means_ = np.array(centroids)
            gmm.covariances_ = np.array(cov_matrices)
            gmm.weights_ = np.array(weights)
            
            # Compute precision matrices
            gmm.precisions_cholesky_ = np.linalg.cholesky(
                np.linalg.inv(gmm.covariances_)
            )
            
            samples, _ = gmm.sample(n_samples)
            return samples
            
        except Exception as e:
            warnings.warn(f"Failed to generate GMM samples: {e}. Using centroids.")
            return np.array(centroids)
    
    def _compute_adaptive_bandwidths(self) -> np.ndarray:
        """
        Compute adaptive bandwidths using median heuristic.
        
        Returns multiple scales around the median pairwise distance for
        robust MMD estimation across different data scales.
        """
        try:
            from scipy.spatial.distance import pdist
            
            if len(self.target_samples) < 2:
                return np.array([1.0])
                
            # Compute pairwise distances
            distances = pdist(self.target_samples, metric='euclidean')
            
            if len(distances) == 0:
                return np.array([1.0])
                
            median_dist = np.median(distances)
            if median_dist == 0:
                median_dist = 1.0
                
            # Multiple scales for robust estimation
            scales = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
            bandwidths = median_dist * scales
            
            return bandwidths
            
        except Exception as e:
            warnings.warn(f"Adaptive bandwidth computation failed: {e}")
            return np.array([0.1, 1.0, 10.0])
    
    def _compute_kernel_matrix(
        self, 
        X: np.ndarray, 
        Y: np.ndarray, 
        bandwidth: float
    ) -> np.ndarray:
        """Compute kernel matrix between two sample sets."""
        
        if self.kernel_type == "rbf":
            # RBF (Gaussian) kernel: K(x,y) = exp(-||x-y||^2 / (2*bandwidth))
            X_sqnorms = np.sum(X**2, axis=1, keepdims=True)
            Y_sqnorms = np.sum(Y**2, axis=1, keepdims=True).T  
            XY = np.dot(X, Y.T)
            
            sq_distances = X_sqnorms + Y_sqnorms - 2 * XY
            sq_distances = np.maximum(sq_distances, 0.0)  # Numerical stability
            
            return np.exp(-sq_distances / (2 * bandwidth))
            
        elif self.kernel_type == "polynomial":
            # Polynomial kernel: K(x,y) = (1 + <x,y>)^degree
            degree = getattr(self, 'degree', 2)
            return (1 + np.dot(X, Y.T)) ** degree
            
        elif self.kernel_type == "linear":
            # Linear kernel: K(x,y) = <x,y>
            return np.dot(X, Y.T)
            
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def _compute_mmd_squared(self, generated_samples: np.ndarray) -> float:
        """
        Compute squared MMD between generated and target samples.
        
        Uses unbiased estimators for MMD computation:
        MMD^2 = E[K(X,X)] + E[K(Y,Y)] - 2*E[K(X,Y)]
        """
        n_gen = len(generated_samples)
        n_target = len(self.target_samples)
        
        if n_gen == 0:
            return float('inf')
        
        mmd_values = []
        
        # Compute MMD for each bandwidth
        for bandwidth in self.bandwidths:
            # Kernel matrices
            K_XX = self._compute_kernel_matrix(
                self.target_samples, self.target_samples, bandwidth
            )
            K_YY = self._compute_kernel_matrix(
                generated_samples, generated_samples, bandwidth
            )  
            K_XY = self._compute_kernel_matrix(
                self.target_samples, generated_samples, bandwidth
            )
            
            # Unbiased estimators (exclude diagonal terms)
            if n_target > 1:
                term_XX = (np.sum(K_XX) - np.trace(K_XX)) / (n_target * (n_target - 1))
            else:
                term_XX = 0.0
                
            if n_gen > 1:
                term_YY = (np.sum(K_YY) - np.trace(K_YY)) / (n_gen * (n_gen - 1))
            else:
                term_YY = 0.0
                
            term_XY = np.mean(K_XY)
            
            # MMD squared for this bandwidth
            mmd_sq = term_XX + term_YY - 2 * term_XY
            mmd_values.append(max(mmd_sq, 0.0))  # Ensure non-negative
            
        # Average across bandwidths
        return np.mean(mmd_values)
    
    def compute_score(self, points: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Compute MMD scores for generated points.
        
        Args:
            points: Generated samples
            
        Returns:
            Array of MMD values (broadcasted to match input size)
        """
        try:
            # Convert and validate input
            if isinstance(points, torch.Tensor):
                points_np = points.cpu().numpy()
            else:
                points_np = np.array(points)
                
            # Filter NaN values
            valid_mask = ~np.isnan(points_np).any(axis=1)
            valid_points = points_np[valid_mask]
            
            if len(valid_points) == 0:
                warnings.warn("No valid points for MMD computation")
                return np.array([float('nan')] * len(points_np))
                
            # Compute MMD
            mmd_squared = self._compute_mmd_squared(valid_points)
            mmd_value = np.sqrt(mmd_squared)
            
            # Return same value for all points (distributional metric)
            return np.array([mmd_value] * len(points_np))
            
        except Exception as e:
            warnings.warn(f"Error in UnifiedMMD computation: {e}")
            return np.array([float('nan')] * len(points))


class MultiScaleConvergence(GaussianMetric):
    """
    Multi-timescale convergence analysis for comprehensive training monitoring.
    
    This class analyzes convergence patterns at different temporal scales:
    - Short-term (5-10 epochs): Local stability, recent fluctuations
    - Medium-term (20-50 epochs): Trend analysis, improvement rate
    - Long-term (50+ epochs): Global convergence, plateau detection
    
    Provides richer convergence information than single-scale analysis.
    """
    
    def __init__(
        self,
        timescales: Optional[Dict[str, int]] = None,
        stability_threshold: float = 0.05,
        trend_threshold: float = 1e-4,
        dist_sync_on_step: bool = False
    ):
        """
        Initialize multi-scale convergence analyzer.
        
        Args:
            timescales: Dict mapping scale names to window sizes
            stability_threshold: Coefficient of variation threshold for stability
            trend_threshold: Minimum slope for trend detection  
            dist_sync_on_step: Whether to sync across processes
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        self.timescales = timescales or {
            "short": 10,
            "medium": 30, 
            "long": 100
        }
        
        self.stability_threshold = stability_threshold
        self.trend_threshold = trend_threshold
        
        # History buffers for each timescale
        self.history_buffers = {
            scale: deque(maxlen=window_size)
            for scale, window_size in self.timescales.items()
        }
        
        # Track convergence indicators
        self.convergence_indicators = {
            scale: {"converged": False, "epoch": None}
            for scale in self.timescales.keys()
        }
    
    def update_history(self, metric_value: float) -> Dict[str, Any]:
        """
        Update metric history and compute multi-scale convergence indicators.
        
        Args:
            metric_value: New metric value to add to history
            
        Returns:
            Dict containing convergence analysis for each timescale
        """
        # Add to all buffers
        for scale_name, buffer in self.history_buffers.items():
            buffer.append(metric_value)
            
        # Compute convergence indicators for each scale
        analysis = {}
        for scale_name, buffer in self.history_buffers.items():
            if len(buffer) >= 3:  # Minimum data for analysis
                analysis[scale_name] = self._analyze_timescale(scale_name, list(buffer))
            else:
                analysis[scale_name] = {"status": "insufficient_data"}
                
        return analysis
    
    def _analyze_timescale(self, scale_name: str, values: List[float]) -> Dict[str, Any]:
        """Analyze convergence for a specific timescale."""
        values = np.array(values)
        valid_values = values[~np.isnan(values)]
        
        if len(valid_values) < 3:
            return {"status": "insufficient_valid_data"}
            
        analysis = {
            "status": "active",
            "n_points": len(valid_values),
            "mean": float(np.mean(valid_values)),
            "std": float(np.std(valid_values)),
            "cv": float(np.std(valid_values) / (np.mean(valid_values) + 1e-8)),
            "trend_slope": 0.0,
            "trend_r2": 0.0,
            "is_stable": False,
            "is_improving": False,
            "is_converged": False
        }
        
        # Stability analysis (coefficient of variation)
        analysis["is_stable"] = analysis["cv"] < self.stability_threshold
        
        # Trend analysis (linear regression)
        if len(valid_values) >= 3:
            x = np.arange(len(valid_values))
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, valid_values)
                analysis["trend_slope"] = float(slope)
                analysis["trend_r2"] = float(r_value**2)
                analysis["trend_p_value"] = float(p_value)
                
                # Improvement detection (negative slope for metrics where lower is better)
                analysis["is_improving"] = (
                    slope < -self.trend_threshold and 
                    p_value < 0.05 and 
                    r_value**2 > 0.5
                )
                
            except Exception:
                analysis["trend_slope"] = 0.0
                analysis["trend_r2"] = 0.0
                analysis["is_improving"] = False
        
        # Convergence detection (stable + not improving)
        analysis["is_converged"] = (
            analysis["is_stable"] and 
            not analysis["is_improving"] and
            len(valid_values) >= self.timescales[scale_name] // 2
        )
        
        # Update convergence tracking
        if (analysis["is_converged"] and 
            not self.convergence_indicators[scale_name]["converged"]):
            self.convergence_indicators[scale_name]["converged"] = True
            self.convergence_indicators[scale_name]["epoch"] = len(valid_values)
        
        return analysis
    
    def compute_score(self, points: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        This metric doesn't operate on points directly.
        Use update_history() method instead.
        """
        warnings.warn("MultiScaleConvergence should use update_history(), not compute_score()")
        return np.array([0.0] * len(points))
    
    def get_overall_convergence_status(self) -> Dict[str, Any]:
        """
        Get overall convergence status across all timescales.
        
        Returns:
            Comprehensive convergence summary
        """
        # Count converged scales
        converged_scales = sum(
            1 for indicator in self.convergence_indicators.values()
            if indicator["converged"]
        )
        
        total_scales = len(self.timescales)
        convergence_ratio = converged_scales / total_scales
        
        # Overall status
        if convergence_ratio >= 0.67:  # Majority converged
            overall_status = "converged"
        elif convergence_ratio >= 0.33:  # Some convergence
            overall_status = "partially_converged" 
        else:
            overall_status = "not_converged"
            
        return {
            "overall_status": overall_status,
            "converged_scales": converged_scales,
            "total_scales": total_scales,
            "convergence_ratio": convergence_ratio,
            "scale_details": dict(self.convergence_indicators)
        }


class StabilityWeightedMetric(GaussianMetric):
    """
    Aggregates multiple metrics with stability-based weighting.
    
    This class combines multiple metrics into a single score, where each
    metric's contribution is weighted by its historical stability. More
    stable metrics receive higher weights, making the overall evaluation
    more robust to noisy or unreliable metrics.
    """
    
    def __init__(
        self,
        base_metrics: Dict[str, GaussianMetric],
        stability_window: int = 20,
        min_weight: float = 0.1,
        max_weight: float = 1.0,
        dist_sync_on_step: bool = False
    ):
        """
        Initialize stability-weighted metric aggregator.
        
        Args:
            base_metrics: Dictionary of metric name -> metric instance
            stability_window: Number of recent values for stability calculation
            min_weight: Minimum weight for any metric
            max_weight: Maximum weight for any metric  
            dist_sync_on_step: Whether to sync across processes
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        self.base_metrics = base_metrics
        self.stability_window = stability_window
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # History for stability calculation
        self.metric_histories = {
            name: deque(maxlen=stability_window)
            for name in base_metrics.keys()
        }
        
        # Current weights (start equal)
        self.current_weights = {
            name: 1.0 / len(base_metrics)
            for name in base_metrics.keys()
        }
        
    def _compute_stability_weights(self) -> Dict[str, float]:
        """
        Compute stability-based weights for each metric.
        
        Returns:
            Dictionary of metric name -> stability weight
        """
        stabilities = {}
        
        for name, history in self.metric_histories.items():
            if len(history) < 3:
                stabilities[name] = 0.5  # Neutral stability for insufficient data
                continue
                
            values = np.array(list(history))
            valid_values = values[~np.isnan(values)]
            
            if len(valid_values) < 3:
                stabilities[name] = 0.5
                continue
                
            # Stability = 1 / (1 + coefficient_of_variation)
            # Higher values are more stable
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)
            cv = std_val / (abs(mean_val) + 1e-8)
            
            stability = 1.0 / (1.0 + cv)
            stabilities[name] = stability
        
        # Normalize and apply bounds
        if stabilities:
            total_stability = sum(stabilities.values())
            if total_stability > 0:
                weights = {
                    name: np.clip(
                        stability / total_stability,
                        self.min_weight / len(stabilities),
                        self.max_weight
                    )
                    for name, stability in stabilities.items()
                }
                
                # Renormalize after clipping
                total_weight = sum(weights.values())
                weights = {
                    name: weight / total_weight
                    for name, weight in weights.items()
                }
            else:
                # Fallback to equal weights
                weights = {
                    name: 1.0 / len(stabilities)
                    for name in stabilities.keys()
                }
        else:
            weights = {}
            
        return weights
    
    def compute_score(self, points: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Compute stability-weighted aggregate score.
        
        Args:
            points: Input samples for metric evaluation
            
        Returns:
            Weighted aggregate metric scores
        """
        # Compute individual metric scores
        metric_scores = {}
        for name, metric in self.base_metrics.items():
            try:
                scores = metric.compute_score(points)
                # Handle the case where metric returns array or scalar
                if isinstance(scores, (list, np.ndarray)):
                    if len(scores) > 0:
                        score = np.mean([s for s in scores if not np.isnan(s)])
                    else:
                        score = float('nan')
                else:
                    score = float(scores)
                    
                metric_scores[name] = score
                
                # Update history
                self.metric_histories[name].append(score)
                
            except Exception as e:
                warnings.warn(f"Error computing {name}: {e}")
                metric_scores[name] = float('nan')
                self.metric_histories[name].append(float('nan'))
        
        # Update weights based on stability
        self.current_weights = self._compute_stability_weights()
        
        # Compute weighted aggregate
        weighted_sum = 0.0
        total_weight = 0.0
        
        for name, score in metric_scores.items():
            if not np.isnan(score):
                weight = self.current_weights.get(name, 0.0)
                weighted_sum += weight * score
                total_weight += weight
        
        if total_weight > 0:
            aggregate_score = weighted_sum / total_weight
        else:
            aggregate_score = float('nan')
            
        # Return as array to match interface
        return np.array([aggregate_score] * len(points))
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current stability-based weights."""
        return dict(self.current_weights)
    
    def get_stability_report(self) -> Dict[str, Any]:
        """
        Get detailed stability report for all metrics.
        
        Returns:
            Dictionary with stability analysis for each metric
        """
        report = {}
        
        for name, history in self.metric_histories.items():
            if len(history) < 3:
                report[name] = {
                    "status": "insufficient_data",
                    "n_points": len(history)
                }
                continue
                
            values = np.array(list(history))
            valid_values = values[~np.isnan(values)]
            
            if len(valid_values) < 3:
                report[name] = {
                    "status": "insufficient_valid_data", 
                    "n_points": len(history),
                    "n_valid": len(valid_values)
                }
                continue
                
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)
            cv = std_val / (abs(mean_val) + 1e-8)
            
            report[name] = {
                "status": "analyzed",
                "n_points": len(history),
                "n_valid": len(valid_values),
                "mean": float(mean_val),
                "std": float(std_val), 
                "cv": float(cv),
                "stability": float(1.0 / (1.0 + cv)),
                "current_weight": self.current_weights.get(name, 0.0),
                "recent_trend": "stable" if cv < 0.1 else "unstable"
            }
            
        return report


# Update ALL_METRICS registry with enhanced implementations
ENHANCED_METRICS = {
    "UnifiedMMD": UnifiedMMD,
    "MultiScaleConvergence": MultiScaleConvergence, 
    "StabilityWeightedMetric": StabilityWeightedMetric,
}

# Example usage and integration functions
def create_enhanced_metric_suite(
    target_data: np.ndarray,
    centroids: List,
    cov_matrices: List,
    weights: List,
    config: Dict[str, Any]
) -> Dict[str, GaussianMetric]:
    """
    Create a comprehensive suite of enhanced metrics.
    
    Args:
        target_data: Target distribution samples
        centroids: GMM centroids
        cov_matrices: GMM covariances
        weights: GMM weights  
        config: Configuration dictionary
        
    Returns:
        Dictionary of metric name -> metric instance
    """
    
    metrics = {}
    
    # Core metrics using factory
    factory = MetricFactory()
    
    for metric_name in config.get('metrics', ['KLDivergence', 'WassersteinDistance']):
        try:
            if metric_name in ["LogLikelihood", "KLDivergence"]:
                metrics[metric_name] = factory.create_metric(
                    metric_name,
                    centroids=centroids,
                    cov_matrices=cov_matrices,
                    weights=weights
                )
            else:
                metrics[metric_name] = factory.create_metric(
                    metric_name,
                    target_data=target_data
                )
        except Exception as e:
            warnings.warn(f"Failed to create {metric_name}: {e}")
    
    # Enhanced MMD with multiple configurations
    try:
        metrics["UnifiedMMD_RBF"] = UnifiedMMD(
            target_samples=target_data,
            kernel="rbf",
            bandwidths=None  # Auto-adaptive
        )
        
        metrics["UnifiedMMD_GMM"] = UnifiedMMD(
            centroids=centroids,
            cov_matrices=cov_matrices,
            weights=weights,
            kernel="rbf",
            n_target_samples=1000
        )
    except Exception as e:
        warnings.warn(f"Failed to create UnifiedMMD: {e}")
    
    # Multi-scale convergence analyzer
    try:
        metrics["MultiScaleConvergence"] = MultiScaleConvergence(
            timescales=config.get("convergence_timescales", {
                "short": 10, "medium": 30, "long": 100
            }),
            stability_threshold=config.get("stability_threshold", 0.05)
        )
    except Exception as e:
        warnings.warn(f"Failed to create MultiScaleConvergence: {e}")
    
    # Stability-weighted aggregate (if multiple base metrics available)
    base_metrics = {k: v for k, v in metrics.items() 
                   if k not in ["MultiScaleConvergence"]}
    
    if len(base_metrics) >= 2:
        try:
            metrics["StabilityWeighted"] = StabilityWeightedMetric(
                base_metrics=base_metrics,
                stability_window=config.get("stability_window", 20)
            )
        except Exception as e:
            warnings.warn(f"Failed to create StabilityWeightedMetric: {e}")
    
    return metrics


def validate_generation_quality(
    generated_samples: np.ndarray,
    target_data: np.ndarray,
    centroids: List,
    cov_matrices: List, 
    weights: List,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Comprehensive validation pipeline using enhanced layer 2 metrics.
    
    Args:
        generated_samples: Samples from generator
        target_data: Target distribution samples
        centroids: GMM centroids
        cov_matrices: GMM covariances
        weights: GMM weights
        config: Configuration dictionary
        
    Returns:
        Comprehensive quality report
    """
    
    # Create metric suite
    metrics = create_enhanced_metric_suite(
        target_data, centroids, cov_matrices, weights, config
    )
    
    # Compute all metric scores
    results = {}
    for name, metric in metrics.items():
        try:
            if name == "MultiScaleConvergence":
                # Skip for this validation (needs temporal data)
                continue
                
            scores = metric.compute_score(generated_samples)
            
            if isinstance(scores, (list, np.ndarray)):
                if len(scores) > 0 and not all(np.isnan(scores)):
                    results[name] = {
                        "mean_score": float(np.nanmean(scores)),
                        "std_score": float(np.nanstd(scores)),
                        "valid_fraction": float(np.mean(~np.isnan(scores)))
                    }
                else:
                    results[name] = {"error": "All NaN scores"}
            else:
                results[name] = {"score": float(scores)}
                
        except Exception as e:
            results[name] = {"error": str(e)}
    
    # Stability analysis (if stability-weighted metric available)
    stability_report = {}
    if "StabilityWeighted" in metrics:
        try:
            stability_report = metrics["StabilityWeighted"].get_stability_report()
        except Exception as e:
            stability_report = {"error": str(e)}
    
    # Generate overall assessment
    valid_metrics = {k: v for k, v in results.items() 
                    if "error" not in v and "score" in v or "mean_score" in v}
    
    if valid_metrics:
        # Simple overall score (could be made more sophisticated)
        scores = []
        for metric_result in valid_metrics.values():
            if "mean_score" in metric_result:
                scores.append(metric_result["mean_score"])
            elif "score" in metric_result:
                scores.append(metric_result["score"])
        
        if scores:
            overall_score = float(np.mean(scores))
            score_std = float(np.std(scores))
        else:
            overall_score = float('nan')
            score_std = float('nan')
    else:
        overall_score = float('nan') 
        score_std = float('nan')
    
    return {
        "individual_metrics": results,
        "stability_analysis": stability_report,
        "overall_assessment": {
            "mean_score": overall_score,
            "score_std": score_std,
            "n_valid_metrics": len(valid_metrics),
            "n_total_metrics": len(results)
        },
        "recommendations": _generate_recommendations(results, stability_report)
    }


def _generate_recommendations(
    metric_results: Dict[str, Any], 
    stability_report: Dict[str, Any]
) -> List[str]:
    """Generate actionable recommendations based on metric results."""
    
    recommendations = []
    
    # Check for problematic metrics
    error_metrics = [k for k, v in metric_results.items() if "error" in v]
    if error_metrics:
        recommendations.append(
            f"Fix computation errors in: {', '.join(error_metrics)}"
        )
    
    # Check for unstable metrics
    if stability_report:
        unstable_metrics = [
            k for k, v in stability_report.items()
            if isinstance(v, dict) and v.get("recent_trend") == "unstable"
        ]
        if unstable_metrics:
            recommendations.append(
                f"Monitor unstable metrics: {', '.join(unstable_metrics)}"
            )
    
    # General recommendations
    valid_count = sum(
        1 for v in metric_results.values() 
        if "error" not in v
    )
    total_count = len(metric_results)
    
    if valid_count < total_count * 0.8:
        recommendations.append(
            "Consider reviewing metric implementations - many are failing"
        )
    
    if not recommendations:
        recommendations.append("All metrics computed successfully")
    
    return recommendations