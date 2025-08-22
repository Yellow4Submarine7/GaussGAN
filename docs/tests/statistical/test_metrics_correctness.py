"""
Statistical validation tests for new metrics mathematical correctness.
Tests theoretical properties, known distribution comparisons, and metric behavior.
"""

import pytest
import torch
import numpy as np
import warnings
from scipy import stats
from scipy.spatial.distance import pdist
from sklearn.mixture import GaussianMixture
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt

from source.metrics import (
    MMDivergence, MMDivergenceFromGMM, MMDDistance, 
    WassersteinDistance, ConvergenceTracker
)


class StatisticalValidator:
    """Utility class for statistical validation of metrics."""
    
    @staticmethod
    def generate_known_distributions(n_samples: int = 1000) -> Dict[str, np.ndarray]:
        """Generate samples from known distributions for testing."""
        np.random.seed(42)  # For reproducibility
        
        distributions = {
            'standard_normal': np.random.normal(0, 1, (n_samples, 2)),
            'shifted_normal': np.random.normal(2, 1, (n_samples, 2)),
            'scaled_normal': np.random.normal(0, 2, (n_samples, 2)),
            'uniform': np.random.uniform(-2, 2, (n_samples, 2)),
            'multimodal': np.vstack([
                np.random.normal([-2, -2], 0.5, (n_samples//2, 2)),
                np.random.normal([2, 2], 0.5, (n_samples//2, 2))
            ])
        }
        
        return distributions
    
    @staticmethod
    def create_gmm_from_samples(samples: np.ndarray, n_components: int = 2) -> GaussianMixture:
        """Fit a GMM to samples and return parameters."""
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(samples)
        return gmm
    
    @staticmethod
    def compute_theoretical_wasserstein_1d(mu1: float, sigma1: float, 
                                         mu2: float, sigma2: float) -> float:
        """Compute theoretical Wasserstein distance between two 1D Gaussians."""
        return abs(mu1 - mu2) + abs(sigma1 - sigma2)
    
    @staticmethod
    def bootstrap_confidence_interval(metric_func, data1: np.ndarray, data2: np.ndarray,
                                    n_bootstrap: int = 100, confidence: float = 0.95) -> Tuple[float, float]:
        """Compute bootstrap confidence interval for a metric."""
        bootstrap_values = []
        
        for _ in range(n_bootstrap):
            # Bootstrap resample
            idx1 = np.random.choice(len(data1), len(data1), replace=True)
            idx2 = np.random.choice(len(data2), len(data2), replace=True)
            
            sample1 = data1[idx1]
            sample2 = data2[idx2]
            
            try:
                value = metric_func(sample1, sample2)
                if np.isfinite(value):
                    bootstrap_values.append(value)
            except Exception:
                continue
        
        if len(bootstrap_values) == 0:
            return np.nan, np.nan
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_values, 100 * alpha / 2)
        upper = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))
        
        return lower, upper


@pytest.mark.statistical
class TestMMDCorrectness:
    """Statistical correctness tests for MMD metrics."""
    
    def test_mmd_identical_distributions_property(self):
        """Test that MMD is zero for identical distributions."""
        np.random.seed(42)
        samples = np.random.randn(500, 2)
        
        mmd = MMDivergence(samples, bandwidths=[1.0])
        scores = mmd.compute_score(samples)
        
        # For identical distributions, MMD should be close to zero
        mean_score = np.mean(scores)
        assert mean_score < 0.1, f"MMD for identical distributions: {mean_score}"
    
    def test_mmd_symmetry_property(self):
        """Test MMD symmetry property: MMD(P, Q) ≈ MMD(Q, P)."""
        np.random.seed(42)
        samples_p = np.random.normal(0, 1, (300, 2))
        samples_q = np.random.normal(1, 1, (300, 2))
        
        mmd_pq = MMDivergence(samples_p, bandwidths=[1.0])
        mmd_qp = MMDivergence(samples_q, bandwidths=[1.0])
        
        score_pq = np.mean(mmd_pq.compute_score(samples_q))
        score_qp = np.mean(mmd_qp.compute_score(samples_p))
        
        # Should be approximately symmetric
        relative_diff = abs(score_pq - score_qp) / max(score_pq, score_qp)
        assert relative_diff < 0.3, f"MMD asymmetry: {score_pq:.3f} vs {score_qp:.3f}"
    
    def test_mmd_triangle_inequality(self):
        """Test MMD triangle inequality: MMD(P, R) ≤ MMD(P, Q) + MMD(Q, R)."""
        np.random.seed(42)
        # Create three distributions along a line
        samples_p = np.random.normal([0, 0], 0.5, (200, 2))
        samples_q = np.random.normal([1, 1], 0.5, (200, 2))
        samples_r = np.random.normal([2, 2], 0.5, (200, 2))
        
        mmd_p = MMDivergence(samples_p, bandwidths=[1.0])
        mmd_q = MMDivergence(samples_q, bandwidths=[1.0])
        
        mmd_pr = np.mean(mmd_p.compute_score(samples_r))
        mmd_pq = np.mean(mmd_p.compute_score(samples_q))
        mmd_qr = np.mean(mmd_q.compute_score(samples_r))
        
        # Triangle inequality
        assert mmd_pr <= mmd_pq + mmd_qr + 0.1, \
            f"Triangle inequality violated: {mmd_pr:.3f} > {mmd_pq:.3f} + {mmd_qr:.3f}"
    
    def test_mmd_scale_invariance_violation(self):
        """Test that MMD is NOT scale invariant (expected behavior)."""
        np.random.seed(42)
        samples_base = np.random.randn(300, 2)
        samples_scaled = samples_base * 2  # Scale by 2
        
        mmd = MMDivergence(samples_base, bandwidths=[1.0])
        
        score_identity = np.mean(mmd.compute_score(samples_base))
        score_scaled = np.mean(mmd.compute_score(samples_scaled))
        
        # MMD should detect the scaling difference
        assert score_scaled > score_identity + 0.1, \
            f"MMD should detect scaling: {score_identity:.3f} vs {score_scaled:.3f}"
    
    def test_mmd_bandwidth_sensitivity(self):
        """Test MMD sensitivity to bandwidth selection."""
        np.random.seed(42)
        target_samples = np.random.randn(300, 2)
        generated_samples = np.random.randn(300, 2) + 0.5  # Shifted
        
        bandwidths = [0.1, 1.0, 10.0]
        scores = []
        
        for bw in bandwidths:
            mmd = MMDivergence(target_samples, bandwidths=[bw])
            score = np.mean(mmd.compute_score(generated_samples))
            scores.append(score)
        
        # Different bandwidths should give different sensitivities
        assert not np.allclose(scores, scores[0], rtol=0.1), \
            f"MMD scores should vary with bandwidth: {scores}"
        
        # All should detect the difference (be positive)
        assert all(s > 0.1 for s in scores), f"All MMD scores should be positive: {scores}"
    
    def test_mmd_convergence_with_sample_size(self):
        """Test MMD convergence properties with increasing sample size."""
        np.random.seed(42)
        
        # Fixed target distribution
        target_samples = np.random.normal(0, 1, (1000, 2))
        mmd = MMDivergence(target_samples, bandwidths=[1.0])
        
        # Test with increasing sample sizes from same distribution
        sample_sizes = [50, 100, 200, 500]
        variances = []
        
        for n in sample_sizes:
            scores = []
            for _ in range(20):  # Multiple trials
                test_samples = np.random.normal(0, 1, (n, 2))
                score = np.mean(mmd.compute_score(test_samples))
                scores.append(score)
            
            variances.append(np.var(scores))
        
        # Variance should generally decrease with sample size
        # (though this is a statistical trend, not strict)
        trend_coefficient = np.corrcoef(sample_sizes, variances)[0, 1]
        assert trend_coefficient < 0.2, \
            f"MMD variance should decrease with sample size, trend: {trend_coefficient:.3f}"
    
    def test_mmd_different_dimensionalities_behavior(self):
        """Test MMD behavior across different dimensionalities."""
        np.random.seed(42)
        
        dimensions = [1, 2, 5, 10]
        mmd_scores = []
        
        for dim in dimensions:
            # Same relative shift in each dimension
            target_samples = np.random.randn(300, dim)
            shifted_samples = np.random.randn(300, dim) + 0.5
            
            mmd = MMDivergence(target_samples, bandwidths=[1.0])
            score = np.mean(mmd.compute_score(shifted_samples))
            mmd_scores.append(score)
        
        # MMD should detect differences in all dimensions
        assert all(s > 0.1 for s in mmd_scores), \
            f"MMD should detect differences in all dimensions: {mmd_scores}"
        
        # Higher dimensions might have different behavior due to curse of dimensionality
        print(f"MMD scores by dimension: {dict(zip(dimensions, mmd_scores))}")


@pytest.mark.statistical
class TestWassersteinCorrectness:
    """Statistical correctness tests for Wasserstein distance."""
    
    def test_wasserstein_identical_distributions_property(self):
        """Test that Wasserstein distance is zero for identical distributions."""
        np.random.seed(42)
        samples = np.random.randn(500, 2)
        
        wd = WassersteinDistance(samples, aggregation="mean")
        distance = wd.compute_score(samples)
        
        # Should be very close to zero
        assert distance < 0.05, f"Wasserstein distance for identical distributions: {distance}"
    
    def test_wasserstein_symmetry_property(self):
        """Test Wasserstein distance symmetry."""
        np.random.seed(42)
        samples1 = np.random.normal(0, 1, (300, 2))
        samples2 = np.random.normal(1, 1, (300, 2))
        
        wd1 = WassersteinDistance(samples1, aggregation="mean")
        wd2 = WassersteinDistance(samples2, aggregation="mean")
        
        dist_12 = wd1.compute_score(samples2)
        dist_21 = wd2.compute_score(samples1)
        
        # Should be symmetric
        relative_diff = abs(dist_12 - dist_21) / max(dist_12, dist_21)
        assert relative_diff < 0.2, f"Wasserstein asymmetry: {dist_12:.3f} vs {dist_21:.3f}"
    
    def test_wasserstein_triangle_inequality(self):
        """Test Wasserstein triangle inequality."""
        np.random.seed(42)
        samples_a = np.random.normal([0, 0], 1, (200, 2))
        samples_b = np.random.normal([1, 1], 1, (200, 2))
        samples_c = np.random.normal([2, 2], 1, (200, 2))
        
        wd_a = WassersteinDistance(samples_a, aggregation="mean")
        wd_b = WassersteinDistance(samples_b, aggregation="mean")
        
        dist_ac = wd_a.compute_score(samples_c)
        dist_ab = wd_a.compute_score(samples_b)
        dist_bc = wd_b.compute_score(samples_c)
        
        # Triangle inequality
        assert dist_ac <= dist_ab + dist_bc + 0.1, \
            f"Triangle inequality violated: {dist_ac:.3f} > {dist_ab:.3f} + {dist_bc:.3f}"
    
    def test_wasserstein_translation_invariance(self):
        """Test Wasserstein translation invariance property."""
        np.random.seed(42)
        samples1 = np.random.randn(300, 2)
        samples2 = np.random.randn(300, 2) * 1.5  # Different scale
        
        # Translate both by same amount
        translation = np.array([10, -5])
        samples1_translated = samples1 + translation
        samples2_translated = samples2 + translation
        
        wd_original = WassersteinDistance(samples1, aggregation="mean")
        wd_translated = WassersteinDistance(samples1_translated, aggregation="mean")
        
        dist_original = wd_original.compute_score(samples2)
        dist_translated = wd_translated.compute_score(samples2_translated)
        
        # Should be approximately equal
        relative_diff = abs(dist_original - dist_translated) / max(dist_original, dist_translated)
        assert relative_diff < 0.1, \
            f"Translation should not affect Wasserstein: {dist_original:.3f} vs {dist_translated:.3f}"
    
    def test_wasserstein_scaling_behavior(self):
        """Test Wasserstein distance scaling behavior."""
        np.random.seed(42)
        samples1 = np.random.randn(300, 2)
        samples2 = np.random.randn(300, 2) + 1  # Unit shift
        
        # Scale everything by factor
        scale_factor = 3
        samples1_scaled = samples1 * scale_factor
        samples2_scaled = samples2 * scale_factor
        
        wd_original = WassersteinDistance(samples1, aggregation="mean")
        wd_scaled = WassersteinDistance(samples1_scaled, aggregation="mean")
        
        dist_original = wd_original.compute_score(samples2)
        dist_scaled = wd_scaled.compute_score(samples2_scaled)
        
        # Distance should scale approximately linearly
        expected_scaled_distance = dist_original * scale_factor
        relative_error = abs(dist_scaled - expected_scaled_distance) / expected_scaled_distance
        
        assert relative_error < 0.3, \
            f"Scaling behavior: {dist_scaled:.3f} vs expected {expected_scaled_distance:.3f}"
    
    def test_wasserstein_1d_theoretical_comparison(self):
        """Test 1D Wasserstein against theoretical values."""
        # Two 1D Gaussians with known parameters
        mu1, sigma1 = 0, 1
        mu2, sigma2 = 2, 1.5
        
        # Generate samples
        np.random.seed(42)
        samples1 = np.random.normal(mu1, sigma1, 1000)
        samples2 = np.random.normal(mu2, sigma2, 1000)
        
        # Compute empirical Wasserstein
        wd = WassersteinDistance(samples1, aggregation="mean")
        empirical_distance = wd.compute_score(samples2)
        
        # Compute theoretical Wasserstein for 1D Gaussians
        theoretical_distance = StatisticalValidator.compute_theoretical_wasserstein_1d(
            mu1, sigma1, mu2, sigma2
        )
        
        # Should be reasonably close
        relative_error = abs(empirical_distance - theoretical_distance) / theoretical_distance
        assert relative_error < 0.2, \
            f"Empirical {empirical_distance:.3f} vs theoretical {theoretical_distance:.3f}"
    
    def test_wasserstein_aggregation_consistency(self):
        """Test consistency of different aggregation methods."""
        np.random.seed(42)
        target_samples = np.random.randn(300, 3)
        generated_samples = np.random.randn(300, 3) + [1, 0.5, 0]  # Different shifts per dim
        
        aggregations = ["mean", "max", "sum"]
        distances = {}
        
        for agg in aggregations:
            wd = WassersteinDistance(target_samples, aggregation=agg)
            distances[agg] = wd.compute_score(generated_samples)
        
        # Basic consistency checks
        assert distances["max"] >= distances["mean"], \
            f"Max should be >= mean: {distances['max']:.3f} vs {distances['mean']:.3f}"
        
        assert distances["sum"] >= distances["max"], \
            f"Sum should be >= max: {distances['sum']:.3f} vs {distances['max']:.3f}"
        
        # All should be positive for different distributions
        assert all(d > 0.1 for d in distances.values()), \
            f"All distances should be positive: {distances}"


@pytest.mark.statistical
class TestConvergenceTrackerCorrectness:
    """Statistical correctness tests for ConvergenceTracker."""
    
    def test_convergence_detection_accuracy(self):
        """Test accuracy of convergence detection."""
        # Test scenarios with known convergence patterns
        scenarios = [
            {
                'name': 'quick_convergence',
                'values': [1.0, 0.5, 0.45, 0.44, 0.44, 0.44],  # Quick convergence
                'patience': 2,
                'min_delta': 0.02,
                'should_converge': True,
                'expected_epoch': 5
            },
            {
                'name': 'slow_improvement',
                'values': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],  # Continuous improvement
                'patience': 2,
                'min_delta': 0.05,
                'should_converge': False,
                'expected_epoch': None
            },
            {
                'name': 'noisy_convergence',
                'values': [1.0, 0.5, 0.52, 0.48, 0.51, 0.49],  # Noisy around 0.5
                'patience': 3,
                'min_delta': 0.05,
                'should_converge': True,
                'expected_epoch': 5
            }
        ]
        
        for scenario in scenarios:
            tracker = ConvergenceTracker(
                patience=scenario['patience'],
                min_delta=scenario['min_delta'],
                monitor_metric="KLDivergence"
            )
            
            convergence_epoch = None
            for epoch, value in enumerate(scenario['values']):
                result = tracker.update(epoch=epoch, metrics={"KLDivergence": value})
                
                if result['converged'] and convergence_epoch is None:
                    convergence_epoch = epoch
            
            if scenario['should_converge']:
                assert tracker.converged, f"Scenario {scenario['name']} should converge"
                if scenario['expected_epoch'] is not None:
                    assert convergence_epoch == scenario['expected_epoch'], \
                        f"Expected convergence at epoch {scenario['expected_epoch']}, got {convergence_epoch}"
            else:
                assert not tracker.converged, f"Scenario {scenario['name']} should not converge"
    
    def test_convergence_robustness_to_noise(self):
        """Test convergence detection robustness to noisy metrics."""
        true_convergence_value = 0.5
        noise_levels = [0.01, 0.05, 0.1]
        
        for noise_level in noise_levels:
            tracker = ConvergenceTracker(
                patience=5,
                min_delta=noise_level * 3,  # Set threshold above noise
                monitor_metric="KLDivergence"
            )
            
            np.random.seed(42)
            
            # Simulate convergence with noise
            converged = False
            for epoch in range(20):
                # Value converges to true_convergence_value with noise
                noisy_value = true_convergence_value + np.random.normal(0, noise_level)
                result = tracker.update(epoch=epoch, metrics={"KLDivergence": noisy_value})
                
                if result['converged']:
                    converged = True
                    break
            
            # Should eventually converge despite noise
            assert converged, f"Should converge with noise level {noise_level}"
    
    def test_loss_stability_calculation_accuracy(self):
        """Test accuracy of loss stability calculations."""
        tracker = ConvergenceTracker(window_size=5)
        
        # Test with known loss patterns
        stable_losses = [1.0, 1.01, 0.99, 1.02, 0.98, 1.0]  # Low variance
        volatile_losses = [1.0, 0.5, 1.5, 0.3, 1.8, 0.7]    # High variance
        
        # Test stable losses
        for epoch, loss in enumerate(stable_losses):
            tracker.update(epoch=epoch, metrics={}, d_loss=loss)
        
        stable_info = tracker.get_convergence_info(len(stable_losses) - 1)
        stable_std = stable_info.get('d_loss_stability', 0)
        
        # Reset and test volatile losses
        tracker.reset()
        for epoch, loss in enumerate(volatile_losses):
            tracker.update(epoch=epoch, metrics={}, d_loss=loss)
        
        volatile_info = tracker.get_convergence_info(len(volatile_losses) - 1)
        volatile_std = volatile_info.get('d_loss_stability', 0)
        
        # Volatile should have higher standard deviation
        assert volatile_std > stable_std, \
            f"Volatile losses should have higher std: {volatile_std:.3f} vs {stable_std:.3f}"
        
        # Values should be reasonable
        assert stable_std < 0.1, f"Stable std too high: {stable_std:.3f}"
        assert volatile_std > 0.3, f"Volatile std too low: {volatile_std:.3f}"
    
    def test_loss_trend_calculation_accuracy(self):
        """Test accuracy of loss trend calculations."""
        tracker = ConvergenceTracker(window_size=4)
        
        # Decreasing trend (improvement)
        decreasing_losses = [2.0, 1.8, 1.6, 1.4, 1.2]
        for epoch, loss in enumerate(decreasing_losses):
            tracker.update(epoch=epoch, metrics={}, g_loss=loss)
        
        decreasing_info = tracker.get_convergence_info(len(decreasing_losses) - 1)
        decreasing_trend = decreasing_info.get('g_loss_trend', 0)
        
        # Reset and test increasing trend
        tracker.reset()
        increasing_losses = [1.0, 1.2, 1.4, 1.6, 1.8]
        for epoch, loss in enumerate(increasing_losses):
            tracker.update(epoch=epoch, metrics={}, g_loss=loss)
        
        increasing_info = tracker.get_convergence_info(len(increasing_losses) - 1)
        increasing_trend = increasing_info.get('g_loss_trend', 0)
        
        # Trends should have correct signs
        assert decreasing_trend < 0, f"Decreasing trend should be negative: {decreasing_trend:.3f}"
        assert increasing_trend > 0, f"Increasing trend should be positive: {increasing_trend:.3f}"
        
        # Magnitudes should be reasonable
        expected_decreasing = -0.2  # Average decrease per epoch
        expected_increasing = 0.2   # Average increase per epoch
        
        assert abs(decreasing_trend - expected_decreasing) < 0.1, \
            f"Decreasing trend magnitude: {decreasing_trend:.3f} vs expected {expected_decreasing:.3f}"
        
        assert abs(increasing_trend - expected_increasing) < 0.1, \
            f"Increasing trend magnitude: {increasing_trend:.3f} vs expected {expected_increasing:.3f}"


@pytest.mark.statistical
class TestMetricComparativeValidation:
    """Comparative validation tests between metrics."""
    
    def test_mmd_vs_wasserstein_correlation(self):
        """Test correlation between MMD and Wasserstein distance."""
        validator = StatisticalValidator()
        distributions = validator.generate_known_distributions(n_samples=500)
        
        # Use standard normal as reference
        reference_dist = distributions['standard_normal']
        
        mmd_scores = []
        wasserstein_scores = []
        
        mmd = MMDivergence(reference_dist, bandwidths=[1.0])
        wd = WassersteinDistance(reference_dist, aggregation="mean")
        
        for dist_name, samples in distributions.items():
            if dist_name != 'standard_normal':  # Skip self-comparison
                mmd_score = np.mean(mmd.compute_score(samples))
                wd_score = wd.compute_score(samples)
                
                mmd_scores.append(mmd_score)
                wasserstein_scores.append(wd_score)
        
        # Should be positively correlated (both measure distributional differences)
        correlation = np.corrcoef(mmd_scores, wasserstein_scores)[0, 1]
        assert correlation > 0.3, f"MMD and Wasserstein should be correlated: {correlation:.3f}"
        
        print(f"MMD vs Wasserstein correlation: {correlation:.3f}")
    
    def test_metric_ordering_consistency(self):
        """Test that different metrics give consistent ordering of distribution differences."""
        # Create distributions with known ordering of similarity to reference
        np.random.seed(42)
        reference = np.random.normal(0, 1, (500, 2))
        
        # Increasing levels of difference from reference
        distributions = {
            'very_similar': np.random.normal(0.1, 1, (500, 2)),      # Small shift
            'moderately_similar': np.random.normal(0.5, 1, (500, 2)), # Medium shift
            'quite_different': np.random.normal(1.5, 1, (500, 2)),    # Large shift
            'very_different': np.random.normal(3.0, 2, (500, 2))      # Large shift + scale
        }
        
        # Compute metrics
        mmd = MMDivergence(reference, bandwidths=[1.0])
        wd = WassersteinDistance(reference, aggregation="mean")
        
        metric_scores = {}
        
        for dist_name, samples in distributions.items():
            mmd_score = np.mean(mmd.compute_score(samples))
            wd_score = wd.compute_score(samples)
            
            metric_scores[dist_name] = {
                'mmd': mmd_score,
                'wasserstein': wd_score
            }
        
        # Check ordering consistency
        dist_names = list(distributions.keys())
        
        # Get rankings for each metric
        mmd_ranking = sorted(dist_names, key=lambda x: metric_scores[x]['mmd'])
        wd_ranking = sorted(dist_names, key=lambda x: metric_scores[x]['wasserstein'])
        
        # Rankings should be similar (Kendall's tau)
        from scipy.stats import kendalltau
        tau, p_value = kendalltau(
            [mmd_ranking.index(name) for name in dist_names],
            [wd_ranking.index(name) for name in dist_names]
        )
        
        assert tau > 0.5, f"Metric rankings should be consistent: tau={tau:.3f}, p={p_value:.3f}"
        
        print(f"MMD ranking: {mmd_ranking}")
        print(f"Wasserstein ranking: {wd_ranking}")
        print(f"Kendall's tau: {tau:.3f}")
    
    def test_metric_sensitivity_analysis(self):
        """Test metric sensitivity to different types of distribution changes."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, (500, 2))
        
        # Different types of changes
        changes = {
            'location_shift': np.random.normal(1, 1, (500, 2)),       # Mean shift
            'scale_change': np.random.normal(0, 2, (500, 2)),         # Variance change
            'shape_change': np.random.exponential(1, (500, 2)) - 1,   # Distribution shape
            'outliers': np.concatenate([
                np.random.normal(0, 1, (450, 2)),
                np.random.normal(0, 1, (50, 2)) * 5  # 10% outliers
            ])
        }
        
        metrics = {
            'MMD': MMDivergence(reference, bandwidths=[1.0]),
            'Wasserstein': WassersteinDistance(reference, aggregation="mean")
        }
        
        sensitivity_results = {}
        
        for change_type, changed_samples in changes.items():
            sensitivity_results[change_type] = {}
            
            for metric_name, metric in metrics.items():
                if metric_name == 'MMD':
                    score = np.mean(metric.compute_score(changed_samples))
                else:
                    score = metric.compute_score(changed_samples)
                
                sensitivity_results[change_type][metric_name] = score
        
        # Print results for analysis
        print("\nMetric Sensitivity Analysis:")
        for change_type in changes:
            print(f"{change_type}:")
            for metric_name in metrics:
                score = sensitivity_results[change_type][metric_name]
                print(f"  {metric_name}: {score:.3f}")
        
        # Basic sanity checks
        for change_type in changes:
            for metric_name in metrics:
                score = sensitivity_results[change_type][metric_name]
                assert score > 0.1, f"{metric_name} should detect {change_type}: {score:.3f}"
    
    def test_metric_stability_across_runs(self):
        """Test metric stability across multiple runs with same distributions."""
        n_runs = 10
        n_samples = 300
        
        metrics = {
            'MMD': lambda ref, gen: np.mean(MMDivergence(ref, bandwidths=[1.0]).compute_score(gen)),
            'Wasserstein': lambda ref, gen: WassersteinDistance(ref).compute_score(gen)
        }
        
        stability_results = {metric_name: [] for metric_name in metrics}
        
        for run in range(n_runs):
            np.random.seed(run)  # Different seed each run
            
            reference = np.random.normal(0, 1, (n_samples, 2))
            generated = np.random.normal(0.5, 1, (n_samples, 2))  # Consistent shift
            
            for metric_name, metric_func in metrics.items():
                try:
                    score = metric_func(reference, generated)
                    stability_results[metric_name].append(score)
                except Exception as e:
                    print(f"Error in {metric_name}, run {run}: {e}")
        
        # Analyze stability
        for metric_name, scores in stability_results.items():
            if len(scores) > 0:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                cv = std_score / mean_score if mean_score > 0 else np.inf
                
                print(f"{metric_name} stability: mean={mean_score:.3f}, std={std_score:.3f}, CV={cv:.3f}")
                
                # Coefficient of variation should be reasonable
                assert cv < 0.5, f"{metric_name} is too unstable: CV={cv:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "statistical"])