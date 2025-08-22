"""
Enhanced test data factories and fixtures for comprehensive testing of new statistical metrics.
Provides standardized test data generation for various testing scenarios.
"""

import pytest
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.mixture import GaussianMixture
from scipy import stats
import warnings

from source.metrics import (
    MMDivergence, MMDivergenceFromGMM, MMDDistance, 
    WassersteinDistance, ConvergenceTracker
)


class MetricsTestDataFactory:
    """Comprehensive test data factory for metrics testing."""
    
    def __init__(self, random_seed: Optional[int] = None):
        """Initialize factory with optional random seed for reproducibility."""
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
    
    # ========================================================================
    # Basic Distribution Generators
    # ========================================================================
    
    def generate_gaussian_mixture(
        self,
        n_samples: int = 1000,
        n_components: int = 2,
        dimensions: int = 2,
        separation: float = 3.0,
        noise_level: float = 1.0
    ) -> Dict[str, Any]:
        """Generate Gaussian mixture model samples and parameters."""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        # Generate centroids with specified separation
        centroids = []
        for i in range(n_components):
            angle = 2 * np.pi * i / n_components
            center = separation * np.array([np.cos(angle), np.sin(angle)])
            if dimensions > 2:
                center = np.concatenate([center, np.zeros(dimensions - 2)])
            centroids.append(center[:dimensions])
        
        # Generate covariance matrices
        cov_matrices = []
        for i in range(n_components):
            # Create random positive definite covariance matrix
            A = np.random.randn(dimensions, dimensions) * noise_level
            cov = np.dot(A, A.T) + np.eye(dimensions) * 0.1  # Ensure positive definite
            cov_matrices.append(cov)
        
        # Equal weights by default
        weights = np.ones(n_components) / n_components
        
        # Generate samples
        samples = []
        labels = []
        samples_per_component = n_samples // n_components
        
        for i, (centroid, cov, weight) in enumerate(zip(centroids, cov_matrices, weights)):
            n_comp_samples = int(n_samples * weight) if i < n_components - 1 else n_samples - len(samples)
            comp_samples = np.random.multivariate_normal(centroid, cov, n_comp_samples)
            samples.extend(comp_samples)
            labels.extend([i] * n_comp_samples)
        
        samples = np.array(samples)
        labels = np.array(labels)
        
        # Shuffle samples
        indices = np.random.permutation(len(samples))
        samples = samples[indices]
        labels = labels[indices]
        
        return {
            'samples': samples,
            'labels': labels,
            'centroids': centroids,
            'cov_matrices': cov_matrices,
            'weights': weights.tolist(),
            'n_components': n_components,
            'dimensions': dimensions
        }
    
    def generate_known_distributions(
        self,
        n_samples: int = 1000,
        dimensions: int = 2
    ) -> Dict[str, np.ndarray]:
        """Generate samples from various known distributions."""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        distributions = {}
        
        # Standard normal
        distributions['standard_normal'] = np.random.normal(0, 1, (n_samples, dimensions))
        
        # Shifted normal
        distributions['shifted_normal'] = np.random.normal(2, 1, (n_samples, dimensions))
        
        # Scaled normal
        distributions['scaled_normal'] = np.random.normal(0, 3, (n_samples, dimensions))
        
        # Uniform distribution
        distributions['uniform'] = np.random.uniform(-2, 2, (n_samples, dimensions))
        
        # Exponential (adjusted for multivariate)
        exp_samples = np.random.exponential(1, (n_samples, dimensions))
        distributions['exponential'] = exp_samples - np.mean(exp_samples, axis=0)
        
        # Beta distribution
        beta_samples = np.random.beta(2, 5, (n_samples, dimensions))
        distributions['beta'] = beta_samples
        
        # Multimodal Gaussian
        if dimensions >= 2:
            comp1 = np.random.normal([-2, -2], 0.5, (n_samples//2, 2))
            comp2 = np.random.normal([2, 2], 0.5, (n_samples//2, 2))
            multimodal = np.vstack([comp1, comp2])
            
            if dimensions > 2:
                extra_dims = np.random.normal(0, 1, (n_samples, dimensions - 2))
                multimodal = np.hstack([multimodal, extra_dims])
            
            distributions['multimodal'] = multimodal
        
        # Heavy-tailed distribution (t-distribution)
        t_samples = np.random.standard_t(3, (n_samples, dimensions))
        distributions['heavy_tailed'] = t_samples
        
        return distributions
    
    def generate_adversarial_distributions(
        self,
        n_samples: int = 1000,
        dimensions: int = 2
    ) -> Dict[str, np.ndarray]:
        """Generate challenging/adversarial distributions for stress testing."""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        distributions = {}
        
        # Distribution with outliers
        main_samples = np.random.normal(0, 1, (int(0.9 * n_samples), dimensions))
        outliers = np.random.normal(0, 10, (int(0.1 * n_samples), dimensions))
        distributions['with_outliers'] = np.vstack([main_samples, outliers])
        
        # Clustered distribution
        n_clusters = 5
        cluster_samples = []
        for i in range(n_clusters):
            center = np.random.uniform(-5, 5, dimensions)
            cluster = np.random.normal(center, 0.3, (n_samples // n_clusters, dimensions))
            cluster_samples.append(cluster)
        distributions['clustered'] = np.vstack(cluster_samples)
        
        # Highly skewed distribution
        skewed = np.random.gamma(1, 2, (n_samples, dimensions))
        distributions['skewed'] = skewed
        
        # Ring/donut distribution (for 2D)
        if dimensions >= 2:
            angles = np.random.uniform(0, 2*np.pi, n_samples)
            radii = np.random.normal(3, 0.3, n_samples)
            ring_x = radii * np.cos(angles)
            ring_y = radii * np.sin(angles)
            ring_samples = np.column_stack([ring_x, ring_y])
            
            if dimensions > 2:
                extra_dims = np.random.normal(0, 0.5, (n_samples, dimensions - 2))
                ring_samples = np.hstack([ring_samples, extra_dims])
            
            distributions['ring'] = ring_samples
        
        # Sparse distribution (mostly zeros with some non-zero values)
        sparse = np.zeros((n_samples, dimensions))
        n_nonzero = int(0.1 * n_samples * dimensions)
        indices = np.random.choice(n_samples * dimensions, n_nonzero, replace=False)
        flat_indices = np.unravel_index(indices, (n_samples, dimensions))
        sparse[flat_indices] = np.random.normal(0, 2, n_nonzero)
        distributions['sparse'] = sparse
        
        return distributions
    
    def generate_edge_case_data(
        self,
        n_samples: int = 100,
        dimensions: int = 2
    ) -> Dict[str, np.ndarray]:
        """Generate edge case data for robustness testing."""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        edge_cases = {}
        
        # Constant data (zero variance)
        edge_cases['constant'] = np.full((n_samples, dimensions), 5.0)
        
        # Data with NaN values
        data_with_nan = np.random.randn(n_samples, dimensions)
        nan_indices = np.random.choice(n_samples, size=n_samples//10, replace=False)
        data_with_nan[nan_indices, 0] = np.nan
        edge_cases['with_nan'] = data_with_nan
        
        # Data with infinite values
        data_with_inf = np.random.randn(n_samples, dimensions)
        inf_indices = np.random.choice(n_samples, size=n_samples//20, replace=False)
        data_with_inf[inf_indices, 0] = np.inf
        data_with_inf[inf_indices[:len(inf_indices)//2], 1] = -np.inf
        edge_cases['with_inf'] = data_with_inf
        
        # Very large values
        edge_cases['very_large'] = np.random.randn(n_samples, dimensions) * 1e6
        
        # Very small values
        edge_cases['very_small'] = np.random.randn(n_samples, dimensions) * 1e-10
        
        # Single point
        edge_cases['single_point'] = np.array([[1.0] * dimensions])
        
        # Empty array
        edge_cases['empty'] = np.array([]).reshape(0, dimensions)
        
        # High precision values
        high_precision = np.random.randn(n_samples, dimensions)
        high_precision += np.random.randn(n_samples, dimensions) * 1e-15
        edge_cases['high_precision'] = high_precision
        
        return edge_cases
    
    # ========================================================================
    # Temporal Data for Convergence Testing
    # ========================================================================
    
    def generate_convergence_scenarios(self) -> Dict[str, List[float]]:
        """Generate various convergence scenarios for testing ConvergenceTracker."""
        scenarios = {}
        
        # Quick convergence
        scenarios['quick_convergence'] = [1.0, 0.5, 0.45, 0.44, 0.44, 0.44, 0.44]
        
        # Slow convergence
        scenarios['slow_convergence'] = [1.0, 0.9, 0.8, 0.7, 0.6, 0.55, 0.52, 0.51, 0.505, 0.502]
        
        # No convergence (continuous improvement)
        scenarios['no_convergence'] = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        
        # Oscillating
        scenarios['oscillating'] = [1.0, 0.5, 0.8, 0.4, 0.7, 0.45, 0.6, 0.48, 0.55, 0.47]
        
        # Noisy convergence
        base_values = [1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        noise = np.random.normal(0, 0.02, len(base_values))
        scenarios['noisy_convergence'] = (np.array(base_values) + noise).tolist()
        
        # Plateau then improvement
        scenarios['plateau_then_improve'] = [1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.5, 0.3, 0.1]
        
        # Early stopping scenario
        scenarios['early_stopping'] = [1.0, 0.5, 0.3, 0.29, 0.28, 0.28, 0.28, 0.28]
        
        return scenarios
    
    def generate_loss_patterns(self) -> Dict[str, List[float]]:
        """Generate various loss patterns for stability analysis."""
        patterns = {}
        
        # Stable decreasing
        patterns['stable_decreasing'] = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8]
        
        # Volatile
        patterns['volatile'] = [2.0, 1.0, 2.5, 0.8, 3.0, 0.5, 2.8, 0.3]
        
        # Increasing (divergence)
        patterns['increasing'] = [1.0, 1.2, 1.5, 1.8, 2.1, 2.5, 3.0]
        
        # Cyclical
        n_epochs = 20
        t = np.linspace(0, 4*np.pi, n_epochs)
        patterns['cyclical'] = (1 + 0.5 * np.sin(t)).tolist()
        
        # Step function
        patterns['step_function'] = [2.0]*5 + [1.0]*5 + [0.5]*5 + [0.1]*5
        
        # Exponential decay
        t = np.arange(15)
        patterns['exponential_decay'] = (2.0 * np.exp(-0.3 * t)).tolist()
        
        return patterns
    
    # ========================================================================
    # Metric-Specific Test Data
    # ========================================================================
    
    def create_mmd_test_suite(
        self,
        n_samples_list: List[int] = [100, 500, 1000],
        dimension_list: List[int] = [2, 5, 10],
        bandwidth_configs: List[List[float]] = None
    ) -> Dict[str, Any]:
        """Create comprehensive test suite for MMD metrics."""
        if bandwidth_configs is None:
            bandwidth_configs = [
                [1.0],
                [0.1, 1.0, 10.0],
                [0.01, 0.1, 1.0, 10.0, 100.0]
            ]
        
        test_suite = {
            'target_distributions': {},
            'generated_distributions': {},
            'bandwidth_configs': bandwidth_configs,
            'expected_properties': {}
        }
        
        for n_samples in n_samples_list:
            for dimensions in dimension_list:
                key = f"n{n_samples}_d{dimensions}"
                
                # Target distributions
                gmm_data = self.generate_gaussian_mixture(
                    n_samples=n_samples,
                    n_components=2,
                    dimensions=dimensions
                )
                test_suite['target_distributions'][key] = gmm_data
                
                # Generated distributions with known relationships
                target_samples = gmm_data['samples']
                
                # Identical (should give MMD ≈ 0)
                test_suite['generated_distributions'][f"{key}_identical"] = {
                    'samples': target_samples.copy(),
                    'expected_mmd_range': (0.0, 0.1),
                    'description': 'Identical distribution'
                }
                
                # Shifted (should give positive MMD)
                shifted_samples = target_samples + np.random.normal(0, 0.1, (1, dimensions))
                test_suite['generated_distributions'][f"{key}_shifted"] = {
                    'samples': shifted_samples,
                    'expected_mmd_range': (0.1, 2.0),
                    'description': 'Slightly shifted distribution'
                }
                
                # Scaled (should give positive MMD)
                scaled_samples = target_samples * 1.5
                test_suite['generated_distributions'][f"{key}_scaled"] = {
                    'samples': scaled_samples,
                    'expected_mmd_range': (0.2, 5.0),
                    'description': 'Scaled distribution'
                }
        
        return test_suite
    
    def create_wasserstein_test_suite(
        self,
        n_samples_list: List[int] = [100, 500, 1000],
        dimension_list: List[int] = [1, 2, 5],
        aggregation_methods: List[str] = ['mean', 'max', 'sum']
    ) -> Dict[str, Any]:
        """Create comprehensive test suite for Wasserstein distance."""
        test_suite = {
            'target_distributions': {},
            'generated_distributions': {},
            'aggregation_methods': aggregation_methods,
            'theoretical_comparisons': {}
        }
        
        for n_samples in n_samples_list:
            for dimensions in dimension_list:
                key = f"n{n_samples}_d{dimensions}"
                
                # Create target distribution
                if dimensions == 1:
                    target_samples = np.random.normal(0, 1, n_samples).reshape(-1, 1)
                else:
                    target_samples = np.random.normal(0, 1, (n_samples, dimensions))
                
                test_suite['target_distributions'][key] = target_samples
                
                # Test cases with known Wasserstein distances
                
                # Identical
                test_suite['generated_distributions'][f"{key}_identical"] = {
                    'samples': target_samples.copy(),
                    'expected_distance_range': (0.0, 0.1),
                    'description': 'Identical distribution'
                }
                
                # Unit shift
                if dimensions == 1:
                    shifted_samples = target_samples + 1.0
                    theoretical_distance = 1.0  # For Gaussian with same variance
                else:
                    shifted_samples = target_samples + np.ones((1, dimensions))
                    theoretical_distance = None  # More complex in higher dimensions
                
                test_suite['generated_distributions'][f"{key}_unit_shift"] = {
                    'samples': shifted_samples,
                    'expected_distance_range': (0.5, 2.0),
                    'theoretical_distance': theoretical_distance,
                    'description': 'Unit shift in all dimensions'
                }
                
                # Scale change
                scaled_samples = target_samples * 2.0
                test_suite['generated_distributions'][f"{key}_scale_2x"] = {
                    'samples': scaled_samples,
                    'expected_distance_range': (0.5, 5.0),
                    'description': '2x scaling'
                }
        
        return test_suite
    
    # ========================================================================
    # Statistical Validation Data
    # ========================================================================
    
    def create_statistical_validation_suite(
        self,
        n_bootstrap: int = 100,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Create data for statistical validation of metrics."""
        validation_suite = {
            'distribution_pairs': {},
            'bootstrap_config': {
                'n_bootstrap': n_bootstrap,
                'confidence_level': confidence_level
            },
            'expected_properties': {}
        }
        
        # Known distribution pairs for validation
        pairs = [
            ('standard_normal', 'shifted_normal', 'location_difference'),
            ('standard_normal', 'scaled_normal', 'scale_difference'),
            ('uniform', 'standard_normal', 'shape_difference'),
            ('multimodal', 'standard_normal', 'modality_difference')
        ]
        
        known_dists = self.generate_known_distributions(n_samples=500, dimensions=2)
        
        for dist1_name, dist2_name, difference_type in pairs:
            if dist1_name in known_dists and dist2_name in known_dists:
                pair_key = f"{dist1_name}_vs_{dist2_name}"
                validation_suite['distribution_pairs'][pair_key] = {
                    'dist1': known_dists[dist1_name],
                    'dist2': known_dists[dist2_name],
                    'difference_type': difference_type,
                    'expected_distance_sign': 'positive'  # All should be > 0
                }
        
        # Identical distributions (should give distance ≈ 0)
        for dist_name, samples in known_dists.items():
            if len(samples) > 200:
                # Split into two halves
                half = len(samples) // 2
                validation_suite['distribution_pairs'][f"{dist_name}_self"] = {
                    'dist1': samples[:half],
                    'dist2': samples[half:],
                    'difference_type': 'identical',
                    'expected_distance_sign': 'zero'
                }
        
        return validation_suite
    
    # ========================================================================
    # Performance Testing Data
    # ========================================================================
    
    def create_performance_test_data(
        self,
        size_scales: List[str] = ['small', 'medium', 'large'],
        complexity_levels: List[str] = ['simple', 'moderate', 'complex']
    ) -> Dict[str, Any]:
        """Create data for performance testing."""
        size_configs = {
            'small': {'n_samples': 100, 'max_dimensions': 5},
            'medium': {'n_samples': 1000, 'max_dimensions': 10},
            'large': {'n_samples': 5000, 'max_dimensions': 20}
        }
        
        complexity_configs = {
            'simple': {'n_components': 1, 'n_bandwidths': 1},
            'moderate': {'n_components': 3, 'n_bandwidths': 3},
            'complex': {'n_components': 5, 'n_bandwidths': 10}
        }
        
        performance_data = {}
        
        for size_scale in size_scales:
            for complexity in complexity_levels:
                key = f"{size_scale}_{complexity}"
                size_config = size_configs[size_scale]
                complexity_config = complexity_configs[complexity]
                
                # Generate data
                gmm_data = self.generate_gaussian_mixture(
                    n_samples=size_config['n_samples'],
                    n_components=complexity_config['n_components'],
                    dimensions=min(size_config['max_dimensions'], 10)
                )
                
                performance_data[key] = {
                    'target_samples': gmm_data['samples'],
                    'generated_samples': np.random.randn(
                        size_config['n_samples'] // 2,
                        gmm_data['dimensions']
                    ),
                    'gmm_params': {
                        'centroids': gmm_data['centroids'],
                        'cov_matrices': gmm_data['cov_matrices'],
                        'weights': gmm_data['weights']
                    },
                    'bandwidths': np.logspace(-1, 1, complexity_config['n_bandwidths']).tolist(),
                    'expected_max_time': self._estimate_max_time(size_scale, complexity),
                    'expected_max_memory_mb': self._estimate_max_memory(size_scale, complexity)
                }
        
        return performance_data
    
    def _estimate_max_time(self, size_scale: str, complexity: str) -> float:
        """Estimate maximum expected computation time."""
        base_times = {'small': 1.0, 'medium': 5.0, 'large': 20.0}
        complexity_multipliers = {'simple': 1.0, 'moderate': 2.0, 'complex': 5.0}
        
        return base_times[size_scale] * complexity_multipliers[complexity]
    
    def _estimate_max_memory(self, size_scale: str, complexity: str) -> float:
        """Estimate maximum expected memory usage in MB."""
        base_memory = {'small': 10, 'medium': 50, 'large': 200}
        complexity_multipliers = {'simple': 1.0, 'moderate': 2.0, 'complex': 3.0}
        
        return base_memory[size_scale] * complexity_multipliers[complexity]


# ========================================================================
# Pytest Fixtures
# ========================================================================

@pytest.fixture
def metrics_test_factory():
    """Provide MetricsTestDataFactory instance."""
    return MetricsTestDataFactory(random_seed=42)

@pytest.fixture
def comprehensive_test_data(metrics_test_factory):
    """Provide comprehensive test data for all metrics."""
    return {
        'known_distributions': metrics_test_factory.generate_known_distributions(),
        'adversarial_distributions': metrics_test_factory.generate_adversarial_distributions(),
        'edge_cases': metrics_test_factory.generate_edge_case_data(),
        'convergence_scenarios': metrics_test_factory.generate_convergence_scenarios(),
        'loss_patterns': metrics_test_factory.generate_loss_patterns()
    }

@pytest.fixture
def mmd_test_suite(metrics_test_factory):
    """Provide comprehensive MMD test suite."""
    return metrics_test_factory.create_mmd_test_suite()

@pytest.fixture
def wasserstein_test_suite(metrics_test_factory):
    """Provide comprehensive Wasserstein test suite."""
    return metrics_test_factory.create_wasserstein_test_suite()

@pytest.fixture
def statistical_validation_suite(metrics_test_factory):
    """Provide statistical validation test suite."""
    return metrics_test_factory.create_statistical_validation_suite()

@pytest.fixture
def performance_test_data(metrics_test_factory):
    """Provide performance testing data."""
    return metrics_test_factory.create_performance_test_data()

@pytest.fixture
def gaussian_mixture_factory():
    """Provide factory function for Gaussian mixtures."""
    def _factory(n_samples=1000, n_components=2, dimensions=2, **kwargs):
        factory = MetricsTestDataFactory(random_seed=42)
        return factory.generate_gaussian_mixture(
            n_samples=n_samples,
            n_components=n_components,
            dimensions=dimensions,
            **kwargs
        )
    return _factory

@pytest.fixture
def metric_instances(comprehensive_test_data):
    """Provide pre-configured metric instances for testing."""
    target_samples = comprehensive_test_data['known_distributions']['standard_normal']
    
    # GMM parameters for metrics that need them
    gmm_params = {
        'centroids': [[0, 0], [2, 2]],
        'cov_matrices': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]],
        'weights': [0.5, 0.5]
    }
    
    return {
        'mmd_divergence': MMDivergence(target_samples, bandwidths=[0.5, 1.0, 2.0]),
        'mmd_from_gmm': MMDivergenceFromGMM(**gmm_params, n_target_samples=500),
        'mmd_distance': MMDDistance(torch.from_numpy(target_samples).float()),
        'wasserstein_mean': WassersteinDistance(target_samples, aggregation="mean"),
        'wasserstein_max': WassersteinDistance(target_samples, aggregation="max"),
        'convergence_tracker': ConvergenceTracker(patience=5, min_delta=0.01)
    }

@pytest.fixture(params=['cpu', 'cuda'])
def device_config(request):
    """Provide device configuration for testing."""
    device_name = request.param
    if device_name == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(device_name)

@pytest.fixture
def reproducible_random_state():
    """Provide reproducible random state context manager."""
    class ReproducibleRandom:
        def __init__(self, seed=42):
            self.seed = seed
            self.np_state = None
            self.torch_state = None
        
        def __enter__(self):
            # Save current states
            self.np_state = np.random.get_state()
            self.torch_state = torch.get_rng_state()
            
            # Set reproducible states
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            # Restore original states
            np.random.set_state(self.np_state)
            torch.set_rng_state(self.torch_state)
    
    return ReproducibleRandom

@pytest.fixture
def benchmark_comparison_data():
    """Provide data for benchmarking against existing metrics."""
    factory = MetricsTestDataFactory(random_seed=42)
    
    # Generate data for comparison with existing metrics
    gmm_data = factory.generate_gaussian_mixture(n_samples=1000, n_components=2)
    
    return {
        'target_samples': gmm_data['samples'],
        'generated_samples': np.random.randn(500, 2) + 0.5,  # Shifted samples
        'gmm_params': {
            'centroids': gmm_data['centroids'],
            'cov_matrices': gmm_data['cov_matrices'],
            'weights': gmm_data['weights']
        },
        'expected_relationships': {
            'all_metrics_positive': True,  # All distances should be > 0
            'mmd_vs_wasserstein_correlation': 0.3,  # Minimum expected correlation
            'metric_ordering_consistency': 0.6  # Minimum Kendall's tau
        }
    }


if __name__ == "__main__":
    # Example usage and testing of the factory
    factory = MetricsTestDataFactory(random_seed=42)
    
    # Generate some test data
    gmm_data = factory.generate_gaussian_mixture()
    print(f"Generated GMM with {len(gmm_data['samples'])} samples")
    
    known_dists = factory.generate_known_distributions()
    print(f"Generated {len(known_dists)} known distributions")
    
    convergence_scenarios = factory.generate_convergence_scenarios()
    print(f"Generated {len(convergence_scenarios)} convergence scenarios")
    
    mmd_suite = factory.create_mmd_test_suite()
    print(f"Created MMD test suite with {len(mmd_suite['target_distributions'])} target distributions")