"""
Performance and regression tests for new statistical metrics.
Tests computational efficiency, scalability, and performance baselines.
"""

import pytest
import torch
import numpy as np
import time
import psutil
import os
from typing import Dict, List, Tuple
import warnings
from contextlib import contextmanager

from source.metrics import (
    MMDivergence, MMDivergenceFromGMM, MMDDistance, 
    WassersteinDistance, ConvergenceTracker,
    LogLikelihood, KLDivergence  # For comparison
)


class PerformanceProfiler:
    """Utility class for profiling performance metrics."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
    
    @contextmanager
    def profile(self):
        """Context manager for profiling execution time and memory."""
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Record initial state
        start_time = time.time()
        start_memory = self.process.memory_info().rss
        start_gpu_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        try:
            yield
        finally:
            # Record final state
            end_time = time.time()
            end_memory = self.process.memory_info().rss
            end_gpu_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            peak_gpu_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            
            # Store results
            self.last_profile = {
                'execution_time': end_time - start_time,
                'memory_delta_mb': (end_memory - start_memory) / (1024 * 1024),
                'gpu_memory_delta_mb': (end_gpu_memory - start_gpu_memory) / (1024 * 1024),
                'peak_gpu_memory_mb': peak_gpu_memory / (1024 * 1024)
            }


@pytest.mark.performance
class TestMetricsPerformance:
    """Performance tests for new statistical metrics."""
    
    @pytest.fixture
    def profiler(self):
        """Provide performance profiler."""
        return PerformanceProfiler()
    
    @pytest.fixture
    def performance_gaussian_params(self):
        """Gaussian parameters for performance testing."""
        return {
            'centroids': [[0, 0], [2, 2], [4, 0], [2, -2]],  # 4 components
            'cov_matrices': [
                [[1, 0], [0, 1]], 
                [[1, 0.5], [0.5, 1]], 
                [[2, 0], [0, 0.5]], 
                [[0.5, -0.2], [-0.2, 1]]
            ],
            'weights': [0.25, 0.25, 0.25, 0.25]
        }
    
    @pytest.mark.parametrize("n_samples", [100, 500, 1000, 2000])
    def test_mmd_divergence_scalability(self, profiler, n_samples):
        """Test MMDivergence scalability with different sample sizes."""
        target_samples = np.random.randn(n_samples, 2)
        generated_samples = np.random.randn(n_samples, 2) + 0.5
        
        mmd = MMDivergence(target_samples, bandwidths=[0.5, 1.0, 2.0])
        
        with profiler.profile():
            scores = mmd.compute_score(generated_samples)
        
        profile = profiler.last_profile
        
        # Performance assertions
        assert profile['execution_time'] < 10.0  # Should complete in 10 seconds
        assert profile['memory_delta_mb'] < 100   # Should not use excessive memory
        
        # Functionality assertions
        assert len(scores) == n_samples
        assert np.all(scores >= 0)
        
        print(f"MMD (n={n_samples}): {profile['execution_time']:.3f}s, "
              f"{profile['memory_delta_mb']:.1f}MB")
    
    @pytest.mark.parametrize("n_dimensions", [2, 5, 10, 20])
    def test_mmd_divergence_dimensionality_performance(self, profiler, n_dimensions):
        """Test MMDivergence performance with different dimensionalities."""
        n_samples = 500
        target_samples = np.random.randn(n_samples, n_dimensions)
        generated_samples = np.random.randn(n_samples, n_dimensions)
        
        mmd = MMDivergence(target_samples, bandwidths=[1.0])
        
        with profiler.profile():
            scores = mmd.compute_score(generated_samples)
        
        profile = profiler.last_profile
        
        # Performance should scale reasonably with dimensions
        assert profile['execution_time'] < 15.0
        assert profile['memory_delta_mb'] < 200
        
        print(f"MMD (d={n_dimensions}): {profile['execution_time']:.3f}s, "
              f"{profile['memory_delta_mb']:.1f}MB")
    
    @pytest.mark.parametrize("n_bandwidths", [1, 3, 5, 10])
    def test_mmd_divergence_bandwidth_performance(self, profiler, n_bandwidths):
        """Test MMDivergence performance with different numbers of bandwidths."""
        target_samples = np.random.randn(1000, 2)
        generated_samples = np.random.randn(1000, 2)
        
        # Create different numbers of bandwidths
        bandwidths = np.logspace(-1, 1, n_bandwidths)  # From 0.1 to 10
        mmd = MMDivergence(target_samples, bandwidths=bandwidths)
        
        with profiler.profile():
            scores = mmd.compute_score(generated_samples)
        
        profile = profiler.last_profile
        
        # Performance should scale linearly with number of bandwidths
        expected_max_time = 2.0 * n_bandwidths  # Rough linear scaling
        assert profile['execution_time'] < expected_max_time
        
        print(f"MMD (bandwidths={n_bandwidths}): {profile['execution_time']:.3f}s")
    
    @pytest.mark.parametrize("n_samples", [100, 500, 1000, 2000])
    def test_wasserstein_distance_scalability(self, profiler, n_samples):
        """Test WassersteinDistance scalability with different sample sizes."""
        target_samples = np.random.randn(n_samples, 2)
        generated_samples = np.random.randn(n_samples, 2) + 0.5
        
        wd = WassersteinDistance(target_samples, aggregation="mean")
        
        with profiler.profile():
            distance = wd.compute_score(generated_samples)
        
        profile = profiler.last_profile
        
        # Wasserstein distance is generally efficient
        assert profile['execution_time'] < 5.0
        assert profile['memory_delta_mb'] < 50
        
        assert isinstance(distance, float)
        assert distance >= 0
        
        print(f"Wasserstein (n={n_samples}): {profile['execution_time']:.3f}s, "
              f"{profile['memory_delta_mb']:.1f}MB")
    
    @pytest.mark.parametrize("n_dimensions", [1, 2, 5, 10, 20])
    def test_wasserstein_distance_dimensionality_performance(self, profiler, n_dimensions):
        """Test WassersteinDistance performance with different dimensionalities."""
        n_samples = 500
        target_samples = np.random.randn(n_samples, n_dimensions)
        generated_samples = np.random.randn(n_samples, n_dimensions)
        
        wd = WassersteinDistance(target_samples, aggregation="mean")
        
        with profiler.profile():
            distance = wd.compute_score(generated_samples)
        
        profile = profiler.last_profile
        
        # Should scale linearly with dimensions
        expected_max_time = 0.5 * n_dimensions
        assert profile['execution_time'] < max(expected_max_time, 2.0)
        
        print(f"Wasserstein (d={n_dimensions}): {profile['execution_time']:.3f}s")
    
    def test_mmd_distance_torch_performance(self, profiler):
        """Test MMDDistance (torch-based) performance."""
        n_samples = 1000
        target_samples = torch.randn(n_samples, 2)
        generated_samples = torch.randn(n_samples, 2)
        
        mmd = MMDDistance(target_samples, gamma=1.0)
        
        with profiler.profile():
            distance = mmd.compute_score(generated_samples)
        
        profile = profiler.last_profile
        
        assert profile['execution_time'] < 3.0  # Should be fast with torch
        assert isinstance(distance, float)
        assert distance >= 0
        
        print(f"MMDDistance (torch): {profile['execution_time']:.3f}s, "
              f"{profile['memory_delta_mb']:.1f}MB")
    
    def test_mmd_from_gmm_performance(self, profiler, performance_gaussian_params):
        """Test MMDivergenceFromGMM performance."""
        n_generated = 1000
        generated_samples = np.random.randn(n_generated, 2)
        
        # Test with different target sample sizes
        for n_target in [500, 1000, 2000]:
            mmd = MMDivergenceFromGMM(
                **performance_gaussian_params,
                n_target_samples=n_target,
                bandwidths=[0.5, 1.0, 2.0]
            )
            
            with profiler.profile():
                scores = mmd.compute_score(generated_samples)
            
            profile = profiler.last_profile
            
            assert profile['execution_time'] < 8.0
            assert len(scores) == n_generated
            
            print(f"MMDFromGMM (target={n_target}): {profile['execution_time']:.3f}s")
    
    def test_convergence_tracker_performance(self, profiler):
        """Test ConvergenceTracker performance with many updates."""
        tracker = ConvergenceTracker(patience=50, window_size=20)
        
        n_epochs = 1000
        
        with profiler.profile():
            for epoch in range(n_epochs):
                metrics = {
                    'KLDivergence': np.random.exponential(1.0),
                    'LogLikelihood': np.random.normal(-2.0, 0.5),
                    'MMDivergence': np.random.exponential(0.5),
                    'WassersteinDistance': np.random.exponential(0.3)
                }
                
                tracker.update(
                    epoch=epoch,
                    metrics=metrics,
                    d_loss=np.random.exponential(1.0),
                    g_loss=np.random.exponential(0.8)
                )
                
                if tracker.should_stop_early():
                    break
        
        profile = profiler.last_profile
        
        # Should be very fast for tracking operations
        assert profile['execution_time'] < 1.0
        assert profile['memory_delta_mb'] < 10
        
        print(f"ConvergenceTracker ({n_epochs} epochs): {profile['execution_time']:.3f}s")
    
    def test_memory_efficiency_large_samples(self, profiler):
        """Test memory efficiency with large sample sizes."""
        n_samples = 5000
        target_samples = np.random.randn(n_samples, 3)
        
        # Test each metric's memory usage
        metrics_to_test = [
            ('MMDivergence', lambda: MMDivergence(target_samples, bandwidths=[1.0])),
            ('WassersteinDistance', lambda: WassersteinDistance(target_samples)),
            ('MMDDistance', lambda: MMDDistance(torch.from_numpy(target_samples).float()))
        ]
        
        memory_results = {}
        
        for metric_name, metric_factory in metrics_to_test:
            # Generate fresh samples for each test
            generated_samples = np.random.randn(n_samples, 3)
            
            metric = metric_factory()
            
            with profiler.profile():
                if metric_name == 'MMDDistance':
                    result = metric.compute_score(torch.from_numpy(generated_samples).float())
                else:
                    result = metric.compute_score(generated_samples)
            
            memory_results[metric_name] = profiler.last_profile['memory_delta_mb']
            
            # Clean up
            del metric, generated_samples
            
            print(f"{metric_name} memory usage: {memory_results[metric_name]:.1f}MB")
        
        # All metrics should use reasonable memory
        for metric_name, memory_mb in memory_results.items():
            assert memory_mb < 500, f"{metric_name} used {memory_mb:.1f}MB"
    
    def test_comparative_performance_vs_existing_metrics(self, profiler, 
                                                        performance_gaussian_params):
        """Compare performance of new metrics vs existing ones."""
        n_samples = 1000
        target_samples = np.random.randn(n_samples, 2)
        generated_samples = np.random.randn(n_samples, 2)
        
        performance_results = {}
        
        # Test existing metrics
        existing_metrics = [
            ('LogLikelihood', lambda: LogLikelihood(**performance_gaussian_params)),
            ('KLDivergence', lambda: KLDivergence(**performance_gaussian_params))
        ]
        
        # Test new metrics
        new_metrics = [
            ('MMDivergence', lambda: MMDivergence(target_samples, bandwidths=[1.0])),
            ('WassersteinDistance', lambda: WassersteinDistance(target_samples)),
            ('MMDDistance', lambda: MMDDistance(torch.from_numpy(target_samples).float()))
        ]
        
        all_metrics = existing_metrics + new_metrics
        
        for metric_name, metric_factory in all_metrics:
            metric = metric_factory()
            
            with profiler.profile():
                if metric_name == 'MMDDistance':
                    result = metric.compute_score(torch.from_numpy(generated_samples).float())
                else:
                    result = metric.compute_score(generated_samples)
            
            performance_results[metric_name] = {
                'time': profiler.last_profile['execution_time'],
                'memory': profiler.last_profile['memory_delta_mb']
            }
            
            print(f"{metric_name}: {performance_results[metric_name]['time']:.3f}s, "
                  f"{performance_results[metric_name]['memory']:.1f}MB")
        
        # New metrics should have reasonable performance compared to existing ones
        existing_avg_time = np.mean([
            performance_results[name]['time'] 
            for name, _ in existing_metrics
        ])
        
        for metric_name, _ in new_metrics:
            new_time = performance_results[metric_name]['time']
            # New metrics should not be more than 10x slower than existing ones
            assert new_time < existing_avg_time * 10, \
                f"{metric_name} is too slow: {new_time:.3f}s vs avg {existing_avg_time:.3f}s"
    
    @pytest.mark.slow
    def test_sustained_performance_stress_test(self, profiler):
        """Test sustained performance under repeated metric computation."""
        n_iterations = 100
        n_samples = 500
        
        target_samples = np.random.randn(n_samples, 2)
        mmd = MMDivergence(target_samples, bandwidths=[0.5, 1.0])
        
        execution_times = []
        memory_deltas = []
        
        for i in range(n_iterations):
            generated_samples = np.random.randn(n_samples, 2)
            
            with profiler.profile():
                scores = mmd.compute_score(generated_samples)
            
            execution_times.append(profiler.last_profile['execution_time'])
            memory_deltas.append(profiler.last_profile['memory_delta_mb'])
        
        # Performance should be stable (no significant degradation)
        early_avg_time = np.mean(execution_times[:10])
        late_avg_time = np.mean(execution_times[-10:])
        
        # Late performance should not be more than 50% worse than early
        assert late_avg_time < early_avg_time * 1.5, \
            f"Performance degraded: {early_avg_time:.3f}s -> {late_avg_time:.3f}s"
        
        # Memory usage should not continuously increase
        avg_memory_delta = np.mean(memory_deltas)
        assert abs(avg_memory_delta) < 10, f"Average memory delta: {avg_memory_delta:.1f}MB"
        
        print(f"Stress test: {n_iterations} iterations, "
              f"avg time: {np.mean(execution_times):.3f}s ± {np.std(execution_times):.3f}s")
    
    def test_batch_vs_individual_computation_efficiency(self, profiler):
        """Test efficiency of batch computation vs individual computation."""
        n_samples = 1000
        target_samples = np.random.randn(n_samples, 2)
        generated_samples = np.random.randn(n_samples, 2)
        
        mmd = MMDivergence(target_samples, bandwidths=[1.0])
        
        # Test batch computation
        with profiler.profile():
            batch_scores = mmd.compute_score(generated_samples)
        batch_time = profiler.last_profile['execution_time']
        
        # Test individual computation (subset for speed)
        individual_times = []
        n_individual = 50  # Test subset
        
        for i in range(n_individual):
            single_sample = generated_samples[i:i+1]
            
            with profiler.profile():
                single_score = mmd.compute_score(single_sample)
            
            individual_times.append(profiler.last_profile['execution_time'])
        
        total_individual_time = sum(individual_times)
        estimated_full_individual_time = total_individual_time * (n_samples / n_individual)
        
        # Batch should be much more efficient
        efficiency_ratio = estimated_full_individual_time / batch_time
        assert efficiency_ratio > 5, f"Batch efficiency ratio: {efficiency_ratio:.1f}"
        
        print(f"Batch efficiency: {efficiency_ratio:.1f}x faster than individual")
    
    @pytest.mark.parametrize("aggregation", ["mean", "max", "sum"])
    def test_wasserstein_aggregation_performance(self, profiler, aggregation):
        """Test performance of different Wasserstein aggregation methods."""
        n_samples = 1000
        n_dims = 10
        
        target_samples = np.random.randn(n_samples, n_dims)
        generated_samples = np.random.randn(n_samples, n_dims)
        
        wd = WassersteinDistance(target_samples, aggregation=aggregation)
        
        with profiler.profile():
            distance = wd.compute_score(generated_samples)
        
        profile = profiler.last_profile
        
        # All aggregation methods should have similar performance
        assert profile['execution_time'] < 5.0
        assert isinstance(distance, float)
        assert distance >= 0
        
        print(f"Wasserstein ({aggregation}): {profile['execution_time']:.3f}s")


@pytest.mark.performance
class TestPerformanceRegression:
    """Regression tests to ensure performance doesn't degrade over time."""
    
    def test_mmd_performance_baseline(self, performance_monitor):
        """Baseline performance test for MMD metrics."""
        # Standard test configuration
        n_samples = 1000
        n_dims = 2
        n_bandwidths = 3
        
        target_samples = np.random.randn(n_samples, n_dims)
        generated_samples = np.random.randn(n_samples, n_dims)
        bandwidths = [0.5, 1.0, 2.0]
        
        mmd = MMDivergence(target_samples, bandwidths=bandwidths)
        
        performance_monitor.start_monitoring()
        scores = mmd.compute_score(generated_samples)
        metrics = performance_monitor.get_metrics()
        
        # Performance baselines (adjust based on typical performance)
        BASELINE_TIME = 2.0  # seconds
        BASELINE_MEMORY = 50  # MB
        
        assert metrics['elapsed_time'] < BASELINE_TIME, \
            f"MMD computation too slow: {metrics['elapsed_time']:.3f}s > {BASELINE_TIME}s"
        
        assert metrics['memory_increase_mb'] < BASELINE_MEMORY, \
            f"MMD memory usage too high: {metrics['memory_increase_mb']:.1f}MB > {BASELINE_MEMORY}MB"
        
        # Functionality check
        assert len(scores) == n_samples
        assert np.all(scores >= 0)
    
    def test_wasserstein_performance_baseline(self, performance_monitor):
        """Baseline performance test for Wasserstein distance."""
        n_samples = 1000
        n_dims = 5
        
        target_samples = np.random.randn(n_samples, n_dims)
        generated_samples = np.random.randn(n_samples, n_dims)
        
        wd = WassersteinDistance(target_samples, aggregation="mean")
        
        performance_monitor.start_monitoring()
        distance = wd.compute_score(generated_samples)
        metrics = performance_monitor.get_metrics()
        
        # Performance baselines
        BASELINE_TIME = 1.0  # seconds
        BASELINE_MEMORY = 20  # MB
        
        assert metrics['elapsed_time'] < BASELINE_TIME, \
            f"Wasserstein computation too slow: {metrics['elapsed_time']:.3f}s > {BASELINE_TIME}s"
        
        assert metrics['memory_increase_mb'] < BASELINE_MEMORY, \
            f"Wasserstein memory usage too high: {metrics['memory_increase_mb']:.1f}MB > {BASELINE_MEMORY}MB"
        
        # Functionality check
        assert isinstance(distance, float)
        assert distance >= 0
    
    def test_convergence_tracker_performance_baseline(self, performance_monitor):
        """Baseline performance test for ConvergenceTracker."""
        tracker = ConvergenceTracker(patience=20, window_size=10)
        
        n_epochs = 500
        
        performance_monitor.start_monitoring()
        
        for epoch in range(n_epochs):
            metrics = {
                'KLDivergence': np.random.exponential(1.0),
                'MMDivergence': np.random.exponential(0.5)
            }
            
            tracker.update(
                epoch=epoch,
                metrics=metrics,
                d_loss=np.random.exponential(1.0),
                g_loss=np.random.exponential(0.8)
            )
        
        metrics = performance_monitor.get_metrics()
        
        # Performance baselines
        BASELINE_TIME = 0.5  # seconds
        BASELINE_MEMORY = 5   # MB
        
        assert metrics['elapsed_time'] < BASELINE_TIME, \
            f"ConvergenceTracker too slow: {metrics['elapsed_time']:.3f}s > {BASELINE_TIME}s"
        
        assert metrics['memory_increase_mb'] < BASELINE_MEMORY, \
            f"ConvergenceTracker memory usage too high: {metrics['memory_increase_mb']:.1f}MB > {BASELINE_MEMORY}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])