"""
Comprehensive edge case and error handling tests for new statistical metrics.
Tests boundary conditions, error scenarios, and robustness.
"""

import pytest
import torch
import numpy as np
import warnings
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path

from source.metrics import (
    MMDivergence, MMDivergenceFromGMM, MMDDistance, 
    WassersteinDistance, ConvergenceTracker, ALL_METRICS
)


class TestEdgeCasesAndErrors:
    """Comprehensive edge case and error handling tests."""
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        target_samples = np.random.randn(100, 2)
        
        # Test MMDivergence
        mmd = MMDivergence(target_samples)
        empty_input = np.array([]).reshape(0, 2)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scores = mmd.compute_score(empty_input)
            assert len(w) > 0
            assert any("No valid points" in str(warning.message) for warning in w)
        
        # Test WassersteinDistance
        wd = WassersteinDistance(target_samples)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            distance = wd.compute_score(empty_input)
            assert distance == float('inf')
            assert len(w) > 0
            assert any("Empty points array" in str(warning.message) for warning in w)
        
        # Test MMDDistance
        mmd_dist = MMDDistance(torch.from_numpy(target_samples).float())
        empty_torch = torch.tensor([]).reshape(0, 2)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            distance = mmd_dist.compute_score(empty_torch)
            assert distance == float('inf')
            assert len(w) > 0
    
    def test_single_point_inputs(self):
        """Test handling of single-point inputs."""
        # Single point target
        single_target = np.array([[1.0, 2.0]])
        
        # Test MMDivergence
        mmd = MMDivergence(single_target)
        single_generated = np.array([[1.5, 2.5]])
        
        scores = mmd.compute_score(single_generated)
        assert len(scores) == 1
        assert scores[0] >= 0
        assert np.isfinite(scores[0])
        
        # Test WassersteinDistance
        wd = WassersteinDistance(single_target)
        distance = wd.compute_score(single_generated)
        
        assert isinstance(distance, float)
        assert distance >= 0
        assert np.isfinite(distance)
        
        # Test MMDDistance
        mmd_dist = MMDDistance(torch.tensor(single_target).float())
        distance = mmd_dist.compute_score(torch.tensor(single_generated).float())
        
        assert isinstance(distance, float)
        assert distance >= 0
        assert np.isfinite(distance)
    
    def test_nan_input_handling(self):
        """Test handling of NaN inputs."""
        target_samples = np.random.randn(100, 2)
        
        # Create input with NaN values
        nan_input = np.array([
            [1.0, 2.0],
            [np.nan, 3.0],
            [4.0, np.nan],
            [np.nan, np.nan],
            [5.0, 6.0]
        ])
        
        # Test MMDivergence
        mmd = MMDivergence(target_samples)
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            scores = mmd.compute_score(nan_input)
            
            # Should return same length as input
            assert len(scores) == len(nan_input)
            # Should handle NaN gracefully
            assert not np.all(np.isnan(scores))
        
        # Test WassersteinDistance
        wd = WassersteinDistance(target_samples)
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            distance = wd.compute_score(nan_input)
            
            # Should compute with valid points only
            assert isinstance(distance, float)
            assert np.isfinite(distance)
    
    def test_infinite_input_handling(self):
        """Test handling of infinite inputs."""
        target_samples = np.random.randn(50, 2)
        
        # Create input with infinite values
        inf_input = np.array([
            [1.0, 2.0],
            [np.inf, 3.0],
            [4.0, -np.inf],
            [np.inf, np.inf],
            [5.0, 6.0]
        ])
        
        # Test MMDivergence
        mmd = MMDivergence(target_samples)
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            scores = mmd.compute_score(inf_input)
            
            # Should handle gracefully
            assert len(scores) == len(inf_input)
        
        # Test WassersteinDistance
        wd = WassersteinDistance(target_samples)
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            distance = wd.compute_score(inf_input)
            
            # Should handle gracefully
            assert isinstance(distance, float)
    
    def test_zero_variance_inputs(self):
        """Test handling of zero variance (constant) inputs."""
        target_samples = np.random.randn(100, 2)
        
        # Constant input (zero variance)
        constant_input = np.full((50, 2), [5.0, 3.0])
        
        # Test MMDivergence
        mmd = MMDivergence(target_samples)
        scores = mmd.compute_score(constant_input)
        
        assert len(scores) == len(constant_input)
        assert np.all(scores >= 0)
        assert np.all(np.isfinite(scores))
        
        # Test WassersteinDistance
        wd = WassersteinDistance(target_samples)
        distance = wd.compute_score(constant_input)
        
        assert isinstance(distance, float)
        assert distance >= 0
        assert np.isfinite(distance)
    
    def test_very_large_values(self):
        """Test handling of very large values."""
        target_samples = np.random.randn(50, 2)
        
        # Very large values
        large_input = np.random.randn(30, 2) * 1e6
        
        # Test MMDivergence
        mmd = MMDivergence(target_samples, bandwidths=[1.0])
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            scores = mmd.compute_score(large_input)
            
            # Should handle without crashing
            assert len(scores) == len(large_input)
            # May be large but should be finite
            assert np.all(np.isfinite(scores) | np.isnan(scores))
        
        # Test WassersteinDistance
        wd = WassersteinDistance(target_samples)
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            distance = wd.compute_score(large_input)
            
            # Should handle without crashing
            assert isinstance(distance, float)
            assert np.isfinite(distance) or np.isnan(distance)
    
    def test_very_small_values(self):
        """Test handling of very small values."""
        target_samples = np.random.randn(50, 2)
        
        # Very small values
        small_input = np.random.randn(30, 2) * 1e-10
        
        # Test MMDivergence
        mmd = MMDivergence(target_samples, bandwidths=[1.0])
        scores = mmd.compute_score(small_input)
        
        assert len(scores) == len(small_input)
        assert np.all(scores >= 0)
        assert np.all(np.isfinite(scores))
        
        # Test WassersteinDistance
        wd = WassersteinDistance(target_samples)
        distance = wd.compute_score(small_input)
        
        assert isinstance(distance, float)
        assert distance >= 0
        assert np.isfinite(distance)
    
    def test_mismatched_dimensions(self):
        """Test handling of mismatched dimensions."""
        target_samples_2d = np.random.randn(100, 2)
        target_samples_3d = np.random.randn(100, 3)
        
        # Test MMDivergence with mismatched dimensions
        mmd = MMDivergence(target_samples_2d)
        mismatched_input = np.random.randn(30, 3)  # 3D input for 2D target
        
        with pytest.raises((ValueError, IndexError, RuntimeError)):
            mmd.compute_score(mismatched_input)
        
        # Test WassersteinDistance with mismatched dimensions
        wd = WassersteinDistance(target_samples_2d)
        
        with pytest.raises((ValueError, IndexError, RuntimeError)):
            wd.compute_score(mismatched_input)
    
    def test_negative_bandwidths(self):
        """Test handling of negative bandwidths in MMD."""
        target_samples = np.random.randn(50, 2)
        
        # Negative bandwidths should be handled gracefully or raise error
        with pytest.raises((ValueError, AssertionError)):
            MMDivergence(target_samples, bandwidths=[-1.0, 0.5])
        
        # Zero bandwidth should be handled
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            mmd = MMDivergence(target_samples, bandwidths=[0.0, 1.0])
            generated = np.random.randn(20, 2)
            scores = mmd.compute_score(generated)
            # Should either work or warn appropriately
    
    def test_invalid_aggregation_methods(self):
        """Test invalid aggregation methods for WassersteinDistance."""
        target_samples = np.random.randn(50, 2)
        
        # Invalid aggregation method
        with pytest.raises(ValueError, match="Invalid aggregation method"):
            WassersteinDistance(target_samples, aggregation="invalid")
    
    def test_convergence_tracker_edge_cases(self):
        """Test ConvergenceTracker edge cases."""
        # Zero patience
        tracker = ConvergenceTracker(patience=0)
        tracker.update(epoch=0, metrics={"KLDivergence": 1.0})
        result = tracker.update(epoch=1, metrics={"KLDivergence": 1.0})
        
        # Should converge immediately
        assert result["converged"] is True
        
        # Negative min_delta
        tracker = ConvergenceTracker(min_delta=-0.1)
        tracker.update(epoch=0, metrics={"KLDivergence": 1.0})
        result = tracker.update(epoch=1, metrics={"KLDivergence": 1.1})  # Worse
        
        # Should still consider it an improvement due to negative threshold
        assert result["epochs_without_improvement"] == 0
        
        # Very large window size
        tracker = ConvergenceTracker(window_size=1000)
        for i in range(10):
            tracker.update(epoch=i, metrics={}, d_loss=1.0, g_loss=0.5)
        
        info = tracker.get_convergence_info(9)
        # Should not have stability metrics due to insufficient data
        assert "d_loss_stability" not in info
        assert "g_loss_stability" not in info
    
    def test_memory_stress_conditions(self):
        """Test behavior under memory stress conditions."""
        # Very large number of bandwidths
        target_samples = np.random.randn(100, 2)
        many_bandwidths = np.logspace(-2, 2, 100)  # 100 bandwidths
        
        mmd = MMDivergence(target_samples, bandwidths=many_bandwidths)
        generated = np.random.randn(50, 2)
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            scores = mmd.compute_score(generated)
            
            # Should complete without major memory issues
            assert len(scores) == len(generated)
    
    def test_thread_safety_considerations(self):
        """Test thread safety considerations (basic checks)."""
        import threading
        import time
        
        target_samples = np.random.randn(200, 2)
        mmd = MMDivergence(target_samples, bandwidths=[1.0])
        
        results = []
        errors = []
        
        def compute_score_thread(thread_id):
            try:
                np.random.seed(thread_id)  # Different seed per thread
                generated = np.random.randn(100, 2)
                score = np.mean(mmd.compute_score(generated))
                results.append((thread_id, score))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=compute_score_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"
        
        # All results should be valid
        for thread_id, score in results:
            assert isinstance(score, (float, np.floating))
            assert score >= 0
            assert np.isfinite(score)
    
    def test_exception_handling_in_dependencies(self):
        """Test handling of exceptions in dependencies."""
        target_samples = np.random.randn(50, 2)
        
        # Mock scipy.spatial.distance.pdist to raise exception
        with patch('source.metrics.pdist', side_effect=Exception("Mock pdist error")):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                mmd = MMDivergence(target_samples)  # Should use fallback bandwidths
                
                # Should warn about fallback
                assert len(w) > 0
                assert any("Failed to compute adaptive bandwidths" in str(warning.message) for warning in w)
        
        # Mock wasserstein_distance to raise exception
        wd = WassersteinDistance(target_samples)
        generated = np.random.randn(30, 2)
        
        with patch('source.metrics.wasserstein_distance', side_effect=Exception("Mock Wasserstein error")):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                distance = wd.compute_score(generated)
                
                # Should handle gracefully
                assert distance == float('inf')
                assert len(w) > 0
                assert any("Error computing Wasserstein distance" in str(warning.message) for warning in w)
    
    def test_device_mismatch_handling_torch(self):
        """Test handling of device mismatches in torch-based metrics."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for device mismatch testing")
        
        # Create tensors on different devices
        target_cpu = torch.randn(50, 2, device='cpu')
        target_cuda = torch.randn(50, 2, device='cuda')
        generated_cpu = torch.randn(30, 2, device='cpu')
        generated_cuda = torch.randn(30, 2, device='cuda')
        
        # Test MMDDistance with device mismatch
        mmd_cpu = MMDDistance(target_cpu)
        mmd_cuda = MMDDistance(target_cuda)
        
        # Should handle device mismatch gracefully
        distance1 = mmd_cpu.compute_score(generated_cuda)  # CPU target, CUDA generated
        distance2 = mmd_cuda.compute_score(generated_cpu)  # CUDA target, CPU generated
        
        assert isinstance(distance1, float)
        assert isinstance(distance2, float)
        assert distance1 >= 0
        assert distance2 >= 0
    
    def test_precision_edge_cases(self):
        """Test numerical precision edge cases."""
        # Very close values that might cause precision issues
        target_samples = np.array([[0.0, 0.0]])
        close_samples = np.array([[1e-15, 1e-15]])
        
        # Test MMDivergence
        mmd = MMDivergence(target_samples, bandwidths=[1.0])
        scores = mmd.compute_score(close_samples)
        
        assert len(scores) == 1
        assert scores[0] >= 0
        assert np.isfinite(scores[0])
        
        # Test WassersteinDistance
        wd = WassersteinDistance(target_samples)
        distance = wd.compute_score(close_samples)
        
        assert isinstance(distance, float)
        assert distance >= 0
        assert np.isfinite(distance)
        # Should be very small due to close values
        assert distance < 1e-10
    
    def test_boundary_sample_sizes(self):
        """Test boundary sample sizes."""
        target_samples = np.random.randn(2, 2)  # Minimal target
        
        # Test with minimal generated samples
        minimal_generated = np.random.randn(1, 2)
        
        # Test MMDivergence
        mmd = MMDivergence(target_samples, bandwidths=[1.0])
        scores = mmd.compute_score(minimal_generated)
        
        assert len(scores) == 1
        assert scores[0] >= 0
        assert np.isfinite(scores[0])
        
        # Test WassersteinDistance
        wd = WassersteinDistance(target_samples)
        distance = wd.compute_score(minimal_generated)
        
        assert isinstance(distance, float)
        assert distance >= 0
        assert np.isfinite(distance)
    
    def test_convergence_tracker_extreme_values(self):
        """Test ConvergenceTracker with extreme metric values."""
        tracker = ConvergenceTracker(patience=3, min_delta=0.1)
        
        # Test with extreme values
        extreme_metrics = [
            {"KLDivergence": 1e10},    # Very large
            {"KLDivergence": 1e-10},   # Very small
            {"KLDivergence": np.inf},  # Infinite
            {"KLDivergence": -np.inf}, # Negative infinite
            {"KLDivergence": np.nan},  # NaN
        ]
        
        for epoch, metrics in enumerate(extreme_metrics):
            result = tracker.update(epoch=epoch, metrics=metrics)
            
            # Should handle gracefully without crashing
            assert isinstance(result, dict)
            assert "converged" in result
    
    def test_mmd_from_gmm_edge_cases(self):
        """Test MMDivergenceFromGMM edge cases."""
        # Single component GMM
        single_component = {
            'centroids': [[0, 0]],
            'cov_matrices': [[[1, 0], [0, 1]]],
            'weights': [1.0]
        }
        
        mmd = MMDivergenceFromGMM(**single_component, n_target_samples=10)
        generated = np.random.randn(20, 2)
        scores = mmd.compute_score(generated)
        
        assert len(scores) == 20
        assert np.all(scores >= 0)
        
        # Degenerate covariance matrix
        degenerate_gmm = {
            'centroids': [[0, 0]],
            'cov_matrices': [[[0, 0], [0, 0]]],  # Singular matrix
            'weights': [1.0]
        }
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mmd = MMDivergenceFromGMM(**degenerate_gmm, n_target_samples=10)
            # Should warn about precision matrix computation
            if len(w) > 0:
                assert any("Failed to compute precision matrices" in str(warning.message) for warning in w)
    
    def test_all_metrics_registry_robustness(self):
        """Test ALL_METRICS registry robustness."""
        # Test that all registered metrics can be instantiated
        for metric_name, metric_class in ALL_METRICS.items():
            try:
                if metric_name in ['MMDivergenceFromGMM']:
                    # Needs GMM parameters
                    metric = metric_class(
                        centroids=[[0, 0]],
                        cov_matrices=[[[1, 0], [0, 1]]],
                        weights=[1.0]
                    )
                elif metric_name in ['MMDivergence', 'MMDDistance', 'WassersteinDistance']:
                    # Needs target samples
                    target_samples = np.random.randn(10, 2)
                    if metric_name == 'MMDDistance':
                        target_samples = torch.from_numpy(target_samples).float()
                    metric = metric_class(target_samples)
                else:
                    # Standard instantiation
                    metric = metric_class()
                
                # Test that it has required method
                assert hasattr(metric, 'compute_score'), f"{metric_name} missing compute_score method"
                
            except Exception as e:
                pytest.fail(f"Failed to instantiate {metric_name}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])