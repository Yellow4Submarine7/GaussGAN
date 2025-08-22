"""
Unit tests for Wasserstein Distance metric implementation.
Tests mathematical correctness, edge cases, and aggregation methods.
"""

import pytest
import torch
import numpy as np
import warnings
from unittest.mock import Mock, patch

from source.metrics import WassersteinDistance
from scipy.stats import wasserstein_distance


class TestWassersteinDistance:
    """Test suite for WassersteinDistance metric."""
    
    def test_initialization_numpy_target_2d(self):
        """Test WassersteinDistance initialization with 2D numpy target samples."""
        target_samples = np.random.randn(100, 2)
        wd = WassersteinDistance(target_samples)
        
        assert wd.target_samples.shape == (100, 2)
        assert wd.n_dims == 2
        assert wd.aggregation == "mean"
        assert isinstance(wd.target_samples, np.ndarray)
    
    def test_initialization_torch_target_2d(self):
        """Test WassersteinDistance initialization with 2D torch target samples."""
        target_samples = torch.randn(50, 3)
        wd = WassersteinDistance(target_samples, aggregation="max")
        
        assert wd.target_samples.shape == (50, 3)
        assert wd.n_dims == 3
        assert wd.aggregation == "max"
        assert isinstance(wd.target_samples, np.ndarray)
    
    def test_initialization_1d_target(self):
        """Test WassersteinDistance initialization with 1D target samples."""
        target_samples = np.random.randn(100)
        wd = WassersteinDistance(target_samples)
        
        assert wd.target_samples.shape == (100,)
        assert wd.n_dims == 1
    
    def test_initialization_filters_nan_values_2d(self):
        """Test that NaN values are filtered from 2D target samples."""
        target_samples = np.array([
            [1.0, 2.0],
            [np.nan, 3.0],
            [4.0, np.nan],
            [5.0, 6.0],
            [np.nan, np.nan]
        ])
        
        wd = WassersteinDistance(target_samples)
        
        # Should keep only [1.0, 2.0] and [5.0, 6.0]
        assert wd.target_samples.shape == (2, 2)
    
    def test_initialization_filters_nan_values_1d(self):
        """Test that NaN values are filtered from 1D target samples."""
        target_samples = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        
        wd = WassersteinDistance(target_samples)
        
        # Should keep [1.0, 3.0, 5.0]
        assert wd.target_samples.shape == (3,)
        np.testing.assert_array_equal(wd.target_samples, [1.0, 3.0, 5.0])
    
    def test_initialization_all_nan_raises_error(self):
        """Test that all NaN target samples raises ValueError."""
        target_samples = np.array([
            [np.nan, np.nan],
            [np.nan, np.nan]
        ])
        
        with pytest.raises(ValueError, match="Target samples contain only NaN values"):
            WassersteinDistance(target_samples)
    
    def test_initialization_invalid_aggregation_raises_error(self):
        """Test that invalid aggregation method raises ValueError."""
        target_samples = np.random.randn(10, 2)
        
        with pytest.raises(ValueError, match="Invalid aggregation method: invalid"):
            WassersteinDistance(target_samples, aggregation="invalid")
    
    def test_valid_aggregation_methods(self):
        """Test all valid aggregation methods."""
        target_samples = np.random.randn(20, 2)
        
        for aggregation in ["mean", "max", "sum"]:
            wd = WassersteinDistance(target_samples, aggregation=aggregation)
            assert wd.aggregation == aggregation
    
    def test_compute_score_1d_case(self):
        """Test compute_score for 1D case."""
        # Create target distribution
        target_samples = np.random.normal(0, 1, 1000)
        wd = WassersteinDistance(target_samples)
        
        # Create generated samples
        generated_samples = np.random.normal(0.5, 1, 500)  # Slightly shifted
        
        distance = wd.compute_score(generated_samples)
        
        assert isinstance(distance, float)
        assert distance >= 0
        assert not np.isnan(distance)
        
        # Should be close to expected shift (0.5)
        assert 0.1 < distance < 1.0  # Reasonable range for this shift
    
    def test_compute_score_2d_case_mean_aggregation(self):
        """Test compute_score for 2D case with mean aggregation."""
        # Create target distribution
        target_samples = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 200)
        wd = WassersteinDistance(target_samples, aggregation="mean")
        
        # Create generated samples (shifted in x-dimension)
        generated_samples = np.random.multivariate_normal([1, 0], [[1, 0], [0, 1]], 100)
        
        distance = wd.compute_score(generated_samples)
        
        assert isinstance(distance, float)
        assert distance >= 0
        assert not np.isnan(distance)
    
    def test_compute_score_2d_case_max_aggregation(self):
        """Test compute_score for 2D case with max aggregation."""
        target_samples = np.random.randn(100, 2)
        wd = WassersteinDistance(target_samples, aggregation="max")
        
        # Create generated samples with different shifts per dimension
        generated_samples = np.random.randn(50, 2)
        generated_samples[:, 0] += 2  # Large shift in first dimension
        generated_samples[:, 1] += 0.1  # Small shift in second dimension
        
        distance = wd.compute_score(generated_samples)
        
        assert isinstance(distance, float)
        assert distance >= 0
        # Max aggregation should be dominated by the large shift
        assert distance > 1.0
    
    def test_compute_score_2d_case_sum_aggregation(self):
        """Test compute_score for 2D case with sum aggregation."""
        target_samples = np.random.randn(100, 2)
        wd = WassersteinDistance(target_samples, aggregation="sum")
        
        generated_samples = np.random.randn(50, 2) + 1  # Shift both dimensions
        
        distance = wd.compute_score(generated_samples)
        
        assert isinstance(distance, float)
        assert distance >= 0
        # Sum should be larger than individual dimension distances
    
    def test_compute_score_torch_input(self):
        """Test compute_score with torch tensor input."""
        target_samples = np.random.randn(50, 2)
        wd = WassersteinDistance(target_samples)
        
        generated_points = torch.randn(30, 2)
        distance = wd.compute_score(generated_points)
        
        assert isinstance(distance, float)
        assert distance >= 0
        assert not np.isnan(distance)
    
    def test_compute_score_empty_input(self):
        """Test compute_score with empty input."""
        target_samples = np.random.randn(50, 2)
        wd = WassersteinDistance(target_samples)
        
        empty_points = np.array([]).reshape(0, 2)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            distance = wd.compute_score(empty_points)
            
            assert distance == float('inf')
            assert len(w) > 0
            assert any("Empty points array" in str(warning.message) for warning in w)
    
    def test_compute_score_all_nan_input_2d(self):
        """Test compute_score with all NaN input (2D case)."""
        target_samples = np.random.randn(50, 2)
        wd = WassersteinDistance(target_samples)
        
        nan_points = np.array([
            [np.nan, np.nan],
            [np.nan, np.nan]
        ])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            distance = wd.compute_score(nan_points)
            
            assert distance == float('inf')
            assert len(w) > 0
            assert any("All generated points contain NaN" in str(warning.message) for warning in w)
    
    def test_compute_score_all_nan_input_1d(self):
        """Test compute_score with all NaN input (1D case)."""
        target_samples = np.random.randn(50)
        wd = WassersteinDistance(target_samples)
        
        nan_points = np.array([np.nan, np.nan, np.nan])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            distance = wd.compute_score(nan_points)
            
            assert distance == float('inf')
            assert len(w) > 0
            assert any("All generated points contain NaN" in str(warning.message) for warning in w)
    
    def test_compute_score_filters_nan_input_2d(self):
        """Test that compute_score filters NaN values from 2D input."""
        target_samples = np.random.randn(50, 2)
        wd = WassersteinDistance(target_samples)
        
        mixed_points = np.array([
            [1.0, 2.0],
            [np.nan, 3.0],
            [4.0, 5.0],
            [6.0, np.nan]
        ])
        
        distance = wd.compute_score(mixed_points)
        
        # Should compute with valid points only ([1.0, 2.0] and [4.0, 5.0])
        assert isinstance(distance, float)
        assert distance >= 0
        assert not np.isnan(distance)
    
    def test_compute_score_filters_nan_input_1d(self):
        """Test that compute_score filters NaN values from 1D input."""
        target_samples = np.random.randn(50)
        wd = WassersteinDistance(target_samples)
        
        mixed_points = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        
        distance = wd.compute_score(mixed_points)
        
        # Should compute with valid points only
        assert isinstance(distance, float)
        assert distance >= 0
        assert not np.isnan(distance)
    
    def test_compute_score_handles_scipy_wasserstein_exceptions(self):
        """Test that compute_score handles scipy.stats.wasserstein_distance exceptions."""
        target_samples = np.random.randn(10, 2)
        wd = WassersteinDistance(target_samples)
        
        # Mock wasserstein_distance to raise an exception
        with patch('source.metrics.wasserstein_distance', side_effect=Exception("Scipy error")):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                generated_points = np.random.randn(5, 2)
                distance = wd.compute_score(generated_points)
                
                assert distance == float('inf')
                assert len(w) > 0
                assert any("Error computing Wasserstein distance" in str(warning.message) for warning in w)
    
    def test_dimension_wise_distance_computation(self):
        """Test that multi-dimensional case computes distance per dimension."""
        # Create target with known properties per dimension
        target_samples = np.zeros((1000, 3))
        target_samples[:, 0] = np.random.normal(0, 1, 1000)  # Standard normal
        target_samples[:, 1] = np.random.normal(5, 2, 1000)  # Shifted and scaled
        target_samples[:, 2] = np.random.uniform(-1, 1, 1000)  # Uniform
        
        wd = WassersteinDistance(target_samples, aggregation="mean")
        
        # Create generated samples with different shifts per dimension
        generated_samples = np.zeros((500, 3))
        generated_samples[:, 0] = np.random.normal(1, 1, 500)    # Shift +1
        generated_samples[:, 1] = np.random.normal(5, 2, 500)    # No shift
        generated_samples[:, 2] = np.random.uniform(-1, 1, 500)  # No shift
        
        distance = wd.compute_score(generated_samples)
        
        # Should be positive due to shift in first dimension
        assert distance > 0
        assert distance < 2  # Should be reasonable given the shift
    
    def test_nan_distance_handling_in_aggregation(self):
        """Test handling of NaN distances during aggregation."""
        target_samples = np.random.randn(10, 2)
        wd = WassersteinDistance(target_samples)
        
        # Mock wasserstein_distance to return NaN for some calls
        mock_distances = [0.5, np.nan, 0.3]  # Second dimension returns NaN
        
        with patch('source.metrics.wasserstein_distance', side_effect=mock_distances):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                generated_points = np.random.randn(5, 2)
                distance = wd.compute_score(generated_points)
                
                # Should handle NaN by removing it and computing with remaining values
                if len(w) > 0:
                    assert any("NaN values detected" in str(warning.message) for warning in w)
                
                # Should compute with valid distances only
                assert isinstance(distance, float)
                # Could be finite or inf depending on how many valid distances remain
    
    def test_all_nan_distances_returns_inf(self):
        """Test that all NaN distances returns infinity."""
        target_samples = np.random.randn(10, 2)
        wd = WassersteinDistance(target_samples)
        
        # Mock wasserstein_distance to always return NaN
        with patch('source.metrics.wasserstein_distance', return_value=np.nan):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                generated_points = np.random.randn(5, 2)
                distance = wd.compute_score(generated_points)
                
                assert distance == float('inf')
                if len(w) > 0:
                    assert any("NaN values detected" in str(warning.message) for warning in w)
    
    def test_aggregation_methods_consistency(self):
        """Test that different aggregation methods give consistent relative results."""
        target_samples = np.random.randn(100, 3)
        generated_samples = np.random.randn(50, 3)
        
        # Create metrics with different aggregations
        wd_mean = WassersteinDistance(target_samples, aggregation="mean")
        wd_max = WassersteinDistance(target_samples, aggregation="max")
        wd_sum = WassersteinDistance(target_samples, aggregation="sum")
        
        distance_mean = wd_mean.compute_score(generated_samples)
        distance_max = wd_max.compute_score(generated_samples)
        distance_sum = wd_sum.compute_score(generated_samples)
        
        # Basic consistency checks
        assert distance_mean >= 0
        assert distance_max >= 0
        assert distance_sum >= 0
        
        # Max should be >= mean (max of values >= mean of values)
        assert distance_max >= distance_mean
        
        # Sum should be >= max for positive values
        assert distance_sum >= distance_max
    
    def test_known_distributions_sanity_check(self):
        """Test with known distributions to verify reasonable behavior."""
        # Create target: standard normal
        np.random.seed(42)
        target_samples = np.random.normal(0, 1, (1000, 2))
        wd = WassersteinDistance(target_samples)
        
        # Test 1: Same distribution should give small distance
        same_samples = np.random.normal(0, 1, (500, 2))
        distance_same = wd.compute_score(same_samples)
        
        # Test 2: Shifted distribution should give larger distance
        shifted_samples = np.random.normal(2, 1, (500, 2))  # Shift by 2
        distance_shifted = wd.compute_score(shifted_samples)
        
        # Test 3: Scaled distribution should give some distance
        scaled_samples = np.random.normal(0, 3, (500, 2))  # Scale by 3
        distance_scaled = wd.compute_score(scaled_samples)
        
        # Assertions
        assert distance_same < distance_shifted  # Shift should be more detectable
        assert distance_shifted > 1.0  # Should detect the shift of 2
        assert distance_scaled > 0.5  # Should detect the scaling
        assert all(d >= 0 for d in [distance_same, distance_shifted, distance_scaled])
    
    def test_edge_case_single_point_distributions(self):
        """Test edge case with single-point distributions."""
        # Single point target
        target_samples = np.array([[1.0, 2.0]])
        wd = WassersteinDistance(target_samples)
        
        # Single point generated (same)
        same_point = np.array([[1.0, 2.0]])
        distance_same = wd.compute_score(same_point)
        
        # Single point generated (different)
        different_point = np.array([[3.0, 4.0]])
        distance_different = wd.compute_score(different_point)
        
        # Same points should give zero distance
        assert distance_same == 0.0
        
        # Different points should give positive distance
        assert distance_different > 0
        
        # Distance should be related to actual difference
        expected_distance = np.mean([abs(3.0 - 1.0), abs(4.0 - 2.0)])  # Mean of abs differences
        assert abs(distance_different - expected_distance) < 0.1
    
    def test_high_dimensional_case(self):
        """Test behavior with high-dimensional data."""
        dimensions = [5, 10, 20]
        
        for dim in dimensions:
            target_samples = np.random.randn(100, dim)
            wd = WassersteinDistance(target_samples)
            
            generated_samples = np.random.randn(50, dim) + 0.5  # Small shift
            distance = wd.compute_score(generated_samples)
            
            assert isinstance(distance, float)
            assert distance >= 0
            assert not np.isnan(distance)
            assert distance < 10  # Should be reasonable for small shift
    
    def test_different_sample_sizes(self):
        """Test with different sample sizes."""
        target_samples = np.random.randn(100, 2)
        wd = WassersteinDistance(target_samples)
        
        sample_sizes = [1, 5, 10, 50, 200]
        
        for size in sample_sizes:
            generated_samples = np.random.randn(size, 2)
            distance = wd.compute_score(generated_samples)
            
            assert isinstance(distance, float)
            assert distance >= 0
            assert not np.isnan(distance)
    
    def test_reproducibility_with_same_input(self):
        """Test that same inputs give same results."""
        target_samples = np.random.randn(50, 2)
        wd = WassersteinDistance(target_samples)
        
        generated_samples = np.random.randn(30, 2)
        
        distance1 = wd.compute_score(generated_samples)
        distance2 = wd.compute_score(generated_samples)
        
        assert distance1 == distance2
    
    @pytest.mark.slow
    def test_computational_efficiency_large_samples(self):
        """Test computational efficiency with larger sample sizes."""
        import time
        
        target_samples = np.random.randn(1000, 2)
        wd = WassersteinDistance(target_samples)
        
        generated_samples = np.random.randn(1000, 2)
        
        start_time = time.time()
        distance = wd.compute_score(generated_samples)
        computation_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert computation_time < 3.0  # seconds
        assert isinstance(distance, float)
        assert distance >= 0
    
    def test_wasserstein_distance_mathematical_properties(self):
        """Test mathematical properties of Wasserstein distance."""
        # Property 1: Non-negativity
        target_samples = np.random.randn(50, 2)
        wd = WassersteinDistance(target_samples)
        
        generated_samples = np.random.randn(30, 2)
        distance = wd.compute_score(generated_samples)
        
        assert distance >= 0
        
        # Property 2: Identity (distance to itself should be 0)
        distance_self = wd.compute_score(target_samples)
        assert distance_self < 0.1  # Should be very small (numerical precision)
        
        # Property 3: Symmetry (in expectation, not exact due to sampling)
        wd2 = WassersteinDistance(generated_samples)
        distance_reverse = wd2.compute_score(target_samples)
        
        # Should be reasonably close (within sampling error)
        assert abs(distance - distance_reverse) / max(distance, distance_reverse) < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])