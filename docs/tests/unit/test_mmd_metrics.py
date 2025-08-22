"""
Unit tests for Maximum Mean Discrepancy (MMD) metric implementations.
Tests mathematical correctness, edge cases, and performance characteristics.
"""

import pytest
import torch
import numpy as np
import warnings
from unittest.mock import Mock, patch, MagicMock
from scipy.spatial.distance import pdist

from source.metrics import MMDivergence, MMDivergenceFromGMM, MMDDistance
from sklearn.mixture import GaussianMixture


class TestMMDivergence:
    """Test suite for MMDivergence metric."""
    
    def test_initialization_with_numpy_target(self):
        """Test MMDivergence initialization with numpy target samples."""
        target_samples = np.random.randn(100, 2)
        mmd = MMDivergence(target_samples)
        
        assert mmd.target_samples.shape == (100, 2)
        assert isinstance(mmd.target_samples, np.ndarray)
        assert mmd.n_target == 100
        assert isinstance(mmd.bandwidths, np.ndarray)
        assert len(mmd.bandwidths) > 0
    
    def test_initialization_with_torch_target(self):
        """Test MMDivergence initialization with torch tensor target samples."""
        target_samples = torch.randn(50, 3)
        mmd = MMDivergence(target_samples)
        
        assert mmd.target_samples.shape == (50, 3)
        assert isinstance(mmd.target_samples, np.ndarray)
        assert mmd.n_target == 50
    
    def test_initialization_with_custom_bandwidths(self):
        """Test MMDivergence initialization with custom bandwidths."""
        target_samples = np.random.randn(50, 2)
        custom_bandwidths = [0.5, 1.0, 2.0]
        
        mmd = MMDivergence(target_samples, bandwidths=custom_bandwidths)
        
        assert np.array_equal(mmd.bandwidths, np.array(custom_bandwidths))
    
    def test_initialization_filters_nan_values(self):
        """Test that NaN values are filtered from target samples during initialization."""
        target_samples = np.array([
            [1.0, 2.0],
            [np.nan, 3.0],
            [4.0, np.nan],
            [5.0, 6.0],
            [np.nan, np.nan]
        ])
        
        mmd = MMDivergence(target_samples)
        
        # Should only keep [1.0, 2.0] and [5.0, 6.0]
        assert mmd.target_samples.shape == (2, 2)
        assert mmd.n_target == 2
    
    def test_initialization_all_nan_raises_error(self):
        """Test that initialization with all NaN values raises ValueError."""
        target_samples = np.array([
            [np.nan, np.nan],
            [np.nan, np.nan]
        ])
        
        with pytest.raises(ValueError, match="No valid target samples"):
            MMDivergence(target_samples)
    
    def test_adaptive_bandwidth_computation(self):
        """Test adaptive bandwidth computation using median heuristic."""
        # Create target samples with known distribution
        np.random.seed(42)
        target_samples = np.random.randn(100, 2)
        
        mmd = MMDivergence(target_samples)
        
        # Should have multiple bandwidths
        assert len(mmd.bandwidths) >= 3
        # Should be positive
        assert np.all(mmd.bandwidths > 0)
        # Should have reasonable scale
        assert np.all(mmd.bandwidths < 100)
    
    def test_adaptive_bandwidth_single_sample(self):
        """Test adaptive bandwidth computation with single sample."""
        target_samples = np.array([[1.0, 2.0]])
        
        mmd = MMDivergence(target_samples)
        
        # Should handle single sample gracefully
        assert len(mmd.bandwidths) >= 1
        assert np.all(mmd.bandwidths > 0)
    
    def test_adaptive_bandwidth_fallback_on_error(self):
        """Test fallback to default bandwidths when computation fails."""
        target_samples = np.random.randn(10, 2)
        
        # Mock pdist to raise an exception
        with patch('source.metrics.pdist', side_effect=Exception("Test error")):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                mmd = MMDivergence(target_samples)
                
                # Should warn about fallback
                assert len(w) > 0
                assert "Failed to compute adaptive bandwidths" in str(w[0].message)
                
                # Should have default bandwidths
                expected_default = np.array([0.1, 1.0, 10.0])
                assert np.array_equal(mmd.bandwidths, expected_default)
    
    def test_rbf_kernel_computation(self):
        """Test RBF kernel matrix computation."""
        target_samples = np.array([[0.0, 0.0], [1.0, 1.0]])
        mmd = MMDivergence(target_samples)
        
        X = np.array([[0.0, 0.0], [2.0, 2.0]])
        Y = np.array([[1.0, 1.0], [3.0, 3.0]])
        bandwidth = 1.0
        
        kernel_matrix = mmd._rbf_kernel(X, Y, bandwidth)
        
        # Check shape
        assert kernel_matrix.shape == (2, 2)
        
        # Check values are in [0, 1] (RBF kernel properties)
        assert np.all(kernel_matrix >= 0)
        assert np.all(kernel_matrix <= 1)
        
        # Diagonal should be exp(-0) = 1 when X == Y
        X_same = np.array([[0.0, 0.0]])
        Y_same = np.array([[0.0, 0.0]])
        kernel_same = mmd._rbf_kernel(X_same, Y_same, bandwidth)
        assert np.allclose(kernel_same, [[1.0]])
    
    def test_rbf_kernel_handles_numerical_issues(self):
        """Test RBF kernel handles numerical precision issues."""
        target_samples = np.random.randn(10, 2)
        mmd = MMDivergence(target_samples)
        
        # Create X and Y that might have negative squared distances due to floating point
        X = np.array([[1e-10, 1e-10]])
        Y = np.array([[0.0, 0.0]])
        
        kernel_matrix = mmd._rbf_kernel(X, Y, 1.0)
        
        # Should not contain NaN or negative values
        assert not np.any(np.isnan(kernel_matrix))
        assert np.all(kernel_matrix >= 0)
    
    def test_mmd_squared_computation(self):
        """Test MMD squared computation."""
        # Create simple target samples
        target_samples = np.array([[0.0, 0.0], [1.0, 1.0]])
        mmd = MMDivergence(target_samples, bandwidths=[1.0])
        
        # Generate some test samples
        generated_samples = np.array([[0.5, 0.5], [1.5, 1.5]])
        
        mmd_squared = mmd._compute_mmd_squared(generated_samples)
        
        # Should be non-negative
        assert mmd_squared >= 0
        assert isinstance(mmd_squared, (float, np.floating))
    
    def test_mmd_squared_empty_samples(self):
        """Test MMD squared computation with empty generated samples."""
        target_samples = np.random.randn(10, 2)
        mmd = MMDivergence(target_samples)
        
        empty_samples = np.array([]).reshape(0, 2)
        mmd_squared = mmd._compute_mmd_squared(empty_samples)
        
        assert mmd_squared == float('inf')
    
    def test_mmd_squared_single_target_sample(self):
        """Test MMD squared computation with single target sample."""
        target_samples = np.array([[1.0, 1.0]])
        mmd = MMDivergence(target_samples, bandwidths=[1.0])
        
        generated_samples = np.array([[2.0, 2.0]])
        mmd_squared = mmd._compute_mmd_squared(generated_samples)
        
        # Should handle single samples gracefully
        assert mmd_squared >= 0
        assert not np.isnan(mmd_squared)
    
    def test_mmd_squared_single_generated_sample(self):
        """Test MMD squared computation with single generated sample."""
        target_samples = np.array([[0.0, 0.0], [1.0, 1.0]])
        mmd = MMDivergence(target_samples, bandwidths=[1.0])
        
        generated_samples = np.array([[0.5, 0.5]])
        mmd_squared = mmd._compute_mmd_squared(generated_samples)
        
        # Should handle single generated sample
        assert mmd_squared >= 0
        assert not np.isnan(mmd_squared)
    
    def test_compute_score_with_torch_input(self):
        """Test compute_score with torch tensor input."""
        target_samples = np.random.randn(50, 2)
        mmd = MMDivergence(target_samples, bandwidths=[0.5, 1.0])
        
        generated_points = torch.randn(30, 2)
        scores = mmd.compute_score(generated_points)
        
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 30
        assert np.all(scores >= 0)  # MMD is non-negative
        assert np.all(np.isfinite(scores) | np.isnan(scores))
    
    def test_compute_score_with_numpy_input(self):
        """Test compute_score with numpy array input."""
        target_samples = np.random.randn(50, 2)
        mmd = MMDivergence(target_samples)
        
        generated_points = np.random.randn(30, 2)
        scores = mmd.compute_score(generated_points)
        
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 30
        assert np.all(scores >= 0)
    
    def test_compute_score_filters_nan_input(self):
        """Test that compute_score filters NaN values from input."""
        target_samples = np.random.randn(50, 2)
        mmd = MMDivergence(target_samples)
        
        generated_points = np.array([
            [1.0, 2.0],
            [np.nan, 3.0],
            [4.0, 5.0],
            [6.0, np.nan]
        ])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scores = mmd.compute_score(generated_points)
            
            # Should return scores for all input points
            assert len(scores) == 4
            # Should warn about NaN values
            if len(w) > 0:
                assert any("No valid points" in str(warning.message) for warning in w)
    
    def test_compute_score_all_nan_input(self):
        """Test compute_score with all NaN input points."""
        target_samples = np.random.randn(50, 2)
        mmd = MMDivergence(target_samples)
        
        generated_points = np.array([
            [np.nan, np.nan],
            [np.nan, np.nan]
        ])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scores = mmd.compute_score(generated_points)
            
            # Should return NaN for all points
            assert len(scores) == 2
            assert np.all(np.isnan(scores))
            # Should warn
            assert len(w) > 0
            assert any("No valid points" in str(warning.message) for warning in w)
    
    def test_compute_score_exception_handling(self):
        """Test compute_score handles exceptions gracefully."""
        target_samples = np.random.randn(10, 2)
        mmd = MMDivergence(target_samples)
        
        # Mock _compute_mmd_squared to raise an exception
        with patch.object(mmd, '_compute_mmd_squared', side_effect=Exception("Test error")):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                generated_points = np.random.randn(5, 2)
                scores = mmd.compute_score(generated_points)
                
                # Should return NaN values
                assert len(scores) == 5
                assert np.all(np.isnan(scores))
                # Should warn
                assert len(w) > 0
                assert any("Error during MMD computation" in str(warning.message) for warning in w)
    
    def test_mmd_properties_identical_distributions(self):
        """Test MMD properties when distributions are identical."""
        # Use same samples for target and generated
        samples = np.random.randn(100, 2)
        mmd = MMDivergence(samples, bandwidths=[1.0])
        
        scores = mmd.compute_score(samples)
        
        # MMD should be close to 0 for identical distributions
        mean_score = np.mean(scores)
        assert mean_score < 0.5  # Should be small
    
    def test_mmd_properties_different_distributions(self):
        """Test MMD properties when distributions are different."""
        target_samples = np.random.normal(0, 1, (100, 2))
        generated_samples = np.random.normal(2, 1, (100, 2))  # Different mean
        
        mmd = MMDivergence(target_samples, bandwidths=[1.0])
        scores = mmd.compute_score(generated_samples)
        
        # MMD should be larger for different distributions
        mean_score = np.mean(scores)
        assert mean_score > 0.1  # Should be non-trivial
    
    def test_mmd_bandwidth_sensitivity(self):
        """Test MMD sensitivity to different bandwidths."""
        target_samples = np.random.randn(50, 2)
        generated_samples = np.random.randn(50, 2) + 1  # Shifted
        
        # Test different bandwidths
        small_bandwidth = MMDivergence(target_samples, bandwidths=[0.1])
        large_bandwidth = MMDivergence(target_samples, bandwidths=[10.0])
        
        scores_small = small_bandwidth.compute_score(generated_samples)
        scores_large = large_bandwidth.compute_score(generated_samples)
        
        # Both should detect the difference, but with different sensitivities
        assert np.mean(scores_small) > 0
        assert np.mean(scores_large) > 0
        
        # Small bandwidth should be more sensitive to local differences
        # Large bandwidth should be more sensitive to global differences
        # (Exact relationship depends on the specific shift)
    
    def test_mmd_multiple_bandwidths(self):
        """Test MMD computation with multiple bandwidths."""
        target_samples = np.random.randn(50, 2)
        generated_samples = np.random.randn(30, 2)
        
        mmd = MMDivergence(target_samples, bandwidths=[0.5, 1.0, 2.0])
        scores = mmd.compute_score(generated_samples)
        
        # Should average over multiple bandwidths
        assert len(scores) == 30
        assert np.all(scores >= 0)
        assert np.all(np.isfinite(scores))
    
    def test_mmd_dimensionality_handling(self):
        """Test MMD with different dimensionalities."""
        dimensions = [1, 2, 5, 10]
        
        for dim in dimensions:
            target_samples = np.random.randn(20, dim)
            generated_samples = np.random.randn(15, dim)
            
            mmd = MMDivergence(target_samples, bandwidths=[1.0])
            scores = mmd.compute_score(generated_samples)
            
            assert len(scores) == 15
            assert np.all(scores >= 0)
            assert not np.any(np.isnan(scores))
    
    @pytest.mark.slow
    def test_mmd_computational_efficiency(self):
        """Test MMD computational efficiency with larger samples."""
        import time
        
        target_samples = np.random.randn(500, 2)
        generated_samples = np.random.randn(500, 2)
        
        mmd = MMDivergence(target_samples, bandwidths=[1.0])
        
        start_time = time.time()
        scores = mmd.compute_score(generated_samples)
        computation_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert computation_time < 5.0  # seconds
        assert len(scores) == 500
        assert np.all(np.isfinite(scores))


class TestMMDivergenceFromGMM:
    """Test suite for MMDivergenceFromGMM metric."""
    
    def test_initialization_valid_gmm_parameters(self):
        """Test initialization with valid GMM parameters."""
        centroids = [[0, 0], [3, 3]]
        cov_matrices = [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
        weights = [0.5, 0.5]
        
        mmd = MMDivergenceFromGMM(centroids, cov_matrices, weights)
        
        # Check GMM setup
        assert mmd.gmm.n_components == 2
        assert np.array_equal(mmd.gmm.means_, np.array(centroids))
        assert np.array_equal(mmd.gmm.covariances_, np.array(cov_matrices))
        assert np.array_equal(mmd.gmm.weights_, np.array(weights))
        
        # Check internal MMD metric
        assert hasattr(mmd, 'mmd_metric')
        assert isinstance(mmd.mmd_metric, MMDivergence)
    
    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        centroids = [[1, 2], [4, 5]]
        cov_matrices = [[[2, 0], [0, 2]], [[1, 0.5], [0.5, 1]]]
        weights = [0.3, 0.7]
        n_target_samples = 500
        bandwidths = [0.5, 2.0]
        
        mmd = MMDivergenceFromGMM(
            centroids, cov_matrices, weights, 
            n_target_samples=n_target_samples,
            bandwidths=bandwidths
        )
        
        assert mmd.gmm.n_components == 2
        # Internal MMD metric should use the specified parameters
        assert len(mmd.mmd_metric.bandwidths) == 2
    
    def test_initialization_handles_singular_covariance(self):
        """Test initialization handles singular covariance matrices."""
        centroids = [[0, 0]]
        # Singular covariance matrix
        cov_matrices = [[[0, 0], [0, 0]]]
        weights = [1.0]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mmd = MMDivergenceFromGMM(centroids, cov_matrices, weights)
            
            # Should warn about precision matrix computation failure
            if len(w) > 0:
                assert any("Failed to compute precision matrices" in str(warning.message) for warning in w)
            
            # Should still create the object
            assert hasattr(mmd, 'mmd_metric')
    
    def test_gmm_sample_generation_failure_fallback(self):
        """Test fallback when GMM sample generation fails."""
        centroids = [[0, 0]]
        cov_matrices = [[[1, 0], [0, 1]]]
        weights = [1.0]
        
        # Mock GMM sample method to fail
        with patch.object(GaussianMixture, 'sample', side_effect=Exception("Sample generation failed")):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                mmd = MMDivergenceFromGMM(centroids, cov_matrices, weights)
                
                # Should warn about sample generation failure
                assert len(w) > 0
                assert any("Failed to generate target samples" in str(warning.message) for warning in w)
                
                # Should fallback to using centroids as target samples
                assert hasattr(mmd, 'mmd_metric')
    
    def test_compute_score_delegation(self):
        """Test that compute_score delegates to internal MMD metric."""
        centroids = [[0, 0], [2, 2]]
        cov_matrices = [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
        weights = [0.5, 0.5]
        
        mmd = MMDivergenceFromGMM(centroids, cov_matrices, weights, n_target_samples=100)
        
        # Mock the internal MMD metric
        mmd.mmd_metric = Mock()
        mmd.mmd_metric.compute_score.return_value = np.array([0.5, 0.3, 0.7])
        
        generated_points = np.random.randn(3, 2)
        scores = mmd.compute_score(generated_points)
        
        # Should delegate to internal metric
        mmd.mmd_metric.compute_score.assert_called_once()
        assert np.array_equal(scores, np.array([0.5, 0.3, 0.7]))
    
    def test_compatibility_with_existing_metrics(self):
        """Test compatibility with existing LogLikelihood and KLDivergence metrics."""
        # Use same parameters as other metrics
        centroids = [[0, 0], [3, 3]]
        cov_matrices = [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
        weights = [0.5, 0.5]
        
        mmd = MMDivergenceFromGMM(centroids, cov_matrices, weights)
        
        # Should accept same format of parameters
        assert mmd.gmm.n_components == 2
        assert len(mmd.gmm.means_) == 2
        assert len(mmd.gmm.covariances_) == 2
        assert len(mmd.gmm.weights_) == 2
    
    def test_different_gmm_configurations(self):
        """Test with different GMM configurations."""
        configurations = [
            {
                'centroids': [[0, 0]],
                'cov_matrices': [[[1, 0], [0, 1]]],
                'weights': [1.0]
            },
            {
                'centroids': [[0, 0], [2, 2], [4, 4]],
                'cov_matrices': [[[1, 0], [0, 1]], [[0.5, 0], [0, 0.5]], [[2, 0], [0, 2]]],
                'weights': [0.33, 0.33, 0.34]
            },
            {
                'centroids': [[-1, -1], [1, 1]],
                'cov_matrices': [[[0.5, 0.2], [0.2, 0.5]], [[1, -0.3], [-0.3, 1]]],
                'weights': [0.7, 0.3]
            }
        ]
        
        for config in configurations:
            mmd = MMDivergenceFromGMM(**config, n_target_samples=50)
            
            # Should work with various configurations
            assert hasattr(mmd, 'mmd_metric')
            
            # Test score computation
            generated_points = np.random.randn(10, 2)
            scores = mmd.compute_score(generated_points)
            
            assert len(scores) == 10
            assert np.all(scores >= 0)


class TestMMDDistance:
    """Test suite for MMDDistance metric."""
    
    def test_initialization_numpy_target(self):
        """Test MMDDistance initialization with numpy target samples."""
        target_samples = np.random.randn(100, 2)
        mmd = MMDDistance(target_samples)
        
        assert mmd.target_samples.shape == (100, 2)
        assert isinstance(mmd.target_samples, torch.Tensor)
        assert mmd.gamma == 1.0
        assert mmd.kernel == "rbf"
    
    def test_initialization_torch_target(self):
        """Test MMDDistance initialization with torch target samples."""
        target_samples = torch.randn(50, 3)
        mmd = MMDDistance(target_samples, gamma=2.0)
        
        assert mmd.target_samples.shape == (50, 3)
        assert mmd.gamma == 2.0
    
    def test_initialization_filters_nan_values(self):
        """Test that NaN values are filtered during initialization."""
        target_samples = torch.tensor([
            [1.0, 2.0],
            [float('nan'), 3.0],
            [4.0, 5.0],
            [float('nan'), float('nan')]
        ])
        
        mmd = MMDDistance(target_samples)
        
        # Should keep only valid samples
        assert mmd.target_samples.shape == (2, 2)
    
    def test_initialization_all_nan_raises_error(self):
        """Test that all NaN target samples raises ValueError."""
        target_samples = torch.tensor([
            [float('nan'), float('nan')],
            [float('nan'), float('nan')]
        ])
        
        with pytest.raises(ValueError, match="Target samples contain only NaN values"):
            MMDDistance(target_samples)
    
    def test_invalid_kernel_raises_error(self):
        """Test that invalid kernel type raises NotImplementedError."""
        target_samples = torch.randn(10, 2)
        
        with pytest.raises(NotImplementedError, match="Kernel polynomial not implemented"):
            MMDDistance(target_samples, kernel="polynomial")
    
    def test_rbf_kernel_computation(self):
        """Test RBF kernel computation."""
        target_samples = torch.randn(10, 2)
        mmd = MMDDistance(target_samples, gamma=1.0)
        
        X = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        Y = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
        
        kernel_matrix = mmd.rbf_kernel(X, Y)
        
        # Check properties
        assert kernel_matrix.shape == (2, 2)
        assert torch.all(kernel_matrix >= 0)
        assert torch.all(kernel_matrix <= 1)
        
        # Self-similarity should be 1
        assert torch.allclose(kernel_matrix[0, 0], torch.tensor(1.0), atol=1e-6)
    
    def test_rbf_kernel_gamma_parameter(self):
        """Test RBF kernel with different gamma parameters."""
        target_samples = torch.randn(10, 2)
        
        X = torch.tensor([[0.0, 0.0]])
        Y = torch.tensor([[1.0, 1.0]])  # Distance sqrt(2)
        
        # Small gamma (broader kernel)
        mmd_small = MMDDistance(target_samples, gamma=0.1)
        kernel_small = mmd_small.rbf_kernel(X, Y)
        
        # Large gamma (narrow kernel)
        mmd_large = MMDDistance(target_samples, gamma=10.0)
        kernel_large = mmd_large.rbf_kernel(X, Y)
        
        # Small gamma should give larger kernel values for same distance
        assert kernel_small > kernel_large
    
    def test_compute_score_torch_input(self):
        """Test compute_score with torch tensor input."""
        target_samples = torch.randn(50, 2)
        mmd = MMDDistance(target_samples)
        
        generated_points = torch.randn(30, 2)
        score = mmd.compute_score(generated_points)
        
        assert isinstance(score, float)
        assert score >= 0
        assert not np.isnan(score)
    
    def test_compute_score_numpy_input(self):
        """Test compute_score with numpy array input."""
        target_samples = torch.randn(50, 2)
        mmd = MMDDistance(target_samples)
        
        generated_points = np.random.randn(30, 2)
        score = mmd.compute_score(generated_points)
        
        assert isinstance(score, float)
        assert score >= 0
    
    def test_compute_score_empty_input(self):
        """Test compute_score with empty input."""
        target_samples = torch.randn(50, 2)
        mmd = MMDDistance(target_samples)
        
        empty_points = np.array([]).reshape(0, 2)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            score = mmd.compute_score(empty_points)
            
            assert score == float('inf')
            assert len(w) > 0
            assert any("Empty points array" in str(warning.message) for warning in w)
    
    def test_compute_score_all_nan_input(self):
        """Test compute_score with all NaN input."""
        target_samples = torch.randn(50, 2)
        mmd = MMDDistance(target_samples)
        
        nan_points = np.array([
            [np.nan, np.nan],
            [np.nan, np.nan]
        ])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            score = mmd.compute_score(nan_points)
            
            assert score == float('inf')
            assert len(w) > 0
            assert any("All generated points contain NaN" in str(warning.message) for warning in w)
    
    def test_compute_score_filters_nan_input(self):
        """Test that compute_score filters NaN values from input."""
        target_samples = torch.randn(50, 2)
        mmd = MMDDistance(target_samples)
        
        mixed_points = np.array([
            [1.0, 2.0],
            [np.nan, 3.0],
            [4.0, 5.0]
        ])
        
        score = mmd.compute_score(mixed_points)
        
        # Should compute with valid points only
        assert isinstance(score, float)
        assert score >= 0
        assert not np.isnan(score)
    
    def test_compute_score_device_handling(self):
        """Test compute_score handles device mismatch."""
        # Create target samples on specific device
        device = torch.device('cpu')
        target_samples = torch.randn(20, 2, device=device)
        mmd = MMDDistance(target_samples)
        
        # Create generated points (should handle device automatically)
        generated_points = torch.randn(10, 2, device=device)
        score = mmd.compute_score(generated_points)
        
        assert isinstance(score, float)
        assert score >= 0
    
    def test_compute_score_exception_handling(self):
        """Test compute_score handles exceptions gracefully."""
        target_samples = torch.randn(10, 2)
        mmd = MMDDistance(target_samples)
        
        # Mock rbf_kernel to raise an exception
        with patch.object(mmd, 'rbf_kernel', side_effect=Exception("Test error")):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                generated_points = torch.randn(5, 2)
                score = mmd.compute_score(generated_points)
                
                assert score == float('inf')
                assert len(w) > 0
                assert any("Error computing MMD distance" in str(warning.message) for warning in w)
    
    def test_mmd_distance_properties_identical_distributions(self):
        """Test MMD distance when distributions are identical."""
        samples = torch.randn(100, 2)
        mmd = MMDDistance(samples)
        
        score = mmd.compute_score(samples)
        
        # Should be close to 0 for identical distributions
        assert score < 0.1
    
    def test_mmd_distance_properties_different_distributions(self):
        """Test MMD distance when distributions are different."""
        target_samples = torch.randn(100, 2)
        generated_samples = torch.randn(100, 2) + 2  # Shifted distribution
        
        mmd = MMDDistance(target_samples)
        score = mmd.compute_score(generated_samples)
        
        # Should detect the difference
        assert score > 0.1
    
    def test_mmd_distance_gamma_sensitivity(self):
        """Test MMD distance sensitivity to gamma parameter."""
        target_samples = torch.randn(50, 2)
        generated_samples = torch.randn(50, 2) + 1  # Shifted
        
        # Different gamma values
        mmd_small = MMDDistance(target_samples, gamma=0.1)
        mmd_large = MMDDistance(target_samples, gamma=10.0)
        
        score_small = mmd_small.compute_score(generated_samples)
        score_large = mmd_large.compute_score(generated_samples)
        
        # Both should detect difference but with different magnitudes
        assert score_small > 0
        assert score_large > 0
    
    def test_mmd_distance_dimensionality(self):
        """Test MMD distance with different dimensionalities."""
        dimensions = [1, 2, 5, 10]
        
        for dim in dimensions:
            target_samples = torch.randn(20, dim)
            generated_samples = torch.randn(15, dim)
            
            mmd = MMDDistance(target_samples)
            score = mmd.compute_score(generated_samples)
            
            assert isinstance(score, float)
            assert score >= 0
            assert not np.isnan(score)
    
    @pytest.mark.slow
    def test_mmd_distance_computational_efficiency(self):
        """Test MMD distance computational efficiency."""
        import time
        
        target_samples = torch.randn(300, 2)
        generated_samples = torch.randn(300, 2)
        
        mmd = MMDDistance(target_samples)
        
        start_time = time.time()
        score = mmd.compute_score(generated_samples)
        computation_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert computation_time < 2.0  # seconds
        assert isinstance(score, float)
        assert score >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])