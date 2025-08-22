#!/usr/bin/env python3
"""
Test script to validate the new MMD implementations.
Tests both MMDivergence and MMDivergenceFromGMM classes.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from source.metrics import MMDivergence, MMDivergenceFromGMM


def test_mmd_basic():
    """Test basic MMD functionality with known distributions."""
    print("Testing MMD basic functionality...")
    
    # Generate target samples from a known distribution
    np.random.seed(42)
    target_samples = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100)
    
    # Create MMD metric
    mmd_metric = MMDivergence(target_samples)
    
    # Test with samples from same distribution (should have low MMD)
    same_dist_samples = torch.tensor(
        np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 50)
    ).float()
    
    mmd_metric.update(same_dist_samples)
    same_dist_mmd = mmd_metric.compute()
    print(f"MMD for same distribution: {same_dist_mmd:.4f}")
    
    # Reset and test with samples from different distribution  
    mmd_metric.reset()
    different_dist_samples = torch.tensor(
        np.random.multivariate_normal([3, 3], [[1, 0], [0, 1]], 50)
    ).float()
    
    mmd_metric.update(different_dist_samples)
    different_dist_mmd = mmd_metric.compute()
    print(f"MMD for different distribution: {different_dist_mmd:.4f}")
    
    # MMD should be lower for same distribution than different distribution
    assert same_dist_mmd < different_dist_mmd, f"Expected same_dist_mmd < different_dist_mmd, got {same_dist_mmd} >= {different_dist_mmd}"
    print("✓ Basic MMD test passed")


def test_mmd_from_gmm():
    """Test MMD metric initialized from GMM parameters."""
    print("\nTesting MMD from GMM...")
    
    # Define GMM parameters (2D Gaussian mixture)
    centroids = [[0, 0], [3, 3]]
    cov_matrices = [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
    weights = [0.5, 0.5]
    
    # Create MMD metric from GMM
    mmd_gmm = MMDivergenceFromGMM(
        centroids=centroids,
        cov_matrices=cov_matrices, 
        weights=weights,
        n_target_samples=500
    )
    
    # Test with samples from same GMM (should have low MMD)
    np.random.seed(42)
    same_gmm_samples = []
    for i, (centroid, cov, weight) in enumerate(zip(centroids, cov_matrices, weights)):
        n_samples = int(50 * weight)
        if i == len(centroids) - 1:  # Last component gets remaining samples
            n_samples = 50 - len(same_gmm_samples)
        samples = np.random.multivariate_normal(centroid, cov, n_samples)
        same_gmm_samples.extend(samples)
    
    same_gmm_samples = torch.tensor(np.array(same_gmm_samples)).float()
    mmd_gmm.update(same_gmm_samples)
    same_gmm_mmd = mmd_gmm.compute()
    print(f"MMD for samples from same GMM: {same_gmm_mmd:.4f}")
    
    # Reset and test with samples from different distribution
    mmd_gmm.reset() 
    different_samples = torch.tensor(
        np.random.multivariate_normal([10, 10], [[2, 0], [0, 2]], 50)
    ).float()
    
    mmd_gmm.update(different_samples)
    different_gmm_mmd = mmd_gmm.compute()
    print(f"MMD for different distribution: {different_gmm_mmd:.4f}")
    
    # MMD should be lower for same GMM than different distribution
    assert same_gmm_mmd < different_gmm_mmd, f"Expected same_gmm_mmd < different_gmm_mmd, got {same_gmm_mmd} >= {different_gmm_mmd}"
    print("✓ MMD from GMM test passed")


def test_mmd_edge_cases():
    """Test MMD with edge cases like NaN values, empty inputs."""
    print("\nTesting MMD edge cases...")
    
    # Target samples with some NaN values
    target_samples = np.array([[1, 2], [3, 4], [np.nan, 6], [7, 8]])
    mmd_metric = MMDivergence(target_samples)
    
    # Generated samples with NaN values
    generated_samples = torch.tensor([[1, 2], [np.nan, 4], [5, 6]]).float()
    
    mmd_metric.update(generated_samples)
    result = mmd_metric.compute()
    print(f"MMD with NaN values: {result:.4f}")
    
    # Should handle NaN gracefully and return a valid number
    assert not np.isnan(result), f"MMD should handle NaN values gracefully, got {result}"
    print("✓ NaN handling test passed")
    
    # Test with very small samples
    mmd_metric.reset()
    small_samples = torch.tensor([[0, 0]]).float()  # Single sample
    mmd_metric.update(small_samples)
    small_result = mmd_metric.compute()
    print(f"MMD with single sample: {small_result:.4f}")
    
    assert not np.isnan(small_result), f"MMD should handle small samples gracefully, got {small_result}"
    print("✓ Small sample test passed")


def test_adaptive_bandwidth():
    """Test that adaptive bandwidth selection works."""
    print("\nTesting adaptive bandwidth selection...")
    
    # Create samples with different scales
    tight_samples = np.random.multivariate_normal([0, 0], [[0.1, 0], [0, 0.1]], 50)
    wide_samples = np.random.multivariate_normal([0, 0], [[10, 0], [0, 10]], 50)
    
    mmd_tight = MMDivergence(tight_samples)
    mmd_wide = MMDivergence(wide_samples)
    
    print(f"Tight distribution bandwidths: {mmd_tight.bandwidths}")
    print(f"Wide distribution bandwidths: {mmd_wide.bandwidths}")
    
    # Wide distribution should have larger bandwidths
    assert np.mean(mmd_wide.bandwidths) > np.mean(mmd_tight.bandwidths), \
        "Wide distribution should have larger bandwidths"
    print("✓ Adaptive bandwidth test passed")


if __name__ == "__main__":
    print("Testing MMD implementation...\n")
    
    try:
        test_mmd_basic()
        test_mmd_from_gmm()
        test_mmd_edge_cases()
        test_adaptive_bandwidth()
        
        print("\n🎉 All MMD tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)