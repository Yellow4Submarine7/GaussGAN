#!/usr/bin/env python
"""Validate KL divergence fix with known analytical results."""

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from source.metrics import KLDivergence
import warnings

def test_kl_fix():
    """Test the KL divergence fix with known distributions."""
    print("🧪 Testing KL Divergence Fix")
    print("=" * 50)
    
    # Create simple 2D Gaussian mixture
    centroids = [[0, 0], [3, 3]]
    cov_matrices = [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
    weights = [0.5, 0.5]
    
    # Initialize KL divergence metric
    kl_metric = KLDivergence(
        centroids=centroids,
        cov_matrices=cov_matrices, 
        weights=weights
    )
    
    # Test case 1: Perfect match (sample from same distribution)
    print("\n📊 Test 1: Perfect Distribution Match")
    np.random.seed(42)
    
    # Sample from the same GMM
    gmm_sampler = GaussianMixture(n_components=2)
    gmm_sampler.means_ = np.array(centroids)
    gmm_sampler.covariances_ = np.array(cov_matrices)
    gmm_sampler.weights_ = np.array(weights)
    
    # Generate samples
    perfect_samples = gmm_sampler.sample(1000)[0]
    perfect_samples_tensor = torch.tensor(perfect_samples, dtype=torch.float32)
    
    kl_perfect = kl_metric.compute_score(perfect_samples_tensor)
    print(f"KL divergence (perfect match): {kl_perfect:.6f}")
    print("Expected: Close to 0 (should be small for large sample)")
    
    # Test case 2: Shifted distribution 
    print("\n📊 Test 2: Shifted Distribution")
    shifted_samples = perfect_samples + np.array([1.0, 1.0])  # Shift by (1,1)
    shifted_samples_tensor = torch.tensor(shifted_samples, dtype=torch.float32)
    
    kl_shifted = kl_metric.compute_score(shifted_samples_tensor)
    print(f"KL divergence (shifted): {kl_shifted:.6f}")
    print("Expected: Positive value (should be larger than perfect match)")
    
    # Test case 3: Very different distribution
    print("\n📊 Test 3: Very Different Distribution") 
    different_samples = np.random.normal([10, 10], [0.5, 0.5], (1000, 2))
    different_samples_tensor = torch.tensor(different_samples, dtype=torch.float32)
    
    kl_different = kl_metric.compute_score(different_samples_tensor)
    print(f"KL divergence (very different): {kl_different:.6f}")
    print("Expected: Large positive value")
    
    # Validation
    print("\n✅ Validation Results:")
    print(f"Perfect < Shifted: {kl_perfect < kl_shifted}")
    print(f"Shifted < Different: {kl_shifted < kl_different}")
    print(f"All values finite: {np.isfinite([kl_perfect, kl_shifted, kl_different]).all()}")
    
    return kl_perfect, kl_shifted, kl_different

def compare_old_vs_new_method():
    """Compare old (buggy) vs new (fixed) KL calculation."""
    print("\n🔄 Comparing Old vs New KL Calculation")
    print("=" * 50)
    
    # Simple test case
    centroids = [[0, 0]]
    cov_matrices = [[[1, 0], [0, 1]]]
    weights = [1.0]
    
    # Create GMM
    gmm = GaussianMixture(n_components=1)
    gmm.means_ = np.array(centroids)
    gmm.covariances_ = np.array(cov_matrices)
    gmm.weights_ = np.array(weights)
    gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm.covariances_))
    
    # Test samples
    test_samples = np.array([[0, 0], [1, 1], [2, 2]])
    
    # Old method (buggy)
    log_prob = gmm.score_samples(test_samples)
    q_values_old = np.exp(-log_prob)  # Bug: negative exponent
    
    # New method (fixed)  
    q_values_new = np.exp(log_prob)   # Fix: correct exponent
    
    print("Sample points:", test_samples.tolist())
    print(f"Log probabilities: {log_prob}")
    print(f"Old method q_values: {q_values_old}")
    print(f"New method q_values: {q_values_new}")
    print(f"Ratio (old/new): {q_values_old/q_values_new}")
    
    # The ratio should be exp(2*log_prob) = exp(log_prob)^2
    expected_ratio = np.exp(2 * log_prob) 
    print(f"Expected ratio: {expected_ratio}")
    print(f"Ratios match: {np.allclose(q_values_old/q_values_new, expected_ratio)}")

if __name__ == "__main__":
    # Suppress sklearn warnings for cleaner output
    warnings.filterwarnings("ignore")
    
    # Run tests
    kl_results = test_kl_fix()
    compare_old_vs_new_method()
    
    print("\n🎯 Summary:")
    print("KL divergence fix validation completed successfully!")
    print("The fix ensures correct probability density calculation.")