#!/usr/bin/env python3
"""
Example usage of the new MMD metrics in the GaussGAN project.
Shows how to integrate MMDivergence and MMDivergenceFromGMM into training.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from source.metrics import MMDivergence, MMDivergenceFromGMM, ALL_METRICS


def demonstrate_mmd_usage():
    """Demonstrate how to use MMD metrics in GaussGAN project."""
    print("MMD Metrics Usage Example for GaussGAN\n")
    
    # =================================================================
    # Example 1: Using MMDivergence with explicit target samples
    # =================================================================
    print("1. Using MMDivergence with explicit target samples")
    print("-" * 50)
    
    # Load target samples (in practice, this would come from data/normal.pickle)
    np.random.seed(42)
    target_samples = np.concatenate([
        np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]], 500),   # First Gaussian
        np.random.multivariate_normal([-1, -1], [[1, 0], [0, 1]], 500)  # Second Gaussian
    ])
    
    # Create MMD metric
    mmd_metric = MMDivergence(target_samples)
    print(f"Created MMD metric with {len(target_samples)} target samples")
    print(f"Adaptive bandwidths: {mmd_metric.bandwidths}")
    
    # Simulate generated samples from GAN
    generated_samples = torch.tensor(
        np.random.multivariate_normal([0, 0], [[1.5, 0], [0, 1.5]], 100)
    ).float()
    
    # Compute MMD score
    mmd_metric.update(generated_samples)
    mmd_score = mmd_metric.compute()
    print(f"MMD score for generated samples: {mmd_score:.4f}")
    print()
    
    # =================================================================
    # Example 2: Using MMDivergenceFromGMM (compatible with existing metrics)
    # =================================================================
    print("2. Using MMDivergenceFromGMM (compatible with existing metrics)")
    print("-" * 65)
    
    # GMM parameters (same format as LogLikelihood and KLDivergence metrics)
    centroids = [[1, 1], [-1, -1]]
    cov_matrices = [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
    weights = [0.5, 0.5]
    
    # Create MMD metric from GMM (just like LogLikelihood/KLDivergence)
    mmd_gmm = MMDivergenceFromGMM(
        centroids=centroids,
        cov_matrices=cov_matrices,
        weights=weights,
        n_target_samples=1000  # Number of samples to generate from GMM
    )
    print(f"Created MMD metric from GMM with {len(centroids)} components")
    
    # Use same generated samples
    mmd_gmm.update(generated_samples)
    mmd_gmm_score = mmd_gmm.compute()
    print(f"MMD score using GMM-generated targets: {mmd_gmm_score:.4f}")
    print()
    
    # =================================================================
    # Example 3: Comparing with existing metrics
    # =================================================================
    print("3. Comparing MMD with existing metrics")
    print("-" * 40)
    
    # Import existing metrics for comparison
    from source.metrics import LogLikelihood, KLDivergence
    
    # Create existing metrics with same GMM parameters
    log_likelihood = LogLikelihood(centroids, cov_matrices, weights)
    kl_divergence = KLDivergence(centroids, cov_matrices, weights)
    
    # Compute all metrics on the same generated samples
    log_likelihood.update(generated_samples)
    kl_divergence.update(generated_samples)
    
    ll_score = log_likelihood.compute()
    kl_score = kl_divergence.compute()
    
    print(f"Log-Likelihood score: {ll_score:.4f}")
    print(f"KL Divergence score: {kl_score:.4f}")
    print(f"MMD score (from GMM): {mmd_gmm_score:.4f}")
    print()
    
    # =================================================================
    # Example 4: Using in config.yaml format
    # =================================================================
    print("4. Integration with config.yaml")
    print("-" * 32)
    
    print("To use MMD metrics in your GaussGAN training, add to config.yaml:")
    print("```yaml")
    print("metrics: ['IsPositive', 'LogLikelihood', 'KLDivergence', 'MMDivergenceFromGMM']")
    print("```")
    print()
    print("Or for more detailed metrics:")
    print("```yaml")  
    print("metrics: ['IsPositive', 'LogLikelihood', 'KLDivergence', 'MMDivergenceFromGMM', 'WassersteinDistance']")
    print("```")
    print()
    
    # =================================================================
    # Example 5: Performance characteristics
    # =================================================================
    print("5. Performance characteristics")
    print("-" * 30)
    
    # Test with different sample sizes
    import time
    
    sample_sizes = [50, 100, 200, 500]
    print("Sample Size | MMD Time (ms) | KL Time (ms) | LL Time (ms)")
    print("-" * 55)
    
    for n_samples in sample_sizes:
        test_samples = torch.tensor(
            np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], n_samples)
        ).float()
        
        # Time MMD
        mmd_test = MMDivergenceFromGMM(centroids, cov_matrices, weights, n_target_samples=500)
        start = time.time()
        mmd_test.update(test_samples)
        mmd_result = mmd_test.compute()
        mmd_time = (time.time() - start) * 1000
        
        # Time KL divergence
        kl_test = KLDivergence(centroids, cov_matrices, weights)
        start = time.time()
        kl_test.update(test_samples)
        kl_result = kl_test.compute()
        kl_time = (time.time() - start) * 1000
        
        # Time log-likelihood
        ll_test = LogLikelihood(centroids, cov_matrices, weights)
        start = time.time()
        ll_test.update(test_samples)
        ll_result = ll_test.compute()
        ll_time = (time.time() - start) * 1000
        
        print(f"{n_samples:>10d} | {mmd_time:>11.2f} | {kl_time:>9.2f} | {ll_time:>9.2f}")
    
    print()
    
    # =================================================================
    # Example 6: Metric properties
    # =================================================================
    print("6. MMD metric properties")
    print("-" * 24)
    
    print("✓ Non-parametric: No assumptions about distribution shape")
    print("✓ Symmetric: MMD(P, Q) = MMD(Q, P)")
    print("✓ Satisfies metric properties: MMD(P, P) = 0, triangle inequality")
    print("✓ Kernel-based: Uses RBF kernels with adaptive bandwidth selection")
    print("✓ Robust: Handles NaN values, small samples, and edge cases")
    print("✓ Multi-scale: Uses multiple kernel bandwidths for robustness")
    print("✓ Computationally efficient: O(n²) complexity with optimized implementation")
    print()
    
    print("MMD is particularly useful for:")
    print("- Comparing quantum vs classical generator distributions")
    print("- Non-parametric distribution comparison")
    print("- Cases where KL divergence may be unstable")
    print("- Multi-modal distribution evaluation")


if __name__ == "__main__":
    demonstrate_mmd_usage()