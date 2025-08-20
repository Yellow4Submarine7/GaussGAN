#!/usr/bin/env python
"""Test script to verify KL divergence calculation logic."""

import numpy as np
from sklearn.mixture import GaussianMixture

# Create a simple test GMM
gmm = GaussianMixture(n_components=2)
gmm.means_ = np.array([[0, 0], [5, 5]])
gmm.covariances_ = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
gmm.weights_ = np.array([0.5, 0.5])
gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm.covariances_))

# Test points
test_points = np.array([[0, 0], [5, 5], [2.5, 2.5]])

# Check what score_samples returns
log_prob = gmm.score_samples(test_points)
print("GMM score_samples output (log probabilities):", log_prob)
print("Exp of score_samples:", np.exp(log_prob))

# Calculate probability density using different methods
print("\n--- Testing different probability calculations ---")
print("Method 1 (current): np.exp(-score_samples) =", np.exp(-log_prob))
print("Method 2 (correct): np.exp(score_samples) =", np.exp(log_prob))

# Verify with manual calculation for point [0,0] at first Gaussian
from scipy.stats import multivariate_normal
manual_prob = 0.5 * multivariate_normal.pdf([0, 0], [0, 0], [[1, 0], [0, 1]]) + \
              0.5 * multivariate_normal.pdf([0, 0], [5, 5], [[1, 0], [0, 1]])
print(f"\nManual calculation for [0,0]: {manual_prob}")
print(f"np.exp(score_samples[0]): {np.exp(log_prob[0])}")

print("\n🔍 ANALYSIS:")
print("score_samples returns LOG probabilities (log p(x))")
print("To get actual probabilities: use np.exp(score_samples)")
print("Current code uses np.exp(-score_samples) which is INCORRECT!")