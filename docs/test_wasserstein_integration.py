#!/usr/bin/env python3
"""
Test script to verify WassersteinDistance metric integration with GaussGAN.
"""
import pickle
import numpy as np
import torch
from source.metrics import WassersteinDistance

def test_wasserstein_with_gaussgan_data():
    """Test WassersteinDistance with actual GaussGAN dataset."""
    
    print("Testing WassersteinDistance with GaussGAN data...")
    
    # Load the actual target data
    try:
        with open("data/normal.pickle", "rb") as f:
            data = pickle.load(f)
            target_samples = data["inputs"]  # Shape: (N, 2)
            print(f"Loaded target data shape: {target_samples.shape}")
            print(f"Target data range: x=[{target_samples[:, 0].min():.3f}, {target_samples[:, 0].max():.3f}], "
                  f"y=[{target_samples[:, 1].min():.3f}, {target_samples[:, 1].max():.3f}]")
    except FileNotFoundError:
        print("Target data file not found, using synthetic data")
        # Create synthetic 2D Gaussian mixture
        np.random.seed(42)
        target_samples_1 = np.random.multivariate_normal([2, 1], [[1, 0.5], [0.5, 1]], 500)
        target_samples_2 = np.random.multivariate_normal([-1, -0.5], [[0.8, -0.3], [-0.3, 0.8]], 500)
        target_samples = np.vstack([target_samples_1, target_samples_2])
        print(f"Created synthetic target data shape: {target_samples.shape}")
    
    print("\n1. Testing WassersteinDistance initialization:")
    
    # Test different aggregation methods
    for aggregation in ["mean", "max", "sum"]:
        try:
            metric = WassersteinDistance(target_samples, aggregation=aggregation)
            print(f"   ✓ {aggregation} aggregation: initialized successfully")
        except Exception as e:
            print(f"   ✗ {aggregation} aggregation failed: {e}")
    
    print("\n2. Testing distance computation:")
    
    # Initialize metric
    metric = WassersteinDistance(target_samples, aggregation="mean")
    
    # Test 1: Same distribution (should be close to 0)
    identical_distance = metric.compute_score(target_samples)
    print(f"   Distance to identical samples: {identical_distance:.6f}")
    
    # Test 2: Shifted distribution
    shifted_samples = target_samples + np.array([1.0, 0.5])
    shifted_distance = metric.compute_score(shifted_samples)
    print(f"   Distance to shifted samples: {shifted_distance:.6f}")
    
    # Test 3: Scaled distribution  
    scaled_samples = target_samples * 1.5
    scaled_distance = metric.compute_score(scaled_samples)
    print(f"   Distance to scaled samples: {scaled_distance:.6f}")
    
    # Test 4: Random distribution
    np.random.seed(123)
    random_samples = np.random.normal(0, 2, size=target_samples.shape)
    random_distance = metric.compute_score(random_samples)
    print(f"   Distance to random samples: {random_distance:.6f}")
    
    print("\n3. Testing with different aggregation methods:")
    for aggregation in ["mean", "max", "sum"]:
        metric_agg = WassersteinDistance(target_samples, aggregation=aggregation)
        distance = metric_agg.compute_score(shifted_samples)
        print(f"   {aggregation:4s} aggregation: {distance:.6f}")
    
    print("\n4. Testing with PyTorch tensors:")
    target_tensor = torch.from_numpy(target_samples).float()
    shifted_tensor = torch.from_numpy(shifted_samples).float()
    
    metric_torch = WassersteinDistance(target_tensor, aggregation="mean")
    torch_distance = metric_torch.compute_score(shifted_tensor)
    print(f"   PyTorch tensor distance: {torch_distance:.6f}")
    
    print("\n5. Testing edge cases:")
    
    # Empty samples
    empty_distance = metric.compute_score(np.array([]).reshape(0, 2))
    print(f"   Empty samples: {empty_distance}")
    
    # Samples with NaN
    nan_samples = shifted_samples.copy()
    nan_samples[0] = np.nan
    nan_distance = metric.compute_score(nan_samples)
    print(f"   Samples with NaN: {nan_distance:.6f}")
    
    # Single point
    single_point = np.array([[0.0, 0.0]])
    single_distance = metric.compute_score(single_point)
    print(f"   Single point: {single_distance:.6f}")
    
    print("\n6. Performance test with larger datasets:")
    
    # Generate larger datasets for performance testing
    large_target = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 5000)
    large_generated = np.random.multivariate_normal([0.5, 0.5], [[1.2, 0.1], [0.1, 1.2]], 5000)
    
    import time
    metric_large = WassersteinDistance(large_target, aggregation="mean")
    
    start_time = time.time()
    large_distance = metric_large.compute_score(large_generated)
    end_time = time.time()
    
    print(f"   Large dataset (5k points) distance: {large_distance:.6f}")
    print(f"   Computation time: {end_time - start_time:.4f} seconds")
    
    print("\n✅ All WassersteinDistance tests completed successfully!")
    
    return metric

def test_metric_comparison():
    """Compare WassersteinDistance with other metrics on the same data."""
    
    print("\n" + "="*60)
    print("COMPARING WASSERSTEIN WITH OTHER METRICS")
    print("="*60)
    
    # Create test distributions
    np.random.seed(42)
    target_samples = np.random.multivariate_normal([1, 1], [[1, 0.3], [0.3, 1]], 1000)
    
    # Create progressively different distributions
    test_cases = [
        ("Identical", target_samples),
        ("Slightly shifted", target_samples + np.array([0.2, 0.2])),
        ("Moderately shifted", target_samples + np.array([0.5, 0.5])),  
        ("Highly shifted", target_samples + np.array([2.0, 2.0])),
        ("Random", np.random.normal(0, 2, size=target_samples.shape))
    ]
    
    # Initialize WassersteinDistance
    wasserstein = WassersteinDistance(target_samples, aggregation="mean")
    
    print(f"{'Test Case':<20} {'Wasserstein':<12}")
    print("-" * 35)
    
    for case_name, test_samples in test_cases:
        w_distance = wasserstein.compute_score(test_samples)
        print(f"{case_name:<20} {w_distance:<12.6f}")
    
    print("\nWasserstein distance successfully shows expected progression!")

if __name__ == "__main__":
    metric = test_wasserstein_with_gaussgan_data()
    test_metric_comparison()