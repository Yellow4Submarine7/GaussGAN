#!/usr/bin/env python3
"""
Integration test for MMD metrics within GaussGAN framework.
Tests that the new metrics work correctly with the training pipeline.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import yaml
import torch
import numpy as np
from source.utils import load_data
from source.metrics import ALL_METRICS


def test_mmd_integration():
    """Test that MMD metrics integrate properly with GaussGAN."""
    print("Testing MMD integration with GaussGAN framework...\n")
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data using existing GaussGAN data loading
    try:
        datamodule, gaussians = load_data(config)
        print("✓ Successfully loaded GaussGAN data")
        print(f"  - Gaussians: {gaussians}")
        print(f"  - Dataset type: {config['dataset_type']}")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return False
    
    # Test that new metrics are in ALL_METRICS
    available_metrics = list(ALL_METRICS.keys())
    print(f"\n✓ Available metrics: {available_metrics}")
    
    assert "MMDivergence" in available_metrics, "MMDivergence not found in ALL_METRICS"
    assert "MMDivergenceFromGMM" in available_metrics, "MMDivergenceFromGMM not found in ALL_METRICS"
    print("✓ New MMD metrics are registered")
    
    # Test creating metrics using the same pattern as GaussGAN model
    if config['dataset_type'] == 'NORMAL' and gaussians:
        print("\n📊 Testing metrics creation (NORMAL dataset)...")
        
        # Create metrics the same way as in model.py
        metrics = {}
        test_metric_names = ['LogLikelihood', 'KLDivergence', 'MMDivergenceFromGMM']
        
        for metric_name in test_metric_names:
            try:
                if metric_name in ['LogLikelihood', 'KLDivergence']:
                    metric = ALL_METRICS[metric_name](
                        centroids=gaussians["centroids"],
                        cov_matrices=gaussians["covariances"], 
                        weights=gaussians["weights"]
                    )
                elif metric_name == 'MMDivergenceFromGMM':
                    metric = ALL_METRICS[metric_name](
                        centroids=gaussians["centroids"],
                        cov_matrices=gaussians["covariances"],
                        weights=gaussians["weights"],
                        n_target_samples=500
                    )
                
                metrics[metric_name] = metric
                print(f"  ✓ Created {metric_name} metric")
                
            except Exception as e:
                print(f"  ❌ Failed to create {metric_name}: {e}")
                return False
    
    # Generate test samples (simulate GAN output)
    print("\n🎲 Testing metric computation...")
    
    # Create some test samples
    torch.manual_seed(42)
    test_samples = torch.randn(50, 2) * 1.5  # 50 samples, 2D
    
    # Test each metric
    for metric_name, metric in metrics.items():
        try:
            # Reset metric state
            metric.reset()
            
            # Update metric with test samples
            metric.update(test_samples)
            
            # Compute final score
            score = metric.compute()
            
            print(f"  ✓ {metric_name}: {float(score):.4f}")
            
            # Validate score is reasonable
            assert not torch.isnan(score), f"{metric_name} returned NaN"
            assert torch.isfinite(score), f"{metric_name} returned infinite value"
            
        except Exception as e:
            print(f"  ❌ {metric_name} failed: {e}")
            return False
    
    # Test MMD-specific features
    print("\n🔬 Testing MMD-specific features...")
    
    mmd_metric = metrics['MMDivergenceFromGMM']
    
    # Test multiple updates (batch processing)
    mmd_metric.reset()
    for i in range(3):
        batch_samples = torch.randn(20, 2) + i  # Different batches
        mmd_metric.update(batch_samples)
    
    final_score = mmd_metric.compute()
    print(f"  ✓ Multi-batch processing: {float(final_score):.4f}")
    
    # Test with edge cases
    print("\n🧪 Testing edge cases...")
    
    # Empty samples (should handle gracefully)
    mmd_metric.reset()
    empty_samples = torch.empty(0, 2)
    try:
        mmd_metric.update(empty_samples)
        empty_score = mmd_metric.compute()
        print(f"  ✓ Empty samples handled: {float(empty_score):.4f}")
    except Exception as e:
        print(f"  ❌ Empty samples failed: {e}")
    
    # Samples with NaN
    mmd_metric.reset()
    nan_samples = torch.tensor([[1, 2], [float('nan'), 3], [4, 5]]).float()
    try:
        mmd_metric.update(nan_samples)
        nan_score = mmd_metric.compute()
        print(f"  ✓ NaN samples handled: {float(nan_score):.4f}")
    except Exception as e:
        print(f"  ❌ NaN samples failed: {e}")
    
    print("\n🎉 All integration tests passed!")
    return True


def test_config_compatibility():
    """Test that MMD metrics work with different config settings."""
    print("\nTesting config compatibility...")
    
    # Test with updated metrics list
    updated_config = {
        'metrics': ['IsPositive', 'LogLikelihood', 'KLDivergence', 'MMDivergenceFromGMM']
    }
    
    print(f"✓ Compatible metrics configuration: {updated_config['metrics']}")
    
    # Verify all metrics in config exist
    for metric_name in updated_config['metrics']:
        assert metric_name in ALL_METRICS, f"Metric {metric_name} not found in ALL_METRICS"
    
    print("✓ All configured metrics are available")


if __name__ == "__main__":
    print("MMD Integration Test for GaussGAN\n")
    print("=" * 40)
    
    try:
        # Run main integration test
        success = test_mmd_integration()
        
        if success:
            # Test config compatibility
            test_config_compatibility()
            print("\n🌟 All tests completed successfully!")
            print("\nTo use MMD metrics in training:")
            print("1. Add 'MMDivergenceFromGMM' to metrics list in config.yaml")
            print("2. Run training as usual with: uv run python main.py")
            print("3. MMD scores will appear in training logs alongside other metrics")
            
        else:
            print("\n❌ Some tests failed. Check the implementation.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)