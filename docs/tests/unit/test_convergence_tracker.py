"""
Unit tests for ConvergenceTracker functionality and accuracy.
Tests convergence detection, early stopping, and tracking accuracy.
"""

import pytest
import numpy as np
import warnings
from unittest.mock import Mock, patch

from source.metrics import ConvergenceTracker


class TestConvergenceTracker:
    """Test suite for ConvergenceTracker."""
    
    def test_initialization_default_parameters(self):
        """Test ConvergenceTracker initialization with default parameters."""
        tracker = ConvergenceTracker()
        
        assert tracker.patience == 10
        assert tracker.min_delta == 1e-4
        assert tracker.monitor_metric == "KLDivergence"
        assert tracker.window_size == 5
        assert tracker.metric_history == {}
        assert tracker.loss_history == {"d_loss": [], "g_loss": []}
        assert tracker.best_metric_value is None
        assert tracker.epochs_without_improvement == 0
        assert tracker.converged is False
        assert tracker.convergence_epoch is None
    
    def test_initialization_custom_parameters(self):
        """Test ConvergenceTracker initialization with custom parameters."""
        tracker = ConvergenceTracker(
            patience=15,
            min_delta=1e-3,
            monitor_metric="LogLikelihood",
            window_size=10
        )
        
        assert tracker.patience == 15
        assert tracker.min_delta == 1e-3
        assert tracker.monitor_metric == "LogLikelihood"
        assert tracker.window_size == 10
    
    def test_update_first_epoch_initialization(self):
        """Test first epoch update initializes tracking properly."""
        tracker = ConvergenceTracker(patience=5, min_delta=0.01)
        
        metrics = {"KLDivergence": 0.5, "LogLikelihood": -2.0}
        result = tracker.update(epoch=0, metrics=metrics, d_loss=1.0, g_loss=0.8)
        
        # Should initialize best metric value
        assert tracker.best_metric_value == 0.5
        assert tracker.epochs_without_improvement == 0
        assert tracker.converged is False
        
        # Should store losses
        assert tracker.loss_history["d_loss"] == [1.0]
        assert tracker.loss_history["g_loss"] == [0.8]
        
        # Should store metrics
        assert tracker.metric_history["KLDivergence"] == [0.5]
        assert tracker.metric_history["LogLikelihood"] == [-2.0]
        
        # Check return value
        assert result["converged"] is False
        assert result["best_metric_value"] == 0.5
        assert result["epochs_without_improvement"] == 0
        assert result["current_epoch"] == 0
    
    def test_update_improvement_detected(self):
        """Test update when improvement is detected."""
        tracker = ConvergenceTracker(patience=3, min_delta=0.1)
        
        # First epoch
        tracker.update(epoch=0, metrics={"KLDivergence": 1.0})
        
        # Second epoch with improvement
        result = tracker.update(epoch=1, metrics={"KLDivergence": 0.8})  # Improvement of 0.2 > 0.1
        
        assert tracker.best_metric_value == 0.8
        assert tracker.epochs_without_improvement == 0
        assert tracker.converged is False
        
        assert result["converged"] is False
        assert result["best_metric_value"] == 0.8
        assert result["epochs_without_improvement"] == 0
    
    def test_update_no_significant_improvement(self):
        """Test update when no significant improvement is detected."""
        tracker = ConvergenceTracker(patience=3, min_delta=0.1)
        
        # First epoch
        tracker.update(epoch=0, metrics={"KLDivergence": 1.0})
        
        # Second epoch with small improvement
        result = tracker.update(epoch=1, metrics={"KLDivergence": 0.95})  # Improvement of 0.05 < 0.1
        
        assert tracker.best_metric_value == 1.0  # Should not update
        assert tracker.epochs_without_improvement == 1
        assert tracker.converged is False
        
        assert result["epochs_without_improvement"] == 1
    
    def test_update_convergence_detection(self):
        """Test convergence detection when patience is exceeded."""
        tracker = ConvergenceTracker(patience=2, min_delta=0.1)
        
        # Initial value
        tracker.update(epoch=0, metrics={"KLDivergence": 1.0})
        
        # No improvement for patience epochs
        tracker.update(epoch=1, metrics={"KLDivergence": 1.0})  # No improvement
        result = tracker.update(epoch=2, metrics={"KLDivergence": 1.0})  # Still no improvement
        
        assert tracker.converged is True
        assert tracker.convergence_epoch == 2
        assert tracker.epochs_without_improvement == 2
        
        assert result["converged"] is True
        assert result["convergence_epoch"] == 2
    
    def test_update_with_nan_metric_values(self):
        """Test update handles NaN metric values properly."""
        tracker = ConvergenceTracker(patience=3, min_delta=0.1)
        
        # First epoch with valid value
        tracker.update(epoch=0, metrics={"KLDivergence": 1.0})
        
        # Second epoch with NaN value
        result = tracker.update(epoch=1, metrics={"KLDivergence": np.nan})
        
        # Should not update best value or convergence status
        assert tracker.best_metric_value == 1.0
        assert tracker.epochs_without_improvement == 0  # NaN should not count
        assert tracker.converged is False
        
        # NaN should not be stored in history
        assert len(tracker.metric_history["KLDivergence"]) == 1
    
    def test_update_with_inf_metric_values(self):
        """Test update handles infinite metric values properly."""
        tracker = ConvergenceTracker(patience=3, min_delta=0.1)
        
        # First epoch with valid value
        tracker.update(epoch=0, metrics={"KLDivergence": 1.0})
        
        # Second epoch with infinite value
        result = tracker.update(epoch=1, metrics={"KLDivergence": np.inf})
        
        # Should not update best value or convergence status
        assert tracker.best_metric_value == 1.0
        assert tracker.epochs_without_improvement == 0
        assert tracker.converged is False
    
    def test_update_multiple_metrics(self):
        """Test update with multiple metrics."""
        tracker = ConvergenceTracker(monitor_metric="KLDivergence")
        
        metrics = {
            "KLDivergence": 0.5,
            "LogLikelihood": -2.0,
            "IsPositive": 0.8,
            "MMDivergence": 0.3
        }
        
        tracker.update(epoch=0, metrics=metrics)
        
        # Should store all metrics
        for metric_name, value in metrics.items():
            assert metric_name in tracker.metric_history
            assert tracker.metric_history[metric_name] == [value]
        
        # Should only monitor specified metric for convergence
        assert tracker.best_metric_value == 0.5  # KLDivergence value
    
    def test_update_missing_monitor_metric(self):
        """Test update when monitor metric is missing from metrics dict."""
        tracker = ConvergenceTracker(monitor_metric="MissingMetric")
        
        result = tracker.update(epoch=0, metrics={"KLDivergence": 0.5})
        
        # Should not initialize best value
        assert tracker.best_metric_value is None
        assert tracker.epochs_without_improvement == 0
        assert tracker.converged is False
    
    def test_update_loss_tracking(self):
        """Test loss tracking functionality."""
        tracker = ConvergenceTracker()
        
        # Update with losses
        tracker.update(epoch=0, metrics={}, d_loss=1.5, g_loss=0.8)
        tracker.update(epoch=1, metrics={}, d_loss=1.2, g_loss=0.9)
        tracker.update(epoch=2, metrics={}, d_loss=1.0, g_loss=0.7)
        
        assert tracker.loss_history["d_loss"] == [1.5, 1.2, 1.0]
        assert tracker.loss_history["g_loss"] == [0.8, 0.9, 0.7]
    
    def test_update_optional_loss_parameters(self):
        """Test update with optional loss parameters."""
        tracker = ConvergenceTracker()
        
        # Update without losses
        tracker.update(epoch=0, metrics={"KLDivergence": 0.5})
        
        # Update with only discriminator loss
        tracker.update(epoch=1, metrics={"KLDivergence": 0.4}, d_loss=1.0)
        
        # Update with only generator loss
        tracker.update(epoch=2, metrics={"KLDivergence": 0.3}, g_loss=0.5)
        
        assert tracker.loss_history["d_loss"] == [1.0]
        assert tracker.loss_history["g_loss"] == [0.5]
    
    def test_get_convergence_info_basic(self):
        """Test get_convergence_info returns correct information."""
        tracker = ConvergenceTracker(patience=5)
        
        # Add some history
        tracker.update(epoch=0, metrics={"KLDivergence": 1.0})
        tracker.update(epoch=1, metrics={"KLDivergence": 0.8})
        
        info = tracker.get_convergence_info(current_epoch=1)
        
        expected_keys = [
            "converged", "convergence_epoch", "epochs_without_improvement",
            "best_metric_value", "current_epoch"
        ]
        
        for key in expected_keys:
            assert key in info
        
        assert info["converged"] is False
        assert info["convergence_epoch"] is None
        assert info["epochs_without_improvement"] == 0
        assert info["best_metric_value"] == 0.8
        assert info["current_epoch"] == 1
    
    def test_get_convergence_info_with_loss_stability(self):
        """Test get_convergence_info includes loss stability metrics."""
        tracker = ConvergenceTracker(window_size=3)
        
        # Add enough loss history for stability calculation
        for i in range(5):
            tracker.update(
                epoch=i,
                metrics={"KLDivergence": 1.0 - i * 0.1},
                d_loss=1.5 - i * 0.1,
                g_loss=0.8 + i * 0.05
            )
        
        info = tracker.get_convergence_info(current_epoch=4)
        
        # Should include stability metrics
        assert "g_loss_stability" in info
        assert "g_loss_trend" in info
        assert "d_loss_stability" in info
        assert "d_loss_trend" in info
        
        # Stability should be non-negative
        assert info["g_loss_stability"] >= 0
        assert info["d_loss_stability"] >= 0
    
    def test_get_convergence_info_insufficient_loss_history(self):
        """Test get_convergence_info when loss history is insufficient."""
        tracker = ConvergenceTracker(window_size=5)
        
        # Add only 2 epochs of data
        tracker.update(epoch=0, metrics={"KLDivergence": 1.0}, d_loss=1.0, g_loss=0.5)
        tracker.update(epoch=1, metrics={"KLDivergence": 0.9}, d_loss=0.9, g_loss=0.6)
        
        info = tracker.get_convergence_info(current_epoch=1)
        
        # Should not include stability metrics
        assert "g_loss_stability" not in info
        assert "d_loss_stability" not in info
    
    def test_should_stop_early_convergence(self):
        """Test should_stop_early returns True when converged."""
        tracker = ConvergenceTracker(patience=1)
        
        # Trigger convergence
        tracker.update(epoch=0, metrics={"KLDivergence": 1.0})
        tracker.update(epoch=1, metrics={"KLDivergence": 1.0})  # No improvement
        
        assert tracker.should_stop_early() is True
    
    def test_should_stop_early_not_converged(self):
        """Test should_stop_early returns False when not converged."""
        tracker = ConvergenceTracker(patience=5)
        
        tracker.update(epoch=0, metrics={"KLDivergence": 1.0})
        tracker.update(epoch=1, metrics={"KLDivergence": 0.8})  # Improvement
        
        assert tracker.should_stop_early() is False
    
    def test_reset_functionality(self):
        """Test reset clears all tracking state."""
        tracker = ConvergenceTracker(patience=3)
        
        # Add some state
        tracker.update(epoch=0, metrics={"KLDivergence": 1.0}, d_loss=1.0, g_loss=0.5)
        tracker.update(epoch=1, metrics={"KLDivergence": 0.8}, d_loss=0.9, g_loss=0.6)
        
        # Verify state exists
        assert len(tracker.metric_history) > 0
        assert len(tracker.loss_history["d_loss"]) > 0
        assert tracker.best_metric_value is not None
        
        # Reset
        tracker.reset()
        
        # Verify state is cleared
        assert tracker.metric_history == {}
        assert tracker.loss_history == {"d_loss": [], "g_loss": []}
        assert tracker.best_metric_value is None
        assert tracker.epochs_without_improvement == 0
        assert tracker.converged is False
        assert tracker.convergence_epoch is None
    
    def test_convergence_with_different_monitor_metrics(self):
        """Test convergence detection with different monitor metrics."""
        metrics_to_test = ["KLDivergence", "LogLikelihood", "MMDivergence"]
        
        for monitor_metric in metrics_to_test:
            tracker = ConvergenceTracker(patience=2, monitor_metric=monitor_metric)
            
            # Simulate no improvement
            for epoch in range(3):
                metrics = {metric: 1.0 for metric in metrics_to_test}
                tracker.update(epoch=epoch, metrics=metrics)
            
            assert tracker.converged is True
            assert tracker.convergence_epoch == 2
    
    def test_improvement_threshold_sensitivity(self):
        """Test sensitivity to different improvement thresholds."""
        small_threshold = ConvergenceTracker(patience=1, min_delta=0.001)
        large_threshold = ConvergenceTracker(patience=1, min_delta=0.1)
        
        # Small improvement
        improvement = 0.01
        
        # Small threshold should detect improvement
        small_threshold.update(epoch=0, metrics={"KLDivergence": 1.0})
        small_threshold.update(epoch=1, metrics={"KLDivergence": 1.0 - improvement})
        assert small_threshold.epochs_without_improvement == 0
        
        # Large threshold should not detect improvement
        large_threshold.update(epoch=0, metrics={"KLDivergence": 1.0})
        large_threshold.update(epoch=1, metrics={"KLDivergence": 1.0 - improvement})
        assert large_threshold.epochs_without_improvement == 1
    
    def test_loss_trend_calculation(self):
        """Test loss trend calculation accuracy."""
        tracker = ConvergenceTracker(window_size=3)
        
        # Add decreasing losses (negative trend)
        decreasing_losses = [1.0, 0.9, 0.8, 0.7, 0.6]
        for i, loss in enumerate(decreasing_losses):
            tracker.update(epoch=i, metrics={}, d_loss=loss)
        
        info = tracker.get_convergence_info(current_epoch=4)
        
        # Should detect negative trend (improvement)
        assert info["d_loss_trend"] < 0
        
        # Reset and test increasing losses
        tracker.reset()
        increasing_losses = [0.5, 0.6, 0.7, 0.8, 0.9]
        for i, loss in enumerate(increasing_losses):
            tracker.update(epoch=i, metrics={}, d_loss=loss)
        
        info = tracker.get_convergence_info(current_epoch=4)
        
        # Should detect positive trend (getting worse)
        assert info["d_loss_trend"] > 0
    
    def test_loss_stability_calculation(self):
        """Test loss stability calculation accuracy."""
        tracker = ConvergenceTracker(window_size=4)
        
        # Add stable losses (low stability/variance)
        stable_losses = [1.0, 1.01, 0.99, 1.02, 0.98]
        for i, loss in enumerate(stable_losses):
            tracker.update(epoch=i, metrics={}, g_loss=loss)
        
        info = tracker.get_convergence_info(current_epoch=4)
        
        # Should detect low stability (small standard deviation)
        stable_std = info["g_loss_stability"]
        
        # Reset and test volatile losses
        tracker.reset()
        volatile_losses = [1.0, 0.5, 1.5, 0.3, 1.8]
        for i, loss in enumerate(volatile_losses):
            tracker.update(epoch=i, metrics={}, g_loss=loss)
        
        info = tracker.get_convergence_info(current_epoch=4)
        
        # Should detect high stability (large standard deviation)
        volatile_std = info["g_loss_stability"]
        
        assert volatile_std > stable_std
    
    def test_early_convergence_scenario(self):
        """Test early convergence detection scenario."""
        tracker = ConvergenceTracker(patience=3, min_delta=0.01)
        
        # Simulate quick convergence
        epoch_metrics = [
            {"KLDivergence": 1.0},
            {"KLDivergence": 0.5},    # Big improvement
            {"KLDivergence": 0.48},   # Small improvement (< min_delta)
            {"KLDivergence": 0.49},   # No improvement
            {"KLDivergence": 0.485},  # No improvement
            {"KLDivergence": 0.490},  # No improvement - should trigger convergence
        ]
        
        for epoch, metrics in enumerate(epoch_metrics):
            result = tracker.update(epoch=epoch, metrics=metrics)
            
            if epoch < 5:
                assert result["converged"] is False
            else:
                assert result["converged"] is True
                assert result["convergence_epoch"] == 5
                break
    
    def test_convergence_robustness_to_noise(self):
        """Test convergence detection robustness to noisy metrics."""
        tracker = ConvergenceTracker(patience=5, min_delta=0.05)
        
        # Base value with noise
        base_value = 1.0
        noise_level = 0.02  # Small noise
        
        np.random.seed(42)  # For reproducibility
        
        for epoch in range(10):
            noisy_value = base_value + np.random.normal(0, noise_level)
            tracker.update(epoch=epoch, metrics={"KLDivergence": noisy_value})
        
        # Should eventually converge despite noise
        assert tracker.converged is True
    
    def test_multiple_metrics_convergence_independence(self):
        """Test that convergence only depends on monitored metric."""
        tracker = ConvergenceTracker(patience=2, monitor_metric="KLDivergence")
        
        # KL divergence improves, but other metrics get worse
        metrics_sequence = [
            {"KLDivergence": 1.0, "LogLikelihood": -1.0},
            {"KLDivergence": 0.8, "LogLikelihood": -2.0},  # KL improves, LL worse
            {"KLDivergence": 0.7, "LogLikelihood": -3.0},  # KL improves, LL worse
        ]
        
        for epoch, metrics in enumerate(metrics_sequence):
            result = tracker.update(epoch=epoch, metrics=metrics)
        
        # Should not converge because KL divergence keeps improving
        assert tracker.converged is False
        assert tracker.epochs_without_improvement == 0
    
    @pytest.mark.parametrize("patience", [1, 3, 5, 10])
    def test_patience_parameter_correctness(self, patience):
        """Test that patience parameter works correctly for different values."""
        tracker = ConvergenceTracker(patience=patience, min_delta=0.1)
        
        # Initialize with a value
        tracker.update(epoch=0, metrics={"KLDivergence": 1.0})
        
        # Add epochs without improvement
        for epoch in range(1, patience + 2):
            result = tracker.update(epoch=epoch, metrics={"KLDivergence": 1.0})
            
            if epoch <= patience:
                assert result["converged"] is False
            else:
                assert result["converged"] is True
                assert result["convergence_epoch"] == patience
                break
    
    def test_edge_case_immediate_convergence(self):
        """Test edge case where convergence happens immediately."""
        tracker = ConvergenceTracker(patience=0, min_delta=0.1)  # Zero patience
        
        tracker.update(epoch=0, metrics={"KLDivergence": 1.0})
        result = tracker.update(epoch=1, metrics={"KLDivergence": 1.0})
        
        # Should converge immediately
        assert result["converged"] is True
        assert result["convergence_epoch"] == 1
    
    def test_convergence_persistence(self):
        """Test that once converged, state persists."""
        tracker = ConvergenceTracker(patience=1)
        
        # Trigger convergence
        tracker.update(epoch=0, metrics={"KLDivergence": 1.0})
        tracker.update(epoch=1, metrics={"KLDivergence": 1.0})
        
        assert tracker.converged is True
        convergence_epoch = tracker.convergence_epoch
        
        # Continue updating
        result = tracker.update(epoch=2, metrics={"KLDivergence": 0.5})  # Even with improvement
        
        # Should remain converged
        assert result["converged"] is True
        assert result["convergence_epoch"] == convergence_epoch


if __name__ == "__main__":
    pytest.main([__file__, "-v"])