"""
ConvergenceTracker: A comprehensive system for analyzing convergence speed in GaussGAN training.

This module provides tools to monitor training progress, detect when generators reach 
quality thresholds, and compute various convergence metrics for comparing different
generator types (classical vs quantum).
"""

import json
import time
import warnings
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import r2_score


class ConvergenceTracker:
    """
    Tracks convergence speed and quality metrics during GAN training.
    
    This class monitors multiple quality metrics over training epochs, detects when 
    generators reach predefined quality thresholds, and provides comprehensive 
    convergence analysis including epochs to convergence, improvement rates, and 
    plateau detection.
    
    Attributes:
        metrics_history: Dictionary storing historical metric values
        convergence_thresholds: Quality thresholds for each metric
        convergence_results: Results of convergence analysis
        plateau_patience: Number of epochs to wait before declaring plateau
        smoothing_window: Window size for smoothing metrics
        min_epochs: Minimum epochs before checking convergence
    """
    
    def __init__(
        self,
        convergence_thresholds: Optional[Dict[str, float]] = None,
        plateau_patience: int = 20,
        smoothing_window: int = 5,
        min_epochs: int = 10,
        save_dir: Optional[str] = None
    ):
        """
        Initialize the ConvergenceTracker.
        
        Args:
            convergence_thresholds: Dict mapping metric names to target values
            plateau_patience: Epochs to wait before declaring plateau
            smoothing_window: Window size for metric smoothing  
            min_epochs: Minimum epochs before convergence checking
            save_dir: Directory to save convergence results
        """
        
        # Default convergence thresholds based on GaussGAN metrics
        self.convergence_thresholds = convergence_thresholds or {
            'KLDivergence': 0.1,          # Lower is better
            'LogLikelihood': -2.0,        # Higher is better (less negative)
            'IsPositive': 0.8,            # Higher is better (proportion positive)
        }
        
        self.plateau_patience = plateau_patience
        self.smoothing_window = smoothing_window
        self.min_epochs = min_epochs
        self.save_dir = Path(save_dir) if save_dir else Path("docs/convergence_analysis")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking data structures
        self.metrics_history: Dict[str, List[float]] = defaultdict(list)
        self.epoch_times: List[float] = []
        self.convergence_results: Dict[str, Dict[str, Any]] = {}
        self.plateau_detected: Dict[str, bool] = defaultdict(bool)
        self.convergence_epochs: Dict[str, Optional[int]] = {}
        
        # Internal tracking
        self._start_time = time.time()
        self._epoch_start_time = None
        self._last_improvement: Dict[str, int] = defaultdict(int)
        self._best_values: Dict[str, float] = {}
        
        # For trend analysis
        self._metric_buffers: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.smoothing_window)
        )
        
    def start_epoch(self) -> None:
        """Mark the start of a new training epoch."""
        self._epoch_start_time = time.time()
        
    def end_epoch(self) -> None:
        """Mark the end of a training epoch and record timing."""
        if self._epoch_start_time is not None:
            epoch_duration = time.time() - self._epoch_start_time
            self.epoch_times.append(epoch_duration)
            self._epoch_start_time = None
    
    def update_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Update metrics history and check for convergence.
        
        Args:
            epoch: Current training epoch
            metrics: Dictionary of metric names to values
        """
        # Store raw metrics
        for metric_name, value in metrics.items():
            if not (np.isnan(value) or np.isinf(value)):
                self.metrics_history[metric_name].append(value)
                self._metric_buffers[metric_name].append(value)
                
                # Track best values
                if metric_name not in self._best_values:
                    self._best_values[metric_name] = value
                    self._last_improvement[metric_name] = epoch
                else:
                    # Check if this is an improvement
                    is_improvement = self._is_improvement(metric_name, value)
                    if is_improvement:
                        self._best_values[metric_name] = value
                        self._last_improvement[metric_name] = epoch
        
        # Check convergence only after minimum epochs
        if epoch >= self.min_epochs:
            self._check_convergence(epoch)
            self._check_plateau(epoch)
    
    def _is_improvement(self, metric_name: str, value: float) -> bool:
        """Check if a metric value represents an improvement."""
        if metric_name not in self._best_values:
            return True
            
        best_value = self._best_values[metric_name]
        
        # For metrics where lower is better
        if metric_name in ['KLDivergence']:
            return value < best_value * 0.99  # 1% improvement threshold
        # For metrics where higher is better
        else:
            return value > best_value * 1.01  # 1% improvement threshold
    
    def _check_convergence(self, epoch: int) -> None:
        """Check if any metrics have reached convergence thresholds."""
        for metric_name, threshold in self.convergence_thresholds.items():
            if (metric_name in self.metrics_history and 
                metric_name not in self.convergence_epochs):
                
                values = self.metrics_history[metric_name]
                if len(values) < self.smoothing_window:
                    continue
                    
                # Use smoothed values for convergence detection
                smoothed_values = self._smooth_metric(values)
                current_value = smoothed_values[-1]
                
                # Check threshold based on metric type
                converged = False
                if metric_name in ['KLDivergence']:  # Lower is better
                    converged = current_value <= threshold
                else:  # Higher is better
                    converged = current_value >= threshold
                    
                if converged:
                    self.convergence_epochs[metric_name] = epoch
                    print(f"Convergence detected for {metric_name} at epoch {epoch}")
    
    def _check_plateau(self, epoch: int) -> None:
        """Check for plateau in metric improvement."""
        for metric_name in self.metrics_history:
            if metric_name in self.plateau_detected and self.plateau_detected[metric_name]:
                continue
                
            epochs_since_improvement = epoch - self._last_improvement.get(metric_name, 0)
            if epochs_since_improvement >= self.plateau_patience:
                self.plateau_detected[metric_name] = True
                print(f"Plateau detected for {metric_name} at epoch {epoch}")
    
    def _smooth_metric(self, values: List[float]) -> np.ndarray:
        """Apply smoothing to metric values using moving average."""
        if len(values) < self.smoothing_window:
            return np.array(values)
            
        smoothed = []
        for i in range(len(values)):
            start_idx = max(0, i - self.smoothing_window + 1)
            window_values = values[start_idx:i+1]
            smoothed.append(np.mean(window_values))
            
        return np.array(smoothed)
    
    def get_convergence_speed(self, metric_name: str) -> Dict[str, Any]:
        """
        Calculate convergence speed metrics for a specific metric.
        
        Args:
            metric_name: Name of the metric to analyze
            
        Returns:
            Dictionary containing convergence speed metrics
        """
        if metric_name not in self.metrics_history:
            return {"error": f"Metric {metric_name} not found in history"}
        
        values = np.array(self.metrics_history[metric_name])
        if len(values) < 2:
            return {"error": "Insufficient data for analysis"}
        
        epochs = np.arange(len(values))
        smoothed_values = self._smooth_metric(values.tolist())
        
        # Calculate improvement rate (slope)
        if len(epochs) >= 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(epochs, smoothed_values)
        else:
            slope = 0
            r_value = 0
        
        # Calculate time to convergence
        convergence_epoch = self.convergence_epochs.get(metric_name)
        
        # Calculate improvement percentage
        if len(smoothed_values) >= 2:
            initial_value = smoothed_values[0]
            final_value = smoothed_values[-1]
            if initial_value != 0:
                improvement_pct = ((final_value - initial_value) / abs(initial_value)) * 100
            else:
                improvement_pct = 0
        else:
            improvement_pct = 0
        
        # Detect convergence pattern
        convergence_pattern = self._analyze_convergence_pattern(smoothed_values)
        
        return {
            "epochs_to_convergence": convergence_epoch,
            "improvement_rate": slope,
            "improvement_percentage": improvement_pct,
            "correlation_coefficient": r_value,
            "converged": convergence_epoch is not None,
            "plateau_detected": self.plateau_detected.get(metric_name, False),
            "convergence_pattern": convergence_pattern,
            "final_value": smoothed_values[-1] if len(smoothed_values) > 0 else None,
            "best_value": self._best_values.get(metric_name),
            "total_epochs": len(values)
        }
    
    def _analyze_convergence_pattern(self, values: np.ndarray) -> str:
        """Analyze the pattern of convergence (fast, slow, oscillating, etc.)."""
        if len(values) < 10:
            return "insufficient_data"
        
        # Split into early and late phases
        mid_point = len(values) // 2
        early_phase = values[:mid_point]
        late_phase = values[mid_point:]
        
        # Calculate variance in each phase
        early_var = np.var(early_phase)
        late_var = np.var(late_phase)
        
        # Calculate overall trend
        slope = (values[-1] - values[0]) / len(values)
        
        # Classify pattern
        if late_var > early_var * 2:
            return "oscillating"
        elif late_var < early_var * 0.1 and abs(slope) < np.std(values) * 0.1:
            return "fast_convergence"
        elif abs(slope) > np.std(values) * 0.5:
            return "steady_improvement"
        else:
            return "slow_convergence"
    
    def get_comparative_analysis(self) -> Dict[str, Any]:
        """
        Generate comparative analysis across all tracked metrics.
        
        Returns:
            Dictionary containing comparative convergence analysis
        """
        analysis = {
            "summary": {},
            "metrics": {},
            "rankings": {},
            "training_efficiency": {}
        }
        
        # Analyze each metric
        for metric_name in self.metrics_history:
            analysis["metrics"][metric_name] = self.get_convergence_speed(metric_name)
        
        # Create rankings
        converged_metrics = [m for m in analysis["metrics"] 
                           if analysis["metrics"][m].get("converged", False)]
        
        # Rank by epochs to convergence (lower is better)
        convergence_ranking = sorted(
            converged_metrics,
            key=lambda m: analysis["metrics"][m]["epochs_to_convergence"] or float('inf')
        )
        
        # Rank by improvement rate (higher absolute value is better for most metrics)
        improvement_ranking = sorted(
            analysis["metrics"].keys(),
            key=lambda m: abs(analysis["metrics"][m]["improvement_rate"]),
            reverse=True
        )
        
        analysis["rankings"] = {
            "fastest_convergence": convergence_ranking,
            "highest_improvement_rate": improvement_ranking,
        }
        
        # Training efficiency metrics
        total_training_time = sum(self.epoch_times) if self.epoch_times else 0
        analysis["training_efficiency"] = {
            "total_epochs": len(self.metrics_history[list(self.metrics_history.keys())[0]]) if self.metrics_history else 0,
            "total_training_time": total_training_time,
            "average_epoch_time": np.mean(self.epoch_times) if self.epoch_times else 0,
            "convergence_efficiency": len(converged_metrics) / len(self.metrics_history) if self.metrics_history else 0
        }
        
        # Summary statistics
        analysis["summary"] = {
            "total_metrics_tracked": len(self.metrics_history),
            "metrics_converged": len(converged_metrics),
            "metrics_plateaued": sum(self.plateau_detected.values()),
            "overall_convergence_rate": len(converged_metrics) / len(self.metrics_history) if self.metrics_history else 0
        }
        
        return analysis
    
    def save_results(self, filename: Optional[str] = None, generator_type: str = "unknown") -> Path:
        """
        Save convergence analysis results to file.
        
        Args:
            filename: Custom filename (optional)
            generator_type: Type of generator being analyzed
            
        Returns:
            Path to saved results file
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"convergence_analysis_{generator_type}_{timestamp}.json"
        
        results_path = self.save_dir / filename
        
        # Prepare results for JSON serialization
        results = {
            "generator_type": generator_type,
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "training_duration": time.time() - self._start_time,
            "convergence_thresholds": self.convergence_thresholds,
            "configuration": {
                "plateau_patience": self.plateau_patience,
                "smoothing_window": self.smoothing_window,
                "min_epochs": self.min_epochs
            },
            "metrics_history": dict(self.metrics_history),
            "epoch_times": self.epoch_times,
            "convergence_analysis": self.get_comparative_analysis()
        }
        
        # Save to JSON
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Convergence analysis saved to: {results_path}")
        return results_path
    
    def plot_convergence_curves(self, save_plots: bool = True, generator_type: str = "unknown") -> Dict[str, plt.Figure]:
        """
        Generate convergence plots for all metrics.
        
        Args:
            save_plots: Whether to save plots to file
            generator_type: Type of generator for plot titles
            
        Returns:
            Dictionary mapping metric names to matplotlib figures
        """
        figures = {}
        
        for metric_name in self.metrics_history:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle(f'{metric_name} Convergence Analysis - {generator_type}', fontsize=16)
            
            values = np.array(self.metrics_history[metric_name])
            epochs = np.arange(len(values))
            smoothed_values = self._smooth_metric(values.tolist())
            
            # Plot 1: Raw and smoothed values
            ax1.plot(epochs, values, alpha=0.3, color='blue', label='Raw values')
            ax1.plot(epochs, smoothed_values, color='blue', linewidth=2, label='Smoothed values')
            
            # Add convergence threshold line
            if metric_name in self.convergence_thresholds:
                threshold = self.convergence_thresholds[metric_name]
                ax1.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold: {threshold}')
                
            # Mark convergence point
            if metric_name in self.convergence_epochs and self.convergence_epochs[metric_name] is not None:
                conv_epoch = self.convergence_epochs[metric_name]
                conv_value = smoothed_values[conv_epoch] if conv_epoch < len(smoothed_values) else smoothed_values[-1]
                ax1.plot(conv_epoch, conv_value, 'ro', markersize=10, label=f'Convergence: Epoch {conv_epoch}')
            
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel(metric_name)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Improvement rate over time
            if len(smoothed_values) > 5:
                # Calculate rolling improvement rate
                window_size = min(10, len(smoothed_values) // 3)
                improvement_rates = []
                for i in range(window_size, len(smoothed_values)):
                    recent_values = smoothed_values[i-window_size:i]
                    slope, _, _, _, _ = stats.linregress(range(window_size), recent_values)
                    improvement_rates.append(slope)
                
                rate_epochs = epochs[window_size:]
                ax2.plot(rate_epochs, improvement_rates, color='green', linewidth=2)
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Improvement Rate')
                ax2.set_title('Rolling Improvement Rate')
                ax2.grid(True, alpha=0.3)
            
            # Add convergence statistics as text
            stats_text = self._get_metric_stats_text(metric_name)
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                    verticalalignment='top', fontfamily='monospace', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            figures[metric_name] = fig
            
            if save_plots:
                plot_path = self.save_dir / f"convergence_{metric_name}_{generator_type}.png"
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Convergence plot saved: {plot_path}")
        
        return figures
    
    def _get_metric_stats_text(self, metric_name: str) -> str:
        """Generate statistics text for metric plots."""
        stats = self.get_convergence_speed(metric_name)
        
        lines = [
            f"Metric: {metric_name}",
            f"Converged: {'Yes' if stats.get('converged', False) else 'No'}",
            f"Epochs to convergence: {stats.get('epochs_to_convergence', 'N/A')}",
            f"Improvement rate: {stats.get('improvement_rate', 0):.4f}",
            f"Final value: {stats.get('final_value', 0):.4f}",
            f"Pattern: {stats.get('convergence_pattern', 'unknown')}"
        ]
        
        return '\n'.join(lines)
    
    def compare_generators(self, other_tracker: 'ConvergenceTracker', 
                          self_name: str = "Generator A", 
                          other_name: str = "Generator B") -> Dict[str, Any]:
        """
        Compare convergence performance between two generators.
        
        Args:
            other_tracker: Another ConvergenceTracker instance
            self_name: Name for this generator
            other_name: Name for the other generator
            
        Returns:
            Comparison results dictionary
        """
        comparison = {
            "generators": {
                self_name: self.get_comparative_analysis(),
                other_name: other_tracker.get_comparative_analysis()
            },
            "head_to_head": {},
            "summary": {}
        }
        
        # Compare common metrics
        common_metrics = set(self.metrics_history.keys()) & set(other_tracker.metrics_history.keys())
        
        for metric in common_metrics:
            self_stats = self.get_convergence_speed(metric)
            other_stats = other_tracker.get_convergence_speed(metric)
            
            comparison["head_to_head"][metric] = {
                self_name: self_stats,
                other_name: other_stats,
                "winner": self._determine_winner(self_stats, other_stats, metric)
            }
        
        # Overall summary
        self_wins = sum(1 for metric in common_metrics 
                       if comparison["head_to_head"][metric]["winner"] == self_name)
        other_wins = sum(1 for metric in common_metrics 
                        if comparison["head_to_head"][metric]["winner"] == other_name)
        
        comparison["summary"] = {
            "total_metrics_compared": len(common_metrics),
            "wins": {self_name: self_wins, other_name: other_wins},
            "overall_winner": self_name if self_wins > other_wins else other_name if other_wins > self_wins else "tie"
        }
        
        return comparison
    
    def _determine_winner(self, stats1: Dict, stats2: Dict, metric_name: str) -> str:
        """Determine which generator performs better for a specific metric."""
        # First check convergence
        conv1 = stats1.get("converged", False)
        conv2 = stats2.get("converged", False)
        
        if conv1 and not conv2:
            return "Generator A"
        elif conv2 and not conv1:
            return "Generator B"
        elif conv1 and conv2:
            # Both converged, compare speed
            epochs1 = stats1.get("epochs_to_convergence", float('inf'))
            epochs2 = stats2.get("epochs_to_convergence", float('inf'))
            return "Generator A" if epochs1 < epochs2 else "Generator B"
        else:
            # Neither converged, compare final values
            final1 = stats1.get("final_value", 0)
            final2 = stats2.get("final_value", 0)
            
            # For KL divergence, lower is better
            if metric_name in ['KLDivergence']:
                return "Generator A" if final1 < final2 else "Generator B"
            else:
                return "Generator A" if final1 > final2 else "Generator B"


def create_convergence_tracker_for_config(config: Dict[str, Any]) -> ConvergenceTracker:
    """
    Create a ConvergenceTracker instance based on training configuration.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        Configured ConvergenceTracker instance
    """
    # Extract relevant parameters from config
    max_epochs = config.get('max_epochs', 50)
    metrics = config.get('metrics', ['IsPositive', 'LogLikelihood', 'KLDivergence'])
    
    # Set appropriate thresholds based on metrics used
    thresholds = {}
    if 'KLDivergence' in metrics:
        thresholds['ValidationStep_FakeData_KLDivergence'] = 0.1
    if 'LogLikelihood' in metrics:
        thresholds['ValidationStep_FakeData_LogLikelihood'] = -2.0
    if 'IsPositive' in metrics:
        thresholds['ValidationStep_FakeData_IsPositive'] = 0.8
    
    # Create tracker with adaptive parameters
    plateau_patience = max(20, max_epochs // 3)  # Adaptive to training length
    smoothing_window = max(5, max_epochs // 10)  # Adaptive smoothing
    
    return ConvergenceTracker(
        convergence_thresholds=thresholds,
        plateau_patience=plateau_patience,
        smoothing_window=smoothing_window,
        min_epochs=max(10, max_epochs // 5)
    )