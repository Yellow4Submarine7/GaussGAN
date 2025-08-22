"""
Specific code modifications for integrating enhanced metrics into GaussGAN.

This file contains the exact code changes needed for each integration point.
All modifications are designed to be minimal and backward-compatible.
"""

# =============================================================================
# MODIFICATION 1: Enhanced _compute_metrics method in source/model.py
# =============================================================================

MODIFICATION_1_MODEL_PY = """
# Replace the existing _compute_metrics method in source/model.py (lines 310-340)
# with this enhanced version:

def _compute_metrics(self, batch):
    \"\"\"
    Enhanced metrics computation with better error handling and parameter management.
    \"\"\"
    metrics = {}
    
    # Early validation of input data
    if torch.is_tensor(batch):
        batch_np = batch.cpu().numpy()
    else:
        batch_np = np.array(batch)
    
    # Filter valid points once for all metrics
    valid_mask = ~np.isnan(batch_np).any(axis=1)
    valid_batch = batch_np[valid_mask]
    
    if len(valid_batch) == 0:
        warnings.warn("No valid points for metrics computation")
        return {metric: float('nan') for metric in self.metrics}
    
    for metric_name in self.metrics:
        try:
            metric_class = ALL_METRICS[metric_name]
            
            # Enhanced parameter selection based on metric type
            if metric_name in ["LogLikelihood", "KLDivergence"]:
                # GMM-based metrics (existing implementation)
                metric_instance = metric_class(
                    centroids=self.gaussians["centroids"],
                    cov_matrices=self.gaussians["covariances"],
                    weights=self.gaussians["weights"],
                )
                
            elif metric_name == "MMDivergenceFromGMM":
                # New MMD metric using GMM samples
                metric_instance = metric_class(
                    centroids=self.gaussians["centroids"],
                    cov_matrices=self.gaussians["covariances"],
                    weights=self.gaussians["weights"],
                    n_target_samples=getattr(self, 'mmd_target_samples', 1000),
                    bandwidths=getattr(self, 'mmd_bandwidths', None)
                )
                
            elif metric_name == "WassersteinDistance":
                # Enhanced Wasserstein with configuration support
                if self.target_data is not None:
                    metric_instance = metric_class(
                        target_samples=self.target_data,
                        aggregation=getattr(self, 'wasserstein_aggregation', 'mean')
                    )
                else:
                    warnings.warn(f"Target data not provided for {metric_name}")
                    metrics[metric_name] = float('nan')
                    continue
                    
            elif metric_name == "MMDDistance":
                # Enhanced MMD with configuration support
                if self.target_data is not None:
                    metric_instance = metric_class(
                        target_samples=self.target_data,
                        kernel=getattr(self, 'mmd_kernel', 'rbf'),
                        gamma=getattr(self, 'mmd_gamma', 1.0)
                    )
                else:
                    warnings.warn(f"Target data not provided for {metric_name}")
                    metrics[metric_name] = float('nan')
                    continue
                    
            else:
                # Simple metrics (IsPositive, etc.)
                metric_instance = metric_class()
            
            # Compute metric with enhanced error handling
            score = metric_instance.compute_score(valid_batch)
            
            # Robust handling of different return types
            if isinstance(score, (list, np.ndarray)):
                valid_scores = [s for s in score if not (np.isnan(s) or np.isinf(s))]
                metrics[metric_name] = float(np.mean(valid_scores)) if valid_scores else float('nan')
            elif np.isnan(score) or np.isinf(score):
                metrics[metric_name] = float('nan')
            else:
                metrics[metric_name] = float(score)
                
        except Exception as e:
            warnings.warn(f"Error computing {metric_name}: {e}")
            metrics[metric_name] = float('nan')
    
    return metrics
"""

# =============================================================================
# MODIFICATION 2: Enhanced validation_step in source/model.py  
# =============================================================================

MODIFICATION_2_MODEL_PY = """
# Add these helper methods to the GaussGan class in source/model.py:

def _compute_validation_metadata(self, fake_data):
    \"\"\"Compute additional validation metadata.\"\"\"
    if torch.is_tensor(fake_data):
        data_np = fake_data.cpu().numpy()
    else:
        data_np = np.array(fake_data)
    
    valid_mask = ~np.isnan(data_np).any(axis=1)
    
    return {
        "validation_samples_count": len(data_np),
        "valid_samples_ratio": float(np.sum(valid_mask)) / len(data_np) if len(data_np) > 0 else 0.0,
        "mean_x": float(np.mean(data_np[valid_mask, 0])) if np.any(valid_mask) else float('nan'),
        "mean_y": float(np.mean(data_np[valid_mask, 1])) if np.any(valid_mask) else float('nan'),
        "std_x": float(np.std(data_np[valid_mask, 0])) if np.any(valid_mask) else float('nan'),
        "std_y": float(np.std(data_np[valid_mask, 1])) if np.any(valid_mask) else float('nan'),
    }

def _log_enhanced_csv_artifacts(self, fake_data, metrics_dict, convergence_info):
    \"\"\"Log enhanced CSV artifacts with metrics metadata.\"\"\"
    try:
        # Standard CSV format (maintain compatibility)
        csv_string = "x,y\\n" + "\\n".join([f"{row[0]},{row[1]}" for row in fake_data])
        
        # Enhanced CSV with metadata header
        if getattr(self, 'log_enhanced_csv', True):
            metadata_lines = [
                f"# Epoch: {self.current_epoch}",
                f"# Generator: {getattr(self, 'generator_type', 'unknown')}",
                f"# Samples: {len(fake_data)}",
                f"# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            ]
            
            # Add metric values as comments
            for metric_name, value in metrics_dict.items():
                if not np.isnan(value):
                    metadata_lines.append(f"# {metric_name}: {value:.6f}")
            
            # Add convergence status
            if convergence_info.get('converged'):
                metadata_lines.append(f"# Converged: True (epoch {convergence_info.get('convergence_epoch')})")
            elif convergence_info.get('epochs_without_improvement') is not None:
                metadata_lines.append(f"# EpochsWithoutImprovement: {convergence_info['epochs_without_improvement']}")
            
            enhanced_csv = "\\n".join(metadata_lines) + "\\n" + csv_string
            
            # Log enhanced version
            self.logger.experiment.log_text(
                text=enhanced_csv,
                artifact_file=f"gaussian_enhanced_epoch_{self.current_epoch:04d}.csv",
                run_id=self.logger.run_id,
            )
        
        # Always log standard version for compatibility
        self.logger.experiment.log_text(
            text=csv_string,
            artifact_file=f"gaussian_generated_epoch_{self.current_epoch:04d}.csv",
            run_id=self.logger.run_id,
        )
        
    except (AttributeError, Exception) as e:
        print(f"Could not log CSV artifacts: {e}")

# Replace the existing validation_step method (lines 118-194) with this enhanced version:

def validation_step(self, batch, batch_idx):
    fake_data = self._generate_fake_data(self.validation_samples).detach()

    # Compute metrics using enhanced method
    metrics_fake = self._compute_metrics(fake_data)
    
    # Compute validation metadata
    validation_metadata = self._compute_validation_metadata(fake_data)

    # Process metrics for logging (existing logic, enhanced)
    avg_metrics_fake = {}
    processed_metrics = {}
    for k, v in metrics_fake.items():
        log_key = f"ValidationStep_FakeData_{k}"
        if v is None or np.isnan(v):
            warnings.warn(f"Metric {k} returned invalid value in validation_step")
            avg_metrics_fake[log_key] = float("nan")
            processed_metrics[k] = float("nan")
        else:
            avg_metrics_fake[log_key] = float(v)
            processed_metrics[k] = float(v)

    # Update convergence tracker (existing logic)
    d_loss = getattr(self, '_last_d_loss', None)
    g_loss = getattr(self, '_last_g_loss', None)
    
    convergence_info = self.convergence_tracker.update(
        epoch=self.current_epoch,
        metrics=processed_metrics,
        d_loss=d_loss,
        g_loss=g_loss
    )
    
    # Prepare enhanced logging dictionary
    convergence_log = {}
    for key, value in convergence_info.items():
        if value is not None and not (np.isnan(value) if isinstance(value, (int, float)) else False):
            convergence_log[f"convergence_{key}"] = value
    
    # Combine all metrics for logging
    all_log_metrics = {
        **avg_metrics_fake, 
        **convergence_log,
        **validation_metadata,
        "epoch": self.current_epoch
    }
    
    # Log to MLflow
    self.log_dict(
        all_log_metrics,
        on_epoch=True,
        on_step=False,
        prog_bar=True,
        logger=True,
        batch_size=batch[0].size(0),
        sync_dist=True,
    )

    # Enhanced CSV logging
    self._log_enhanced_csv_artifacts(fake_data, metrics_fake, convergence_info)

    return {
        "fake_data": fake_data, 
        "metrics": avg_metrics_fake,
        "convergence_info": convergence_info,
        "validation_metadata": validation_metadata,
        "raw_metrics": metrics_fake
    }
"""

# =============================================================================
# MODIFICATION 3: Enhanced __init__ method in source/model.py
# =============================================================================

MODIFICATION_3_MODEL_PY = """
# Add these lines to the GaussGan.__init__ method after line 55:

# Enhanced metric configuration parameters
self.wasserstein_aggregation = kwargs.get("wasserstein_aggregation", "mean")
self.mmd_kernel = kwargs.get("mmd_kernel", "rbf") 
self.mmd_gamma = kwargs.get("mmd_gamma", 1.0)
self.mmd_target_samples = kwargs.get("mmd_target_samples", 1000)
self.mmd_bandwidths = kwargs.get("mmd_bandwidths", None)
self.log_enhanced_csv = kwargs.get("log_enhanced_csv", True)

# Add stability tracking if enabled
self.stability_tracking = kwargs.get("stability_tracking", False)
if self.stability_tracking:
    from .metrics import StabilityTracker  # Import when needed
    self.stability_tracker = StabilityTracker(
        window_size=kwargs.get("stability_window", 10),
        stability_threshold=kwargs.get("stability_threshold", 0.1)
    )
else:
    self.stability_tracker = None
"""

# =============================================================================
# MODIFICATION 4: Add StabilityTracker to source/metrics.py
# =============================================================================

MODIFICATION_4_METRICS_PY = """
# Add this class to the end of source/metrics.py, before the ALL_METRICS dictionary:

class StabilityTracker:
    \"\"\"
    Track stability metrics across training epochs for convergence analysis.
    
    Measures stability as coefficient of variation over a sliding window,
    helping detect when metrics have stabilized during training.
    \"\"\"
    
    def __init__(self, window_size=10, stability_threshold=0.1):
        \"\"\"
        Initialize stability tracker.
        
        Args:
            window_size: Number of recent values to consider for stability
            stability_threshold: Coefficient of variation threshold for stability
        \"\"\"
        self.window_size = window_size
        self.stability_threshold = stability_threshold
        self.metric_windows = {}
        self.stability_scores = {}
        
    def update(self, metrics_dict):
        \"\"\"
        Update stability tracking with new metrics.
        
        Args:
            metrics_dict: Dictionary of metric name -> value pairs
            
        Returns:
            Dictionary of stability information
        \"\"\"
        stability_info = {}
        
        for metric_name, value in metrics_dict.items():
            if not (np.isnan(value) or np.isinf(value)):
                # Initialize window if needed
                if metric_name not in self.metric_windows:
                    self.metric_windows[metric_name] = []
                
                # Add to window
                self.metric_windows[metric_name].append(float(value))
                
                # Maintain window size
                if len(self.metric_windows[metric_name]) > self.window_size:
                    self.metric_windows[metric_name].pop(0)
                
                # Compute stability if window is full
                if len(self.metric_windows[metric_name]) >= self.window_size:
                    values = np.array(self.metric_windows[metric_name])
                    stability_score = self._compute_stability_score(values)
                    self.stability_scores[metric_name] = stability_score
                    
                    stability_info[f"{metric_name}_stability"] = stability_score
                    stability_info[f"{metric_name}_is_stable"] = stability_score < self.stability_threshold
                    stability_info[f"{metric_name}_trend"] = self._compute_trend(values)
        
        return stability_info
    
    def _compute_stability_score(self, values):
        \"\"\"Compute stability score as coefficient of variation.\"\"\"
        if len(values) < 2:
            return float('inf')
        
        mean_val = np.mean(values)
        if abs(mean_val) < 1e-10:  # Near zero mean
            return float('inf')
        
        return float(np.std(values) / abs(mean_val))
    
    def _compute_trend(self, values):
        \"\"\"Compute trend direction (-1: decreasing, 0: stable, 1: increasing).\"\"\"
        if len(values) < 3:
            return 0.0
        
        # Simple linear trend using first and last values
        trend = (values[-1] - values[0]) / len(values)
        return float(np.sign(trend))
    
    def get_stability_summary(self):
        \"\"\"Get summary of current stability status.\"\"\"
        summary = {
            'tracked_metrics': list(self.stability_scores.keys()),
            'stable_metrics': [
                metric for metric, score in self.stability_scores.items()
                if score < self.stability_threshold
            ],
            'overall_stability': (
                len([s for s in self.stability_scores.values() if s < self.stability_threshold]) /
                len(self.stability_scores) if self.stability_scores else 0.0
            )
        }
        return summary
    
    def reset(self):
        \"\"\"Reset stability tracking.\"\"\"
        self.metric_windows.clear()
        self.stability_scores.clear()
"""

# =============================================================================
# MODIFICATION 5: Update ALL_METRICS dictionary in source/metrics.py
# =============================================================================

MODIFICATION_5_METRICS_PY = """
# Update the ALL_METRICS dictionary at the end of source/metrics.py:

ALL_METRICS = {
    "IsPositive": IsPositive,
    "LogLikelihood": LogLikelihood,
    "KLDivergence": KLDivergence,
    "WassersteinDistance": WassersteinDistance,
    "MMDDistance": MMDDistance,
    "MMDivergence": MMDivergence,
    "MMDivergenceFromGMM": MMDivergenceFromGMM,
}
"""

# =============================================================================
# MODIFICATION 6: Enhanced configuration in config.yaml
# =============================================================================

MODIFICATION_6_CONFIG_YAML = """
# Add these lines to config.yaml after the existing metrics section:

# Enhanced metrics configuration
wasserstein_aggregation: "mean"  # Options: mean, max, sum  
mmd_kernel: "rbf"               # Kernel type for MMD distance
mmd_gamma: 1.0                  # RBF kernel bandwidth parameter
mmd_target_samples: 1000        # Number of target samples for MMD from GMM
mmd_bandwidths: null            # Custom bandwidths (null for adaptive)

# Stability analysis parameters
stability_tracking: true        # Enable stability analysis
stability_window: 10            # Number of epochs for stability window  
stability_threshold: 0.1        # Coefficient of variation threshold

# Enhanced logging parameters
log_enhanced_csv: true          # Enable metadata in CSV artifacts
"""

# =============================================================================
# MODIFICATION 7: Enhanced main.py parameter passing
# =============================================================================

MODIFICATION_7_MAIN_PY = """
# Add these parameters to the model initialization in main.py (around line 147):

# Enhanced metric parameters
wasserstein_aggregation=final_args.get("wasserstein_aggregation", "mean"),
mmd_kernel=final_args.get("mmd_kernel", "rbf"),
mmd_gamma=final_args.get("mmd_gamma", 1.0),
mmd_target_samples=final_args.get("mmd_target_samples", 1000),
mmd_bandwidths=final_args.get("mmd_bandwidths", None),
log_enhanced_csv=final_args.get("log_enhanced_csv", True),

# Stability tracking parameters  
stability_tracking=final_args.get("stability_tracking", False),
stability_window=final_args.get("stability_window", 10),
stability_threshold=final_args.get("stability_threshold", 0.1),
"""

# =============================================================================
# IMPLEMENTATION SUMMARY
# =============================================================================

IMPLEMENTATION_SUMMARY = """
Summary of Required Changes:
===========================

1. source/model.py:
   - Replace _compute_metrics method with enhanced version
   - Replace validation_step method with enhanced version  
   - Add helper methods: _compute_validation_metadata, _log_enhanced_csv_artifacts
   - Add parameters to __init__ method for enhanced configuration

2. source/metrics.py:
   - Add StabilityTracker class before ALL_METRICS dictionary
   - Ensure ALL_METRICS includes all metric types

3. config.yaml:
   - Add enhanced metric configuration parameters
   - Add stability tracking parameters
   - Add enhanced logging parameters

4. main.py:
   - Pass enhanced parameters to model initialization

Expected Benefits:
=================
- Robust error handling for all metric computations
- Enhanced CSV artifacts with training metadata
- Stability analysis for convergence detection
- Improved MLflow logging with additional metadata
- Backward compatibility with existing training pipelines
- Configurable metric parameters for experimentation

Testing Strategy:
================
1. Run basic training with classical generator to verify compatibility
2. Test enhanced CSV logging contains proper metadata
3. Verify MLflow logs include stability metrics
4. Check error handling with edge cases (NaN values, empty batches)
5. Validate convergence detection works with stability analysis
"""

if __name__ == "__main__":
    print("GaussGAN Enhanced Metrics Integration")
    print("====================================")
    print(IMPLEMENTATION_SUMMARY)