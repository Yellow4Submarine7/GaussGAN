# Layer 2 Metrics Integration Guide - GaussGAN

## Summary

After analyzing the GaussGAN metrics system, **all required layer 2 statistical measures are already implemented**:

- ✅ **Maximum Mean Discrepancy (MMD)**: 3 implementations available
- ✅ **Wasserstein Distance**: Complete with multi-dimensional support  
- ✅ **Convergence Tracking**: Comprehensive system in `/source/convergence.py`
- ✅ **Generation Stability**: Full analyzer in `/docs/stability_analyzer.py`

This guide focuses on **integration enhancements** to leverage the existing robust foundation.

## Quick Start: Using Existing Metrics

### 1. Basic Configuration

```yaml
# config.yaml
metrics: ['IsPositive', 'LogLikelihood', 'KLDivergence', 'WassersteinDistance', 'MMDDistance']

# Convergence tracking  
convergence_patience: 10
convergence_min_delta: 1e-4
convergence_monitor: "KLDivergence"
convergence_window: 5
```

### 2. Advanced Configuration with Enhanced Features

```yaml
# Enhanced metric configuration
metrics:
  - name: "KLDivergence"
    weight: 1.0
    monitor_convergence: true
  - name: "MMDDistance"
    weight: 0.8
    kernel: "rbf" 
    gamma: 1.0
  - name: "WassersteinDistance"
    weight: 0.6
    aggregation: "mean"

# Multi-scale convergence analysis
convergence:
  timescales:
    short: 10    # Local stability
    medium: 30   # Trend analysis
    long: 100    # Global convergence
  stability_threshold: 0.05
  enable_multi_scale: true
  
# Stability analysis
stability:
  enable_weighted_metrics: true
  stability_window: 20
  min_weight: 0.1
  max_weight: 1.0
```

## Enhanced Usage Patterns

### 1. Using the MetricFactory for Consistent Initialization

```python
from docs.enhanced_metrics_examples import MetricFactory, validate_generation_quality

# Consistent metric creation
factory = MetricFactory()

# GMM-based metrics
kl_metric = factory.create_metric(
    "KLDivergence",
    centroids=gmm_centroids,
    cov_matrices=gmm_covariances, 
    weights=gmm_weights
)

# Distance-based metrics
mmd_metric = factory.create_metric(
    "MMDDistance",
    target_data=target_samples,
    kernel="rbf",
    gamma=1.0
)
```

### 2. Multi-Scale Convergence Analysis

```python
from docs.enhanced_metrics_examples import MultiScaleConvergence

# Initialize multi-scale tracker
convergence_tracker = MultiScaleConvergence(
    timescales={"short": 10, "medium": 30, "long": 100},
    stability_threshold=0.05
)

# During training loop
for epoch in range(max_epochs):
    # ... training step ...
    
    # Compute validation metric
    kl_value = kl_metric.compute_score(generated_samples)
    
    # Update multi-scale analysis
    convergence_analysis = convergence_tracker.update_history(kl_value)
    
    # Check convergence status
    overall_status = convergence_tracker.get_overall_convergence_status()
    
    if overall_status["overall_status"] == "converged":
        print(f"Multi-scale convergence detected at epoch {epoch}")
        break
```

### 3. Stability-Weighted Metric Aggregation  

```python
from docs.enhanced_metrics_examples import StabilityWeightedMetric

# Create base metrics
base_metrics = {
    "KL": factory.create_metric("KLDivergence", centroids=centroids, ...),
    "MMD": factory.create_metric("MMDDistance", target_data=target_data, ...),
    "Wasserstein": factory.create_metric("WassersteinDistance", target_data=target_data, ...)
}

# Stability-weighted aggregator
weighted_metric = StabilityWeightedMetric(
    base_metrics=base_metrics,
    stability_window=20
)

# Use in training
aggregate_score = weighted_metric.compute_score(generated_samples)
current_weights = weighted_metric.get_current_weights()
stability_report = weighted_metric.get_stability_report()
```

### 4. Comprehensive Validation Pipeline

```python
from docs.enhanced_metrics_examples import validate_generation_quality

# Complete quality assessment
quality_report = validate_generation_quality(
    generated_samples=generator.sample(500),
    target_data=target_distribution_samples,
    centroids=gmm_centroids,
    cov_matrices=gmm_covariances,
    weights=gmm_weights,
    config={
        'metrics': ['KLDivergence', 'MMDDistance', 'WassersteinDistance'],
        'stability_threshold': 0.05,
        'stability_window': 20
    }
)

print("Individual Metrics:", quality_report["individual_metrics"])
print("Stability Analysis:", quality_report["stability_analysis"]) 
print("Overall Assessment:", quality_report["overall_assessment"])
print("Recommendations:", quality_report["recommendations"])
```

## Integration with Existing Training Loop

### 1. Enhanced Model Integration

```python
# In source/model.py - enhanced _compute_metrics method
def _compute_metrics_enhanced(self, batch):
    """Enhanced metric computation with stability tracking."""
    
    # Use existing metric computation
    basic_metrics = self._compute_metrics(batch)  # Original method
    
    # Add enhanced metrics if configured
    if hasattr(self, 'enhanced_metrics_enabled') and self.enhanced_metrics_enabled:
        
        # Multi-scale convergence tracking
        if hasattr(self, 'convergence_tracker'):
            for metric_name, metric_value in basic_metrics.items():
                if metric_name == self.convergence_monitor_metric:
                    convergence_analysis = self.convergence_tracker.update_history(metric_value)
                    basic_metrics[f"{metric_name}_convergence"] = convergence_analysis
        
        # Stability-weighted aggregate
        if hasattr(self, 'stability_weighted_metric'):
            try:
                aggregate_score = self.stability_weighted_metric.compute_score(batch)
                basic_metrics["StabilityWeighted"] = float(np.mean(aggregate_score))
                
                # Add stability report periodically
                if self.current_epoch % 10 == 0:
                    stability_report = self.stability_weighted_metric.get_stability_report()
                    # Log stability report to MLflow or save to file
                    
            except Exception as e:
                warnings.warn(f"Stability-weighted metric failed: {e}")
    
    return basic_metrics

def setup_enhanced_metrics(self, config):
    """Setup enhanced metrics during model initialization."""
    
    if config.get('enable_enhanced_metrics', False):
        self.enhanced_metrics_enabled = True
        
        # Multi-scale convergence
        if config.get('enable_multi_scale_convergence', False):
            self.convergence_tracker = MultiScaleConvergence(
                timescales=config.get('convergence_timescales'),
                stability_threshold=config.get('stability_threshold', 0.05)
            )
            self.convergence_monitor_metric = config.get('convergence_monitor', 'KLDivergence')
        
        # Stability-weighted metrics
        if config.get('enable_weighted_metrics', False):
            # Create base metrics for weighting
            base_metrics = {}
            factory = MetricFactory()
            
            for metric_name in config.get('weighted_base_metrics', ['KLDivergence', 'MMDDistance']):
                try:
                    if metric_name in ["LogLikelihood", "KLDivergence"]:
                        metric = factory.create_metric(
                            metric_name,
                            centroids=self.gaussians["centroids"],
                            cov_matrices=self.gaussians["covariances"],
                            weights=self.gaussians["weights"]
                        )
                    else:
                        metric = factory.create_metric(
                            metric_name,
                            target_data=self.target_data
                        )
                    base_metrics[metric_name] = metric
                except Exception as e:
                    warnings.warn(f"Failed to create base metric {metric_name}: {e}")
            
            if len(base_metrics) >= 2:
                self.stability_weighted_metric = StabilityWeightedMetric(
                    base_metrics=base_metrics,
                    stability_window=config.get('stability_window', 20)
                )
```

### 2. Configuration Integration

```python
# Enhanced configuration parsing in main.py or config loading
def load_enhanced_config(config_path):
    """Load configuration with enhanced metrics support."""
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Enhanced metrics configuration
    enhanced_config = {
        'enable_enhanced_metrics': config.get('enhanced_metrics', {}).get('enabled', False),
        'enable_multi_scale_convergence': config.get('convergence', {}).get('enable_multi_scale', False),
        'convergence_timescales': config.get('convergence', {}).get('timescales', {"short": 10, "medium": 30, "long": 100}),
        'enable_weighted_metrics': config.get('stability', {}).get('enable_weighted_metrics', False),
        'stability_window': config.get('stability', {}).get('stability_window', 20),
        'stability_threshold': config.get('stability', {}).get('threshold', 0.05)
    }
    
    config.update(enhanced_config)
    return config
```

### 3. Enhanced Logging and Visualization

```python
# Enhanced logging for layer 2 metrics
def log_enhanced_metrics(trainer, metrics_dict):
    """Enhanced logging with stability and convergence information."""
    
    # Log basic metrics (existing functionality)
    for metric_name, metric_value in metrics_dict.items():
        if not metric_name.endswith('_convergence'):
            trainer.log(f"val/{metric_name}", metric_value)
    
    # Log convergence information
    for metric_name, metric_value in metrics_dict.items():
        if metric_name.endswith('_convergence') and isinstance(metric_value, dict):
            for scale, analysis in metric_value.items():
                if isinstance(analysis, dict) and analysis.get('status') == 'active':
                    trainer.log(f"convergence/{metric_name}_{scale}_cv", analysis['cv'])
                    trainer.log(f"convergence/{metric_name}_{scale}_stable", int(analysis['is_stable']))
                    trainer.log(f"convergence/{metric_name}_{scale}_slope", analysis['trend_slope'])

# Enhanced visualization
def create_enhanced_plots(metrics_history, save_path):
    """Create enhanced visualizations including stability and convergence."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Existing metric plots
    for i, (metric_name, values) in enumerate(metrics_history.items()):
        if i < 4:  # First 4 metrics
            ax = axes[i//2, i%2]
            ax.plot(values, label=metric_name)
            ax.set_title(f"{metric_name} Over Time")
            ax.legend()
    
    # Additional stability analysis plot
    if 'StabilityWeighted' in metrics_history:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        stability_values = metrics_history['StabilityWeighted']
        
        # Plot stability-weighted metric with confidence bands
        epochs = range(len(stability_values))
        ax2.plot(epochs, stability_values, label='Stability-Weighted Score', linewidth=2)
        
        # Add rolling statistics
        window = 10
        if len(stability_values) > window:
            rolling_mean = np.convolve(stability_values, np.ones(window)/window, mode='valid')
            rolling_epochs = range(window-1, len(stability_values))
            ax2.plot(rolling_epochs, rolling_mean, '--', alpha=0.7, label=f'Rolling Mean ({window} epochs)')
        
        ax2.set_title('Stability-Weighted Metric Evolution')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}_stability.png", dpi=150, bbox_inches='tight')
        
    plt.tight_layout()
    plt.savefig(f"{save_path}_enhanced.png", dpi=150, bbox_inches='tight')
```

## Best Practices and Recommendations

### 1. Performance Considerations

```python
# Efficient metric computation
def compute_metrics_efficiently(generator, target_data, config):
    """Compute metrics efficiently with caching and batching."""
    
    # Generate samples once
    samples = generator.sample(config['validation_samples'])
    
    # Compute heavy metrics (MMD, Wasserstein) less frequently
    if config.get('heavy_metrics_frequency', 5) and epoch % config['heavy_metrics_frequency'] == 0:
        # Compute MMD and Wasserstein
        pass
    
    # Compute lightweight metrics every epoch
    # KL divergence, log likelihood
    pass
```

### 2. Error Handling and Robustness

```python
def robust_metric_computation(metric_func, samples, **kwargs):
    """Robust wrapper for metric computation with fallbacks."""
    
    try:
        return metric_func(samples, **kwargs)
    except Exception as e:
        warnings.warn(f"Metric computation failed: {e}")
        
        # Fallback strategies
        if len(samples) == 0:
            return float('inf')
        elif np.all(np.isnan(samples)):
            return float('nan')
        else:
            # Try with reduced sample size
            try:
                reduced_samples = samples[:len(samples)//2]
                return metric_func(reduced_samples, **kwargs)
            except:
                return float('nan')
```

### 3. Configuration Validation

```python
def validate_enhanced_config(config):
    """Validate enhanced metrics configuration."""
    
    errors = []
    
    # Check timescales
    if 'convergence_timescales' in config:
        timescales = config['convergence_timescales']
        if not isinstance(timescales, dict):
            errors.append("convergence_timescales must be a dictionary")
        elif any(not isinstance(v, int) or v < 1 for v in timescales.values()):
            errors.append("All timescale values must be positive integers")
    
    # Check stability parameters
    stability_threshold = config.get('stability_threshold', 0.05)
    if not 0 < stability_threshold < 1:
        errors.append("stability_threshold must be between 0 and 1")
    
    # Check weights
    if 'weighted_base_metrics' in config:
        base_metrics = config['weighted_base_metrics']
        if not isinstance(base_metrics, list) or len(base_metrics) < 2:
            errors.append("weighted_base_metrics must be a list with at least 2 metrics")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
    
    return True
```

## Migration Guide

To integrate enhanced metrics into existing GaussGAN training:

### 1. Minimal Integration (No Code Changes)

Simply update `config.yaml`:

```yaml
# Add to existing config.yaml
enhanced_metrics:
  enabled: false  # Keep existing behavior

# But enable better convergence tracking
convergence_patience: 15
convergence_min_delta: 1e-5  
convergence_monitor: "KLDivergence"
```

### 2. Partial Integration (Config Only)

```yaml
enhanced_metrics:
  enabled: true

convergence:
  enable_multi_scale: true
  timescales:
    short: 10
    medium: 30
    long: 100

stability:
  enable_weighted_metrics: true
  stability_window: 20
```

### 3. Full Integration

Add enhanced metric computation to your training loop and enable all features.

## Conclusion

The GaussGAN project has **excellent layer 2 statistical measures already implemented**. The enhancements provided focus on:

1. **Better integration patterns** (MetricFactory, configuration)
2. **Multi-scale analysis** (temporal convergence patterns) 
3. **Stability-aware aggregation** (robust metric combination)
4. **Comprehensive validation** (automated quality assessment)

These enhancements leverage the solid existing foundation while providing more sophisticated analysis capabilities for comparing quantum vs classical generators.