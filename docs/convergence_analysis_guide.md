# GaussGAN Convergence Speed Analysis System

This comprehensive system provides tools to measure and analyze convergence speed across different generator types in the GaussGAN project, enabling quantitative comparison between classical and quantum generators.

## Overview

The convergence analysis system consists of several key components:

1. **ConvergenceTracker**: Core tracking class that monitors metrics over time
2. **ConvergenceCallback**: PyTorch Lightning callback for seamless integration
3. **Training Integration**: Automated setup for existing training pipelines
4. **Experiment Runner**: Scripts for running comparative experiments
5. **Analysis Tools**: Comprehensive reporting and visualization

## Key Features

### Metric Tracking
- **Real-time monitoring** of quality metrics (KL divergence, log-likelihood, position validation)
- **Smoothed trend analysis** using configurable moving averages
- **Historical data storage** with JSON export capabilities

### Convergence Detection
- **Threshold-based convergence** detection for each metric
- **Adaptive thresholds** based on metric types and expected ranges
- **Early stopping** detection when improvements plateau

### Speed Analysis
- **Epochs to convergence** measurement for each metric
- **Improvement rate calculation** using linear regression
- **Convergence pattern classification** (fast, slow, oscillating, steady)
- **Training efficiency metrics** (time per epoch, convergence per hour)

### Comparative Analysis
- **Head-to-head comparisons** between generator types
- **Ranking systems** for speed, efficiency, and overall performance
- **Statistical significance** testing for performance differences
- **Comprehensive reporting** with visualizations

## Quick Start

### 1. Basic Integration (Minimal Changes)

Add convergence tracking to existing training with minimal code changes:

```python
from source.training_integration import setup_convergence_tracking

# In your main training script
tracker, callback = setup_convergence_tracking("config.yaml", "classical_normal")

# Add to your PyTorch Lightning trainer
trainer = pl.Trainer(callbacks=[callback, ...other_callbacks])
trainer.fit(model, datamodule)
```

### 2. Running Single Experiments

Train a single generator with convergence tracking:

```bash
# The convergence tracking is now automatically enabled in main.py
uv run python main.py --generator_type classical_normal --max_epochs 100
uv run python main.py --generator_type quantum_samples --max_epochs 100
```

### 3. Automated Comparative Experiments

Run comprehensive comparative experiments:

```bash
# Compare classical vs quantum generators
python convergence_experiment.py --generators classical_normal quantum_samples --epochs 100

# Compare multiple generator types
python convergence_experiment.py --generators classical_normal classical_uniform quantum_samples quantum_shadows --epochs 75

# Analyze existing results without running new experiments
python convergence_experiment.py --comparative-analysis
```

## Configuration

### Default Convergence Thresholds

The system uses the following default quality thresholds:

```python
{
    'KLDivergence': 0.1,          # Lower is better - good distribution match
    'LogLikelihood': -2.0,        # Higher is better - good probability fit  
    'IsPositive': 0.8,            # Higher is better - 80% positive x values
}
```

### Customizing Thresholds

```python
custom_thresholds = {
    'ValidationStep_FakeData_KLDivergence': 0.05,  # More strict
    'ValidationStep_FakeData_LogLikelihood': -1.5,  # More strict
    'ValidationStep_FakeData_IsPositive': 0.9,      # More strict
}

tracker = ConvergenceTracker(
    convergence_thresholds=custom_thresholds,
    plateau_patience=30,  # Wait 30 epochs before declaring plateau
    smoothing_window=10,  # Use 10-epoch moving average
)
```

### Configuration Parameters

- **`plateau_patience`**: Epochs to wait before declaring convergence plateau (default: 20)
- **`smoothing_window`**: Window size for metric smoothing (default: 5)
- **`min_epochs`**: Minimum epochs before checking convergence (default: 10)
- **`save_frequency`**: How often to save intermediate results (default: every 50 epochs)
- **`plot_frequency`**: How often to generate convergence plots (default: every 25 epochs)

## Output and Results

### Directory Structure

```
docs/
├── convergence_analysis/
│   ├── convergence_analysis_classical_normal_20250821_143022.json
│   ├── convergence_analysis_quantum_samples_20250821_145510.json
│   ├── convergence_KLDivergence_classical_normal.png
│   ├── convergence_LogLikelihood_quantum_samples.png
│   └── comparative_analysis_20250821_150045.json
└── convergence_experiments/
    └── generator_comparison_classical_normal_vs_quantum_samples_20250821.json
```

### Analysis JSON Structure

```json
{
  "generator_type": "classical_normal",
  "analysis_timestamp": "2025-08-21 14:30:22",
  "training_duration": 1247.5,
  "convergence_analysis": {
    "summary": {
      "total_metrics_tracked": 3,
      "metrics_converged": 2,
      "overall_convergence_rate": 0.67
    },
    "metrics": {
      "ValidationStep_FakeData_KLDivergence": {
        "epochs_to_convergence": 45,
        "improvement_rate": -0.0023,
        "converged": true,
        "convergence_pattern": "steady_improvement"
      }
    }
  }
}
```

## Understanding the Metrics

### Convergence Speed Metrics

1. **Epochs to Convergence**: Number of training epochs required to reach quality threshold
2. **Improvement Rate**: Slope of metric improvement over time (linear regression)
3. **Improvement Percentage**: Total improvement from start to end of training
4. **Convergence Pattern**: Classification of improvement behavior

### Convergence Patterns

- **`fast_convergence`**: Rapid improvement with early stabilization
- **`steady_improvement`**: Consistent linear improvement throughout training
- **`slow_convergence`**: Gradual improvement with high variance
- **`oscillating`**: High variance with unstable convergence behavior

### Efficiency Metrics

- **Training Time per Epoch**: Average time spent per training epoch
- **Convergence per Hour**: Rate of metric convergence normalized by training time
- **Overall Convergence Rate**: Proportion of tracked metrics that successfully converged

## Advanced Usage

### Manual Convergence Tracking

For custom training loops or advanced analysis:

```python
from source.convergence import ConvergenceTracker

# Create tracker
tracker = ConvergenceTracker(
    convergence_thresholds={'my_metric': 0.1},
    plateau_patience=25
)

# Training loop
for epoch in range(max_epochs):
    tracker.start_epoch()
    
    # ... your training code ...
    
    # Update with validation metrics
    val_metrics = {'my_metric': current_metric_value}
    tracker.update_metrics(epoch, val_metrics)
    
    tracker.end_epoch()

# Get results
analysis = tracker.get_comparative_analysis()
tracker.save_results(generator_type="my_generator")
```

### Custom Metric Integration

Add your own convergence metrics:

```python
# Define custom convergence detection
def custom_convergence_check(values, threshold):
    return np.mean(values[-5:]) >= threshold  # Last 5 epochs above threshold

# Integrate with tracker
tracker.add_custom_metric("MyMetric", custom_convergence_check, threshold=0.95)
```

### Comparative Analysis

Compare multiple completed experiments:

```python
from source.training_integration import ConvergenceTrainingManager

manager = ConvergenceTrainingManager()

# Compare completed experiments
comparison = manager.compare_completed_experiments([
    'classical_normal', 
    'quantum_samples', 
    'quantum_shadows'
])

# Generate human-readable report
report = manager.generate_experiment_report([
    'classical_normal', 
    'quantum_samples'
])
print(report)
```

## Visualization

The system automatically generates convergence plots showing:

1. **Raw and smoothed metric values** over training epochs
2. **Convergence thresholds** as reference lines
3. **Convergence detection points** marked on the curves
4. **Rolling improvement rates** to show training dynamics
5. **Statistical summaries** with key convergence metrics

### Plot Types

- **Individual metric plots**: Detailed analysis for each quality metric
- **Comparative plots**: Side-by-side comparison of different generators
- **Training efficiency plots**: Time-based convergence analysis
- **Pattern analysis plots**: Classification of convergence behaviors

## Troubleshooting

### Common Issues

1. **No Convergence Detected**
   - Check if thresholds are appropriate for your metrics
   - Increase `max_epochs` to allow more training time
   - Verify metric values are being logged correctly

2. **Noisy Convergence Detection**
   - Increase `smoothing_window` for more stable detection
   - Adjust `plateau_patience` to avoid premature plateau detection
   - Check for numerical instabilities in metric calculations

3. **Missing Results Files**
   - Ensure `docs/convergence_analysis/` directory has write permissions
   - Check that training completed successfully
   - Verify callback was properly added to trainer

### Performance Considerations

- **Large Smoothing Windows**: May delay convergence detection but provide more stable results
- **Frequent Saving**: More robust but may slow training slightly
- **Multiple Metrics**: Comprehensive analysis but increased computational overhead

## Examples

### Basic Comparison Experiment

```bash
# Run comparative experiment between classical and quantum generators
python convergence_experiment.py \
    --generators classical_normal quantum_samples \
    --epochs 100 \
    --config-overrides '{"batch_size": 512, "learning_rate": 0.0005}'
```

### Analysis Only

```bash
# Analyze existing results without running new experiments
python convergence_experiment.py --comparative-analysis
```

### Custom Configuration

```python
from source.convergence import ConvergenceTracker

# High-precision tracking for research
tracker = ConvergenceTracker(
    convergence_thresholds={
        'ValidationStep_FakeData_KLDivergence': 0.01,  # Very strict
        'ValidationStep_FakeData_LogLikelihood': -1.0,  # High quality
    },
    plateau_patience=50,     # Long patience for slow quantum training
    smoothing_window=15,     # Heavy smoothing for stability
    min_epochs=25           # Wait longer before checking convergence
)
```

## Best Practices

1. **Threshold Setting**: Start with default thresholds and adjust based on your specific use case
2. **Training Duration**: Allow sufficient epochs for convergence (quantum generators typically need more)
3. **Multiple Runs**: Run multiple experiments with different seeds for statistical reliability
4. **Resource Planning**: Quantum experiments take significantly longer - plan accordingly
5. **Result Verification**: Always verify convergence results by inspecting the generated plots

## Integration with Existing Code

The convergence tracking system is designed to integrate seamlessly with the existing GaussGAN codebase:

- **Minimal Changes**: Only requires adding the callback to your trainer
- **Automatic Detection**: Works with existing validation metrics without modification
- **Backward Compatibility**: Does not interfere with existing logging or checkpointing
- **Optional Usage**: Can be disabled by simply not including the callback

This system provides a comprehensive solution for quantitatively comparing the convergence speed and efficiency of different generator architectures in the GaussGAN project.