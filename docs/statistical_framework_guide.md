# GaussGAN Statistical Analysis Framework - Complete Guide

## Table of Contents

1. [Overview](#overview)
2. [Framework Components](#framework-components)
3. [Installation and Setup](#installation-and-setup)
4. [Quick Start Guide](#quick-start-guide)
5. [Detailed Usage Examples](#detailed-usage-examples)
6. [Advanced Features](#advanced-features)
7. [Interpreting Results](#interpreting-results)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [API Reference](#api-reference)

## Overview

The GaussGAN Statistical Analysis Framework provides a comprehensive toolkit for comparing quantum vs classical generators through rigorous statistical analysis. The framework implements robust statistical methods, convergence analysis, stability assessment, and automated reporting to ensure reliable and reproducible results.

### Key Features

- **Multi-Run Experiment Management**: Automated execution and tracking of multiple training runs
- **Statistical Significance Testing**: Comprehensive hypothesis testing with multiple comparison correction
- **Convergence Analysis**: Pattern classification and survival analysis for convergence behavior
- **Stability Assessment**: Multi-dimensional stability metrics and risk scoring
- **Automated Reporting**: Publication-ready reports in multiple formats (PDF, HTML, LaTeX)
- **Interactive Visualizations**: Dynamic plots and dashboards for exploratory analysis

### Scientific Rigor

The framework implements state-of-the-art statistical methods including:
- Bootstrap confidence intervals
- ANOVA-based variance decomposition
- Survival analysis with Kaplan-Meier estimators
- Ensemble outlier detection
- Effect size calculations (Cohen's d, eta-squared)
- Multiple comparison corrections (Bonferroni, FDR)

## Framework Components

### 1. Multi-Run Experiment Manager (`multi_run_experiment_runner.py`)

Orchestrates multiple training runs with systematic parameter variation:

```python
from docs.multi_run_experiment_runner import ExperimentRunner

# Initialize experiment runner
runner = ExperimentRunner(
    base_config_path="config.yaml",
    output_dir="docs/experiments",
    max_parallel=2
)

# Create experiment plan
experiment_plan = runner.create_experiment_plan(
    generator_types=['classical_normal', 'quantum_samples'],
    n_runs_per_type=20,
    seed_range=(42, 142)
)

# Execute experiments
results = runner.run_experiments(experiment_plan)
```

### 2. Statistical Analyzer (`statistical_analysis_framework.py`)

Performs comprehensive statistical comparisons:

```python
from docs.statistical_analysis_framework import StatisticalAnalyzer

analyzer = StatisticalAnalyzer(alpha=0.05, correction_method='bonferroni')

# Compare all generators for a specific metric
comparisons = analyzer.compare_all_generators(runs, 'KLDivergence')

# Extract descriptive statistics
stats = analyzer.compute_descriptive_statistics(runs, 'KLDivergence')
```

### 3. Convergence Analyzer (`convergence_analysis_tools.py`)

Analyzes convergence patterns and speed:

```python
from docs.convergence_analysis_tools import AdvancedConvergenceAnalyzer

conv_analyzer = AdvancedConvergenceAnalyzer()

# Perform comprehensive convergence analysis
convergence_results = conv_analyzer.analyze_convergence_events(
    experimental_runs=runs,
    metrics_to_analyze=['KLDivergence', 'LogLikelihood']
)

# Create convergence visualizations
conv_analyzer.create_convergence_visualizations(convergence_results)
```

### 4. Stability Analyzer (`stability_variance_analyzer.py`)

Assesses stability and variance characteristics:

```python
from docs.stability_variance_analyzer import StabilityAnalyzer

stability_analyzer = StabilityAnalyzer()

# Analyze stability across runs
stability_results = stability_analyzer.analyze_stability(
    experimental_runs=runs,
    metrics_to_analyze=['KLDivergence', 'LogLikelihood']
)

# Create stability visualizations
stability_analyzer.create_stability_visualizations(stability_results)
```

### 5. Comprehensive Reporting (`comprehensive_reporting_system.py`)

Generates publication-ready reports:

```python
from docs.comprehensive_reporting_system import ReportGenerator

report_gen = ReportGenerator(
    output_dir="docs/reports",
    report_title="GaussGAN Quantum vs Classical Analysis"
)

# Generate all report formats
generated_files = report_gen.generate_comprehensive_report(
    statistical_results=statistical_results,
    convergence_results=convergence_results,
    stability_results=stability_results
)
```

## Installation and Setup

### Prerequisites

```bash
# Python dependencies
pip install numpy pandas scipy matplotlib seaborn scikit-learn
pip install plotly jinja2  # Optional: for interactive plots and LaTeX reports
pip install lifelines      # Optional: for survival analysis

# GaussGAN project dependencies
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install lightning mlflow optuna pennylane torch-geometric
```

### Directory Structure

```
GaussGAN/
├── docs/
│   ├── statistical_analysis_framework.py
│   ├── multi_run_experiment_runner.py
│   ├── convergence_analysis_tools.py
│   ├── stability_variance_analyzer.py
│   ├── comprehensive_reporting_system.py
│   ├── statistical_analysis_demo.py
│   └── outputs/
│       ├── experiments/
│       ├── statistical_analysis/
│       ├── convergence_analysis/
│       ├── stability_analysis/
│       └── reports/
├── source/
│   ├── model.py
│   ├── metrics.py
│   └── ...
└── config.yaml
```

## Quick Start Guide

### 1. Complete Analysis Workflow

```python
# Import all components
from docs.multi_run_experiment_runner import ExperimentRunner
from docs.statistical_analysis_framework import StatisticalAnalyzer
from docs.convergence_analysis_tools import AdvancedConvergenceAnalyzer
from docs.stability_variance_analyzer import StabilityAnalyzer
from docs.comprehensive_reporting_system import ReportGenerator

# Step 1: Run experiments
runner = ExperimentRunner()
experiment_plan = runner.create_experiment_plan(
    generator_types=['classical_normal', 'classical_uniform', 
                     'quantum_samples', 'quantum_shadows'],
    n_runs_per_type=15
)
experimental_runs = runner.run_experiments(experiment_plan)

# Step 2: Statistical analysis
stat_analyzer = StatisticalAnalyzer()
statistical_results = {}
for metric in ['KLDivergence', 'LogLikelihood', 'WassersteinDistance']:
    statistical_results[metric] = stat_analyzer.compare_all_generators(
        experimental_runs, metric
    )

# Step 3: Convergence analysis
conv_analyzer = AdvancedConvergenceAnalyzer()
convergence_results = conv_analyzer.analyze_convergence_events(experimental_runs)

# Step 4: Stability analysis
stability_analyzer = StabilityAnalyzer()
stability_results = stability_analyzer.analyze_stability(experimental_runs)

# Step 5: Generate reports
report_gen = ReportGenerator()
reports = report_gen.generate_comprehensive_report(
    statistical_results, convergence_results, stability_results
)

print("Analysis complete! Check docs/reports/ for results.")
```

### 2. Demo with Simulated Data

For immediate testing without running experiments:

```python
from docs.statistical_analysis_demo import main

# Run complete demo with simulated data
main()
```

This demo:
- Generates realistic experimental data (80 runs across 4 generator types)
- Performs all analyses
- Creates visualizations
- Generates summary report
- Takes ~2-3 minutes to complete

## Detailed Usage Examples

### Example 1: Custom Experiment Configuration

```python
# Create custom experiment configuration
runner = ExperimentRunner()

# Advanced experiment plan
experiment_plan = runner.create_experiment_plan(
    generator_types=['classical_normal', 'quantum_shadows'],
    n_runs_per_type=25,
    seed_range=(100, 200),
    max_epochs_range=(40, 80),
    batch_sizes=[256, 512],
    learning_rates=[0.001, 0.0005],
    killer_modes=[False, True]
)

# Run with progress monitoring
results = runner.run_experiments(
    experiment_plan, 
    parallel=True, 
    save_interval=3
)

# Load previous results
previous_results = runner.load_results()
```

### Example 2: Focused Statistical Analysis

```python
# Initialize with custom parameters
analyzer = StatisticalAnalyzer(
    alpha=0.01,  # More stringent significance level
    correction_method='fdr_bh'  # False Discovery Rate correction
)

# Group runs by generator type
groups = analyzer.group_runs_by_generator(experimental_runs)

# Detailed analysis for specific metric
metric = 'KLDivergence'
for gen1, gen2 in itertools.combinations(groups.keys(), 2):
    values1 = analyzer.extract_metric_values(groups[gen1], metric)
    values2 = analyzer.extract_metric_values(groups[gen2], metric)
    
    # Perform statistical test
    test_result = analyzer.perform_t_test(values1, values2)
    print(f"{gen1} vs {gen2}: {test_result.interpretation}")
    
    # Bootstrap confidence interval
    ci_lower, ci_upper = analyzer.bootstrap_confidence_interval(values1)
    print(f"{gen1} 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Permutation test for robustness
    p_perm = analyzer.permutation_test(values1, values2)
    print(f"Permutation test p-value: {p_perm:.4f}")
```

### Example 3: Advanced Convergence Analysis

```python
# Initialize with custom thresholds
conv_analyzer = AdvancedConvergenceAnalyzer()
conv_analyzer.thresholds = {
    'KLDivergence': 0.20,      # Stricter convergence criterion
    'LogLikelihood': -2.5,
    'WassersteinDistance': 0.10
}

# Analyze specific aspects
results = conv_analyzer.analyze_convergence_events(experimental_runs)

# Pattern classification details
pattern_analysis = results['pattern_analysis']
print("Convergence Pattern Distribution:")
for pattern, frequency in pattern_analysis['pattern_frequency'].items():
    print(f"  {pattern}: {frequency} occurrences")

# Survival analysis (if lifelines available)
survival_analysis = results['survival_analysis']
for metric, curves in survival_analysis.items():
    print(f"\n{metric} Survival Analysis:")
    for gen_type, curve_data in curves.items():
        median_time = curve_data.get('median_survival_time', 'Not reached')
        print(f"  {gen_type}: Median time to convergence = {median_time}")

# Generate detailed visualizations
conv_analyzer.create_convergence_visualizations(results)
```

### Example 4: Comprehensive Stability Assessment

```python
# Initialize with specific outlier detection
stability_analyzer = StabilityAnalyzer(
    confidence_level=0.99,  # 99% confidence intervals
)

# Custom outlier detection parameters
stability_analyzer.outlier_detector.contamination = 0.05  # Expect 5% outliers

# Perform analysis
results = stability_analyzer.analyze_stability(experimental_runs)

# Access detailed stability metrics
stability_metrics = results['stability_metrics']
for gen_type, metrics in stability_metrics.items():
    print(f"\n{gen_type} Stability Assessment:")
    for metric_name, stability in metrics.items():
        if hasattr(stability, 'consistency_score'):
            print(f"  {metric_name}:")
            print(f"    Consistency Score: {stability.consistency_score:.3f}")
            print(f"    Risk Score: {stability.risk_score:.3f}")
            print(f"    Outlier Rate: {stability.outlier_rate:.1%}")

# Risk assessment summary
risk_assessment = results['risk_assessment']
for gen_type, assessment in risk_assessment.items():
    print(f"\n{gen_type} Risk Assessment:")
    print(f"  Overall Risk: {assessment['risk_classification']}")
    print(f"  Deployment Recommendation: {assessment['deployment_recommendation']}")

# Create comprehensive visualizations
stability_analyzer.create_stability_visualizations(results)
```

### Example 5: Custom Report Generation

```python
# Initialize with custom parameters
report_gen = ReportGenerator(
    output_dir="custom_reports",
    report_title="Production Readiness Analysis",
    author="ML Engineering Team",
    institution="Research Lab"
)

# Generate specific report types
reports = report_gen.generate_comprehensive_report(
    statistical_results=statistical_results,
    convergence_results=convergence_results,
    stability_results=stability_results,
    experiment_metadata={
        'experiment_purpose': 'Production deployment evaluation',
        'hardware_config': 'GPU cluster with 8x A100',
        'dataset_version': 'v2.1',
        'additional_notes': 'Extended runs for stability assessment'
    }
)

# Access individual report files
executive_summary = reports['executive_summary']
technical_report = reports['technical_report']
interactive_dashboard = reports['interactive_dashboard']
data_tables = reports['data_tables']

print(f"Executive summary: {executive_summary}")
print(f"Technical report: {technical_report}")
print(f"Interactive dashboard: {interactive_dashboard}")
print(f"Data tables directory: {data_tables}")
```

## Advanced Features

### 1. Integration with Existing Training

The framework integrates seamlessly with the existing GaussGAN training infrastructure:

```python
# In your training script
from source.metrics import ConvergenceTracker

# The framework automatically uses existing convergence tracking
tracker = ConvergenceTracker(
    patience=10,
    min_delta=1e-4,
    monitor_metric="KLDivergence"
)

# During training
convergence_info = tracker.update(
    epoch=current_epoch,
    metrics=validation_metrics,
    d_loss=discriminator_loss,
    g_loss=generator_loss
)

# The framework extracts this information automatically
```

### 2. Custom Metrics Integration

Add your own metrics to the analysis:

```python
# Define custom metric
class CustomMetric(GaussianMetric):
    def compute_score(self, points):
        # Your custom metric calculation
        return custom_scores

# Register with framework
from source.metrics import ALL_METRICS
ALL_METRICS['CustomMetric'] = CustomMetric

# Use in analysis
results = analyzer.compare_all_generators(runs, 'CustomMetric')
```

### 3. Parallel Processing

For large-scale experiments:

```python
# Enable parallel experiment execution
runner = ExperimentRunner(max_parallel=4)

# For analysis, use multiprocessing
import multiprocessing as mp

def analyze_metric_subset(metric_subset):
    return {metric: analyzer.compare_all_generators(runs, metric) 
            for metric in metric_subset}

# Split metrics across processes
metrics = ['KLDivergence', 'LogLikelihood', 'WassersteinDistance', 'MMDDistance']
metric_chunks = np.array_split(metrics, mp.cpu_count())

with mp.Pool() as pool:
    results_chunks = pool.map(analyze_metric_subset, metric_chunks)

# Combine results
combined_results = {}
for chunk in results_chunks:
    combined_results.update(chunk)
```

### 4. External Data Integration

Load experimental data from external sources:

```python
# Load from MLflow
import mlflow

# Search for experiments
experiment = mlflow.get_experiment_by_name("GaussGAN_Comparison")
runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Convert to framework format
experimental_runs = []
for _, run_data in runs_df.iterrows():
    run = ExperimentRun(
        generator_type=run_data['tags.generator_type'],
        seed=int(run_data['tags.seed']),
        run_id=run_data['run_id'],
        final_metrics={
            'KLDivergence': run_data['metrics.KLDivergence'],
            'LogLikelihood': run_data['metrics.LogLikelihood']
        },
        # ... other fields
    )
    experimental_runs.append(run)

# Proceed with analysis
results = analyzer.compare_all_generators(experimental_runs, 'KLDivergence')
```

## Interpreting Results

### 1. Statistical Significance

```python
# Understanding test results
test_result = analyzer.perform_t_test(group1_values, group2_values)

if test_result.significant:
    print(f"Significant difference found (p={test_result.p_value:.4f})")
    
    # Interpret effect size
    effect_size = test_result.effect_size
    if abs(effect_size) < 0.2:
        magnitude = "small"
    elif abs(effect_size) < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"
    
    print(f"Effect size: {effect_size:.3f} ({magnitude} effect)")
    
    # Check statistical power
    if test_result.power and test_result.power < 0.8:
        print("Warning: Low statistical power, consider more samples")
```

### 2. Convergence Patterns

```python
# Interpreting convergence analysis
convergence_results = conv_analyzer.analyze_convergence_events(runs)

pattern_analysis = convergence_results['pattern_analysis']
for pattern, performance in pattern_analysis['pattern_performance'].items():
    conv_rate = performance['convergence_rate']
    avg_epochs = performance['avg_epochs_to_convergence']
    
    print(f"{pattern}: {conv_rate:.1%} convergence rate, "
          f"average {avg_epochs:.1f} epochs")
    
    # Interpretation guidelines
    if conv_rate > 0.9:
        reliability = "Highly reliable"
    elif conv_rate > 0.7:
        reliability = "Moderately reliable"
    else:
        reliability = "Unreliable"
    
    print(f"  Assessment: {reliability}")
```

### 3. Stability Assessment

```python
# Risk level interpretation
risk_assessment = stability_results['risk_assessment']

for gen_type, assessment in risk_assessment.items():
    risk_score = assessment['overall_risk_score']
    risk_class = assessment['risk_classification']
    deployment_rec = assessment['deployment_recommendation']
    
    print(f"{gen_type}:")
    print(f"  Risk Score: {risk_score:.3f}")
    print(f"  Classification: {risk_class}")
    print(f"  Recommendation: {deployment_rec}")
    
    # Decision guidelines
    if risk_class == "Low Risk":
        decision = "Suitable for production deployment"
    elif risk_class == "Medium Risk":
        decision = "Deploy with monitoring and rollback plan"
    else:
        decision = "Requires further investigation before deployment"
    
    print(f"  Decision: {decision}")
```

### 4. Overall Recommendations

```python
# Synthesizing results for decision making
def make_recommendation(statistical_results, convergence_results, stability_results):
    recommendations = []
    
    # Best performer by metrics
    best_performers = statistical_results['summary_insights']['best_performing_generators']
    most_frequent_best = max(best_performers.values(), 
                           key=lambda x: list(best_performers.values()).count(x))
    recommendations.append(f"Best overall performer: {most_frequent_best['generator']}")
    
    # Most reliable converger
    conv_rates = convergence_results['speed_comparison']['KLDivergence']
    most_reliable = max(conv_rates.keys(), 
                       key=lambda x: conv_rates[x]['convergence_rate'])
    recommendations.append(f"Most reliable convergence: {most_reliable}")
    
    # Lowest risk
    risk_assessment = stability_results['risk_assessment']
    lowest_risk = min(risk_assessment.keys(), 
                     key=lambda x: risk_assessment[x]['overall_risk_score'])
    recommendations.append(f"Lowest risk: {lowest_risk}")
    
    return recommendations

recommendations = make_recommendation(
    statistical_results, convergence_results, stability_results
)

for rec in recommendations:
    print(f"• {rec}")
```

## Best Practices

### 1. Experimental Design

- **Sample Size**: Aim for at least 15-20 runs per generator type for robust statistics
- **Randomization**: Use different random seeds for each run to ensure independence
- **Balanced Design**: Equal number of runs across all generator types
- **Replication**: Include technical replicates with identical configurations

```python
# Good experimental design
experiment_plan = runner.create_experiment_plan(
    generator_types=['classical_normal', 'classical_uniform', 
                     'quantum_samples', 'quantum_shadows'],
    n_runs_per_type=20,          # Adequate sample size
    seed_range=(42, 142),        # Wide range of seeds
    max_epochs_range=(40, 60),   # Consistent training duration
    batch_sizes=[256],           # Control for batch size effect
    learning_rates=[0.001]       # Control for learning rate effect
)
```

### 2. Statistical Analysis

- **Multiple Comparisons**: Always correct for multiple comparisons when testing multiple generator pairs
- **Effect Sizes**: Report effect sizes alongside p-values for practical significance
- **Assumptions**: Check statistical test assumptions (normality, equal variances)
- **Robustness**: Use non-parametric tests when assumptions are violated

```python
# Robust statistical analysis
analyzer = StatisticalAnalyzer(
    alpha=0.05,
    correction_method='bonferroni'  # Conservative correction
)

# Check assumptions before testing
for gen_type, runs in groups.items():
    values = analyzer.extract_metric_values(runs, 'KLDivergence')
    
    # Test normality
    is_normal, p_normal = analyzer.test_normality(values)
    print(f"{gen_type} normality: {is_normal} (p={p_normal:.4f})")
    
    # If not normal, use non-parametric tests
    if not is_normal:
        print(f"Using Mann-Whitney U test for {gen_type}")
```

### 3. Convergence Analysis

- **Thresholds**: Set meaningful convergence thresholds based on domain knowledge
- **Pattern Recognition**: Use convergence patterns to guide hyperparameter tuning
- **Early Stopping**: Implement early stopping based on convergence analysis

```python
# Domain-informed convergence thresholds
conv_analyzer.thresholds = {
    'KLDivergence': 0.2,        # Based on literature benchmarks
    'LogLikelihood': -2.5,      # Domain expert input
    'WassersteinDistance': 0.1  # Practical significance threshold
}
```

### 4. Stability Assessment

- **Outlier Handling**: Use ensemble outlier detection for robustness
- **Risk Tolerance**: Adjust risk thresholds based on application requirements
- **Monitoring**: Implement ongoing stability monitoring in production

```python
# Production-oriented stability analysis
stability_analyzer = StabilityAnalyzer(confidence_level=0.99)

# Strict outlier detection for production
stability_analyzer.outlier_detector.contamination = 0.02

# Monitor key stability metrics
key_metrics = ['consistency_score', 'reliability_score', 'risk_score']
```

### 5. Reporting and Documentation

- **Reproducibility**: Document all parameters and random seeds
- **Version Control**: Track framework and data versions
- **Peer Review**: Have statistical analyses reviewed by domain experts
- **Transparency**: Report all analyses performed, not just significant results

```python
# Comprehensive documentation
report_gen = ReportGenerator()
reports = report_gen.generate_comprehensive_report(
    statistical_results=statistical_results,
    convergence_results=convergence_results,
    stability_results=stability_results,
    experiment_metadata={
        'framework_version': '1.0.0',
        'data_version': '2023-12-01',
        'analysis_date': datetime.now().isoformat(),
        'reviewer': 'Senior Data Scientist',
        'random_seeds': list(range(42, 142)),
        'exclusion_criteria': 'Runs with >90% NaN values',
        'preprocessing_steps': ['Outlier detection', 'Normalization']
    }
)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Insufficient Data

**Problem**: Not enough experimental runs for reliable statistics

**Symptoms**:
```
Warning: Only 3 runs available for classical_normal
Statistical test failed: insufficient data
```

**Solution**:
```python
# Check data availability
groups = analyzer.group_runs_by_generator(runs)
for gen_type, gen_runs in groups.items():
    print(f"{gen_type}: {len(gen_runs)} runs")
    if len(gen_runs) < 10:
        print(f"Warning: {gen_type} has insufficient data")

# Run more experiments if needed
additional_runs = runner.create_experiment_plan(
    generator_types=['classical_normal'],  # Focus on missing data
    n_runs_per_type=15
)
new_results = runner.run_experiments(additional_runs)
all_runs = runs + new_results
```

#### 2. Convergence Issues

**Problem**: Many runs fail to converge

**Symptoms**:
```
Pattern analysis shows 70% 'divergent' patterns
Low convergence rates across all generators
```

**Solution**:
```python
# Investigate convergence issues
convergence_results = conv_analyzer.analyze_convergence_events(runs)

# Check pattern distribution
patterns = convergence_results['pattern_analysis']['pattern_frequency']
print("Pattern distribution:", patterns)

# Adjust thresholds if too strict
conv_analyzer.thresholds = {
    'KLDivergence': 0.5,  # More lenient threshold
    'LogLikelihood': -3.5
}

# Or extend training duration
experiment_plan = runner.create_experiment_plan(
    max_epochs_range=(60, 100)  # Longer training
)
```

#### 3. High Variance

**Problem**: Extremely high variance in results

**Symptoms**:
```
Coefficient of variation > 2.0
Risk scores > 0.8 for all generators
Large confidence intervals
```

**Solution**:
```python
# Investigate variance sources
stability_results = stability_analyzer.analyze_stability(runs)

# Check outlier impact
outlier_analysis = stability_results['outlier_analysis']
for gen_type, analysis in outlier_analysis['outlier_impact'].items():
    impact = analysis.get('relative_impact_on_mean', 0)
    if impact > 0.1:
        print(f"{gen_type} heavily affected by outliers: {impact:.1%}")

# Use robust statistics
analyzer = StatisticalAnalyzer()
for gen_type, runs in groups.items():
    values = analyzer.extract_metric_values(runs, 'KLDivergence')
    
    # Compare mean vs median
    mean_val = np.mean(values)
    median_val = np.median(values)
    print(f"{gen_type}: mean={mean_val:.4f}, median={median_val:.4f}")
```

#### 4. Memory Issues

**Problem**: Out of memory during large-scale analysis

**Symptoms**:
```
MemoryError: Unable to allocate array
Process killed during experiment execution
```

**Solution**:
```python
# Process in chunks
def analyze_in_chunks(runs, chunk_size=50):
    results = []
    for i in range(0, len(runs), chunk_size):
        chunk = runs[i:i+chunk_size]
        chunk_results = analyzer.compare_all_generators(chunk, 'KLDivergence')
        results.append(chunk_results)
    return results

# Reduce parallel processing
runner = ExperimentRunner(max_parallel=1)

# Use memory-efficient data structures
import gc
gc.collect()  # Force garbage collection
```

#### 5. Visualization Issues

**Problem**: Plots not displaying or saving correctly

**Symptoms**:
```
Matplotlib backend not available
Plotly plots not interactive
PDF generation fails
```

**Solution**:
```python
# Check and set matplotlib backend
import matplotlib
print(f"Current backend: {matplotlib.get_backend()}")

# For headless environments
matplotlib.use('Agg')

# Install missing dependencies
# pip install plotly kaleido  # For Plotly static image export

# Check Plotly availability
try:
    import plotly
    print("Plotly available")
except ImportError:
    print("Plotly not available, interactive plots disabled")

# Alternative: Save individual plots
plt.figure(figsize=(10, 6))
# ... create plot ...
plt.savefig('manual_plot.png', dpi=300, bbox_inches='tight')
plt.close()
```

### Performance Optimization

#### 1. Speed Up Experiments

```python
# Use smaller networks for initial testing
config_fast = {
    'nn_gen': "[64,64]",      # Smaller networks
    'nn_disc': "[64,64]",
    'max_epochs': 20,         # Fewer epochs
    'batch_size': 128,        # Smaller batches
    'validation_samples': 100 # Fewer validation samples
}

# Reduce quantum circuit complexity
config_quantum = {
    'quantum_qubits': 4,      # Fewer qubits
    'quantum_layers': 1,      # Fewer layers
    'quantum_shots': 50       # Fewer shots
}
```

#### 2. Optimize Analysis

```python
# Pre-filter runs
valid_runs = [run for run in runs if run.converged and run.final_metrics]

# Focus on key metrics first
priority_metrics = ['KLDivergence', 'LogLikelihood']

# Use sampling for large datasets
if len(runs) > 100:
    sample_runs = np.random.choice(runs, size=100, replace=False)
    quick_results = analyzer.compare_all_generators(sample_runs, 'KLDivergence')
```

## API Reference

### Core Classes

#### ExperimentRunner
```python
class ExperimentRunner:
    def __init__(self, base_config_path: str, output_dir: str, max_parallel: int = 1)
    def create_experiment_plan(self, generator_types: List[str], n_runs_per_type: int, **kwargs) -> List[ExperimentConfig]
    def run_experiments(self, configs: List[ExperimentConfig], parallel: bool = False) -> List[ExperimentResult]
    def load_results(self) -> List[ExperimentResult]
```

#### StatisticalAnalyzer
```python
class StatisticalAnalyzer:
    def __init__(self, alpha: float = 0.05, correction_method: str = 'bonferroni')
    def compare_all_generators(self, runs: List[ExperimentRun], metric_name: str) -> Dict[str, Dict[str, StatisticalTest]]
    def perform_t_test(self, group1: List[float], group2: List[float]) -> StatisticalTest
    def bootstrap_confidence_interval(self, data: List[float], confidence_level: float = 0.95) -> Tuple[float, float]
```

#### AdvancedConvergenceAnalyzer
```python
class AdvancedConvergenceAnalyzer:
    def __init__(self, output_dir: Union[str, Path] = "docs/convergence_analysis")
    def analyze_convergence_events(self, experimental_runs: List[Any], metrics_to_analyze: List[str] = None) -> Dict[str, Any]
    def create_convergence_visualizations(self, analysis_results: Dict[str, Any])
```

#### StabilityAnalyzer
```python
class StabilityAnalyzer:
    def __init__(self, output_dir: Union[str, Path] = "docs/stability_analysis", confidence_level: float = 0.95)
    def analyze_stability(self, experimental_runs: List[Any], metrics_to_analyze: List[str] = None) -> Dict[str, Any]
    def create_stability_visualizations(self, analysis_results: Dict[str, Any])
```

#### ReportGenerator
```python
class ReportGenerator:
    def __init__(self, output_dir: Union[str, Path], report_title: str, author: str = "", institution: str = "")
    def generate_comprehensive_report(self, statistical_results: Dict[str, Any], convergence_results: Dict[str, Any], stability_results: Dict[str, Any], experiment_metadata: Dict[str, Any] = None) -> Dict[str, Path]
```

### Data Structures

#### ExperimentRun
```python
@dataclass
class ExperimentRun:
    generator_type: str
    seed: int
    run_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    convergence_epoch: Optional[int] = None
    final_metrics: Dict[str, float] = None
    metric_history: Dict[str, List[float]] = None
    loss_history: Dict[str, List[float]] = None
    hyperparameters: Dict[str, Any] = None
    training_duration: Optional[float] = None
    converged: bool = False
```

#### StatisticalTest
```python
@dataclass
class StatisticalTest:
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    power: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""
    significant: bool = False
```

This comprehensive guide provides everything needed to effectively use the GaussGAN Statistical Analysis Framework for rigorous quantum vs classical generator comparison. The framework ensures reproducible, statistically sound results suitable for scientific publication and production deployment decisions.