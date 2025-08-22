"""
Regression testing suite for performance baseline validation.
Prevents performance degradation during development and tracks improvements.
"""

import pytest
import torch
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings
from typing import Dict, List, Any, Optional

from source.nn import ClassicalNoise, QuantumNoise, QuantumShadowNoise, MLPGenerator
from source.metrics import LogLikelihood, KLDivergence, IsPositive


class PerformanceBaseline:
    """Manager for performance baselines and regression detection."""
    
    def __init__(self, baseline_dir: Path):
        """
        Initialize performance baseline manager.
        
        Args:
            baseline_dir: Directory to store baseline files
        """
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        
        # Thresholds for regression detection
        self.regression_thresholds = {
            'generation_time': 1.5,        # 50% slower is regression
            'memory_usage': 1.3,           # 30% more memory is regression
            'kl_divergence': 2.0,          # 100% worse KL is regression
            'log_likelihood': 0.5,         # 50% worse log-likelihood is regression
            'convergence_epochs': 1.5      # 50% more epochs is regression
        }
        
        # Minimum improvement thresholds for baseline updates
        self.improvement_thresholds = {
            'generation_time': 0.9,        # 10% faster
            'memory_usage': 0.9,           # 10% less memory
            'kl_divergence': 0.8,          # 20% better KL
            'log_likelihood': 1.1,         # 10% better log-likelihood
            'convergence_epochs': 0.8      # 20% fewer epochs
        }
    
    def save_baseline(self, baseline_name: str, metrics: Dict[str, Any], 
                     metadata: Optional[Dict] = None):
        """
        Save performance baseline to disk.
        
        Args:
            baseline_name: Name of the baseline
            metrics: Performance metrics dictionary
            metadata: Additional metadata (config, git hash, etc.)
        """
        baseline_data = {
            'metrics': metrics,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'regression_thresholds': self.regression_thresholds.copy(),
            'improvement_thresholds': self.improvement_thresholds.copy()
        }
        
        baseline_file = self.baseline_dir / f"{baseline_name}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        json_data = convert_numpy(baseline_data)
        
        with open(baseline_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Baseline saved: {baseline_file}")
    
    def load_baseline(self, baseline_name: str) -> Optional[Dict]:
        """
        Load performance baseline from disk.
        
        Args:
            baseline_name: Name of the baseline to load
            
        Returns:
            Baseline data dictionary or None if not found
        """
        baseline_file = self.baseline_dir / f"{baseline_name}.json"
        
        if not baseline_file.exists():
            return None
        
        with open(baseline_file, 'r') as f:
            return json.load(f)
    
    def detect_regression(self, baseline_name: str, current_metrics: Dict[str, Any],
                         strict: bool = False) -> Dict[str, Any]:
        """
        Detect performance regression compared to baseline.
        
        Args:
            baseline_name: Name of baseline to compare against
            current_metrics: Current performance metrics
            strict: If True, use stricter thresholds
            
        Returns:
            Dictionary with regression analysis results
        """
        baseline = self.load_baseline(baseline_name)
        
        if baseline is None:
            return {
                'error': f'Baseline {baseline_name} not found',
                'status': 'no_baseline',
                'recommendations': ['Create initial baseline by running with --save-baseline']
            }
        
        baseline_metrics = baseline['metrics']
        thresholds = baseline.get('regression_thresholds', self.regression_thresholds)
        
        if strict:
            # Use stricter thresholds for critical tests
            thresholds = {k: v * 0.8 for k, v in thresholds.items()}
        
        regression_results = {
            'baseline_name': baseline_name,
            'baseline_timestamp': baseline.get('timestamp', 'unknown'),
            'regressions': [],
            'improvements': [],
            'stable_metrics': [],
            'missing_metrics': [],
            'new_metrics': [],
            'overall_status': 'pass'
        }
        
        # Check each metric for regression
        for metric, current_value in current_metrics.items():
            if metric not in baseline_metrics:
                regression_results['new_metrics'].append(metric)
                continue
            
            baseline_value = baseline_metrics[metric]
            
            # Skip if either value is not numeric
            if not isinstance(current_value, (int, float, np.number)) or \
               not isinstance(baseline_value, (int, float, np.number)):
                continue
            
            # Skip if baseline is zero (would cause division by zero)
            if baseline_value == 0:
                continue
            
            ratio = current_value / baseline_value
            threshold = thresholds.get(metric, 1.2)  # Default 20% regression threshold
            improvement_threshold = self.improvement_thresholds.get(metric, 0.9)
            
            if ratio > threshold:
                # Regression detected
                regression_results['regressions'].append({
                    'metric': metric,
                    'baseline_value': baseline_value,
                    'current_value': current_value,
                    'ratio': ratio,
                    'threshold': threshold,
                    'percent_change': (ratio - 1) * 100
                })
                regression_results['overall_status'] = 'regression'
                
            elif ratio < improvement_threshold:
                # Significant improvement detected
                regression_results['improvements'].append({
                    'metric': metric,
                    'baseline_value': baseline_value,
                    'current_value': current_value,
                    'ratio': ratio,
                    'threshold': improvement_threshold,
                    'percent_change': (ratio - 1) * 100
                })
                
            else:
                # Stable performance
                regression_results['stable_metrics'].append({
                    'metric': metric,
                    'baseline_value': baseline_value,
                    'current_value': current_value,
                    'ratio': ratio,
                    'percent_change': (ratio - 1) * 100
                })
        
        # Check for missing metrics
        for metric in baseline_metrics:
            if metric not in current_metrics:
                regression_results['missing_metrics'].append(metric)
        
        # Generate recommendations
        recommendations = []
        if regression_results['regressions']:
            recommendations.append("Performance regression detected - investigate before merge")
            recommendations.append("Consider profiling slow operations")
            
        if regression_results['improvements']:
            recommendations.append("Performance improvements detected - consider updating baseline")
            
        if regression_results['missing_metrics']:
            recommendations.append(f"Missing metrics: {regression_results['missing_metrics']}")
            
        regression_results['recommendations'] = recommendations
        
        return regression_results
    
    def generate_regression_report(self, regression_results: Dict[str, Any]) -> str:
        """Generate human-readable regression report."""
        report = []
        report.append("=" * 80)
        report.append("PERFORMANCE REGRESSION ANALYSIS REPORT")
        report.append("=" * 80)
        
        if 'error' in regression_results:
            report.append(f"ERROR: {regression_results['error']}")
            return "\n".join(report)
        
        report.append(f"Baseline: {regression_results['baseline_name']}")
        report.append(f"Baseline Date: {regression_results['baseline_timestamp']}")
        report.append(f"Overall Status: {regression_results['overall_status'].upper()}")
        report.append("")
        
        # Regressions
        if regression_results['regressions']:
            report.append("🔴 PERFORMANCE REGRESSIONS DETECTED:")
            report.append("-" * 50)
            for reg in regression_results['regressions']:
                report.append(f"  {reg['metric']}:")
                report.append(f"    Baseline: {reg['baseline_value']:.4f}")
                report.append(f"    Current:  {reg['current_value']:.4f}")
                report.append(f"    Change:   {reg['percent_change']:+.1f}% (threshold: {(reg['threshold']-1)*100:.1f}%)")
                report.append("")
        
        # Improvements  
        if regression_results['improvements']:
            report.append("🟢 PERFORMANCE IMPROVEMENTS DETECTED:")
            report.append("-" * 50)
            for imp in regression_results['improvements']:
                report.append(f"  {imp['metric']}:")
                report.append(f"    Baseline: {imp['baseline_value']:.4f}")
                report.append(f"    Current:  {imp['current_value']:.4f}")
                report.append(f"    Change:   {imp['percent_change']:+.1f}%")
                report.append("")
        
        # Stable metrics
        if regression_results['stable_metrics']:
            report.append("⚫ STABLE PERFORMANCE:")
            report.append("-" * 30)
            for stable in regression_results['stable_metrics']:
                report.append(f"  {stable['metric']}: {stable['percent_change']:+.1f}%")
            report.append("")
        
        # New/missing metrics
        if regression_results['new_metrics']:
            report.append(f"📊 NEW METRICS: {', '.join(regression_results['new_metrics'])}")
            
        if regression_results['missing_metrics']:
            report.append(f"❓ MISSING METRICS: {', '.join(regression_results['missing_metrics'])}")
        
        # Recommendations
        if regression_results['recommendations']:
            report.append("")
            report.append("💡 RECOMMENDATIONS:")
            report.append("-" * 20)
            for rec in regression_results['recommendations']:
                report.append(f"  • {rec}")
        
        report.append("\n" + "=" * 80)
        return "\n".join(report)


class GeneratorPerformanceCollector:
    """Collects comprehensive performance metrics for generators."""
    
    def __init__(self, device, config, gaussian_params):
        self.device = device
        self.config = config
        self.gaussian_params = gaussian_params
    
    def create_generator(self, generator_type: str):
        """Create generator of specified type."""
        if generator_type == 'classical_normal':
            noise_gen = ClassicalNoise(
                z_dim=self.config['z_dim'],
                generator_type='classical_normal'
            )
        elif generator_type == 'quantum_samples':
            noise_gen = QuantumNoise(
                num_qubits=self.config['z_dim'],
                num_layers=self.config['quantum_layers']
            )
        elif generator_type == 'quantum_shadows':
            noise_gen = QuantumShadowNoise(
                z_dim=self.config['z_dim'],
                num_qubits=self.config['quantum_qubits'],
                num_layers=self.config['quantum_layers'],
                num_basis=self.config['quantum_basis']
            )
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")
        
        mlp_gen = MLPGenerator(
            non_linearity=self.config['non_linearity'],
            hidden_dims=self.config['nn_gen'],
            z_dim=self.config['z_dim'],
            std_scale=self.config['std_scale'],
            min_std=self.config['min_std']
        )
        
        generator = torch.nn.Sequential(noise_gen, mlp_gen)
        return generator.to(self.device)
    
    def collect_comprehensive_metrics(self, generator_types: List[str], 
                                    n_runs: int = 5) -> Dict[str, Dict[str, float]]:
        """Collect comprehensive performance metrics."""
        all_metrics = {}
        
        for gen_type in generator_types:
            try:
                generator = self.create_generator(gen_type)
                metrics = self._collect_single_generator_metrics(generator, gen_type, n_runs)
                all_metrics[gen_type] = metrics
                
            except Exception as e:
                print(f"Error collecting metrics for {gen_type}: {e}")
                all_metrics[gen_type] = {'error': str(e)}
        
        return all_metrics
    
    def _collect_single_generator_metrics(self, generator, gen_type: str, 
                                        n_runs: int) -> Dict[str, float]:
        """Collect metrics for a single generator."""
        import time
        import psutil
        
        metrics = defaultdict(list)
        
        # Performance metrics
        batch_sizes = [32, 64, 128] if self.device.type == 'cuda' else [32, 64]
        
        for batch_size in batch_sizes:
            # Timing tests
            times = []
            memory_usage = []
            
            for _ in range(n_runs):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                
                start_time = time.time()
                
                with torch.no_grad():
                    samples = generator(batch_size)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                elapsed_time = time.time() - start_time
                times.append(elapsed_time)
                
                if torch.cuda.is_available():
                    memory_usage.append(torch.cuda.max_memory_allocated() / 1024**2)  # MB
            
            metrics[f'generation_time_batch_{batch_size}'] = np.mean(times)
            metrics[f'generation_time_std_batch_{batch_size}'] = np.std(times)
            metrics[f'samples_per_second_batch_{batch_size}'] = batch_size / np.mean(times)
            
            if memory_usage:
                metrics[f'peak_memory_mb_batch_{batch_size}'] = np.mean(memory_usage)
                metrics[f'memory_per_sample_kb_batch_{batch_size}'] = np.mean(memory_usage) * 1024 / batch_size
        
        # Quality metrics
        log_likelihood = LogLikelihood(**self.gaussian_params)
        kl_divergence = KLDivergence(**self.gaussian_params)
        is_positive = IsPositive()
        
        quality_runs = []
        for _ in range(n_runs):
            with torch.no_grad():
                samples = generator(500)  # Fixed size for quality testing
                
                ll_scores = log_likelihood.compute_score(samples)
                kl_score = kl_divergence.compute_score(samples)
                pos_scores = is_positive.compute_score(samples)
                
                quality_runs.append({
                    'log_likelihood': np.mean(ll_scores),
                    'kl_divergence': kl_score,
                    'positive_ratio': np.mean([1 if p > 0 else 0 for p in pos_scores])
                })
        
        # Aggregate quality metrics
        for metric_name in ['log_likelihood', 'kl_divergence', 'positive_ratio']:
            values = [run[metric_name] for run in quality_runs if not (np.isnan(run[metric_name]) or np.isinf(run[metric_name]))]
            if values:
                metrics[metric_name] = np.mean(values)
                metrics[f'{metric_name}_std'] = np.std(values)
        
        # Convert defaultdict to regular dict with float values
        return {k: float(v) for k, v in metrics.items()}


@pytest.mark.regression
@pytest.mark.slow
class TestPerformanceBaselines:
    """Test suite for performance baseline validation and regression detection."""
    
    def setup_method(self):
        """Set up baseline testing environment."""
        self.generator_types = ['classical_normal']
        
        # Add quantum generators if available
        try:
            import pennylane as qml
            self.generator_types.append('quantum_samples')
        except ImportError:
            print("PennyLane not available, only testing classical generators")
    
    def test_baseline_manager_basic_operations(self, temp_directory):
        """Test basic baseline manager operations."""
        baseline_manager = PerformanceBaseline(temp_directory)
        
        # Test saving baseline
        test_metrics = {
            'generation_time': 0.1,
            'memory_usage': 100.0,
            'kl_divergence': 0.5,
            'log_likelihood': -2.0
        }
        
        test_metadata = {
            'generator_type': 'classical_normal',
            'config': {'z_dim': 4, 'batch_size': 64}
        }
        
        baseline_manager.save_baseline('test_baseline', test_metrics, test_metadata)
        
        # Test loading baseline
        loaded_baseline = baseline_manager.load_baseline('test_baseline')
        
        assert loaded_baseline is not None
        assert loaded_baseline['metrics'] == test_metrics
        assert loaded_baseline['metadata'] == test_metadata
        assert 'timestamp' in loaded_baseline
    
    def test_regression_detection_scenarios(self, temp_directory):
        """Test various regression detection scenarios."""
        baseline_manager = PerformanceBaseline(temp_directory)
        
        # Create baseline
        baseline_metrics = {
            'generation_time': 0.10,
            'memory_usage': 100.0,
            'kl_divergence': 0.50,
            'log_likelihood': -2.0
        }
        
        baseline_manager.save_baseline('regression_test', baseline_metrics)
        
        # Test 1: No regression (stable performance)
        stable_metrics = {
            'generation_time': 0.11,  # 10% slower (within threshold)
            'memory_usage': 105.0,    # 5% more memory
            'kl_divergence': 0.52,    # Slightly worse
            'log_likelihood': -2.1    # Slightly worse
        }
        
        results = baseline_manager.detect_regression('regression_test', stable_metrics)
        assert results['overall_status'] == 'pass'
        assert len(results['regressions']) == 0
        
        # Test 2: Clear regression
        regression_metrics = {
            'generation_time': 0.20,  # 100% slower
            'memory_usage': 150.0,    # 50% more memory
            'kl_divergence': 1.5,     # 200% worse
            'log_likelihood': -4.0    # 100% worse
        }
        
        results = baseline_manager.detect_regression('regression_test', regression_metrics)
        assert results['overall_status'] == 'regression'
        assert len(results['regressions']) > 0
        
        # Test 3: Improvements
        improved_metrics = {
            'generation_time': 0.05,  # 50% faster
            'memory_usage': 80.0,     # 20% less memory
            'kl_divergence': 0.35,    # 30% better
            'log_likelihood': -1.5    # 25% better
        }
        
        results = baseline_manager.detect_regression('regression_test', improved_metrics)
        assert len(results['improvements']) > 0
        
        # Test 4: Missing baseline
        results = baseline_manager.detect_regression('nonexistent_baseline', stable_metrics)
        assert results['status'] == 'no_baseline'
        assert 'error' in results
    
    def test_performance_collection_integration(self, device, test_config, 
                                               gaussian_mixture_params, temp_directory):
        """Test integration with performance collection."""
        collector = GeneratorPerformanceCollector(device, test_config, gaussian_mixture_params)
        baseline_manager = PerformanceBaseline(temp_directory)
        
        # Collect current performance metrics
        metrics = collector.collect_comprehensive_metrics(
            self.generator_types[:1], n_runs=2  # Reduced for testing
        )
        
        # Validate metrics collection
        for gen_type in self.generator_types[:1]:
            if 'error' not in metrics[gen_type]:
                gen_metrics = metrics[gen_type]
                
                # Check required metrics exist
                assert 'generation_time_batch_32' in gen_metrics
                assert 'kl_divergence' in gen_metrics
                assert 'log_likelihood' in gen_metrics
                
                # Save as baseline
                baseline_manager.save_baseline(
                    f'integration_test_{gen_type}',
                    gen_metrics,
                    {'generator_type': gen_type, 'test_config': test_config}
                )
                
                # Test regression detection with same metrics (should be stable)
                regression_results = baseline_manager.detect_regression(
                    f'integration_test_{gen_type}', gen_metrics
                )
                
                assert regression_results['overall_status'] == 'pass'
                
                # Generate and validate report
                report = baseline_manager.generate_regression_report(regression_results)
                assert isinstance(report, str)
                assert "REGRESSION ANALYSIS REPORT" in report
    
    def test_baseline_versioning_and_history(self, temp_directory):
        """Test baseline versioning and historical tracking."""
        baseline_manager = PerformanceBaseline(temp_directory)
        
        # Simulate evolution of performance over time
        versions = [
            ('v1.0', {'generation_time': 0.10, 'kl_divergence': 0.60}),
            ('v1.1', {'generation_time': 0.08, 'kl_divergence': 0.55}),  # Improvement
            ('v1.2', {'generation_time': 0.12, 'kl_divergence': 0.50}),  # Mixed
            ('v2.0', {'generation_time': 0.06, 'kl_divergence': 0.40}),  # Major improvement
        ]
        
        # Save all versions
        for version, metrics in versions:
            baseline_manager.save_baseline(f'history_test_{version}', metrics, 
                                         {'version': version})
        
        # Test progression
        for i in range(1, len(versions)):
            current_version, current_metrics = versions[i]
            prev_version, prev_metrics = versions[i-1]
            
            # Load previous baseline and compare
            prev_baseline = baseline_manager.load_baseline(f'history_test_{prev_version}')
            assert prev_baseline is not None
            
            regression_results = baseline_manager.detect_regression(
                f'history_test_{prev_version}', current_metrics
            )
            
            # v1.1 should show improvements over v1.0
            if current_version == 'v1.1':
                assert len(regression_results['improvements']) > 0
    
    def test_strict_regression_thresholds(self, temp_directory):
        """Test strict regression thresholds for critical paths."""
        baseline_manager = PerformanceBaseline(temp_directory)
        
        baseline_metrics = {'generation_time': 0.10, 'kl_divergence': 0.50}
        baseline_manager.save_baseline('strict_test', baseline_metrics)
        
        # Metrics that would pass normal thresholds but fail strict ones
        borderline_metrics = {'generation_time': 0.12, 'kl_divergence': 0.55}  # 20% and 10% worse
        
        # Normal threshold test
        normal_results = baseline_manager.detect_regression('strict_test', borderline_metrics, strict=False)
        
        # Strict threshold test  
        strict_results = baseline_manager.detect_regression('strict_test', borderline_metrics, strict=True)
        
        # Strict should be more sensitive
        assert len(strict_results['regressions']) >= len(normal_results['regressions'])
    
    def test_report_generation_comprehensiveness(self, temp_directory):
        """Test comprehensive report generation."""
        baseline_manager = PerformanceBaseline(temp_directory)
        
        baseline_metrics = {
            'generation_time': 0.10,
            'memory_usage': 100.0,
            'kl_divergence': 0.50,
            'log_likelihood': -2.0,
            'convergence_epochs': 20
        }
        
        baseline_manager.save_baseline('report_test', baseline_metrics)
        
        # Current metrics with mixed results
        current_metrics = {
            'generation_time': 0.20,    # Regression
            'memory_usage': 80.0,       # Improvement  
            'kl_divergence': 0.48,      # Stable
            'log_likelihood': -1.5,     # Improvement
            'new_metric': 0.95          # New metric
        }
        # Missing: convergence_epochs
        
        regression_results = baseline_manager.detect_regression('report_test', current_metrics)
        report = baseline_manager.generate_regression_report(regression_results)
        
        # Validate report content
        assert "REGRESSION ANALYSIS REPORT" in report
        assert "PERFORMANCE REGRESSIONS DETECTED" in report
        assert "PERFORMANCE IMPROVEMENTS DETECTED" in report
        assert "STABLE PERFORMANCE" in report
        assert "NEW METRICS" in report
        assert "MISSING METRICS" in report
        assert "RECOMMENDATIONS" in report
        
        # Check specific content
        assert "generation_time" in report  # Should be in regressions
        assert "memory_usage" in report     # Should be in improvements
        assert "kl_divergence" in report    # Should be in stable
        assert "new_metric" in report       # Should be in new metrics
        assert "convergence_epochs" in report  # Should be in missing metrics
        
        print(f"\nGenerated comprehensive report:\n{report}")
    
    def test_edge_cases_and_error_handling(self, temp_directory):
        """Test edge cases and error handling in baseline management."""
        baseline_manager = PerformanceBaseline(temp_directory)
        
        # Test with NaN and infinite values
        problematic_metrics = {
            'normal_metric': 0.5,
            'nan_metric': float('nan'),
            'inf_metric': float('inf'),
            'zero_metric': 0.0,
            'negative_metric': -0.1
        }
        
        baseline_manager.save_baseline('edge_case_test', problematic_metrics)
        
        current_metrics = {
            'normal_metric': 0.6,
            'nan_metric': 0.4,      # Now normal
            'inf_metric': 0.3,      # Now normal
            'zero_metric': 0.1,     # Changed from zero
            'negative_metric': -0.05 # Still negative
        }
        
        # Should handle gracefully without crashing
        results = baseline_manager.detect_regression('edge_case_test', current_metrics)
        assert 'error' not in results
        
        # Test empty metrics
        empty_metrics = {}
        results = baseline_manager.detect_regression('edge_case_test', empty_metrics)
        assert len(results['missing_metrics']) > 0
        
        # Test non-numeric values
        mixed_metrics = {
            'numeric': 0.5,
            'string': 'hello',
            'list': [1, 2, 3],
            'dict': {'nested': 'value'}
        }
        
        baseline_manager.save_baseline('mixed_test', mixed_metrics)
        results = baseline_manager.detect_regression('mixed_test', mixed_metrics)
        # Should process only numeric values


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])