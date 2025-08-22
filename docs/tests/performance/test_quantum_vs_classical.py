"""
Performance benchmarking tests for quantum vs classical generators.
Comprehensive comparison of training speed, memory usage, and generation quality.
"""

import pytest
import torch
import numpy as np
import time
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial

from source.nn import ClassicalNoise, QuantumNoise, QuantumShadowNoise, MLPGenerator
from source.model import GaussGan
from source.metrics import LogLikelihood, KLDivergence, IsPositive


class QuantumClassicalBenchmark:
    """Comprehensive benchmarking suite for quantum vs classical generators."""
    
    def __init__(self, device, test_config, gaussian_mixture_params):
        self.device = device
        self.config = test_config
        self.gaussian_params = gaussian_mixture_params
        self.results = defaultdict(dict)
    
    def create_generator(self, generator_type):
        """Create generator of specified type."""
        # Create noise generator
        if generator_type == 'classical_normal':
            noise_gen = ClassicalNoise(
                z_dim=self.config['z_dim'],
                generator_type='classical_normal'
            )
        elif generator_type == 'classical_uniform':
            noise_gen = ClassicalNoise(
                z_dim=self.config['z_dim'],
                generator_type='classical_uniform'
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
        
        # Create MLP generator
        mlp_gen = MLPGenerator(
            non_linearity=self.config['non_linearity'],
            hidden_dims=self.config['nn_gen'],
            z_dim=self.config['z_dim'],
            std_scale=self.config['std_scale'],
            min_std=self.config['min_std']
        )
        
        generator = torch.nn.Sequential(noise_gen, mlp_gen)
        return generator.to(self.device)
    
    def benchmark_forward_pass(self, generator_types, batch_sizes, n_iterations=10):
        """Benchmark forward pass performance across different batch sizes."""
        results = {}
        
        for gen_type in generator_types:
            try:
                generator = self.create_generator(gen_type)
                results[gen_type] = {}
                
                for batch_size in batch_sizes:
                    times = []
                    memory_usage = []
                    
                    for _ in range(n_iterations):
                        # Clear cache
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                        
                        # Measure forward pass
                        start_time = time.time()
                        
                        with torch.no_grad():
                            generated = generator(batch_size)
                        
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                        
                        elapsed_time = time.time() - start_time
                        times.append(elapsed_time)
                        
                        if torch.cuda.is_available():
                            memory_usage.append(torch.cuda.max_memory_allocated() / 1024**2)  # MB
                        
                    results[gen_type][batch_size] = {
                        'mean_time': np.mean(times),
                        'std_time': np.std(times),
                        'mean_memory': np.mean(memory_usage) if memory_usage else 0,
                        'samples_per_second': batch_size / np.mean(times)
                    }
                    
            except Exception as e:
                print(f"Skipping {gen_type} due to error: {e}")
                results[gen_type] = {'error': str(e)}
        
        return results
    
    def benchmark_training_convergence(self, generator_types, target_kl=1.0, max_epochs=50):
        """Benchmark training convergence speed for different generators."""
        results = {}
        
        # Create test data
        test_data = self._create_test_data()
        
        for gen_type in generator_types:
            try:
                print(f"Testing convergence for {gen_type}...")
                
                # Create model
                generator = self.create_generator(gen_type)
                discriminator = self._create_discriminator()
                predictor = self._create_predictor()
                
                model = GaussGan(
                    generator=generator,
                    discriminator=discriminator,
                    predictor=predictor,
                    optimizer=partial(torch.optim.Adam, lr=self.config['learning_rate']),
                    killer=False,  # Disable for clean comparison
                    n_critic=self.config['n_critic'],
                    grad_penalty=self.config['grad_penalty'],
                    metrics=['KLDivergence'],
                    gaussians=self.gaussian_params,
                    validation_samples=self.config['validation_samples']
                )
                model.to(self.device)
                
                # Training loop
                kl_history = []
                time_history = []
                epoch_times = []
                
                for epoch in range(max_epochs):
                    epoch_start = time.time()
                    
                    # Simulate training step
                    batch = (test_data[:self.config['batch_size']], 
                            torch.zeros(self.config['batch_size']))
                    
                    # Training step
                    model.training_step(batch, 0)
                    
                    # Validation
                    with torch.no_grad():
                        fake_data = model._generate_fake_data(self.config['validation_samples'])
                        metrics = model._compute_metrics(fake_data)
                        current_kl = metrics.get('KLDivergence', float('inf'))
                        if not np.isnan(current_kl) and not np.isinf(current_kl):
                            kl_history.append(current_kl)
                        else:
                            kl_history.append(float('inf'))
                    
                    epoch_time = time.time() - epoch_start
                    epoch_times.append(epoch_time)
                    time_history.append(np.sum(epoch_times))
                    
                    # Check convergence
                    if len(kl_history) > 5:
                        recent_kl = np.mean(kl_history[-5:])
                        if recent_kl < target_kl:
                            break
                
                # Find convergence epoch
                convergence_epoch = len(kl_history)
                for i, kl in enumerate(kl_history):
                    if kl < target_kl:
                        convergence_epoch = i + 1
                        break
                
                results[gen_type] = {
                    'convergence_epoch': convergence_epoch,
                    'final_kl': kl_history[-1] if kl_history else float('inf'),
                    'kl_history': kl_history,
                    'time_to_convergence': time_history[convergence_epoch-1] if convergence_epoch <= len(time_history) else time_history[-1],
                    'mean_epoch_time': np.mean(epoch_times),
                    'total_time': np.sum(epoch_times)
                }
                
            except Exception as e:
                print(f"Error testing convergence for {gen_type}: {e}")
                results[gen_type] = {'error': str(e)}
        
        return results
    
    def benchmark_generation_quality(self, generator_types, n_samples=1000, n_runs=5):
        """Benchmark generation quality across multiple runs."""
        results = {}
        
        for gen_type in generator_types:
            try:
                generator = self.create_generator(gen_type)
                quality_metrics = defaultdict(list)
                
                for run in range(n_runs):
                    with torch.no_grad():
                        generated = generator(n_samples)
                    
                    # Compute quality metrics
                    log_likelihood = LogLikelihood(**self.gaussian_params)
                    kl_divergence = KLDivergence(**self.gaussian_params)
                    is_positive = IsPositive()
                    
                    ll_score = log_likelihood.compute_score(generated)
                    kl_score = kl_divergence.compute_score(generated)
                    pos_score = is_positive.compute_score(generated)
                    
                    quality_metrics['log_likelihood'].append(np.mean(ll_score))
                    quality_metrics['kl_divergence'].append(kl_score)
                    quality_metrics['positive_ratio'].append(np.mean([1 if p > 0 else 0 for p in pos_score]))
                
                # Aggregate results
                results[gen_type] = {}
                for metric, values in quality_metrics.items():
                    results[gen_type][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'values': values
                    }
                    
            except Exception as e:
                print(f"Error testing quality for {gen_type}: {e}")
                results[gen_type] = {'error': str(e)}
        
        return results
    
    def benchmark_memory_usage(self, generator_types, batch_sizes):
        """Benchmark memory usage for different generators and batch sizes."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available for memory benchmarking")
        
        results = {}
        
        for gen_type in generator_types:
            try:
                results[gen_type] = {}
                
                for batch_size in batch_sizes:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    
                    generator = self.create_generator(gen_type)
                    initial_memory = torch.cuda.memory_allocated()
                    
                    # Forward pass
                    with torch.no_grad():
                        generated = generator(batch_size)
                    
                    peak_memory = torch.cuda.max_memory_allocated()
                    final_memory = torch.cuda.memory_allocated()
                    
                    results[gen_type][batch_size] = {
                        'initial_memory_mb': initial_memory / 1024**2,
                        'peak_memory_mb': peak_memory / 1024**2,
                        'final_memory_mb': final_memory / 1024**2,
                        'memory_increase_mb': (peak_memory - initial_memory) / 1024**2,
                        'memory_per_sample_kb': (peak_memory - initial_memory) / batch_size / 1024
                    }
                    
            except Exception as e:
                print(f"Error testing memory for {gen_type}: {e}")
                results[gen_type] = {'error': str(e)}
        
        return results
    
    def _create_test_data(self):
        """Create test data for training."""
        np.random.seed(42)
        # 2-component Gaussian mixture
        comp1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 500)
        comp2 = np.random.multivariate_normal([3, 3], [[1, 0], [0, 1]], 500)
        data = np.vstack([comp1, comp2])
        return torch.tensor(data, dtype=torch.float32).to(self.device)
    
    def _create_discriminator(self):
        """Create discriminator for testing."""
        from source.nn import MLPDiscriminator
        disc = MLPDiscriminator(
            non_linearity=self.config['non_linearity'],
            hidden_dims=self.config['nn_disc']
        )
        return disc.to(self.device)
    
    def _create_predictor(self):
        """Create predictor for testing."""
        from source.nn import MLPDiscriminator
        pred = MLPDiscriminator(
            non_linearity=self.config['non_linearity'],
            hidden_dims=self.config['nn_validator']
        )
        return pred.to(self.device)


@pytest.mark.performance
@pytest.mark.slow
class TestQuantumClassicalPerformance:
    """Test suite for quantum vs classical performance comparison."""
    
    def setup_method(self):
        """Set up test environment."""
        self.generator_types = ['classical_normal', 'classical_uniform']
        
        # Add quantum generators if available
        try:
            import pennylane as qml
            self.generator_types.extend(['quantum_samples', 'quantum_shadows'])
        except ImportError:
            print("PennyLane not available, skipping quantum generators")
    
    def test_forward_pass_performance(self, device, test_config, gaussian_mixture_params):
        """Test forward pass performance across different batch sizes."""
        benchmark = QuantumClassicalBenchmark(device, test_config, gaussian_mixture_params)
        batch_sizes = [32, 64, 128, 256] if device.type == 'cuda' else [32, 64]
        
        results = benchmark.benchmark_forward_pass(
            self.generator_types, batch_sizes, n_iterations=5
        )
        
        # Validate results
        for gen_type in self.generator_types:
            if 'error' not in results[gen_type]:
                assert len(results[gen_type]) == len(batch_sizes)
                
                for batch_size in batch_sizes:
                    metrics = results[gen_type][batch_size]
                    assert metrics['mean_time'] > 0
                    assert metrics['samples_per_second'] > 0
                    assert metrics['std_time'] >= 0
        
        # Print summary
        print("\n=== Forward Pass Performance Summary ===")
        for gen_type, data in results.items():
            if 'error' not in data:
                print(f"\n{gen_type.upper()}:")
                for batch_size, metrics in data.items():
                    print(f"  Batch {batch_size}: {metrics['samples_per_second']:.1f} samples/sec, "
                          f"{metrics['mean_time']*1000:.1f}ms")
    
    def test_memory_usage(self, device, test_config, gaussian_mixture_params):
        """Test memory usage across different batch sizes."""
        if device.type != 'cuda':
            pytest.skip("GPU not available for memory testing")
        
        benchmark = QuantumClassicalBenchmark(device, test_config, gaussian_mixture_params)
        batch_sizes = [64, 128, 256, 512]
        
        results = benchmark.benchmark_memory_usage(self.generator_types, batch_sizes)
        
        # Validate results
        for gen_type in self.generator_types:
            if 'error' not in results[gen_type]:
                for batch_size in batch_sizes:
                    metrics = results[gen_type][batch_size]
                    assert metrics['memory_increase_mb'] >= 0
                    assert metrics['memory_per_sample_kb'] >= 0
        
        # Print summary
        print("\n=== Memory Usage Summary ===")
        for gen_type, data in results.items():
            if 'error' not in data:
                print(f"\n{gen_type.upper()}:")
                for batch_size, metrics in data.items():
                    print(f"  Batch {batch_size}: {metrics['memory_increase_mb']:.1f} MB, "
                          f"{metrics['memory_per_sample_kb']:.2f} KB/sample")
    
    def test_generation_quality(self, device, test_config, gaussian_mixture_params):
        """Test generation quality consistency across runs."""
        benchmark = QuantumClassicalBenchmark(device, test_config, gaussian_mixture_params)
        
        results = benchmark.benchmark_generation_quality(
            self.generator_types, n_samples=500, n_runs=3
        )
        
        # Validate results
        for gen_type in self.generator_types:
            if 'error' not in results[gen_type]:
                data = results[gen_type]
                
                # Check that we have quality metrics
                assert 'kl_divergence' in data
                assert 'log_likelihood' in data
                assert 'positive_ratio' in data
                
                # Check that metrics are reasonable
                assert not np.isnan(data['kl_divergence']['mean'])
                assert not np.isnan(data['log_likelihood']['mean'])
                assert 0 <= data['positive_ratio']['mean'] <= 1
        
        # Print summary
        print("\n=== Generation Quality Summary ===")
        for gen_type, data in results.items():
            if 'error' not in data:
                print(f"\n{gen_type.upper()}:")
                print(f"  KL Divergence: {data['kl_divergence']['mean']:.3f} ± {data['kl_divergence']['std']:.3f}")
                print(f"  Log Likelihood: {data['log_likelihood']['mean']:.3f} ± {data['log_likelihood']['std']:.3f}")
                print(f"  Positive Ratio: {data['positive_ratio']['mean']:.3f} ± {data['positive_ratio']['std']:.3f}")
    
    @pytest.mark.slow
    def test_training_convergence(self, device, test_config, gaussian_mixture_params):
        """Test training convergence speed comparison."""
        benchmark = QuantumClassicalBenchmark(device, test_config, gaussian_mixture_params)
        
        # Use shorter training for testing
        results = benchmark.benchmark_training_convergence(
            self.generator_types, target_kl=2.0, max_epochs=10
        )
        
        # Validate results
        for gen_type in self.generator_types:
            if 'error' not in results[gen_type]:
                data = results[gen_type]
                assert data['convergence_epoch'] > 0
                assert data['mean_epoch_time'] > 0
                assert data['total_time'] > 0
                assert len(data['kl_history']) > 0
        
        # Print summary
        print("\n=== Training Convergence Summary ===")
        for gen_type, data in results.items():
            if 'error' not in data:
                print(f"\n{gen_type.upper()}:")
                print(f"  Convergence Epoch: {data['convergence_epoch']}")
                print(f"  Final KL: {data['final_kl']:.3f}")
                print(f"  Time to Convergence: {data['time_to_convergence']:.1f}s")
                print(f"  Mean Epoch Time: {data['mean_epoch_time']:.2f}s")
    
    def test_performance_comparison_statistical_significance(self, device, test_config, 
                                                           gaussian_mixture_params, 
                                                           statistical_tester):
        """Test statistical significance of performance differences."""
        if len(self.generator_types) < 2:
            pytest.skip("Need at least 2 generator types for comparison")
        
        benchmark = QuantumClassicalBenchmark(device, test_config, gaussian_mixture_params)
        
        # Run quality benchmark multiple times for statistical analysis
        all_results = {}
        n_runs = 5
        
        for run in range(n_runs):
            run_results = benchmark.benchmark_generation_quality(
                self.generator_types, n_samples=200, n_runs=1
            )
            
            for gen_type in self.generator_types:
                if 'error' not in run_results[gen_type]:
                    if gen_type not in all_results:
                        all_results[gen_type] = defaultdict(list)
                    
                    for metric, value in run_results[gen_type].items():
                        all_results[gen_type][metric].append(value['mean'])
        
        # Statistical comparison between first two available generators
        available_gens = [gen for gen in self.generator_types if gen in all_results]
        if len(available_gens) >= 2:
            gen1, gen2 = available_gens[:2]
            
            print(f"\n=== Statistical Comparison: {gen1} vs {gen2} ===")
            
            for metric in ['kl_divergence', 'log_likelihood']:
                if metric in all_results[gen1] and metric in all_results[gen2]:
                    values1 = all_results[gen1][metric]
                    values2 = all_results[gen2][metric]
                    
                    # Calculate effect size
                    effect_size = statistical_tester.calculate_effect_size(values1, values2)
                    
                    # Perform t-test
                    from scipy import stats
                    t_stat, p_value = stats.ttest_ind(values1, values2)
                    
                    print(f"\n{metric.upper()}:")
                    print(f"  {gen1}: {np.mean(values1):.3f} ± {np.std(values1):.3f}")
                    print(f"  {gen2}: {np.mean(values2):.3f} ± {np.std(values2):.3f}")
                    print(f"  Effect size (Cohen's d): {effect_size:.3f}")
                    print(f"  T-test p-value: {p_value:.4f}")
                    print(f"  Significant difference (p<0.05): {p_value < 0.05}")


@pytest.mark.performance
def test_save_benchmark_results(device, test_config, gaussian_mixture_params, temp_directory):
    """Save benchmark results for later analysis."""
    benchmark = QuantumClassicalBenchmark(device, test_config, gaussian_mixture_params)
    generator_types = ['classical_normal']
    
    try:
        import pennylane as qml
        generator_types.append('quantum_samples')
    except ImportError:
        pass
    
    # Run benchmarks
    forward_results = benchmark.benchmark_forward_pass(
        generator_types, [32, 64], n_iterations=3
    )
    
    quality_results = benchmark.benchmark_generation_quality(
        generator_types, n_samples=200, n_runs=2
    )
    
    # Combine results
    full_results = {
        'forward_pass': forward_results,
        'generation_quality': quality_results,
        'test_config': test_config,
        'device': str(device)
    }
    
    # Save results
    results_file = temp_directory / 'benchmark_results.json'
    
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
    
    json_results = convert_numpy(full_results)
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    assert results_file.exists()
    
    # Verify we can load the results
    with open(results_file, 'r') as f:
        loaded_results = json.load(f)
    
    assert 'forward_pass' in loaded_results
    assert 'generation_quality' in loaded_results
    print(f"Benchmark results saved to: {results_file}")


if __name__ == "__main__":
    # Allow running this module directly for benchmarking
    pytest.main([__file__ + "::TestQuantumClassicalPerformance", "-v", "-s"])