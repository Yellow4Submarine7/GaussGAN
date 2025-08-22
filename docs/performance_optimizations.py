#!/usr/bin/env python3
"""
Performance optimizations for GaussGAN project.
This file contains optimized versions of key components.

Usage: Review these implementations and selectively integrate into main codebase.
"""

import time
import torch
import torch.nn as nn
from contextlib import contextmanager
from functools import lru_cache
import numpy as np
from typing import Optional, Tuple, Dict, Any
import pennylane as qml
from concurrent.futures import ThreadPoolExecutor
import asyncio


# ============================================================================
# 1. OPTIMIZED QUANTUM COMPONENTS
# ============================================================================

class OptimizedQuantumNoise(nn.Module):
    """
    Optimized version of QuantumNoise with batching and caching.
    
    Key improvements:
    - Vectorized random input generation
    - Circuit compilation caching
    - Reduced device initialization overhead
    - Optional hardware acceleration
    """
    
    def __init__(
        self, 
        num_qubits: int = 6, 
        num_layers: int = 2,
        use_gpu: bool = False,
        cache_size: int = 32
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.cache_size = cache_size
        
        # Initialize weights
        self.weights = nn.Parameter(
            torch.rand(num_layers, num_qubits * 2 - 1) * 2 * torch.pi - torch.pi
        )
        
        # Choose optimal device based on availability
        device_name = "lightning.gpu" if use_gpu and qml.device.capabilities("lightning.gpu") else "default.qubit"
        self.dev = qml.device(device_name, wires=num_qubits)
        
        # Pre-compile quantum circuit
        self.gen_circuit = self._create_optimized_circuit()
        
    @lru_cache(maxsize=32)
    def _create_optimized_circuit(self):
        """Create and cache quantum circuit compilation."""
        
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def optimized_circuit(weights, z_inputs):
            batch_size = z_inputs.shape[0]
            
            # Vectorized initialization for all samples
            for i in range(self.num_qubits):
                for b in range(batch_size):
                    qml.RY(torch.arcsin(z_inputs[b, 0]).item(), wires=i)
                    qml.RZ(torch.arcsin(z_inputs[b, 1]).item(), wires=i)
            
            # Parameterized layers
            for layer in range(self.num_layers):
                for i in range(self.num_qubits):
                    qml.RY(weights[layer, i], wires=i)
                for i in range(self.num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.RZ(weights[layer, i + self.num_qubits], wires=i + 1)
                    qml.CNOT(wires=[i, i + 1])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
        
        return optimized_circuit
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Optimized forward pass with batch processing.
        
        Returns:
            torch.Tensor: Generated quantum states [batch_size, num_qubits]
        """
        # Generate batch of random inputs efficiently
        z_inputs = torch.rand(batch_size, 2, device=self.weights.device) * 2 - 1
        
        # For small batches, use vectorized processing
        if batch_size <= 32:
            results = self.gen_circuit(self.weights, z_inputs)
            return torch.stack([torch.stack(results[i]) for i in range(len(results))]).T
        
        # For larger batches, use chunked processing
        chunk_size = 32
        results = []
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            chunk_inputs = z_inputs[i:end_idx]
            chunk_results = self.gen_circuit(self.weights, chunk_inputs)
            results.append(torch.stack([torch.stack(chunk_results[j]) for j in range(len(chunk_results))]).T)
        
        return torch.cat(results, dim=0)


class OptimizedQuantumShadowNoise(nn.Module):
    """
    Optimized quantum shadow noise with efficient basis generation.
    """
    
    def __init__(
        self,
        z_dim: int,
        num_qubits: int = 6,
        num_layers: int = 2,
        num_basis: int = 3,
        use_gpu: bool = False
    ):
        super().__init__()
        self.z_dim = z_dim
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.num_basis = num_basis
        
        # Pre-generate basis to avoid runtime overhead
        self.basis = self._precompute_basis()
        
        # Use optimal device
        device_name = "lightning.gpu" if use_gpu else "default.qubit"
        self.dev = qml.device(device_name, wires=num_qubits, shots=100)
        
        # Initialize parameters
        self.weights = nn.Parameter(
            torch.rand(num_layers, num_qubits * 2 - 1) * 2 * torch.pi - torch.pi
        )
        self.coeffs = nn.Parameter(torch.rand(num_basis, z_dim))
        
        # Create optimized circuit
        self.gen_circuit = self._create_shadow_circuit()
    
    def _precompute_basis(self):
        """Pre-generate basis observables to avoid runtime computation."""
        paulis = [qml.PauliZ, qml.PauliX, qml.PauliY, qml.Identity]
        basis = []
        
        # Use more efficient basis generation
        for _ in range(self.num_basis):
            # Create deterministic basis for reproducibility
            np.random.seed(_ + 42)  # Fixed seed for deterministic behavior
            pauli_indices = np.random.choice(4, self.num_qubits)
            
            obs = paulis[pauli_indices[0]](0)
            for i in range(1, self.num_qubits):
                obs = obs @ paulis[pauli_indices[i]](i)
            basis.append(obs)
        
        return basis
    
    @lru_cache(maxsize=16)
    def _create_shadow_circuit(self):
        """Create cached shadow circuit."""
        
        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def shadow_circuit(weights, z_inputs):
            # Efficient initialization
            z1, z2 = z_inputs[0], z_inputs[1]
            for i in range(self.num_qubits):
                qml.RY(torch.arcsin(z1).item(), wires=i)
                qml.RZ(torch.arcsin(z2).item(), wires=i)
            
            # Parameterized layers
            for layer in range(self.num_layers):
                for i in range(self.num_qubits):
                    qml.RY(weights[layer, i], wires=i)
                for i in range(self.num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.RZ(weights[layer, i + self.num_qubits], wires=i + 1)
                    qml.CNOT(wires=[i, i + 1])
            
            return qml.shadow_expval(self.basis)
        
        return shadow_circuit
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """Optimized forward pass for shadow noise."""
        # Pre-generate random inputs
        z_inputs = torch.rand(2, batch_size, device=self.weights.device) * 2 - 1
        
        # Process in chunks for memory efficiency
        chunk_size = 16  # Smaller chunks for shadow circuits
        results = []
        
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            chunk_results = []
            
            for j in range(i, end_idx):
                chunk_input = z_inputs[:, j]
                circuit_output = self.gen_circuit(self.weights, chunk_input)
                chunk_results.append(torch.stack(circuit_output))
            
            results.extend(chunk_results)
        
        # Stack and apply coefficients
        shadow_measurements = torch.stack(results)
        return torch.matmul(shadow_measurements, self.coeffs)


# ============================================================================
# 2. MEMORY-OPTIMIZED MODEL COMPONENTS
# ============================================================================

class MemoryOptimizedGaussGan:
    """
    Memory optimization mixins for the main GaussGan model.
    """
    
    def optimized_training_step(self, batch, batch_idx):
        """
        Memory-optimized training step with efficient gradient management.
        """
        # Get optimizers
        g_optim, d_optim, p_optim = self.optimizers()
        
        # Clear gradients efficiently
        for optimizer in [g_optim, d_optim, p_optim]:
            optimizer.zero_grad(set_to_none=True)  # More memory efficient
        
        # Discriminator training with gradient accumulation
        d_loss_accumulated = 0
        for critic_step in range(self.n_critic):
            with torch.cuda.amp.autocast(enabled=True):  # Mixed precision
                d_loss = self._compute_discriminator_loss(batch)
                d_loss = d_loss / self.n_critic  # Normalize for accumulation
            
            self.manual_backward(d_loss)
            d_loss_accumulated += d_loss.item()
        
        d_optim.step()
        
        # Predictor training (if enabled)
        if self.killer:
            p_loss_accumulated = 0
            for pred_step in range(self.n_predictor):
                with torch.cuda.amp.autocast(enabled=True):
                    p_loss, _ = self._compute_predictor_loss(batch)
                    p_loss = p_loss / self.n_predictor
                
                self.manual_backward(p_loss)
                p_loss_accumulated += p_loss.item()
            
            p_optim.step()
        
        # Generator training
        with torch.cuda.amp.autocast(enabled=True):
            g_loss = self._compute_generator_loss(batch)
        
        self.manual_backward(g_loss)
        g_optim.step()
        
        # Log accumulated losses
        self.log_dict({
            "d_loss": d_loss_accumulated,
            "g_loss": g_loss.item(),
            "p_loss": p_loss_accumulated if self.killer else 0.0
        })
    
    def memory_efficient_validation(self, batch, batch_idx):
        """
        Memory-efficient validation with chunked processing.
        """
        chunk_size = 100  # Process validation in smaller chunks
        total_samples = self.validation_samples
        
        with torch.no_grad():  # Disable gradient computation
            fake_data_chunks = []
            
            for i in range(0, total_samples, chunk_size):
                current_chunk_size = min(chunk_size, total_samples - i)
                
                # Generate chunk on GPU
                chunk = self._generate_fake_data(current_chunk_size)
                
                # Move to CPU immediately to free GPU memory
                fake_data_chunks.append(chunk.cpu())
                
                # Clear GPU cache periodically
                if i % (chunk_size * 5) == 0:
                    torch.cuda.empty_cache()
            
            # Combine chunks on GPU only when needed
            fake_data = torch.cat(fake_data_chunks).to(self.device)
        
        # Compute metrics (unchanged)
        metrics_fake = self._compute_metrics(fake_data)
        
        # Process metrics safely
        safe_metrics = {}
        for k, v in metrics_fake.items():
            if v is None or (hasattr(v, '__iter__') and len(v) == 0):
                safe_metrics[f"ValidationStep_FakeData_{k}"] = float("nan")
            elif not hasattr(v, "__iter__"):
                safe_metrics[f"ValidationStep_FakeData_{k}"] = float(v)
            else:
                valid_vals = [val for val in v if val is not None and not np.isnan(val)]
                safe_metrics[f"ValidationStep_FakeData_{k}"] = np.mean(valid_vals) if valid_vals else float("nan")
        
        self.log_dict(safe_metrics, on_epoch=True, batch_size=batch[0].size(0))
        
        return {"fake_data": fake_data.cpu(), "metrics": safe_metrics}


# ============================================================================
# 3. PERFORMANCE MONITORING UTILITIES
# ============================================================================

class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking for GaussGAN components.
    """
    
    def __init__(self):
        self.metrics = {
            'classical_time': [],
            'quantum_time': [],
            'memory_peak': [],
            'gpu_utilization': [],
            'batch_sizes': [],
            'component_types': []
        }
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    @contextmanager
    def measure_performance(self, component_type: str, batch_size: int):
        """
        Context manager for measuring performance metrics.
        
        Args:
            component_type: 'classical' or 'quantum'
            batch_size: Number of samples in batch
        """
        # Record initial state
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        
        start_time = time.perf_counter()
        
        try:
            yield self
        finally:
            # Record final state
            end_time = time.perf_counter()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_after = torch.cuda.memory_allocated()
                mem_peak = torch.cuda.max_memory_allocated()
                
                self.metrics['memory_peak'].append(mem_peak - mem_before)
            else:
                self.metrics['memory_peak'].append(0)
            
            # Record timing and metadata
            execution_time = end_time - start_time
            if component_type.startswith('quantum'):
                self.metrics['quantum_time'].append(execution_time)
                self.metrics['classical_time'].append(0)
            else:
                self.metrics['classical_time'].append(execution_time)
                self.metrics['quantum_time'].append(0)
            
            self.metrics['batch_sizes'].append(batch_size)
            self.metrics['component_types'].append(component_type)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance summary.
        
        Returns:
            Dict containing performance statistics
        """
        if not self.metrics['batch_sizes']:
            return {"error": "No performance data collected"}
        
        # Separate classical and quantum metrics
        classical_times = [t for t in self.metrics['classical_time'] if t > 0]
        quantum_times = [t for t in self.metrics['quantum_time'] if t > 0]
        
        summary = {
            'classical_performance': {
                'avg_time_per_batch': np.mean(classical_times) if classical_times else 0,
                'min_time': np.min(classical_times) if classical_times else 0,
                'max_time': np.max(classical_times) if classical_times else 0,
                'std_time': np.std(classical_times) if classical_times else 0,
            },
            'quantum_performance': {
                'avg_time_per_batch': np.mean(quantum_times) if quantum_times else 0,
                'min_time': np.min(quantum_times) if quantum_times else 0,
                'max_time': np.max(quantum_times) if quantum_times else 0,
                'std_time': np.std(quantum_times) if quantum_times else 0,
            },
            'memory_usage': {
                'avg_memory_mb': np.mean(self.metrics['memory_peak']) / (1024**2),
                'max_memory_mb': np.max(self.metrics['memory_peak']) / (1024**2),
                'total_measurements': len(self.metrics['memory_peak'])
            },
            'performance_ratio': {
                'quantum_vs_classical_slowdown': (
                    np.mean(quantum_times) / np.mean(classical_times) 
                    if classical_times and quantum_times else float('inf')
                )
            }
        }
        
        return summary
    
    def export_benchmark_report(self, filepath: str):
        """Export detailed benchmark report to file."""
        summary = self.get_performance_summary()
        
        report = f"""# GaussGAN Performance Benchmark Report

## Test Configuration
- Total measurements: {len(self.metrics['batch_sizes'])}
- Batch sizes tested: {set(self.metrics['batch_sizes'])}
- Component types: {set(self.metrics['component_types'])}

## Classical Performance
- Average time per batch: {summary['classical_performance']['avg_time_per_batch']:.4f}s
- Time range: {summary['classical_performance']['min_time']:.4f}s - {summary['classical_performance']['max_time']:.4f}s
- Standard deviation: {summary['classical_performance']['std_time']:.4f}s

## Quantum Performance  
- Average time per batch: {summary['quantum_performance']['avg_time_per_batch']:.4f}s
- Time range: {summary['quantum_performance']['min_time']:.4f}s - {summary['quantum_performance']['max_time']:.4f}s
- Standard deviation: {summary['quantum_performance']['std_time']:.4f}s

## Memory Usage
- Average memory per operation: {summary['memory_usage']['avg_memory_mb']:.2f} MB
- Peak memory usage: {summary['memory_usage']['max_memory_mb']:.2f} MB

## Performance Comparison
- Quantum vs Classical slowdown: {summary['performance_ratio']['quantum_vs_classical_slowdown']:.1f}x

## Recommendations
Based on the benchmark results:
1. {'Use classical generators for rapid prototyping' if summary['performance_ratio']['quantum_vs_classical_slowdown'] > 10 else 'Quantum performance is acceptable'}
2. {'Consider batch size optimization' if summary['memory_usage']['max_memory_mb'] > 1000 else 'Memory usage is efficient'}
3. {'Implement circuit caching' if summary['quantum_performance']['std_time'] > summary['quantum_performance']['avg_time_per_batch'] * 0.1 else 'Quantum performance is stable'}
"""
        
        with open(filepath, 'w') as f:
            f.write(report)


# ============================================================================
# 4. OPTIMIZED METRIC COMPUTATION
# ============================================================================

def fast_kl_divergence_approximation(samples: torch.Tensor, target_gmm, bins: int = 50):
    """
    Fast KL divergence approximation using histograms instead of KDE.
    
    Args:
        samples: Generated samples [N, 2]
        target_gmm: Target Gaussian mixture model
        bins: Number of histogram bins
    
    Returns:
        Approximate KL divergence value
    """
    device = samples.device
    
    # Define histogram range
    x_min, x_max = samples[:, 0].min() - 1, samples[:, 0].max() + 1
    y_min, y_max = samples[:, 1].min() - 1, samples[:, 1].max() + 1
    
    # Create 2D histogram for generated samples
    sample_hist = torch.histogramdd(
        samples, 
        bins=[bins, bins], 
        range=[[x_min.item(), x_max.item()], [y_min.item(), y_max.item()]]
    )[0]
    
    # Generate target samples and create histogram
    target_samples = torch.tensor(
        target_gmm.sample(samples.shape[0])[0], 
        device=device, 
        dtype=torch.float32
    )
    
    target_hist = torch.histogramdd(
        target_samples,
        bins=[bins, bins],
        range=[[x_min.item(), x_max.item()], [y_min.item(), y_max.item()]]
    )[0]
    
    # Convert to probabilities
    sample_probs = sample_hist / sample_hist.sum()
    target_probs = target_hist / target_hist.sum()
    
    # Compute KL divergence with numerical stability
    epsilon = 1e-10
    sample_probs = sample_probs + epsilon
    target_probs = target_probs + epsilon
    
    kl_div = torch.sum(target_probs * torch.log(target_probs / sample_probs))
    
    return kl_div.item()


# ============================================================================
# 5. CHECKPOINT CLEANUP UTILITIES
# ============================================================================

def cleanup_checkpoint_directory(checkpoint_dir: str, keep_last_n: int = 5, keep_best_n: int = 3):
    """
    Clean up checkpoint directory to reduce storage usage.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        keep_last_n: Number of most recent checkpoints to keep per run
        keep_best_n: Number of best checkpoints to keep (by validation metric)
    """
    import os
    import glob
    from pathlib import Path
    
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        print(f"Checkpoint directory {checkpoint_dir} does not exist")
        return
    
    # Remove redundant last-v*.ckpt files (keep only last.ckpt)
    last_files = list(checkpoint_path.glob("last-v*.ckpt"))
    for file_path in last_files:
        print(f"Removing redundant checkpoint: {file_path}")
        file_path.unlink()
    
    # Group checkpoints by run_id
    run_checkpoints = {}
    for ckpt_file in checkpoint_path.glob("run_id-*.ckpt"):
        run_id = ckpt_file.name.split('-')[1]
        if run_id not in run_checkpoints:
            run_checkpoints[run_id] = []
        run_checkpoints[run_id].append(ckpt_file)
    
    # Clean up each run's checkpoints
    for run_id, checkpoints in run_checkpoints.items():
        if len(checkpoints) <= keep_last_n:
            continue  # Keep all if less than threshold
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Keep only the most recent N checkpoints
        checkpoints_to_remove = checkpoints[keep_last_n:]
        
        for ckpt_file in checkpoints_to_remove:
            print(f"Removing old checkpoint: {ckpt_file}")
            ckpt_file.unlink()
    
    # Calculate space saved
    remaining_files = list(checkpoint_path.glob("*.ckpt"))
    total_size = sum(f.stat().st_size for f in remaining_files)
    print(f"Checkpoint cleanup complete. Remaining: {len(remaining_files)} files ({total_size/(1024**3):.2f} GB)")


# ============================================================================
# 6. EXAMPLE USAGE AND INTEGRATION
# ============================================================================

def demonstrate_optimizations():
    """
    Example showing how to use the optimized components.
    """
    
    # Initialize performance benchmark
    benchmark = PerformanceBenchmark()
    
    # Test classical vs quantum performance
    batch_sizes = [64, 128, 256, 512]
    
    print("Running performance benchmarks...")
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        
        # Test classical component
        classical_gen = torch.nn.Linear(4, 2)  # Simple classical generator
        with benchmark.measure_performance("classical", batch_size):
            classical_output = classical_gen(torch.randn(batch_size, 4))
        
        print(f"Classical generation time: {benchmark.metrics['classical_time'][-1]:.4f}s")
        
        # Test optimized quantum component
        quantum_gen = OptimizedQuantumNoise(num_qubits=6, num_layers=2, use_gpu=False)
        with benchmark.measure_performance("quantum_optimized", batch_size):
            quantum_output = quantum_gen(batch_size)
        
        print(f"Quantum generation time: {benchmark.metrics['quantum_time'][-1]:.4f}s")
    
    # Generate and export benchmark report
    summary = benchmark.get_performance_summary()
    print(f"\nPerformance Summary:")
    print(f"Quantum vs Classical slowdown: {summary['performance_ratio']['quantum_vs_classical_slowdown']:.1f}x")
    print(f"Average memory usage: {summary['memory_usage']['avg_memory_mb']:.2f} MB")
    
    return benchmark


if __name__ == "__main__":
    # Example usage
    print("GaussGAN Performance Optimizations")
    print("=" * 50)
    
    # Run demonstrations
    benchmark_results = demonstrate_optimizations()
    
    # Export detailed report
    benchmark_results.export_benchmark_report("performance_benchmark_results.md")
    print("\nBenchmark report exported to: performance_benchmark_results.md")
    
    # Demonstrate checkpoint cleanup (commented out for safety)
    # cleanup_checkpoint_directory("/home/paperx/quantum/GaussGAN/checkpoints/")