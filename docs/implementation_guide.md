# GaussGAN Performance Optimization Implementation Guide

## Overview

This guide provides step-by-step instructions for implementing the performance optimizations identified in the analysis. Each optimization is prioritized by impact and implementation difficulty.

## Priority 1: Critical Storage Issues (Immediate Action Required)

### Checkpoint Cleanup
**Impact**: Free 3.5-4GB disk space immediately
**Difficulty**: Low
**Implementation Time**: 5 minutes

```bash
# WARNING: This will delete old checkpoints. Backup important ones first!

# 1. Navigate to project directory
cd /home/paperx/quantum/GaussGAN

# 2. Count current checkpoints
echo "Current checkpoint count: $(ls checkpoints/ | wc -l)"
echo "Current checkpoint size: $(du -sh checkpoints/)"

# 3. Remove redundant last-v*.ckpt files (keep only last.ckpt)
rm checkpoints/last-v*.ckpt

# 4. Keep only last 5 checkpoints per run_id (manual selection recommended)
# List runs to identify which ones to keep:
ls checkpoints/run_id-* | cut -d'-' -f2 | sort -u

# Example cleanup for old runs (adjust run_ids as needed):
# find checkpoints/ -name "run_id-OLD_RUN_ID-*" -type f | head -n -5 | xargs rm

echo "After cleanup:"
echo "Remaining checkpoints: $(ls checkpoints/ | wc -l)"
echo "Remaining size: $(du -sh checkpoints/)"
```

### Immediate Configuration Updates

Update `config.yaml` with optimized settings:

```yaml
# Performance-optimized config.yaml additions
batch_size: 512          # Increased from 256 for better GPU utilization
validation_samples: 200  # Reduced from 500 for faster validation

# Memory optimization
grad_penalty: 0.1       # Reduced from 0.2 for less memory pressure
n_critic: 3            # Reduced from 5 for faster training
n_predictor: 2         # Reduced from 5 for efficiency

# Checkpoint optimization
checkpoint_frequency: 10  # Save every 10 epochs instead of 5
keep_last_n: 3           # Keep only last 3 checkpoints
```

## Priority 2: Memory and GPU Optimizations (High Impact)

### Enable Mixed Precision Training

**File**: `main.py`
**Lines to modify**: Trainer initialization (~line 161)

```python
# Replace the existing trainer configuration:
trainer = Trainer(
    max_epochs=final_args["max_epochs"],
    accelerator=final_args["accelerator"],
    logger=mlflow_logger,
    log_every_n_steps=5,
    limit_val_batches=2,
    callbacks=[checkpoint_callback],
    # ADD THESE OPTIMIZATIONS:
    precision=16,                    # Enable mixed precision
    accumulate_grad_batches=2,       # Gradient accumulation
    gradient_clip_val=1.0,           # Prevent gradient explosion
    enable_checkpointing=True,
    enable_model_summary=False,      # Reduce startup overhead
)
```

### Optimize DataLoader Settings

**File**: `source/data.py`
**Lines to modify**: DataLoader configurations (~lines 52-76)

```python
def train_dataloader(self):
    return DataLoader(
        self.train_dataset,
        batch_size=self.hparams.batch_size,
        # OPTIMIZED SETTINGS:
        num_workers=4,               # Parallel data loading
        pin_memory=True,            # Faster GPU transfer
        persistent_workers=True,     # Keep workers alive
        drop_last=True,             # Consistent batch sizes
        prefetch_factor=2,          # Prefetch batches
    )

def val_dataloader(self):
    return DataLoader(
        self.val_dataset,
        batch_size=self.hparams.batch_size,
        # OPTIMIZED SETTINGS:
        num_workers=2,               # Fewer workers for validation
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,            # Keep all validation data
    )
```

### Memory-Efficient Validation

**File**: `source/model.py`
**Function**: `validation_step` (~line 94)

```python
def validation_step(self, batch, batch_idx):
    # MEMORY-OPTIMIZED VALIDATION
    chunk_size = 100  # Process in chunks
    total_samples = self.validation_samples
    
    with torch.no_grad():  # Disable gradient computation
        fake_data_chunks = []
        
        for i in range(0, total_samples, chunk_size):
            current_chunk_size = min(chunk_size, total_samples - i)
            
            # Generate chunk
            chunk = self._generate_fake_data(current_chunk_size)
            
            # Move to CPU immediately to free GPU memory
            fake_data_chunks.append(chunk.cpu())
            
            # Clear GPU cache periodically
            if i % (chunk_size * 5) == 0:
                torch.cuda.empty_cache()
        
        # Combine chunks on GPU only when needed for metrics
        fake_data = torch.cat(fake_data_chunks).to(self.device)
    
    # Rest of validation logic remains the same...
    metrics_fake = self._compute_metrics(fake_data)
    # ... (existing code)
```

## Priority 3: Training Speed Optimizations (Medium Impact)

### Optimize Training Step with Gradient Management

**File**: `source/model.py`
**Function**: `training_step` (~line 69)

```python
def training_step(self, batch, batch_idx):
    # Get optimizers
    g_optim, d_optim, p_optim = self.optimizers()
    
    # OPTIMIZED GRADIENT CLEARING
    for optimizer in [g_optim, d_optim, p_optim]:
        optimizer.zero_grad(set_to_none=True)  # More memory efficient
    
    # Train discriminator with mixed precision
    d_loss_total = 0
    for critic_step in range(self.n_critic):
        # Use autocast for mixed precision
        with torch.cuda.amp.autocast(enabled=True):
            d_loss = self._compute_discriminator_loss(batch)
            d_loss = d_loss / self.n_critic  # Normalize for accumulation
        
        self.manual_backward(d_loss)
        d_loss_total += d_loss.item()
    
    d_optim.step()
    
    # Train predictor (if enabled)
    if self.killer:
        p_loss_total = 0
        for pred_step in range(self.n_predictor):
            with torch.cuda.amp.autocast(enabled=True):
                p_loss, _ = self._compute_predictor_loss(batch)
                p_loss = p_loss / self.n_predictor
            
            self.manual_backward(p_loss)
            p_loss_total += p_loss.item()
        
        p_optim.step()
    
    # Train generator
    with torch.cuda.amp.autocast(enabled=True):
        g_loss = self._compute_generator_loss(batch)
    
    self.manual_backward(g_loss)
    g_optim.step()
    
    # Log losses efficiently
    self.log_dict({
        "d_loss": d_loss_total,
        "g_loss": g_loss.item(),
        "p_loss": p_loss_total if self.killer else 0.0
    }, on_step=True, on_epoch=False)
```

### Reduce Metric Computation Frequency

**File**: `source/model.py`
**Function**: `validation_step` (~line 94)

```python
def validation_step(self, batch, batch_idx):
    fake_data = self._generate_fake_data(self.validation_samples).detach()
    
    # CONDITIONAL METRIC COMPUTATION
    # Compute expensive metrics less frequently
    if self.current_epoch % 5 == 0 or self.current_epoch < 10:
        # Full validation every 5 epochs or during early training
        metrics_fake = self._compute_metrics(fake_data)
    else:
        # Lightweight validation - only position metric
        metrics_fake = {
            'IsPositive': self._compute_single_metric('IsPositive', fake_data)
        }
    
    # Rest of the function remains the same...
```

## Priority 4: Quantum Circuit Optimizations (Research Impact)

### Optimize Quantum Noise Generation

**File**: `source/nn.py`
**Class**: `QuantumNoise` (~line 12)

```python
class QuantumNoise(nn.Module):
    def __init__(self, num_qubits: int = 6, num_layers: int = 2, use_caching: bool = True):
        super(QuantumNoise, self).__init__()
        
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.use_caching = use_caching
        
        # OPTIMIZED WEIGHT INITIALIZATION
        self.weights = nn.Parameter(
            torch.rand(num_layers, (num_qubits * 2 - 1)) * 2 * torch.pi - torch.pi
        )
        
        # USE OPTIMAL DEVICE
        try:
            # Try GPU-accelerated quantum simulator if available
            dev = qml.device("lightning.gpu", wires=num_qubits)
            print("Using GPU-accelerated quantum simulator")
        except:
            # Fall back to CPU simulator
            dev = qml.device("default.qubit", wires=num_qubits)
            print("Using CPU quantum simulator")
        
        # Create quantum node with optimized settings
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def gen_circuit(w):
            # OPTIMIZED RANDOM INPUT GENERATION
            # Pre-generate random values to avoid repeated calls
            z1, z2 = torch.rand(2, device=w.device) * 2 - 1
            
            # Initialize qubits efficiently
            for i in range(num_qubits):
                qml.RY(torch.arcsin(z1).item(), wires=i)
                qml.RZ(torch.arcsin(z2).item(), wires=i)
            
            # Parameterized layers (unchanged)
            for l in range(num_layers):
                for i in range(num_qubits):
                    qml.RY(w[l][i], wires=i)
                for i in range(num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.RZ(w[l][i + num_qubits], wires=i + 1)
                    qml.CNOT(wires=[i, i + 1])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
        
        self.gen_circuit = gen_circuit
    
    def forward(self, batch_size: int):
        # BATCHED PROCESSING FOR EFFICIENCY
        if batch_size <= 32:
            # Small batches: process normally
            sample_list = [
                torch.concat([tensor.unsqueeze(0) for tensor in self.gen_circuit(self.weights)])
                for _ in range(batch_size)
            ]
        else:
            # Large batches: process in chunks
            chunk_size = 32
            sample_list = []
            for i in range(0, batch_size, chunk_size):
                chunk_samples = [
                    torch.concat([tensor.unsqueeze(0) for tensor in self.gen_circuit(self.weights)])
                    for _ in range(min(chunk_size, batch_size - i))
                ]
                sample_list.extend(chunk_samples)
        
        noise = torch.stack(tuple(sample_list)).float()
        return noise
```

### Optimize Quantum Shadow Noise

**File**: `source/nn.py`  
**Class**: `QuantumShadowNoise` (~line 60)

```python
class QuantumShadowNoise(nn.Module):
    def __init__(self, z_dim: int, *, num_qubits: int = 6, num_layers: int = 2, num_basis: int = 3):
        super(QuantumShadowNoise, self).__init__()
        
        self.z_dim = z_dim
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.num_basis = num_basis
        
        # PRECOMPUTE BASIS TO AVOID RUNTIME OVERHEAD
        self.basis = self._precompute_optimal_basis()
        
        # Use optimal device with reduced shots for speed
        dev = qml.device("default.qubit", wires=num_qubits, shots=50)  # Reduced from 100
        
        # Rest of initialization...
        self.weights = nn.Parameter(
            torch.rand(num_layers, (num_qubits * 2 - 1)) * 2 * torch.pi - torch.pi
        )
        self.coeffs = nn.Parameter(torch.rand(num_basis, self.z_dim))
    
    def _precompute_optimal_basis(self):
        """Precompute basis observables for efficiency."""
        import numpy as np
        
        paulis = [qml.PauliZ, qml.PauliX, qml.PauliY, qml.Identity]
        basis = []
        
        # Use fixed seed for reproducible basis generation
        np.random.seed(42)
        
        for b in range(self.num_basis):
            # Generate basis more efficiently
            pauli_choices = np.random.choice(4, self.num_qubits)
            obs = paulis[pauli_choices[0]](0)
            
            for i in range(1, self.num_qubits):
                obs = obs @ paulis[pauli_choices[i]](i)
            
            basis.append(obs)
        
        return basis
    
    # Rest of the class implementation...
```

## Priority 5: Monitoring and Profiling Setup

### Add Performance Monitoring

Create a new file: `source/performance_monitor.py`

```python
import time
import torch
from contextlib import contextmanager
from typing import Dict, Any
import psutil
import GPUtil

class PerformanceMonitor:
    """Real-time performance monitoring for training."""
    
    def __init__(self):
        self.metrics = []
        self.current_batch_start = None
    
    @contextmanager
    def monitor_batch(self, batch_size: int, component: str):
        """Monitor a single batch processing."""
        # Record initial state
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_mem_before = torch.cuda.memory_allocated()
        
        cpu_percent_before = psutil.cpu_percent()
        start_time = time.time()
        
        try:
            yield self
        finally:
            # Record final state
            end_time = time.time()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                gpu_mem_after = torch.cuda.memory_allocated()
                gpu_utilization = GPUtil.getGPUs()[0].load * 100 if GPUtil.getGPUs() else 0
            else:
                gpu_mem_after = gpu_mem_before = 0
                gpu_utilization = 0
            
            cpu_percent_after = psutil.cpu_percent()
            
            # Store metrics
            batch_metrics = {
                'timestamp': start_time,
                'component': component,
                'batch_size': batch_size,
                'duration': end_time - start_time,
                'gpu_memory_used': gpu_mem_after - gpu_mem_before,
                'gpu_utilization': gpu_utilization,
                'cpu_utilization': (cpu_percent_before + cpu_percent_after) / 2,
                'samples_per_second': batch_size / (end_time - start_time)
            }
            
            self.metrics.append(batch_metrics)
    
    def get_latest_metrics(self, n: int = 10) -> Dict[str, Any]:
        """Get summary of latest n batches."""
        recent_metrics = self.metrics[-n:] if self.metrics else []
        
        if not recent_metrics:
            return {}
        
        return {
            'avg_duration': sum(m['duration'] for m in recent_metrics) / len(recent_metrics),
            'avg_gpu_memory': sum(m['gpu_memory_used'] for m in recent_metrics) / len(recent_metrics),
            'avg_gpu_util': sum(m['gpu_utilization'] for m in recent_metrics) / len(recent_metrics),
            'avg_samples_per_sec': sum(m['samples_per_second'] for m in recent_metrics) / len(recent_metrics),
            'total_batches': len(self.metrics)
        }
```

### Integrate Monitoring in Main Model

**File**: `source/model.py`
**Add to class initialization**:

```python
class GaussGan(LightningModule):
    def __init__(self, generator, discriminator, predictor, optimizer, **kwargs):
        # ... existing initialization ...
        
        # ADD PERFORMANCE MONITORING
        from .performance_monitor import PerformanceMonitor
        self.perf_monitor = PerformanceMonitor()
    
    def training_step(self, batch, batch_idx):
        batch_size = batch[0].size(0)
        
        # Monitor training step performance
        with self.perf_monitor.monitor_batch(batch_size, 'training'):
            # ... existing training step code ...
            pass
        
        # Log performance metrics periodically
        if batch_idx % 50 == 0:
            perf_metrics = self.perf_monitor.get_latest_metrics(10)
            self.log_dict({
                f"perf/avg_duration": perf_metrics.get('avg_duration', 0),
                f"perf/gpu_utilization": perf_metrics.get('avg_gpu_util', 0),
                f"perf/samples_per_sec": perf_metrics.get('avg_samples_per_sec', 0),
            }, on_step=True)
```

## Implementation Timeline

### Week 1: Critical Issues
- [ ] Checkpoint cleanup (Day 1)
- [ ] Configuration updates (Day 1)
- [ ] Mixed precision training (Day 2)
- [ ] Memory-efficient validation (Day 3)

### Week 2: Performance Optimizations
- [ ] DataLoader optimization (Day 1)
- [ ] Training step optimization (Day 2-3)
- [ ] Metric computation optimization (Day 4)
- [ ] Performance monitoring setup (Day 5)

### Week 3: Quantum Optimizations
- [ ] Quantum circuit batching (Day 1-2)
- [ ] Shadow noise optimization (Day 3-4)
- [ ] Hardware backend testing (Day 5)

### Week 4: Testing and Validation
- [ ] Benchmark all optimizations (Day 1-2)
- [ ] Regression testing (Day 3-4)
- [ ] Documentation updates (Day 5)

## Testing and Validation

### Performance Regression Tests

Create `tests/test_performance.py`:

```python
import pytest
import torch
import time
from source.model import GaussGan
from source.nn import QuantumNoise, ClassicalNoise

def test_quantum_vs_classical_performance():
    """Ensure quantum optimizations don't break functionality."""
    batch_size = 64
    
    # Test classical generator
    classical_gen = ClassicalNoise(z_dim=4, generator_type="classical_normal")
    start = time.time()
    classical_output = classical_gen(batch_size)
    classical_time = time.time() - start
    
    # Test quantum generator
    quantum_gen = QuantumNoise(num_qubits=4, num_layers=2)
    start = time.time()
    quantum_output = quantum_gen(batch_size)
    quantum_time = time.time() - start
    
    # Verify outputs have correct shape
    assert classical_output.shape == (batch_size, 4)
    assert quantum_output.shape == (batch_size, 4)
    
    # Performance sanity check
    print(f"Classical: {classical_time:.4f}s, Quantum: {quantum_time:.4f}s")
    print(f"Quantum/Classical ratio: {quantum_time/classical_time:.1f}x")

def test_memory_optimization():
    """Test that memory optimizations don't cause issues."""
    # This would test the memory-efficient validation
    # and gradient clearing optimizations
    pass

if __name__ == "__main__":
    test_quantum_vs_classical_performance()
```

### Benchmarking Script

Create `scripts/run_benchmarks.py`:

```python
#!/usr/bin/env python3
"""
Comprehensive benchmarking script for GaussGAN optimizations.
"""

import sys
import time
import torch
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from source.nn import QuantumNoise, ClassicalNoise, QuantumShadowNoise
from docs.performance_optimizations import PerformanceBenchmark

def run_comprehensive_benchmark():
    """Run full performance benchmark suite."""
    
    benchmark = PerformanceBenchmark()
    batch_sizes = [32, 64, 128, 256, 512]
    
    print("GaussGAN Comprehensive Performance Benchmark")
    print("=" * 60)
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        results[batch_size] = {}
        
        # Test Classical Normal
        classical_gen = ClassicalNoise(z_dim=4, generator_type="classical_normal")
        with benchmark.measure_performance("classical_normal", batch_size):
            output = classical_gen(batch_size)
        
        results[batch_size]['classical'] = benchmark.metrics['classical_time'][-1]
        print(f"  Classical: {results[batch_size]['classical']:.4f}s")
        
        # Test Quantum Basic
        quantum_gen = QuantumNoise(num_qubits=4, num_layers=2)
        with benchmark.measure_performance("quantum_basic", batch_size):
            output = quantum_gen(batch_size)
        
        results[batch_size]['quantum'] = benchmark.metrics['quantum_time'][-1]
        print(f"  Quantum: {results[batch_size]['quantum']:.4f}s")
        
        # Calculate and display ratio
        ratio = results[batch_size]['quantum'] / results[batch_size]['classical']
        print(f"  Ratio: {ratio:.1f}x")
    
    # Generate comprehensive report
    benchmark.export_benchmark_report("benchmark_results.md")
    
    return results

if __name__ == "__main__":
    results = run_comprehensive_benchmark()
    print("\nBenchmark completed. Results exported to benchmark_results.md")
```

## Rollback Plan

If optimizations cause issues, here's the rollback procedure:

1. **Revert Configuration**: Restore original `config.yaml`
2. **Disable Mixed Precision**: Remove `precision=16` from trainer
3. **Restore Original DataLoader**: Remove optimization parameters
4. **Revert Training Step**: Restore original gradient handling
5. **Git Checkout**: Use git to revert specific files if needed

```bash
# Example rollback commands
git checkout HEAD -- config.yaml
git checkout HEAD -- source/model.py
git checkout HEAD -- source/data.py
```

## Success Metrics

Monitor these metrics to verify optimization success:

1. **Training Speed**: Target 2-3x improvement
2. **Memory Usage**: Target 30-50% reduction
3. **Storage Usage**: Target 80% reduction in checkpoint storage
4. **GPU Utilization**: Target >60% for classical, >30% for quantum
5. **Samples/Second**: Target 2x improvement in throughput

## Support and Troubleshooting

Common issues and solutions:

1. **CUDA Out of Memory**: Reduce batch size or enable gradient checkpointing
2. **Quantum Circuit Errors**: Verify PennyLane version and device availability
3. **Performance Regression**: Use benchmarking script to identify bottlenecks
4. **Training Instability**: Adjust mixed precision settings or learning rates

For additional support, refer to the performance monitoring logs and benchmark reports generated by the optimization tools.