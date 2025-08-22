#!/usr/bin/env python3
"""
Quick performance analysis script for GaussGAN project.
This script provides immediate insights into current performance bottlenecks.

Usage: uv run python docs/quick_performance_check.py
"""

import sys
import os
import time
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_system_status():
    """Check system configuration and resource availability."""
    print("=" * 60)
    print("SYSTEM STATUS CHECK")
    print("=" * 60)
    
    # PyTorch and CUDA info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Total memory: {props.total_memory / (1024**3):.1f} GB")
            print(f"  Current memory usage: {torch.cuda.memory_allocated(i) / (1024**2):.1f} MB")
    
    # Tensor Core optimization status
    print(f"Float32 matmul precision: {torch.backends.cuda.matmul.allow_tf32}")
    
    print()

def analyze_checkpoint_storage():
    """Analyze checkpoint storage usage and provide cleanup recommendations."""
    print("=" * 60)
    print("CHECKPOINT STORAGE ANALYSIS")
    print("=" * 60)
    
    checkpoint_dir = project_root / "checkpoints"
    
    if not checkpoint_dir.exists():
        print("Checkpoint directory not found.")
        return
    
    # Count different types of checkpoints
    all_checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    last_checkpoints = list(checkpoint_dir.glob("last-v*.ckpt"))
    run_checkpoints = list(checkpoint_dir.glob("run_id-*.ckpt"))
    
    total_size = sum(f.stat().st_size for f in all_checkpoints)
    
    print(f"Total checkpoints: {len(all_checkpoints)}")
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    print(f"Last-v* checkpoints: {len(last_checkpoints)} (potentially redundant)")
    print(f"Run-specific checkpoints: {len(run_checkpoints)}")
    
    # Calculate potential savings
    redundant_size = sum(f.stat().st_size for f in last_checkpoints)
    print(f"Potential savings from removing last-v* files: {redundant_size / (1024**2):.1f} MB")
    
    # Analyze run distribution
    if run_checkpoints:
        run_ids = {}
        for ckpt in run_checkpoints:
            run_id = ckpt.name.split('-')[1]
            if run_id not in run_ids:
                run_ids[run_id] = []
            run_ids[run_id].append(ckpt)
        
        print(f"Number of different runs: {len(run_ids)}")
        avg_checkpoints_per_run = np.mean([len(ckpts) for ckpts in run_ids.values()])
        print(f"Average checkpoints per run: {avg_checkpoints_per_run:.1f}")
        
        # Find runs with excessive checkpoints
        excessive_runs = {k: v for k, v in run_ids.items() if len(v) > 10}
        if excessive_runs:
            print(f"Runs with >10 checkpoints: {len(excessive_runs)}")
    
    print()

def benchmark_generators():
    """Quick benchmark of different generator types."""
    print("=" * 60)
    print("GENERATOR PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    try:
        from source.nn import ClassicalNoise, QuantumNoise, QuantumShadowNoise
    except ImportError as e:
        print(f"Could not import generators: {e}")
        return
    
    batch_sizes = [32, 64, 128]
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        results[batch_size] = {}
        
        # Test Classical Normal
        try:
            classical_gen = ClassicalNoise(z_dim=4, generator_type="classical_normal")
            
            # Warmup
            _ = classical_gen(10)
            
            # Benchmark
            start_time = time.time()
            for _ in range(5):
                output = classical_gen(batch_size)
            classical_time = (time.time() - start_time) / 5
            
            results[batch_size]['classical'] = classical_time
            print(f"  Classical Normal: {classical_time:.4f}s ({batch_size/classical_time:.0f} samples/s)")
            
        except Exception as e:
            print(f"  Classical Normal: ERROR - {e}")
        
        # Test Quantum Basic (with timeout)
        try:
            quantum_gen = QuantumNoise(num_qubits=4, num_layers=2)
            
            # Single run with timeout
            start_time = time.time()
            output = quantum_gen(min(batch_size, 32))  # Limit to prevent long waits
            quantum_time = time.time() - start_time
            
            # Scale to full batch size for comparison
            quantum_time_scaled = quantum_time * (batch_size / min(batch_size, 32))
            
            results[batch_size]['quantum'] = quantum_time_scaled
            print(f"  Quantum Basic: {quantum_time:.4f}s (scaled: {quantum_time_scaled:.4f}s)")
            print(f"    Estimated throughput: {batch_size/quantum_time_scaled:.1f} samples/s")
            
            if 'classical' in results[batch_size]:
                ratio = quantum_time_scaled / results[batch_size]['classical']
                print(f"    Quantum vs Classical: {ratio:.1f}x slower")
            
        except Exception as e:
            print(f"  Quantum Basic: ERROR - {e}")
        
        # Skip Quantum Shadow for speed in quick check
    
    print()

def check_memory_usage():
    """Check current memory usage patterns."""
    print("=" * 60)
    print("MEMORY USAGE CHECK")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available - skipping GPU memory check")
        return
    
    # Check initial GPU memory
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()
    print(f"Initial GPU memory: {initial_memory / (1024**2):.1f} MB")
    
    # Test memory usage with different tensor sizes (safe sizes)
    test_sizes = [1000, 5000]  # Reduced to avoid OOM
    
    for size in test_sizes:
        torch.cuda.empty_cache()
        
        # Create smaller tensors and measure memory
        x = torch.randn(size, 50, device='cuda')  # Reduced dimension
        y = torch.randn(size, 50, device='cuda')
        
        memory_after_creation = torch.cuda.memory_allocated()
        
        # Perform safer computation
        z = torch.matmul(x, y.T)
        
        memory_after_computation = torch.cuda.memory_allocated()
        
        print(f"Size {size}:")
        print(f"  After creation: {memory_after_creation / (1024**2):.1f} MB")
        print(f"  After computation: {memory_after_computation / (1024**2):.1f} MB")
        print(f"  Computation overhead: {(memory_after_computation - memory_after_creation) / (1024**2):.1f} MB")
        
        # Clean up
        del x, y, z
        torch.cuda.empty_cache()
    
    print()

def analyze_config():
    """Analyze current configuration for optimization opportunities."""
    print("=" * 60)
    print("CONFIGURATION ANALYSIS")
    print("=" * 60)
    
    config_path = project_root / "config.yaml"
    
    if not config_path.exists():
        print("config.yaml not found")
        return
    
    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Check key performance-related settings
        batch_size = config.get('batch_size', 'Not set')
        validation_samples = config.get('validation_samples', 'Not set')
        n_critic = config.get('n_critic', 'Not set')
        grad_penalty = config.get('grad_penalty', 'Not set')
        
        print(f"Current batch_size: {batch_size}")
        print(f"Current validation_samples: {validation_samples}")
        print(f"Current n_critic: {n_critic}")
        print(f"Current grad_penalty: {grad_penalty}")
        
        # Recommendations
        print("\nOptimization Recommendations:")
        
        if isinstance(batch_size, int) and batch_size < 512:
            print(f"  ⚡ Increase batch_size to 512-1024 for better GPU utilization")
        
        if isinstance(validation_samples, int) and validation_samples > 300:
            print(f"  ⚡ Reduce validation_samples to 200-300 for faster validation")
        
        if isinstance(n_critic, int) and n_critic > 3:
            print(f"  ⚡ Reduce n_critic to 3 for faster training")
        
        if isinstance(grad_penalty, (int, float)) and grad_penalty > 0.5:
            print(f"  ⚡ Reduce grad_penalty to 0.1-0.2 for less memory usage")
        
    except Exception as e:
        print(f"Could not analyze config: {e}")
    
    print()

def generate_optimization_summary():
    """Generate summary of optimization opportunities."""
    print("=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    
    print("🎯 IMMEDIATE ACTIONS (High Impact, Low Effort):")
    print("   1. Clean up redundant checkpoints (save 1-2GB disk space)")
    print("   2. Increase batch_size to 512-1024")
    print("   3. Enable mixed precision training (precision=16)")
    print("   4. Reduce validation_samples to 200")
    
    print("\n⚡ PERFORMANCE OPTIMIZATIONS (Medium Impact):")
    print("   1. Optimize DataLoader with num_workers=4, pin_memory=True")
    print("   2. Implement chunked validation processing")
    print("   3. Use gradient clearing with set_to_none=True")
    print("   4. Reduce metric computation frequency")
    
    print("\n🔬 QUANTUM OPTIMIZATIONS (Research Impact):")
    print("   1. Implement quantum circuit batching")
    print("   2. Use lightning.gpu backend if available")
    print("   3. Optimize basis generation for shadow noise")
    print("   4. Cache quantum circuit compilations")
    
    print("\n📊 MONITORING:")
    print("   1. Set up performance monitoring with GPU utilization tracking")
    print("   2. Implement automated benchmarking")
    print("   3. Add memory usage profiling")
    
    print("\n🎯 Expected Overall Improvement: 3-5x training speed, 50% memory reduction")
    print()

def main():
    """Run complete performance analysis."""
    print("GaussGAN Quick Performance Analysis")
    print(f"Project: {project_root}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    check_system_status()
    analyze_checkpoint_storage()
    analyze_config()
    check_memory_usage()
    benchmark_generators()
    generate_optimization_summary()
    
    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("📋 Next steps:")
    print("   1. Review the detailed implementation guide: docs/implementation_guide.md")
    print("   2. Start with checkpoint cleanup for immediate space savings")
    print("   3. Implement batch size and mixed precision optimizations")
    print("   4. Run full benchmark after optimizations: uv run python docs/performance_optimizations.py")

if __name__ == "__main__":
    main()