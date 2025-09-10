"""
PennyLane and CUDA Compatibility Check Script
Check compatibility issues between quantum simulation library and GPU environment
"""

import torch
import pennylane as qml
import numpy as np
import sys
import gc
import traceback

def check_torch_cuda():
    """Check PyTorch CUDA environment"""
    print("=" * 50)
    print("PyTorch CUDA Environment Check")
    print("=" * 50)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name()}")
        
        # Memory information
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        
        print(f"GPU total memory: {total_memory:.2f} GB")
        print(f"Allocated memory: {allocated:.2f} GB")
        print(f"Cached memory: {cached:.2f} GB")
        print(f"Available memory: {total_memory - allocated - cached:.2f} GB")

def check_pennylane_info():
    """Check PennyLane information"""
    print("\n" + "=" * 50)
    print("PennyLane Environment Check")
    print("=" * 50)
    
    print(f"PennyLane version: {qml.__version__}")
    
    # Check available devices
    print("\nAvailable quantum devices:")
    try:
        available_devices = qml.device._get_device_entrypoints()
        for device_name in available_devices:
            print(f"  - {device_name}")
    except Exception as e:
        print(f"  Unable to get device list: {e}")

def test_basic_quantum_circuit():
    """Test basic quantum circuit"""
    print("\n" + "=" * 50)
    print("Basic Quantum Circuit Test")
    print("=" * 50)
    
    try:
        # Create small quantum device
        dev = qml.device("default.qubit", wires=2)
        
        @qml.qnode(dev, interface="torch")
        def simple_circuit(x):
            qml.RY(x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))
        
        # Test single call
        x = torch.tensor(0.5, requires_grad=True)
        result = simple_circuit(x)
        print(f"Single circuit call successful: {result}")
        
        # Test gradient calculation
        loss = result ** 2
        loss.backward()
        print(f"Gradient calculation successful: {x.grad}")
        
    except Exception as e:
        print(f"Basic quantum circuit test failed: {e}")
        traceback.print_exc()

def test_batch_quantum_processing():
    """Test batch quantum processing"""
    print("\n" + "=" * 50)
    print("Batch Quantum Processing Test")
    print("=" * 50)
    
    batch_sizes = [1, 4, 8, 16, 32]
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        
        try:
            # Create quantum device
            dev = qml.device("default.qubit", wires=3)
            
            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def batch_circuit(params):
                qml.RY(params[0], wires=0)
                qml.RY(params[1], wires=1)
                qml.CNOT(wires=[0, 1])
                qml.RZ(params[2], wires=1)
                return [qml.expval(qml.PauliZ(i)) for i in range(3)]
            
            # Prepare parameters
            params = torch.rand(3, requires_grad=True)
            
            # Simulate batch processing (similar to QuantumNoise approach)
            results = []
            for i in range(batch_size):
                # Add small amount of randomness
                noise = torch.randn(3) * 0.1
                circuit_params = params + noise
                
                circuit_output = batch_circuit(circuit_params)
                result = torch.stack([tensor for tensor in circuit_output])
                results.append(result)
                
                # Check memory usage
                if torch.cuda.is_available() and i % 8 == 7:
                    allocated = torch.cuda.memory_allocated() / 1024**2
                    print(f"  Step {i+1}/{batch_size}: GPU memory {allocated:.1f} MB")
            
            # Final stacking
            batch_output = torch.stack(results)
            print(f"  Batch processing successful! Output shape: {batch_output.shape}")
            
            # Clean up memory
            del results, batch_output
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
        except Exception as e:
            print(f"  Batch size {batch_size} failed: {e}")
            # Try to recover
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

def test_memory_stress():
    """Memory stress test"""
    print("\n" + "=" * 50)
    print("Memory Stress Test")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("Skipping GPU memory stress test (no CUDA support)")
        return
    
    try:
        # Record initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"Initial GPU memory: {initial_memory:.1f} MB")
        
        # Create many quantum devices (simulate original problem)
        devices = []
        circuits = []
        
        for i in range(10):  # Create 10 devices
            dev = qml.device("default.qubit", wires=6)  # Use original configuration
            devices.append(dev)
            
            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def circuit(w):
                for j in range(6):
                    qml.RY(w[j], wires=j)
                for j in range(5):
                    qml.CNOT(wires=[j, j+1])
                return [qml.expval(qml.PauliZ(k)) for k in range(6)]
            
            circuits.append(circuit)
            
            current_memory = torch.cuda.memory_allocated() / 1024**2
            print(f"Device {i+1}: GPU memory {current_memory:.1f} MB (+{current_memory-initial_memory:.1f})")
            
            if current_memory > 1000:  # Stop if exceeds 1GB
                print("Memory usage too high, stopping test")
                break
        
        print(f"Created {len(devices)} quantum devices")
        
    except Exception as e:
        print(f"Memory stress test failed: {e}")
    finally:
        # Clean up
        torch.cuda.empty_cache()
        gc.collect()

def check_pennylane_lightning():
    """Check PennyLane Lightning accelerator"""
    print("\n" + "=" * 50)
    print("PennyLane Lightning Check")
    print("=" * 50)
    
    try:
        # Try to use Lightning device
        dev_lightning = qml.device("lightning.qubit", wires=4)
        
        @qml.qnode(dev_lightning, interface="torch")
        def lightning_circuit(x):
            qml.RY(x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))
        
        x = torch.tensor(0.3)
        result = lightning_circuit(x)
        print(f"Lightning device available: {result}")
        
    except Exception as e:
        print(f"Lightning device not available: {e}")
        print("Suggestion: Consider using Lightning to improve performance")

def main():
    """Main check function"""
    print("GaussGAN Quantum Environment Compatibility Check")
    print("=" * 80)
    
    check_torch_cuda()
    check_pennylane_info()
    test_basic_quantum_circuit()
    test_batch_quantum_processing()
    test_memory_stress()
    check_pennylane_lightning()
    
    print("\n" + "=" * 80)
    print("Check complete!")
    print("=" * 80)
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"Final GPU memory usage: {final_memory:.1f} MB")

if __name__ == "__main__":
    main()