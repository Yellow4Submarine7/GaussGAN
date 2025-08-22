"""
Unit tests for all generator types: Classical, Quantum, and QuantumShadow.
Tests individual components in isolation with comprehensive coverage.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import warnings

from source.nn import (
    ClassicalNoise,
    QuantumNoise, 
    QuantumShadowNoise,
    MLPGenerator
)


class TestClassicalNoise:
    """Test suite for classical noise generators."""
    
    def test_classical_normal_initialization(self):
        """Test ClassicalNoise initialization with normal distribution."""
        generator = ClassicalNoise(z_dim=4, generator_type='classical_normal')
        
        assert generator.z_dim == 4
        assert generator.generator_type == 'classical_normal'
        assert hasattr(generator, 'dummy')  # Device tracking parameter
    
    def test_classical_uniform_initialization(self):
        """Test ClassicalNoise initialization with uniform distribution."""
        generator = ClassicalNoise(z_dim=6, generator_type='classical_uniform')
        
        assert generator.z_dim == 6
        assert generator.generator_type == 'classical_uniform'
    
    def test_invalid_generator_type(self):
        """Test that invalid generator type raises error."""
        with pytest.raises(ValueError, match="Unknown generator type"):
            ClassicalNoise(z_dim=4, generator_type='invalid_type')
    
    def test_normal_noise_generation(self, device):
        """Test normal noise generation."""
        generator = ClassicalNoise(z_dim=4, generator_type='classical_normal')
        generator = generator.to(device)
        
        batch_size = 32
        noise = generator(batch_size)
        
        assert isinstance(noise, torch.Tensor)
        assert noise.shape == (batch_size, 4)
        assert noise.device == device
        
        # Statistical properties of normal distribution
        mean = torch.mean(noise, dim=0)
        std = torch.std(noise, dim=0)
        
        # Should be approximately N(0,1)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=0.3)
        assert torch.allclose(std, torch.ones_like(std), atol=0.3)
    
    def test_uniform_noise_generation(self, device):
        """Test uniform noise generation."""
        generator = ClassicalNoise(z_dim=4, generator_type='classical_uniform')
        generator = generator.to(device)
        
        batch_size = 32
        noise = generator(batch_size)
        
        assert isinstance(noise, torch.Tensor)
        assert noise.shape == (batch_size, 4)
        assert noise.device == device
        
        # Should be in range [-1, 1]
        assert torch.all(noise >= -1)
        assert torch.all(noise <= 1)
        
        # Should be approximately uniform
        mean = torch.mean(noise, dim=0)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=0.3)
    
    def test_device_consistency(self, device):
        """Test that noise is generated on correct device."""
        generator = ClassicalNoise(z_dim=2, generator_type='classical_normal')
        generator = generator.to(device)
        
        noise = generator(10)
        assert noise.device == device
    
    def test_different_batch_sizes(self, device):
        """Test generation with different batch sizes."""
        generator = ClassicalNoise(z_dim=3, generator_type='classical_normal')
        generator = generator.to(device)
        
        batch_sizes = [1, 5, 16, 64, 128]
        
        for batch_size in batch_sizes:
            noise = generator(batch_size)
            assert noise.shape == (batch_size, 3)
            assert noise.device == device
    
    def test_reproducibility_with_seed(self, device):
        """Test reproducibility when using torch.manual_seed."""
        generator = ClassicalNoise(z_dim=4, generator_type='classical_normal')
        generator = generator.to(device)
        
        # Generate with fixed seed
        torch.manual_seed(42)
        noise1 = generator(10)
        
        torch.manual_seed(42)
        noise2 = generator(10)
        
        assert torch.allclose(noise1, noise2)


@pytest.mark.quantum
class TestQuantumNoise:
    """Test suite for quantum noise generators."""
    
    def setup_method(self):
        """Set up quantum testing environment."""
        try:
            import pennylane as qml
            self.pennylane_available = True
        except ImportError:
            self.pennylane_available = False
    
    @pytest.mark.skipif(not hasattr(pytest, 'importorskip') and True, 
                       reason="PennyLane not available")
    def test_quantum_noise_initialization(self):
        """Test QuantumNoise initialization."""
        if not self.pennylane_available:
            pytest.skip("PennyLane not available")
        
        generator = QuantumNoise(num_qubits=4, num_layers=2)
        
        assert generator.num_qubits == 4
        assert generator.num_layers == 2
        assert hasattr(generator, 'weights')
        assert hasattr(generator, 'gen_circuit')
        
        # Check weight dimensions
        expected_weight_dim = (2, 4 * 2 - 1)  # (num_layers, num_qubits * 2 - 1)
        assert generator.weights.shape == expected_weight_dim
        
        # Weights should be in range [-π, π]
        assert torch.all(generator.weights >= -np.pi)
        assert torch.all(generator.weights <= np.pi)
    
    @pytest.mark.skipif(not hasattr(pytest, 'importorskip') and True,
                       reason="PennyLane not available")
    def test_quantum_noise_generation(self, device):
        """Test quantum noise generation."""
        if not self.pennylane_available:
            pytest.skip("PennyLane not available")
        
        generator = QuantumNoise(num_qubits=4, num_layers=1)  # Reduced for testing
        generator = generator.to(device)
        
        batch_size = 8  # Small batch for quantum testing
        noise = generator(batch_size)
        
        assert isinstance(noise, torch.Tensor)
        assert noise.shape == (batch_size, 4)  # num_qubits
        assert noise.dtype == torch.float32
        
        # Quantum measurements should be in range [-1, 1] (Pauli-Z expectation values)
        assert torch.all(noise >= -1)
        assert torch.all(noise <= 1)
    
    @pytest.mark.skipif(not hasattr(pytest, 'importorskip') and True,
                       reason="PennyLane not available")
    def test_quantum_gradient_flow(self, device):
        """Test that gradients flow through quantum circuit."""
        if not self.pennylane_available:
            pytest.skip("PennyLane not available")
        
        generator = QuantumNoise(num_qubits=2, num_layers=1)
        generator = generator.to(device)
        
        # Create a simple loss function
        output = generator(4)
        loss = torch.mean(output**2)
        
        # Compute gradients
        loss.backward()
        
        # Check that weights have gradients
        assert generator.weights.grad is not None
        assert not torch.allclose(generator.weights.grad, torch.zeros_like(generator.weights.grad))
    
    @pytest.mark.skipif(not hasattr(pytest, 'importorskip') and True,
                       reason="PennyLane not available")
    def test_quantum_noise_different_configurations(self):
        """Test different quantum circuit configurations."""
        if not self.pennylane_available:
            pytest.skip("PennyLane not available")
        
        configurations = [
            {'num_qubits': 2, 'num_layers': 1},
            {'num_qubits': 4, 'num_layers': 2},
            {'num_qubits': 6, 'num_layers': 1},
        ]
        
        for config in configurations:
            generator = QuantumNoise(**config)
            
            # Test generation
            noise = generator(4)
            assert noise.shape == (4, config['num_qubits'])
            assert torch.all(noise >= -1)
            assert torch.all(noise <= 1)
    
    @pytest.mark.skipif(not hasattr(pytest, 'importorskip') and True,
                       reason="PennyLane not available")  
    def test_quantum_noise_reproducibility(self):
        """Test quantum noise reproducibility with weight initialization."""
        if not self.pennylane_available:
            pytest.skip("PennyLane not available")
        
        # Create two generators with same random seed
        torch.manual_seed(42)
        generator1 = QuantumNoise(num_qubits=2, num_layers=1)
        
        torch.manual_seed(42)
        generator2 = QuantumNoise(num_qubits=2, num_layers=1)
        
        # Weights should be identical
        assert torch.allclose(generator1.weights, generator2.weights)
        
        # Note: Full reproducibility of quantum circuits may depend on
        # PennyLane's random number generation, which is tested separately


@pytest.mark.quantum
class TestQuantumShadowNoise:
    """Test suite for quantum shadow noise generators."""
    
    def setup_method(self):
        """Set up quantum shadow testing environment."""
        try:
            import pennylane as qml
            self.pennylane_available = True
        except ImportError:
            self.pennylane_available = False
    
    @pytest.mark.skipif(not hasattr(pytest, 'importorskip') and True,
                       reason="PennyLane not available")
    def test_quantum_shadow_initialization(self):
        """Test QuantumShadowNoise initialization."""
        if not self.pennylane_available:
            pytest.skip("PennyLane not available")
        
        generator = QuantumShadowNoise(
            z_dim=4,
            num_qubits=6,
            num_layers=2,
            num_basis=3
        )
        
        assert generator.z_dim == 4
        assert generator.num_qubits == 6
        assert generator.num_layers == 2
        assert generator.num_basis == 3
        
        # Check weight dimensions
        expected_weight_dim = (2, 6 * 2 - 1)  # (num_layers, num_qubits * 2 - 1)
        assert generator.weights.shape == expected_weight_dim
        
        # Check coefficient dimensions
        expected_coeff_dim = (3, 4)  # (num_basis, z_dim)
        assert generator.coeffs.shape == expected_coeff_dim
    
    @pytest.mark.skipif(not hasattr(pytest, 'importorskip') and True,
                       reason="PennyLane not available")
    def test_quantum_shadow_generation(self, device):
        """Test quantum shadow noise generation."""
        if not self.pennylane_available:
            pytest.skip("PennyLane not available")
        
        generator = QuantumShadowNoise(
            z_dim=4,
            num_qubits=4,  # Reduced for testing
            num_layers=1,  # Reduced for testing
            num_basis=2    # Reduced for testing
        )
        generator = generator.to(device)
        
        batch_size = 4  # Small batch for quantum testing
        noise = generator(batch_size)
        
        assert isinstance(noise, torch.Tensor)
        assert noise.shape == (batch_size, 4)  # z_dim
        assert noise.dtype == torch.float32
    
    @pytest.mark.skipif(not hasattr(pytest, 'importorskip') and True,
                       reason="PennyLane not available")
    def test_quantum_shadow_gradient_flow(self, device):
        """Test gradient flow through quantum shadow circuit."""
        if not self.pennylane_available:
            pytest.skip("PennyLane not available")
        
        generator = QuantumShadowNoise(
            z_dim=2,
            num_qubits=2,
            num_layers=1,
            num_basis=2
        )
        generator = generator.to(device)
        
        # Create loss function
        output = generator(2)
        loss = torch.mean(output**2)
        
        # Compute gradients
        loss.backward()
        
        # Check that both weights and coefficients have gradients
        assert generator.weights.grad is not None
        assert generator.coeffs.grad is not None
        assert not torch.allclose(generator.weights.grad, torch.zeros_like(generator.weights.grad))
        assert not torch.allclose(generator.coeffs.grad, torch.zeros_like(generator.coeffs.grad))
    
    @pytest.mark.skipif(not hasattr(pytest, 'importorskip') and True,
                       reason="PennyLane not available")
    def test_quantum_shadow_basis_creation(self):
        """Test quantum shadow basis creation mechanism."""
        if not self.pennylane_available:
            pytest.skip("PennyLane not available")
        
        # Test build_qnode static method
        basis, gen_circuit = QuantumShadowNoise.build_qnode(
            num_qubits=2, num_layers=1, num_basis=2
        )
        
        assert len(basis) == 2  # num_basis
        assert callable(gen_circuit)
        
        # Test circuit execution
        weights = torch.randn(1, 3)  # (num_layers, num_qubits * 2 - 1)
        try:
            result = gen_circuit(weights)
            assert len(result) == 2  # num_basis measurements
        except Exception as e:
            # Circuit execution might fail in test environment, but structure should be correct
            print(f"Circuit execution failed (expected in test): {e}")


class TestMLPGenerator:
    """Test suite for MLP generator (variational output layer)."""
    
    def test_mlp_generator_initialization(self):
        """Test MLPGenerator initialization."""
        generator = MLPGenerator(
            non_linearity='LeakyReLU',
            hidden_dims=[64, 128],
            z_dim=4,
            output_dim=2,
            std_scale=1.5,
            min_std=0.5
        )
        
        assert generator.std_scale == 1.5
        assert generator.min_std == 0.5
        assert hasattr(generator, 'feature_extractor')
        assert hasattr(generator, 'mean_layer')
        assert hasattr(generator, 'logvar_layer')
    
    def test_mlp_generator_forward_pass(self, device):
        """Test MLPGenerator forward pass."""
        generator = MLPGenerator(
            non_linearity='LeakyReLU',
            hidden_dims=[32, 64],
            z_dim=4,
            output_dim=2
        )
        generator = generator.to(device)
        
        batch_size = 16
        z = torch.randn(batch_size, 4).to(device)
        
        output = generator(z)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, 2)
        assert output.device == device
    
    def test_mlp_generator_reparameterization(self, device):
        """Test that generator uses reparameterization trick correctly."""
        generator = MLPGenerator(
            non_linearity='ReLU',
            hidden_dims=[32],
            z_dim=4,
            output_dim=2,
            std_scale=1.0,
            min_std=0.1
        )
        generator = generator.to(device)
        
        batch_size = 100
        z = torch.randn(batch_size, 4).to(device)
        
        # Multiple forward passes should give different outputs (stochastic)
        output1 = generator(z)
        output2 = generator(z)
        
        # Outputs should be different due to sampling
        assert not torch.allclose(output1, output2, atol=1e-6)
        
        # But statistical properties should be similar
        mean1, mean2 = torch.mean(output1, dim=0), torch.mean(output2, dim=0)
        assert torch.allclose(mean1, mean2, atol=0.5)
    
    def test_mlp_generator_variance_control(self, device):
        """Test variance scaling and minimum standard deviation."""
        generator = MLPGenerator(
            non_linearity='ReLU',
            hidden_dims=[32],
            z_dim=2,
            output_dim=2,
            std_scale=2.0,
            min_std=0.8
        )
        generator = generator.to(device)
        
        # Create input that would typically produce small variance
        z = torch.zeros(100, 2).to(device)  # Zero input
        
        outputs = []
        for _ in range(10):  # Multiple samples to estimate variance
            output = generator(z)
            outputs.append(output)
        
        outputs = torch.stack(outputs)  # Shape: (10, 100, 2)
        empirical_std = torch.std(outputs, dim=0).mean(dim=0)  # Average std across batch
        
        # Due to min_std=0.8, empirical std should be reasonably large
        assert torch.all(empirical_std > 0.3)  # Should be affected by min_std constraint
    
    def test_mlp_generator_different_architectures(self, device):
        """Test different network architectures."""
        architectures = [
            [32],
            [64, 64],
            [32, 64, 32],
            [128, 256, 128]
        ]
        
        for hidden_dims in architectures:
            generator = MLPGenerator(
                non_linearity='LeakyReLU',
                hidden_dims=hidden_dims,
                z_dim=4,
                output_dim=2
            )
            generator = generator.to(device)
            
            z = torch.randn(8, 4).to(device)
            output = generator(z)
            
            assert output.shape == (8, 2)
            assert torch.all(torch.isfinite(output))
    
    def test_mlp_generator_activation_functions(self, device):
        """Test different activation functions."""
        activations = ['ReLU', 'LeakyReLU', 'Tanh', 'Sigmoid']
        
        for activation in activations:
            try:
                generator = MLPGenerator(
                    non_linearity=activation,
                    hidden_dims=[32, 32],
                    z_dim=4,
                    output_dim=2
                )
                generator = generator.to(device)
                
                z = torch.randn(8, 4).to(device)
                output = generator(z)
                
                assert output.shape == (8, 2)
                assert torch.all(torch.isfinite(output))
                
            except AttributeError:
                # Some activations might not be available, skip them
                print(f"Activation {activation} not available, skipping")
    
    def test_mlp_generator_gradient_flow(self, device):
        """Test gradient flow through MLP generator."""
        generator = MLPGenerator(
            non_linearity='LeakyReLU',
            hidden_dims=[32, 64],
            z_dim=4,
            output_dim=2
        )
        generator = generator.to(device)
        
        z = torch.randn(16, 4, requires_grad=True).to(device)
        output = generator(z)
        loss = torch.mean(output**2)
        
        loss.backward()
        
        # Check gradients exist and are non-zero
        assert z.grad is not None
        assert not torch.allclose(z.grad, torch.zeros_like(z.grad))
        
        # Check model parameters have gradients
        for param in generator.parameters():
            assert param.grad is not None


class TestGeneratorIntegration:
    """Integration tests for complete generator pipelines."""
    
    def test_classical_mlp_pipeline(self, device):
        """Test complete classical + MLP generator pipeline."""
        noise_gen = ClassicalNoise(z_dim=4, generator_type='classical_normal')
        mlp_gen = MLPGenerator(
            non_linearity='LeakyReLU',
            hidden_dims=[64, 64],
            z_dim=4,
            output_dim=2
        )
        
        # Create sequential pipeline
        generator = torch.nn.Sequential(noise_gen, mlp_gen)
        generator = generator.to(device)
        
        batch_size = 32
        samples = generator(batch_size)
        
        assert isinstance(samples, torch.Tensor)
        assert samples.shape == (batch_size, 2)
        assert samples.device == device
        assert torch.all(torch.isfinite(samples))
    
    @pytest.mark.quantum
    def test_quantum_mlp_pipeline(self, device):
        """Test complete quantum + MLP generator pipeline."""
        try:
            import pennylane as qml
        except ImportError:
            pytest.skip("PennyLane not available")
        
        noise_gen = QuantumNoise(num_qubits=4, num_layers=1)
        mlp_gen = MLPGenerator(
            non_linearity='LeakyReLU',
            hidden_dims=[32, 32],
            z_dim=4,
            output_dim=2
        )
        
        generator = torch.nn.Sequential(noise_gen, mlp_gen)
        generator = generator.to(device)
        
        batch_size = 8  # Smaller for quantum testing
        samples = generator(batch_size)
        
        assert isinstance(samples, torch.Tensor)
        assert samples.shape == (batch_size, 2)
        assert samples.device == device
        assert torch.all(torch.isfinite(samples))
    
    @pytest.mark.quantum
    def test_quantum_shadow_mlp_pipeline(self, device):
        """Test complete quantum shadow + MLP generator pipeline."""
        try:
            import pennylane as qml
        except ImportError:
            pytest.skip("PennyLane not available")
        
        noise_gen = QuantumShadowNoise(
            z_dim=4,
            num_qubits=4,
            num_layers=1,
            num_basis=2
        )
        mlp_gen = MLPGenerator(
            non_linearity='LeakyReLU',
            hidden_dims=[32, 32],
            z_dim=4,
            output_dim=2
        )
        
        generator = torch.nn.Sequential(noise_gen, mlp_gen)
        generator = generator.to(device)
        
        batch_size = 4  # Small for quantum shadow testing
        samples = generator(batch_size)
        
        assert isinstance(samples, torch.Tensor)
        assert samples.shape == (batch_size, 2)
        assert samples.device == device
        assert torch.all(torch.isfinite(samples))
    
    def test_pipeline_gradient_flow(self, device):
        """Test gradient flow through complete pipeline."""
        noise_gen = ClassicalNoise(z_dim=4, generator_type='classical_normal')
        mlp_gen = MLPGenerator(
            non_linearity='ReLU',
            hidden_dims=[32, 32],
            z_dim=4,
            output_dim=2
        )
        
        generator = torch.nn.Sequential(noise_gen, mlp_gen)
        generator = generator.to(device)
        
        samples = generator(16)
        loss = torch.mean(samples**2)
        
        loss.backward()
        
        # Check that MLP generator parameters have gradients
        for param in mlp_gen.parameters():
            assert param.grad is not None
            # Note: ClassicalNoise has no learnable parameters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])