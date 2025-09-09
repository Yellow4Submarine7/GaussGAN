"""Fixed neural network modules with quantum memory leak resolved"""
import random
from abc import ABC, abstractmethod

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from torch import nn
from torch.nn.functional import one_hot


class QuantumNoise(nn.Module):
    def build_qnode(self, num_qubits, num_layers):
        dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(dev, interface="torch", diff_method="best")
        def gen_circuit(w, z_values):  # Random values passed as parameters
            # No random calls inside the circuit!
            z1, z2 = z_values[0], z_values[1]
            
            for i in range(num_qubits):
                qml.RY(np.arcsin(z1), wires=i)
                qml.RZ(np.arcsin(z2), wires=i)
            
            for l in range(num_layers):
                for i in range(num_qubits):
                    qml.RY(w[l][i], wires=i)
                for i in range(num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.RZ(w[l][i + num_qubits], wires=i + 1)
                    qml.CNOT(wires=[i, i + 1])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
        
        return gen_circuit

    def __init__(self, num_qubits: int = 8, num_layers: int = 3, z_dim: int = 4):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.z_dim = z_dim
        self.gen_circuit = self.build_qnode(num_qubits, num_layers)
        
        self.weights = nn.Parameter(
            torch.rand(num_layers, (num_qubits * 2 - 1)) * 2 * torch.pi - torch.pi
        )
        
        # Add projection layer to match z_dim
        self.projection = nn.Linear(num_qubits, z_dim)

    def forward(self, batch_size: int):
        # Use torch.no_grad() to avoid computation graph accumulation
        with torch.no_grad():
            sample_list = []
            for _ in range(batch_size):
                # Generate random values outside the circuit
                z_values = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
                # Pass to circuit
                output = self.gen_circuit(self.weights.detach(), z_values)
                sample = torch.stack([tensor for tensor in output])
                sample_list.append(sample)
            
            quantum_output = torch.stack(sample_list).float()
        
        # Project to z_dim with gradients enabled
        return self.projection(quantum_output)


class QuantumShadowNoise(nn.Module):
    def build_qnode(self, num_qubits, num_layers, num_basis):
        import random as py_random  # Use alias to avoid confusion
        
        paulis = [qml.PauliX, qml.PauliY, qml.PauliZ]
        
        def create_tensor_observable(num_qubits, paulis):
            obs = qml.Identity(0)
            for i in range(1, num_qubits):
                obs = obs @ py_random.choice(paulis)(i)
            return obs
        
        basis = [create_tensor_observable(num_qubits, paulis) for _ in range(num_basis)]
        dev = qml.device("default.qubit", wires=num_qubits, shots=100)
        
        @qml.qnode(dev, interface="torch", diff_method="best")
        def gen_circuit(w, z_values):  # Random values as parameters
            z1, z2 = z_values[0], z_values[1]
            
            for i in range(num_qubits):
                qml.RY(np.arcsin(z1), wires=i)
                qml.RZ(np.arcsin(z2), wires=i)
            
            for l in range(num_layers):
                for i in range(num_qubits):
                    qml.RY(w[l][i], wires=i)
                for i in range(num_qubits - 1):
                    qml.CNOT(wires=[i, i+1])
                    qml.RZ(w[l][i+num_qubits], wires=i+1)
                    qml.CNOT(wires=[i, i+1])
            
            return qml.shadow_expval(basis)
        
        return basis, gen_circuit

    def __init__(self, num_qubits: int = 8, num_layers: int = 3, num_basis: int = 3, z_dim: int = 4):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.num_basis = num_basis
        self.z_dim = z_dim
        
        self.basis, self.gen_circuit = self.build_qnode(num_qubits, num_layers, num_basis)
        
        self.weights = nn.Parameter(
            torch.rand(num_layers, (num_qubits * 2 - 1)) * 2 * torch.pi - torch.pi
        )
        
        # Add projection layer to match z_dim
        self.projection = nn.Linear(num_basis, z_dim)

    def forward(self, batch_size: int):
        with torch.no_grad():
            sample_list = []
            for _ in range(batch_size):
                z_values = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
                output = self.gen_circuit(self.weights.detach(), z_values)
                sample = torch.cat([tensor.unsqueeze(0) for tensor in output])
                sample_list.append(sample)
            
            quantum_output = torch.stack(sample_list).float()
        
        # Project to z_dim with gradients enabled
        return self.projection(quantum_output)


class ClassicalNoise(nn.Module):
    def __init__(self, z_dim, generator_type):
        super().__init__()
        self.z_dim = z_dim
        self.generator_type = generator_type
        # Create a dummy parameter to track the device
        self.register_buffer("dummy", torch.zeros(1))

    def forward(self, batch_size):
        # Get the device from the dummy parameter
        device = self.dummy.device

        if self.generator_type == "classical_normal":
            return torch.randn(batch_size, self.z_dim, device=device)
        elif self.generator_type == "classical_uniform":
            return torch.rand(batch_size, self.z_dim, device=device) * 2 - 1
        else:
            raise ValueError(f"Unknown generator type: {self.generator_type}")


class MLPGenerator(nn.Module):
    def __init__(self, non_linearity, hidden_dims, z_dim, output_dim=2, std_scale=1.5, min_std=0.5):
        super(MLPGenerator, self).__init__()
        layers = []
        #layers.append(nn.Sigmoid())
        non_linearity = getattr(nn, non_linearity, nn.LeakyReLU)
        
        self.std_scale = std_scale
        self.min_std = min_std

        current_dim = z_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(current_dim, hdim))
            layers.append(nn.BatchNorm1d(hdim))
            layers.append(non_linearity())
            current_dim = hdim
        self.mean_layer = nn.Linear(current_dim, output_dim)
        self.logvar_layer = nn.Linear(current_dim, output_dim)
        nn.init.xavier_uniform_(self.logvar_layer.weight, gain=2.0)
        nn.init.constant_(self.logvar_layer.bias, 0.5)
        
        self.feature_extractor = nn.Sequential(*layers)

    def forward(self, z):
        features = self.feature_extractor(z)
        #Set the output layer to a normal distribution format.
        mean = self.mean_layer(features)
        log_var = self.logvar_layer(features)
        std = torch.exp(0.5 * log_var) * self.std_scale
        std = torch.clamp(std, min=self.min_std)
        
        #reparameterization allows gradients to flow through mean and std
        eps = torch.randn_like(std)
        return mean + eps * std


class MLPDiscriminator(nn.Module):
    def __init__(self, non_linearity, hidden_dims, input_dim=2, output_dim=1):
        super(MLPDiscriminator, self).__init__()
        layers = []
        #non_linearity = getattr(nn, non_linearity)
        non_linearity = getattr(nn, non_linearity, nn.LeakyReLU)

        current_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(current_dim, hdim))
            #layers.append(nn.Sigmoid())
            layers.append(non_linearity())
            current_dim = hdim
        layers.append(nn.Linear(current_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)