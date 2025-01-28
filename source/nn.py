import random

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from torch import nn
from torch.nn.functional import one_hot
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv
from torch_geometric.utils import one_hot

# from .data import extract_graphs_from_features
# from .layers import GraphAggregation, GraphConvolution, MultiDenseLayers


class QuantumMolGanNoise(nn.Module):
    def __init__(self, num_qubits: int = 8, num_layers: int = 3):
        super(QuantumMolGanNoise, self).__init__()

        # Register the parameters with the module
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        # Initialize weights with PyTorch (between -pi and pi) and register as a learnable parameter
        self.weights = nn.Parameter(
            torch.rand(num_layers, (num_qubits * 2 - 1)) * 2 * torch.pi - torch.pi
        )

        # Initialize the device
        dev = qml.device("default.qubit", wires=num_qubits)

        # Define the quantum circuit
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def gen_circuit(w):
            # random noise as generator input
            z1 = random.uniform(-1, 1)
            z2 = random.uniform(-1, 1)
            # construct generator circuit for both atom vector and node matrix
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

        self.gen_circuit = gen_circuit

    def forward(self, batch_size: int):
        sample_list = [
            torch.concat(
                [tensor.unsqueeze(0) for tensor in self.gen_circuit(self.weights)]
            )
            for _ in range(batch_size)
        ]
        noise = torch.stack(tuple(sample_list)).float()
        return noise


class QuantumShadowNoise(nn.Module):
    @staticmethod
    def build_qnode(num_qubits, num_layers, num_basis):

        paulis = [qml.PauliZ, qml.PauliX, qml.PauliY, qml.Identity]
        basis = [
            qml.operation.Tensor(*[random.choice(paulis)(i) for i in range(num_qubits)])
            for _ in range(num_basis)
        ]

        dev = qml.device("default.qubit", wires=num_qubits, shots=300)

        @qml.qnode(dev, interface="torch", diff_method="best")
        def gen_circuit(w):
            z1 = random.uniform(-1, 1)
            z2 = random.uniform(-1, 1)
            # construct generator circuit for both atom vector and node matrix
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
            return qml.shadow_expval(basis)

        return basis, gen_circuit

    def __init__(
        self,
        z_dim: int,
        *,
        num_qubits: int = 8,
        num_layers: int = 3,
        num_basis: int = 3,
    ):
        super(QuantumShadowNoise, self).__init__()

        # Register the parameters with the module
        self.z_dim = z_dim
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.num_basis = num_basis

        self.basis, self.gen_circuit = self.build_qnode(
            num_qubits, num_layers, num_basis
        )

        # Initialize weights with PyTorch (between -pi and pi) and register as a learnable parameter
        self.weights = nn.Parameter(
            torch.rand(num_layers, (num_qubits * 2 - 1)) * 2 * torch.pi - torch.pi
        )
        self.coeffs = nn.Parameter(torch.rand(num_basis, self.z_dim))

    def forward(self, batch_size: int):
        sample_list = [
            torch.cat(
                [tensor.unsqueeze(0) for tensor in self.gen_circuit(self.weights)]
            )
            for _ in range(batch_size)
        ]
        noise = torch.stack(tuple(sample_list)).float()
        noise = torch.matmul(noise, self.coeffs)
        return noise

class QuantumGenerator(MLPGenerator):
    """Quantum Generator network of MolGAN"""

    def __init__(
        self,
        dataset,
        *,
        conv_dims=[128, 256],
        z_dim=8,
        dropout=0.0,
        use_shadows=False,
    ):
        super(QuantumGenerator, self).__init__(
            dataset,
            conv_dims=conv_dims,
            z_dim=z_dim,
            dropout=dropout,
        )
        if use_shadows:
            self.noise_generator = QuantumShadowNoise(z_dim)
        else:
            self.noise_generator = QuantumMolGanNoise(z_dim)

    def _generate_z(self, batch_size):
        return self.noise_generator(batch_size)
    
    

class MLPDiscriminator(nn.Module):
    def __init__(self, dataset, input_dim=2, hidden_dim=10, output_dim=1):
        super(MLPDiscriminator, self).__init__()
        self.first_linear = nn.Linear(input_dim, hidden_dim)
        self.first_dropout = nn.Dropout(0.5)
        self.second_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.first_linear(x))
        x = self.first_dropout(x)
        x = torch.sigmoid(self.second_linear(x))
        return x

class MLPGenerator(nn.Module):
    def __init__(self, z_dim=2, hidden_dims=[10], output_dim=1, use_conv=False):
        super(MLPGenerator, self).__init__()
        layers = []
        current_dim = z_dim

        for hdim in hidden_dims:
            layers.append(nn.Linear(current_dim, hdim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            current_dim = hdim

        layers.append(nn.Linear(current_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



# class MLPGenerator(nn.Module):
#     def __init__(self, z_dim, hidden_dims=[128, 64], use_conv=False):
#         super(MLPGenerator, self).__init__()
        
#         self.layers = []
#         self.current_dim = z_dim
        
#         # Optional convolutional layer
#         if use_conv:
#             self.conv = nn.Conv1d(1, 16, kernel_size=3, padding=1)
#             self.current_dim = (self.current_dim // 2) * 16
#             self.layers.append(nn.Flatten())
        
#         # Fully connected layers
#         for hidden_dim in hidden_dims:
#             self.layers.extend([
#                 nn.Linear(self.current_dim, hidden_dim),
#                 nn.Tanh(),
#                 nn.Dropout(0.1)
#             ])
#             self.current_dim = hidden_dim
            
#         # Output layer
#         self.layers.append(nn.Linear(self.current_dim, 1))
        
#         self.model = nn.Sequential(*self.layers)
#         self.use_conv = use_conv


#     def forward(self, x):
#         if self.use_conv:
#             x = x.unsqueeze(1)  # Add channel dimension
#             x = self.conv(x)
#             x = x.flatten(start_dim=1)  # Flatten for linear layers
        
#         x = self.model(x)
#         return x
    



    

# class Generator(nn.Module):
#     """Generator network of MolGAN"""

#     def __init__(
#         self,
#         dataset,
#         *,
#         conv_dims=[128, 256],
#         z_dim=8,
#         dropout=0.0,
#     ):
#         super(Generator, self).__init__()
#         self.dataset = dataset
#         self.conv_dims = conv_dims
#         self.z_dim = z_dim
#         self.dropout = dropout

#         # question amine: what is this vertexes, edges, nodes in the context of the NN?
#         self.vertexes = 5
#         self.edges = 5
#         self.nodes = 5

        # self.multi_dense_layers = MLP(
        #     self.z_dim,
        #     self.conv_dims,
        #     activation_layer=nn.Tanh,
        #     dropout=self.dropout,
        # )

#         self.multi_dense_layers = MultiDenseLayers(
#             z_dim,
#             self.conv_dims,
#             nn.Tanh(),
#             dropout_rate=self.dropout,
#         )
#         # question amine: why self.conv_dims[-1]?
#         self.edges_layer = nn.Linear(
#             self.conv_dims[-1], self.edges * self.vertexes * self.vertexes
#         )
#         self.nodes_layer = nn.Linear(self.conv_dims[-1], self.vertexes * self.nodes)
#         self.dropout_layer = nn.Dropout(self.dropout)

#     def _generate_z(self, batch_size):
#         return torch.rand(batch_size, self.z_dim).to(next(self.parameters()).device)

#     def forward(self, batch_size):
#         z = self._generate_z(batch_size)
#         z = z.to(next(self.parameters()).device)
#         output = self.multi_dense_layers(z)
#         edges_logits = self.edges_layer(output).view(
#             -1, self.edges, self.vertexes, self.vertexes
#         )
#         edges_logits = (edges_logits + edges_logits.permute(0, 1, 3, 2)) / 2
#         edges_logits = self.dropout_layer(edges_logits.permute(0, 2, 3, 1))

#         nodes_logits = self.nodes_layer(output)
#         nodes_logits = self.dropout_layer(
#             nodes_logits.view(-1, self.vertexes, self.nodes)
#         )

#         return edges_logits, nodes_logits





# class Discriminator(nn.Module):
#     """Discriminator network of MolGAN"""

#     def __init__(
#         self,
#         dataset,
#         *,
#         conv_dims=[[128, 64], 128, [128, 64]],
#         with_features=False,
#         f_dim=0,
#         dropout=0.0,
#     ):
#         super(Discriminator, self).__init__()
#         self.dataset = dataset
#         self.conv_dims = conv_dims
#         self.with_features = with_features
#         self.f_dim = f_dim
#         self.dropout = dropout

#         self._initialize()

#     def _initialize(self):
#         # question amine: what is m_dim, b_dim in the context of the NN?

#         m_dim = 5
#         b_dim = 5
#         self.activation_f = nn.Tanh()
#         graph_conv_dim, aux_dim, linear_dim = self.conv_dims
#         self.gcn_layer = GraphConvolution(
#             m_dim, graph_conv_dim, b_dim, self.with_features, self.f_dim, self.dropout
#         )
#         self.agg_layer = GraphAggregation(
#             graph_conv_dim[-1] + m_dim,
#             aux_dim,
#             self.activation_f,
#             self.with_features,
#             self.f_dim,
#             self.dropout,
#         )
#         self.multi_dense_layers = MultiDenseLayers(
#             aux_dim,
#             linear_dim,
#             self.activation_f,
#             self.dropout,
#         )
#         self.output_layer = nn.Linear(linear_dim[-1], 1)

#     def forward(self, adjacency_tensor, hidden, node, activation=None):
#         adj = adjacency_tensor[:, :, :, 1:].permute(0, 3, 1, 2)
#         h = self.gcn_layer(node, adj, hidden)
#         h = self.agg_layer(node, h, hidden)

#         h = self.multi_dense_layers(h)

#         output = self.output_layer(h)
#         output = activation(output) if activation is not None else output

#         return output, h


# class GNNDiscriminator(nn.Module):
#     """Discriminator network for MolGAN."""

#     def __init__(self, dataset, conv_dim=128, dropout=0.1):
#         super().__init__()
#         self.dataset = dataset
#         self.conv_dim = conv_dim
#         self._initialize_layers()

#     def _initialize_layers(self):
#         m_dim = self.dataset.atom_num_types
#         b_dim = self.dataset.bond_num_types

#         self.gnn_layer = GATConv(
#             in_channels=m_dim, out_channels=self.conv_dim, edge_dim=b_dim
#         )
#         self.output_layer = nn.Linear(self.conv_dim, 1)

#     def forward(self, adjacency_tensor, hidden, node, activation=None):
#         # Extract batch graphs from adjacency and node feature tensors
#         batch_graphs = [
#             extract_graphs_from_features({"A": a, "X": x})
#             for a, x in zip(adjacency_tensor, node)
#         ]
#         batch_graphs = Batch.from_data_list(batch_graphs)

#         # Convert node and edge attributes to one-hot encodings
#         # m_dim = self.dataset.atom_num_types
#         # b_dim = self.dataset.bond_num_types
#         # batch_graphs.x = one_hot(batch_graphs.x, m_dim).float()
#         # batch_graphs.edge_attr = one_hot(batch_graphs.edge_attr, b_dim).float()

#         # Pass through GNN layer
#         batch_features = self.gnn_layer(
#             batch_graphs.x, batch_graphs.edge_index, batch_graphs.edge_attr
#         )

#         # Reshape and aggregate graph-level features
#         batch_features = batch_features.reshape(
#             batch_graphs.num_graphs, -1, self.conv_dim
#         )
#         batch_features = batch_features.mean(dim=1)

#         # Pass through output layer
#         output = self.output_layer(batch_features)

#         # Apply activation if provided
#         if activation is not None:
#             output = activation(output)

#         return output, None
