"""
Test data factories and generators for comprehensive testing.
Provides various data scenarios for testing quantum vs classical generators.
"""

import numpy as np
import torch
from typing import Tuple, List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import pickle
from pathlib import Path

from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal


class DistributionType(Enum):
    """Enumeration of supported distribution types."""
    GAUSSIAN_SINGLE = "gaussian_single"
    GAUSSIAN_MIXTURE = "gaussian_mixture"  
    UNIFORM = "uniform"
    SPIRAL = "spiral"
    MOONS = "moons"
    CIRCLES = "circles"
    CHECKERBOARD = "checkerboard"
    SWISS_ROLL = "swiss_roll"
    

@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    distribution_type: DistributionType
    n_samples: int = 1000
    n_dimensions: int = 2
    noise_level: float = 0.1
    random_state: int = 42
    # Gaussian mixture specific
    n_components: int = 2
    separation: float = 3.0
    # Distribution specific parameters
    params: Dict[str, Any] = None


class StatisticalDataFactory:
    """Factory for creating various statistical distributions for testing."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the data factory.
        
        Args:
            random_state: Random seed for reproducible data generation
        """
        self.random_state = random_state
        np.random.seed(random_state)
    
    def create_gaussian_mixture(
        self, 
        n_samples: int = 1000,
        n_components: int = 2,
        n_dimensions: int = 2,
        separation: float = 3.0,
        covariance_type: str = 'spherical',
        weights: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Create Gaussian mixture model data.
        
        Args:
            n_samples: Number of samples to generate
            n_components: Number of Gaussian components
            n_dimensions: Dimensionality of data
            separation: Distance between component centers
            covariance_type: Type of covariance ('spherical', 'diagonal', 'full')
            weights: Component weights (defaults to uniform)
            
        Returns:
            Tuple of (data, parameters) where parameters contain GMM info
        """
        if weights is None:
            weights = [1.0 / n_components] * n_components
        
        # Create component centers
        if n_components == 1:
            centers = [[0.0] * n_dimensions]
        elif n_components == 2:
            centers = [
                [-separation/2, 0.0] if n_dimensions == 2 else [-separation/2] + [0.0] * (n_dimensions-1),
                [separation/2, 0.0] if n_dimensions == 2 else [separation/2] + [0.0] * (n_dimensions-1)
            ]
        else:
            # Arrange components in a circle for higher numbers
            angles = np.linspace(0, 2*np.pi, n_components, endpoint=False)
            centers = []
            for angle in angles:
                center = [separation * np.cos(angle), separation * np.sin(angle)]
                if n_dimensions > 2:
                    center.extend([0.0] * (n_dimensions - 2))
                centers.append(center)
        
        # Create covariance matrices
        if covariance_type == 'spherical':
            covariances = [np.eye(n_dimensions) for _ in range(n_components)]
        elif covariance_type == 'diagonal':
            covariances = [
                np.diag(np.random.uniform(0.5, 2.0, n_dimensions)) 
                for _ in range(n_components)
            ]
        elif covariance_type == 'full':
            covariances = []
            for _ in range(n_components):
                # Generate positive definite covariance matrix
                A = np.random.randn(n_dimensions, n_dimensions)
                cov = np.dot(A, A.transpose()) + np.eye(n_dimensions) * 0.1
                covariances.append(cov)
        else:
            raise ValueError(f"Unknown covariance type: {covariance_type}")
        
        # Generate samples
        samples = []
        labels = []
        
        for i in range(n_components):
            n_component_samples = int(n_samples * weights[i])
            if i == n_components - 1:  # Ensure exact sample count
                n_component_samples = n_samples - len(samples)
                
            component_samples = np.random.multivariate_normal(
                centers[i], covariances[i], n_component_samples
            )
            samples.extend(component_samples)
            labels.extend([i] * n_component_samples)
        
        data = np.array(samples)
        labels = np.array(labels)
        
        # Shuffle the data
        indices = np.random.permutation(len(data))
        data = data[indices]
        labels = labels[indices]
        
        parameters = {
            'centroids': centers,
            'covariances': covariances,
            'weights': weights,
            'labels': labels,
            'n_components': n_components,
            'distribution_type': 'gaussian_mixture'
        }
        
        return data, parameters
    
    def create_spiral_data(
        self,
        n_samples: int = 1000,
        noise: float = 0.1,
        n_spirals: int = 2
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Create spiral-shaped data."""
        n_per_spiral = n_samples // n_spirals
        data = []
        labels = []
        
        for spiral_idx in range(n_spirals):
            t = np.linspace(0, 4 * np.pi, n_per_spiral)
            # Offset each spiral
            t_offset = t + spiral_idx * (2 * np.pi / n_spirals)
            
            x = t_offset * np.cos(t_offset) * 0.1
            y = t_offset * np.sin(t_offset) * 0.1
            
            # Add noise
            x += np.random.normal(0, noise, len(x))
            y += np.random.normal(0, noise, len(y))
            
            spiral_data = np.column_stack([x, y])
            data.append(spiral_data)
            labels.extend([spiral_idx] * len(spiral_data))
        
        data = np.vstack(data)
        labels = np.array(labels)
        
        # Shuffle
        indices = np.random.permutation(len(data))
        data = data[indices]
        labels = labels[indices]
        
        parameters = {
            'n_spirals': n_spirals,
            'noise_level': noise,
            'labels': labels,
            'distribution_type': 'spiral'
        }
        
        return data, parameters
    
    def create_moons_data(
        self,
        n_samples: int = 1000,
        noise: float = 0.1,
        separation: float = 1.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Create two interleaving moons."""
        n_per_moon = n_samples // 2
        
        # First moon
        t1 = np.linspace(0, np.pi, n_per_moon)
        x1 = np.cos(t1)
        y1 = np.sin(t1)
        
        # Second moon (offset and flipped)
        t2 = np.linspace(0, np.pi, n_samples - n_per_moon)
        x2 = 1 - np.cos(t2) + separation
        y2 = -np.sin(t2) - 0.5
        
        # Combine and add noise
        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])
        
        x += np.random.normal(0, noise, len(x))
        y += np.random.normal(0, noise, len(y))
        
        data = np.column_stack([x, y])
        labels = np.concatenate([np.zeros(n_per_moon), np.ones(n_samples - n_per_moon)])
        
        # Shuffle
        indices = np.random.permutation(len(data))
        data = data[indices]
        labels = labels[indices]
        
        parameters = {
            'separation': separation,
            'noise_level': noise,
            'labels': labels,
            'distribution_type': 'moons'
        }
        
        return data, parameters
    
    def create_circles_data(
        self,
        n_samples: int = 1000,
        noise: float = 0.1,
        factor: float = 0.5
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Create two concentric circles."""
        n_outer = n_samples // 2
        n_inner = n_samples - n_outer
        
        # Outer circle
        theta_outer = np.random.uniform(0, 2*np.pi, n_outer)
        x_outer = np.cos(theta_outer)
        y_outer = np.sin(theta_outer)
        
        # Inner circle
        theta_inner = np.random.uniform(0, 2*np.pi, n_inner)
        x_inner = factor * np.cos(theta_inner)
        y_inner = factor * np.sin(theta_inner)
        
        # Combine and add noise
        x = np.concatenate([x_outer, x_inner])
        y = np.concatenate([y_outer, y_inner])
        
        x += np.random.normal(0, noise, len(x))
        y += np.random.normal(0, noise, len(y))
        
        data = np.column_stack([x, y])
        labels = np.concatenate([np.ones(n_outer), np.zeros(n_inner)])
        
        # Shuffle
        indices = np.random.permutation(len(data))
        data = data[indices]
        labels = labels[indices]
        
        parameters = {
            'factor': factor,
            'noise_level': noise,
            'labels': labels,
            'distribution_type': 'circles'
        }
        
        return data, parameters
    
    def create_checkerboard_data(
        self,
        n_samples: int = 1000,
        n_clusters_per_side: int = 3,
        noise: float = 0.1
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Create checkerboard pattern data."""
        data = []
        labels = []
        
        samples_per_square = n_samples // (n_clusters_per_side ** 2)
        
        for i in range(n_clusters_per_side):
            for j in range(n_clusters_per_side):
                # Determine if this square should have data (checkerboard pattern)
                if (i + j) % 2 == 0:
                    # Generate samples in this square
                    x = np.random.uniform(i, i + 1, samples_per_square)
                    y = np.random.uniform(j, j + 1, samples_per_square)
                    
                    # Add noise
                    x += np.random.normal(0, noise, len(x))
                    y += np.random.normal(0, noise, len(y))
                    
                    square_data = np.column_stack([x, y])
                    data.append(square_data)
                    labels.extend([i * n_clusters_per_side + j] * len(square_data))
        
        data = np.vstack(data)
        labels = np.array(labels)
        
        # Shuffle
        indices = np.random.permutation(len(data))
        data = data[indices]
        labels = labels[indices]
        
        parameters = {
            'n_clusters_per_side': n_clusters_per_side,
            'noise_level': noise,
            'labels': labels,
            'distribution_type': 'checkerboard'
        }
        
        return data, parameters
    
    def create_swiss_roll_data(
        self,
        n_samples: int = 1000,
        noise: float = 0.1,
        hole: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Create Swiss roll manifold data."""
        t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
        height = 21 * np.random.rand(n_samples)
        
        x = t * np.cos(t)
        y = height
        z = t * np.sin(t)
        
        # Add noise
        x += np.random.normal(0, noise, n_samples)
        y += np.random.normal(0, noise, n_samples) 
        z += np.random.normal(0, noise, n_samples)
        
        if hole:
            # Remove samples from the center to create a hole
            distances = np.sqrt(x**2 + z**2)
            valid_indices = distances > 5  # Remove center samples
            x, y, z = x[valid_indices], y[valid_indices], z[valid_indices]
            t = t[valid_indices]
        
        # For 2D visualization, use x and y
        data = np.column_stack([x, y])
        
        parameters = {
            'noise_level': noise,
            'hole': hole,
            't_values': t,  # Parameter values for coloring
            '3d_coordinates': np.column_stack([x, y, z]),
            'distribution_type': 'swiss_roll'
        }
        
        return data, parameters


class EdgeCaseDataFactory:
    """Factory for creating edge case and challenging test data."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def create_empty_dataset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Create empty dataset for edge case testing."""
        return np.empty((0, 2)), {'distribution_type': 'empty'}
    
    def create_single_point_dataset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Create dataset with single point."""
        return np.array([[0.0, 0.0]]), {'distribution_type': 'single_point'}
    
    def create_nan_dataset(self, n_samples: int = 100) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Create dataset with NaN values."""
        data = np.random.randn(n_samples, 2)
        # Introduce NaN values
        nan_indices = np.random.choice(n_samples, size=n_samples//10, replace=False)
        data[nan_indices] = np.nan
        
        return data, {'distribution_type': 'nan_data', 'nan_count': len(nan_indices)}
    
    def create_infinite_dataset(self, n_samples: int = 100) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Create dataset with infinite values."""
        data = np.random.randn(n_samples, 2)
        # Introduce infinite values
        inf_indices = np.random.choice(n_samples, size=n_samples//20, replace=False)
        data[inf_indices, 0] = np.inf
        data[inf_indices[::2], 1] = -np.inf  # Mix positive and negative infinity
        
        return data, {'distribution_type': 'infinite_data', 'inf_count': len(inf_indices)}
    
    def create_extreme_scale_dataset(self, n_samples: int = 1000) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Create dataset with extreme scale differences."""
        # One dimension very small, other very large
        x = np.random.normal(0, 1e-10, n_samples)  # Very small scale
        y = np.random.normal(0, 1e10, n_samples)   # Very large scale
        
        data = np.column_stack([x, y])
        return data, {'distribution_type': 'extreme_scale'}
    
    def create_high_dimensional_dataset(
        self, 
        n_samples: int = 1000,
        n_dimensions: int = 50
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Create high-dimensional dataset."""
        data = np.random.randn(n_samples, n_dimensions)
        return data, {'distribution_type': 'high_dimensional', 'n_dimensions': n_dimensions}
    
    def create_collinear_dataset(self, n_samples: int = 1000) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Create dataset with perfect collinearity."""
        x = np.random.randn(n_samples)
        y = 2 * x + 3  # Perfect linear relationship
        
        data = np.column_stack([x, y])
        return data, {'distribution_type': 'collinear'}
    
    def create_constant_dataset(self, n_samples: int = 1000) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Create dataset with constant values."""
        data = np.full((n_samples, 2), [1.5, -2.3])
        return data, {'distribution_type': 'constant'}


class ConfigurableDataFactory:
    """Factory that combines multiple data generation approaches with configuration."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.statistical_factory = StatisticalDataFactory(random_state)
        self.edge_case_factory = EdgeCaseDataFactory(random_state)
        
    def create_from_config(self, config: DatasetConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Create dataset from configuration object."""
        np.random.seed(config.random_state)
        
        if config.distribution_type == DistributionType.GAUSSIAN_SINGLE:
            return self.statistical_factory.create_gaussian_mixture(
                n_samples=config.n_samples,
                n_components=1,
                n_dimensions=config.n_dimensions
            )
        elif config.distribution_type == DistributionType.GAUSSIAN_MIXTURE:
            return self.statistical_factory.create_gaussian_mixture(
                n_samples=config.n_samples,
                n_components=config.n_components,
                n_dimensions=config.n_dimensions,
                separation=config.separation
            )
        elif config.distribution_type == DistributionType.SPIRAL:
            return self.statistical_factory.create_spiral_data(
                n_samples=config.n_samples,
                noise=config.noise_level
            )
        elif config.distribution_type == DistributionType.MOONS:
            return self.statistical_factory.create_moons_data(
                n_samples=config.n_samples,
                noise=config.noise_level
            )
        elif config.distribution_type == DistributionType.CIRCLES:
            return self.statistical_factory.create_circles_data(
                n_samples=config.n_samples,
                noise=config.noise_level
            )
        elif config.distribution_type == DistributionType.CHECKERBOARD:
            return self.statistical_factory.create_checkerboard_data(
                n_samples=config.n_samples,
                noise=config.noise_level
            )
        elif config.distribution_type == DistributionType.SWISS_ROLL:
            return self.statistical_factory.create_swiss_roll_data(
                n_samples=config.n_samples,
                noise=config.noise_level
            )
        else:
            raise ValueError(f"Unknown distribution type: {config.distribution_type}")
    
    def create_test_suite(self) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
        """Create comprehensive test suite with various distributions."""
        test_suite = {}
        
        # Standard distributions
        distributions = [
            DistributionType.GAUSSIAN_SINGLE,
            DistributionType.GAUSSIAN_MIXTURE,
            DistributionType.SPIRAL,
            DistributionType.MOONS,
            DistributionType.CIRCLES
        ]
        
        for dist_type in distributions:
            config = DatasetConfig(
                distribution_type=dist_type,
                n_samples=500,  # Smaller for test suite
                random_state=self.random_state
            )
            
            data, params = self.create_from_config(config)
            test_suite[dist_type.value] = (data, params)
        
        # Edge cases
        edge_cases = {
            'empty': self.edge_case_factory.create_empty_dataset(),
            'single_point': self.edge_case_factory.create_single_point_dataset(),
            'nan_data': self.edge_case_factory.create_nan_dataset(100),
            'infinite_data': self.edge_case_factory.create_infinite_dataset(100),
            'extreme_scale': self.edge_case_factory.create_extreme_scale_dataset(200),
            'collinear': self.edge_case_factory.create_collinear_dataset(200),
            'constant': self.edge_case_factory.create_constant_dataset(200)
        }
        
        test_suite.update(edge_cases)
        return test_suite
    
    def save_dataset(
        self, 
        data: np.ndarray, 
        parameters: Dict[str, Any], 
        filepath: Union[str, Path]
    ):
        """Save dataset to disk in pickle format."""
        filepath = Path(filepath)
        
        dataset = {
            'data': data,
            'parameters': parameters
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f)
    
    def load_dataset(self, filepath: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load dataset from disk."""
        filepath = Path(filepath)
        
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        
        return dataset['data'], dataset['parameters']


class BenchmarkDataFactory:
    """Factory for creating standardized benchmark datasets."""
    
    BENCHMARK_DATASETS = {
        'easy_gaussian': DatasetConfig(
            distribution_type=DistributionType.GAUSSIAN_MIXTURE,
            n_samples=1000,
            n_components=2,
            separation=4.0,
            noise_level=0.1
        ),
        'medium_gaussian': DatasetConfig(
            distribution_type=DistributionType.GAUSSIAN_MIXTURE,
            n_samples=1000,
            n_components=3,
            separation=2.0,
            noise_level=0.2
        ),
        'hard_gaussian': DatasetConfig(
            distribution_type=DistributionType.GAUSSIAN_MIXTURE,
            n_samples=1000,
            n_components=5,
            separation=1.0,
            noise_level=0.3
        ),
        'spiral_easy': DatasetConfig(
            distribution_type=DistributionType.SPIRAL,
            n_samples=1000,
            noise_level=0.1
        ),
        'spiral_hard': DatasetConfig(
            distribution_type=DistributionType.SPIRAL,
            n_samples=1000,
            noise_level=0.3
        ),
        'moons': DatasetConfig(
            distribution_type=DistributionType.MOONS,
            n_samples=1000,
            noise_level=0.15
        )
    }
    
    def __init__(self, random_state: int = 42):
        self.factory = ConfigurableDataFactory(random_state)
    
    def get_benchmark_dataset(self, name: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Get a standardized benchmark dataset by name."""
        if name not in self.BENCHMARK_DATASETS:
            available = ', '.join(self.BENCHMARK_DATASETS.keys())
            raise ValueError(f"Unknown benchmark dataset: {name}. Available: {available}")
        
        config = self.BENCHMARK_DATASETS[name]
        return self.factory.create_from_config(config)
    
    def get_all_benchmark_datasets(self) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
        """Get all benchmark datasets."""
        datasets = {}
        for name in self.BENCHMARK_DATASETS:
            datasets[name] = self.get_benchmark_dataset(name)
        return datasets
    
    def create_difficulty_progression(self) -> List[Tuple[str, np.ndarray, Dict[str, Any]]]:
        """Create datasets with increasing difficulty for progressive testing."""
        progression = [
            'easy_gaussian',
            'moons', 
            'spiral_easy',
            'medium_gaussian',
            'spiral_hard',
            'hard_gaussian'
        ]
        
        datasets = []
        for name in progression:
            data, params = self.get_benchmark_dataset(name)
            datasets.append((name, data, params))
        
        return datasets


# Utility functions for PyTest integration
def torch_tensor_from_numpy(data: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    """Convert numpy array to PyTorch tensor."""
    return torch.tensor(data, dtype=torch.float32, device=device)


def create_pytorch_dataloader(
    data: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    device: str = 'cpu'
) -> torch.utils.data.DataLoader:
    """Create PyTorch DataLoader from numpy data."""
    from torch.utils.data import TensorDataset, DataLoader
    
    tensor_data = torch_tensor_from_numpy(data, device)
    # Create dummy labels (zeros) for compatibility with GAN training
    labels = torch.zeros(len(tensor_data), device=device)
    
    dataset = TensorDataset(tensor_data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Export main classes and functions
__all__ = [
    'StatisticalDataFactory',
    'EdgeCaseDataFactory', 
    'ConfigurableDataFactory',
    'BenchmarkDataFactory',
    'DatasetConfig',
    'DistributionType',
    'torch_tensor_from_numpy',
    'create_pytorch_dataloader'
]