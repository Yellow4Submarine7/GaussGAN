"""
Pytest configuration and shared fixtures for GaussGAN testing.
Provides common test utilities, fixtures, and configuration for comprehensive testing.
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
import yaml
import pickle
from unittest.mock import Mock

# Import project modules
from source.nn import (
    ClassicalNoise, 
    QuantumNoise, 
    QuantumShadowNoise,
    MLPGenerator,
    MLPDiscriminator
)
from source.model import GaussGan
from source.metrics import LogLikelihood, KLDivergence, IsPositive
from source.data import GaussianDataModule


@pytest.fixture(scope="session")
def device():
    """Provide appropriate device for testing (CPU for CI, GPU if available)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps") 
    else:
        return torch.device("cpu")


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration with optimized parameters for fast testing."""
    return {
        'z_dim': 4,
        'generator_type': 'classical_normal',
        'max_epochs': 5,  # Reduced for testing
        'batch_size': 64,  # Smaller batch for testing
        'learning_rate': 0.001,
        'grad_penalty': 0.2,
        'n_critic': 2,  # Reduced for testing
        'n_predictor': 2,
        'rl_weight': 10,
        'killer': False,
        'validation_samples': 100,  # Reduced for testing
        'nn_gen': [64, 64],  # Smaller networks for testing
        'nn_disc': [64, 64],
        'nn_validator': [32, 32],
        'non_linearity': 'LeakyReLU',
        'std_scale': 1.1,
        'min_std': 0.5,
        'quantum_qubits': 4,  # Reduced for testing
        'quantum_layers': 1,  # Reduced for testing
        'quantum_basis': 2,   # Reduced for testing
        'quantum_shots': 50,  # Reduced for testing
        'accelerator': 'cpu',
        'seed': 42
    }


@pytest.fixture(scope="session")
def benchmark_config():
    """Provide configuration optimized for performance benchmarking."""
    return {
        'z_dim': 4,
        'batch_size': 256,
        'max_epochs': 20,
        'learning_rate': 0.001,
        'validation_samples': 1000,
        'nn_gen': [256, 256],
        'nn_disc': [256, 256],
        'quantum_qubits': 6,
        'quantum_layers': 2,
        'quantum_basis': 3,
        'quantum_shots': 100
    }


@pytest.fixture
def test_data_2d():
    """Generate simple 2D test data for validation."""
    np.random.seed(42)
    # Create a simple 2-component Gaussian mixture
    n_samples = 500
    
    # Component 1: centered at (0, 0)
    comp1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], n_samples // 2)
    # Component 2: centered at (3, 3)  
    comp2 = np.random.multivariate_normal([3, 3], [[1, 0], [0, 1]], n_samples // 2)
    
    data = np.vstack([comp1, comp2])
    return torch.tensor(data, dtype=torch.float32)


@pytest.fixture
def gaussian_mixture_params():
    """Provide parameters for a 2-component Gaussian mixture model."""
    return {
        'centroids': [[0, 0], [3, 3]],
        'cov_matrices': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]],
        'weights': [0.5, 0.5]
    }


@pytest.fixture
def classical_generator(test_config, device):
    """Create classical noise generator for testing."""
    noise_gen = ClassicalNoise(
        z_dim=test_config['z_dim'],
        generator_type='classical_normal'
    )
    mlp_gen = MLPGenerator(
        non_linearity=test_config['non_linearity'],
        hidden_dims=test_config['nn_gen'],
        z_dim=test_config['z_dim'],
        std_scale=test_config['std_scale'],
        min_std=test_config['min_std']
    )
    generator = torch.nn.Sequential(noise_gen, mlp_gen)
    return generator.to(device)


@pytest.fixture
def quantum_generator(test_config, device):
    """Create quantum noise generator for testing."""
    try:
        noise_gen = QuantumNoise(
            num_qubits=test_config['z_dim'],
            num_layers=test_config['quantum_layers']
        )
        mlp_gen = MLPGenerator(
            non_linearity=test_config['non_linearity'],
            hidden_dims=test_config['nn_gen'],
            z_dim=test_config['z_dim'],
            std_scale=test_config['std_scale'],
            min_std=test_config['min_std']
        )
        generator = torch.nn.Sequential(noise_gen, mlp_gen)
        return generator.to(device)
    except Exception as e:
        pytest.skip(f"Quantum generator not available: {e}")


@pytest.fixture
def quantum_shadow_generator(test_config, device):
    """Create quantum shadow noise generator for testing."""
    try:
        noise_gen = QuantumShadowNoise(
            z_dim=test_config['z_dim'],
            num_qubits=test_config['quantum_qubits'],
            num_layers=test_config['quantum_layers'],
            num_basis=test_config['quantum_basis']
        )
        mlp_gen = MLPGenerator(
            non_linearity=test_config['non_linearity'],
            hidden_dims=test_config['nn_gen'],
            z_dim=test_config['z_dim'],
            std_scale=test_config['std_scale'],
            min_std=test_config['min_std']
        )
        generator = torch.nn.Sequential(noise_gen, mlp_gen)
        return generator.to(device)
    except Exception as e:
        pytest.skip(f"Quantum shadow generator not available: {e}")


@pytest.fixture
def discriminator(test_config, device):
    """Create discriminator for testing."""
    disc = MLPDiscriminator(
        non_linearity=test_config['non_linearity'],
        hidden_dims=test_config['nn_disc']
    )
    return disc.to(device)


@pytest.fixture
def predictor(test_config, device):
    """Create predictor/value network for testing."""
    pred = MLPDiscriminator(
        non_linearity=test_config['non_linearity'],
        hidden_dims=test_config['nn_validator']
    )
    return pred.to(device)


@pytest.fixture
def gauss_gan_model(classical_generator, discriminator, predictor, 
                   test_config, gaussian_mixture_params, device):
    """Create complete GaussGan model for testing."""
    from functools import partial
    
    model = GaussGan(
        generator=classical_generator,
        discriminator=discriminator,
        predictor=predictor,
        optimizer=partial(torch.optim.Adam, lr=test_config['learning_rate']),
        killer=test_config['killer'],
        n_critic=test_config['n_critic'],
        grad_penalty=test_config['grad_penalty'],
        rl_weight=test_config['rl_weight'],
        n_predictor=test_config['n_predictor'],
        metrics=['LogLikelihood', 'KLDivergence', 'IsPositive'],
        gaussians=gaussian_mixture_params,
        validation_samples=test_config['validation_samples'],
        non_linearity=test_config['non_linearity']
    )
    return model.to(device)


@pytest.fixture
def datamodule(test_data_2d, test_config):
    """Create Lightning DataModule for testing."""
    # Mock datamodule with test data
    from torch.utils.data import DataLoader, TensorDataset
    
    class TestDataModule:
        def __init__(self, data, batch_size):
            self.data = data
            self.batch_size = batch_size
            
        def train_dataloader(self):
            dataset = TensorDataset(self.data, torch.zeros(len(self.data)))
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
        def val_dataloader(self):
            dataset = TensorDataset(self.data, torch.zeros(len(self.data)))
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
    
    return TestDataModule(test_data_2d, test_config['batch_size'])


@pytest.fixture
def metrics_suite(gaussian_mixture_params):
    """Create metric instances for testing."""
    return {
        'log_likelihood': LogLikelihood(**gaussian_mixture_params),
        'kl_divergence': KLDivergence(**gaussian_mixture_params),
        'is_positive': IsPositive()
    }


@pytest.fixture
def temp_directory():
    """Provide temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_mlflow_logger():
    """Mock MLFlow logger for testing."""
    logger = Mock()
    logger.run_id = "test_run_id"
    logger.experiment = Mock()
    logger.log_hyperparams = Mock()
    return logger


# Test data factories
class TestDataFactory:
    """Factory for creating various types of test data."""
    
    @staticmethod
    def create_gaussian_mixture(n_components=2, n_samples=1000, seed=42):
        """Create Gaussian mixture data."""
        np.random.seed(seed)
        data_points = []
        
        for i in range(n_components):
            center = [i * 3, i * 3]  # Spread components apart
            cov = [[1, 0], [0, 1]]   # Unit covariance
            samples = np.random.multivariate_normal(
                center, cov, n_samples // n_components
            )
            data_points.append(samples)
            
        return torch.tensor(np.vstack(data_points), dtype=torch.float32)
    
    @staticmethod
    def create_edge_case_data():
        """Create data with edge cases (NaN, inf, etc.)."""
        # Mix of normal data with edge cases
        normal_data = np.random.randn(100, 2)
        edge_data = np.array([
            [np.nan, 1.0],        # NaN values
            [np.inf, 2.0],        # Infinite values
            [-np.inf, 3.0],       # Negative infinite
            [1e10, 1e10],         # Very large values
            [1e-10, 1e-10],       # Very small values
        ])
        
        combined = np.vstack([normal_data, edge_data])
        return torch.tensor(combined, dtype=torch.float32)
    
    @staticmethod
    def create_challenging_distribution(complexity_level='medium'):
        """Create challenging distributions for testing."""
        if complexity_level == 'simple':
            # Single Gaussian
            return torch.randn(1000, 2)
        elif complexity_level == 'medium':
            # Multi-modal with different scales
            comp1 = torch.randn(300, 2) * 0.5
            comp2 = torch.randn(300, 2) * 2.0 + torch.tensor([5, 5])
            comp3 = torch.randn(400, 2) * 1.0 + torch.tensor([-3, 2])
            return torch.vstack([comp1, comp2, comp3])
        elif complexity_level == 'hard':
            # Highly non-linear distribution
            t = torch.linspace(0, 4*np.pi, 1000)
            x = t * torch.cos(t) * 0.1
            y = t * torch.sin(t) * 0.1
            noise = torch.randn(1000, 2) * 0.1
            spiral = torch.stack([x, y], dim=1) + noise
            return spiral


@pytest.fixture
def test_data_factory():
    """Provide test data factory."""
    return TestDataFactory


# Performance testing utilities
@pytest.fixture
def performance_monitor():
    """Provide utilities for performance monitoring."""
    import time
    import psutil
    import torch
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            
        def start_monitoring(self):
            self.start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                self.start_memory = torch.cuda.memory_allocated()
            else:
                self.start_memory = psutil.Process().memory_info().rss
                
        def get_metrics(self):
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated()
                current_memory = torch.cuda.memory_allocated()
            else:
                current_memory = psutil.Process().memory_info().rss
                peak_memory = current_memory  # Approximate for CPU
                
            return {
                'elapsed_time': elapsed_time,
                'peak_memory_mb': peak_memory / 1024 / 1024,
                'current_memory_mb': current_memory / 1024 / 1024,
                'memory_increase_mb': (current_memory - self.start_memory) / 1024 / 1024
            }
    
    return PerformanceMonitor


# Statistical testing utilities
@pytest.fixture  
def statistical_tester():
    """Provide statistical testing utilities."""
    from scipy import stats
    
    class StatisticalTester:
        @staticmethod
        def test_distribution_similarity(data1, data2, alpha=0.05):
            """Test if two datasets come from similar distributions."""
            # Kolmogorov-Smirnov test for each dimension
            p_values = []
            for dim in range(data1.shape[1]):
                _, p_value = stats.ks_2samp(data1[:, dim], data2[:, dim])
                p_values.append(p_value)
            return np.array(p_values), np.all(np.array(p_values) > alpha)
        
        @staticmethod
        def test_performance_regression(current_metrics, baseline_metrics, 
                                     threshold=0.05, alpha=0.05):
            """Test if current performance is significantly worse than baseline."""
            improvements = []
            p_values = []
            
            for metric_name in baseline_metrics:
                if metric_name in current_metrics:
                    current = np.array(current_metrics[metric_name])
                    baseline = np.array(baseline_metrics[metric_name])
                    
                    # Paired t-test (assuming paired samples)
                    if len(current) == len(baseline):
                        _, p_value = stats.ttest_rel(current, baseline)
                    else:
                        _, p_value = stats.ttest_ind(current, baseline)
                    
                    improvement = (np.mean(current) - np.mean(baseline)) / np.mean(baseline)
                    improvements.append(improvement)
                    p_values.append(p_value)
            
            significant_regression = any(
                imp < -threshold and p_val < alpha 
                for imp, p_val in zip(improvements, p_values)
            )
            
            return {
                'improvements': improvements,
                'p_values': p_values,
                'significant_regression': significant_regression
            }
        
        @staticmethod
        def calculate_effect_size(group1, group2):
            """Calculate Cohen's d effect size."""
            pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                                 (len(group2) - 1) * np.var(group2, ddof=1)) / 
                                (len(group1) + len(group2) - 2))
            return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    return StatisticalTester


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "quantum: marks tests requiring quantum backends")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "performance: marks performance benchmark tests")
    config.addinivalue_line("markers", "statistical: marks statistical validation tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names/paths."""
    for item in items:
        # Add quantum marker for quantum-related tests
        if "quantum" in item.name.lower() or "quantum" in str(item.fspath):
            item.add_marker(pytest.mark.quantum)
            
        # Add slow marker for performance tests
        if "performance" in item.name.lower() or "benchmark" in item.name.lower():
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.performance)
            
        # Add integration marker for integration tests
        if "integration" in str(item.fspath) or "end_to_end" in item.name.lower():
            item.add_marker(pytest.mark.integration)
            
        # Add statistical marker for statistical tests
        if "statistical" in str(item.fspath) or "significance" in item.name.lower():
            item.add_marker(pytest.mark.statistical)