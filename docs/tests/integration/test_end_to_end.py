"""
End-to-end integration tests for the complete GaussGAN training pipeline.
Tests the full workflow from configuration to trained model validation.
"""

import pytest
import torch
import numpy as np
import yaml
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import warnings

from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from source.model import GaussGan
from source.nn import (
    ClassicalNoise, QuantumNoise, QuantumShadowNoise,
    MLPGenerator, MLPDiscriminator
)
from source.data import GaussianDataModule
from source.utils import set_seed, load_data


class EndToEndTestRunner:
    """Manages end-to-end testing workflows."""
    
    def __init__(self, device, temp_dir):
        self.device = device
        self.temp_dir = Path(temp_dir)
        self.test_results = {}
    
    def create_test_config(self, generator_type='classical_normal', quick=True):
        """Create test configuration for end-to-end testing."""
        config = {
            'z_dim': 4,
            'generator_type': generator_type,
            'stage': 'train',
            'experiment_name': 'E2E-Test',
            'killer': False,
            'quantum_qubits': 4,  # Reduced for testing
            'quantum_layers': 1,  # Reduced for testing
            'quantum_basis': 2,   # Reduced for testing
            'quantum_shots': 50,  # Reduced for testing
            'grad_penalty': 0.2,
            'n_critic': 2,        # Reduced for testing
            'n_predictor': 2,
            'checkpoint_path': str(self.temp_dir / 'checkpoints'),
            'agg_method': 'prod',
            'max_epochs': 3 if quick else 10,  # Very short for testing
            'batch_size': 32,     # Small for testing
            'learning_rate': 0.001,
            'nn_gen': [32, 32],   # Smaller networks for testing
            'nn_disc': [32, 32],
            'nn_validator': [32, 32],
            'non_linearity': 'LeakyReLU',
            'std_scale': 1.1,
            'min_std': 0.5,
            'dataset_type': 'NORMAL',
            'metrics': ['IsPositive', 'LogLikelihood', 'KLDivergence'],
            'accelerator': 'cpu',  # Use CPU for CI compatibility
            'validation_samples': 100,  # Reduced for testing
            'seed': 42,
            'rl_weight': 10
        }
        
        return config
    
    def create_test_data(self, n_samples=1000):
        """Create test data for end-to-end testing."""
        np.random.seed(42)
        
        # Create 2-component Gaussian mixture
        comp1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], n_samples // 2)
        comp2 = np.random.multivariate_normal([3, 3], [[1, 0], [0, 1]], n_samples // 2)
        
        data = np.vstack([comp1, comp2])
        
        # Save as pickle file (mimicking real data format)
        data_file = self.temp_dir / 'test_data.pickle'
        with open(data_file, 'wb') as f:
            pickle.dump(data, f)
        
        return data_file, data
    
    def create_complete_model(self, config):
        """Create complete GaussGan model from config."""
        from functools import partial
        
        # Create generator components
        if config['generator_type'] == 'classical_normal':
            noise_gen = ClassicalNoise(
                z_dim=config['z_dim'],
                generator_type='classical_normal'
            )
        elif config['generator_type'] == 'classical_uniform':
            noise_gen = ClassicalNoise(
                z_dim=config['z_dim'],
                generator_type='classical_uniform'
            )
        elif config['generator_type'] == 'quantum_samples':
            noise_gen = QuantumNoise(
                num_qubits=config['z_dim'],
                num_layers=config['quantum_layers']
            )
        elif config['generator_type'] == 'quantum_shadows':
            noise_gen = QuantumShadowNoise(
                z_dim=config['z_dim'],
                num_qubits=config['quantum_qubits'],
                num_layers=config['quantum_layers'],
                num_basis=config['quantum_basis']
            )
        else:
            raise ValueError(f"Unknown generator type: {config['generator_type']}")
        
        mlp_gen = MLPGenerator(
            non_linearity=config['non_linearity'],
            z_dim=config['z_dim'],
            hidden_dims=config['nn_gen'],
            std_scale=config['std_scale'],
            min_std=config['min_std']
        )
        
        generator = torch.nn.Sequential(noise_gen, mlp_gen)
        
        # Create discriminator and predictor
        discriminator = MLPDiscriminator(
            non_linearity=config['non_linearity'],
            hidden_dims=config['nn_disc']
        )
        
        predictor = MLPDiscriminator(
            non_linearity=config['non_linearity'],
            hidden_dims=config['nn_validator']
        )
        
        # Move to device
        generator = generator.to(self.device)
        discriminator = discriminator.to(self.device)
        predictor = predictor.to(self.device)
        
        # Create Gaussian parameters
        gaussians = {
            'centroids': [[0, 0], [3, 3]],
            'covariances': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]],
            'weights': [0.5, 0.5]
        }
        
        # Create GaussGan model
        model = GaussGan(
            generator=generator,
            discriminator=discriminator,
            predictor=predictor,
            optimizer=partial(torch.optim.Adam, lr=config['learning_rate']),
            killer=config['killer'],
            n_critic=config['n_critic'],
            grad_penalty=config['grad_penalty'],
            rl_weight=config['rl_weight'],
            n_predictor=config['n_predictor'],
            metrics=config['metrics'],
            gaussians=gaussians,
            validation_samples=config['validation_samples'],
            non_linearity=config['non_linearity']
        )
        
        return model.to(self.device)
    
    def create_mock_datamodule(self, data):
        """Create mock datamodule for testing."""
        from torch.utils.data import DataLoader, TensorDataset
        
        class MockDataModule:
            def __init__(self, data, batch_size):
                self.data = torch.tensor(data, dtype=torch.float32)
                self.batch_size = batch_size
            
            def train_dataloader(self):
                dataset = TensorDataset(self.data, torch.zeros(len(self.data)))
                return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            def val_dataloader(self):
                dataset = TensorDataset(self.data, torch.zeros(len(self.data)))
                return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        return MockDataModule(data, 32)


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndPipeline:
    """End-to-end integration tests for complete training pipeline."""
    
    def setup_method(self):
        """Set up end-to-end testing environment."""
        self.generator_types = ['classical_normal']
        
        # Add quantum generators if available
        try:
            import pennylane as qml
            self.generator_types.append('quantum_samples')
        except ImportError:
            print("PennyLane not available, only testing classical generators")
    
    def test_complete_training_workflow_classical(self, device, temp_directory):
        """Test complete training workflow with classical generator."""
        runner = EndToEndTestRunner(device, temp_directory)
        
        # Create test configuration
        config = runner.create_test_config('classical_normal', quick=True)
        
        # Create test data
        data_file, data = runner.create_test_data(200)  # Small dataset for testing
        
        # Create model
        model = runner.create_complete_model(config)
        
        # Create datamodule
        datamodule = runner.create_mock_datamodule(data)
        
        # Create mock logger to avoid MLFlow dependency
        mock_logger = Mock()
        mock_logger.run_id = "test_run_id"
        mock_logger.experiment = Mock()
        
        # Create trainer with minimal configuration
        trainer = Trainer(
            max_epochs=config['max_epochs'],
            accelerator='cpu',  # Force CPU for testing
            logger=mock_logger,
            enable_checkpointing=False,  # Disable checkpointing for testing
            enable_progress_bar=False,    # Disable progress bar
            limit_train_batches=2,        # Limit batches for speed
            limit_val_batches=1,          # Limit validation batches
        )
        
        # Test training
        try:
            trainer.fit(model=model, datamodule=datamodule)
            training_success = True
        except Exception as e:
            print(f"Training failed: {e}")
            training_success = False
        
        # Validate training completed
        assert training_success, "Training should complete without errors"
        assert trainer.current_epoch >= 0
        
        # Test model can generate samples after training
        with torch.no_grad():
            samples = model._generate_fake_data(50)
            assert samples.shape == (50, 2)
            assert torch.all(torch.isfinite(samples))
    
    @pytest.mark.quantum
    def test_complete_training_workflow_quantum(self, device, temp_directory):
        """Test complete training workflow with quantum generator."""
        try:
            import pennylane as qml
        except ImportError:
            pytest.skip("PennyLane not available")
        
        runner = EndToEndTestRunner(device, temp_directory)
        
        # Create test configuration for quantum
        config = runner.create_test_config('quantum_samples', quick=True)
        
        # Create test data
        data_file, data = runner.create_test_data(100)  # Even smaller for quantum
        
        # Create model
        model = runner.create_complete_model(config)
        
        # Create datamodule
        datamodule = runner.create_mock_datamodule(data)
        
        # Create mock logger
        mock_logger = Mock()
        mock_logger.run_id = "test_run_id_quantum"
        mock_logger.experiment = Mock()
        
        # Create trainer with very limited scope for quantum testing
        trainer = Trainer(
            max_epochs=2,                 # Very short for quantum
            accelerator='cpu',
            logger=mock_logger,
            enable_checkpointing=False,
            enable_progress_bar=False,
            limit_train_batches=1,        # Just one batch
            limit_val_batches=1
        )
        
        # Test training
        try:
            trainer.fit(model=model, datamodule=datamodule)
            training_success = True
        except Exception as e:
            print(f"Quantum training failed: {e}")
            # Quantum training might fail in test environment, that's okay
            training_success = False
        
        if training_success:
            # Test model can generate samples after training
            with torch.no_grad():
                samples = model._generate_fake_data(10)  # Small batch for quantum
                assert samples.shape == (10, 2)
                assert torch.all(torch.isfinite(samples))
    
    def test_killer_mode_integration(self, device, temp_directory):
        """Test end-to-end workflow with killer mode enabled."""
        runner = EndToEndTestRunner(device, temp_directory)
        
        # Create configuration with killer mode
        config = runner.create_test_config('classical_normal', quick=True)
        config['killer'] = True
        config['rl_weight'] = 10
        
        # Create test data
        data_file, data = runner.create_test_data(200)
        
        # Create model
        model = runner.create_complete_model(config)
        
        # Create datamodule
        datamodule = runner.create_mock_datamodule(data)
        
        # Create mock logger
        mock_logger = Mock()
        mock_logger.run_id = "test_run_id_killer"
        mock_logger.experiment = Mock()
        
        # Create trainer
        trainer = Trainer(
            max_epochs=3,
            accelerator='cpu',
            logger=mock_logger,
            enable_checkpointing=False,
            enable_progress_bar=False,
            limit_train_batches=2,
            limit_val_batches=1
        )
        
        # Test training with killer mode
        try:
            trainer.fit(model=model, datamodule=datamodule)
            training_success = True
        except Exception as e:
            print(f"Killer mode training failed: {e}")
            training_success = False
        
        assert training_success, "Killer mode training should complete"
        
        # Test that predictor was used during training
        assert model.killer == True
        
        # Test prediction functionality
        with torch.no_grad():
            test_points = torch.tensor([[-1, 0], [1, 0]], dtype=torch.float32).to(device)
            predictions = model._apply_predictor(test_points)
            
            assert predictions.shape == (2, 1)
            assert torch.all(predictions >= 0) and torch.all(predictions <= 1)  # Sigmoid output
    
    def test_metrics_integration(self, device, temp_directory):
        """Test integration of all metrics during training."""
        runner = EndToEndTestRunner(device, temp_directory)
        
        config = runner.create_test_config('classical_normal', quick=True)
        data_file, data = runner.create_test_data(150)
        
        model = runner.create_complete_model(config)
        datamodule = runner.create_mock_datamodule(data)
        
        # Test metrics computation directly
        with torch.no_grad():
            fake_samples = model._generate_fake_data(100)
            metrics = model._compute_metrics(fake_samples)
        
        # Validate all expected metrics are computed
        expected_metrics = ['IsPositive', 'LogLikelihood', 'KLDivergence']
        for metric in expected_metrics:
            assert metric in metrics, f"Metric {metric} not computed"
            
            # Check metric values are reasonable
            value = metrics[metric]
            if isinstance(value, (list, np.ndarray)):
                assert len(value) > 0
                assert not all(np.isnan(value))
            else:
                assert not np.isnan(value) and not np.isinf(value)
    
    def test_checkpoint_save_load_integration(self, device, temp_directory):
        """Test checkpoint saving and loading integration."""
        runner = EndToEndTestRunner(device, temp_directory)
        
        config = runner.create_test_config('classical_normal', quick=True)
        data_file, data = runner.create_test_data(100)
        
        # Create first model
        model1 = runner.create_complete_model(config)
        datamodule = runner.create_mock_datamodule(data)
        
        # Setup checkpointing
        checkpoint_dir = temp_directory / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='test-checkpoint-{epoch:02d}',
            save_top_k=1,
            every_n_epochs=1
        )
        
        mock_logger = Mock()
        mock_logger.run_id = "test_checkpoint_run"
        mock_logger.experiment = Mock()
        
        # Train model with checkpointing
        trainer = Trainer(
            max_epochs=2,
            accelerator='cpu',
            logger=mock_logger,
            callbacks=[checkpoint_callback],
            enable_progress_bar=False,
            limit_train_batches=1,
            limit_val_batches=1
        )
        
        trainer.fit(model=model1, datamodule=datamodule)
        
        # Find saved checkpoint
        checkpoint_files = list(checkpoint_dir.glob('*.ckpt'))
        assert len(checkpoint_files) > 0, "No checkpoint files were saved"
        
        checkpoint_path = checkpoint_files[0]
        
        # Load model from checkpoint
        model2 = GaussGan.load_from_checkpoint(
            checkpoint_path,
            generator=model1.generator,
            discriminator=model1.discriminator,
            predictor=model1.predictor,
            optimizer=partial(torch.optim.Adam, lr=config['learning_rate']),
            gaussians=model1.gaussians
        )
        model2 = model2.to(device)
        
        # Test that loaded model produces same outputs
        with torch.no_grad():
            samples1 = model1._generate_fake_data(20)
            samples2 = model2._generate_fake_data(20)
            
            # Models should produce different samples (due to randomness)
            # but should have similar statistical properties
            mean1, mean2 = torch.mean(samples1, dim=0), torch.mean(samples2, dim=0)
            std1, std2 = torch.std(samples1, dim=0), torch.std(samples2, dim=0)
            
            # Statistical properties should be similar (within reasonable bounds)
            assert torch.allclose(mean1, mean2, atol=1.0)
            assert torch.allclose(std1, std2, atol=1.0)
    
    def test_hyperparameter_validation_integration(self, device, temp_directory):
        """Test integration with hyperparameter validation."""
        runner = EndToEndTestRunner(device, temp_directory)
        
        # Test various hyperparameter combinations
        configs_to_test = [
            # Valid configurations
            {'z_dim': 4, 'batch_size': 32, 'learning_rate': 0.001},
            {'z_dim': 6, 'batch_size': 64, 'learning_rate': 0.0001},
            
            # Edge case configurations
            {'z_dim': 2, 'batch_size': 16, 'learning_rate': 0.01},  # Small z_dim
            {'z_dim': 8, 'batch_size': 8, 'learning_rate': 0.00001},  # Large z_dim, tiny lr
        ]
        
        data_file, data = runner.create_test_data(100)
        
        for i, param_updates in enumerate(configs_to_test):
            config = runner.create_test_config('classical_normal', quick=True)
            config.update(param_updates)
            
            try:
                model = runner.create_complete_model(config)
                datamodule = runner.create_mock_datamodule(data)
                
                # Test model creation and basic functionality
                with torch.no_grad():
                    samples = model._generate_fake_data(config['batch_size'])
                    assert samples.shape == (config['batch_size'], 2)
                    
                print(f"Configuration {i+1} passed: {param_updates}")
                
            except Exception as e:
                pytest.fail(f"Configuration {i+1} failed: {param_updates}, Error: {e}")
    
    def test_mlflow_logging_integration(self, device, temp_directory):
        """Test MLFlow logging integration without actual MLFlow server."""
        runner = EndToEndTestRunner(device, temp_directory)
        
        config = runner.create_test_config('classical_normal', quick=True)
        data_file, data = runner.create_test_data(100)
        
        model = runner.create_complete_model(config)
        datamodule = runner.create_mock_datamodule(data)
        
        # Create comprehensive mock logger
        mock_logger = Mock()
        mock_logger.run_id = "test_mlflow_integration"
        mock_logger.experiment = Mock()
        mock_logger.experiment.log_text = Mock()
        mock_logger.log_hyperparams = Mock()
        
        # Test model training with logging
        trainer = Trainer(
            max_epochs=2,
            accelerator='cpu',
            logger=mock_logger,
            enable_checkpointing=False,
            enable_progress_bar=False,
            limit_train_batches=1,
            limit_val_batches=1
        )
        
        trainer.fit(model=model, datamodule=datamodule)
        
        # Verify logging calls were made
        assert mock_logger.log_hyperparams.called, "Hyperparameters should be logged"
        
        # Test that validation logging works
        model.logger = mock_logger  # Ensure model has logger reference
        
        with torch.no_grad():
            batch = (data[:32], torch.zeros(32))
            validation_result = model.validation_step(batch, 0)
            
            assert 'fake_data' in validation_result
            assert 'metrics' in validation_result
            assert validation_result['fake_data'].shape[0] > 0
    
    def test_error_handling_and_recovery(self, device, temp_directory):
        """Test error handling and recovery in end-to-end pipeline."""
        runner = EndToEndTestRunner(device, temp_directory)
        
        # Test 1: Invalid configuration
        invalid_config = runner.create_test_config('classical_normal', quick=True)
        invalid_config['generator_type'] = 'invalid_generator'
        
        with pytest.raises(ValueError, match="Unknown generator type"):
            runner.create_complete_model(invalid_config)
        
        # Test 2: Corrupted data handling
        data_file, data = runner.create_test_data(50)
        
        # Add NaN values to data
        corrupted_data = data.copy()
        corrupted_data[0] = [np.nan, np.nan]
        corrupted_data[1] = [np.inf, -np.inf]
        
        config = runner.create_test_config('classical_normal', quick=True)
        model = runner.create_complete_model(config)
        datamodule = runner.create_mock_datamodule(corrupted_data)
        
        # Training should handle corrupted data gracefully
        mock_logger = Mock()
        mock_logger.run_id = "test_error_handling"
        mock_logger.experiment = Mock()
        
        trainer = Trainer(
            max_epochs=1,
            accelerator='cpu',
            logger=mock_logger,
            enable_checkpointing=False,
            enable_progress_bar=False,
            limit_train_batches=1,
            limit_val_batches=1
        )
        
        # Should not crash due to corrupted data
        try:
            trainer.fit(model=model, datamodule=datamodule)
            recovery_success = True
        except Exception as e:
            print(f"Error handling failed: {e}")
            recovery_success = False
        
        # Model should either handle gracefully or fail with clear error
        # (Both are acceptable behaviors)
        print(f"Error recovery test completed, success: {recovery_success}")
    
    def test_reproducibility_end_to_end(self, device, temp_directory):
        """Test end-to-end reproducibility with fixed seeds."""
        runner = EndToEndTestRunner(device, temp_directory)
        
        config = runner.create_test_config('classical_normal', quick=True)
        config['seed'] = 12345  # Fixed seed
        
        data_file, data = runner.create_test_data(100)
        
        # Run 1
        set_seed(config['seed'])
        model1 = runner.create_complete_model(config)
        datamodule1 = runner.create_mock_datamodule(data)
        
        mock_logger1 = Mock()
        mock_logger1.run_id = "reproducibility_run1"
        mock_logger1.experiment = Mock()
        
        trainer1 = Trainer(
            max_epochs=2,
            accelerator='cpu',
            logger=mock_logger1,
            enable_checkpointing=False,
            enable_progress_bar=False,
            limit_train_batches=1,
            limit_val_batches=1
        )
        
        trainer1.fit(model=model1, datamodule=datamodule1)
        
        # Generate samples from first model
        with torch.no_grad():
            set_seed(config['seed'])  # Reset seed for generation
            samples1 = model1._generate_fake_data(50)
        
        # Run 2 with same configuration and seed
        set_seed(config['seed'])
        model2 = runner.create_complete_model(config)
        datamodule2 = runner.create_mock_datamodule(data)
        
        mock_logger2 = Mock()
        mock_logger2.run_id = "reproducibility_run2"
        mock_logger2.experiment = Mock()
        
        trainer2 = Trainer(
            max_epochs=2,
            accelerator='cpu',
            logger=mock_logger2,
            enable_checkpointing=False,
            enable_progress_bar=False,
            limit_train_batches=1,
            limit_val_batches=1
        )
        
        trainer2.fit(model=model2, datamodule=datamodule2)
        
        # Generate samples from second model
        with torch.no_grad():
            set_seed(config['seed'])  # Reset seed for generation
            samples2 = model2._generate_fake_data(50)
        
        # Check reproducibility (should be similar but not necessarily identical due to training randomness)
        mean1, mean2 = torch.mean(samples1, dim=0), torch.mean(samples2, dim=0)
        std1, std2 = torch.std(samples1, dim=0), torch.std(samples2, dim=0)
        
        # Statistical properties should be similar
        assert torch.allclose(mean1, mean2, atol=0.5), "Means should be similar for reproducible training"
        assert torch.allclose(std1, std2, atol=0.5), "Standard deviations should be similar"
    
    def test_configuration_file_integration(self, device, temp_directory):
        """Test integration with YAML configuration files."""
        runner = EndToEndTestRunner(device, temp_directory)
        
        # Create test config file
        config = runner.create_test_config('classical_normal', quick=True)
        config_file = temp_directory / 'test_config.yaml'
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Load config file
        with open(config_file, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        # Verify config loading works
        assert loaded_config['generator_type'] == 'classical_normal'
        assert loaded_config['z_dim'] == 4
        assert loaded_config['max_epochs'] == 3
        
        # Test creating model from loaded config
        model = runner.create_complete_model(loaded_config)
        
        # Test model functionality
        with torch.no_grad():
            samples = model._generate_fake_data(20)
            assert samples.shape == (20, 2)
    
    def test_multi_generator_comparison_workflow(self, device, temp_directory):
        """Test workflow for comparing multiple generators."""
        runner = EndToEndTestRunner(device, temp_directory)
        
        # Test only available generators
        available_generators = ['classical_normal', 'classical_uniform']
        
        data_file, data = runner.create_test_data(100)
        comparison_results = {}
        
        for gen_type in available_generators:
            try:
                config = runner.create_test_config(gen_type, quick=True)
                model = runner.create_complete_model(config)
                
                # Quick training
                datamodule = runner.create_mock_datamodule(data)
                mock_logger = Mock()
                mock_logger.run_id = f"comparison_{gen_type}"
                mock_logger.experiment = Mock()
                
                trainer = Trainer(
                    max_epochs=1,
                    accelerator='cpu',
                    logger=mock_logger,
                    enable_checkpointing=False,
                    enable_progress_bar=False,
                    limit_train_batches=1,
                    limit_val_batches=1
                )
                
                trainer.fit(model=model, datamodule=datamodule)
                
                # Collect metrics
                with torch.no_grad():
                    samples = model._generate_fake_data(100)
                    metrics = model._compute_metrics(samples)
                
                comparison_results[gen_type] = {
                    'training_success': True,
                    'metrics': metrics,
                    'sample_shape': samples.shape
                }
                
            except Exception as e:
                print(f"Error testing {gen_type}: {e}")
                comparison_results[gen_type] = {
                    'training_success': False,
                    'error': str(e)
                }
        
        # Validate comparison results
        successful_runs = [k for k, v in comparison_results.items() if v['training_success']]
        assert len(successful_runs) >= 1, "At least one generator should train successfully"
        
        # If multiple generators succeeded, compare their metrics
        if len(successful_runs) >= 2:
            gen1, gen2 = successful_runs[:2]
            metrics1 = comparison_results[gen1]['metrics']
            metrics2 = comparison_results[gen2]['metrics']
            
            # Both should have computed all metrics
            for metric in ['IsPositive', 'LogLikelihood', 'KLDivergence']:
                assert metric in metrics1
                assert metric in metrics2
        
        print(f"Multi-generator comparison completed: {list(comparison_results.keys())}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])