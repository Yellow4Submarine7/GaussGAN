"""
Integration tests for new statistical metrics with the existing GaussGAN training pipeline.
Tests metric integration, training loop compatibility, and end-to-end functionality.
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import warnings

from source.model import GaussGan
from source.metrics import (
    MMDivergence, MMDivergenceFromGMM, MMDDistance, 
    WassersteinDistance, ConvergenceTracker, ALL_METRICS
)
from source.nn import ClassicalNoise, MLPGenerator, MLPDiscriminator
import lightning as L


class TestNewMetricsIntegration:
    """Test integration of new metrics with GaussGAN training pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def integration_config(self):
        """Configuration optimized for integration testing."""
        return {
            'z_dim': 2,
            'max_epochs': 3,  # Very short for integration testing
            'batch_size': 16,
            'learning_rate': 0.01,
            'grad_penalty': 0.2,
            'n_critic': 1,  # Reduced for faster testing
            'n_predictor': 1,
            'rl_weight': 10,
            'killer': False,
            'validation_samples': 50,  # Reduced for faster testing
            'nn_gen': [32, 32],
            'nn_disc': [32, 32],
            'nn_validator': [16, 16],
            'non_linearity': 'LeakyReLU',
            'std_scale': 1.1,
            'min_std': 0.5,
            'accelerator': 'cpu',
        }
    
    @pytest.fixture
    def integration_gaussian_params(self):
        """Simple Gaussian parameters for integration testing."""
        return {
            'centroids': [[0, 0], [2, 2]],
            'cov_matrices': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]],
            'weights': [0.5, 0.5]
        }
    
    @pytest.fixture
    def basic_generator(self, integration_config, device):
        """Create basic generator for integration testing."""
        noise_gen = ClassicalNoise(
            z_dim=integration_config['z_dim'],
            generator_type='classical_normal'
        )
        mlp_gen = MLPGenerator(
            non_linearity=integration_config['non_linearity'],
            hidden_dims=integration_config['nn_gen'],
            z_dim=integration_config['z_dim'],
            std_scale=integration_config['std_scale'],
            min_std=integration_config['min_std']
        )
        generator = torch.nn.Sequential(noise_gen, mlp_gen)
        return generator.to(device)
    
    @pytest.fixture
    def basic_discriminator(self, integration_config, device):
        """Create basic discriminator for integration testing."""
        disc = MLPDiscriminator(
            non_linearity=integration_config['non_linearity'],
            hidden_dims=integration_config['nn_disc']
        )
        return disc.to(device)
    
    @pytest.fixture
    def basic_predictor(self, integration_config, device):
        """Create basic predictor for integration testing."""
        pred = MLPDiscriminator(
            non_linearity=integration_config['non_linearity'],
            hidden_dims=integration_config['nn_validator']
        )
        return pred.to(device)
    
    def test_mmd_metrics_in_gaussgan_model(self, basic_generator, basic_discriminator, 
                                         basic_predictor, integration_config, 
                                         integration_gaussian_params, device):
        """Test MMD metrics integration in GaussGan model."""
        from functools import partial
        
        # Test with MMD metrics
        mmd_metrics = ['MMDivergence', 'MMDivergenceFromGMM', 'MMDDistance']
        
        model = GaussGan(
            generator=basic_generator,
            discriminator=basic_discriminator,
            predictor=basic_predictor,
            optimizer=partial(torch.optim.Adam, lr=integration_config['learning_rate']),
            killer=integration_config['killer'],
            n_critic=integration_config['n_critic'],
            grad_penalty=integration_config['grad_penalty'],
            rl_weight=integration_config['rl_weight'],
            n_predictor=integration_config['n_predictor'],
            metrics=mmd_metrics,
            gaussians=integration_gaussian_params,
            validation_samples=integration_config['validation_samples'],
            non_linearity=integration_config['non_linearity']
        )
        model = model.to(device)
        
        # Test metric initialization
        assert len(model.metrics) == len(mmd_metrics)
        
        # Test validation step
        model.eval()
        with torch.no_grad():
            validation_metrics = model.validation_step(batch=None, batch_idx=0)
        
        # Should compute all MMD metrics
        for metric_name in mmd_metrics:
            assert f"val_{metric_name}" in validation_metrics
            value = validation_metrics[f"val_{metric_name}"]
            assert isinstance(value, (float, torch.Tensor))
            if isinstance(value, torch.Tensor):
                assert value.numel() == 1
    
    def test_wasserstein_distance_in_gaussgan_model(self, basic_generator, basic_discriminator,
                                                   basic_predictor, integration_config,
                                                   integration_gaussian_params, device):
        """Test WassersteinDistance metric integration in GaussGan model."""
        from functools import partial
        
        model = GaussGan(
            generator=basic_generator,
            discriminator=basic_discriminator,
            predictor=basic_predictor,
            optimizer=partial(torch.optim.Adam, lr=integration_config['learning_rate']),
            killer=integration_config['killer'],
            n_critic=integration_config['n_critic'],
            grad_penalty=integration_config['grad_penalty'],
            rl_weight=integration_config['rl_weight'],
            n_predictor=integration_config['n_predictor'],
            metrics=['WassersteinDistance'],
            gaussians=integration_gaussian_params,
            validation_samples=integration_config['validation_samples'],
            non_linearity=integration_config['non_linearity']
        )
        model = model.to(device)
        
        # Test validation step
        model.eval()
        with torch.no_grad():
            validation_metrics = model.validation_step(batch=None, batch_idx=0)
        
        assert "val_WassersteinDistance" in validation_metrics
        value = validation_metrics["val_WassersteinDistance"]
        assert isinstance(value, (float, torch.Tensor))
        assert value >= 0  # Wasserstein distance should be non-negative
    
    def test_combined_old_and_new_metrics(self, basic_generator, basic_discriminator,
                                        basic_predictor, integration_config,
                                        integration_gaussian_params, device):
        """Test integration of old and new metrics together."""
        from functools import partial
        
        # Mix of old and new metrics
        combined_metrics = [
            'LogLikelihood',      # Old metric
            'KLDivergence',       # Old metric
            'IsPositive',         # Old metric
            'MMDivergence',       # New metric
            'WassersteinDistance' # New metric
        ]
        
        model = GaussGan(
            generator=basic_generator,
            discriminator=basic_discriminator,
            predictor=basic_predictor,
            optimizer=partial(torch.optim.Adam, lr=integration_config['learning_rate']),
            killer=integration_config['killer'],
            n_critic=integration_config['n_critic'],
            grad_penalty=integration_config['grad_penalty'],
            rl_weight=integration_config['rl_weight'],
            n_predictor=integration_config['n_predictor'],
            metrics=combined_metrics,
            gaussians=integration_gaussian_params,
            validation_samples=integration_config['validation_samples'],
            non_linearity=integration_config['non_linearity']
        )
        model = model.to(device)
        
        # Test validation step
        model.eval()
        with torch.no_grad():
            validation_metrics = model.validation_step(batch=None, batch_idx=0)
        
        # All metrics should be computed
        for metric_name in combined_metrics:
            assert f"val_{metric_name}" in validation_metrics
            value = validation_metrics[f"val_{metric_name}"]
            assert isinstance(value, (float, torch.Tensor))
    
    def test_metric_computation_with_small_samples(self, basic_generator, basic_discriminator,
                                                  basic_predictor, integration_config,
                                                  integration_gaussian_params, device):
        """Test metric computation with very small sample sizes."""
        from functools import partial
        
        # Use very small validation samples
        small_config = integration_config.copy()
        small_config['validation_samples'] = 5  # Very small
        
        model = GaussGan(
            generator=basic_generator,
            discriminator=basic_discriminator,
            predictor=basic_predictor,
            optimizer=partial(torch.optim.Adam, lr=small_config['learning_rate']),
            killer=small_config['killer'],
            n_critic=small_config['n_critic'],
            grad_penalty=small_config['grad_penalty'],
            rl_weight=small_config['rl_weight'],
            n_predictor=small_config['n_predictor'],
            metrics=['MMDivergence', 'WassersteinDistance'],
            gaussians=integration_gaussian_params,
            validation_samples=small_config['validation_samples'],
            non_linearity=small_config['non_linearity']
        )
        model = model.to(device)
        
        # Should handle small samples gracefully
        model.eval()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validation_metrics = model.validation_step(batch=None, batch_idx=0)
        
        # Metrics should be computed (may have warnings)
        assert "val_MMDivergence" in validation_metrics
        assert "val_WassersteinDistance" in validation_metrics
    
    def test_metric_error_handling_in_training(self, basic_generator, basic_discriminator,
                                             basic_predictor, integration_config,
                                             integration_gaussian_params, device):
        """Test metric error handling during training."""
        from functools import partial
        
        model = GaussGan(
            generator=basic_generator,
            discriminator=basic_discriminator,
            predictor=basic_predictor,
            optimizer=partial(torch.optim.Adam, lr=integration_config['learning_rate']),
            killer=integration_config['killer'],
            n_critic=integration_config['n_critic'],
            grad_penalty=integration_config['grad_penalty'],
            rl_weight=integration_config['rl_weight'],
            n_predictor=integration_config['n_predictor'],
            metrics=['MMDivergence'],
            gaussians=integration_gaussian_params,
            validation_samples=integration_config['validation_samples'],
            non_linearity=integration_config['non_linearity']
        )
        model = model.to(device)
        
        # Mock metric to raise an exception
        with patch.object(model.metrics[0], 'compute_score', side_effect=Exception("Test error")):
            model.eval()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                validation_metrics = model.validation_step(batch=None, batch_idx=0)
            
            # Should handle error gracefully and continue
            assert "val_MMDivergence" in validation_metrics
            # Value should be NaN or inf
            value = validation_metrics["val_MMDivergence"]
            assert np.isnan(value) or np.isinf(value)
    
    def test_lightning_trainer_integration(self, basic_generator, basic_discriminator,
                                         basic_predictor, integration_config,
                                         integration_gaussian_params, device, temp_dir):
        """Test integration with Lightning Trainer."""
        from functools import partial
        from torch.utils.data import DataLoader, TensorDataset
        
        model = GaussGan(
            generator=basic_generator,
            discriminator=basic_discriminator,
            predictor=basic_predictor,
            optimizer=partial(torch.optim.Adam, lr=integration_config['learning_rate']),
            killer=integration_config['killer'],
            n_critic=integration_config['n_critic'],
            grad_penalty=integration_config['grad_penalty'],
            rl_weight=integration_config['rl_weight'],
            n_predictor=integration_config['n_predictor'],
            metrics=['MMDivergence', 'WassersteinDistance'],
            gaussians=integration_gaussian_params,
            validation_samples=integration_config['validation_samples'],
            non_linearity=integration_config['non_linearity']
        )
        
        # Create minimal dataloader
        dummy_data = torch.randn(32, 2)
        dummy_labels = torch.zeros(32)
        dataset = TensorDataset(dummy_data, dummy_labels)
        dataloader = DataLoader(dataset, batch_size=integration_config['batch_size'])
        
        # Create trainer
        trainer = L.Trainer(
            max_epochs=1,
            accelerator='cpu',
            devices=1,
            default_root_dir=str(temp_dir),
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False
        )
        
        # Test training
        try:
            trainer.fit(model, dataloader, dataloader)
            training_successful = True
        except Exception as e:
            print(f"Training failed with error: {e}")
            training_successful = False
        
        # Should complete without major errors
        assert training_successful
    
    def test_convergence_tracker_integration(self, basic_generator, basic_discriminator,
                                           basic_predictor, integration_config,
                                           integration_gaussian_params, device):
        """Test ConvergenceTracker integration with training loop."""
        from functools import partial
        
        # Create model with convergence tracker
        model = GaussGan(
            generator=basic_generator,
            discriminator=basic_discriminator,
            predictor=basic_predictor,
            optimizer=partial(torch.optim.Adam, lr=integration_config['learning_rate']),
            killer=integration_config['killer'],
            n_critic=integration_config['n_critic'],
            grad_penalty=integration_config['grad_penalty'],
            rl_weight=integration_config['rl_weight'],
            n_predictor=integration_config['n_predictor'],
            metrics=['KLDivergence'],
            gaussians=integration_gaussian_params,
            validation_samples=integration_config['validation_samples'],
            non_linearity=integration_config['non_linearity']
        )
        model = model.to(device)
        
        # Create convergence tracker
        convergence_tracker = ConvergenceTracker(
            patience=2,
            min_delta=0.1,
            monitor_metric="KLDivergence"
        )
        
        # Simulate training epochs
        for epoch in range(5):
            model.train()
            # Simulate training step
            batch = torch.randn(integration_config['batch_size'], 2).to(device)
            loss = model.training_step((batch, torch.zeros(len(batch))), batch_idx=0)
            
            model.eval()
            # Validation step
            with torch.no_grad():
                val_metrics = model.validation_step(batch=None, batch_idx=0)
            
            # Update convergence tracker
            metrics_dict = {
                key.replace("val_", ""): value 
                for key, value in val_metrics.items() 
                if key.startswith("val_")
            }
            
            convergence_info = convergence_tracker.update(
                epoch=epoch,
                metrics=metrics_dict,
                d_loss=loss.item() if hasattr(loss, 'item') else float(loss),
                g_loss=loss.item() if hasattr(loss, 'item') else float(loss)
            )
            
            # Check convergence info structure
            assert "converged" in convergence_info
            assert "epochs_without_improvement" in convergence_info
            assert "best_metric_value" in convergence_info
            
            # If converged, break
            if convergence_tracker.should_stop_early():
                print(f"Early stopping at epoch {epoch}")
                break
    
    def test_metric_memory_efficiency(self, basic_generator, basic_discriminator,
                                    basic_predictor, integration_config,
                                    integration_gaussian_params, device):
        """Test memory efficiency of new metrics during training."""
        from functools import partial
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        model = GaussGan(
            generator=basic_generator,
            discriminator=basic_discriminator,
            predictor=basic_predictor,
            optimizer=partial(torch.optim.Adam, lr=integration_config['learning_rate']),
            killer=integration_config['killer'],
            n_critic=integration_config['n_critic'],
            grad_penalty=integration_config['grad_penalty'],
            rl_weight=integration_config['rl_weight'],
            n_predictor=integration_config['n_predictor'],
            metrics=['MMDivergence', 'MMDivergenceFromGMM', 'WassersteinDistance'],
            gaussians=integration_gaussian_params,
            validation_samples=integration_config['validation_samples'],
            non_linearity=integration_config['non_linearity']
        )
        model = model.to(device)
        
        # Run multiple validation steps
        model.eval()
        for _ in range(10):
            with torch.no_grad():
                validation_metrics = model.validation_step(batch=None, batch_idx=0)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100, f"Memory increased by {memory_increase:.1f}MB"
    
    def test_all_metrics_registry_integration(self, integration_gaussian_params):
        """Test that all new metrics are properly registered in ALL_METRICS."""
        expected_new_metrics = [
            'MMDivergence', 'MMDivergenceFromGMM', 'MMDDistance', 'WassersteinDistance'
        ]
        
        for metric_name in expected_new_metrics:
            assert metric_name in ALL_METRICS
            metric_class = ALL_METRICS[metric_name]
            
            # Test instantiation
            if metric_name in ['MMDivergenceFromGMM']:
                # These need GMM parameters
                metric = metric_class(**integration_gaussian_params)
            elif metric_name in ['MMDivergence', 'MMDDistance', 'WassersteinDistance']:
                # These need target samples
                target_samples = np.random.randn(50, 2)
                metric = metric_class(target_samples)
            else:
                # Generic instantiation
                metric = metric_class()
            
            assert hasattr(metric, 'compute_score')
    
    def test_metric_configuration_flexibility(self, basic_generator, basic_discriminator,
                                            basic_predictor, integration_config,
                                            integration_gaussian_params, device):
        """Test flexibility in metric configuration."""
        from functools import partial
        
        # Test different metric combinations
        metric_combinations = [
            ['MMDivergence'],
            ['WassersteinDistance'],
            ['MMDivergence', 'WassersteinDistance'],
            ['LogLikelihood', 'MMDivergence'],
            ['KLDivergence', 'WassersteinDistance', 'IsPositive'],
        ]
        
        for metrics in metric_combinations:
            model = GaussGan(
                generator=basic_generator,
                discriminator=basic_discriminator,
                predictor=basic_predictor,
                optimizer=partial(torch.optim.Adam, lr=integration_config['learning_rate']),
                killer=integration_config['killer'],
                n_critic=integration_config['n_critic'],
                grad_penalty=integration_config['grad_penalty'],
                rl_weight=integration_config['rl_weight'],
                n_predictor=integration_config['n_predictor'],
                metrics=metrics,
                gaussians=integration_gaussian_params,
                validation_samples=integration_config['validation_samples'],
                non_linearity=integration_config['non_linearity']
            )
            model = model.to(device)
            
            # Test validation step
            model.eval()
            with torch.no_grad():
                validation_metrics = model.validation_step(batch=None, batch_idx=0)
            
            # All requested metrics should be computed
            for metric_name in metrics:
                assert f"val_{metric_name}" in validation_metrics
    
    def test_metric_nan_inf_handling_in_training_loop(self, basic_generator, basic_discriminator,
                                                     basic_predictor, integration_config,
                                                     integration_gaussian_params, device):
        """Test handling of NaN/Inf values from metrics in training loop."""
        from functools import partial
        
        model = GaussGan(
            generator=basic_generator,
            discriminator=basic_discriminator,
            predictor=basic_predictor,
            optimizer=partial(torch.optim.Adam, lr=integration_config['learning_rate']),
            killer=integration_config['killer'],
            n_critic=integration_config['n_critic'],
            grad_penalty=integration_config['grad_penalty'],
            rl_weight=integration_config['rl_weight'],
            n_predictor=integration_config['n_predictor'],
            metrics=['MMDivergence'],
            gaussians=integration_gaussian_params,
            validation_samples=integration_config['validation_samples'],
            non_linearity=integration_config['non_linearity']
        )
        model = model.to(device)
        
        # Mock metric to return NaN/Inf
        original_compute_score = model.metrics[0].compute_score
        def mock_compute_score(*args, **kwargs):
            return np.array([np.nan, np.inf, 0.5])  # Mix of NaN, Inf, and valid
        
        model.metrics[0].compute_score = mock_compute_score
        
        # Should handle gracefully
        model.eval()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            validation_metrics = model.validation_step(batch=None, batch_idx=0)
        
        # Should still produce a result
        assert "val_MMDivergence" in validation_metrics
        
        # Restore original method
        model.metrics[0].compute_score = original_compute_score
    
    @pytest.mark.slow
    def test_extended_training_with_new_metrics(self, basic_generator, basic_discriminator,
                                              basic_predictor, integration_config,
                                              integration_gaussian_params, device):
        """Test extended training with new metrics for stability."""
        from functools import partial
        from torch.utils.data import DataLoader, TensorDataset
        
        # Extended config
        extended_config = integration_config.copy()
        extended_config['max_epochs'] = 10
        
        model = GaussGan(
            generator=basic_generator,
            discriminator=basic_discriminator,
            predictor=basic_predictor,
            optimizer=partial(torch.optim.Adam, lr=extended_config['learning_rate']),
            killer=extended_config['killer'],
            n_critic=extended_config['n_critic'],
            grad_penalty=extended_config['grad_penalty'],
            rl_weight=extended_config['rl_weight'],
            n_predictor=extended_config['n_predictor'],
            metrics=['MMDivergence', 'WassersteinDistance', 'KLDivergence'],
            gaussians=integration_gaussian_params,
            validation_samples=extended_config['validation_samples'],
            non_linearity=extended_config['non_linearity']
        )
        
        # Create dataloader
        dummy_data = torch.randn(128, 2)
        dummy_labels = torch.zeros(128)
        dataset = TensorDataset(dummy_data, dummy_labels)
        dataloader = DataLoader(dataset, batch_size=extended_config['batch_size'])
        
        # Track metrics over time
        metric_history = []
        
        # Simulate training epochs
        for epoch in range(extended_config['max_epochs']):
            model.train()
            
            # Training step
            for batch_idx, batch in enumerate(dataloader):
                loss = model.training_step(batch, batch_idx)
                if batch_idx >= 2:  # Limit batches for testing
                    break
            
            # Validation step
            model.eval()
            with torch.no_grad():
                val_metrics = model.validation_step(batch=None, batch_idx=0)
            
            metric_history.append(val_metrics)
        
        # Verify metrics were computed throughout training
        assert len(metric_history) == extended_config['max_epochs']
        
        for epoch_metrics in metric_history:
            assert "val_MMDivergence" in epoch_metrics
            assert "val_WassersteinDistance" in epoch_metrics
            assert "val_KLDivergence" in epoch_metrics
            
            # Values should be finite or NaN (but not crash)
            for value in epoch_metrics.values():
                assert np.isfinite(value) or np.isnan(value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])