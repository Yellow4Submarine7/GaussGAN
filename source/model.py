import tempfile
from pathlib import Path
import time
import os

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
import pdb
import warnings


from .metrics import ALL_METRICS, ConvergenceTracker


class GaussGan(LightningModule):

    def __init__(self, generator, discriminator, predictor, optimizer, **kwargs):
        super().__init__()
        self.automatic_optimization = False  # Disable automatic optimization
        self.rl_weight = kwargs.get("rl_weight", 1.0)
        self.save_hyperparameters(
            ignore=[
                "generator",
                "discriminator",
                "predictor",
                "optimizer",
            ],
        )
        self.generator = generator
        self.discriminator = discriminator
        self.predictor = predictor
        self.optimizer = optimizer
        self.metrics = kwargs.get("metrics", [])
        self.killer = kwargs.get("killer", False)
        self.validation_samples = kwargs.get("validation_samples", 1000)
        self.n_critic = kwargs.get("n_critic", 5)
        self.n_predictor = kwargs.get("n_predictor", 2)
        self.grad_penalty = kwargs.get("grad_penalty", 10.0)
        self.gaussians = kwargs.get("gaussians", {})
        self.non_linearity = kwargs.get("non_linearity", False)  # :(
        
        # Initialize convergence tracker
        self.convergence_tracker = ConvergenceTracker(
            patience=kwargs.get("convergence_patience", 10),
            min_delta=kwargs.get("convergence_min_delta", 1e-4),
            monitor_metric=kwargs.get("convergence_monitor", "KLDivergence"),
            window_size=kwargs.get("convergence_window", 5)
        )
        
        # Store reference data for new metrics initialization
        self.target_data = kwargs.get("target_data", None)
        
        # Store generator type and experiment name for visualization
        self.generator_type = kwargs.get("generator_type", "unknown")
        self.experiment_name = kwargs.get("experiment_name", "default_experiment")
        
        # Initialize epoch history for cumulative visualization
        self.epoch_history = []

    def configure_optimizers(self):
        g_optim = self.optimizer(self.generator.parameters(), 
                            betas=(0.5, 0.9))
        d_optim = self.optimizer(self.discriminator.parameters(), 
                            betas=(0.5, 0.9))
        p_optim = self.optimizer(self.predictor.parameters()) #default betas=(0.9, 0.999)
        return [g_optim, d_optim, p_optim], []

    def _calculate_gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=weight,
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def training_step(self, batch, batch_idx):
        g_optim, d_optim, p_optim = self.optimizers()
        d_loss_total = 0.0
        for _ in range(self.n_critic):
            d_optim.zero_grad()
            d_loss = self._compute_discriminator_loss(batch)
            self.manual_backward(d_loss)
            d_optim.step()
            d_loss_total += d_loss.item()
        d_loss_avg = d_loss_total / self.n_critic
        if self.killer:
            for _ in range(self.n_predictor):
                p_optim.zero_grad()
                p_loss, _ = self._compute_predictor_loss(batch)
                self.manual_backward(p_loss)
                p_optim.step()
        g_optim.zero_grad()
        g_loss = self._compute_generator_loss(batch)
        self.manual_backward(g_loss)
        g_optim.step()
        
        # Store losses for convergence tracking
        self._last_d_loss = d_loss_avg
        self._last_g_loss = g_loss.item()
        
        # Log losses for convergence tracking
        self.log("train_d_loss", d_loss_avg, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_g_loss", g_loss.item(), on_step=True, on_epoch=True, prog_bar=False)
        
        return {"d_loss": d_loss_avg, "g_loss": g_loss.item()}

    def validation_step(self, batch, batch_idx):
        fake_data = self._generate_fake_data(self.validation_samples).detach()

        # Compute and log metrics on generated data
        metrics_fake = self._compute_metrics(fake_data)

        # Add safety check for None values in metrics
        avg_metrics_fake = {}
        processed_metrics = {}
        for k, v in metrics_fake.items():
            if v is None:
                # Handle None case - log a warning and skip or set to a default value
                warnings.warn(f"Metric {k} returned None in validation_step")
                avg_metrics_fake[f"ValidationStep_FakeData_{k}"] = float("nan")
                processed_metrics[k] = float("nan")
            elif not hasattr(v, "__iter__"):
                # Handle non-iterable case (like a single number)
                avg_metrics_fake[f"ValidationStep_FakeData_{k}"] = v
                processed_metrics[k] = v
            else:
                # Original calculation for iterable results
                valid_vals = [val for val in v if val is not None]
                if valid_vals:
                    metric_value = np.mean(valid_vals)
                    avg_metrics_fake[f"ValidationStep_FakeData_{k}"] = metric_value
                    processed_metrics[k] = metric_value
                else:
                    avg_metrics_fake[f"ValidationStep_FakeData_{k}"] = float("nan")
                    processed_metrics[k] = float("nan")

        # Update convergence tracker with current metrics
        # Get recent losses from trainer if available
        d_loss = getattr(self, '_last_d_loss', None)
        g_loss = getattr(self, '_last_g_loss', None)
        
        convergence_info = self.convergence_tracker.update(
            epoch=self.current_epoch,
            metrics=processed_metrics,
            d_loss=d_loss,
            g_loss=g_loss
        )
        
        # Log convergence information
        convergence_log = {}
        for key, value in convergence_info.items():
            if value is not None:
                convergence_log[f"convergence_{key}"] = value
        
        # Combine all metrics for logging
        all_log_metrics = {**avg_metrics_fake, **convergence_log}
        
        self.log_dict(
            all_log_metrics,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
            batch_size=batch[0].size(0),
            sync_dist=True,
        )

        # Generate epoch-by-epoch visualization
        self._generate_epoch_visualization(fake_data)
        
        # Rest of the method remains unchanged
        csv_string = "x,y\n" + "\n".join([f"{row[0]},{row[1]}" for row in fake_data])
        
        # 方案1: 尝试MLflow记录
        mlflow_success = False
        try:
            self.logger.experiment.log_text(
                text=csv_string,
                artifact_file=f"gaussian_generated_epoch_{self.current_epoch:04d}.csv",
                run_id=self.logger.run_id,
            )
            mlflow_success = True
            print(f"Validation step completed for epoch {self.current_epoch} - CSV logged to MLflow")
        except Exception as e:
            print(f"MLflow CSV logging failed: {e}")
        
        # 方案2: 如果MLflow失败，保存到本地文件
        if not mlflow_success:
            try:
                import os
                os.makedirs("generated_samples", exist_ok=True)
                csv_file = f"generated_samples/gaussian_generated_epoch_{self.current_epoch:04d}_run_{self.logger.run_id}.csv"
                with open(csv_file, 'w') as f:
                    f.write(csv_string)
                print(f"Validation step completed for epoch {self.current_epoch} - CSV saved locally to {csv_file}")
            except Exception as e:
                print(f"Local CSV save also failed: {e}")
                print(f"Validation step completed for epoch {self.current_epoch} - No CSV saved")

        return {
            "fake_data": fake_data, 
            "metrics": avg_metrics_fake,
            "convergence_info": convergence_info
        }

    def on_validation_epoch_end(self):
        """Check for early stopping based on convergence."""
        if hasattr(self, 'convergence_tracker') and self.convergence_tracker.should_stop_early():
            print(f"Training converged at epoch {self.convergence_tracker.convergence_epoch}")
            print(f"Best {self.convergence_tracker.monitor_metric}: {self.convergence_tracker.best_metric_value}")
            
            # Log convergence event
            self.log("converged", 1.0, on_epoch=True)
            self.log("convergence_epoch", float(self.convergence_tracker.convergence_epoch), on_epoch=True)
            
            # Note: Lightning will handle early stopping if a callback is configured
            # This method primarily logs the convergence information

    def _generate_fake_data(self, batch_size):
        # 添加时间测量
        start_time = time.time()
        
        # Convert batch_size from tensor to int if needed
        if isinstance(batch_size, torch.Tensor):
            batch_size = batch_size.item()

        # Only pass batch_size to the generator
        fake_gaussians = self.generator(batch_size)
        
        # 记录时间
        elapsed = time.time() - start_time
        # 每10个批次记录一次时间
        if hasattr(self, 'step_counter'):
            self.step_counter += 1
        else:
            self.step_counter = 0
        
        if self.step_counter % 10 == 0:
            self.log("generator_time", elapsed, on_step=True, on_epoch=True)
            print(f"Generator forward time: {elapsed:.2f}s for batch size {batch_size}")

        # Ensure the output is on the same device as the model
        if fake_gaussians.device != self.device:
            fake_gaussians = fake_gaussians.to(self.device)

        return fake_gaussians

    def _apply_discriminator(self, x):
        return self.discriminator(x)

    def _apply_predictor(self, x):
        return torch.sigmoid(self.predictor(x))

    def _compute_discriminator_loss(self, batch):

        fake_gaussians = self._generate_fake_data(batch[0].size(0))
        fake_gaussians = fake_gaussians.detach()
        d_fake = self._apply_discriminator(fake_gaussians)
        d_real = self._apply_discriminator(batch[0])

        # thanks cedric villani
        d_loss = d_fake.mean() - d_real.mean()

        # Compute gradient penalty
        eps = torch.rand(batch[0].size(0), 1).to(self.device)
        fake_inter = (eps * batch[0] + (1.0 - eps) * fake_gaussians).requires_grad_(
            True
        )
        d_inter = self._apply_discriminator(fake_inter)
        grad_penalty_score = self._calculate_gradient_penalty(d_inter, fake_inter)
        d_loss += self.grad_penalty * grad_penalty_score
        return d_loss

    def _compute_generator_loss(self, batch):
        x_fake = self._generate_fake_data(batch[0].size(0))
        d_fake = self._apply_discriminator(x_fake)
        gan_loss = -d_fake.mean()

        if self.killer:
            rl_weight = getattr(self, "rl_weight", 1.0)
            
            # 分别计算负 x 轴和正 x 轴点的惩罚
            neg_x_mask = x_fake[:, 0] < 0
            
            if torch.any(neg_x_mask):
                # 对负 x 轴的点施加更大惩罚
                neg_x_points = x_fake[neg_x_mask]
                neg_x_penalty = -self._apply_predictor(neg_x_points).mean() * rl_weight * 5.0
                
                # 对正 x 轴的点保持正常奖励
                pos_x_mask = ~neg_x_mask
                if torch.any(pos_x_mask):
                    pos_x_points = x_fake[pos_x_mask]
                    pos_x_reward = -self._apply_predictor(pos_x_points).mean() * rl_weight
                else:
                    pos_x_reward = 0.0
                    
                rl_loss = neg_x_penalty + pos_x_reward
            else:
                # 所有点都在正 x 轴，使用正常奖励
                rl_loss = -self._apply_predictor(x_fake).mean() * rl_weight
        else:
            rl_loss = 0

        g_loss = gan_loss + rl_loss
        return g_loss

    def _compute_predictor_loss(self, batch):
        x, _ = batch  # 忽略原始标签
        v = self._apply_predictor(x)
        
        # 创建新标签：x > 0 的点为 1，x < 0 的点为 0
        targets = (x[:, 0] > 0).float().unsqueeze(1)
        
        # 使用 BCE 损失（因为输出经过了 sigmoid）
        #p_loss = nn.HuberLoss()(v, r)
        p_loss = F.binary_cross_entropy(v, targets)
        return p_loss, {"Predictor_loss": p_loss}

    def _compute_metrics(self, batch):
        metrics = {}
        for metric in self.metrics:
            if metric == "LogLikelihood" or metric == "KLDivergence":
                metrics[metric] = ALL_METRICS[metric](
                    centroids=self.gaussians["centroids"],
                    cov_matrices=self.gaussians["covariances"],
                    weights=self.gaussians["weights"],
                ).compute_score(batch)
            elif metric == "WassersteinDistance":
                if self.target_data is not None:
                    metrics[metric] = ALL_METRICS[metric](
                        target_samples=self.target_data,
                        aggregation="mean"
                    ).compute_score(batch)
                else:
                    warnings.warn(f"Target data not provided for {metric}, skipping")
                    metrics[metric] = float('nan')
            elif metric == "MMDDistance":
                if self.target_data is not None:
                    metrics[metric] = ALL_METRICS[metric](
                        target_samples=self.target_data,
                        kernel="rbf",
                        gamma=1.0
                    ).compute_score(batch)
                else:
                    warnings.warn(f"Target data not provided for {metric}, skipping")
                    metrics[metric] = float('nan')
            elif metric == "MMDivergence":
                if self.target_data is not None:
                    metrics[metric] = ALL_METRICS[metric](
                        target_samples=self.target_data
                    ).compute_score(batch)
                else:
                    warnings.warn(f"Target data not provided for {metric}, skipping")
                    metrics[metric] = float('nan')
            else:
                metrics[metric] = ALL_METRICS[metric]().compute_score(batch)
        return metrics
    
    def _generate_epoch_visualization(self, generated_data):
        """Generate cumulative subplot grid showing all epochs' generated vs target distributions."""
        try:
            # Store current epoch's data (convert to numpy if tensor)
            if generated_data is not None and len(generated_data) > 0:
                if isinstance(generated_data, torch.Tensor):
                    gen_data = generated_data.detach().cpu().numpy()
                else:
                    gen_data = generated_data
                self.epoch_history.append(gen_data)
            else:
                # Store empty array if no data
                self.epoch_history.append(np.array([]))
            
            # Limit history to prevent memory issues (keep last 100 epochs)
            if len(self.epoch_history) > 100:
                self.epoch_history = self.epoch_history[-100:]
            
            # Create images directory structure
            images_dir = f"images/{self.experiment_name}"
            os.makedirs(images_dir, exist_ok=True)
            
            # Calculate grid size for subplots
            num_epochs = len(self.epoch_history)
            if num_epochs == 1:
                rows, cols = 1, 1
            else:
                cols = max(2, int(np.ceil(np.sqrt(num_epochs))))
                rows = int(np.ceil(num_epochs / cols))
            
            # Create figure with dynamic size
            fig_width = min(20, cols * 4)  # Max 20 inches wide
            fig_height = min(16, rows * 3)  # Max 16 inches tall
            fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
            
            # Ensure axes is always a 2D array
            if rows == 1 and cols == 1:
                axes = np.array([[axes]])
            elif rows == 1:
                axes = axes.reshape(1, -1)
            elif cols == 1:
                axes = axes.reshape(-1, 1)
            
            # Generate target distribution (same for all subplots)
            mean1 = torch.tensor([-5.0, 5.0])
            cov1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
            mean2 = torch.tensor([5.0, 5.0])
            cov2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
            
            # Generate target samples
            dist1 = torch.distributions.MultivariateNormal(mean1, cov1)
            dist2 = torch.distributions.MultivariateNormal(mean2, cov2)
            n_target_samples = 500  # Reduced for performance with many subplots
            
            target1 = dist1.sample((n_target_samples,))
            target2 = dist2.sample((n_target_samples,))
            
            # Plot each epoch in the grid
            for epoch_idx in range(num_epochs):
                row = epoch_idx // cols
                col = epoch_idx % cols
                ax = axes[row, col]
                
                # Plot target distribution
                ax.scatter(target1[:, 0], target1[:, 1], color="blue", alpha=0.4, s=1, label="Target -1" if epoch_idx == 0 else "")
                ax.scatter(target2[:, 0], target2[:, 1], color="red", alpha=0.4, s=1, label="Target +1" if epoch_idx == 0 else "")
                
                # Plot generated data for this epoch
                if len(self.epoch_history[epoch_idx]) > 0:
                    ax.scatter(self.epoch_history[epoch_idx][:, 0], self.epoch_history[epoch_idx][:, 1], 
                             s=1, color='black', alpha=0.6, label="Generated" if epoch_idx == 0 else "")
                
                # Add reference lines
                ax.axhline(y=0, color="gray", linestyle="--", alpha=0.2)
                ax.axvline(x=0, color="gray", linestyle="--", alpha=0.2)
                
                # Set plot properties
                ax.set_xlim(-11, 11)
                ax.set_ylim(-11, 11)
                ax.set_title(f'Epoch {epoch_idx}', fontsize=8)
                ax.grid(True, alpha=0.2)
                
                # Only show axis labels on edge plots to save space
                if row == rows - 1:  # Bottom row
                    ax.set_xlabel('X', fontsize=8)
                if col == 0:  # Left column
                    ax.set_ylabel('Y', fontsize=8)
                
                # Smaller tick labels
                ax.tick_params(labelsize=6)
            
            # Hide empty subplots
            for epoch_idx in range(num_epochs, rows * cols):
                row = epoch_idx // cols
                col = epoch_idx % cols
                axes[row, col].set_visible(False)
            
            # Add overall title and legend
            fig.suptitle(f'{self.generator_type.title()} Generator Training Progress ({num_epochs} epochs)', fontsize=14)
            
            # Add legend only to the first subplot
            if num_epochs > 0:
                axes[0, 0].legend(fontsize=8, loc='upper right')
            
            # Save plot with single filename (overwrites previous)
            filename = f"{images_dir}/training_{self.generator_type}.png"
            plt.tight_layout()
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Epoch {self.current_epoch}: Cumulative visualization updated ({num_epochs} epochs) -> {filename}")
            
        except Exception as e:
            # Don't break training if visualization fails
            print(f"Warning: Failed to generate epoch visualization: {e}")
            plt.close('all')  # Clean up any open figures
