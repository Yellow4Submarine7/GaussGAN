import tempfile
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
import pdb
import warnings


from .metrics import ALL_METRICS


class GaussGan(LightningModule):

    def __init__(self, generator, discriminator, predictor, optimizer, **kwargs):
        super().__init__()
        self.automatic_optimization = False  # Disable automatic optimization
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
        self.grad_penalty = kwargs.get("grad_penalty", 10.0)
        self.gaussians = kwargs.get("gaussians", {})
        self.non_linearity = kwargs.get("non_linearity", False)  # :(

    def configure_optimizers(self):
        # pdb.set_trace()
        g_optim = self.optimizer(self.generator.parameters())
        d_optim = self.optimizer(self.discriminator.parameters())
        p_optim = self.optimizer(self.predictor.parameters())
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
        data_gaussians, _ = batch

        if torch.isnan(data_gaussians).all():
            self.log(
                "TrainingSkipped",
                1,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            return

        # Access the optimizers
        g_optim, d_optim, p_optim = self.optimizers()
        # data, labels = batch

        # train discriminator
        d_optim.zero_grad()
        d_loss = self._compute_discriminator_loss(batch)
        self.manual_backward(d_loss)
        d_optim.step()
        self.log(
            "DiscriminatorLoss",
            d_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch[0].size(0),
            sync_dist=True,
        )

        # train predictor
        if self.killer == True:
            p_loss, p_aux = self._compute_predictor_loss(batch)
            self.manual_backward(p_loss)
            p_optim.step()
            p_optim.zero_grad()
            self.log(
                "PredictorLoss",
                p_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch[0].size(0),
                sync_dist=True,
            )
            for key, value in p_aux.items():
                self.log(
                    key,
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    batch_size=batch[0].size(0),
                    sync_dist=True,
                )

        if (batch_idx % self.n_critic) == 0:
            # train generator
            g_optim.zero_grad()
            g_loss = self._compute_generator_loss(batch)
            self.manual_backward(g_loss)
            g_optim.step()
            self.log(
                "GeneratorLoss",
                g_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch[0].size(0),
                sync_dist=True,
            )

    def validation_step(self, batch, batch_idx):
        fake_data = self._generate_fake_data(self.validation_samples).detach()

        # Compute and log metrics on generated data
        metrics_fake = self._compute_metrics(fake_data)

        # Add safety check for None values in metrics
        avg_metrics_fake = {}
        for k, v in metrics_fake.items():
            if v is None:
                # Handle None case - log a warning and skip or set to a default value
                warnings.warn(f"Metric {k} returned None in validation_step")
                avg_metrics_fake[f"ValidationStep_FakeData_{k}"] = float("nan")
            elif not hasattr(v, "__iter__"):
                # Handle non-iterable case (like a single number)
                avg_metrics_fake[f"ValidationStep_FakeData_{k}"] = v
            else:
                # Original calculation for iterable results
                valid_vals = [val for val in v if val is not None]
                if valid_vals:
                    avg_metrics_fake[f"ValidationStep_FakeData_{k}"] = np.mean(
                        valid_vals
                    )
                else:
                    avg_metrics_fake[f"ValidationStep_FakeData_{k}"] = float("nan")

        self.log_dict(
            avg_metrics_fake,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
            batch_size=batch[0].size(0),
            sync_dist=True,
        )

        # Rest of the method remains unchanged
        csv_string = "x,y\n" + "\n".join([f"{row[0]},{row[1]}" for row in fake_data])
        try:
            self.logger.experiment.log_text(
                text=csv_string,
                artifact_file=f"gaussian_generated_epoch_{self.current_epoch:04d}.csv",
                run_id=self.logger.run_id,
            )
        except AttributeError:
            print("Could not log the CSV file as an artifact.")

        return {"fake_data": fake_data, "metrics": avg_metrics_fake}

    def _generate_fake_data(self, batch_size):
        # Convert batch_size from tensor to int if needed
        if isinstance(batch_size, torch.Tensor):
            batch_size = batch_size.item()

        # Only pass batch_size to the generator
        fake_gaussians = self.generator(batch_size)

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

        if self.killer == True:
            rl_loss = -self._apply_predictor(x_fake).mean()
        else:
            rl_loss = 0

        g_loss = gan_loss + rl_loss
        return g_loss

    def _compute_predictor_loss(self, batch):
        x, r = batch
        v = self._apply_predictor(x)
        p_loss = nn.HuberLoss()(v, r)
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
            else:
                metrics[metric] = ALL_METRICS[metric]().compute_score(batch)
        return metrics
