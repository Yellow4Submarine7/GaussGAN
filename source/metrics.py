import gzip
import math
import pickle
import warnings
from functools import partial

import numpy as np
import torch

from torchmetrics import Metric
from sklearn.mixture import GaussianMixture


class GaussianMetric(Metric):
    """Base class for calculating Gaussian metrics. Implements basic functionality for metric calculation and aggregation."""

    def __init__(self, dist_sync_on_step=False):
        """Initializes state for metric computation."""
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        #
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, points):
        """Updates the metric's state with the provided molecules."""
        scores = self.compute_score(points)
        scores = np.array(scores, dtype=np.float32)
        self.score += torch.tensor(scores).sum()
        self.total += len(points)

    def compute(self):
        """Computes the final metric score."""
        return self.score.float() / self.total.float()

    def compute_score(self, mols):
        """Method to compute the score of the given molecules. Should be implemented by subclasses."""
        raise NotImplementedError("This method needs to be implemented by subclasses")


class IsPositive(GaussianMetric):
    """Calculates the uniqueness of molecules within a batch."""

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

    def compute_score(self, points):
        return [-1 if point[0] < 0 else 1 for point in points]


class LogLikelihood(GaussianMetric):
    """Calculates the uniqueness of molecules within a batch."""

    def __init__(self, centroids, cov_matrices, weights, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # Create GaussianMixture model

        self.gmm = GaussianMixture(n_components=len(centroids), covariance_type="full")

        self.gmm.means_ = np.array(centroids)
        self.gmm.covariances_ = np.array(cov_matrices)
        self.gmm.weights_ = weights
        self.gmm.precisions_cholesky_ = np.linalg.cholesky(
            np.linalg.inv(self.gmm.covariances_)
        )

    def compute_score(self, points):


        # Filter out points with NaN values
        # points = np.array(points)
        points = points.cpu().numpy()
        nan_indices = np.isnan(points).any(axis=1)

        points = points[~nan_indices]
        #import pdb
        #pdb.set_trace()
        #points = points[~np.isnan(points).any(axis=1)]


        return self.gmm.score_samples(points)


ALL_METRICS = {
    "IsPositive": IsPositive,
    "LogLikelihood": LogLikelihood,
}
