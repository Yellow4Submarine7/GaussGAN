import gzip
import math
import pickle
import warnings
from functools import partial

import numpy as np
import torch

from torchmetrics import Metric


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




ALL_METRICS = {
    "IsPositive": IsPositive,

}

