import gzip
import math
import pickle
import warnings
from functools import partial
from typing import Union, List, Dict, Optional
import numpy as np
from scipy.stats import gaussian_kde, wasserstein_distance
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
        # import pdb
        # pdb.set_trace()
        # points = points[~np.isnan(points).any(axis=1)]

        try:
            return_value = self.gmm.score_samples(points)
        except Exception as e:
            warnings.warn(f"Error during GMM score computation: {e}")
            return_value = np.zeros(len(points))
        return return_value


class KLDivergence(GaussianMetric):
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
        points = points.cpu().numpy()
        points = points[~np.isnan(points).any(axis=1)]
        samples_nn = np.array(points)

        # 估计生成分布P(x)
        kde = gaussian_kde(samples_nn.T)
        p_estimates = kde(samples_nn.T)

        # 计算目标分布Q(x)
        q_values = np.exp(self.gmm.score_samples(samples_nn))   # 修复：移除错误的负号

        # 过滤无效值
        valid_indices = (p_estimates > 0) & (q_values > 0)
        if not np.any(valid_indices):
            warnings.warn("No valid values for KL divergence calculation")
            return np.array([float("nan")])

        p_valid = p_estimates[valid_indices]
        q_valid = q_values[valid_indices]

        # 计算KL(Q||P)而不是KL(P||Q)
        kl_divergence = np.mean(np.log(q_valid) - np.log(p_valid))

        return kl_divergence


class MMDivergence(GaussianMetric):
    """
    Maximum Mean Discrepancy (MMD) metric for measuring distance between 
    generated samples and target distribution using RBF (Gaussian) kernels.
    
    MMD provides a non-parametric test statistic for determining whether two 
    distributions are different. It uses the mean embeddings in a reproducing 
    kernel Hilbert space (RKHS) to compare distributions.
    """

    def __init__(self, target_samples, bandwidths=None, dist_sync_on_step=False):
        """
        Initialize MMD metric with target distribution samples.
        
        Args:
            target_samples: Reference samples from target distribution, shape (n_samples, n_dims)
            bandwidths: List of bandwidth parameters for RBF kernels. If None, uses 
                       adaptive bandwidths based on median heuristic.
            dist_sync_on_step: Whether to synchronize metric state across devices
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        # Convert target samples to numpy array and validate
        if torch.is_tensor(target_samples):
            self.target_samples = target_samples.detach().cpu().numpy()
        else:
            self.target_samples = np.array(target_samples)
            
        # Filter out NaN values from target samples
        valid_mask = ~np.isnan(self.target_samples).any(axis=1)
        self.target_samples = self.target_samples[valid_mask]
        
        if len(self.target_samples) == 0:
            raise ValueError("No valid target samples provided (all contain NaN)")
            
        # Set up kernel bandwidths
        if bandwidths is None:
            self.bandwidths = self._compute_adaptive_bandwidths()
        else:
            self.bandwidths = np.array(bandwidths)
            
        self.n_target = len(self.target_samples)

    def _compute_adaptive_bandwidths(self):
        """
        Compute adaptive bandwidths using median heuristic.
        
        Returns:
            Array of bandwidth parameters for multiple scales
        """
        # Compute pairwise distances
        try:
            from scipy.spatial.distance import pdist
            distances = pdist(self.target_samples, metric='euclidean')
            
            if len(distances) == 0:
                # Fallback for single sample
                return np.array([1.0])
                
            median_dist = np.median(distances)
            if median_dist == 0:
                median_dist = 1.0
                
            # Use multiple scales around the median
            bandwidths = median_dist * np.array([0.1, 0.5, 1.0, 2.0, 5.0])
            return bandwidths
            
        except Exception as e:
            warnings.warn(f"Failed to compute adaptive bandwidths: {e}. Using default values.")
            return np.array([0.1, 1.0, 10.0])

    def _rbf_kernel(self, X, Y, bandwidth):
        """
        Compute RBF (Gaussian) kernel matrix between two sample sets.
        
        Args:
            X: First sample set, shape (n_samples_X, n_dims)
            Y: Second sample set, shape (n_samples_Y, n_dims) 
            bandwidth: Bandwidth parameter (sigma^2)
            
        Returns:
            Kernel matrix K(X,Y) of shape (n_samples_X, n_samples_Y)
        """
        # Compute squared Euclidean distances
        X_sqnorms = np.sum(X**2, axis=1, keepdims=True)  # (n_X, 1)
        Y_sqnorms = np.sum(Y**2, axis=1, keepdims=True).T  # (1, n_Y)
        XY = np.dot(X, Y.T)  # (n_X, n_Y)
        
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
        sq_distances = X_sqnorms + Y_sqnorms - 2 * XY
        
        # Avoid numerical issues
        sq_distances = np.maximum(sq_distances, 0.0)
        
        # Apply RBF kernel: exp(-||x-y||^2 / (2 * bandwidth))
        return np.exp(-sq_distances / (2 * bandwidth))

    def _compute_mmd_squared(self, generated_samples):
        """
        Compute squared MMD between generated samples and target samples.
        
        Args:
            generated_samples: Generated samples, shape (n_gen, n_dims)
            
        Returns:
            Squared MMD value (scalar)
        """
        n_gen = len(generated_samples)
        
        if n_gen == 0:
            return float('inf')
            
        mmd_values = []
        
        # Compute MMD for each bandwidth and average
        for bandwidth in self.bandwidths:
            # K_XX: kernel between target samples
            K_XX = self._rbf_kernel(self.target_samples, self.target_samples, bandwidth)
            
            # K_YY: kernel between generated samples  
            K_YY = self._rbf_kernel(generated_samples, generated_samples, bandwidth)
            
            # K_XY: kernel between target and generated samples
            K_XY = self._rbf_kernel(self.target_samples, generated_samples, bandwidth)
            
            # MMD^2 = E[K(X,X)] + E[K(Y,Y)] - 2*E[K(X,Y)]
            # Unbiased estimators:
            # E[K(X,X)] ≈ (1/(n(n-1))) * sum_{i≠j} K(x_i, x_j)
            # E[K(Y,Y)] ≈ (1/(m(m-1))) * sum_{i≠j} K(y_i, y_j) 
            # E[K(X,Y)] ≈ (1/(nm)) * sum_i sum_j K(x_i, y_j)
            
            # For K_XX, exclude diagonal (i=j terms)
            if self.n_target > 1:
                K_XX_sum = np.sum(K_XX) - np.trace(K_XX)
                term_XX = K_XX_sum / (self.n_target * (self.n_target - 1))
            else:
                term_XX = 0.0
                
            # For K_YY, exclude diagonal  
            if n_gen > 1:
                K_YY_sum = np.sum(K_YY) - np.trace(K_YY)
                term_YY = K_YY_sum / (n_gen * (n_gen - 1))
            else:
                term_YY = 0.0
                
            # For K_XY, use all terms
            term_XY = np.mean(K_XY)
            
            # Compute MMD^2 for this bandwidth
            mmd_sq = term_XX + term_YY - 2 * term_XY
            mmd_values.append(max(mmd_sq, 0.0))  # Ensure non-negative
            
        # Return average MMD across bandwidths
        return np.mean(mmd_values)

    def compute_score(self, points):
        """
        Compute MMD scores for generated points.
        
        Args:
            points: Generated points, shape (n_points, n_dims)
            
        Returns:
            Array of MMD values (one value per batch, broadcasted to match input size)
        """
        try:
            # Convert to numpy and filter NaN values
            if torch.is_tensor(points):
                points_np = points.detach().cpu().numpy()
            else:
                points_np = np.array(points)
                
            # Filter out NaN values
            valid_mask = ~np.isnan(points_np).any(axis=1)
            valid_points = points_np[valid_mask]
            
            if len(valid_points) == 0:
                warnings.warn("No valid points for MMD computation (all contain NaN)")
                return np.array([float('nan')] * len(points_np))
                
            # Compute MMD between valid generated points and target samples
            mmd_squared = self._compute_mmd_squared(valid_points)
            mmd_value = np.sqrt(mmd_squared)
            
            # Return the same MMD value for all input points
            # (since MMD is a distributional comparison, not per-sample)
            return np.array([mmd_value] * len(points_np))
            
        except Exception as e:
            warnings.warn(f"Error during MMD computation: {e}")
            return np.array([float('nan')] * len(points))


class MMDivergenceFromGMM(GaussianMetric):
    """
    MMD metric that generates target samples from a Gaussian Mixture Model,
    compatible with existing KLDivergence and LogLikelihood metrics.
    """
    
    def __init__(self, centroids, cov_matrices, weights, n_target_samples=1000, 
                 bandwidths=None, dist_sync_on_step=False):
        """
        Initialize MMD metric using GMM parameters to generate target samples.
        
        Args:
            centroids: List of centroid coordinates for each Gaussian component
            cov_matrices: List of covariance matrices for each component
            weights: List of mixture weights for each component
            n_target_samples: Number of samples to generate from the GMM
            bandwidths: Kernel bandwidths (if None, uses adaptive selection)
            dist_sync_on_step: Whether to synchronize metric state across devices
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        # Create and configure GMM
        self.gmm = GaussianMixture(n_components=len(centroids), covariance_type="full")
        self.gmm.means_ = np.array(centroids)
        self.gmm.covariances_ = np.array(cov_matrices) 
        self.gmm.weights_ = np.array(weights)
        
        # Compute precision matrices for GMM
        try:
            self.gmm.precisions_cholesky_ = np.linalg.cholesky(
                np.linalg.inv(self.gmm.covariances_)
            )
        except np.linalg.LinAlgError:
            warnings.warn("Failed to compute precision matrices, using identity")
            self.gmm.precisions_cholesky_ = np.array([
                np.linalg.cholesky(np.eye(len(centroids[0]))) 
                for _ in centroids
            ])
        
        # Generate target samples from GMM
        try:
            target_samples, _ = self.gmm.sample(n_target_samples)
            self.mmd_metric = MMDivergence(target_samples, bandwidths, dist_sync_on_step)
        except Exception as e:
            warnings.warn(f"Failed to generate target samples from GMM: {e}")
            # Fallback: use centroids as target samples
            self.mmd_metric = MMDivergence(centroids, bandwidths, dist_sync_on_step)
    
    def compute_score(self, points):
        """Delegate to the internal MMD metric."""
        return self.mmd_metric.compute_score(points)


class WassersteinDistance(GaussianMetric):
    """
    Calculates the 1-Wasserstein distance (Earth Mover's Distance) between generated and target distributions.
    
    For 2D distributions, computes the distance for each dimension separately and aggregates them.
    The Wasserstein distance measures the minimum cost to transform one distribution into another,
    providing a robust metric for distribution comparison that is particularly useful for GANs.
    """

    def __init__(
        self, 
        target_samples: Union[np.ndarray, torch.Tensor],
        aggregation: str = "mean",
        dist_sync_on_step: bool = False
    ):
        """
        Initialize the Wasserstein distance metric.
        
        Args:
            target_samples: Reference distribution samples (N x D array where D is dimensionality)
            aggregation: How to combine distances across dimensions. Options:
                        - "mean": Average distance across dimensions
                        - "max": Maximum distance across dimensions  
                        - "sum": Sum of distances across dimensions
            dist_sync_on_step: Whether to synchronize metric state across processes
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        # Convert target samples to numpy array and store
        if isinstance(target_samples, torch.Tensor):
            self.target_samples = target_samples.cpu().numpy()
        else:
            self.target_samples = np.array(target_samples)
            
        # Remove any NaN values from target samples
        if len(self.target_samples.shape) == 1:
            # 1D case
            valid_mask = ~np.isnan(self.target_samples)
        else:
            # Multi-dimensional case
            valid_mask = ~np.isnan(self.target_samples).any(axis=1)
        self.target_samples = self.target_samples[valid_mask]
        
        if len(self.target_samples) == 0:
            raise ValueError("Target samples contain only NaN values")
            
        # Store aggregation method
        if aggregation not in ["mean", "max", "sum"]:
            raise ValueError(f"Invalid aggregation method: {aggregation}. Use 'mean', 'max', or 'sum'")
        self.aggregation = aggregation
        
        # Store dimensionality
        self.n_dims = self.target_samples.shape[1] if len(self.target_samples.shape) > 1 else 1

    def compute_score(self, points: Union[np.ndarray, torch.Tensor]) -> Union[float, List[float]]:
        """
        Compute Wasserstein distance between generated points and target distribution.
        
        Args:
            points: Generated samples (N x D array)
            
        Returns:
            Scalar Wasserstein distance value
        """
        # Convert to numpy array
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        else:
            points = np.array(points)
            
        # Handle edge cases
        if len(points) == 0:
            warnings.warn("Empty points array provided to WassersteinDistance")
            return float('inf')
            
        # Remove NaN values
        if len(points.shape) == 1:
            # 1D case
            valid_mask = ~np.isnan(points)
        else:
            # Multi-dimensional case
            valid_mask = ~np.isnan(points).any(axis=1)
        points_clean = points[valid_mask]
        
        if len(points_clean) == 0:
            warnings.warn("All generated points contain NaN values")
            return float('inf')
            
        try:
            # For 1D case
            if self.n_dims == 1:
                target_1d = self.target_samples.flatten() if len(self.target_samples.shape) > 1 else self.target_samples
                points_1d = points_clean.flatten() if len(points_clean.shape) > 1 else points_clean
                distance = wasserstein_distance(target_1d, points_1d)
                return distance
                
            # For multi-dimensional case, compute distance per dimension
            distances = []
            for dim in range(self.n_dims):
                target_dim = self.target_samples[:, dim]
                points_dim = points_clean[:, dim]
                
                # Compute Wasserstein distance for this dimension
                dim_distance = wasserstein_distance(target_dim, points_dim)
                distances.append(dim_distance)
                
            distances = np.array(distances)
            
            # Handle any NaN distances
            if np.any(np.isnan(distances)):
                warnings.warn("NaN values detected in dimension-wise Wasserstein distances")
                distances = distances[~np.isnan(distances)]
                if len(distances) == 0:
                    return float('inf')
                    
            # Aggregate distances across dimensions
            if self.aggregation == "mean":
                return float(np.mean(distances))
            elif self.aggregation == "max":
                return float(np.max(distances))
            elif self.aggregation == "sum":
                return float(np.sum(distances))
                
        except Exception as e:
            warnings.warn(f"Error computing Wasserstein distance: {e}")
            return float('inf')
            
        return float('inf')  # Fallback


class MMDDistance(GaussianMetric):
    """
    Maximum Mean Discrepancy (MMD) distance metric using RBF kernel.
    
    MMD is a kernel-based statistical test that measures the distance between
    distributions in a reproducing kernel Hilbert space (RKHS). It provides
    a robust non-parametric comparison between generated and target distributions.
    """

    def __init__(
        self,
        target_samples: Union[np.ndarray, torch.Tensor],
        kernel: str = "rbf",
        gamma: float = 1.0,
        dist_sync_on_step: bool = False
    ):
        """
        Initialize MMD distance metric.
        
        Args:
            target_samples: Reference distribution samples (N x D array)
            kernel: Kernel type ('rbf' only supported currently)
            gamma: RBF kernel bandwidth parameter (higher = more localized)
            dist_sync_on_step: Whether to synchronize metric state across processes
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        # Convert target samples to torch tensor
        if isinstance(target_samples, np.ndarray):
            self.target_samples = torch.from_numpy(target_samples).float()
        else:
            self.target_samples = target_samples.float()
            
        # Remove NaN values
        valid_mask = ~torch.isnan(self.target_samples).any(dim=1)
        self.target_samples = self.target_samples[valid_mask]
        
        if len(self.target_samples) == 0:
            raise ValueError("Target samples contain only NaN values")
            
        self.gamma = gamma
        self.kernel = kernel
        
        if kernel != "rbf":
            raise NotImplementedError(f"Kernel {kernel} not implemented. Only 'rbf' is supported.")

    def rbf_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute RBF (Gaussian) kernel matrix between two sets of points.
        
        K(x,y) = exp(-gamma * ||x-y||^2)
        
        Args:
            X: First set of points (N1 x D)
            Y: Second set of points (N2 x D)
            
        Returns:
            Kernel matrix (N1 x N2)
        """
        # Compute pairwise squared distances
        pairwise_dists_sq = torch.cdist(X, Y) ** 2
        return torch.exp(-self.gamma * pairwise_dists_sq)

    def compute_score(self, points: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Compute MMD distance between generated points and target distribution.
        
        MMD^2 = E[K(x,x')] - 2*E[K(x,y)] + E[K(y,y')]
        where x,x' ~ generated distribution, y,y' ~ target distribution
        
        Args:
            points: Generated samples (N x D array)
            
        Returns:
            MMD distance value
        """
        # Convert to torch tensor
        if isinstance(points, np.ndarray):
            X = torch.from_numpy(points).float()
        else:
            X = points.float()
            
        # Handle edge cases
        if len(X) == 0:
            warnings.warn("Empty points array provided to MMDDistance")
            return float('inf')
            
        # Remove NaN values
        valid_mask = ~torch.isnan(X).any(dim=1)
        X = X[valid_mask]
        
        if len(X) == 0:
            warnings.warn("All generated points contain NaN values")
            return float('inf')
            
        try:
            # Ensure both tensors are on the same device
            Y = self.target_samples
            if X.device != Y.device:
                Y = Y.to(X.device)
                
            # Compute kernel matrices
            K_XX = self.rbf_kernel(X, X).mean()
            K_YY = self.rbf_kernel(Y, Y).mean()
            K_XY = self.rbf_kernel(X, Y).mean()
            
            # Compute MMD squared
            mmd_squared = K_XX + K_YY - 2 * K_XY
            
            # Take square root and ensure non-negative
            mmd = torch.sqrt(torch.clamp(mmd_squared, min=0.0))
            
            return float(mmd.item())
            
        except Exception as e:
            warnings.warn(f"Error computing MMD distance: {e}")
            return float('inf')


class ConvergenceTracker:
    """
    Tracks training convergence based on loss and metric stability.
    
    This class monitors training progress and detects convergence based on:
    - Loss stabilization (small changes over recent epochs)
    - Metric stability (validation metrics not improving)
    - Early stopping criteria
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        monitor_metric: str = "KLDivergence",
        window_size: int = 5
    ):
        """
        Initialize convergence tracker.
        
        Args:
            patience: Number of epochs to wait before declaring convergence
            min_delta: Minimum change to qualify as an improvement
            monitor_metric: Primary metric to monitor for convergence
            window_size: Number of recent epochs to consider for stability
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor_metric = monitor_metric
        self.window_size = window_size
        
        # Tracking variables
        self.metric_history: Dict[str, List[float]] = {}
        self.loss_history: Dict[str, List[float]] = {"d_loss": [], "g_loss": []}
        self.best_metric_value: Optional[float] = None
        self.epochs_without_improvement = 0
        self.converged = False
        self.convergence_epoch: Optional[int] = None
        
    def update(
        self,
        epoch: int,
        metrics: Dict[str, float],
        d_loss: Optional[float] = None,
        g_loss: Optional[float] = None
    ) -> Dict[str, Union[bool, int, float]]:
        """
        Update convergence tracking with new metrics.
        
        Args:
            epoch: Current training epoch
            metrics: Dictionary of validation metrics
            d_loss: Discriminator loss (optional)
            g_loss: Generator loss (optional)
            
        Returns:
            Dictionary containing convergence information
        """
        # Store losses
        if d_loss is not None:
            self.loss_history["d_loss"].append(d_loss)
        if g_loss is not None:
            self.loss_history["g_loss"].append(g_loss)
            
        # Store metrics
        for metric_name, value in metrics.items():
            if metric_name not in self.metric_history:
                self.metric_history[metric_name] = []
            if not (np.isnan(value) or np.isinf(value)):
                self.metric_history[metric_name].append(value)
            
        # Check convergence based on monitor metric
        if self.monitor_metric in metrics:
            current_value = metrics[self.monitor_metric]
            
            if not (np.isnan(current_value) or np.isinf(current_value)):
                # Initialize best value if first valid measurement
                if self.best_metric_value is None:
                    self.best_metric_value = current_value
                    self.epochs_without_improvement = 0
                else:
                    # Check if current value is better (lower for most metrics)
                    improvement = self.best_metric_value - current_value
                    
                    if improvement > self.min_delta:
                        self.best_metric_value = current_value
                        self.epochs_without_improvement = 0
                    else:
                        self.epochs_without_improvement += 1
                        
                # Check for convergence
                if (self.epochs_without_improvement >= self.patience and 
                    not self.converged):
                    self.converged = True
                    self.convergence_epoch = epoch
                    
        return self.get_convergence_info(epoch)
        
    def get_convergence_info(self, current_epoch: int) -> Dict[str, Union[bool, int, float]]:
        """
        Get current convergence information.
        
        Args:
            current_epoch: Current training epoch
            
        Returns:
            Dictionary with convergence status and statistics
        """
        info = {
            "converged": self.converged,
            "convergence_epoch": self.convergence_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
            "best_metric_value": self.best_metric_value,
            "current_epoch": current_epoch
        }
        
        # Add loss stability metrics
        if len(self.loss_history["g_loss"]) >= self.window_size:
            recent_g_losses = self.loss_history["g_loss"][-self.window_size:]
            info["g_loss_stability"] = float(np.std(recent_g_losses))
            info["g_loss_trend"] = float(np.mean(np.diff(recent_g_losses)))
            
        if len(self.loss_history["d_loss"]) >= self.window_size:
            recent_d_losses = self.loss_history["d_loss"][-self.window_size:]
            info["d_loss_stability"] = float(np.std(recent_d_losses))
            info["d_loss_trend"] = float(np.mean(np.diff(recent_d_losses)))
            
        return info
        
    def should_stop_early(self) -> bool:
        """
        Determine if training should stop early based on convergence.
        
        Returns:
            True if early stopping criteria are met
        """
        return self.converged
        
    def reset(self):
        """Reset convergence tracker for new training run."""
        self.metric_history.clear()
        self.loss_history = {"d_loss": [], "g_loss": []}
        self.best_metric_value = None
        self.epochs_without_improvement = 0
        self.converged = False
        self.convergence_epoch = None


ALL_METRICS = {
    "IsPositive": IsPositive,
    "LogLikelihood": LogLikelihood,
    "KLDivergence": KLDivergence,
    "WassersteinDistance": WassersteinDistance,
    "MMDDistance": MMDDistance,
    "MMDivergence": MMDivergence,
    "MMDivergenceFromGMM": MMDivergenceFromGMM,
}
