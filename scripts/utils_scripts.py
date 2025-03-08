import argparse
import torch


def get_parser():
    parser = argparse.ArgumentParser(description="Generate and save datasets.")
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["NORMAL", "UNIFORM"],
        default="NORMAL",
        help="Type of dataset to generate",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=10000,
        help="Number of points in the dataset",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="GaussGAN-Optuna",
        help="Name of the experiment",
    )
    args = parser.parse_args()
    return args


def generate_dataset(n_points, mean1, cov1, mean2, cov2):
    # Create MultivariateNormal distributions
    dist1 = torch.distributions.MultivariateNormal(mean1, cov1)
    dist2 = torch.distributions.MultivariateNormal(mean2, cov2)

    # Sample points from the distributions
    inps1 = dist1.sample((n_points,))
    targs1 = -torch.ones(n_points, 1)
    inps2 = dist2.sample((n_points,))
    targs2 = torch.ones(n_points, 1)

    # Combine the inputs and targets
    inputs = torch.cat([inps1, inps2])
    targets = torch.cat([targs1, targs2])

    return inps1, inps2, targs1, targs2, inputs, targets
