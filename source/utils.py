import argparse
from torch.utils.data import TensorDataset
from source.data import GaussianDataModule
import pickle
import torch
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)  # 1
    np.random.seed(seed)  # 2
    torch.manual_seed(seed)  # 3
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # 4


def load_data(args):

    if args["dataset_type"] == "UNIFORM":
        with open("data/uniform.pickle", "rb") as f:
            data = pickle.load(f)
            gaussians = {}

    elif args["dataset_type"] == "NORMAL":
        with open("data/normal.pickle", "rb") as f:
            data = pickle.load(f)
            gaussians = {
                "centroids": [data["mean1"], data["mean2"]],
                "covariances": [data["cov1"], data["cov2"]],
                "weights": [0.5],
            }

    inputs, targets = data["inputs"], data["targets"]

    dataset = TensorDataset(inputs, targets)
    datamodule = GaussianDataModule(dataset, batch_size=args["batch_size"])
    return datamodule, gaussians


def return_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train the GaussGan model")

    parser.add_argument(
        "--z_dim",
        type=int,
        help="Dimension of the latent space",
    )
    parser.add_argument(
        "--generator_type",
        type=str,
        help="Type of generator to use ('classical_uniform', 'classical_normal', 'quantum_samples', 'quantum_shadows')",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        help="Maximum number of epochs to train",
    )
    parser.add_argument(
        "--grad_penalty",
        type=float,
        help="Gradient penalty regularization factor of Wasserstain GAN",
    )
    parser.add_argument(
        "--n_critic",
        type=int,
        help="Number of discriminator updates per generator update",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate for the optimizer",
    )

    parser.add_argument(
        "--stage",
        type=str,
        help="Stage to run ('train', 'test')",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to the checkpoint directory",
    )

    parser.add_argument(
        "--dataset_type",
        type=str,
        help="Distribution of the generator ('NORMAL', 'UNIFORM') ",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name of the experiment",
    )

    parser.add_argument(
        "--metrics",
        nargs="+",  # This allows multiple values
        help="List of metrics to compute",
    )

    parser.add_argument(
        "--agg_method",
        type=str,
        help="Aggregation method for the rewards.",
    )

    parser.add_argument(
        "--accelerator",
        type=str,
        help="Device to use",
    )
    parser.add_argument(
        "--killer",
        type=bool,
        help="Kill one gaussian",
    )
    parser.add_argument(
        "--validation_samples",
        type=int,
        help="Number of validation samples step",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--nn_gen",
        type=str,
        help="Neural network architecture for the generator",
    )
    parser.add_argument(
        "--nn_disc",
        type=str,
        help="Neural network architecture for the discriminator",
    )
    parser.add_argument(
        "--nn_validator",
        type=str,
        help="Neural network architecture for the validator",
    )
    parser.add_argument(
        "--non_linearity",
        type=str,
        help="Non-linear activation function to use in neural networks",
    )

    parser.add_argument(
        "--quantum_qubits",
        type=int,
        help="Number of qubits for quantum circuit",
    )
    parser.add_argument(
        "--quantum_layers",
        type=int,
        help="Number of layers for quantum circuit",
    )
    parser.add_argument(
        "--quantum_shots",
        type=int,
        help="Number of shots for quantum measurement",
    )
    parser.add_argument(
        "--quantum_basis",
        type=int,
        help="Number of basis for quantum shadow",
    )

    return parser
