import argparse


import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the GaussGan model"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="train",
        help="Stage to run ('train', 'test')",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "--generator_type",
        type=str,
        default="classical",
        help="Type of generator to use ('classical', 'quantum')",
    )
    parser.add_argument(
        "--use_shadows",
        type=bool,
        default=False,  # or shadow
        help="Use shadow noise generator for the quantum generator",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/dataset.pkl",
        help="Path to the Gaussian dataset file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="Maximum number of epochs to train",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--grad_penalty",
        type=float,
        default=10,
        help="Gradient penalty regularization factor of Wasserstain GAN",
    )
    parser.add_argument(
        "--process_method",
        type=str,
        default="soft_gumbel",
        help="Method to process the output probabilities ('soft_gumbel', 'hard_gumbel')",
    )
    parser.add_argument(
        "--agg_method",
        type=str,
        default="prod",
        help="Aggregation method for the rewards.",
    )
    parser.add_argument(
        "--n_critic",
        type=int,
        default=2,
        help="Number of discriminator updates per generator update",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=5000,
        help="Number of pints in the training set",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--killer",
        type=bool,
        default=False,
        help="Kill one gaussian",
    )
    parser.add_argument(
        "--z_dim",
        type=int,
        default=100,
        help="Dimension of the latent space",
    )
    parser.add_argument(
        "--validation_samples",
        type=int,
        default=2000,
        help="Number of validation samples step",
    )
    return parser.parse_args()