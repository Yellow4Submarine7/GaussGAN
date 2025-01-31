import argparse
from datetime import datetime
from functools import partial
import subprocess

import pickle
import mlflow
import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import pdb

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

# from source.data import GaussianDataModule #, GaussianDataset
from source.model import GaussGan

from source.nn import (
    MLPDiscriminator,
    MLPGenerator,
    QuantumNoise,
    ClassicalNoise,
    QuantumShadowNoise,
)



from source.utils import parse_args, load_data
import random
import numpy as np


import torch.multiprocessing as mp


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


def main():

    args = parse_args()

    device = torch.device(
        "cuda" if args.accelerator == "gpu" and torch.cuda.is_available() else "cpu"
    )

    datamodule, gaussians = load_data(args)


    if (
        args.generator_type == "classical_uniform"
        or args.generator_type == "classical_normal"
    ):
        G_part_1 = ClassicalNoise(z_dim=args.z_dim, generator_type=args.generator_type)
    elif args.generator_type == "quantum_sampels":
        G_part_1 = QuantumNoise(
            z_dim=args.z_dim,
        )
    elif args.generator_type == "quantum_shadows":
        G_part_1 = QuantumShadowNoise(
            z_dim=args.z_dim,
        )
    else:
        raise ValueError("Invalid generator type")

    G_part_2 = MLPGenerator(
        z_dim=args.z_dim,
        hidden_dims=4 * [128],
    )

    G = torch.nn.Sequential(G_part_1, G_part_2)
    D = MLPDiscriminator(
        hidden_dims=4 * [128],
    )
    V = MLPDiscriminator(
        hidden_dims=[1, 1],
    )
    G.to(device)
    D.to(device)
    V.to(device)

    print("Nets created")

    # Setup the GaussGan model
    model = GaussGan(
        G,
        D,
        V,
        optimizer=partial(torch.optim.RAdam, lr=args.learning_rate, betas=(0.9, 0.99)),
        killer=args.killer,
        n_critic=args.n_critic,
        gradient_penalty=args.grad_penalty,
        metrics=["IsPositive", "LogLikelihood"],
        gaussians=gaussians,
        validation_samples=args.validation_samples,
    )
    model.to(device)

    # if args.checkpoint_path is not None:
    #     # Load the best model
    #     model = GaussGan.load_from_checkpoint(
    #         "checkpoints/" + args.checkpoint_path + ".ckpt",
    #         dataset=dataset,
    #         generator=G,
    #         discriminator=D,
    #         predictor=V,
    #         optimizer=partial(torch.optim.RMSprop, lr=args.learning_rate),
    #     )
    #     model.to(device)

    # Define the checkpoint callback
    current_date_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    filename = f"best-checkpoint-{args.generator_type}-{current_date_time}"

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=filename,
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=5,  # Save every 5 epochs
    )

    mlflow_logger = MLFlowLogger(experiment_name=args.experiment_name)
    run_id = mlflow_logger.run_id
    print(f"---Run ID: {run_id}")

    hparams = vars(args)
    mlflow_logger.log_hyperparams(hparams)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        logger=mlflow_logger,
        log_every_n_steps=1,
        limit_val_batches=3,
        callbacks=[checkpoint_callback],
    )

    if args.stage == "train":
        trainer.fit(model=model, datamodule=datamodule)
    elif args.stage == "test":
        trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    #mp.set_start_method('spawn', force=True)
    main()
