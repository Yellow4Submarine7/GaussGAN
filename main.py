import argparse
from datetime import datetime
from functools import partial
import subprocess
import ast
import yaml
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
from source.utils import set_seed, load_data


from source.nn import (
    MLPDiscriminator,
    MLPGenerator,
    QuantumNoise,
    ClassicalNoise,
    QuantumShadowNoise,
)


from source.utils import return_parser, load_data
import random
import numpy as np


import torch.multiprocessing as mp

from source.utils import set_seed


def main():

    parser = return_parser()

    with open("config.yaml", "r") as file:
        final_args = yaml.safe_load(file)
        cmd_args = parser.parse_args()

        update_dict = {k: v for k, v in vars(cmd_args).items() if v is not None}
        run_instance = "_".join(f"{k}-{v}" for k, v in update_dict.items())
        random_number = random.randint(0, 10000)
        run_instance = f"{random_number}_" + run_instance
        final_args.update(update_dict)

    set_seed(final_args["seed"])

    # pdb.set_trace()

    device = torch.device(
        "cuda"
        if final_args["accelerator"] == "gpu" and torch.cuda.is_available()
        else (
            "mps"
            if final_args["accelerator"] == "mps" and torch.backends.mps.is_available()
            else "cpu"
        )
    )

    datamodule, gaussians = load_data(final_args)

    if (
        final_args["generator_type"] == "classical_uniform"
        or final_args["generator_type"] == "classical_normal"
    ):
        G_part_1 = ClassicalNoise(
            z_dim=final_args["z_dim"], generator_type=final_args["generator_type"]
        )
    elif final_args["generator_type"] == "quantum_sampels":
        G_part_1 = QuantumNoise(
            z_dim=final_args["z_dim"],
        )
    elif final_args["generator_type"] == "quantum_shadows":
        G_part_1 = QuantumShadowNoise(
            z_dim=final_args["z_dim"],
        )
    else:
        raise ValueError("Invalid generator type")
    G_part_2 = MLPGenerator(
        non_linearity=final_args["non_linearity"],
        z_dim=final_args["z_dim"],
        hidden_dims=ast.literal_eval(final_args["nn_gen"]),
    )
    G = torch.nn.Sequential(G_part_1, G_part_2)

    D = MLPDiscriminator(
        non_linearity=final_args["non_linearity"],
        hidden_dims=ast.literal_eval(final_args["nn_disc"]),
    )

    V = MLPDiscriminator(
        non_linearity=final_args["non_linearity"],
        hidden_dims=ast.literal_eval(final_args["nn_validator"]),
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
        optimizer=partial(
            # torch.optim.RAdam, lr=final_args["learning_rate"], betas=(0.9, 0.99)
            torch.optim.Adam,
            lr=final_args["learning_rate"],
        ),
        killer=final_args["killer"],
        n_critic=final_args["n_critic"],
        grad_penalty=final_args["grad_penalty"],
        metrics=list(final_args["metrics"]),
        gaussians=gaussians,
        validation_samples=final_args["validation_samples"],
        non_linearity=final_args[
            "non_linearity"
        ],  # code stinks, we don't need to pass it here, is only to save this
    )
    model.to(device)

    mlflow_logger = MLFlowLogger(
        experiment_name=final_args["experiment_name"], run_name=run_instance
    )
    run_id = mlflow_logger.run_id

    print(f"---Run ID: {run_id}")

    filename = f"run_id-{run_id}"

    # LOGGING WITH MLFLOW
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=filename + "-{epoch:03d}",  # Add epoch number to filename
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=5,  # Save every 5 epochs
        save_last=True,  # Save the last checkpoint
    )

    hparams = final_args
    mlflow_logger.log_hyperparams(hparams)

    trainer = Trainer(
        max_epochs=final_args["max_epochs"],
        accelerator=final_args["accelerator"],
        logger=mlflow_logger,
        log_every_n_steps=1,
        limit_val_batches=3,
        callbacks=[checkpoint_callback],
    )

    if final_args["stage"] == "train":
        trainer.fit(model=model, datamodule=datamodule)
    elif final_args["stage"] == "test":
        trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
