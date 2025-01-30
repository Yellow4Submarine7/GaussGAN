import argparse
from datetime import datetime
from functools import partial
import subprocess


import mlflow
import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt


from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger, WandbLogger

from source.data import GaussianDataModule #, GaussianDataset
from source.model import GaussGan
from source.nn import MLPDiscriminator, MLPGenerator, QuantumNoise, ClassicalNoise, QuantumShadowNoise

from source.utils import parse_args


def plot_dataset(inps1, inps2):

    plt.figure(figsize=(8, 6))
    plt.scatter(inps1[:, 0], inps1[:, 1], color='blue', label='Class -1')
    plt.scatter(inps2[:, 0], inps2[:, 1], color='red', label='Class 1')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.title('2D Scatter Plot of Gaussian Distributions')
    plt.savefig("dataset.png")

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

def main():
    args = parse_args()
    device = torch.device(
        "cuda" if args.accelerator == "gpu" and torch.cuda.is_available() else "cpu"
    )



    # mean1 = torch.tensor([-6, 3]).float()
    # cov1 = torch.tensor([[2,0], [0, 2]]).float()  # Valid covariance matrix

    # mean2 = torch.tensor([6, 3]).float()
    # cov2 = torch.tensor([[2, 0], [0, 2]]).float()  # Valid covariance matrix


    # inps1, inps2, targs1, targs2, inputs, targets = generate_dataset(args.dataset_size, mean1, cov1, mean2, cov2)
    # plot_dataset(inps1, inps2)
    

    # Create dataset by sampling from a uniform distribution in the range [-10, 10]
    inputs = 20 * torch.rand((args.dataset_size, 2)) - 10
    targets = torch.zeros((args.dataset_size, 1))

    
    # plot the dataset
    plt.figure(figsize=(8, 6))
    plt.scatter(inputs[:, 0], inputs[:, 1], color='blue', label='Class -1')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.title('2D Scatter Plot of Uniform Distributions')
    plt.savefig("images/uniform_dataset.png")


    dataset = TensorDataset(
        inputs, targets
    )
    datamodule = GaussianDataModule(dataset, batch_size=args.batch_size)






    # Initialize networks
    if args.generator_type == "classical":
        G_part_1 = ClassicalNoise(z_dim=args.z_dim, classical_generator_type=args.classical_generator_type)

    elif args.generator_type == "quantum_samples":
        G_part_1 = QuantumNoise(
            z_dim=args.z_dim,
        )
    elif args.generator_type == "quantum_shadows":
        G_part_1 = QuantumShadowNoise(
            z_dim=args.z_dim,
        )
    
    G_part_2 = MLPGenerator(
        z_dim=args.z_dim,
        hidden_dims=4*[128],
    )

    G = torch.nn.Sequential(G_part_1, G_part_2)
    D = MLPDiscriminator(hidden_dims=4*[128],)
    V = MLPDiscriminator(hidden_dims=[1, 1],)
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
        grad_penalty=args.grad_penalty,
        killer=args.killer,
        n_critic=args.n_critic,
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
    if args.generator_type == "classical":
        filename = f"best-checkpoint-classical-{current_date_time}"
    elif args.use_shadows:
        filename = f"best-checkpoint-quantum-shadows-{current_date_time}"
    else:
        filename = f"best-checkpoint-quantum-no-shadows-{current_date_time}"


    mlflow_logger = MLFlowLogger(experiment_name="GaussGan")

    hparams = vars(args)
    mlflow_logger.log_hyperparams(hparams)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        logger=mlflow_logger,
        log_every_n_steps=1,
        limit_val_batches=3,
        # callbacks=[checkpoint_callback],
    )

    if args.stage == "train":
        # Start training
        trainer.fit(model=model, datamodule=datamodule)
    elif args.stage == "test":
        # Start testing
        trainer.test(model=model, datamodule=datamodule)



if __name__ == "__main__":
    main()
    #subprocess.run(["python", "scripts/plot_gaussians.py"])

    #print("TODO implement class Generator so to make easily swtich from quantum to classical noise generator")
