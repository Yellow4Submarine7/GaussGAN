import torch
import pickle
import os
import argparse
import matplotlib.pyplot as plt
import utils_scripts


def plot_dataset(inps1, inps2):

    plt.figure(figsize=(8, 6))
    plt.scatter(inps1[:, 0], inps1[:, 1], color="blue", label="Class -1")
    plt.scatter(inps2[:, 0], inps2[:, 1], color="red", label="Class +1")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.title("2D Scatter Plot of Gaussian Distributions")
    plt.savefig("images/gaussian_dataset.png")


def plot_uniform(inputs):
    # plot the dataset
    plt.figure(figsize=(8, 6))
    plt.scatter(inputs[:, 0], inputs[:, 1], color="blue", label="Class -1")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.title("2D Scatter Plot of Uniform Distributions")
    plt.savefig("images/uniform_dataset.png")


mean1 = torch.tensor([-5.0, 5.0])
cov1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
mean2 = torch.tensor([5.0, 5.0])
cov2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])


if __name__ == "__main__":

    args = utils_scripts.get_parser()

    # Ensure the directory exists
    os.makedirs("data", exist_ok=True)

    if args.dataset_type == "NORMAL":

        inps1, inps2, targs1, targs2, inputs, targets = utils_scripts.generate_dataset(
            args.dataset_size, mean1, cov1, mean2, cov2
        )

        with open("data/normal.pickle", "wb") as f:
            pickle.dump(
                {
                    "inputs": inputs,
                    "targets": targets,
                    "mean1": mean1,
                    "mean2": mean2,
                    "cov1": cov1,
                    "cov2": cov2,
                },
                f,
            )
        plot_dataset(inps1, inps2)

    elif args.dataset_type == "UNIFORM":
        inputs = 20 * torch.rand((args.dataset_size, 2)) - 10
        targets = torch.zeros((args.dataset_size, 1))

        with open("data/uniform.pickle", "wb") as f:
            pickle.dump({"inputs": inputs, "targets": targets}, f)
        plot_uniform(inputs)
