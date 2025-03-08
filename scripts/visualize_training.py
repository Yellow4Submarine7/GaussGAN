import mlflow
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import utils_scripts


def plot_training(fig, axes, csv_artifacts, run_id, filename):

    for idx, csv_path in enumerate(csv_artifacts):
        # print("Plotting artifact", csv_path)
        data_path = client.download_artifacts(run_id, csv_path)
        df = pd.read_csv(data_path, header=None, sep=",")
        # Handle 1D vs 2D axes array
        if axes.ndim == 1:
            ax = axes[idx % 4]
        else:
            ax = axes[idx // 4, idx % 4]

        df = (
            df.drop(0).apply(pd.to_numeric).round(3)
        )  # Drop the first row and convert to numeric

        # Convert to NumPy array
        data = df.values
        x_values = data[:, 0]
        y_values = data[:, 1]
        # print(len(inps1[:, 0]), len(inps1[:, 1]))

        if np.all(np.isnan(x_values)) or np.all(
            np.isnan(y_values)
        ):  # se tutto sta nan, skippa e non plottare..
            break

        ax.scatter(inps1[:, 0], inps1[:, 1], color="blue", label="Class -1")
        ax.scatter(inps2[:, 0], inps2[:, 1], color="red", label="Class +1")

        ax.scatter(x_values, y_values, s=3)

        ax.axhline(y=0, color="r", linestyle="--")
        ax.axvline(x=0, color="r", linestyle="--")

        ax.set_xlim(-11, 11)
        ax.set_ylim(-11, 11)

        ax.set_title(f"Epoch {idx+1}")

    fig.suptitle(f"Run ID: {run_id}", fontsize=16)

    print(f"Saving plot to {filename}")
    plt.savefig(filename)
    # plt.tight_layout()
    # plt.show()
    plt.close()


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


mean1 = torch.tensor([-5.0, 5.0])
cov1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
mean2 = torch.tensor([5.0, 5.0])
cov2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])


if __name__ == "__main__":

    client = mlflow.tracking.MlflowClient()
    args = utils_scripts.get_parser()

    experiment = client.get_experiment_by_name("GaussGAN-Optuna")
    print("Experiment ID:", experiment.experiment_id)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attribute.start_time desc"],
        # max_results=256,
    )
    print("Found", len(runs), "runs")

    inps1, inps2, targs1, targs2, inputs, targets = utils_scripts.generate_dataset(
        args.dataset_size, mean1, cov1, mean2, cov2
    )
    base = 0
    for i, run in enumerate(runs[base:]):

        run_id = run.info.run_id
        print("Index", base + i, "Runid:", run_id)

        filename = f"images/training_{experiment.experiment_id}_{run_id}.png"

        if os.path.exists(filename):
            print("Skipping")
            continue
        else:
            artifacts = client.list_artifacts(run_id)

            csv_artifacts = [a.path for a in artifacts if a.path.endswith(".csv")]

            fig_rows = int((len(csv_artifacts) + 3) // 4)  # why?!
            fig, axes = plt.subplots(
                nrows=fig_rows, ncols=4, figsize=(12, 3 * fig_rows)
            )

            plot_training(fig, axes, csv_artifacts, run_id, filename)
