import mlflow
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import utils_scripts

def plot_latest_training(csv_artifacts, run_id, filename):
    fig_rows = int((len(csv_artifacts) + 3) // 4)
    fig, axes = plt.subplots(nrows=fig_rows, ncols=4, figsize=(12, 3 * fig_rows))
    
    mean1 = torch.tensor([-5.0, 5.0])
    cov1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mean2 = torch.tensor([5.0, 5.0])
    cov2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    
    inps1, inps2, _, _, _, _ = utils_scripts.generate_dataset(
        1000, mean1, cov1, mean2, cov2
    )

    client = mlflow.tracking.MlflowClient()

    for idx, csv_path in enumerate(csv_artifacts):
        data_path = client.download_artifacts(run_id, csv_path)
        df = pd.read_csv(data_path, header=None, sep=",")
        
        if axes.ndim == 1:
            ax = axes[idx % 4]
        else:
            ax = axes[idx // 4, idx % 4]

        df = df.drop(0).apply(pd.to_numeric).round(3)
        data = df.values
        x_values = data[:, 0]
        y_values = data[:, 1]

        if np.all(np.isnan(x_values)) or np.all(np.isnan(y_values)):
            break

        ax.scatter(inps1[:, 0], inps1[:, 1], color="blue", label="Class -1", alpha=0.5)
        ax.scatter(inps2[:, 0], inps2[:, 1], color="red", label="Class +1", alpha=0.5)
        
        ax.scatter(x_values, y_values, s=3, color='black', alpha=0.5, label='Generated')

        ax.axhline(y=0, color="r", linestyle="--", alpha=0.3)
        ax.axvline(x=0, color="r", linestyle="--", alpha=0.3)

        ax.set_xlim(-11, 11)
        ax.set_ylim(-11, 11)
        ax.set_title(f"Epoch {idx+1}")
        
        if idx == 0:
            ax.legend()

    fig.suptitle(f"Latest Training Run (ID: {run_id})", fontsize=16)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    os.makedirs("images", exist_ok=True)
    
    client = mlflow.tracking.MlflowClient()
    
    experiment = client.get_experiment_by_name("GaussGAN-manual")
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attribute.start_time desc"],
        max_results=1
    )
    
    if not runs:
        print("No runs found in the experiment")
        return
        
    latest_run = runs[0]
    run_id = latest_run.info.run_id
    print(f"Processing latest run: {run_id}")
    
    artifacts = client.list_artifacts(run_id)
    csv_artifacts = [a.path for a in artifacts if a.path.endswith(".csv")]
    
    if not csv_artifacts:
        print("No CSV artifacts found in the run")
        return
        
    filename = f"images/training_latest.png"
    
    plot_latest_training(csv_artifacts, run_id, filename)
    print(f"Training visualization saved to {filename}")

if __name__ == "__main__":
    main() 