import mlflow
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

experiment_id = "0"
client = mlflow.tracking.MlflowClient()

experiment = client.get_experiment_by_name("default")
# experiment_id = experiment.experiment_id
runs = client.search_runs(experiment_ids=[experiment_id], order_by=["attribute.start_time desc"], max_results=1)
run_id = runs[0].info.run_id


artifacts = client.list_artifacts(run_id)

csv_artifacts = [a.path for a in artifacts if a.path.endswith(".csv")]
csv_artifacts=csv_artifacts

print("Found CSV artifacts")

fig_rows = (len(csv_artifacts) + 3) // 4
fig, axes = plt.subplots(nrows=fig_rows, ncols=4, figsize=(12, 3*fig_rows))



for idx, csv_path in enumerate(csv_artifacts):
    # print("Plotting artifact", csv_path)
    data_path = client.download_artifacts(run_id, csv_path)
    df = pd.read_csv(data_path, header=None, sep=",")
    ax = axes[idx//4, idx%4]

    df = df.drop(0).apply(pd.to_numeric).round(3)  # Drop the first row and convert to numeric

    # Convert to NumPy array
    data = df.values
    x_values = data[:, 0]
    y_values = data[:, 1]

    ax.scatter(x_values, y_values, s=3)

    ax.axhline(y=0, color='r', linestyle='--')
    ax.axvline(x=0, color='r', linestyle='--')

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)


    ax.set_title(f"Epoch {idx+1}")

fig.suptitle(f"Run ID: {run_id}", fontsize=16)

filename = f"training_{experiment_id}_{run_id}.png"
print(f"Saving plot to {filename}")
plt.savefig(filename)
plt.savefig("training.png")
plt.tight_layout()
#plt.show()