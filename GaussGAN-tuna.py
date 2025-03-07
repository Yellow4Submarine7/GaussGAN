import optuna
import subprocess
import mlflow
import argparse
import os
import re
import pdb


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning of GaussGAN with Optuna"
    )
    parser.add_argument(
        "--n_trials", type=int, default=50, help="Number of Optuna trials"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="Maximum epochs for training"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="GaussGAN-Optuna",
        help="Name of the MLflow experiment",
    )
    return parser.parse_args()


def objective(trial):
    # Define the hyperparameters to tune
    lr = trial.suggest_categorical("learning_rate", [0.01, 0.001, 0.0001])
    seed = trial.suggest_categorical("seed", [0, 1, 2, 3, 4, 5, 6])
    n_critic = trial.suggest_categorical("n_critic", [3, 4, 5, 6, 8])

    # Neural network architectures
    nn_gen_options = ["[32,32]", "[32,64]", "[64,64]", "[128,128]", "[256,256]"]
    nn_disc_options = ["[32,32]", "[32,64]", "[64,64]", "[128,128]", "[256,256]"]

    nn_gen = trial.suggest_categorical("nn_gen", nn_gen_options)
    nn_disc = trial.suggest_categorical("nn_disc", nn_disc_options)

    z_dim = trial.suggest_categorical("z_dim", [3, 5, 10, 12])
    non_linearity = trial.suggest_categorical(
        "non_linearity", ["ReLU", "LeakyReLU", "Tanh", "Sigmoid"]
    )
    grad_penalty = trial.suggest_categorical("grad_penalty", [5, 10, 15])

    # Build command to run
    cmd = [
        "python",
        "main.py",
        "--max_epochs",
        str(args.max_epochs),
        "--seed",
        str(seed),
        "--n_critic",
        str(n_critic),
        "--nn_gen",
        nn_gen,
        "--nn_disc",
        nn_disc,
        "--experiment_name",
        args.experiment_name,
        "--z_dim",
        str(z_dim),
        "--learning_rate",
        str(lr),
        "--non_linearity",
        non_linearity,
        "--grad_penalty",
        str(grad_penalty),
        "--accelerator",
        "gpu",
    ]

    # Run the command and capture the output
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout.strip().split("\n")

    # print(cmd, output)

    # import pdb

    # pdb.set_trace()

    run_id = None
    for line in output:
        match = re.search(r"---Run ID: (\w+)", line)
        if match:
            print("Matched", match.group(1))
            run_id = match.group(1)
            break

    if run_id is None:
        raise ValueError("Run ID not found in the output")

    # Retrieve the maximum log-likelihood from MLFlow
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    max_log_likelihood = run.data.metrics["ValidationStep_FakeData_KLDivergence"]

    return max_log_likelihood


if __name__ == "__main__":
    args = parse_args()
    study = optuna.create_study(direction="maximize")

    def print_callback(study, trial):
        print(
            f"Trial {trial.number} finished with value: {trial.value} and parameters: {trial.params}"
        )

    study.optimize(
        objective,
        n_trials=args.n_trials,
        callbacks=[print_callback],
        n_jobs=5,  # Use all available CPU cores (-1), or specify a number like 4
    )

    print("Best hyperparameters: ", study.best_params)
    print("Best value: ", study.best_value)
