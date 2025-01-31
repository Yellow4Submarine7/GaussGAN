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
    return parser.parse_args()


def objective(trial):
    # Define the hyperparameters to tune
    generator_type = trial.suggest_categorical(
        "generator_type", ["classical_uniform", "classical_normal"]
    )  # , "quantum_samples", "quantum_shadows"])
    grad_penalty = trial.suggest_int("grad_penalty", 3, 15, step=4)
    n_critic = trial.suggest_int("n_critic", 3, 15, step=4)
    z_dim = trial.suggest_int("z_dim", 2, 102, step=10)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, step=10)

    # Construct the command to run main.py with the current hyperparameters
    command = [
        "python",
        "main.py",
        "--generator_type",
        generator_type,
        "--grad_penalty",
        str(grad_penalty),
        "--n_critic",
        str(n_critic),
        "--z_dim",
        str(z_dim),
        "--learning_rate",
        str(learning_rate),
        "--experiment_name", "GaussGAN-tuna",
    ]

    # Run the command and capture the output
    result = subprocess.run(command, capture_output=True, text=True)
    output = result.stdout.strip().split("\n")

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

    max_log_likelihood = run.data.metrics["ValidationStep_FakeData_LogLikelihood"]


    return max_log_likelihood


if __name__ == "__main__":
    args = parse_args()
    study = optuna.create_study(direction="maximize")

    def print_callback(study, trial):
        print(
            f"Trial {trial.number} finished with value: {trial.value} and parameters: {trial.params}"
        )


    study.optimize(
        objective, n_trials=args.n_trials, callbacks=[print_callback]
    )

    print("Best hyperparameters: ", study.best_params)
    print("Best value: ", study.best_value)
