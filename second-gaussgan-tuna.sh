
#!/bin/bash

# First, create the MLflow experiment if it doesn't exist
python - <<EOF
import mlflow
from mlflow.exceptions import MlflowException

def setup_experiment(experiment_name):
    try:
        # Try to get existing experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            # Create new experiment if it doesn't exist
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        
        # Set as active experiment
        mlflow.set_experiment(experiment_name)
        print(f"Using experiment '{experiment_name}' with ID: {experiment_id}")
    except MlflowException as e:
        print(f"Error setting up experiment: {e}")
        exit(1)

setup_experiment("GaussGAN-Search")
EOF

LEARNING_RATES=(0.001 0.01 0.1)
SEED=(0 1 2 3)
NCRITIC=(2 4 6 8)
NN_GEN=("[16,16]" "[32,32]")
NN_DISC=("[16,16]" "[32,32]")
Z_DIM=(9)

# Create a function to run the python script
run_experiment() {
    local lr=$1
    local seed=$2
    local nc=$3
    local nn_gen=$4
    local nn_disc=$5
    local z_dim=$6

    python main.py \
        --max_epochs 400 \
        --learning_rate "$lr" \
        --seed "$seed" \
        --n_critic "$nc" \
        --nn_gen "$nn_gen" \
        --nn_disc "$nn_disc" \
        --experiment_name "GaussGAN-Search" \
        --z_dim "$z_dim"
}

export -f run_experiment

# Generate the combinations and run them in parallel
parallel -j 7 run_experiment ::: "${LEARNING_RATES[@]}" ::: "${SEED[@]}" ::: "${NCRITIC[@]}" ::: "${NN_GEN[@]}" ::: "${NN_DISC[@]}"  ::: "${Z_DIM[@]}"
