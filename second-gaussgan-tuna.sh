#!/bin/bash

# First, create the MLflow experiment if it doesn't exist
python - <<'EOF'
import mlflow
from mlflow.exceptions import MlflowException

def setup_experiment(experiment_name):
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        mlflow.set_experiment(experiment_name)
        print(f"Using experiment '{experiment_name}' with ID: {experiment_id}")
    except MlflowException as e:
        print(f"Error setting up experiment: {e}")
        exit(1)

setup_experiment("Big-GaussGAN-Search")
EOF

# Define arrays with parameter combinations
# LEARNING_RATES=(0.01 0.001 0.0001)
# SEED=(0 1 2 3 4 5 6)
# NCRITIC=(3 4 5 6 8)
# NN_GEN=("[32,32]" "[32,64]" "[64,64]" "[128,128]" "[256,256]")
# NN_DISC=("[32,32]" "[32,64]" "[64,64]" "[128,128]" "[256,256]")
# Z_DIM=(3 5 10 12)
# NON_LINEARITY=("ReLU" "LeakyReLU" "Tanh" "Sigmoid")
# GRAD_PENALTIES=(5 10 15)
LEARNING_RATES=(0.001)        # 0.001 0.0001)
SEED=(0 1 2 3 4 5 6)
NCRITIC=(3 5) # 6 8)
NN_GEN=("[32,32]" "[32,64]" "[64,64]" "[128,128]" "[256,256]")
NN_DISC=("[32,32]" "[32,64]" "[64,64]" "[128,128]" "[256,256]")
Z_DIM=(3 10) # 12)
NON_LINEARITY=("ReLU") # "LeakyReLU" "Tanh" "Sigmoid")
GRAD_PENALTIES=(10) # 10 15)





# Create a function to run the python script
run_experiment() {
    local lr=$1
    local seed=$2
    local nc=$3
    local nn_gen=$4
    local nn_disc=$5
    local z_dim=$6
    local non_linearity=$7
    local gp=$8

    python main.py \
        --max_epochs 150 \
        --seed "$seed" \
        --n_critic "$nc" \
        --nn_gen "$nn_gen" \
        --nn_disc "$nn_disc" \
        --experiment_name "Big-GaussGAN-Search" \
        --z_dim "$z_dim" \
        --learning_rate "$lr" \
        --non_linearity "$non_linearity" \
        --grad_penalty "$gp" \
        --accelerator cuda
}

export -f run_experiment

# Generate the combinations and run them in parallel
parallel -j 25 run_experiment ::: "${LEARNING_RATES[@]}" ::: "${SEED[@]}" ::: "${NCRITIC[@]}" ::: "${NN_GEN[@]}" ::: "${NN_DISC[@]}" ::: "${Z_DIM[@]}" ::: "${NON_LINEARITY[@]}"  ::: "${GRAD_PENALTIES[@]}"