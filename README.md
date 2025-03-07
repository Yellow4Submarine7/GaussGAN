# GaussGAN Model Training and Testing Guide

This guide provides instructions on how to train and test the GaussGAN model using classical or quantum generators, with or without shadow noise. It explains the command-line arguments, describes how checkpoints are saved with date-time stamps in the `checkpoints` folder, and guides you through running the script in various configurations.


## Table of Contents

- [Setup Instructions](#setup-instructions)
- [Command-Line Arguments](#command-line-arguments)
- [Training the Model](#training-the-model)
- [Examples](#examples)
- [Additional Notes](#additional-notes)

## Setup Instructions

1. **Create conda environment:**
   ```bash
   conda create --name gaussgan python=3.10
   conda activate gaussgan
   ```

2. **Install dependencies:**
   ```bash
   pip install torch lightning mlflow optuna pennylane torch-geometric
   ```

## Command-Line Arguments

The main script accepts several command-line arguments:

- **`--z_dim`**: Dimension of the latent space (default: 3)
- **`--generator_type`**: Type of generator to use:
  - `classical_uniform`: Uniform distribution generator
  - `classical_normal`: Normal distribution generator
  - `quantum_samples`: Quantum circuit generator
  - `quantum_shadows`: Quantum shadows generator
- **`--max_epochs`**: Maximum training epochs (default: 100)
- **`--grad_penalty`**: Gradient penalty factor for WGAN (default: 10)
- **`--n_critic`**: Number of discriminator updates per generator update (default: 5)
- **`--batch_size`**: Batch size for training (default: 16)
- **`--learning_rate`**: Learning rate (default: 0.001)
- **`--dataset_type`**: Type of target distribution (`NORMAL` or `UNIFORM`)
- **`--killer`**: Enable killing one gaussian (default: False)
- **`--validation_samples`**: Number of validation samples (default: 100)

## Training the Model

### Using Classical Generator

```bash
python main.py --generator_type classical_normal --max_epochs 100
```

### Using Quantum Generator

```bash
python main.py --generator_type quantum_samples --max_epochs 100
```

### Using Quantum Shadows

```bash
python main.py --generator_type quantum_shadows --max_epochs 100
```

## Examples

### Training on Gaussian Distribution

```bash
python main.py \
    --generator_type classical_normal \
    --dataset_type NORMAL \
    --max_epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001
```

### Training on Uniform Distribution

```bash
python main.py \
    --generator_type classical_uniform \
    --dataset_type UNIFORM \
    --max_epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001
```

## Additional Notes

- **Checkpoints:**
  - Model checkpoints are saved in the `checkpoints/` directory
  - Filenames include generator type and timestamp
  - Format: `best-checkpoint-{generator_type}-{YYYYMMDD-HHMMSS}.ckpt`

- **Logging:**
  - Training metrics are logged using MLflow
  - Generated samples are saved as CSV files
  - Training visualizations are saved as PNG files

- **Hyperparameter Tuning:**
  - Use `GaussGAN-tuna.py` for automated hyperparameter optimization
  - Explores generator types, gradient penalty, critic updates, and latent dimensions
  - Optimizes for maximum log-likelihood of generated samples

## Performance Metrics

The model tracks several metrics during training:
- Log-likelihood of generated samples
- Discriminator and generator losses
- Position validation for generated points

## Visualization

Training progress can be visualized using:
```bash
python scripts/visualize_training.py
```

This will create plots showing the evolution of generated samples across training epochs.