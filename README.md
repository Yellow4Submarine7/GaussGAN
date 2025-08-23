# GaussGAN: Quantum-Classical Hybrid Generative Adversarial Network

A Wasserstein GAN with quantum circuit generators for 2D distribution generation. This research project explores the integration of classical and quantum machine learning approaches for generative modeling.


## Features

- **Multiple Generator Types**: Classical (uniform/normal), quantum circuits, quantum shadow noise
- **WGAN-GP Architecture**: Wasserstein GAN with gradient penalty for stable training
- **Quantum Integration**: PennyLane quantum circuits with PyTorch autodiff
- **Value Network "Killer"**: Reinforcement learning to shape distributions
- **Comprehensive Metrics**: Log-likelihood, KL divergence, Wasserstein distance, MMD
- **Hyperparameter Optimization**: Automated tuning with Optuna

## Table of Contents

- [Setup Instructions](#setup-instructions)
- [Quick Start](#quick-start)
- [Command-Line Arguments](#command-line-arguments)
- [Training the Model](#training-the-model)
- [Examples](#examples)
- [Project Structure](#project-structure)
- [Additional Notes](#additional-notes)

## Setup Instructions

### Option 1: Using pip (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Yellow4Submarine7/GaussGAN.git
   cd GaussGAN
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using uv (Faster)

1. **Install uv:**
   ```bash
   pip install uv
   ```

2. **Create environment and install dependencies:**
   ```bash
   uv venv .venv --python 3.10
   uv pip install -r requirements.txt
   ```

3. **Run commands with uv:**
   ```bash
   uv run python main.py --generator_type classical_normal --max_epochs 50
   ```

## Quick Start

```bash
# Train with classical generator
python main.py --generator_type classical_normal --max_epochs 50

# Train with quantum circuit generator
python main.py --generator_type quantum_samples --max_epochs 50

# Train with quantum shadows
python main.py --generator_type quantum_shadows --max_epochs 50
```

## Command-Line Arguments

The main script accepts several command-line arguments:

### Generator Configuration
- **`--generator_type`**: Type of generator to use:
  - `classical_uniform`: Uniform distribution generator
  - `classical_normal`: Normal distribution generator (default)
  - `quantum_samples`: Quantum circuit generator with parameterized gates
  - `quantum_shadows`: Quantum shadows with exponential measurement efficiency
- **`--z_dim`**: Dimension of the latent space (default: 4)

### Training Parameters
- **`--max_epochs`**: Maximum training epochs (default: 50)
- **`--batch_size`**: Batch size for training (default: 256)
- **`--learning_rate`**: Learning rate (default: 0.001)
- **`--grad_penalty`**: Gradient penalty factor for WGAN-GP (default: 0.2)
- **`--n_critic`**: Number of discriminator updates per generator update (default: 5)

### Dataset and Features
- **`--dataset_type`**: Type of target distribution (`NORMAL` or `UNIFORM`)
- **`--killer`**: Enable value network to shape distributions (default: false)
- **`--validation_samples`**: Number of validation samples (default: 500)

### Quantum Circuit Parameters
- **`--quantum_qubits`**: Number of qubits (default: 6)
- **`--quantum_layers`**: Number of quantum layers (default: 2)
- **`--quantum_shots`**: Number of measurement shots (default: 100)

## Training the Model

### Using Classical Generator

```bash
python main.py --generator_type classical_normal --max_epochs 50
```

### Using Quantum Generator

```bash
python main.py --generator_type quantum_samples --max_epochs 50
```

### Using Quantum Shadows

```bash
python main.py --generator_type quantum_shadows --max_epochs 50
```

### With Killer Mode (Distribution Shaping)

```bash
python main.py --generator_type classical_normal --killer true --max_epochs 50
```

## Examples

### Training on Gaussian Mixture (Default)

```bash
python main.py \
    --generator_type classical_normal \
    --dataset_type NORMAL \
    --max_epochs 50 \
    --batch_size 256 \
    --learning_rate 0.001
```

### Quantum Circuit with Custom Parameters

```bash
python main.py \
    --generator_type quantum_samples \
    --quantum_qubits 8 \
    --quantum_layers 3 \
    --quantum_shots 200 \
    --max_epochs 50
```

### Hyperparameter Optimization

```bash
python GaussGAN-tuna.py
```

## Project Structure

```
GaussGAN/
├── main.py                 # Main training script
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
├── source/
│   ├── model.py          # WGAN-GP model implementation
│   ├── nn.py             # Generator and discriminator networks
│   ├── data.py           # Data loading and processing
│   └── metrics.py        # Evaluation metrics
├── scripts/
│   ├── visualize_training.py  # Training visualization
│   └── visualize_latest.py    # Latest results visualization
└── data/
    ├── normal.pickle     # Gaussian mixture target
    └── uniform.pickle    # Uniform distribution target
```

## Additional Notes

### Checkpoints
- Model checkpoints are saved in the `checkpoints/` directory
- Filenames include generator type and timestamp
- Format: `best-checkpoint-{generator_type}-{YYYYMMDD-HHMMSS}.ckpt`

### Logging and Metrics
- Training metrics are logged using MLflow
- Tracked metrics include:
  - **LogLikelihood**: GMM-based likelihood estimation
  - **KLDivergence**: KL(Q||P) using KDE
  - **WassersteinDistance**: Earth mover's distance
  - **MMDDistance**: Maximum mean discrepancy
  - **IsPositive**: Distribution position validation

### Visualization

```bash
# Visualize training progress
python scripts/visualize_training.py

# Visualize latest results
python scripts/visualize_latest.py

# Visualize dataset
python scripts/visualize_data.py
```

### Performance Optimizations
- Tensor Core optimization with `torch.set_float32_matmul_precision('medium')`
- Reduced quantum parameters for faster training (6 qubits, 2 layers, 100 shots)
- WGAN-GP for stable training without mode collapse

## Research Context

This project explores:
- Quantum circuit expressivity in generative modeling
- Shadow tomography for efficient quantum measurements
- Integration of quantum and classical components in hybrid models
- Comparison of quantum vs classical generators for 2D distributions

## License

This project is open source. Contributions are welcome!

## Citation

If you use this code in your research, please cite:
```
@software{gaussgan2024,
  title = {GaussGAN: Quantum-Classical Hybrid Generative Adversarial Network},
  author = {Yellow4Submarine7},
  year = {2024},
  url = {https://github.com/Yellow4Submarine7/GaussGAN}
}
```