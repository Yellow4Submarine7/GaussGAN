# GaussGAN: Quantum-Classical Hybrid Generative Adversarial Network

A Wasserstein GAN with quantum circuit generators for 2D distribution generation. This research project explores the integration of classical and quantum machine learning approaches for generative modeling.


## Features

- **Multiple Generator Types**: 
  - Classical generators with Gaussian parameterization (uniform/normal)
  - Direct coordinate generators without distributional assumptions (5 architectures)
  - Quantum circuit generators with PennyLane integration
  - Quantum shadow noise with exponential measurement efficiency
- **WGAN-GP Architecture**: Wasserstein GAN with gradient penalty for stable training
- **Distribution-Agnostic Generation**: New direct generators that output coordinates without mean+std assumptions
- **Value Network "Killer"**: Reinforcement learning to shape distributions
- **Comprehensive Metrics**: Log-likelihood, KL divergence, Wasserstein distance, MMD, convergence tracking
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
# Train with classical generator (Gaussian parameterization)
python main.py --generator_type classical_normal --max_epochs 50

# Train with direct coordinate generator (no distributional assumptions)
python main.py --generator_type direct_stable --max_epochs 50

# Train with quantum circuit generator
python main.py --generator_type quantum_samples --max_epochs 50

# Train with quantum shadows
python main.py --generator_type quantum_shadows --max_epochs 50

# Test direct generator implementations
python test_direct_generator.py
```

## Command-Line Arguments

The main script accepts several command-line arguments:

### Generator Configuration
- **`--generator_type`**: Type of generator to use:
  - **Classical (with Gaussian parameterization)**:
    - `classical_uniform`: Uniform distribution generator
    - `classical_normal`: Normal distribution generator (default)
  - **Direct Coordinate Generators (no distributional assumptions)**:
    - `direct_bounded`: Simple bounded output generator
    - `direct_residual`: Generator with residual connections
    - `direct_progressive`: Progressive complexity generator
    - `direct_stable`: Combined stability techniques (recommended)
    - `direct_gradient`: Gradient-controlled generator
  - **Quantum Generators**:
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

### Using Classical Generator (Gaussian Parameterization)

```bash
python main.py --generator_type classical_normal --max_epochs 50
```

### Using Direct Coordinate Generator (Distribution-Agnostic)

```bash
# Recommended: stable direct generator with multiple stability techniques
python main.py --generator_type direct_stable --max_epochs 50

# Alternative architectures
python main.py --generator_type direct_bounded --max_epochs 50    # Simple bounded
python main.py --generator_type direct_residual --max_epochs 50   # With residual connections
python main.py --generator_type direct_progressive --max_epochs 50 # Progressive complexity
python main.py --generator_type direct_gradient --max_epochs 50   # Gradient controlled
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
python main.py --generator_type direct_stable --killer true --max_epochs 50
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

### Comparing Direct vs Parameterized Generators

```bash
# Traditional approach with Gaussian parameterization
python main.py --generator_type classical_normal --max_epochs 50

# Direct approach without distributional assumptions
python main.py --generator_type direct_stable --max_epochs 50

# Test both approaches
python test_direct_generator.py
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
├── main.py                     # Main training script
├── config.yaml                # Configuration file
├── requirements.txt           # Python dependencies
├── test_direct_generator.py   # Test script for direct generators
├── source/
│   ├── model.py              # WGAN-GP model implementation
│   ├── nn.py                 # Classical and quantum generators
│   ├── direct_generators.py  # Direct coordinate generators (no Gaussian assumption)
│   ├── data.py               # Data loading and processing
│   ├── metrics.py            # Evaluation metrics
│   └── convergence.py        # Convergence tracking and analysis
├── scripts/
│   ├── visualize_training.py # Training visualization
│   └── visualize_latest.py   # Latest results visualization
└── data/
    ├── normal.pickle         # Gaussian mixture target
    └── uniform.pickle        # Uniform distribution target
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
- Validation frequency set to once per epoch for consistent visualization
- Direct generators use multiple stability techniques (bounded outputs, residual connections, spectral normalization)

## Research Context

This project explores:
- **Distribution-agnostic generation**: Direct coordinate generators without Gaussian assumptions
- **Quantum circuit expressivity**: Quantum generators for non-classical distributions
- **Shadow tomography**: Efficient quantum measurements with exponential advantages
- **Hybrid architectures**: Integration of quantum and classical components
- **Comparative analysis**: Classical (parameterized vs direct) and quantum generators
- **Training stability**: Multiple architectural techniques for stable GAN training

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