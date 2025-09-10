# 🔄 Automated Run Comparison Tool

## Quick Start

```bash
uv run python compare_latest.py
```

One command to complete all operations!

## Features

✅ **Automatic Discovery of Latest Runs**
- Quantum runs: `quantum_samples`, `quantum_shadows`  
- Classical runs: `classical_normal`, `classical_uniform`

✅ **Complete Metrics Calculation**
- KL Divergence
- Wasserstein Distance
- Maximum Mean Discrepancy (MMD)
- Log Likelihood

✅ **6-Subplot Visualization**
- Training metrics comparison
- Loss function curves
- Professional chart layout

✅ **Data Export**
- High-resolution PNG charts
- Detailed CSV data files

## Output Example

```
🔍 GaussGAN Latest Runs Comparison
==================================================
✅ Found quantum run: quantum_samples (4 epochs)
✅ Found classical run: classical_normal (30 epochs)
✅ Comparison plot saved as: latest_comparison_20250822_235724.png
📄 Detailed data saved as: latest_comparison_data_20250822_235724.csv
```

## Core Features

- **Zero Configuration**: No need to specify run IDs or parameters
- **Intelligent Processing**: Automatically handles different epoch counts
- **Fully Compatible**: Uses the project's built-in metrics.py
- **Fault-Tolerant Design**: Gracefully handles exceptions

It's that simple! 🚀