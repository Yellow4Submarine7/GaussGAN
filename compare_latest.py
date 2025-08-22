#!/usr/bin/env python3
"""
Automatic comparison of latest quantum and classical GAN runs.

This script automatically:
1. Finds the most recent quantum run (quantum_samples or quantum_shadows)
2. Finds the most recent classical run (classical_normal or classical_uniform)
3. Loads training data and computes metrics using source/metrics.py
4. Generates 6-subplot comparison visualization
5. Handles different epoch lengths gracefully

Usage:
    python compare_latest.py
"""

import os
import sys
import glob
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import torch
import yaml

# Add source directory to path for metrics import
sys.path.append('source')
from metrics import LogLikelihood, KLDivergence, MMDivergenceFromGMM, WassersteinDistance

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class RunFinder:
    """Find and organize MLflow runs by type and recency."""
    
    def __init__(self, mlruns_path: str = "mlruns"):
        self.mlruns_path = Path(mlruns_path)
        self.quantum_types = {"quantum_samples", "quantum_shadows"}
        self.classical_types = {"classical_normal", "classical_uniform"}
    
    def _get_run_info(self, run_path: Path) -> Optional[Dict]:
        """Extract run information from MLflow run directory."""
        try:
            # Read generator type
            gen_type_file = run_path / "params" / "generator_type"
            if not gen_type_file.exists():
                return None
            
            with open(gen_type_file, 'r') as f:
                generator_type = f.read().strip()
            
            # Get run creation time from meta.yaml
            meta_file = run_path / "meta.yaml"
            if meta_file.exists():
                import yaml
                try:
                    with open(meta_file, 'r') as f:
                        meta = yaml.safe_load(f)
                    start_time = meta.get('start_time', 0)
                except:
                    start_time = 0
            else:
                # Fallback to directory modification time
                start_time = run_path.stat().st_mtime * 1000  # Convert to milliseconds
            
            # Check if artifacts exist
            artifacts_path = run_path / "artifacts"
            csv_files = list(artifacts_path.glob("gaussian_generated_epoch_*.csv")) if artifacts_path.exists() else []
            
            return {
                "run_id": run_path.name,
                "generator_type": generator_type,
                "start_time": start_time,
                "run_path": run_path,
                "artifacts_path": artifacts_path,
                "csv_files": sorted(csv_files),
                "n_epochs": len(csv_files)
            }
        except Exception as e:
            print(f"Warning: Failed to read run info from {run_path}: {e}")
            return None
    
    def find_runs_by_type(self, run_types: set) -> List[Dict]:
        """Find all runs of specified types, sorted by recency."""
        runs = []
        
        # Search through all experiment directories
        for exp_dir in self.mlruns_path.glob("*/"):
            if exp_dir.name in ['0', 'models'] or not exp_dir.is_dir():
                continue
                
            # Search through all runs in this experiment
            for run_dir in exp_dir.glob("*/"):
                if not run_dir.is_dir():
                    continue
                    
                run_info = self._get_run_info(run_dir)
                if run_info and run_info["generator_type"] in run_types:
                    runs.append(run_info)
        
        # Sort by start time (most recent first)
        runs.sort(key=lambda x: x["start_time"], reverse=True)
        return runs
    
    def find_latest_runs(self) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Find the most recent quantum and classical runs."""
        quantum_runs = self.find_runs_by_type(self.quantum_types)
        classical_runs = self.find_runs_by_type(self.classical_types)
        
        latest_quantum = quantum_runs[0] if quantum_runs else None
        latest_classical = classical_runs[0] if classical_runs else None
        
        return latest_quantum, latest_classical


class MetricsCalculator:
    """Calculate metrics for generated samples using source/metrics.py."""
    
    def __init__(self, target_data_path: str = "data/normal.pickle"):
        """Initialize with target distribution."""
        self.target_data_path = target_data_path
        self.target_data = self._load_target_data()
        self.metrics = self._setup_metrics()
    
    def _load_target_data(self) -> np.ndarray:
        """Load target distribution data."""
        try:
            with open(self.target_data_path, 'rb') as f:
                target_data = pickle.load(f)
            
            # Handle different data formats (torch tensor, dict, list, etc.)
            if isinstance(target_data, dict):
                # If it's a dict, look for common keys
                if 'inputs' in target_data:
                    target_data = target_data['inputs']
                elif 'data' in target_data:
                    target_data = target_data['data']
                else:
                    # Use the first tensor-like value
                    target_data = next(iter(target_data.values()))
            
            # Convert to numpy array
            if torch.is_tensor(target_data):
                target_data = target_data.cpu().numpy()
            else:
                target_data = np.array(target_data)
            
            # Ensure it's 2D
            if target_data.ndim == 1:
                target_data = target_data.reshape(-1, 1)
            
            print(f"Loaded target data: {target_data.shape} samples")
            return target_data
            
        except Exception as e:
            print(f"Error loading target data: {e}")
            # Fallback: create dummy target data (two Gaussian components)
            np.random.seed(42)
            comp1 = np.random.multivariate_normal([-2, -2], [[1, 0], [0, 1]], 500)
            comp2 = np.random.multivariate_normal([2, 2], [[1, 0], [0, 1]], 500)
            return np.vstack([comp1, comp2])
    
    def _setup_metrics(self) -> Dict:
        """Setup metric calculators with target distribution parameters."""
        # For normal.pickle, we assume two Gaussian components
        # These parameters should match the actual target distribution
        centroids = [[-2, -2], [2, 2]]
        cov_matrices = [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
        weights = [0.5, 0.5]
        
        return {
            "LogLikelihood": LogLikelihood(centroids, cov_matrices, weights),
            "KLDivergence": KLDivergence(centroids, cov_matrices, weights),
            "MMD": MMDivergenceFromGMM(centroids, cov_matrices, weights),
            "Wasserstein": WassersteinDistance(self.target_data)
        }
    
    def calculate_metrics_for_samples(self, samples: np.ndarray) -> Dict[str, float]:
        """Calculate all metrics for given samples."""
        results = {}
        
        # Convert to torch tensor for metrics
        import torch
        samples_tensor = torch.from_numpy(samples).float()
        
        for metric_name, metric_calc in self.metrics.items():
            try:
                if metric_name in ["LogLikelihood", "KLDivergence", "MMD"]:
                    # These metrics work with single batches
                    metric_calc.update(samples_tensor)
                    value = float(metric_calc.compute().item())
                    metric_calc.reset()  # Reset for next calculation
                else:
                    # Wasserstein distance returns a single value
                    value = float(metric_calc.compute_score(samples))
                
                results[metric_name] = value
            except Exception as e:
                print(f"Warning: Failed to compute {metric_name}: {e}")
                results[metric_name] = np.nan
        
        return results


class RunComparator:
    """Compare two training runs and generate visualizations."""
    
    def __init__(self):
        self.metrics_calc = MetricsCalculator()
    
    def load_run_data(self, run_info: Dict) -> pd.DataFrame:
        """Load training data from a run and calculate metrics."""
        print(f"Loading data from {run_info['generator_type']} run: {run_info['run_id'][:8]}...")
        
        data_rows = []
        csv_files = run_info["csv_files"]
        
        for csv_file in csv_files:
            try:
                # Extract epoch number from filename
                epoch_str = csv_file.stem.split('_')[-1]  # e.g., "0023" from "gaussian_generated_epoch_0023.csv"
                epoch = int(epoch_str)
                
                # Load generated samples
                samples_df = pd.read_csv(csv_file)
                samples = samples_df[['x', 'y']].values
                
                # Calculate metrics
                metrics = self.metrics_calc.calculate_metrics_for_samples(samples)
                
                # Add epoch and basic info
                row = {
                    'epoch': epoch,
                    'generator_type': run_info['generator_type'],
                    'run_id': run_info['run_id'],
                    'n_samples': len(samples),
                    **metrics
                }
                data_rows.append(row)
                
            except Exception as e:
                print(f"Warning: Failed to process {csv_file}: {e}")
                continue
        
        df = pd.DataFrame(data_rows)
        print(f"Processed {len(df)} epochs for {run_info['generator_type']}")
        return df.sort_values('epoch')
    
    def load_losses_from_mlflow(self, run_info: Dict) -> Dict[str, List[float]]:
        """Load loss data from MLflow metrics."""
        losses = {"generator": [], "discriminator": []}
        
        try:
            metrics_path = run_info["run_path"] / "metrics"
            
            # Try multiple possible generator loss file names
            gen_loss_files = [
                "train_g_loss_epoch",    # New format
                "GeneratorLoss",         # Old format  
                "train_g_loss_step"      # Step-based format
            ]
            
            for gen_loss_name in gen_loss_files:
                gen_loss_file = metrics_path / gen_loss_name
                if gen_loss_file.exists():
                    with open(gen_loss_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                losses["generator"].append(float(parts[1]))
                    break  # Use first found file
            
            # Try multiple possible discriminator loss file names
            disc_loss_files = [
                "train_d_loss_epoch",    # New format
                "DiscriminatorLoss",     # Old format
                "train_d_loss_step"      # Step-based format
            ]
            
            for disc_loss_name in disc_loss_files:
                disc_loss_file = metrics_path / disc_loss_name
                if disc_loss_file.exists():
                    with open(disc_loss_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                losses["discriminator"].append(float(parts[1]))
                    break  # Use first found file
            
        except Exception as e:
            print(f"Warning: Failed to load losses for {run_info['run_id']}: {e}")
        
        return losses
    
    def create_comparison_plot(self, quantum_df: pd.DataFrame, classical_df: pd.DataFrame,
                             quantum_losses: Dict, classical_losses: Dict,
                             quantum_info: Dict, classical_info: Dict) -> plt.Figure:
        """Create 6-subplot comparison visualization."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Quantum vs Classical GAN Comparison\n'
                    f'Quantum: {quantum_info["generator_type"]} | '
                    f'Classical: {classical_info["generator_type"]}', 
                    fontsize=16, fontweight='bold')
        
        # Define colors
        quantum_color = '#FF6B6B'  # Red
        classical_color = '#4ECDC4'  # Teal
        
        # 1. KL Divergence
        ax = axes[0, 0]
        if 'KLDivergence' in quantum_df.columns and 'KLDivergence' in classical_df.columns:
            ax.plot(quantum_df['epoch'], quantum_df['KLDivergence'], 
                   color=quantum_color, linewidth=2, label=f'Quantum ({quantum_info["generator_type"]})', alpha=0.8)
            ax.plot(classical_df['epoch'], classical_df['KLDivergence'], 
                   color=classical_color, linewidth=2, label=f'Classical ({classical_info["generator_type"]})', alpha=0.8)
        ax.set_title('KL Divergence', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('KL Divergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Wasserstein Distance  
        ax = axes[0, 1]
        if 'Wasserstein' in quantum_df.columns and 'Wasserstein' in classical_df.columns:
            ax.plot(quantum_df['epoch'], quantum_df['Wasserstein'], 
                   color=quantum_color, linewidth=2, label=f'Quantum ({quantum_info["generator_type"]})', alpha=0.8)
            ax.plot(classical_df['epoch'], classical_df['Wasserstein'], 
                   color=classical_color, linewidth=2, label=f'Classical ({classical_info["generator_type"]})', alpha=0.8)
        ax.set_title('Wasserstein Distance', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Distance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. MMD 
        ax = axes[0, 2]
        if 'MMD' in quantum_df.columns and 'MMD' in classical_df.columns:
            ax.plot(quantum_df['epoch'], quantum_df['MMD'], 
                   color=quantum_color, linewidth=2, label=f'Quantum ({quantum_info["generator_type"]})', alpha=0.8)
            ax.plot(classical_df['epoch'], classical_df['MMD'], 
                   color=classical_color, linewidth=2, label=f'Classical ({classical_info["generator_type"]})', alpha=0.8)
        ax.set_title('Maximum Mean Discrepancy (MMD)', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MMD')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Generator Loss
        ax = axes[1, 0]
        if quantum_losses["generator"] and classical_losses["generator"]:
            q_epochs = range(len(quantum_losses["generator"]))
            c_epochs = range(len(classical_losses["generator"]))
            ax.plot(q_epochs, quantum_losses["generator"], 
                   color=quantum_color, linewidth=2, label=f'Quantum ({quantum_info["generator_type"]})', alpha=0.8)
            ax.plot(c_epochs, classical_losses["generator"], 
                   color=classical_color, linewidth=2, label=f'Classical ({classical_info["generator_type"]})', alpha=0.8)
        ax.set_title('Generator Loss', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Discriminator Loss
        ax = axes[1, 1] 
        if quantum_losses["discriminator"] and classical_losses["discriminator"]:
            q_epochs = range(len(quantum_losses["discriminator"]))
            c_epochs = range(len(classical_losses["discriminator"]))
            ax.plot(q_epochs, quantum_losses["discriminator"], 
                   color=quantum_color, linewidth=2, label=f'Quantum ({quantum_info["generator_type"]})', alpha=0.8)
            ax.plot(c_epochs, classical_losses["discriminator"], 
                   color=classical_color, linewidth=2, label=f'Classical ({classical_info["generator_type"]})', alpha=0.8)
        ax.set_title('Discriminator Loss', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Log Likelihood
        ax = axes[1, 2]
        if 'LogLikelihood' in quantum_df.columns and 'LogLikelihood' in classical_df.columns:
            ax.plot(quantum_df['epoch'], quantum_df['LogLikelihood'], 
                   color=quantum_color, linewidth=2, label=f'Quantum ({quantum_info["generator_type"]})', alpha=0.8)
            ax.plot(classical_df['epoch'], classical_df['LogLikelihood'], 
                   color=classical_color, linewidth=2, label=f'Classical ({classical_info["generator_type"]})', alpha=0.8)
        ax.set_title('Log Likelihood', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Log Likelihood')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def main():
    """Main function to run the comparison."""
    print("🔍 GaussGAN Latest Runs Comparison")
    print("=" * 50)
    
    # Find latest runs
    print("1. Finding latest quantum and classical runs...")
    finder = RunFinder()
    quantum_info, classical_info = finder.find_latest_runs()
    
    if not quantum_info:
        print("❌ No quantum runs found!")
        return
    
    if not classical_info:
        print("❌ No classical runs found!")
        return
    
    print(f"✅ Found quantum run: {quantum_info['generator_type']} ({quantum_info['n_epochs']} epochs)")
    print(f"   Run ID: {quantum_info['run_id']}")
    print(f"✅ Found classical run: {classical_info['generator_type']} ({classical_info['n_epochs']} epochs)")
    print(f"   Run ID: {classical_info['run_id']}")
    print()
    
    # Load and process data
    print("2. Loading training data and computing metrics...")
    comparator = RunComparator()
    
    try:
        quantum_df = comparator.load_run_data(quantum_info)
        classical_df = comparator.load_run_data(classical_info)
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return
    
    # Load loss data
    print("3. Loading loss data from MLflow...")
    quantum_losses = comparator.load_losses_from_mlflow(quantum_info)
    classical_losses = comparator.load_losses_from_mlflow(classical_info)
    
    # Debug: Print loss data info
    print(f"   Quantum losses: Generator={len(quantum_losses['generator'])} points, Discriminator={len(quantum_losses['discriminator'])} points")
    print(f"   Classical losses: Generator={len(classical_losses['generator'])} points, Discriminator={len(classical_losses['discriminator'])} points")
    
    # Create comparison plot
    print("4. Generating comparison visualization...")
    try:
        fig = comparator.create_comparison_plot(
            quantum_df, classical_df,
            quantum_losses, classical_losses, 
            quantum_info, classical_info
        )
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"latest_comparison_{timestamp}.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Comparison plot saved as: {output_file}")
        
        # Show statistics
        print("\n📊 Summary Statistics:")
        print("-" * 30)
        print("Quantum Run:")
        print(f"  - Type: {quantum_info['generator_type']}")
        print(f"  - Epochs: {len(quantum_df)}")
        if 'KLDivergence' in quantum_df.columns:
            final_kl = quantum_df['KLDivergence'].iloc[-1]
            print(f"  - Final KL Divergence: {final_kl:.4f}")
        
        print("\nClassical Run:")
        print(f"  - Type: {classical_info['generator_type']}")
        print(f"  - Epochs: {len(classical_df)}")
        if 'KLDivergence' in classical_df.columns:
            final_kl = classical_df['KLDivergence'].iloc[-1]
            print(f"  - Final KL Divergence: {final_kl:.4f}")
        
        # Save data to CSV for further analysis
        output_csv = f"latest_comparison_data_{timestamp}.csv"
        quantum_df['run_type'] = 'quantum'
        classical_df['run_type'] = 'classical'
        combined_df = pd.concat([quantum_df, classical_df], ignore_index=True)
        combined_df.to_csv(output_csv, index=False)
        print(f"📄 Detailed data saved as: {output_csv}")
        
        # Don't call plt.show() to avoid display issues in headless environments
        # plt.show()
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()