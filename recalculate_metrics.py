#!/usr/bin/env python3
"""
Post-processing script to recalculate metrics from saved CSV files.

This script reads the gaussian_generated_epoch_*.csv files from MLflow artifacts
and recalculates all metrics using the existing metrics implementation from source/metrics.py.

Advantages:
- Faster training (no complex metrics calculation during training)
- Can add new metrics without retraining
- Can recalculate with different parameters
- Robust to training interruptions
"""

import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import yaml

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import mlflow

from source.metrics import (
    LogLikelihood, KLDivergence, MMDivergenceFromGMM,
    WassersteinDistance, MMDDistance, IsPositive
)


class MetricsRecalculator:
    """Recalculates metrics for saved generated samples."""
    
    def __init__(self, target_data_path: str = "data/normal.pickle"):
        """
        Initialize the metrics recalculator.
        
        Args:
            target_data_path: Path to the target distribution pickle file
        """
        self.target_data_path = target_data_path
        self.target_data = None
        self.gaussians = None
        self.metrics_instances = {}
        
        # Load target distribution
        self._load_target_distribution()
        
    def _load_target_distribution(self):
        """Load the target distribution from pickle file."""
        try:
            with open(self.target_data_path, 'rb') as f:
                data = pickle.load(f)
                
            self.target_data = data['inputs']
            if torch.is_tensor(self.target_data):
                self.target_data = self.target_data.numpy()
                
            # Extract Gaussian parameters
            self.gaussians = {
                'centroids': [data['mean1'].numpy(), data['mean2'].numpy()],
                'cov_matrices': [data['cov1'].numpy(), data['cov2'].numpy()],
                'weights': np.array([0.5, 0.5])  # Equal weights for the two Gaussians
            }
            
            print(f"✅ Loaded target distribution with {len(self.target_data)} samples")
            print(f"   - Mean 1: {self.gaussians['centroids'][0]}")
            print(f"   - Mean 2: {self.gaussians['centroids'][1]}")
            
        except Exception as e:
            raise ValueError(f"Failed to load target distribution: {e}")
    
    def _initialize_metrics(self, selected_metrics: List[str] = None, 
                           fast_mode: bool = False) -> Dict:
        """
        Initialize metrics with target distribution parameters.
        
        Args:
            selected_metrics: List of metric names to calculate. If None, calculates all.
            fast_mode: If True, uses faster settings and skips heavy computations
        """
        all_metrics = {
            'IsPositive': lambda: IsPositive(),
            'LogLikelihood': lambda: LogLikelihood(
                centroids=self.gaussians['centroids'],
                cov_matrices=self.gaussians['cov_matrices'],
                weights=self.gaussians['weights']
            ),
            'WassersteinDistance': lambda: WassersteinDistance(
                target_samples=self.target_data[:5000] if fast_mode else self.target_data,
                aggregation="mean"
            ),
            'MMDDistance': lambda: MMDDistance(
                target_samples=self.target_data[:2000] if fast_mode else self.target_data[:5000],
                gamma=1.0
            ),
            'MMDivergenceFromGMM': lambda: MMDivergenceFromGMM(
                centroids=self.gaussians['centroids'],
                cov_matrices=self.gaussians['cov_matrices'],
                weights=self.gaussians['weights'],
                n_target_samples=500 if fast_mode else 1000
            )
        }
        
        # KL Divergence is expensive, add only if not in fast mode or explicitly requested
        if not fast_mode or (selected_metrics and 'KLDivergence' in selected_metrics):
            all_metrics['KLDivergence'] = lambda: KLDivergence(
                centroids=self.gaussians['centroids'],
                cov_matrices=self.gaussians['cov_matrices'],
                weights=self.gaussians['weights']
            )
        
        # Select which metrics to initialize
        if selected_metrics:
            available_metrics = set(all_metrics.keys())
            requested_metrics = set(selected_metrics)
            invalid_metrics = requested_metrics - available_metrics
            
            if invalid_metrics:
                print(f"⚠️  Warning: Unknown metrics: {invalid_metrics}")
            
            selected_metrics = list(requested_metrics & available_metrics)
        else:
            selected_metrics = list(all_metrics.keys())
        
        metrics = {}
        
        try:
            for metric_name in selected_metrics:
                metrics[metric_name] = all_metrics[metric_name]()
                
            print(f"✅ Initialized {len(metrics)} metrics: {list(metrics.keys())}")
            if fast_mode:
                print("🚀 Running in fast mode with reduced sample sizes")
            
        except Exception as e:
            print(f"❌ Error initializing metrics: {e}")
            
        return metrics
    
    def _load_csv_samples(self, csv_path: str) -> np.ndarray:
        """Load generated samples from CSV file."""
        try:
            df = pd.read_csv(csv_path)
            if 'x' in df.columns and 'y' in df.columns:
                samples = df[['x', 'y']].values
            else:
                # Assume first two columns are coordinates
                samples = df.iloc[:, :2].values
                
            # Filter out invalid samples
            valid_mask = ~(np.isnan(samples).any(axis=1) | np.isinf(samples).any(axis=1))
            samples = samples[valid_mask]
            
            return samples
            
        except Exception as e:
            warnings.warn(f"Failed to load CSV {csv_path}: {e}")
            return np.array([]).reshape(0, 2)
    
    def _calculate_metric_score(self, metric, samples: np.ndarray) -> float:
        """Calculate metric score for given samples."""
        try:
            if len(samples) == 0:
                return float('nan')
                
            # Convert to torch tensor for compatibility
            samples_tensor = torch.from_numpy(samples).float()
            
            # Reset metric state
            metric.reset()
            
            # Update metric with samples
            metric.update(samples_tensor)
            
            # Compute final score
            score = metric.compute()
            
            if torch.is_tensor(score):
                score = score.item()
                
            return float(score)
            
        except Exception as e:
            warnings.warn(f"Error calculating metric {type(metric).__name__}: {e}")
            return float('nan')
    
    def process_experiment(self, experiment_path: str, 
                          output_file: Optional[str] = None,
                          selected_metrics: List[str] = None,
                          fast_mode: bool = False) -> pd.DataFrame:
        """
        Process a single experiment (run) and calculate metrics for all epochs.
        
        Args:
            experiment_path: Path to the MLflow run directory containing artifacts/
            output_file: Optional path to save results CSV
            selected_metrics: List of metrics to calculate. If None, calculates all.
            fast_mode: If True, uses faster settings
            
        Returns:
            DataFrame with metrics for each epoch
        """
        artifacts_path = Path(experiment_path) / "artifacts"
        
        if not artifacts_path.exists():
            raise ValueError(f"Artifacts directory not found: {artifacts_path}")
            
        # Find all CSV files
        csv_files = sorted(list(artifacts_path.glob("gaussian_generated_epoch_*.csv")))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {artifacts_path}")
            
        print(f"📊 Processing {len(csv_files)} epochs from {experiment_path}")
        
        # Initialize metrics
        metrics = self._initialize_metrics(selected_metrics, fast_mode)
        
        results = []
        
        # Process each epoch
        for csv_file in tqdm(csv_files, desc="Calculating metrics"):
            # Extract epoch number from filename
            epoch_str = csv_file.stem.split('_')[-1]  # Gets '0000' from 'epoch_0000'
            epoch = int(epoch_str)
            
            # Load samples
            samples = self._load_csv_samples(str(csv_file))
            
            if len(samples) == 0:
                print(f"⚠️  Warning: No valid samples in epoch {epoch}")
                continue
                
            # Calculate all metrics
            epoch_metrics = {
                'epoch': epoch,
                'n_samples': len(samples),
                'samples_file': csv_file.name
            }
            
            for metric_name, metric_instance in metrics.items():
                score = self._calculate_metric_score(metric_instance, samples)
                epoch_metrics[metric_name] = score
                
            results.append(epoch_metrics)
            
        # Convert to DataFrame
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('epoch').reset_index(drop=True)
        
        # Save results if output file specified
        if output_file:
            df_results.to_csv(output_file, index=False)
            print(f"💾 Saved results to {output_file}")
            
        return df_results
    
    def process_all_experiments(self, mlruns_dir: str = "mlruns", 
                              output_dir: str = "docs/recalculated_metrics") -> Dict[str, pd.DataFrame]:
        """
        Process all experiments in the MLflow directory.
        
        Args:
            mlruns_dir: MLflow runs directory
            output_dir: Directory to save results
            
        Returns:
            Dictionary mapping run_id -> results DataFrame
        """
        mlruns_path = Path(mlruns_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        
        # Find all experiment directories
        for experiment_dir in mlruns_path.iterdir():
            if not experiment_dir.is_dir() or experiment_dir.name == "0":
                continue
                
            # Find all run directories within each experiment
            for run_dir in experiment_dir.iterdir():
                if not run_dir.is_dir() or run_dir.name == "meta.yaml":
                    continue
                    
                run_id = run_dir.name
                artifacts_dir = run_dir / "artifacts"
                
                # Check if this run has CSV files
                if not artifacts_dir.exists():
                    continue
                    
                csv_files = list(artifacts_dir.glob("gaussian_generated_epoch_*.csv"))
                if not csv_files:
                    continue
                    
                print(f"\n🔄 Processing run: {run_id}")
                
                try:
                    # Process this run
                    output_file = output_path / f"metrics_{run_id}.csv"
                    results_df = self.process_experiment(str(run_dir), str(output_file))
                    all_results[run_id] = results_df
                    
                    print(f"✅ Processed {len(results_df)} epochs for run {run_id}")
                    
                except Exception as e:
                    print(f"❌ Error processing run {run_id}: {e}")
                    
        return all_results
    
    def generate_summary_report(self, all_results: Dict[str, pd.DataFrame], 
                              output_file: str = "docs/metrics_summary.csv"):
        """Generate a summary report across all runs."""
        summary_data = []
        
        for run_id, df in all_results.items():
            if df.empty:
                continue
                
            # Calculate summary statistics for each metric
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            metric_cols = [col for col in numeric_cols if col not in ['epoch', 'n_samples']]
            
            summary = {
                'run_id': run_id,
                'total_epochs': len(df),
                'avg_samples_per_epoch': df['n_samples'].mean(),
            }
            
            for metric in metric_cols:
                valid_values = df[metric].dropna()
                if len(valid_values) > 0:
                    summary.update({
                        f'{metric}_final': df[metric].iloc[-1] if not pd.isna(df[metric].iloc[-1]) else float('nan'),
                        f'{metric}_best': valid_values.min() if metric in ['KLDivergence', 'WassersteinDistance', 'MMDDistance'] else valid_values.max(),
                        f'{metric}_mean': valid_values.mean(),
                        f'{metric}_std': valid_values.std(),
                    })
                    
            summary_data.append(summary)
            
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(output_file, index=False)
            print(f"📋 Generated summary report: {output_file}")
            
            return summary_df
        else:
            print("❌ No valid results to summarize")
            return pd.DataFrame()


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="Recalculate metrics from saved CSV files")
    parser.add_argument(
        "--run_path", "-r", 
        type=str, 
        help="Path to specific MLflow run directory"
    )
    parser.add_argument(
        "--mlruns_dir", "-m", 
        type=str, 
        default="mlruns",
        help="MLflow runs directory (default: mlruns)"
    )
    parser.add_argument(
        "--output_dir", "-o", 
        type=str,
        default="docs/recalculated_metrics",
        help="Output directory for results (default: docs/recalculated_metrics)"
    )
    parser.add_argument(
        "--target_data", "-t",
        type=str,
        default="data/normal.pickle",
        help="Path to target distribution data (default: data/normal.pickle)"
    )
    parser.add_argument(
        "--summary_only", "-s",
        action="store_true",
        help="Only generate summary report (skip individual processing)"
    )
    parser.add_argument(
        "--metrics", 
        type=str, 
        nargs="*",
        help="Specific metrics to calculate (e.g., --metrics IsPositive LogLikelihood). If not specified, calculates all available metrics."
    )
    parser.add_argument(
        "--fast", "-f",
        action="store_true",
        help="Fast mode: use reduced sample sizes and skip expensive computations"
    )
    
    args = parser.parse_args()
    
    # Initialize recalculator
    recalculator = MetricsRecalculator(target_data_path=args.target_data)
    
    if args.run_path:
        # Process single run
        print(f"🎯 Processing single run: {args.run_path}")
        output_file = Path(args.output_dir) / f"metrics_{Path(args.run_path).name}.csv"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        results_df = recalculator.process_experiment(
            args.run_path, 
            str(output_file),
            selected_metrics=args.metrics,
            fast_mode=args.fast
        )
        
        print(f"\n📈 Results summary for run {Path(args.run_path).name}:")
        print(f"   - Total epochs: {len(results_df)}")
        print(f"   - Average samples per epoch: {results_df['n_samples'].mean():.1f}")
        
        # Show final metrics
        final_row = results_df.iloc[-1]
        for col in results_df.columns:
            if col not in ['epoch', 'n_samples', 'samples_file']:
                value = final_row[col]
                if not pd.isna(value):
                    print(f"   - Final {col}: {value:.4f}")
                    
    else:
        # Process all runs
        if not args.summary_only:
            print(f"🎯 Processing all runs in {args.mlruns_dir}")
            all_results = recalculator.process_all_experiments(args.mlruns_dir, args.output_dir)
        else:
            # Load existing results for summary
            output_path = Path(args.output_dir)
            if not output_path.exists():
                print("❌ Output directory doesn't exist. Run without --summary_only first.")
                return
                
            all_results = {}
            for csv_file in output_path.glob("metrics_*.csv"):
                run_id = csv_file.stem.replace("metrics_", "")
                all_results[run_id] = pd.read_csv(csv_file)
        
        # Generate summary report
        if all_results:
            summary_file = Path(args.output_dir) / "summary_report.csv"
            summary_df = recalculator.generate_summary_report(all_results, str(summary_file))
            
            print(f"\n📊 Summary across {len(all_results)} runs:")
            print(f"   - Total runs processed: {len(summary_df)}")
            if not summary_df.empty:
                print(f"   - Average epochs per run: {summary_df['total_epochs'].mean():.1f}")
        else:
            print("❌ No results found to process")


if __name__ == "__main__":
    main()