"""
Multi-Run Experiment Runner for Statistical Analysis
==================================================

This module provides a practical implementation for running multiple GaussGAN 
experiments systematically for statistical comparison between quantum and 
classical generators.

Key Features:
- Integration with existing GaussGAN training infrastructure
- Automated experiment execution with different seeds and configurations
- Results collection and storage for statistical analysis
- Progress monitoring and error handling
- Configurable experiment parameters and resource management
"""

import os
import sys
import subprocess
import yaml
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import concurrent.futures
import threading
import queue
import psutil

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import MLflow for experiment tracking
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not available. Experiment tracking will be limited.")


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    run_id: str
    generator_type: str
    seed: int
    max_epochs: int
    batch_size: int
    learning_rate: float
    killer: bool = False
    quantum_qubits: int = 6
    quantum_layers: int = 2
    quantum_shots: int = 100
    additional_args: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_args is None:
            self.additional_args = {}


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    config: ExperimentConfig
    success: bool
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    final_metrics: Dict[str, float] = None
    convergence_epoch: Optional[int] = None
    error_message: str = ""
    checkpoint_path: str = ""
    mlflow_run_id: str = ""
    
    def __post_init__(self):
        if self.final_metrics is None:
            self.final_metrics = {}


class ExperimentRunner:
    """
    Manages execution of multiple GaussGAN experiments for statistical analysis.
    
    This class coordinates the execution of multiple training runs with different
    configurations, collecting results for comprehensive statistical comparison.
    """
    
    def __init__(
        self,
        base_config_path: str = "config.yaml",
        output_dir: str = "docs/statistical_analysis/experiments",
        max_parallel: int = 1,
        gpu_memory_limit: float = 0.8
    ):
        """
        Initialize experiment runner.
        
        Args:
            base_config_path: Path to base configuration file
            output_dir: Directory to store experiment results
            max_parallel: Maximum number of parallel experiments
            gpu_memory_limit: GPU memory usage limit (0.0-1.0)
        """
        self.base_config_path = Path(base_config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_parallel = max_parallel
        self.gpu_memory_limit = gpu_memory_limit
        
        # Load base configuration
        with open(self.base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        # Setup logging
        log_file = self.output_dir / 'experiment_runner.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.results: List[ExperimentResult] = []
        self.results_file = self.output_dir / 'experiment_results.json'
        self.progress_file = self.output_dir / 'experiment_progress.json'
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        
        # MLflow setup
        if MLFLOW_AVAILABLE:
            self.setup_mlflow()
    
    def setup_mlflow(self):
        """Setup MLflow for experiment tracking."""
        try:
            # Set MLflow tracking URI to local directory
            mlflow_dir = self.output_dir / 'mlruns'
            mlflow_dir.mkdir(exist_ok=True)
            mlflow.set_tracking_uri(f"file://{mlflow_dir}")
            
            # Create or get experiment
            experiment_name = "GaussGAN_Statistical_Analysis"
            try:
                mlflow.create_experiment(experiment_name)
            except mlflow.exceptions.MlflowException:
                pass  # Experiment already exists
            
            mlflow.set_experiment(experiment_name)
            self.logger.info(f"MLflow experiment setup: {experiment_name}")
            
        except Exception as e:
            self.logger.warning(f"MLflow setup failed: {e}")
    
    def create_experiment_plan(
        self,
        generator_types: List[str] = None,
        n_runs_per_type: int = 10,
        seed_range: Tuple[int, int] = (42, 142),
        max_epochs_range: Tuple[int, int] = (30, 80),
        batch_sizes: List[int] = None,
        learning_rates: List[float] = None,
        killer_modes: List[bool] = None
    ) -> List[ExperimentConfig]:
        """
        Create a comprehensive experiment plan.
        
        Args:
            generator_types: List of generator types to test
            n_runs_per_type: Number of runs per generator type
            seed_range: Range of seeds to use
            max_epochs_range: Range of max epochs to test
            batch_sizes: List of batch sizes to test
            learning_rates: List of learning rates to test
            killer_modes: List of killer mode settings
            
        Returns:
            List of experiment configurations
        """
        if generator_types is None:
            generator_types = ['classical_normal', 'classical_uniform', 
                             'quantum_samples', 'quantum_shadows']
        if batch_sizes is None:
            batch_sizes = [256]
        if learning_rates is None:
            learning_rates = [0.001]
        if killer_modes is None:
            killer_modes = [False]
        
        experiments = []
        seed_start, seed_end = seed_range
        
        for gen_type in generator_types:
            for run_idx in range(n_runs_per_type):
                # Generate systematic seed
                seed = seed_start + len(experiments)
                if seed > seed_end:
                    seed = np.random.randint(seed_start, seed_end)
                
                # Sample hyperparameters
                max_epochs = np.random.randint(*max_epochs_range)
                batch_size = np.random.choice(batch_sizes)
                learning_rate = np.random.choice(learning_rates)
                killer = np.random.choice(killer_modes)
                
                # Create run ID
                run_id = f"{gen_type}_run{run_idx:03d}_seed{seed}_ep{max_epochs}"
                
                config = ExperimentConfig(
                    run_id=run_id,
                    generator_type=gen_type,
                    seed=seed,
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    killer=killer
                )
                
                # Add quantum-specific parameters
                if 'quantum' in gen_type:
                    config.quantum_qubits = 6
                    config.quantum_layers = 2
                    config.quantum_shots = 100
                
                experiments.append(config)
        
        self.logger.info(f"Created experiment plan with {len(experiments)} experiments")
        self.logger.info(f"Generator types: {generator_types}")
        self.logger.info(f"Runs per type: {n_runs_per_type}")
        
        return experiments
    
    def execute_single_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Execute a single experiment.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Experiment result
        """
        result = ExperimentResult(
            config=config,
            success=False,
            start_time=datetime.now()
        )
        
        try:
            self.logger.info(f"Starting experiment: {config.run_id}")
            
            # Check resource availability
            if not self.resource_monitor.check_resources():
                raise RuntimeError("Insufficient resources to run experiment")
            
            # Create temporary config file
            temp_config = self._create_temp_config(config)
            temp_config_path = self.output_dir / f"temp_config_{config.run_id}.yaml"
            
            with open(temp_config_path, 'w') as f:
                yaml.dump(temp_config, f)
            
            # Build command
            cmd = self._build_training_command(config, temp_config_path)
            
            # Execute training
            start_time = time.time()
            process_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout
                cwd=Path(__file__).parent.parent  # Run from project root
            )
            
            duration = time.time() - start_time
            result.duration_seconds = duration
            result.end_time = datetime.now()
            
            # Check if training was successful
            if process_result.returncode == 0:
                result.success = True
                
                # Extract results from MLflow or checkpoint
                self._extract_experiment_results(config, result)
                
                self.logger.info(f"Completed experiment: {config.run_id} "
                               f"(duration: {duration:.1f}s)")
            else:
                result.error_message = process_result.stderr
                self.logger.error(f"Experiment {config.run_id} failed: {result.error_message}")
            
            # Cleanup
            if temp_config_path.exists():
                temp_config_path.unlink()
                
        except subprocess.TimeoutExpired:
            result.error_message = "Training timeout exceeded"
            result.end_time = datetime.now()
            self.logger.error(f"Experiment {config.run_id} timed out")
            
        except Exception as e:
            result.error_message = str(e)
            result.end_time = datetime.now()
            self.logger.error(f"Experiment {config.run_id} failed with exception: {e}")
        
        return result
    
    def _create_temp_config(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Create temporary configuration for experiment."""
        temp_config = self.base_config.copy()
        
        # Update with experiment-specific parameters
        temp_config.update({
            'generator_type': config.generator_type,
            'seed': config.seed,
            'max_epochs': config.max_epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'killer': config.killer,
            'experiment_name': f"statistical_analysis_{config.run_id}",
            'quantum_qubits': config.quantum_qubits,
            'quantum_layers': config.quantum_layers,
            'quantum_shots': config.quantum_shots
        })
        
        # Add additional arguments
        temp_config.update(config.additional_args)
        
        return temp_config
    
    def _build_training_command(
        self, 
        config: ExperimentConfig, 
        config_path: Path
    ) -> List[str]:
        """Build the training command."""
        cmd = [
            'uv', 'run', 'python', 'main.py',
            '--generator_type', config.generator_type,
            '--max_epochs', str(config.max_epochs),
            '--seed', str(config.seed),
            '--experiment_name', f"statistical_analysis_{config.run_id}"
        ]
        
        if config.killer:
            cmd.extend(['--killer', 'true'])
        
        return cmd
    
    def _extract_experiment_results(
        self, 
        config: ExperimentConfig, 
        result: ExperimentResult
    ):
        """Extract results from completed experiment."""
        try:
            # Try to get results from MLflow
            if MLFLOW_AVAILABLE:
                self._extract_mlflow_results(config, result)
            
            # Try to get results from checkpoint
            self._extract_checkpoint_results(config, result)
            
        except Exception as e:
            self.logger.warning(f"Failed to extract results for {config.run_id}: {e}")
    
    def _extract_mlflow_results(self, config: ExperimentConfig, result: ExperimentResult):
        """Extract results from MLflow run."""
        try:
            # Find the most recent run for this experiment
            experiment = mlflow.get_experiment_by_name("GaussGAN_Statistical_Analysis")
            if experiment is None:
                return
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.run_id = '{config.run_id}'",
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if len(runs) > 0:
                run = runs.iloc[0]
                result.mlflow_run_id = run['run_id']
                
                # Extract metrics
                metrics = {}
                for col in run.index:
                    if col.startswith('metrics.'):
                        metric_name = col.replace('metrics.', '')
                        metrics[metric_name] = run[col]
                
                result.final_metrics = metrics
                
        except Exception as e:
            self.logger.warning(f"Failed to extract MLflow results: {e}")
    
    def _extract_checkpoint_results(self, config: ExperimentConfig, result: ExperimentResult):
        """Extract results from checkpoint files."""
        try:
            # Look for checkpoint files
            checkpoint_dir = Path("checkpoints")
            if not checkpoint_dir.exists():
                return
            
            # Find checkpoint file for this run
            pattern = f"*{config.run_id}*"
            checkpoint_files = list(checkpoint_dir.glob(pattern))
            
            if checkpoint_files:
                result.checkpoint_path = str(checkpoint_files[0])
                
                # Try to extract metrics from checkpoint filename or logs
                # This would depend on your specific checkpoint naming convention
                
        except Exception as e:
            self.logger.warning(f"Failed to extract checkpoint results: {e}")
    
    def run_experiments(
        self, 
        experiment_configs: List[ExperimentConfig],
        parallel: bool = False,
        save_interval: int = 5
    ) -> List[ExperimentResult]:
        """
        Execute all experiments in the plan.
        
        Args:
            experiment_configs: List of experiment configurations
            parallel: Whether to run experiments in parallel
            save_interval: Save results every N experiments
            
        Returns:
            List of experiment results
        """
        self.logger.info(f"Starting {len(experiment_configs)} experiments")
        
        # Save experiment plan
        plan_file = self.output_dir / 'experiment_plan.json'
        with open(plan_file, 'w') as f:
            json.dump([asdict(config) for config in experiment_configs], f, indent=2)
        
        if parallel and self.max_parallel > 1:
            results = self._run_experiments_parallel(experiment_configs, save_interval)
        else:
            results = self._run_experiments_sequential(experiment_configs, save_interval)
        
        # Save final results
        self.save_results()
        
        # Generate summary
        self._generate_execution_summary()
        
        return results
    
    def _run_experiments_sequential(
        self, 
        configs: List[ExperimentConfig], 
        save_interval: int
    ) -> List[ExperimentResult]:
        """Run experiments sequentially."""
        results = []
        
        progress_bar = tqdm(configs, desc="Running experiments")
        for i, config in enumerate(progress_bar):
            progress_bar.set_description(f"Running: {config.run_id}")
            
            result = self.execute_single_experiment(config)
            results.append(result)
            self.results.append(result)
            
            # Update progress
            progress_bar.set_postfix({
                'Success': sum(1 for r in results if r.success),
                'Failed': sum(1 for r in results if not r.success)
            })
            
            # Periodic saving
            if (i + 1) % save_interval == 0:
                self.save_results()
                self.save_progress(i + 1, len(configs))
        
        return results
    
    def _run_experiments_parallel(
        self, 
        configs: List[ExperimentConfig], 
        save_interval: int
    ) -> List[ExperimentResult]:
        """Run experiments in parallel."""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            # Submit all experiments
            future_to_config = {
                executor.submit(self.execute_single_experiment, config): config 
                for config in configs
            }
            
            # Collect results as they complete
            with tqdm(total=len(configs), desc="Running experiments") as pbar:
                for future in concurrent.futures.as_completed(future_to_config):
                    config = future_to_config[future]
                    try:
                        result = future.result()
                        results.append(result)
                        self.results.append(result)
                        
                        pbar.update(1)
                        pbar.set_postfix({
                            'Success': sum(1 for r in results if r.success),
                            'Failed': sum(1 for r in results if not r.success)
                        })
                        
                        # Periodic saving
                        if len(results) % save_interval == 0:
                            self.save_results()
                            
                    except Exception as e:
                        self.logger.error(f"Experiment {config.run_id} failed: {e}")
        
        return results
    
    def save_results(self):
        """Save experiment results to file."""
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(self.results),
            'successful_experiments': sum(1 for r in self.results if r.success),
            'failed_experiments': sum(1 for r in self.results if not r.success),
            'results': []
        }
        
        for result in self.results:
            result_dict = asdict(result)
            # Convert datetime objects to strings
            if result_dict['start_time']:
                result_dict['start_time'] = result.start_time.isoformat()
            if result_dict['end_time']:
                result_dict['end_time'] = result.end_time.isoformat()
            results_data['results'].append(result_dict)
        
        with open(self.results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        self.logger.info(f"Saved {len(self.results)} experiment results")
    
    def save_progress(self, completed: int, total: int):
        """Save progress information."""
        progress_data = {
            'timestamp': datetime.now().isoformat(),
            'completed': completed,
            'total': total,
            'progress_percentage': (completed / total) * 100 if total > 0 else 0,
            'estimated_remaining_time': self._estimate_remaining_time(completed, total)
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def _estimate_remaining_time(self, completed: int, total: int) -> str:
        """Estimate remaining time based on completed experiments."""
        if completed == 0 or len(self.results) == 0:
            return "Unknown"
        
        # Calculate average duration of completed experiments
        successful_results = [r for r in self.results if r.success and r.duration_seconds > 0]
        if not successful_results:
            return "Unknown"
        
        avg_duration = np.mean([r.duration_seconds for r in successful_results])
        remaining_experiments = total - completed
        remaining_seconds = remaining_experiments * avg_duration
        
        # Convert to human-readable format
        hours = int(remaining_seconds // 3600)
        minutes = int((remaining_seconds % 3600) // 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    
    def _generate_execution_summary(self):
        """Generate a summary of experiment execution."""
        if not self.results:
            return
        
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        summary = {
            'execution_summary': {
                'total_experiments': len(self.results),
                'successful': len(successful),
                'failed': len(failed),
                'success_rate': len(successful) / len(self.results) if self.results else 0,
                'total_duration_hours': sum(r.duration_seconds for r in self.results) / 3600,
                'avg_duration_minutes': np.mean([r.duration_seconds for r in successful]) / 60 if successful else 0
            },
            'generator_type_summary': {},
            'failure_analysis': {}
        }
        
        # Summary by generator type
        generator_types = set(r.config.generator_type for r in self.results)
        for gen_type in generator_types:
            gen_results = [r for r in self.results if r.config.generator_type == gen_type]
            gen_successful = [r for r in gen_results if r.success]
            
            summary['generator_type_summary'][gen_type] = {
                'total': len(gen_results),
                'successful': len(gen_successful),
                'success_rate': len(gen_successful) / len(gen_results) if gen_results else 0,
                'avg_duration_minutes': np.mean([r.duration_seconds for r in gen_successful]) / 60 if gen_successful else 0
            }
        
        # Failure analysis
        if failed:
            error_counts = {}
            for result in failed:
                error_type = result.error_message.split(':')[0] if result.error_message else 'Unknown'
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            summary['failure_analysis'] = {
                'common_errors': error_counts,
                'failure_rate_by_generator': {
                    gen_type: sum(1 for r in failed if r.config.generator_type == gen_type)
                    for gen_type in generator_types
                }
            }
        
        # Save summary
        summary_file = self.output_dir / 'execution_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info("Generated execution summary")
        self.logger.info(f"Success rate: {summary['execution_summary']['success_rate']:.2%}")
        self.logger.info(f"Total duration: {summary['execution_summary']['total_duration_hours']:.1f} hours")
    
    def load_results(self) -> List[ExperimentResult]:
        """Load previously saved experiment results."""
        if not self.results_file.exists():
            self.logger.warning("No saved results found")
            return []
        
        with open(self.results_file, 'r') as f:
            data = json.load(f)
        
        results = []
        for result_data in data['results']:
            # Convert datetime strings back to datetime objects
            if 'start_time' in result_data and isinstance(result_data['start_time'], str):
                result_data['start_time'] = datetime.fromisoformat(result_data['start_time'])
            if 'end_time' in result_data and isinstance(result_data['end_time'], str):
                result_data['end_time'] = datetime.fromisoformat(result_data['end_time'])
            
            # Reconstruct config
            config_data = result_data['config']
            config = ExperimentConfig(**config_data)
            result_data['config'] = config
            
            result = ExperimentResult(**result_data)
            results.append(result)
        
        self.results = results
        self.logger.info(f"Loaded {len(results)} experiment results")
        return results


class ResourceMonitor:
    """Monitor system resources during experiment execution."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def check_resources(self) -> bool:
        """Check if sufficient resources are available."""
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                self.logger.warning(f"High CPU usage: {cpu_percent}%")
                return False
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                self.logger.warning(f"High memory usage: {memory.percent}%")
                return False
            
            # Check disk space
            disk = psutil.disk_usage('/')
            if disk.percent > 95:
                self.logger.warning(f"Low disk space: {disk.percent}% used")
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Resource check failed: {e}")
            return True  # Assume OK if check fails
    
    def get_resource_status(self) -> Dict[str, float]:
        """Get current resource usage."""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent
            }
        except Exception as e:
            self.logger.warning(f"Failed to get resource status: {e}")
            return {}


def create_sample_experiment_plan():
    """Create a sample experiment plan for demonstration."""
    runner = ExperimentRunner(
        base_config_path="config.yaml",
        output_dir="docs/statistical_analysis/experiments",
        max_parallel=1
    )
    
    # Create a small test plan
    experiment_plan = runner.create_experiment_plan(
        generator_types=['classical_normal', 'quantum_samples'],
        n_runs_per_type=3,
        seed_range=(42, 50),
        max_epochs_range=(20, 30),
        batch_sizes=[256],
        learning_rates=[0.001],
        killer_modes=[False]
    )
    
    print(f"Created experiment plan with {len(experiment_plan)} experiments:")
    for config in experiment_plan:
        print(f"  {config.run_id}: {config.generator_type}, seed={config.seed}, epochs={config.max_epochs}")
    
    return runner, experiment_plan


if __name__ == "__main__":
    # Demonstrate the experiment runner
    runner, plan = create_sample_experiment_plan()
    
    print("\nTo execute the experiments, run:")
    print("results = runner.run_experiments(plan)")
    print("\nTo load previous results:")
    print("results = runner.load_results()")