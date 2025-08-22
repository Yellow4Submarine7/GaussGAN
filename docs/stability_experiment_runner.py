"""
Stability Experiment Runner for GaussGAN

This script runs multiple training experiments with different random seeds
to generate data for stability analysis. It supports running experiments
for different generator types and collecting comprehensive metrics.

Features:
- Automated multi-seed experiment execution
- Parallel/sequential experiment running
- Comprehensive error handling and logging
- Integration with existing MLflow tracking
- Configurable experiment parameters
- Results aggregation and validation

Usage:
    python stability_experiment_runner.py --generator_types classical_normal quantum_samples --num_seeds 20 --max_epochs 50
    
Author: Created for GaussGAN quantum vs classical generator comparison
"""

import os
import sys
import argparse
import subprocess
import time
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from docs.stability_analyzer import StabilityAnalyzer


@dataclass
class ExperimentConfig:
    """Configuration for a stability experiment."""
    generator_type: str
    seed: int
    max_epochs: int
    experiment_name: str
    additional_params: Dict[str, Any] = None


class StabilityExperimentRunner:
    """
    Manages running multiple GaussGAN experiments for stability analysis.
    
    This class handles:
    - Experiment configuration generation
    - Parallel/sequential execution
    - Progress tracking and logging
    - Results collection and validation
    - Integration with StabilityAnalyzer
    """
    
    def __init__(self, base_config_file: str = "config.yaml", 
                 output_dir: str = "docs/stability_analysis",
                 log_level: str = "INFO"):
        """
        Initialize the experiment runner.
        
        Args:
            base_config_file: Path to base configuration file
            output_dir: Directory for output files and logs
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.base_config_file = base_config_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging(log_level)
        
        # Load base configuration
        self.base_config = self._load_base_config()
        
        # Track experiments
        self.planned_experiments: List[ExperimentConfig] = []
        self.completed_experiments: List[str] = []
        self.failed_experiments: List[Dict] = []
        
    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('StabilityExperimentRunner')
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create file handler
        log_file = self.output_dir / 'stability_experiments.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler  
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration from YAML file."""
        try:
            with open(self.base_config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Loaded base config from {self.base_config_file}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load base config: {e}")
            return {}
    
    def generate_experiment_configs(self, 
                                  generator_types: List[str],
                                  num_seeds: int,
                                  seed_start: int = 42,
                                  max_epochs: int = None,
                                  experiment_name: str = "GaussGAN-stability",
                                  additional_params: Dict[str, Any] = None) -> List[ExperimentConfig]:
        """
        Generate experiment configurations for stability testing.
        
        Args:
            generator_types: List of generator types to test
            num_seeds: Number of different seeds to run for each generator
            seed_start: Starting seed value
            max_epochs: Override max epochs (uses config default if None)
            experiment_name: MLflow experiment name
            additional_params: Additional parameters to override in config
            
        Returns:
            List of experiment configurations
        """
        configs = []
        
        if max_epochs is None:
            max_epochs = self.base_config.get('max_epochs', 50)
        
        for generator_type in generator_types:
            for i in range(num_seeds):
                seed = seed_start + i
                
                config = ExperimentConfig(
                    generator_type=generator_type,
                    seed=seed,
                    max_epochs=max_epochs,
                    experiment_name=experiment_name,
                    additional_params=additional_params or {}
                )
                
                configs.append(config)
        
        self.planned_experiments = configs
        self.logger.info(f"Generated {len(configs)} experiment configurations")
        
        return configs
    
    def run_single_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """
        Run a single experiment with the given configuration.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Dictionary with experiment results and metadata
        """
        start_time = time.time()
        
        try:
            # Build command line arguments
            cmd_args = [
                'uv', 'run', 'python', 'main.py',
                '--generator_type', config.generator_type,
                '--seed', str(config.seed),
                '--max_epochs', str(config.max_epochs),
                '--experiment_name', config.experiment_name
            ]
            
            # Add additional parameters
            if config.additional_params:
                for key, value in config.additional_params.items():
                    cmd_args.extend([f'--{key}', str(value)])
            
            self.logger.info(f"Starting experiment: {config.generator_type}, seed={config.seed}")
            self.logger.debug(f"Command: {' '.join(cmd_args)}")
            
            # Run the experiment
            result = subprocess.run(
                cmd_args,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                self.logger.info(f"Experiment completed successfully: {config.generator_type}, seed={config.seed}")
                return {
                    'status': 'success',
                    'config': config,
                    'duration': duration,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode
                }
            else:
                self.logger.error(f"Experiment failed: {config.generator_type}, seed={config.seed}")
                return {
                    'status': 'failed',
                    'config': config,
                    'duration': duration,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode,
                    'error': f"Non-zero return code: {result.returncode}"
                }
                
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            error_msg = f"Experiment timed out after {duration:.1f} seconds"
            self.logger.error(f"{error_msg}: {config.generator_type}, seed={config.seed}")
            return {
                'status': 'timeout',
                'config': config,
                'duration': duration,
                'error': error_msg
            }
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(f"{error_msg}: {config.generator_type}, seed={config.seed}")
            return {
                'status': 'error',
                'config': config,
                'duration': duration,
                'error': error_msg
            }
    
    def run_experiments_sequential(self, configs: List[ExperimentConfig]) -> List[Dict[str, Any]]:
        """
        Run experiments sequentially (one after another).
        
        Args:
            configs: List of experiment configurations
            
        Returns:
            List of experiment results
        """
        results = []
        
        self.logger.info(f"Starting sequential execution of {len(configs)} experiments")
        
        for i, config in enumerate(configs, 1):
            self.logger.info(f"Running experiment {i}/{len(configs)}")
            result = self.run_single_experiment(config)
            results.append(result)
            
            # Track results
            if result['status'] == 'success':
                self.completed_experiments.append(f"{config.generator_type}_seed{config.seed}")
            else:
                self.failed_experiments.append({
                    'config': config,
                    'error': result.get('error', 'Unknown error')
                })
            
            # Save intermediate results
            if i % 5 == 0:  # Save every 5 experiments
                self._save_intermediate_results(results)
        
        self.logger.info(f"Sequential execution completed. "
                        f"Success: {len(self.completed_experiments)}, "
                        f"Failed: {len(self.failed_experiments)}")
        
        return results
    
    def run_experiments_parallel(self, configs: List[ExperimentConfig], 
                                max_workers: int = 2) -> List[Dict[str, Any]]:
        """
        Run experiments in parallel.
        
        Args:
            configs: List of experiment configurations
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of experiment results
        """
        results = []
        
        self.logger.info(f"Starting parallel execution of {len(configs)} experiments "
                        f"with {max_workers} workers")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all experiments
            future_to_config = {
                executor.submit(self.run_single_experiment, config): config
                for config in configs
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Track results
                    if result['status'] == 'success':
                        self.completed_experiments.append(f"{config.generator_type}_seed{config.seed}")
                    else:
                        self.failed_experiments.append({
                            'config': config,
                            'error': result.get('error', 'Unknown error')
                        })
                    
                    self.logger.info(f"Completed: {config.generator_type}, seed={config.seed}, "
                                   f"status={result['status']}")
                    
                except Exception as e:
                    self.logger.error(f"Future failed for {config.generator_type}, "
                                    f"seed={config.seed}: {e}")
                    
                    self.failed_experiments.append({
                        'config': config,
                        'error': f"Future execution failed: {str(e)}"
                    })
        
        self.logger.info(f"Parallel execution completed. "
                        f"Success: {len(self.completed_experiments)}, "
                        f"Failed: {len(self.failed_experiments)}")
        
        return results
    
    def _save_intermediate_results(self, results: List[Dict[str, Any]]) -> None:
        """Save intermediate results to avoid data loss."""
        results_file = self.output_dir / 'intermediate_results.json'
        
        # Prepare serializable data
        serializable_results = []
        for result in results:
            serializable_result = result.copy()
            # Convert ExperimentConfig to dict
            if 'config' in serializable_result:
                config = serializable_result['config']
                if hasattr(config, '__dict__'):
                    serializable_result['config'] = {
                        'generator_type': config.generator_type,
                        'seed': config.seed,
                        'max_epochs': config.max_epochs,
                        'experiment_name': config.experiment_name,
                        'additional_params': config.additional_params
                    }
            serializable_results.append(serializable_result)
        
        try:
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save intermediate results: {e}")
    
    def run_stability_experiment_suite(self,
                                     generator_types: List[str],
                                     num_seeds: int = 10,
                                     max_epochs: int = 50,
                                     parallel: bool = False,
                                     max_workers: int = 2,
                                     experiment_name: str = "GaussGAN-stability") -> Dict[str, Any]:
        """
        Run a complete stability experiment suite.
        
        Args:
            generator_types: List of generator types to test
            num_seeds: Number of seeds per generator type
            max_epochs: Maximum training epochs
            parallel: Whether to run experiments in parallel
            max_workers: Number of parallel workers (if parallel=True)
            experiment_name: MLflow experiment name
            
        Returns:
            Dictionary with experiment results and summary
        """
        self.logger.info("=== Starting Stability Experiment Suite ===")
        self.logger.info(f"Generator types: {generator_types}")
        self.logger.info(f"Seeds per generator: {num_seeds}")
        self.logger.info(f"Max epochs: {max_epochs}")
        self.logger.info(f"Parallel execution: {parallel}")
        
        # Generate experiment configurations
        configs = self.generate_experiment_configs(
            generator_types=generator_types,
            num_seeds=num_seeds,
            max_epochs=max_epochs,
            experiment_name=experiment_name
        )
        
        # Run experiments
        start_time = time.time()
        
        if parallel:
            results = self.run_experiments_parallel(configs, max_workers)
        else:
            results = self.run_experiments_sequential(configs)
        
        total_duration = time.time() - start_time
        
        # Generate summary
        summary = {
            'experiment_suite_info': {
                'generator_types': generator_types,
                'num_seeds': num_seeds,
                'max_epochs': max_epochs,
                'parallel': parallel,
                'total_experiments': len(configs),
                'total_duration_seconds': total_duration,
                'experiment_name': experiment_name
            },
            'results_summary': {
                'successful_experiments': len(self.completed_experiments),
                'failed_experiments': len(self.failed_experiments),
                'success_rate': len(self.completed_experiments) / len(configs) if configs else 0
            },
            'results': results,
            'failed_experiments_details': self.failed_experiments
        }
        
        # Save complete results
        self._save_experiment_suite_results(summary)
        
        self.logger.info("=== Stability Experiment Suite Completed ===")
        self.logger.info(f"Total duration: {total_duration:.1f} seconds")
        self.logger.info(f"Success rate: {summary['results_summary']['success_rate']:.2%}")
        
        return summary
    
    def _save_experiment_suite_results(self, summary: Dict[str, Any]) -> None:
        """Save complete experiment suite results."""
        results_file = self.output_dir / 'experiment_suite_results.json'
        
        try:
            # Prepare serializable data
            serializable_summary = self._make_serializable(summary)
            
            with open(results_file, 'w') as f:
                json.dump(serializable_summary, f, indent=2, default=str)
            
            self.logger.info(f"Experiment suite results saved to {results_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save experiment suite results: {e}")
    
    def _make_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            # Convert dataclass or object to dict
            if hasattr(obj, 'generator_type'):  # ExperimentConfig
                return {
                    'generator_type': obj.generator_type,
                    'seed': obj.seed,
                    'max_epochs': obj.max_epochs,
                    'experiment_name': obj.experiment_name,
                    'additional_params': obj.additional_params
                }
            else:
                return obj.__dict__
        else:
            return obj
    
    def run_analysis_after_experiments(self, experiment_name: str = "GaussGAN-stability") -> None:
        """
        Run stability analysis after experiments are completed.
        
        Args:
            experiment_name: MLflow experiment name to analyze
        """
        self.logger.info("=== Running Stability Analysis ===")
        
        try:
            # Initialize analyzer
            analyzer = StabilityAnalyzer(
                experiment_name=experiment_name,
                stability_threshold=0.15
            )
            
            # Load results from MLflow
            loaded_count = analyzer.load_from_mlflow()
            self.logger.info(f"Loaded {loaded_count} experiments for analysis")
            
            if loaded_count > 0:
                # Generate comprehensive report
                report = analyzer.generate_stability_report(str(self.output_dir))
                
                # Display summary
                summary_df = analyzer.get_stability_summary()
                self.logger.info(f"Analysis complete. Summary:\n{summary_df.to_string()}")
                
                return report
            else:
                self.logger.warning("No experiments found for analysis")
                return None
                
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return None


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run stability experiments for GaussGAN generator comparison"
    )
    
    parser.add_argument(
        '--generator_types',
        nargs='+',
        default=['classical_normal', 'quantum_samples'],
        help='Generator types to test (default: classical_normal quantum_samples)'
    )
    
    parser.add_argument(
        '--num_seeds',
        type=int,
        default=10,
        help='Number of different seeds per generator type (default: 10)'
    )
    
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=50,
        help='Maximum training epochs (default: 50)'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run experiments in parallel'
    )
    
    parser.add_argument(
        '--max_workers',
        type=int,
        default=2,
        help='Maximum parallel workers (default: 2)'
    )
    
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='GaussGAN-stability',
        help='MLflow experiment name (default: GaussGAN-stability)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='docs/stability_analysis',
        help='Output directory (default: docs/stability_analysis)'
    )
    
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--run_analysis',
        action='store_true',
        help='Run stability analysis after experiments'
    )
    
    parser.add_argument(
        '--analysis_only',
        action='store_true',
        help='Only run analysis, skip experiments'
    )
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = StabilityExperimentRunner(
        output_dir=args.output_dir,
        log_level=args.log_level
    )
    
    try:
        if args.analysis_only:
            # Only run analysis
            runner.run_analysis_after_experiments(args.experiment_name)
        else:
            # Run experiments
            summary = runner.run_stability_experiment_suite(
                generator_types=args.generator_types,
                num_seeds=args.num_seeds,
                max_epochs=args.max_epochs,
                parallel=args.parallel,
                max_workers=args.max_workers,
                experiment_name=args.experiment_name
            )
            
            # Print summary
            print("\n=== EXPERIMENT SUITE SUMMARY ===")
            print(f"Total experiments: {summary['experiment_suite_info']['total_experiments']}")
            print(f"Successful: {summary['results_summary']['successful_experiments']}")
            print(f"Failed: {summary['results_summary']['failed_experiments']}")
            print(f"Success rate: {summary['results_summary']['success_rate']:.2%}")
            print(f"Total duration: {summary['experiment_suite_info']['total_duration_seconds']:.1f} seconds")
            
            # Run analysis if requested
            if args.run_analysis:
                time.sleep(2)  # Give MLflow time to flush
                runner.run_analysis_after_experiments(args.experiment_name)
    
    except KeyboardInterrupt:
        print("\nExperiment suite interrupted by user")
        runner.logger.info("Experiment suite interrupted by user")
    except Exception as e:
        print(f"Experiment suite failed: {e}")
        runner.logger.error(f"Experiment suite failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())