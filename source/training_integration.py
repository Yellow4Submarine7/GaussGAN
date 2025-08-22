"""
Training Integration Module for ConvergenceTracker

This module provides easy integration of the ConvergenceTracker into the existing
GaussGAN training pipeline with minimal code changes.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from .convergence import ConvergenceTracker, create_convergence_tracker_for_config


class ConvergenceCallback(Callback):
    """
    PyTorch Lightning callback for automated convergence tracking.
    
    This callback integrates seamlessly with the existing Lightning training loop
    and automatically tracks convergence metrics during validation.
    """
    
    def __init__(
        self, 
        convergence_tracker: Optional[ConvergenceTracker] = None,
        generator_type: str = "unknown",
        save_frequency: int = 50,  # Save results every N epochs
        plot_frequency: int = 25   # Generate plots every N epochs
    ):
        """
        Initialize the convergence callback.
        
        Args:
            convergence_tracker: Pre-configured tracker (optional)
            generator_type: Name/type of the generator being trained
            save_frequency: How often to save intermediate results
            plot_frequency: How often to generate convergence plots
        """
        super().__init__()
        self.convergence_tracker = convergence_tracker
        self.generator_type = generator_type
        self.save_frequency = save_frequency
        self.plot_frequency = plot_frequency
        self._epoch_count = 0
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Called at the start of each training epoch."""
        if self.convergence_tracker:
            self.convergence_tracker.start_epoch()
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of each training epoch."""
        if self.convergence_tracker:
            self.convergence_tracker.end_epoch()
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called at the end of each validation epoch."""
        if not self.convergence_tracker:
            return
        
        # Extract metrics from the logged values
        logged_metrics = trainer.logged_metrics
        
        # Filter for validation metrics
        validation_metrics = {
            key: float(value) for key, value in logged_metrics.items()
            if key.startswith('ValidationStep_FakeData_') and not key.endswith('_epoch')
        }
        
        if validation_metrics:
            current_epoch = trainer.current_epoch
            self.convergence_tracker.update_metrics(current_epoch, validation_metrics)
            
            # Periodic saving and plotting
            if current_epoch > 0 and current_epoch % self.save_frequency == 0:
                self.convergence_tracker.save_results(generator_type=self.generator_type)
            
            if current_epoch > 0 and current_epoch % self.plot_frequency == 0:
                self.convergence_tracker.plot_convergence_curves(
                    save_plots=True, 
                    generator_type=self.generator_type
                )
        
        self._epoch_count += 1
    
    def on_train_end(self, trainer, pl_module):
        """Called at the end of training."""
        if self.convergence_tracker:
            # Final save and plot generation
            self.convergence_tracker.save_results(generator_type=self.generator_type)
            self.convergence_tracker.plot_convergence_curves(
                save_plots=True, 
                generator_type=self.generator_type
            )
            
            # Print summary
            analysis = self.convergence_tracker.get_comparative_analysis()
            print("\n" + "="*60)
            print(f"CONVERGENCE ANALYSIS SUMMARY - {self.generator_type}")
            print("="*60)
            print(f"Total epochs trained: {analysis['training_efficiency']['total_epochs']}")
            print(f"Metrics converged: {analysis['summary']['metrics_converged']}/{analysis['summary']['total_metrics_tracked']}")
            print(f"Convergence rate: {analysis['summary']['overall_convergence_rate']:.2%}")
            
            if analysis['rankings']['fastest_convergence']:
                print(f"Fastest converging metric: {analysis['rankings']['fastest_convergence'][0]}")
            
            print("="*60)


class ConvergenceTrainingManager:
    """
    High-level manager for convergence-aware training experiments.
    
    This class provides a simple interface for running convergence experiments
    and comparing different generator types.
    """
    
    def __init__(self, base_config_path: str = "config.yaml"):
        """
        Initialize the training manager.
        
        Args:
            base_config_path: Path to the base configuration file
        """
        self.base_config_path = Path(base_config_path)
        self.results_dir = Path("docs/convergence_experiments")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_trackers = {}
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load training configuration from YAML file."""
        config_path = Path(config_path) if config_path else self.base_config_path
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def create_convergence_experiment(
        self, 
        generator_type: str,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> tuple[Dict[str, Any], ConvergenceTracker, ConvergenceCallback]:
        """
        Create a complete convergence tracking setup for an experiment.
        
        Args:
            generator_type: Type of generator (e.g., 'classical_normal', 'quantum_samples')
            config_overrides: Optional configuration overrides
            
        Returns:
            Tuple of (config, tracker, callback)
        """
        # Load base configuration
        config = self.load_config()
        
        # Apply overrides
        if config_overrides:
            config.update(config_overrides)
        
        # Ensure generator type is set
        config['generator_type'] = generator_type
        
        # Create convergence tracker
        tracker = create_convergence_tracker_for_config(config)
        
        # Create callback
        callback = ConvergenceCallback(
            convergence_tracker=tracker,
            generator_type=generator_type,
            save_frequency=max(10, config.get('max_epochs', 50) // 5),
            plot_frequency=max(25, config.get('max_epochs', 50) // 2)
        )
        
        # Store tracker for later comparison
        self.experiment_trackers[generator_type] = tracker
        
        return config, tracker, callback
    
    def run_comparative_experiment(
        self, 
        generator_types: list[str],
        max_epochs: int = 50,
        save_comparison: bool = True
    ) -> Dict[str, Any]:
        """
        Run a comparative convergence experiment across multiple generator types.
        
        Args:
            generator_types: List of generator types to compare
            max_epochs: Number of epochs to train each generator
            save_comparison: Whether to save comparison results
            
        Returns:
            Comparison results dictionary
        """
        print(f"Starting comparative experiment with generators: {generator_types}")
        print(f"Training for {max_epochs} epochs each...")
        
        # Note: This method sets up the experiments but doesn't run the actual training
        # The user will need to run the training separately for each generator type
        
        experiment_configs = {}
        
        for gen_type in generator_types:
            config_overrides = {
                'max_epochs': max_epochs,
                'generator_type': gen_type
            }
            
            config, tracker, callback = self.create_convergence_experiment(
                gen_type, config_overrides
            )
            
            experiment_configs[gen_type] = {
                'config': config,
                'tracker': tracker,
                'callback': callback
            }
        
        return experiment_configs
    
    def compare_completed_experiments(
        self, 
        generator_types: list[str],
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Compare results from completed experiments.
        
        Args:
            generator_types: List of generator types that were trained
            save_results: Whether to save comparison results
            
        Returns:
            Detailed comparison results
        """
        if len(generator_types) < 2:
            raise ValueError("Need at least 2 generator types for comparison")
        
        # Get trackers for comparison
        trackers = {}
        for gen_type in generator_types:
            if gen_type in self.experiment_trackers:
                trackers[gen_type] = self.experiment_trackers[gen_type]
            else:
                print(f"Warning: No tracker found for {gen_type}")
        
        if len(trackers) < 2:
            raise ValueError("Need at least 2 completed experiments for comparison")
        
        # Perform pairwise comparisons
        comparison_results = {}
        generator_names = list(trackers.keys())
        
        for i in range(len(generator_names)):
            for j in range(i + 1, len(generator_names)):
                gen1, gen2 = generator_names[i], generator_names[j]
                comparison_key = f"{gen1}_vs_{gen2}"
                
                comparison_results[comparison_key] = trackers[gen1].compare_generators(
                    trackers[gen2], gen1, gen2
                )
        
        # Create overall ranking
        overall_results = {
            "pairwise_comparisons": comparison_results,
            "generator_summaries": {
                gen: tracker.get_comparative_analysis() 
                for gen, tracker in trackers.items()
            }
        }
        
        if save_results:
            self._save_comparison_results(overall_results, generator_types)
        
        return overall_results
    
    def _save_comparison_results(self, results: Dict[str, Any], generator_types: list[str]):
        """Save comparison results to file."""
        import json
        import time
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"generator_comparison_{'_vs_'.join(generator_types)}_{timestamp}.json"
        results_path = self.results_dir / filename
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Comparison results saved to: {results_path}")
    
    def generate_experiment_report(self, generator_types: list[str]) -> str:
        """
        Generate a human-readable report of the convergence experiment.
        
        Args:
            generator_types: List of generator types to include in report
            
        Returns:
            Formatted report string
        """
        if not all(gen in self.experiment_trackers for gen in generator_types):
            return "Error: Not all generator types have been tracked"
        
        report_lines = [
            "GAUSSGAN CONVERGENCE EXPERIMENT REPORT",
            "=" * 50,
            "",
            f"Generator Types Tested: {', '.join(generator_types)}",
            f"Report Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Individual generator summaries
        for gen_type in generator_types:
            tracker = self.experiment_trackers[gen_type]
            analysis = tracker.get_comparative_analysis()
            
            report_lines.extend([
                f"Generator: {gen_type.upper()}",
                "-" * 30,
                f"Total Epochs: {analysis['training_efficiency']['total_epochs']}",
                f"Metrics Converged: {analysis['summary']['metrics_converged']}/{analysis['summary']['total_metrics_tracked']}",
                f"Overall Convergence Rate: {analysis['summary']['overall_convergence_rate']:.2%}",
                f"Average Epoch Time: {analysis['training_efficiency']['average_epoch_time']:.2f}s",
                ""
            ])
            
            # Detail each metric
            for metric_name, metric_stats in analysis['metrics'].items():
                if 'ValidationStep_FakeData_' in metric_name:
                    clean_name = metric_name.replace('ValidationStep_FakeData_', '')
                    converged_status = "✓" if metric_stats.get('converged', False) else "✗"
                    conv_epoch = metric_stats.get('epochs_to_convergence', 'N/A')
                    
                    report_lines.append(
                        f"  {clean_name}: {converged_status} "
                        f"(Epoch {conv_epoch}, Pattern: {metric_stats.get('convergence_pattern', 'unknown')})"
                    )
            
            report_lines.append("")
        
        # Overall comparison if multiple generators
        if len(generator_types) > 1:
            comparison = self.compare_completed_experiments(generator_types, save_results=False)
            
            report_lines.extend([
                "COMPARATIVE ANALYSIS",
                "=" * 30,
                ""
            ])
            
            # Find best performers
            best_convergence = {}
            for gen_type in generator_types:
                summary = comparison['generator_summaries'][gen_type]
                best_convergence[gen_type] = summary['summary']['overall_convergence_rate']
            
            best_generator = max(best_convergence, key=best_convergence.get)
            
            report_lines.extend([
                f"Overall Best Performer: {best_generator}",
                f"Convergence Rate: {best_convergence[best_generator]:.2%}",
                "",
                "Detailed Comparison:",
            ])
            
            for comparison_key, comp_data in comparison['pairwise_comparisons'].items():
                winner = comp_data['summary']['overall_winner']
                total_metrics = comp_data['summary']['total_metrics_compared']
                win_count = comp_data['summary']['wins'][winner]
                
                report_lines.append(f"  {comparison_key}: {winner} wins ({win_count}/{total_metrics} metrics)")
        
        return "\n".join(report_lines)


def setup_convergence_tracking(config_path: str = "config.yaml", generator_type: str = None) -> tuple[ConvergenceTracker, ConvergenceCallback]:
    """
    Simple setup function for convergence tracking with existing training code.
    
    Args:
        config_path: Path to configuration file
        generator_type: Type of generator (if not in config)
        
    Returns:
        Tuple of (tracker, callback) ready for use
        
    Example:
        # In your main training script:
        tracker, callback = setup_convergence_tracking("config.yaml", "classical_normal")
        
        # Add callback to trainer
        trainer = pl.Trainer(callbacks=[callback, ...other_callbacks])
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Use provided generator_type or get from config
    gen_type = generator_type or config.get('generator_type', 'unknown')
    
    # Create tracker
    tracker = create_convergence_tracker_for_config(config)
    
    # Create callback
    callback = ConvergenceCallback(
        convergence_tracker=tracker,
        generator_type=gen_type,
        save_frequency=max(10, config.get('max_epochs', 50) // 5),
        plot_frequency=max(25, config.get('max_epochs', 50) // 2)
    )
    
    return tracker, callback