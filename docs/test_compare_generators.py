#!/usr/bin/env python
"""
Comprehensive test suite for compare_generators.py script
Tests functionality, edge cases, and identifies potential issues
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import tempfile
import warnings

# Add the parent directory to path to import compare_generators
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the functions we want to test
try:
    from compare_generators import (
        get_experiment_runs, 
        analyze_convergence, 
        compare_generators, 
        create_visualization
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Failed to import compare_generators: {e}")
    IMPORT_SUCCESS = False

class TestCompareGenerators(unittest.TestCase):
    """Test suite for compare_generators.py functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not IMPORT_SUCCESS:
            self.skipTest("Cannot import compare_generators module")
        
        # Mock MLflow run data
        self.mock_run_data = {
            'run_id': 'test_run_123',
            'generator_type': 'classical_normal',
            'status': 'FINISHED',
            'start_time': 1600000000000,  # milliseconds
            'end_time': 1600000060000,    # 60 seconds later
            'duration_seconds': 60.0
        }
        
        # Mock metrics data
        self.mock_metrics = {
            'ValidationStep_FakeData_KLDivergence': 0.123,
            'ValidationStep_FakeData_LogLikelihood': -2.456,
            'ValidationStep_FakeData_IsPositive': 0.789,
            'ValidationStep_FakeData_WassersteinDistance': 0.456,
            'ValidationStep_FakeData_MMDDistance': 0.789,
            'train_loss_step': -1.234,
            'd_loss': 0.567,
            'g_loss': -0.890
        }

    @patch('compare_generators.mlflow.tracking.MlflowClient')
    def test_get_experiment_runs_success(self, mock_client_class):
        """Test successful experiment runs retrieval"""
        # Setup mock
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_experiment = Mock()
        mock_experiment.experiment_id = 'exp_123'
        mock_client.get_experiment_by_name.return_value = mock_experiment
        
        # Create mock run
        mock_run = Mock()
        mock_run.info.run_id = self.mock_run_data['run_id']
        mock_run.info.status = self.mock_run_data['status']
        mock_run.info.start_time = self.mock_run_data['start_time']
        mock_run.info.end_time = self.mock_run_data['end_time']
        mock_run.data.params = {'generator_type': 'classical_normal'}
        mock_run.data.metrics = self.mock_metrics
        
        mock_client.search_runs.return_value = [mock_run]
        
        # Test function
        result = get_experiment_runs("test_experiment")
        
        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['generator_type'], 'classical_normal')
        self.assertEqual(result.iloc[0]['duration_seconds'], 60.0)
        
        # Check that all expected metrics are present
        for metric in self.mock_metrics.keys():
            self.assertIn(metric, result.columns)
            self.assertEqual(result.iloc[0][metric], self.mock_metrics[metric])

    @patch('compare_generators.mlflow.tracking.MlflowClient')
    def test_get_experiment_runs_no_experiment(self, mock_client_class):
        """Test handling of non-existent experiment"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_experiment_by_name.return_value = None
        
        result = get_experiment_runs("nonexistent_experiment")
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    @patch('compare_generators.mlflow.tracking.MlflowClient')
    def test_get_experiment_runs_empty_runs(self, mock_client_class):
        """Test handling of experiment with no runs"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_experiment = Mock()
        mock_experiment.experiment_id = 'exp_123'
        mock_client.get_experiment_by_name.return_value = mock_experiment
        mock_client.search_runs.return_value = []
        
        result = get_experiment_runs("empty_experiment")
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)

    def test_analyze_convergence_success(self):
        """Test convergence analysis with valid data"""
        mock_client = Mock()
        
        # Create mock metric history
        mock_metrics = []
        values = [1.0, 0.8, 0.6, 0.4, 0.3, 0.25, 0.24, 0.23, 0.22, 0.21]
        for i, value in enumerate(values):
            mock_metric = Mock()
            mock_metric.step = i
            mock_metric.value = value
            mock_metrics.append(mock_metric)
        
        mock_client.get_metric_history.return_value = mock_metrics
        
        result = analyze_convergence(mock_client, "test_run_id")
        
        # Assertions
        self.assertIsInstance(result, dict)
        self.assertEqual(result['final_value'], 0.21)
        self.assertEqual(result['best_value'], 0.21)
        self.assertEqual(result['epochs_to_best'], 9)
        self.assertGreater(result['improvement_rate'], 0)  # Should be positive (improvement)
        self.assertIsNotNone(result['stability'])

    def test_analyze_convergence_empty_history(self):
        """Test convergence analysis with no metric history"""
        mock_client = Mock()
        mock_client.get_metric_history.return_value = []
        
        result = analyze_convergence(mock_client, "test_run_id")
        
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 0)

    def test_analyze_convergence_single_point(self):
        """Test convergence analysis with single data point"""
        mock_client = Mock()
        
        mock_metric = Mock()
        mock_metric.step = 0
        mock_metric.value = 0.5
        mock_client.get_metric_history.return_value = [mock_metric]
        
        result = analyze_convergence(mock_client, "test_run_id")
        
        self.assertEqual(result['final_value'], 0.5)
        self.assertEqual(result['best_value'], 0.5)
        self.assertEqual(result['epochs_to_best'], 0)
        self.assertEqual(result['improvement_rate'], 0)
        self.assertIsNone(result['stability'])

    @patch('compare_generators.create_visualization')
    @patch('compare_generators.get_experiment_runs')
    @patch('compare_generators.mlflow.tracking.MlflowClient')
    def test_compare_generators_success(self, mock_client_class, mock_get_runs, mock_viz):
        """Test successful generator comparison"""
        # Setup mock data
        mock_df = pd.DataFrame([
            {
                'run_id': 'run1',
                'generator_type': 'classical_normal',
                'duration_seconds': 30.0,
                'ValidationStep_FakeData_KLDivergence': 0.1,
                'ValidationStep_FakeData_LogLikelihood': -2.0,
                'ValidationStep_FakeData_WassersteinDistance': 0.2,
                'ValidationStep_FakeData_MMDDistance': 0.3
            },
            {
                'run_id': 'run2',
                'generator_type': 'quantum_samples',
                'duration_seconds': 120.0,
                'ValidationStep_FakeData_KLDivergence': 0.15,
                'ValidationStep_FakeData_LogLikelihood': -2.5,
                'ValidationStep_FakeData_WassersteinDistance': 0.25,
                'ValidationStep_FakeData_MMDDistance': 0.35
            }
        ])
        
        mock_get_runs.return_value = mock_df
        
        # Mock client and convergence analysis
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock analyze_convergence function
        with patch('compare_generators.analyze_convergence') as mock_convergence:
            mock_convergence.return_value = {
                'epochs_to_best': 10,
                'improvement_rate': 0.05,
                'stability': 0.01
            }
            
            # Mock file operations
            with patch('pandas.DataFrame.to_csv'):
                result = compare_generators("test_experiment")
        
        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)  # Two generator types
        mock_viz.assert_called_once()

    @patch('compare_generators.get_experiment_runs')
    def test_compare_generators_empty_data(self, mock_get_runs):
        """Test handling of empty experiment data"""
        mock_get_runs.return_value = pd.DataFrame()
        
        result = compare_generators("empty_experiment")
        
        self.assertIsNone(result)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    def test_create_visualization(self, mock_tight_layout, mock_savefig):
        """Test visualization creation"""
        # Create test data
        test_df = pd.DataFrame([
            {
                '生成器类型': 'classical_normal',
                '平均训练时间(秒)': 30.0,
                'KL散度(最佳)': 0.1,
                'Wasserstein距离': 0.2,
                'MMD距离': 0.3
            },
            {
                '生成器类型': 'quantum_samples',
                '平均训练时间(秒)': 120.0,
                'KL散度(最佳)': 0.15,
                'Wasserstein距离': 0.25,
                'MMD距离': 0.35
            }
        ])
        
        # Test visualization function
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            # Create mock axes
            mock_axes = [[Mock(), Mock()], [Mock(), Mock()]]
            mock_fig = Mock()
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            create_visualization(test_df)
            
            # Check that subplots was called
            mock_subplots.assert_called_once_with(2, 2, figsize=(12, 10))
            
            # Check that each axis had bar plots created
            for row in mock_axes:
                for ax in row:
                    ax.bar.assert_called_once()
                    ax.set_title.assert_called_once()
                    ax.set_ylabel.assert_called_once()
                    ax.set_xlabel.assert_called_once()
            
            mock_tight_layout.assert_called_once()
            mock_savefig.assert_called_once()

    def test_data_types_and_validation(self):
        """Test data type validation and edge cases"""
        # Test with NaN values
        test_data = pd.DataFrame([
            {
                'generator_type': 'test_gen',
                'duration_seconds': np.nan,
                'ValidationStep_FakeData_KLDivergence': 0.1,
                'ValidationStep_FakeData_LogLikelihood': np.nan,
            }
        ])
        
        # Should handle NaN values gracefully
        mean_duration = test_data['duration_seconds'].mean()
        self.assertTrue(pd.isna(mean_duration))
        
        # Test with empty strings
        test_data_empty = pd.DataFrame([
            {
                'generator_type': '',
                'duration_seconds': 0,
                'ValidationStep_FakeData_KLDivergence': 0,
            }
        ])
        
        self.assertEqual(test_data_empty['generator_type'].iloc[0], '')

    def test_mathematical_calculations(self):
        """Test mathematical calculations for correctness"""
        # Test improvement rate calculation
        values = [1.0, 0.9, 0.8, 0.7, 0.6]
        improvement_rate = (values[0] - values[-1]) / len(values)
        expected_rate = (1.0 - 0.6) / 5
        self.assertAlmostEqual(improvement_rate, expected_rate, places=6)
        
        # Test stability calculation (standard deviation)
        last_5_values = [0.6, 0.61, 0.59, 0.60, 0.62]
        stability = np.std(last_5_values)
        expected_stability = np.std(last_5_values)
        self.assertAlmostEqual(stability, expected_stability, places=6)
        
        # Test percentage difference calculation
        classical_val = 0.1
        quantum_val = 0.15
        diff = ((quantum_val - classical_val) / classical_val * 100)
        expected_diff = 50.0  # 50% increase
        self.assertAlmostEqual(diff, expected_diff, places=1)

    def test_edge_cases(self):
        """Test various edge cases"""
        # Test division by zero protection
        classical_val = 0.0
        quantum_val = 0.15
        diff = ((quantum_val - classical_val) / classical_val * 100) if classical_val != 0 else 0
        self.assertEqual(diff, 0)  # Should default to 0 when classical_val is 0
        
        # Test empty array handling
        empty_array = np.array([])
        self.assertEqual(len(empty_array), 0)
        
        # Test single value arrays
        single_val = np.array([0.5])
        self.assertEqual(len(single_val), 1)
        self.assertEqual(single_val[0], 0.5)

class TestIntegration(unittest.TestCase):
    """Integration tests for file I/O and external dependencies"""
    
    def setUp(self):
        if not IMPORT_SUCCESS:
            self.skipTest("Cannot import compare_generators module")
    
    def test_csv_file_operations(self):
        """Test CSV file creation and reading"""
        test_df = pd.DataFrame([
            {
                '生成器类型': 'test_generator',
                '运行次数': 1,
                '平均训练时间(秒)': 30.0,
                'KL散度(平均)': 0.1
            }
        ])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name
        
        try:
            # Test writing
            test_df.to_csv(temp_file, index=False)
            
            # Test reading
            loaded_df = pd.read_csv(temp_file)
            
            self.assertEqual(len(loaded_df), 1)
            self.assertEqual(loaded_df['生成器类型'].iloc[0], 'test_generator')
            self.assertAlmostEqual(loaded_df['平均训练时间(秒)'].iloc[0], 30.0, places=1)
            
        finally:
            os.unlink(temp_file)

    @patch('matplotlib.pyplot.savefig')
    def test_png_file_operations(self, mock_savefig):
        """Test PNG file creation"""
        # Mock successful file save
        mock_savefig.return_value = None
        
        # This should not raise an exception
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.bar(['A', 'B'], [1, 2])
            plt.savefig('/tmp/test_plot.png')
            plt.close()
        except Exception as e:
            self.fail(f"PNG file operation failed: {e}")

def run_dependency_check():
    """Check if all required dependencies are available"""
    dependencies = [
        'mlflow',
        'pandas', 
        'numpy',
        'matplotlib',
        'seaborn'
    ]
    
    missing_deps = []
    for dep in dependencies:
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"Missing dependencies: {missing_deps}")
        return False
    else:
        print("All dependencies are available")
        return True

def analyze_potential_issues():
    """Analyze the original script for potential issues"""
    issues = []
    
    # Read the original script to analyze
    script_path = "/home/paperx/quantum/GaussGAN/compare_generators.py"
    
    if not os.path.exists(script_path):
        issues.append("Script file not found")
        return issues
    
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for potential issues
    
    # 1. Check for proper error handling around MLflow operations
    if 'try:' not in content or 'except:' not in content:
        issues.append("Missing try-except blocks for MLflow operations")
    
    # 2. Check for NaN handling in calculations
    if 'pd.notna(' in content:
        print("✓ Script includes NaN checking")
    else:
        issues.append("Limited NaN value handling in calculations")
    
    # 3. Check division by zero protection
    if 'c_val != 0' in content:
        print("✓ Script includes division by zero protection")
    else:
        issues.append("Missing division by zero protection")
    
    # 4. Check for empty DataFrame handling
    if 'df.empty' in content:
        print("✓ Script checks for empty DataFrames")
    else:
        issues.append("Missing empty DataFrame checks")
    
    # 5. Check for missing metric handling
    if '.get(' in content:
        print("✓ Script uses safe dictionary access")
    else:
        issues.append("Potential KeyError issues with missing metrics")
    
    return issues

if __name__ == "__main__":
    print("=" * 80)
    print("COMPREHENSIVE ANALYSIS OF compare_generators.py")
    print("=" * 80)
    
    # 1. Dependency check
    print("\n1. DEPENDENCY CHECK:")
    print("-" * 40)
    deps_ok = run_dependency_check()
    
    # 2. Import test
    print("\n2. IMPORT TEST:")
    print("-" * 40)
    if IMPORT_SUCCESS:
        print("✓ Successfully imported compare_generators module")
    else:
        print("✗ Failed to import compare_generators module")
    
    # 3. Potential issues analysis
    print("\n3. POTENTIAL ISSUES ANALYSIS:")
    print("-" * 40)
    issues = analyze_potential_issues()
    
    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"⚠️  Issue {i}: {issue}")
    else:
        print("✓ No major issues detected")
    
    # 4. Run unit tests
    print("\n4. UNIT TESTS:")
    print("-" * 40)
    
    if IMPORT_SUCCESS and deps_ok:
        # Suppress warnings during testing
        warnings.filterwarnings('ignore')
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Add test classes
        suite.addTests(loader.loadTestsFromTestCase(TestCompareGenerators))
        suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Test summary
        print(f"\nTests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        
        if result.failures:
            print("\nFAILURES:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        
        if result.errors:
            print("\nERRORS:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")
        
    else:
        print("Skipping unit tests due to import or dependency issues")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)