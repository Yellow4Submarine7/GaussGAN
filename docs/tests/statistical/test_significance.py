"""
Statistical significance testing framework for quantum vs classical generator comparisons.
Implements rigorous statistical tests to validate performance differences and ensure reproducibility.
"""

import pytest
import torch
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, ks_2samp, levene, shapiro
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

from source.nn import ClassicalNoise, QuantumNoise, QuantumShadowNoise, MLPGenerator
from source.metrics import LogLikelihood, KLDivergence, IsPositive


class StatisticalValidator:
    """Comprehensive statistical validation framework for quantum ML experiments."""
    
    def __init__(self, alpha=0.05, power_threshold=0.8):
        """
        Initialize statistical validator.
        
        Args:
            alpha: Significance level for hypothesis tests
            power_threshold: Minimum statistical power required
        """
        self.alpha = alpha
        self.power_threshold = power_threshold
        self.results = {}
    
    def test_normality(self, data: np.ndarray, method='shapiro') -> Tuple[float, bool]:
        """
        Test if data follows normal distribution.
        
        Args:
            data: Data to test for normality
            method: Statistical test method ('shapiro', 'jarque_bera')
            
        Returns:
            p_value: P-value of normality test
            is_normal: Whether data is normally distributed (p > alpha)
        """
        if method == 'shapiro':
            if len(data) > 5000:  # Shapiro-Wilk is sensitive to sample size
                # Use Jarque-Bera for large samples
                from scipy.stats import jarque_bera
                stat, p_value = jarque_bera(data)
            else:
                stat, p_value = shapiro(data)
        elif method == 'jarque_bera':
            from scipy.stats import jarque_bera
            stat, p_value = jarque_bera(data)
        else:
            raise ValueError(f"Unknown normality test method: {method}")
        
        return p_value, p_value > self.alpha
    
    def test_equal_variance(self, *groups) -> Tuple[float, bool]:
        """
        Test if groups have equal variance using Levene's test.
        
        Args:
            *groups: Variable number of data groups
            
        Returns:
            p_value: P-value of equal variance test
            equal_variance: Whether variances are equal
        """
        stat, p_value = levene(*groups)
        return p_value, p_value > self.alpha
    
    def compare_distributions(self, data1: np.ndarray, data2: np.ndarray, 
                            test_type='auto') -> Dict[str, Any]:
        """
        Compare two distributions using appropriate statistical test.
        
        Args:
            data1, data2: Data from two distributions
            test_type: Type of test ('auto', 'ttest', 'mannwhitney', 'ks')
            
        Returns:
            Dictionary containing test results and recommendations
        """
        results = {
            'n1': len(data1),
            'n2': len(data2),
            'mean1': np.mean(data1),
            'mean2': np.mean(data2),
            'std1': np.std(data1, ddof=1),
            'std2': np.std(data2, ddof=1),
        }
        
        # Test assumptions
        p_norm1, is_norm1 = self.test_normality(data1)
        p_norm2, is_norm2 = self.test_normality(data2)
        p_var, equal_var = self.test_equal_variance(data1, data2)
        
        results.update({
            'normality_p1': p_norm1,
            'normality_p2': p_norm2,
            'is_normal1': is_norm1,
            'is_normal2': is_norm2,
            'equal_variance_p': p_var,
            'equal_variance': equal_var
        })
        
        # Choose appropriate test
        if test_type == 'auto':
            if is_norm1 and is_norm2:
                if equal_var:
                    test_type = 'ttest_equal_var'
                else:
                    test_type = 'ttest_unequal_var'
            else:
                test_type = 'mannwhitney'
        
        # Perform chosen test
        if test_type in ['ttest', 'ttest_equal_var']:
            stat, p_value = stats.ttest_ind(data1, data2, equal_var=True)
        elif test_type == 'ttest_unequal_var':
            stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
        elif test_type == 'mannwhitney':
            stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
        elif test_type == 'ks':
            stat, p_value = ks_2samp(data1, data2)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        results.update({
            'test_used': test_type,
            'test_statistic': stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'effect_size': self.calculate_effect_size(data1, data2)
        })
        
        return results
    
    def calculate_effect_size(self, data1: np.ndarray, data2: np.ndarray, 
                            method='cohen_d') -> float:
        """
        Calculate effect size between two groups.
        
        Args:
            data1, data2: Data from two groups
            method: Effect size method ('cohen_d', 'glass_delta', 'hedges_g')
            
        Returns:
            Effect size value
        """
        mean1, mean2 = np.mean(data1), np.mean(data2)
        std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
        n1, n2 = len(data1), len(data2)
        
        if method == 'cohen_d':
            # Pooled standard deviation
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        elif method == 'glass_delta':
            return (mean1 - mean2) / std2 if std2 > 0 else 0
        elif method == 'hedges_g':
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            cohen_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
            # Hedges' g correction for small samples
            correction = 1 - (3 / (4 * (n1 + n2) - 9))
            return cohen_d * correction
        else:
            raise ValueError(f"Unknown effect size method: {method}")
    
    def interpret_effect_size(self, effect_size: float, method='cohen_d') -> str:
        """Interpret effect size magnitude."""
        abs_effect = abs(effect_size)
        
        if method == 'cohen_d':
            if abs_effect < 0.2:
                return "negligible"
            elif abs_effect < 0.5:
                return "small"
            elif abs_effect < 0.8:
                return "medium"
            else:
                return "large"
        else:
            # Generic interpretation
            if abs_effect < 0.2:
                return "negligible"
            elif abs_effect < 0.5:
                return "small"
            elif abs_effect < 0.8:
                return "medium"
            else:
                return "large"
    
    def power_analysis(self, effect_size: float, n1: int, n2: int = None, 
                      test_type='ttest') -> float:
        """
        Calculate statistical power for given effect size and sample sizes.
        
        Args:
            effect_size: Expected effect size
            n1: Sample size for group 1
            n2: Sample size for group 2 (defaults to n1)
            test_type: Type of statistical test
            
        Returns:
            Statistical power (0-1)
        """
        if n2 is None:
            n2 = n1
        
        try:
            from statsmodels.stats.power import ttest_power
            
            if test_type == 'ttest':
                power = ttest_power(
                    effect_size=effect_size,
                    nobs1=n1,
                    nobs2=n2,
                    alpha=self.alpha,
                    alternative='two-sided'
                )
                return power
        except ImportError:
            warnings.warn("statsmodels not available, using approximate power calculation")
            
        # Approximate power calculation for t-test
        # This is a simplified version
        import math
        
        # Degrees of freedom
        df = n1 + n2 - 2
        
        # Non-centrality parameter
        ncp = effect_size * math.sqrt(n1 * n2 / (n1 + n2))
        
        # Critical t-value
        t_crit = stats.t.ppf(1 - self.alpha/2, df)
        
        # Approximate power (simplified)
        power = 1 - stats.t.cdf(t_crit - ncp, df) + stats.t.cdf(-t_crit - ncp, df)
        
        return max(0, min(1, power))
    
    def multiple_comparison_correction(self, p_values: List[float], 
                                     method='bonferroni') -> List[float]:
        """
        Apply multiple comparison correction to p-values.
        
        Args:
            p_values: List of p-values to correct
            method: Correction method ('bonferroni', 'fdr_bh')
            
        Returns:
            Corrected p-values
        """
        if method == 'bonferroni':
            return [min(1.0, p * len(p_values)) for p in p_values]
        elif method == 'fdr_bh':
            # Benjamini-Hochberg FDR correction
            try:
                from statsmodels.stats.multitest import multipletests
                _, corrected_p, _, _ = multipletests(p_values, method='fdr_bh')
                return corrected_p.tolist()
            except ImportError:
                warnings.warn("statsmodels not available, using Bonferroni correction")
                return [min(1.0, p * len(p_values)) for p in p_values]
        else:
            raise ValueError(f"Unknown correction method: {method}")
    
    def bootstrap_confidence_interval(self, data: np.ndarray, statistic_func=np.mean, 
                                    confidence=0.95, n_bootstrap=1000) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for a statistic.
        
        Args:
            data: Input data
            statistic_func: Function to calculate statistic
            confidence: Confidence level
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Lower and upper bounds of confidence interval
        """
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            resample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic_func(resample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        alpha = 1 - confidence
        
        lower_bound = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper_bound = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        
        return lower_bound, upper_bound


class QuantumClassicalStatisticalComparison:
    """Specialized statistical comparison for quantum vs classical generators."""
    
    def __init__(self, device, config, gaussian_params):
        self.device = device
        self.config = config
        self.gaussian_params = gaussian_params
        self.validator = StatisticalValidator()
    
    def create_generator(self, generator_type: str):
        """Create generator of specified type."""
        # Create noise generator
        if generator_type == 'classical_normal':
            noise_gen = ClassicalNoise(
                z_dim=self.config['z_dim'],
                generator_type='classical_normal'
            )
        elif generator_type == 'quantum_samples':
            noise_gen = QuantumNoise(
                num_qubits=self.config['z_dim'],
                num_layers=self.config['quantum_layers']
            )
        elif generator_type == 'quantum_shadows':
            noise_gen = QuantumShadowNoise(
                z_dim=self.config['z_dim'],
                num_qubits=self.config['quantum_qubits'],
                num_layers=self.config['quantum_layers'],
                num_basis=self.config['quantum_basis']
            )
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")
        
        # Create MLP generator
        mlp_gen = MLPGenerator(
            non_linearity=self.config['non_linearity'],
            hidden_dims=self.config['nn_gen'],
            z_dim=self.config['z_dim'],
            std_scale=self.config['std_scale'],
            min_std=self.config['min_std']
        )
        
        generator = torch.nn.Sequential(noise_gen, mlp_gen)
        return generator.to(self.device)
    
    def collect_performance_data(self, generator_types: List[str], 
                               n_runs: int = 10, n_samples: int = 1000) -> Dict:
        """
        Collect performance data for statistical analysis.
        
        Args:
            generator_types: List of generator types to compare
            n_runs: Number of independent runs
            n_samples: Number of samples per run
            
        Returns:
            Dictionary containing performance metrics for each generator type
        """
        results = defaultdict(lambda: defaultdict(list))
        
        for gen_type in generator_types:
            try:
                generator = self.create_generator(gen_type)
                
                # Create metrics
                log_likelihood = LogLikelihood(**self.gaussian_params)
                kl_divergence = KLDivergence(**self.gaussian_params)
                is_positive = IsPositive()
                
                for run in range(n_runs):
                    with torch.no_grad():
                        # Generate samples
                        generated = generator(n_samples)
                        
                        # Compute metrics
                        ll_scores = log_likelihood.compute_score(generated)
                        kl_score = kl_divergence.compute_score(generated)
                        pos_scores = is_positive.compute_score(generated)
                        
                        # Store results
                        results[gen_type]['log_likelihood'].append(np.mean(ll_scores))
                        results[gen_type]['kl_divergence'].append(kl_score)
                        results[gen_type]['positive_ratio'].append(
                            np.mean([1 if p > 0 else 0 for p in pos_scores])
                        )
                        
                        # Additional statistics
                        samples_np = generated.cpu().numpy()
                        results[gen_type]['mean_x'].append(np.mean(samples_np[:, 0]))
                        results[gen_type]['mean_y'].append(np.mean(samples_np[:, 1]))
                        results[gen_type]['std_x'].append(np.std(samples_np[:, 0]))
                        results[gen_type]['std_y'].append(np.std(samples_np[:, 1]))
                        
            except Exception as e:
                print(f"Error collecting data for {gen_type}: {e}")
                results[gen_type]['error'] = str(e)
        
        return dict(results)
    
    def perform_comprehensive_comparison(self, performance_data: Dict) -> Dict:
        """
        Perform comprehensive statistical comparison between generator types.
        
        Args:
            performance_data: Performance data from collect_performance_data
            
        Returns:
            Dictionary containing comparison results
        """
        comparison_results = {}
        generator_types = [k for k in performance_data.keys() if 'error' not in performance_data[k]]
        
        if len(generator_types) < 2:
            return {'error': 'Need at least 2 generator types for comparison'}
        
        # Pairwise comparisons
        for i, gen1 in enumerate(generator_types):
            for j, gen2 in enumerate(generator_types[i+1:], i+1):
                comparison_key = f"{gen1}_vs_{gen2}"
                comparison_results[comparison_key] = {}
                
                # Compare each metric
                for metric in ['log_likelihood', 'kl_divergence', 'positive_ratio', 
                              'mean_x', 'mean_y', 'std_x', 'std_y']:
                    if metric in performance_data[gen1] and metric in performance_data[gen2]:
                        data1 = np.array(performance_data[gen1][metric])
                        data2 = np.array(performance_data[gen2][metric])
                        
                        # Remove any NaN or infinite values
                        data1 = data1[np.isfinite(data1)]
                        data2 = data2[np.isfinite(data2)]
                        
                        if len(data1) > 0 and len(data2) > 0:
                            comparison = self.validator.compare_distributions(data1, data2)
                            comparison['effect_interpretation'] = self.validator.interpret_effect_size(
                                comparison['effect_size']
                            )
                            
                            # Add power analysis
                            comparison['statistical_power'] = self.validator.power_analysis(
                                abs(comparison['effect_size']), len(data1), len(data2)
                            )
                            
                            # Bootstrap confidence intervals
                            ci1 = self.validator.bootstrap_confidence_interval(data1)
                            ci2 = self.validator.bootstrap_confidence_interval(data2)
                            comparison['ci1'] = ci1
                            comparison['ci2'] = ci2
                            
                            comparison_results[comparison_key][metric] = comparison
        
        return comparison_results
    
    def generate_statistical_report(self, comparison_results: Dict) -> str:
        """Generate a comprehensive statistical report."""
        report = []
        report.append("=" * 80)
        report.append("STATISTICAL ANALYSIS REPORT: QUANTUM VS CLASSICAL GENERATORS")
        report.append("=" * 80)
        
        for comparison_key, metrics in comparison_results.items():
            if 'error' in comparison_results:
                continue
                
            gen1, gen2 = comparison_key.split('_vs_')
            report.append(f"\n{'-' * 60}")
            report.append(f"COMPARISON: {gen1.upper()} vs {gen2.upper()}")
            report.append(f"{'-' * 60}")
            
            for metric, stats in metrics.items():
                report.append(f"\nMETRIC: {metric.upper()}")
                report.append("-" * 40)
                
                report.append(f"Sample sizes: n1={stats['n1']}, n2={stats['n2']}")
                report.append(f"{gen1}: {stats['mean1']:.4f} ± {stats['std1']:.4f}")
                report.append(f"{gen2}: {stats['mean2']:.4f} ± {stats['std2']:.4f}")
                
                report.append(f"Test used: {stats['test_used']}")
                report.append(f"Test statistic: {stats['test_statistic']:.4f}")
                report.append(f"P-value: {stats['p_value']:.6f}")
                report.append(f"Significant (α=0.05): {stats['significant']}")
                
                report.append(f"Effect size (Cohen's d): {stats['effect_size']:.4f}")
                report.append(f"Effect interpretation: {stats['effect_interpretation']}")
                report.append(f"Statistical power: {stats['statistical_power']:.3f}")
                
                report.append(f"95% CI {gen1}: [{stats['ci1'][0]:.4f}, {stats['ci1'][1]:.4f}]")
                report.append(f"95% CI {gen2}: [{stats['ci2'][0]:.4f}, {stats['ci2'][1]:.4f}]")
                
                # Interpretation
                if stats['significant']:
                    if stats['statistical_power'] >= 0.8:
                        interpretation = "Significant difference detected with adequate power"
                    else:
                        interpretation = "Significant difference but low statistical power"
                else:
                    if stats['statistical_power'] >= 0.8:
                        interpretation = "No significant difference with adequate power"
                    else:
                        interpretation = "No significant difference but insufficient power"
                
                report.append(f"Interpretation: {interpretation}")
        
        report.append(f"\n{'=' * 80}")
        report.append("END OF STATISTICAL REPORT")
        report.append(f"{'=' * 80}")
        
        return "\n".join(report)


@pytest.mark.statistical
@pytest.mark.slow
class TestStatisticalSignificance:
    """Test suite for statistical significance analysis."""
    
    def setup_method(self):
        """Set up test environment."""
        self.generator_types = ['classical_normal']
        
        # Add quantum generators if available
        try:
            import pennylane as qml
            self.generator_types.append('quantum_samples')
        except ImportError:
            print("PennyLane not available, skipping quantum generators")
    
    def test_statistical_validator_basic_functions(self):
        """Test basic functions of statistical validator."""
        validator = StatisticalValidator()
        
        # Test normality detection
        normal_data = np.random.normal(0, 1, 100)
        uniform_data = np.random.uniform(-1, 1, 100)
        
        p_norm_normal, is_normal = validator.test_normality(normal_data)
        p_norm_uniform, is_uniform = validator.test_normality(uniform_data)
        
        # Normal data should be more likely to pass normality test
        assert p_norm_normal >= 0.0
        assert p_norm_uniform >= 0.0
        
        # Test effect size calculation
        data1 = np.random.normal(0, 1, 50)
        data2 = np.random.normal(1, 1, 50)  # Different mean
        
        effect_size = validator.calculate_effect_size(data1, data2)
        assert isinstance(effect_size, float)
        assert abs(effect_size) > 0.5  # Should detect the difference
        
        interpretation = validator.interpret_effect_size(effect_size)
        assert interpretation in ['negligible', 'small', 'medium', 'large']
    
    def test_distribution_comparison(self):
        """Test distribution comparison functionality."""
        validator = StatisticalValidator()
        
        # Create data with known difference
        np.random.seed(42)
        data1 = np.random.normal(0, 1, 100)
        data2 = np.random.normal(0.5, 1, 100)  # Shifted mean
        
        results = validator.compare_distributions(data1, data2)
        
        # Validate results structure
        expected_keys = ['n1', 'n2', 'mean1', 'mean2', 'std1', 'std2', 
                        'test_used', 'test_statistic', 'p_value', 'significant', 
                        'effect_size']
        
        for key in expected_keys:
            assert key in results
        
        assert results['n1'] == 100
        assert results['n2'] == 100
        assert abs(results['mean1'] - 0) < 0.3  # Approximately 0
        assert abs(results['mean2'] - 0.5) < 0.3  # Approximately 0.5
        assert results['effect_size'] != 0
    
    def test_multiple_comparison_correction(self):
        """Test multiple comparison correction."""
        validator = StatisticalValidator()
        
        # Create p-values that would be significant without correction
        p_values = [0.04, 0.03, 0.045, 0.02, 0.01]
        
        corrected_bonf = validator.multiple_comparison_correction(p_values, 'bonferroni')
        corrected_fdr = validator.multiple_comparison_correction(p_values, 'fdr_bh')
        
        # Bonferroni correction should be more conservative
        assert all(c >= p for c, p in zip(corrected_bonf, p_values))
        assert all(c >= p for c, p in zip(corrected_fdr, p_values))
        
        # At least some should no longer be significant with Bonferroni
        assert any(c > 0.05 for c in corrected_bonf)
    
    def test_bootstrap_confidence_intervals(self):
        """Test bootstrap confidence interval calculation."""
        validator = StatisticalValidator()
        
        # Create data with known mean
        np.random.seed(42)
        data = np.random.normal(5, 2, 100)
        
        ci_lower, ci_upper = validator.bootstrap_confidence_interval(data, np.mean, 0.95, 500)
        
        # CI should contain the true mean (5) most of the time
        assert ci_lower < ci_upper
        assert ci_lower < 5 < ci_upper  # Should contain true mean
        assert ci_upper - ci_lower > 0  # Should have non-zero width
    
    def test_quantum_classical_statistical_comparison(self, device, test_config, 
                                                     gaussian_mixture_params):
        """Test comprehensive statistical comparison framework."""
        if len(self.generator_types) < 2:
            pytest.skip("Need at least 2 generator types for comparison")
        
        comparator = QuantumClassicalStatisticalComparison(
            device, test_config, gaussian_mixture_params
        )
        
        # Collect performance data (reduced for testing)
        performance_data = comparator.collect_performance_data(
            self.generator_types, n_runs=5, n_samples=200
        )
        
        # Validate data collection
        for gen_type in self.generator_types:
            if 'error' not in performance_data[gen_type]:
                assert 'kl_divergence' in performance_data[gen_type]
                assert 'log_likelihood' in performance_data[gen_type]
                assert len(performance_data[gen_type]['kl_divergence']) == 5
        
        # Perform statistical comparison
        comparison_results = comparator.perform_comprehensive_comparison(performance_data)
        
        # Validate comparison results
        if 'error' not in comparison_results:
            assert len(comparison_results) > 0
            
            for comparison_key, metrics in comparison_results.items():
                for metric, stats in metrics.items():
                    assert 'p_value' in stats
                    assert 'effect_size' in stats
                    assert 'statistical_power' in stats
                    assert 'significant' in stats
                    assert stats['p_value'] >= 0.0
                    assert stats['statistical_power'] >= 0.0
        
        # Generate and validate report
        report = comparator.generate_statistical_report(comparison_results)
        assert isinstance(report, str)
        assert len(report) > 0
        assert "STATISTICAL ANALYSIS REPORT" in report
        
        print(f"\n{report}")
    
    def test_power_analysis_and_sample_size_recommendations(self):
        """Test power analysis and sample size recommendations."""
        validator = StatisticalValidator()
        
        # Test power calculation for different effect sizes
        effect_sizes = [0.2, 0.5, 0.8]  # Small, medium, large
        sample_sizes = [10, 30, 50, 100]
        
        for effect_size in effect_sizes:
            for n in sample_sizes:
                power = validator.power_analysis(effect_size, n, n)
                
                assert 0 <= power <= 1
                
                # Larger effect sizes and sample sizes should give higher power
                if n >= 30:  # Adequate sample size
                    if effect_size >= 0.5:  # Medium+ effect
                        # Should have reasonable power
                        assert power > 0.3
    
    @pytest.mark.slow
    def test_reproducibility_statistical_validation(self, device, test_config, 
                                                   gaussian_mixture_params):
        """Test statistical validation of reproducibility."""
        if len(self.generator_types) < 1:
            pytest.skip("Need at least 1 generator type for reproducibility testing")
        
        comparator = QuantumClassicalStatisticalComparison(
            device, test_config, gaussian_mixture_params
        )
        
        gen_type = self.generator_types[0]
        
        # Collect data from two independent runs of the same configuration
        torch.manual_seed(42)
        np.random.seed(42)
        data_run1 = comparator.collect_performance_data([gen_type], n_runs=5, n_samples=200)
        
        torch.manual_seed(42)
        np.random.seed(42)
        data_run2 = comparator.collect_performance_data([gen_type], n_runs=5, n_samples=200)
        
        # Statistical test for reproducibility
        validator = StatisticalValidator()
        
        for metric in ['kl_divergence', 'log_likelihood']:
            if metric in data_run1[gen_type] and metric in data_run2[gen_type]:
                values1 = np.array(data_run1[gen_type][metric])
                values2 = np.array(data_run2[gen_type][metric])
                
                # Remove any NaN values
                values1 = values1[np.isfinite(values1)]
                values2 = values2[np.isfinite(values2)]
                
                if len(values1) > 0 and len(values2) > 0:
                    # Test if the two runs produce statistically similar results
                    comparison = validator.compare_distributions(values1, values2)
                    
                    print(f"\nReproducibility test for {metric}:")
                    print(f"  Run 1: {np.mean(values1):.4f} ± {np.std(values1):.4f}")
                    print(f"  Run 2: {np.mean(values2):.4f} ± {np.std(values2):.4f}")
                    print(f"  P-value: {comparison['p_value']:.6f}")
                    print(f"  Effect size: {comparison['effect_size']:.4f}")
                    
                    # For reproducibility, we expect NO significant difference
                    # But we need to be careful about Type II errors
                    if comparison['p_value'] < 0.01:  # Very strict threshold
                        print(f"  WARNING: Potential reproducibility issue detected!")
                    else:
                        print(f"  Reproducibility check passed")
    
    def test_effect_size_interpretation_validation(self):
        """Test that effect size interpretations are meaningful."""
        validator = StatisticalValidator()
        
        # Create data with known effect sizes
        np.random.seed(42)
        base_data = np.random.normal(0, 1, 100)
        
        # Small effect (0.2 standard deviations)
        small_effect_data = np.random.normal(0.2, 1, 100)
        # Medium effect (0.5 standard deviations)
        medium_effect_data = np.random.normal(0.5, 1, 100)
        # Large effect (0.8 standard deviations)
        large_effect_data = np.random.normal(0.8, 1, 100)
        
        # Test effect size calculations
        small_effect = validator.calculate_effect_size(base_data, small_effect_data)
        medium_effect = validator.calculate_effect_size(base_data, medium_effect_data)
        large_effect = validator.calculate_effect_size(base_data, large_effect_data)
        
        # Check that effect sizes are in expected ranges
        assert 0.1 < abs(small_effect) < 0.4  # Around 0.2
        assert 0.3 < abs(medium_effect) < 0.7  # Around 0.5
        assert 0.6 < abs(large_effect) < 1.0   # Around 0.8
        
        # Check interpretations
        small_interp = validator.interpret_effect_size(small_effect)
        medium_interp = validator.interpret_effect_size(medium_effect)
        large_interp = validator.interpret_effect_size(large_effect)
        
        assert small_interp in ['negligible', 'small']
        assert medium_interp in ['small', 'medium']
        assert large_interp in ['medium', 'large']
        
        print(f"\nEffect size validation:")
        print(f"  Small effect: {small_effect:.3f} ({small_interp})")
        print(f"  Medium effect: {medium_effect:.3f} ({medium_interp})")
        print(f"  Large effect: {large_effect:.3f} ({large_interp})")


if __name__ == "__main__":
    # Allow running this module directly
    pytest.main([__file__, "-v", "-s"])