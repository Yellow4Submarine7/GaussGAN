"""
Comprehensive Reporting System for GaussGAN Statistical Analysis
==============================================================

This module provides a complete reporting system that aggregates results from
all analysis components and generates publication-ready reports, visualizations,
and summaries for the quantum vs classical generator comparison.

Key Features:
- Multi-format report generation (PDF, HTML, Markdown, LaTeX)
- Interactive visualizations and dashboards
- Executive summaries and technical deep-dives
- Publication-ready figures and tables
- Automated insights and recommendations
- Export to various academic formats
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime
import json
import warnings

# Report generation
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

# Interactive plotting
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Interactive visualizations will be limited.")

# LaTeX and academic formatting
try:
    from jinja2 import Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    print("Warning: Jinja2 not available. Template-based report generation will be limited.")

# Statistical summary tables
from scipy import stats
import textwrap


class ReportGenerator:
    """
    Comprehensive report generator for statistical analysis results.
    
    This class creates publication-ready reports combining results from
    statistical analysis, convergence analysis, and stability analysis.
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path] = "docs/comprehensive_reports",
        report_title: str = "GaussGAN Quantum vs Classical Generator Analysis",
        author: str = "Statistical Analysis Framework",
        institution: str = ""
    ):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
            report_title: Title for the reports
            author: Author name
            institution: Institution name
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.report_title = report_title
        self.author = author
        self.institution = institution
        
        self.logger = logging.getLogger(__name__)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def generate_comprehensive_report(
        self,
        statistical_results: Dict[str, Any],
        convergence_results: Dict[str, Any],
        stability_results: Dict[str, Any],
        experiment_metadata: Dict[str, Any] = None
    ) -> Dict[str, Path]:
        """
        Generate comprehensive report in multiple formats.
        
        Args:
            statistical_results: Results from statistical analysis
            convergence_results: Results from convergence analysis
            stability_results: Results from stability analysis
            experiment_metadata: Additional experiment metadata
            
        Returns:
            Dictionary with paths to generated reports
        """
        self.logger.info("Generating comprehensive reports...")
        
        if experiment_metadata is None:
            experiment_metadata = {}
        
        # Combine all results
        combined_results = {
            'statistical': statistical_results,
            'convergence': convergence_results,
            'stability': stability_results,
            'metadata': experiment_metadata,
            'generation_timestamp': datetime.now().isoformat()
        }
        
        generated_files = {}
        
        # 1. Executive Summary (PDF)
        generated_files['executive_summary'] = self._generate_executive_summary(combined_results)
        
        # 2. Technical Report (PDF)
        generated_files['technical_report'] = self._generate_technical_report(combined_results)
        
        # 3. Interactive Dashboard (HTML)
        if PLOTLY_AVAILABLE:
            generated_files['interactive_dashboard'] = self._generate_interactive_dashboard(combined_results)
        
        # 4. Markdown Summary
        generated_files['markdown_summary'] = self._generate_markdown_summary(combined_results)
        
        # 5. Data Tables (CSV)
        generated_files['data_tables'] = self._export_data_tables(combined_results)
        
        # 6. Publication Figures
        generated_files['publication_figures'] = self._generate_publication_figures(combined_results)
        
        # 7. LaTeX Report (if template available)
        if JINJA2_AVAILABLE:
            generated_files['latex_report'] = self._generate_latex_report(combined_results)
        
        self.logger.info(f"Reports generated: {list(generated_files.keys())}")
        return generated_files
    
    def _generate_executive_summary(self, results: Dict[str, Any]) -> Path:
        """Generate executive summary PDF."""
        self.logger.info("Generating executive summary...")
        
        output_file = self.output_dir / 'executive_summary.pdf'
        
        with PdfPages(output_file) as pdf:
            # Page 1: Overview and Key Findings
            self._create_overview_page(results, pdf)
            
            # Page 2: Performance Comparison
            self._create_performance_summary_page(results, pdf)
            
            # Page 3: Recommendations
            self._create_recommendations_page(results, pdf)
        
        return output_file
    
    def _create_overview_page(self, results: Dict[str, Any], pdf: PdfPages):
        """Create overview page for executive summary."""
        fig = plt.figure(figsize=(8.5, 11))
        
        # Title section
        fig.suptitle(self.report_title, fontsize=20, fontweight='bold', y=0.95)
        
        # Create text content
        overview_text = self._generate_overview_text(results)
        
        # Add overview text
        ax_text = fig.add_subplot(111)
        ax_text.text(0.05, 0.85, overview_text, transform=ax_text.transAxes,
                    fontsize=12, verticalalignment='top', wrap=True)
        ax_text.axis('off')
        
        # Add key metrics summary
        key_metrics = self._extract_key_metrics(results)
        metrics_text = self._format_key_metrics(key_metrics)
        
        ax_text.text(0.05, 0.45, "KEY PERFORMANCE INDICATORS", 
                    transform=ax_text.transAxes,
                    fontsize=14, fontweight='bold')
        
        ax_text.text(0.05, 0.40, metrics_text, transform=ax_text.transAxes,
                    fontsize=11, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        # Add footer with generation info
        footer_text = f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by {self.author}"
        ax_text.text(0.05, 0.05, footer_text, transform=ax_text.transAxes,
                    fontsize=9, style='italic')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _generate_overview_text(self, results: Dict[str, Any]) -> str:
        """Generate overview text for the report."""
        metadata = results.get('metadata', {})
        statistical = results.get('statistical', {})
        
        # Extract basic statistics
        total_runs = metadata.get('total_runs', 'Unknown')
        generator_types = metadata.get('generator_types', ['Unknown'])
        metrics_analyzed = metadata.get('metrics_analyzed', ['Unknown'])
        
        overview = f"""
EXECUTIVE SUMMARY

This report presents a comprehensive statistical analysis comparing quantum and classical 
generators in the GaussGAN project. The analysis is based on {total_runs} experimental 
runs across {len(generator_types)} generator types: {', '.join(generator_types)}.

SCOPE OF ANALYSIS:
• Experimental runs: {total_runs}
• Generator types: {', '.join(generator_types)}
• Metrics analyzed: {', '.join(metrics_analyzed)}
• Analysis dimensions: Statistical significance, convergence patterns, stability assessment

METHODOLOGY:
• Statistical significance testing with multiple comparison correction
• Convergence pattern analysis with survival curves
• Multi-dimensional stability assessment
• Outlier detection and robustness evaluation
• Risk assessment for production deployment

The analysis employs rigorous statistical methods including bootstrap confidence intervals,
ANOVA-based variance decomposition, and ensemble outlier detection to ensure robust
and reliable conclusions.
        """.strip()
        
        return overview
    
    def _extract_key_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics for summary."""
        key_metrics = {}
        
        # Statistical results
        statistical = results.get('statistical', {})
        if 'summary_insights' in statistical:
            insights = statistical['summary_insights']
            key_metrics['best_performers'] = insights.get('best_performing_generators', {})
            key_metrics['significant_differences'] = len(insights.get('significant_differences', []))
        
        # Convergence results
        convergence = results.get('convergence', {})
        if 'speed_comparison' in convergence:
            # Calculate average convergence rates
            speed_comp = convergence['speed_comparison']
            avg_convergence_rates = {}
            for metric, comparisons in speed_comp.items():
                for gen_type, stats in comparisons.items():
                    if gen_type not in avg_convergence_rates:
                        avg_convergence_rates[gen_type] = []
                    avg_convergence_rates[gen_type].append(stats.get('convergence_rate', 0))
            
            for gen_type in avg_convergence_rates:
                avg_convergence_rates[gen_type] = np.mean(avg_convergence_rates[gen_type])
            
            key_metrics['convergence_rates'] = avg_convergence_rates
        
        # Stability results
        stability = results.get('stability', {})
        if 'risk_assessment' in stability:
            risk_levels = {}
            for gen_type, assessment in stability['risk_assessment'].items():
                risk_levels[gen_type] = assessment.get('risk_classification', 'Unknown')
            key_metrics['risk_levels'] = risk_levels
        
        return key_metrics
    
    def _format_key_metrics(self, key_metrics: Dict[str, Any]) -> str:
        """Format key metrics for display."""
        lines = []
        
        # Best performers
        if 'best_performers' in key_metrics:
            lines.append("BEST PERFORMING GENERATORS:")
            for metric, info in key_metrics['best_performers'].items():
                lines.append(f"  • {metric}: {info.get('generator', 'Unknown')}")
        
        # Convergence rates
        if 'convergence_rates' in key_metrics:
            lines.append("\nCONVERGENCE RATES:")
            for gen_type, rate in key_metrics['convergence_rates'].items():
                lines.append(f"  • {gen_type}: {rate:.1%}")
        
        # Risk levels
        if 'risk_levels' in key_metrics:
            lines.append("\nRISK CLASSIFICATIONS:")
            for gen_type, risk in key_metrics['risk_levels'].items():
                lines.append(f"  • {gen_type}: {risk}")
        
        # Significant differences
        if 'significant_differences' in key_metrics:
            lines.append(f"\nSTATISTICALLY SIGNIFICANT DIFFERENCES: {key_metrics['significant_differences']}")
        
        return '\n'.join(lines)
    
    def _create_performance_summary_page(self, results: Dict[str, Any], pdf: PdfPages):
        """Create performance summary page."""
        fig = plt.figure(figsize=(8.5, 11))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Performance Summary', fontsize=16, fontweight='bold', y=0.95)
        
        # Plot 1: Performance comparison (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_performance_overview(results, ax1)
        
        # Plot 2: Convergence comparison (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_convergence_overview(results, ax2)
        
        # Plot 3: Stability comparison (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_stability_overview(results, ax3)
        
        # Plot 4: Risk assessment (middle right)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_risk_overview(results, ax4)
        
        # Text summary (bottom)
        ax_summary = fig.add_subplot(gs[2, :])
        summary_text = self._generate_performance_summary_text(results)
        ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                       fontsize=10, verticalalignment='top', wrap=True)
        ax_summary.axis('off')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_overview(self, results: Dict[str, Any], ax):
        """Plot performance overview."""
        statistical = results.get('statistical', {})
        
        if 'descriptive_statistics' in statistical:
            desc_stats = statistical['descriptive_statistics']
            
            # Focus on KL Divergence as primary metric
            if 'KLDivergence' in desc_stats:
                generators = list(desc_stats['KLDivergence'].keys())
                means = [desc_stats['KLDivergence'][gen]['mean'] for gen in generators]
                stds = [desc_stats['KLDivergence'][gen]['std'] for gen in generators]
                
                bars = ax.bar(generators, means, yerr=stds, capsize=5, alpha=0.7)
                ax.set_title('KL Divergence Performance')
                ax.set_ylabel('KL Divergence')
                ax.tick_params(axis='x', rotation=45)
                
                # Color bars by performance (lower is better)
                colors = ['green' if m == min(means) else 'orange' if m < np.median(means) else 'red' 
                         for m in means]
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
        else:
            ax.text(0.5, 0.5, 'Performance data\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Performance Overview')
    
    def _plot_convergence_overview(self, results: Dict[str, Any], ax):
        """Plot convergence overview."""
        convergence = results.get('convergence', {})
        
        if 'speed_comparison' in convergence and 'KLDivergence' in convergence['speed_comparison']:
            comparison = convergence['speed_comparison']['KLDivergence']
            
            generators = list(comparison.keys())
            conv_rates = [comparison[gen]['convergence_rate'] for gen in generators]
            
            bars = ax.bar(generators, conv_rates, alpha=0.7)
            ax.set_title('Convergence Rates')
            ax.set_ylabel('Convergence Rate')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            
            # Color bars by convergence rate
            colors = ['green' if r > 0.8 else 'orange' if r > 0.6 else 'red' for r in conv_rates]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        else:
            ax.text(0.5, 0.5, 'Convergence data\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Convergence Overview')
    
    def _plot_stability_overview(self, results: Dict[str, Any], ax):
        """Plot stability overview."""
        stability = results.get('stability', {})
        
        if 'stability_metrics' in stability:
            # Calculate average consistency scores
            stability_metrics = stability['stability_metrics']
            
            avg_consistency = {}
            for gen_type, metrics in stability_metrics.items():
                consistency_scores = []
                for metric_name, metric_obj in metrics.items():
                    if hasattr(metric_obj, 'consistency_score'):
                        consistency_scores.append(metric_obj.consistency_score)
                    elif isinstance(metric_obj, dict) and 'consistency_score' in metric_obj:
                        consistency_scores.append(metric_obj['consistency_score'])
                
                if consistency_scores:
                    avg_consistency[gen_type] = np.mean(consistency_scores)
            
            if avg_consistency:
                generators = list(avg_consistency.keys())
                scores = list(avg_consistency.values())
                
                bars = ax.bar(generators, scores, alpha=0.7)
                ax.set_title('Stability Scores')
                ax.set_ylabel('Consistency Score')
                ax.set_ylim(0, 1)
                ax.tick_params(axis='x', rotation=45)
                
                # Color bars by stability
                colors = ['green' if s > 0.8 else 'orange' if s > 0.6 else 'red' for s in scores]
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
            else:
                ax.text(0.5, 0.5, 'Stability data\nnot available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Stability Overview')
        else:
            ax.text(0.5, 0.5, 'Stability data\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Stability Overview')
    
    def _plot_risk_overview(self, results: Dict[str, Any], ax):
        """Plot risk overview."""
        stability = results.get('stability', {})
        
        if 'risk_assessment' in stability:
            risk_assessment = stability['risk_assessment']
            
            generators = list(risk_assessment.keys())
            risk_scores = [risk_assessment[gen]['overall_risk_score'] for gen in generators]
            
            bars = ax.bar(generators, risk_scores, alpha=0.7)
            ax.set_title('Risk Assessment')
            ax.set_ylabel('Risk Score')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            
            # Color bars by risk level (lower is better)
            colors = ['green' if r < 0.3 else 'orange' if r < 0.6 else 'red' for r in risk_scores]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        else:
            ax.text(0.5, 0.5, 'Risk data\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Risk Overview')
    
    def _generate_performance_summary_text(self, results: Dict[str, Any]) -> str:
        """Generate performance summary text."""
        lines = []
        
        lines.append("PERFORMANCE ANALYSIS SUMMARY")
        lines.append("=" * 40)
        
        # Statistical insights
        statistical = results.get('statistical', {})
        if 'summary_insights' in statistical:
            insights = statistical['summary_insights']
            
            if 'best_performing_generators' in insights:
                lines.append("\nBest Performing Generators:")
                for metric, info in insights['best_performing_generators'].items():
                    lines.append(f"  • {metric}: {info.get('generator', 'Unknown')}")
            
            if 'significant_differences' in insights:
                sig_count = len(insights['significant_differences'])
                lines.append(f"\nStatistically significant differences found: {sig_count}")
        
        # Convergence insights
        convergence = results.get('convergence', {})
        if 'recommendations' in convergence:
            recs = convergence['recommendations']
            if 'fastest_generator' in recs:
                lines.append(f"\nFastest converging: {recs['fastest_generator']}")
        
        # Stability insights
        stability = results.get('stability', {})
        if 'recommendations' in stability:
            recs = stability['recommendations']
            if 'most_stable_generator' in recs:
                lines.append(f"\nMost stable: {recs['most_stable_generator']}")
        
        return '\n'.join(lines)
    
    def _create_recommendations_page(self, results: Dict[str, Any], pdf: PdfPages):
        """Create recommendations page."""
        fig = plt.figure(figsize=(8.5, 11))
        
        # Title
        fig.suptitle('Recommendations and Conclusions', fontsize=16, fontweight='bold', y=0.95)
        
        # Generate recommendations text
        recommendations_text = self._generate_comprehensive_recommendations(results)
        
        # Add recommendations
        ax_text = fig.add_subplot(111)
        ax_text.text(0.05, 0.85, recommendations_text, transform=ax_text.transAxes,
                    fontsize=11, verticalalignment='top', wrap=True)
        ax_text.axis('off')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _generate_comprehensive_recommendations(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive recommendations."""
        lines = []
        
        lines.append("COMPREHENSIVE RECOMMENDATIONS")
        lines.append("=" * 50)
        
        # Performance recommendations
        lines.append("\n1. PERFORMANCE OPTIMIZATION")
        statistical = results.get('statistical', {})
        if 'summary_insights' in statistical:
            insights = statistical['summary_insights']
            best_performers = insights.get('best_performing_generators', {})
            
            if best_performers:
                lines.append("\nRecommended generators by metric:")
                for metric, info in best_performers.items():
                    lines.append(f"  • {metric}: {info.get('generator', 'Unknown')}")
                
                # Find most frequently recommended generator
                generator_counts = {}
                for info in best_performers.values():
                    gen = info.get('generator', 'Unknown')
                    generator_counts[gen] = generator_counts.get(gen, 0) + 1
                
                if generator_counts:
                    top_generator = max(generator_counts.keys(), key=lambda x: generator_counts[x])
                    lines.append(f"\nOverall recommended generator: {top_generator}")
                    lines.append(f"(Best performance in {generator_counts[top_generator]} out of {len(best_performers)} metrics)")
        
        # Convergence recommendations
        lines.append("\n2. CONVERGENCE OPTIMIZATION")
        convergence = results.get('convergence', {})
        if 'recommendations' in convergence:
            for key, rec in convergence['recommendations'].items():
                lines.append(f"  • {key.replace('_', ' ').title()}: {rec}")
        
        # Stability recommendations
        lines.append("\n3. STABILITY AND RELIABILITY")
        stability = results.get('stability', {})
        if 'recommendations' in stability:
            for key, rec in stability['recommendations'].items():
                lines.append(f"  • {key.replace('_', ' ').title()}: {rec}")
        
        # Risk management
        if 'risk_assessment' in stability:
            lines.append("\n4. RISK MANAGEMENT")
            risk_assessment = stability['risk_assessment']
            
            low_risk_gens = [gen for gen, assess in risk_assessment.items() 
                            if assess.get('risk_classification') == 'Low Risk']
            high_risk_gens = [gen for gen, assess in risk_assessment.items() 
                             if assess.get('risk_classification') in ['High Risk', 'Very High Risk']]
            
            if low_risk_gens:
                lines.append(f"  • Low-risk generators for production: {', '.join(low_risk_gens)}")
            if high_risk_gens:
                lines.append(f"  • High-risk generators requiring caution: {', '.join(high_risk_gens)}")
        
        # General recommendations
        lines.append("\n5. GENERAL RECOMMENDATIONS")
        lines.append("  • Conduct additional runs for generators showing high variance")
        lines.append("  • Monitor convergence patterns during training for early stopping")
        lines.append("  • Implement ensemble methods combining multiple generator types")
        lines.append("  • Regular performance validation with out-of-sample data")
        
        lines.append("\n6. FUTURE WORK")
        lines.append("  • Investigate hyperparameter sensitivity for quantum generators")
        lines.append("  • Explore hybrid quantum-classical approaches")
        lines.append("  • Develop adaptive training strategies based on convergence patterns")
        lines.append("  • Study performance scaling with problem complexity")
        
        return '\n'.join(lines)
    
    def _generate_technical_report(self, results: Dict[str, Any]) -> Path:
        """Generate detailed technical report."""
        self.logger.info("Generating technical report...")
        
        output_file = self.output_dir / 'technical_report.pdf'
        
        with PdfPages(output_file) as pdf:
            # Title page
            self._create_title_page(pdf)
            
            # Statistical analysis section
            self._create_statistical_analysis_section(results, pdf)
            
            # Convergence analysis section
            self._create_convergence_analysis_section(results, pdf)
            
            # Stability analysis section
            self._create_stability_analysis_section(results, pdf)
            
            # Detailed results tables
            self._create_detailed_results_tables(results, pdf)
            
            # Methodology appendix
            self._create_methodology_appendix(pdf)
        
        return output_file
    
    def _create_title_page(self, pdf: PdfPages):
        """Create title page for technical report."""
        fig = plt.figure(figsize=(8.5, 11))
        
        # Title
        fig.text(0.5, 0.7, self.report_title, ha='center', va='center',
                fontsize=24, fontweight='bold')
        
        # Subtitle
        fig.text(0.5, 0.6, 'Technical Report', ha='center', va='center',
                fontsize=18)
        
        # Author and date
        fig.text(0.5, 0.4, f'Author: {self.author}', ha='center', va='center',
                fontsize=14)
        
        if self.institution:
            fig.text(0.5, 0.35, f'Institution: {self.institution}', ha='center', va='center',
                    fontsize=12)
        
        fig.text(0.5, 0.25, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                ha='center', va='center', fontsize=12)
        
        # Abstract box
        abstract_text = """
        This technical report presents a comprehensive statistical analysis of quantum versus 
        classical generators in the GaussGAN project. The analysis employs rigorous statistical 
        methods including significance testing, convergence analysis, and stability assessment 
        to provide evidence-based recommendations for generator selection and optimization.
        """
        
        fig.text(0.5, 0.15, abstract_text.strip(), ha='center', va='center',
                fontsize=11, wrap=True,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        fig.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_statistical_analysis_section(self, results: Dict[str, Any], pdf: PdfPages):
        """Create statistical analysis section."""
        # This would include detailed statistical plots and tables
        pass  # Implementation would continue here
    
    def _create_convergence_analysis_section(self, results: Dict[str, Any], pdf: PdfPages):
        """Create convergence analysis section."""
        # This would include convergence plots and analysis
        pass  # Implementation would continue here
    
    def _create_stability_analysis_section(self, results: Dict[str, Any], pdf: PdfPages):
        """Create stability analysis section."""
        # This would include stability plots and analysis
        pass  # Implementation would continue here
    
    def _create_detailed_results_tables(self, results: Dict[str, Any], pdf: PdfPages):
        """Create detailed results tables."""
        # This would include comprehensive data tables
        pass  # Implementation would continue here
    
    def _create_methodology_appendix(self, pdf: PdfPages):
        """Create methodology appendix."""
        # This would include detailed methodology descriptions
        pass  # Implementation would continue here
    
    def _generate_interactive_dashboard(self, results: Dict[str, Any]) -> Path:
        """Generate interactive HTML dashboard."""
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available, skipping interactive dashboard")
            return None
        
        self.logger.info("Generating interactive dashboard...")
        
        # Create interactive plots
        figures = self._create_interactive_plots(results)
        
        # Create HTML dashboard
        html_content = self._create_dashboard_html(figures, results)
        
        output_file = self.output_dir / 'interactive_dashboard.html'
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        return output_file
    
    def _create_interactive_plots(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Create interactive plots using Plotly."""
        figures = {}
        
        # Performance comparison plot
        if 'statistical' in results and 'descriptive_statistics' in results['statistical']:
            desc_stats = results['statistical']['descriptive_statistics']
            
            if 'KLDivergence' in desc_stats:
                generators = list(desc_stats['KLDivergence'].keys())
                means = [desc_stats['KLDivergence'][gen]['mean'] for gen in generators]
                stds = [desc_stats['KLDivergence'][gen]['std'] for gen in generators]
                
                fig = go.Figure(data=[
                    go.Bar(x=generators, y=means, error_y=dict(type='data', array=stds),
                          name='KL Divergence')
                ])
                fig.update_layout(title='Performance Comparison - KL Divergence',
                                xaxis_title='Generator Type',
                                yaxis_title='KL Divergence')
                
                figures['performance'] = pyo.plot(fig, output_type='div', include_plotlyjs=False)
        
        return figures
    
    def _create_dashboard_html(self, figures: Dict[str, str], results: Dict[str, Any]) -> str:
        """Create HTML dashboard content."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>GaussGAN Analysis Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; margin-bottom: 20px; }
                .section { margin-bottom: 30px; }
                .plot-container { margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
                <p>Interactive dashboard for quantum vs classical generator analysis</p>
                <p>Generated: {timestamp}</p>
            </div>
            
            <div class="section">
                <h2>Performance Analysis</h2>
                <div class="plot-container">
                    {performance_plot}
                </div>
            </div>
            
            <div class="section">
                <h2>Summary Insights</h2>
                <div>{summary_insights}</div>
            </div>
        </body>
        </html>
        """
        
        summary_insights = self._format_insights_for_html(results)
        
        return html_template.format(
            title=self.report_title,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            performance_plot=figures.get('performance', '<p>Performance plot not available</p>'),
            summary_insights=summary_insights
        )
    
    def _format_insights_for_html(self, results: Dict[str, Any]) -> str:
        """Format insights for HTML display."""
        html_lines = []
        
        statistical = results.get('statistical', {})
        if 'summary_insights' in statistical:
            insights = statistical['summary_insights']
            
            if 'best_performing_generators' in insights:
                html_lines.append("<h3>Best Performing Generators</h3>")
                html_lines.append("<ul>")
                for metric, info in insights['best_performing_generators'].items():
                    html_lines.append(f"<li><strong>{metric}:</strong> {info.get('generator', 'Unknown')}</li>")
                html_lines.append("</ul>")
        
        return '\n'.join(html_lines)
    
    def _generate_markdown_summary(self, results: Dict[str, Any]) -> Path:
        """Generate markdown summary report."""
        self.logger.info("Generating markdown summary...")
        
        output_file = self.output_dir / 'analysis_summary.md'
        
        markdown_content = self._create_markdown_content(results)
        
        with open(output_file, 'w') as f:
            f.write(markdown_content)
        
        return output_file
    
    def _create_markdown_content(self, results: Dict[str, Any]) -> str:
        """Create markdown content."""
        lines = []
        
        # Header
        lines.append(f"# {self.report_title}")
        lines.append("")
        lines.append(f"**Author:** {self.author}")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(self._generate_overview_text(results))
        lines.append("")
        
        # Key Findings
        lines.append("## Key Findings")
        lines.append("")
        
        statistical = results.get('statistical', {})
        if 'summary_insights' in statistical:
            insights = statistical['summary_insights']
            
            if 'best_performing_generators' in insights:
                lines.append("### Best Performing Generators")
                for metric, info in insights['best_performing_generators'].items():
                    lines.append(f"- **{metric}:** {info.get('generator', 'Unknown')}")
                lines.append("")
        
        # Recommendations
        lines.append("## Recommendations")
        lines.append("")
        lines.append(self._generate_comprehensive_recommendations(results))
        
        return '\n'.join(lines)
    
    def _export_data_tables(self, results: Dict[str, Any]) -> Path:
        """Export data tables to CSV files."""
        self.logger.info("Exporting data tables...")
        
        tables_dir = self.output_dir / 'data_tables'
        tables_dir.mkdir(exist_ok=True)
        
        # Export statistical results
        statistical = results.get('statistical', {})
        if 'descriptive_statistics' in statistical:
            self._export_descriptive_statistics(statistical['descriptive_statistics'], tables_dir)
        
        # Export convergence results
        convergence = results.get('convergence', {})
        if 'speed_comparison' in convergence:
            self._export_convergence_data(convergence['speed_comparison'], tables_dir)
        
        # Export stability results
        stability = results.get('stability', {})
        if 'stability_metrics' in stability:
            self._export_stability_data(stability['stability_metrics'], tables_dir)
        
        return tables_dir
    
    def _export_descriptive_statistics(self, desc_stats: Dict[str, Any], output_dir: Path):
        """Export descriptive statistics to CSV."""
        for metric, generators in desc_stats.items():
            data = []
            for gen_type, stats in generators.items():
                row = {'generator_type': gen_type, 'metric': metric}
                row.update(stats)
                data.append(row)
            
            df = pd.DataFrame(data)
            output_file = output_dir / f'descriptive_statistics_{metric}.csv'
            df.to_csv(output_file, index=False)
    
    def _export_convergence_data(self, speed_comparison: Dict[str, Any], output_dir: Path):
        """Export convergence data to CSV."""
        data = []
        for metric, generators in speed_comparison.items():
            for gen_type, stats in generators.items():
                row = {'metric': metric, 'generator_type': gen_type}
                row.update(stats)
                data.append(row)
        
        df = pd.DataFrame(data)
        output_file = output_dir / 'convergence_analysis.csv'
        df.to_csv(output_file, index=False)
    
    def _export_stability_data(self, stability_metrics: Dict[str, Any], output_dir: Path):
        """Export stability data to CSV."""
        data = []
        for gen_type, metrics in stability_metrics.items():
            for metric_name, stability_obj in metrics.items():
                row = {'generator_type': gen_type, 'metric': metric_name}
                
                # Handle both object and dictionary formats
                if hasattr(stability_obj, '__dict__'):
                    row.update(stability_obj.__dict__)
                elif isinstance(stability_obj, dict):
                    row.update(stability_obj)
                
                data.append(row)
        
        df = pd.DataFrame(data)
        output_file = output_dir / 'stability_analysis.csv'
        df.to_csv(output_file, index=False)
    
    def _generate_publication_figures(self, results: Dict[str, Any]) -> Path:
        """Generate publication-ready figures."""
        self.logger.info("Generating publication figures...")
        
        figures_dir = self.output_dir / 'publication_figures'
        figures_dir.mkdir(exist_ok=True)
        
        # Set publication style
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 12,
            'axes.linewidth': 1.5,
            'axes.labelweight': 'bold',
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
        
        # Create publication figures
        self._create_main_results_figure(results, figures_dir)
        self._create_convergence_figure(results, figures_dir)
        self._create_stability_figure(results, figures_dir)
        
        return figures_dir
    
    def _create_main_results_figure(self, results: Dict[str, Any], output_dir: Path):
        """Create main results figure for publication."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Performance comparison
        self._plot_performance_overview(results, axes[0, 0])
        
        # Convergence comparison
        self._plot_convergence_overview(results, axes[0, 1])
        
        # Stability comparison
        self._plot_stability_overview(results, axes[1, 0])
        
        # Risk assessment
        self._plot_risk_overview(results, axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(output_dir / 'main_results.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'main_results.pdf', bbox_inches='tight')
        plt.close()
    
    def _create_convergence_figure(self, results: Dict[str, Any], output_dir: Path):
        """Create detailed convergence figure."""
        # Implementation would create detailed convergence plots
        pass
    
    def _create_stability_figure(self, results: Dict[str, Any], output_dir: Path):
        """Create detailed stability figure."""
        # Implementation would create detailed stability plots
        pass
    
    def _generate_latex_report(self, results: Dict[str, Any]) -> Path:
        """Generate LaTeX report."""
        if not JINJA2_AVAILABLE:
            self.logger.warning("Jinja2 not available, skipping LaTeX report")
            return None
        
        self.logger.info("Generating LaTeX report...")
        
        # LaTeX template would be defined here
        latex_template = """
        \\documentclass{article}
        \\usepackage[utf8]{inputenc}
        \\usepackage{graphicx}
        \\usepackage{booktabs}
        
        \\title{ {{- title -}} }
        \\author{ {{- author -}} }
        \\date{ {{- date -}} }
        
        \\begin{document}
        \\maketitle
        
        \\section{Executive Summary}
        {{- summary -}}
        
        \\section{Results}
        {{- results -}}
        
        \\section{Conclusions}
        {{- conclusions -}}
        
        \\end{document}
        """
        
        template = Template(latex_template)
        
        latex_content = template.render(
            title=self.report_title,
            author=self.author,
            date=datetime.now().strftime('%Y-%m-%d'),
            summary=self._generate_overview_text(results),
            results="Detailed results would be inserted here",
            conclusions=self._generate_comprehensive_recommendations(results)
        )
        
        output_file = self.output_dir / 'technical_report.tex'
        with open(output_file, 'w') as f:
            f.write(latex_content)
        
        return output_file


def demonstrate_reporting_system():
    """Demonstrate the comprehensive reporting system."""
    print("Comprehensive Reporting System Demo")
    print("=" * 50)
    
    # Create sample results
    sample_results = {
        'statistical': {
            'summary_insights': {
                'best_performing_generators': {
                    'KLDivergence': {'generator': 'classical_normal', 'value': 0.12},
                    'LogLikelihood': {'generator': 'classical_normal', 'value': -2.3}
                },
                'significant_differences': [
                    {'metric': 'KLDivergence', 'generator1': 'classical_normal', 
                     'generator2': 'quantum_samples', 'p_value': 0.001}
                ]
            },
            'descriptive_statistics': {
                'KLDivergence': {
                    'classical_normal': {'mean': 0.12, 'std': 0.03, 'count': 20},
                    'quantum_samples': {'mean': 0.22, 'std': 0.06, 'count': 20}
                }
            }
        },
        'convergence': {
            'speed_comparison': {
                'KLDivergence': {
                    'classical_normal': {'convergence_rate': 0.95, 'mean_epochs_to_convergence': 25},
                    'quantum_samples': {'convergence_rate': 0.75, 'mean_epochs_to_convergence': 35}
                }
            },
            'recommendations': {
                'fastest_generator': 'classical_normal shows fastest convergence'
            }
        },
        'stability': {
            'risk_assessment': {
                'classical_normal': {'overall_risk_score': 0.2, 'risk_classification': 'Low Risk'},
                'quantum_samples': {'overall_risk_score': 0.6, 'risk_classification': 'Medium Risk'}
            },
            'recommendations': {
                'most_stable_generator': 'classical_normal shows highest stability'
            }
        },
        'metadata': {
            'total_runs': 80,
            'generator_types': ['classical_normal', 'quantum_samples'],
            'metrics_analyzed': ['KLDivergence', 'LogLikelihood']
        }
    }
    
    # Initialize report generator
    generator = ReportGenerator(
        output_dir="docs/demo_reports",
        report_title="Demo: GaussGAN Analysis",
        author="Demo User"
    )
    
    # Generate reports
    generated_files = generator.generate_comprehensive_report(
        statistical_results=sample_results['statistical'],
        convergence_results=sample_results['convergence'],
        stability_results=sample_results['stability'],
        experiment_metadata=sample_results['metadata']
    )
    
    print("✓ Report generator initialized")
    print("✓ Sample results created")
    print(f"✓ Generated {len(generated_files)} report types")
    
    print("\nGenerated files:")
    for report_type, file_path in generated_files.items():
        if file_path:
            print(f"  • {report_type}: {file_path}")
    
    return generator


if __name__ == "__main__":
    generator = demonstrate_reporting_system()