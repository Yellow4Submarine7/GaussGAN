#!/usr/bin/env python
"""
Comprehensive test runner and coverage analysis tool for GaussGAN.
Provides automated test execution with performance tracking and reporting.
"""

import argparse
import subprocess
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import shutil

try:
    import pytest
    import coverage
except ImportError:
    print("Please install pytest and coverage: pip install pytest coverage")
    sys.exit(1)


class TestRunner:
    """Comprehensive test runner with coverage analysis and reporting."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize test runner.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent.parent
        self.test_dir = self.project_root / "docs" / "tests"
        self.source_dir = self.project_root / "source"
        self.reports_dir = self.project_root / "test_reports"
        
        # Create reports directory
        self.reports_dir.mkdir(exist_ok=True)
        
        # Test categories and their configurations
        self.test_categories = {
            'unit': {
                'path': 'unit/',
                'markers': 'not slow and not quantum',
                'timeout': 300,
                'description': 'Fast unit tests for individual components'
            },
            'integration': {
                'path': 'integration/',
                'markers': 'integration and not slow',
                'timeout': 600,
                'description': 'Integration tests for component interactions'
            },
            'quantum': {
                'path': '.',
                'markers': 'quantum and not slow',
                'timeout': 900,
                'description': 'Quantum component tests'
            },
            'performance': {
                'path': 'performance/',
                'markers': 'performance',
                'timeout': 1800,
                'description': 'Performance benchmarking tests'
            },
            'statistical': {
                'path': 'statistical/',
                'markers': 'statistical',
                'timeout': 1200,
                'description': 'Statistical validation tests'
            },
            'regression': {
                'path': 'regression/',
                'markers': 'regression',
                'timeout': 900,
                'description': 'Performance regression tests'
            },
            'all': {
                'path': '.',
                'markers': '',
                'timeout': 3600,
                'description': 'Complete test suite'
            }
        }
        
    def run_tests(
        self,
        category: str = 'unit',
        coverage: bool = True,
        parallel: bool = False,
        verbose: bool = True,
        save_baselines: bool = False,
        strict: bool = False
    ) -> Dict[str, Any]:
        """
        Run tests for specified category.
        
        Args:
            category: Test category to run
            coverage: Enable coverage analysis
            parallel: Enable parallel test execution
            verbose: Enable verbose output
            save_baselines: Save performance baselines
            strict: Use strict mode (fail on warnings)
            
        Returns:
            Dictionary containing test results and metrics
        """
        if category not in self.test_categories:
            available = ', '.join(self.test_categories.keys())
            raise ValueError(f"Unknown test category: {category}. Available: {available}")
        
        config = self.test_categories[category]
        test_path = self.test_dir / config['path']
        
        print(f"\n{'=' * 80}")
        print(f"RUNNING {category.upper()} TESTS")
        print(f"{'=' * 80}")
        print(f"Description: {config['description']}")
        print(f"Test path: {test_path}")
        print(f"Timeout: {config['timeout']} seconds")
        print(f"{'=' * 80}\n")
        
        # Build pytest command
        cmd = ['python', '-m', 'pytest', str(test_path)]
        
        # Add markers
        if config['markers']:
            cmd.extend(['-m', config['markers']])
        
        # Add coverage
        if coverage:
            cmd.extend([
                '--cov=source',
                '--cov-report=html:htmlcov',
                '--cov-report=xml:coverage.xml',
                '--cov-report=term-missing',
                '--cov-branch'
            ])
        
        # Add parallelization
        if parallel:
            try:
                import pytest_xdist
                cmd.extend(['-n', 'auto'])
            except ImportError:
                print("Warning: pytest-xdist not available, running sequentially")
        
        # Add verbosity
        if verbose:
            cmd.append('-v')
        
        # Add strict mode
        if strict:
            cmd.extend(['--strict-markers', '--strict-config'])
        
        # Add timeout
        cmd.extend(['--timeout', str(config['timeout'])])
        
        # Add JUnit XML output
        xml_report = self.reports_dir / f'{category}_results.xml'
        cmd.extend(['--junit-xml', str(xml_report)])
        
        # Add JSON report if pytest-json-report is available
        try:
            import pytest_json_report
            json_report = self.reports_dir / f'{category}_results.json'
            cmd.extend(['--json-report', '--json-report-file', str(json_report)])
        except ImportError:
            json_report = None
        
        # Add baseline saving for performance tests
        if save_baselines and category in ['performance', 'regression']:
            baseline_dir = self.reports_dir / 'baselines'
            baseline_dir.mkdir(exist_ok=True)
            cmd.extend(['--save-baselines', '--baseline-dir', str(baseline_dir)])
        
        # Record execution details
        start_time = time.time()
        execution_details = {
            'category': category,
            'command': ' '.join(cmd),
            'start_time': datetime.now().isoformat(),
            'config': config.copy(),
            'options': {
                'coverage': coverage,
                'parallel': parallel,
                'verbose': verbose,
                'save_baselines': save_baselines,
                'strict': strict
            }
        }
        
        # Run tests
        print(f"Executing: {' '.join(cmd)}")
        print("-" * 80)
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=False,  # Show output in real-time
                timeout=config['timeout'] + 60  # Add buffer to pytest timeout
            )
            
            exit_code = result.returncode
            success = exit_code == 0
            
        except subprocess.TimeoutExpired:
            print(f"\nERROR: Tests timed out after {config['timeout']} seconds")
            exit_code = -1
            success = False
        except KeyboardInterrupt:
            print("\nTests interrupted by user")
            exit_code = -2
            success = False
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Collect results
        execution_details.update({
            'end_time': datetime.now().isoformat(),
            'duration_seconds': duration,
            'exit_code': exit_code,
            'success': success
        })
        
        # Parse results from XML/JSON if available
        if xml_report.exists():
            execution_details['xml_report'] = str(xml_report)
            
        if json_report and json_report.exists():
            try:
                with open(json_report) as f:
                    test_results = json.load(f)
                    execution_details.update({
                        'json_report': str(json_report),
                        'tests_collected': test_results.get('summary', {}).get('total', 0),
                        'tests_passed': test_results.get('summary', {}).get('passed', 0),
                        'tests_failed': test_results.get('summary', {}).get('failed', 0),
                        'tests_skipped': test_results.get('summary', {}).get('skipped', 0),
                    })
            except Exception as e:
                print(f"Warning: Could not parse JSON report: {e}")
        
        # Parse coverage if enabled
        if coverage and Path('coverage.xml').exists():
            execution_details['coverage_report'] = str(Path('coverage.xml').absolute())
            execution_details['coverage_html'] = str(Path('htmlcov').absolute())
        
        # Save execution summary
        summary_file = self.reports_dir / f'{category}_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(execution_details, f, indent=2)
        
        print(f"\n{'=' * 80}")
        print(f"{category.upper()} TESTS COMPLETED")
        print(f"{'=' * 80}")
        print(f"Success: {success}")
        print(f"Exit code: {exit_code}")
        print(f"Duration: {duration:.1f} seconds")
        print(f"Summary saved to: {summary_file}")
        
        if coverage and Path('htmlcov/index.html').exists():
            print(f"Coverage report: file://{Path('htmlcov/index.html').absolute()}")
        
        print(f"{'=' * 80}\n")
        
        return execution_details
    
    def run_test_suite(
        self,
        categories: List[str] = None,
        fast_only: bool = False,
        coverage: bool = True,
        parallel: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run multiple test categories in sequence.
        
        Args:
            categories: List of categories to run (default: ['unit', 'integration'])
            fast_only: Only run fast tests (exclude slow markers)
            coverage: Enable coverage analysis
            parallel: Enable parallel execution
            
        Returns:
            Dictionary of results for each category
        """
        if categories is None:
            if fast_only:
                categories = ['unit']
            else:
                categories = ['unit', 'integration', 'statistical']
        
        results = {}
        total_start = time.time()
        
        print(f"\n🚀 STARTING TEST SUITE")
        print(f"Categories: {', '.join(categories)}")
        print(f"Fast only: {fast_only}")
        print(f"Coverage: {coverage}")
        print(f"Parallel: {parallel}")
        
        for category in categories:
            try:
                result = self.run_tests(
                    category=category,
                    coverage=coverage,
                    parallel=parallel
                )
                results[category] = result
                
                if not result['success']:
                    print(f"❌ {category} tests failed, continuing with remaining tests...")
                else:
                    print(f"✅ {category} tests passed")
                    
            except Exception as e:
                print(f"❌ Error running {category} tests: {e}")
                results[category] = {
                    'success': False,
                    'error': str(e),
                    'category': category
                }
        
        total_duration = time.time() - total_start
        
        # Generate summary
        summary = {
            'total_duration': total_duration,
            'categories_run': len(categories),
            'categories_passed': sum(1 for r in results.values() if r.get('success', False)),
            'categories_failed': sum(1 for r in results.values() if not r.get('success', False)),
            'results': results
        }
        
        # Save overall summary
        summary_file = self.reports_dir / 'test_suite_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n🏁 TEST SUITE COMPLETED")
        print(f"Total duration: {total_duration:.1f} seconds")
        print(f"Categories passed: {summary['categories_passed']}/{summary['categories_run']}")
        print(f"Summary saved to: {summary_file}")
        
        return summary
    
    def generate_coverage_report(self) -> Dict[str, Any]:
        """Generate detailed coverage analysis report."""
        print("\n📊 GENERATING COVERAGE REPORT")
        
        # Initialize coverage
        cov = coverage.Coverage(source=['source'])
        
        # Check if coverage data exists
        if not Path('.coverage').exists():
            print("No coverage data found. Run tests with --coverage first.")
            return {'error': 'No coverage data'}
        
        cov.load()
        
        # Generate reports
        html_dir = self.reports_dir / 'coverage_html'
        html_dir.mkdir(exist_ok=True)
        
        xml_file = self.reports_dir / 'coverage.xml'
        json_file = self.reports_dir / 'coverage.json'
        
        # HTML report
        cov.html_report(directory=str(html_dir))
        
        # XML report
        cov.xml_report(outfile=str(xml_file))
        
        # JSON report (custom)
        coverage_data = {}
        
        # Get overall coverage
        total_coverage = cov.report(show_missing=False, file=None)
        
        # Get file-level coverage
        analysis = cov.get_data()
        
        file_coverage = {}
        for filename in analysis.measured_files():
            if filename.startswith(str(self.source_dir)):
                rel_path = Path(filename).relative_to(self.source_dir)
                
                # Get coverage analysis for this file
                try:
                    file_analysis = cov.analysis2(filename)
                    executed_lines = len(file_analysis.executed)
                    missing_lines = len(file_analysis.missing)
                    total_lines = executed_lines + missing_lines
                    
                    file_coverage[str(rel_path)] = {
                        'total_lines': total_lines,
                        'executed_lines': executed_lines,
                        'missing_lines': missing_lines,
                        'coverage_percent': (executed_lines / total_lines * 100) if total_lines > 0 else 0,
                        'missing_line_numbers': list(file_analysis.missing)
                    }
                except Exception as e:
                    print(f"Error analyzing {filename}: {e}")
        
        coverage_data = {
            'timestamp': datetime.now().isoformat(),
            'total_coverage_percent': total_coverage,
            'files': file_coverage,
            'html_report': str(html_dir / 'index.html'),
            'xml_report': str(xml_file),
            'summary': {
                'total_files': len(file_coverage),
                'fully_covered_files': len([f for f in file_coverage.values() if f['coverage_percent'] == 100]),
                'uncovered_files': len([f for f in file_coverage.values() if f['coverage_percent'] == 0]),
                'average_coverage': sum(f['coverage_percent'] for f in file_coverage.values()) / len(file_coverage) if file_coverage else 0
            }
        }
        
        # Save JSON report
        with open(json_file, 'w') as f:
            json.dump(coverage_data, f, indent=2)
        
        print(f"📈 Coverage Report Generated")
        print(f"Overall coverage: {total_coverage:.1f}%")
        print(f"HTML report: file://{html_dir / 'index.html'}")
        print(f"JSON report: {json_file}")
        
        return coverage_data
    
    def clean_reports(self):
        """Clean old test reports and coverage data."""
        print("🧹 Cleaning old test reports...")
        
        # Remove reports directory
        if self.reports_dir.exists():
            shutil.rmtree(self.reports_dir)
            self.reports_dir.mkdir()
        
        # Remove coverage files
        coverage_files = ['.coverage', 'coverage.xml', 'htmlcov']
        for file_path in coverage_files:
            path = Path(file_path)
            if path.exists():
                if path.is_file():
                    path.unlink()
                else:
                    shutil.rmtree(path)
        
        print("✅ Reports cleaned")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GaussGAN Test Runner and Coverage Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_runner.py --category unit                    # Run unit tests
  python test_runner.py --category all --parallel         # Run all tests in parallel
  python test_runner.py --suite --fast                    # Run fast test suite
  python test_runner.py --coverage-only                   # Generate coverage report only
  python test_runner.py --clean                           # Clean old reports
        """
    )
    
    parser.add_argument('--category', choices=TestRunner().test_categories.keys(),
                       default='unit', help='Test category to run')
    
    parser.add_argument('--suite', action='store_true',
                       help='Run test suite (multiple categories)')
    
    parser.add_argument('--categories', nargs='+',
                       help='Specific categories for test suite')
    
    parser.add_argument('--fast', action='store_true',
                       help='Run only fast tests (exclude slow markers)')
    
    parser.add_argument('--no-coverage', action='store_true',
                       help='Disable coverage analysis')
    
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel test execution')
    
    parser.add_argument('--strict', action='store_true',
                       help='Use strict mode (fail on warnings)')
    
    parser.add_argument('--save-baselines', action='store_true',
                       help='Save performance baselines')
    
    parser.add_argument('--coverage-only', action='store_true',
                       help='Generate coverage report only')
    
    parser.add_argument('--clean', action='store_true',
                       help='Clean old reports and exit')
    
    parser.add_argument('--project-root', type=Path,
                       help='Project root directory')
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = TestRunner(args.project_root)
    
    # Clean reports if requested
    if args.clean:
        runner.clean_reports()
        return 0
    
    # Generate coverage report only
    if args.coverage_only:
        runner.generate_coverage_report()
        return 0
    
    # Run test suite
    if args.suite:
        categories = args.categories
        if args.fast and not categories:
            categories = ['unit']
        
        results = runner.run_test_suite(
            categories=categories,
            fast_only=args.fast,
            coverage=not args.no_coverage,
            parallel=args.parallel
        )
        
        # Generate coverage report if enabled
        if not args.no_coverage:
            runner.generate_coverage_report()
        
        # Exit with error if any category failed
        failed_categories = [cat for cat, result in results['results'].items() 
                           if not result.get('success', False)]
        
        if failed_categories:
            print(f"\n❌ Failed categories: {', '.join(failed_categories)}")
            return 1
        else:
            print(f"\n✅ All test categories passed!")
            return 0
    
    # Run single category
    else:
        result = runner.run_tests(
            category=args.category,
            coverage=not args.no_coverage,
            parallel=args.parallel,
            save_baselines=args.save_baselines,
            strict=args.strict
        )
        
        # Generate coverage report if enabled
        if not args.no_coverage:
            runner.generate_coverage_report()
        
        return 0 if result['success'] else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)