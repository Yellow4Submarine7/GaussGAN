#!/usr/bin/env python
"""
Specialized test runner for new statistical metrics.
Focuses on the comprehensive testing of MMD, Wasserstein, and ConvergenceTracker.
"""

import sys
import os
import subprocess
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any

# Add source to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_pytest_command(
    test_files: List[str],
    markers: str = "",
    verbose: bool = True,
    coverage: bool = True,
    timeout: int = 300
) -> Dict[str, Any]:
    """Run pytest with specified parameters."""
    cmd = ["python", "-m", "pytest"]
    
    # Add test files
    cmd.extend(test_files)
    
    # Add markers
    if markers:
        cmd.extend(["-m", markers])
    
    # Add verbosity
    if verbose:
        cmd.extend(["-v", "-s"])
    
    # Add coverage
    if coverage:
        cmd.extend([
            "--cov=source.metrics",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov/new_metrics"
        ])
    
    # Add timeout
    cmd.extend(["--timeout", str(timeout)])
    
    # Add warnings
    cmd.extend(["--disable-warnings"])
    
    print(f"Running: {' '.join(cmd)}")
    print("-" * 80)
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, timeout=timeout + 60)
        exit_code = result.returncode
        success = exit_code == 0
    except subprocess.TimeoutExpired:
        print(f"Tests timed out after {timeout} seconds")
        exit_code = -1
        success = False
    except KeyboardInterrupt:
        print("Tests interrupted by user")
        exit_code = -2
        success = False
    
    duration = time.time() - start_time
    
    return {
        "command": " ".join(cmd),
        "exit_code": exit_code,
        "success": success,
        "duration": duration
    }


def main():
    """Main entry point for new metrics test runner."""
    parser = argparse.ArgumentParser(
        description="Test runner for new statistical metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Categories:
  unit        - Unit tests for MMD, Wasserstein, ConvergenceTracker
  integration - Integration tests with training pipeline
  performance - Performance and scalability tests
  statistical - Statistical correctness validation
  edge        - Edge cases and error handling
  all         - Run all test categories

Examples:
  python run_new_metrics_tests.py unit                    # Run unit tests
  python run_new_metrics_tests.py all --no-coverage     # Run all without coverage
  python run_new_metrics_tests.py performance --timeout 600  # Performance tests with longer timeout
        """
    )
    
    parser.add_argument(
        "category",
        choices=["unit", "integration", "performance", "statistical", "edge", "all"],
        help="Test category to run"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage reporting"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet output (less verbose)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Test timeout in seconds (default: 300)"
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip slow tests"
    )
    
    args = parser.parse_args()
    
    # Test definitions
    test_configs = {
        "unit": {
            "files": [
                "docs/tests/unit/test_mmd_metrics.py",
                "docs/tests/unit/test_wasserstein_distance.py",
                "docs/tests/unit/test_convergence_tracker.py"
            ],
            "markers": "not slow" if args.fast else "",
            "description": "Unit tests for new metrics"
        },
        "integration": {
            "files": [
                "docs/tests/integration/test_new_metrics_integration.py"
            ],
            "markers": "integration and not slow" if args.fast else "integration",
            "description": "Integration tests with training pipeline"
        },
        "performance": {
            "files": [
                "docs/tests/performance/test_metrics_performance.py"
            ],
            "markers": "performance and not slow" if args.fast else "performance",
            "description": "Performance and scalability tests"
        },
        "statistical": {
            "files": [
                "docs/tests/statistical/test_metrics_correctness.py"
            ],
            "markers": "statistical and not slow" if args.fast else "statistical",
            "description": "Statistical correctness validation"
        },
        "edge": {
            "files": [
                "docs/tests/unit/test_edge_cases_and_errors.py"
            ],
            "markers": "not slow" if args.fast else "",
            "description": "Edge cases and error handling"
        },
        "all": {
            "files": [
                "docs/tests/unit/test_mmd_metrics.py",
                "docs/tests/unit/test_wasserstein_distance.py",
                "docs/tests/unit/test_convergence_tracker.py",
                "docs/tests/unit/test_edge_cases_and_errors.py",
                "docs/tests/integration/test_new_metrics_integration.py",
                "docs/tests/performance/test_metrics_performance.py",
                "docs/tests/statistical/test_metrics_correctness.py"
            ],
            "markers": "not slow" if args.fast else "",
            "description": "All new metrics tests"
        }
    }
    
    # Get test configuration
    config = test_configs[args.category]
    
    # Filter existing files
    existing_files = []
    for file_path in config["files"]:
        if Path(file_path).exists():
            existing_files.append(file_path)
        else:
            print(f"Warning: Test file not found: {file_path}")
    
    if not existing_files:
        print(f"Error: No test files found for category '{args.category}'")
        return 1
    
    print("=" * 80)
    print(f"NEW METRICS TEST RUNNER - {args.category.upper()}")
    print("=" * 80)
    print(f"Description: {config['description']}")
    print(f"Test files: {len(existing_files)}")
    for file_path in existing_files:
        print(f"  - {file_path}")
    print(f"Markers: {config['markers'] or 'none'}")
    print(f"Coverage: {'disabled' if args.no_coverage else 'enabled'}")
    print(f"Timeout: {args.timeout} seconds")
    if args.fast:
        print("Fast mode: skipping slow tests")
    print("=" * 80)
    
    # Run tests
    result = run_pytest_command(
        test_files=existing_files,
        markers=config["markers"],
        verbose=not args.quiet,
        coverage=not args.no_coverage,
        timeout=args.timeout
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Category: {args.category}")
    print(f"Success: {'✅ PASSED' if result['success'] else '❌ FAILED'}")
    print(f"Exit code: {result['exit_code']}")
    print(f"Duration: {result['duration']:.1f} seconds")
    
    if not args.no_coverage and result['success']:
        print(f"Coverage report: file://{Path('htmlcov/new_metrics/index.html').absolute()}")
    
    print("=" * 80)
    
    # Quick usage tips
    if not result['success']:
        print("\n💡 Troubleshooting tips:")
        print("  - Check if all dependencies are installed: pip install -r requirements.txt")
        print("  - Verify source code is in the correct path")
        print("  - Try running with --fast to skip slow tests")
        print("  - Check specific test failures for details")
    
    return 0 if result['success'] else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)