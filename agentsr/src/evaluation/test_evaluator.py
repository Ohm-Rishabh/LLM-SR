#!/usr/bin/env python3
"""
Test script for the evaluation module.

This script demonstrates how to use the evaluation module with synthetic examples.
It tests both symbolic and numerical evaluation capabilities.
"""

import numpy as np
import logging
from pathlib import Path
import sys

# Add parent directory to path to import evaluation module
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation import SRAgentEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)


def test_perfect_match():
    """Test case where discovered equation matches ground truth exactly."""
    print("\n" + "=" * 70)
    print("TEST 1: Perfect Match")
    print("=" * 70)

    # Generate synthetic data: y = x^2 + 2*z
    np.random.seed(42)
    n_samples = 100
    x = np.random.uniform(-5, 5, n_samples)
    z = np.random.uniform(-5, 5, n_samples)
    y = x**2 + 2*z

    # Create test data (format: [y, x, z])
    test_data = np.column_stack([y, x, z])

    # Evaluate
    evaluator = SRAgentEvaluator(symbolic_model="gpt-4o", symbolic_temperature=0.0)

    results = evaluator.evaluate(
        discovered_equation="x**2 + 2*z",
        ground_truth_equation="x**2 + 2*z",
        test_data=test_data,
        symbols=["y", "x", "z"],
        check_symbolic=True
    )

    # Print results
    print("\nResults:")
    print(f"Symbolically Equivalent: {results['symbolic_accuracy']['is_equivalent']}")
    print(f"Reasoning: {results['symbolic_accuracy']['reasoning']}")
    print(f"\nTest MSE: {results['numerical_metrics']['test']['metrics']['mse']:.6e}")
    print(f"Test R²: {results['numerical_metrics']['test']['metrics']['r2']:.6f}")

    return results


def test_equivalent_with_different_constants():
    """Test case where equations are equivalent with different constant values."""
    print("\n" + "=" * 70)
    print("TEST 2: Equivalent with Different Constants")
    print("=" * 70)

    # Generate synthetic data: y = 3*x^2 + 4*z
    np.random.seed(43)
    n_samples = 100
    x = np.random.uniform(-5, 5, n_samples)
    z = np.random.uniform(-5, 5, n_samples)
    y = 3*x**2 + 4*z

    test_data = np.column_stack([y, x, z])

    evaluator = SRAgentEvaluator(symbolic_model="gpt-4o")

    results = evaluator.evaluate(
        discovered_equation="3*x**2 + 4*z",
        ground_truth_equation="a*x**2 + b*z",  # Generic form with parameters
        test_data=test_data,
        symbols=["y", "x", "z"],
        check_symbolic=True
    )

    print("\nResults:")
    print(f"Symbolically Equivalent: {results['symbolic_accuracy']['is_equivalent']}")
    print(f"Reasoning: {results['symbolic_accuracy']['reasoning']}")
    print(f"\nTest MSE: {results['numerical_metrics']['test']['metrics']['mse']:.6e}")
    print(f"Test R²: {results['numerical_metrics']['test']['metrics']['r2']:.6f}")

    return results


def test_approximate_match():
    """Test case where discovered equation is approximately correct."""
    print("\n" + "=" * 70)
    print("TEST 3: Approximate Match")
    print("=" * 70)

    # Generate synthetic data: y = x^2 + 2*z
    np.random.seed(44)
    n_samples = 100
    x = np.random.uniform(-5, 5, n_samples)
    z = np.random.uniform(-5, 5, n_samples)
    y = x**2 + 2*z

    test_data = np.column_stack([y, x, z])

    evaluator = SRAgentEvaluator(symbolic_model="gpt-4o")

    # Discovered equation is slightly different: x^2 + 1.9*z
    results = evaluator.evaluate(
        discovered_equation="x**2 + 1.9*z",
        ground_truth_equation="x**2 + 2*z",
        test_data=test_data,
        symbols=["y", "x", "z"],
        check_symbolic=True
    )

    print("\nResults:")
    print(f"Symbolically Equivalent: {results['symbolic_accuracy']['is_equivalent']}")
    print(f"Reasoning: {results['symbolic_accuracy']['reasoning']}")
    print(f"\nTest MSE: {results['numerical_metrics']['test']['metrics']['mse']:.6e}")
    print(f"Test NMSE: {results['numerical_metrics']['test']['metrics']['nmse']:.6e}")
    print(f"Test R²: {results['numerical_metrics']['test']['metrics']['r2']:.6f}")

    return results


def test_wrong_equation():
    """Test case where discovered equation is completely wrong."""
    print("\n" + "=" * 70)
    print("TEST 4: Wrong Equation")
    print("=" * 70)

    # Generate synthetic data: y = x^2 + 2*z
    np.random.seed(45)
    n_samples = 100
    x = np.random.uniform(-5, 5, n_samples)
    z = np.random.uniform(-5, 5, n_samples)
    y = x**2 + 2*z

    test_data = np.column_stack([y, x, z])

    evaluator = SRAgentEvaluator(symbolic_model="gpt-4o")

    # Completely wrong equation
    results = evaluator.evaluate(
        discovered_equation="x + z",
        ground_truth_equation="x**2 + 2*z",
        test_data=test_data,
        symbols=["y", "x", "z"],
        check_symbolic=True
    )

    print("\nResults:")
    print(f"Symbolically Equivalent: {results['symbolic_accuracy']['is_equivalent']}")
    print(f"Reasoning: {results['symbolic_accuracy']['reasoning']}")
    print(f"\nTest MSE: {results['numerical_metrics']['test']['metrics']['mse']:.6e}")
    print(f"Test NMSE: {results['numerical_metrics']['test']['metrics']['nmse']:.6e}")
    print(f"Test R²: {results['numerical_metrics']['test']['metrics']['r2']:.6f}")

    return results


def test_numerical_only():
    """Test numerical evaluation without symbolic checking."""
    print("\n" + "=" * 70)
    print("TEST 5: Numerical Evaluation Only (Skip Symbolic)")
    print("=" * 70)

    # Generate synthetic data: y = sin(x) * z
    np.random.seed(46)
    n_samples = 100
    x = np.random.uniform(0, 2*np.pi, n_samples)
    z = np.random.uniform(-5, 5, n_samples)
    y = np.sin(x) * z

    test_data = np.column_stack([y, x, z])

    evaluator = SRAgentEvaluator()

    results = evaluator.evaluate(
        discovered_equation="sin(x) * z",
        ground_truth_equation="sin(x) * z",
        test_data=test_data,
        symbols=["y", "x", "z"],
        check_symbolic=False  # Skip symbolic evaluation
    )

    print("\nResults:")
    print(f"Symbolic check skipped: {results['symbolic_accuracy'] is None}")
    print(f"\nTest MSE: {results['numerical_metrics']['test']['metrics']['mse']:.6e}")
    print(f"Test R²: {results['numerical_metrics']['test']['metrics']['r2']:.6f}")

    return results


def main():
    """Run all test cases."""
    print("\n" + "=" * 70)
    print("EVALUATION MODULE TEST SUITE")
    print("=" * 70)

    try:
        # Run tests
        test_perfect_match()
        test_equivalent_with_different_constants()
        test_approximate_match()
        test_wrong_equation()
        test_numerical_only()

        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETED")
        print("=" * 70)

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
