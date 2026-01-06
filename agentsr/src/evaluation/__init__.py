"""
Evaluation Module for Symbolic Regression Agent.

This module provides comprehensive evaluation capabilities for symbolic regression
results, including:
- Symbolic accuracy: Check if discovered equation is symbolically equivalent to ground truth
- Numerical precision: Compute MSE, NMSE, R², Kendall Tau, MAPE metrics

Usage:
    from evaluation import SRAgentEvaluator

    evaluator = SRAgentEvaluator()
    results = evaluator.evaluate(
        discovered_equation="x**2 + y",
        ground_truth_equation="a*x**2 + b*y",
        test_data=test_array,
        symbols=["z", "x", "y"]
    )
"""

from .symbolic_evaluator import SymbolicAccuracyEvaluator
from .numerical_evaluator import NumericalEvaluator
from .evaluator import SRAgentEvaluator

__all__ = [
    'SymbolicAccuracyEvaluator',
    'NumericalEvaluator',
    'SRAgentEvaluator',
]
