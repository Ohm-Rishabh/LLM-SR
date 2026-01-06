"""
Numerical Precision Evaluator.

This module implements numerical evaluation metrics for discovered equations,
including MSE, NMSE, R², Kendall Tau, and MAPE. The implementation is adapted
from llm-srbench/bench/pipelines.py.
"""

import logging
import numpy as np
import sympy as sp
from typing import Dict, Any, Optional, Callable, List
from scipy.stats import kendalltau
from sklearn.metrics import mean_absolute_percentage_error

logger = logging.getLogger(__name__)


class NumericalEvaluator:
    """
    Evaluator for computing numerical metrics between discovered equations and data.

    This evaluator computes various numerical precision metrics by evaluating
    the discovered equation on test data and comparing with ground truth values.
    """

    def __init__(self):
        """Initialize the numerical evaluator."""
        pass

    def compute_metrics(self, y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Compute numerical metrics between predicted and true values.

        Metrics computed:
        - MSE: Mean Squared Error
        - NMSE: Normalized Mean Squared Error (MSE / variance)
        - R²: Coefficient of determination
        - KDT: Kendall Tau correlation coefficient
        - MAPE: Mean Absolute Percentage Error

        Args:
            y_pred: Predicted values from the discovered equation
            y_true: Ground truth values

        Returns:
            Dictionary containing all computed metrics
        """
        # Handle NaN values by filtering them out
        valid_mask = ~np.isnan(y_pred) & ~np.isnan(y_true) & np.isfinite(y_pred) & np.isfinite(y_true)

        if not np.any(valid_mask):
            logger.warning("All predictions are NaN or infinite")
            return {
                "mse": float('inf'),
                "nmse": float('inf'),
                "r2": -float('inf'),
                "kdt": 0.0,
                "mape": float('inf'),
                "num_valid_points": 0,
            }

        y_pred_valid = y_pred[valid_mask]
        y_true_valid = y_true[valid_mask]

        # Compute variance
        var = np.var(y_true_valid)

        # MSE
        mse = np.mean((y_true_valid - y_pred_valid) ** 2)

        # NMSE (normalized by variance)
        nmse = mse / var if var != 0 else float('inf')

        # R²
        ss_res = np.sum((y_true_valid - y_pred_valid) ** 2)
        ss_tot = np.sum((y_true_valid - y_true_valid.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else -float('inf')

        # Kendall Tau
        try:
            kdt = kendalltau(y_true_valid, y_pred_valid)[0]
        except Exception as e:
            logger.warning(f"Failed to compute Kendall Tau: {e}")
            kdt = 0.0

        # MAPE
        try:
            mape = mean_absolute_percentage_error(y_true_valid, y_pred_valid)
        except Exception as e:
            logger.warning(f"Failed to compute MAPE: {e}")
            mape = float('inf')

        return {
            "mse": float(mse),
            "nmse": float(nmse),
            "r2": float(r2),
            "kdt": float(kdt),
            "mape": float(mape),
            "num_valid_points": int(np.sum(valid_mask)),
        }

    def evaluate_equation_string(
        self,
        equation_str: str,
        X: np.ndarray,
        y_true: np.ndarray,
        symbols: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate an equation string on data and compute metrics.

        Args:
            equation_str: Mathematical equation as a string (e.g., "x**2 + y")
            X: Input data array of shape (n_samples, n_features)
            y_true: Ground truth output values of shape (n_samples,)
            symbols: List of variable names corresponding to X columns

        Returns:
            Dictionary containing metrics and evaluation status
        """
        logger.info(f"Evaluating equation: {equation_str}")
        logger.info(f"Data shape: X={X.shape}, y={y_true.shape}")
        logger.info(f"Symbols: {symbols}")

        try:
            # Create lambda function from equation string
            lambda_fn = self._string_to_lambda(equation_str, symbols)

            # Evaluate on data
            y_pred = lambda_fn(X)

            # Compute metrics
            metrics = self.compute_metrics(y_pred, y_true)

            return {
                "success": True,
                "metrics": metrics,
                "equation": equation_str,
                "error": None
            }

        except Exception as e:
            logger.error(f"Failed to evaluate equation: {e}")
            return {
                "success": False,
                "metrics": None,
                "equation": equation_str,
                "error": str(e)
            }

    def _string_to_lambda(self, equation_str: str, symbols: List[str]) -> Callable:
        """
        Convert equation string to a lambda function.

        Args:
            equation_str: Mathematical equation as a string
            symbols: List of variable names

        Returns:
            Lambda function that takes X array and returns predictions

        Raises:
            ValueError: If equation cannot be parsed
        """
        try:
            # Parse equation with sympy
            sympy_symbols = sp.symbols(' '.join(symbols))
            if not isinstance(sympy_symbols, tuple):
                sympy_symbols = (sympy_symbols,)

            expr = sp.sympify(equation_str)

            # Convert to lambda function
            # The lambda should take an array X of shape (n_samples, n_features)
            lambda_fn = sp.lambdify(sympy_symbols, expr, modules=['numpy'])

            # Wrap to handle array input correctly
            def wrapped_lambda(X):
                if X.ndim == 1:
                    X = X.reshape(1, -1)

                # Unpack columns as separate arguments
                args = [X[:, i] for i in range(X.shape[1])]
                result = lambda_fn(*args)

                # Ensure result is 1D array
                if isinstance(result, (int, float)):
                    result = np.array([result])
                elif not isinstance(result, np.ndarray):
                    result = np.array(result)

                return result.flatten()

            return wrapped_lambda

        except Exception as e:
            logger.error(f"Failed to convert equation to lambda: {e}")
            raise ValueError(f"Cannot parse equation '{equation_str}': {e}")

    def evaluate_with_train_test_split(
        self,
        equation_str: str,
        train_data: np.ndarray,
        test_data: np.ndarray,
        symbols: List[str],
        ood_test_data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluate equation on train, test, and optional OOD test sets.

        Args:
            equation_str: Mathematical equation as a string
            train_data: Training data array of shape (n_samples, n_features + 1)
                       First column is the output, remaining are inputs
            test_data: Test data array with same format as train_data
            symbols: List of variable names (output symbol first, then inputs)
            ood_test_data: Optional out-of-distribution test data

        Returns:
            Dictionary containing metrics for train, test, and optionally OOD test
        """
        results = {}

        # Note: Following llm-srbench convention where first column is output
        # Input symbols are symbols[1:] (excluding the output symbol)
        input_symbols = symbols[1:] if len(symbols) > 1 else symbols

        # Evaluate on training data
        X_train = train_data[:, 1:]
        y_train = train_data[:, 0]
        results['train'] = self.evaluate_equation_string(
            equation_str, X_train, y_train, input_symbols
        )

        # Evaluate on test data
        X_test = test_data[:, 1:]
        y_test = test_data[:, 0]
        results['test'] = self.evaluate_equation_string(
            equation_str, X_test, y_test, input_symbols
        )

        # Evaluate on OOD test data if provided
        if ood_test_data is not None:
            X_ood = ood_test_data[:, 1:]
            y_ood = ood_test_data[:, 0]
            results['ood_test'] = self.evaluate_equation_string(
                equation_str, X_ood, y_ood, input_symbols
            )

        return results
