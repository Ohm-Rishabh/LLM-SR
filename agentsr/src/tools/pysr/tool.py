#!/usr/bin/env python3
"""
PySR Tool - Symbolic Regression using PySR

This script reads configuration from environment variables set by ToolSwitchNode
and executes symbolic regression using PySRRegressor.

Environment Variables:
    WORKSPACE_INPUT: Path to workspace input directory
    WORKSPACE_OUTPUT: Path to workspace output directory
    WORKSPACE_LOGS: Path to workspace logs directory
    WORKSPACE_SCRATCH: Path to workspace scratch directory
    TOOL_ARG_*: Tool arguments from the LLM's tool_call JSON
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pysr import PySRRegressor, TemplateExpressionSpec
import warnings
from pathlib import Path

# Add parent directory to path to import common utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.result_manager import write_result

# Import template utilities
from template_utils import combine_template_equation

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE) in percent.

    Args:
        y_true: Array-like of true target values.
        y_pred: Array-like of predicted values.

    Returns:
        MAPE as a float in percentage (e.g., 12.3 means 12.3%).
        Returns np.nan if MAPE is undefined (e.g., all y_true are zero or invalid).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Valid entries: finite and non-zero true values
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0)

    if not np.any(mask):
        return np.nan

    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
    return float(mape)


def parse_env_arg(key, default=None, arg_type=str):
    """
    Parse an environment variable argument.

    Args:
        key: Environment variable name (without TOOL_ARG_ prefix)
        default: Default value if not present
        arg_type: Type to convert to (str, int, float, bool, list, dict)

    Returns:
        Parsed value or default
    """
    env_key = f"TOOL_ARG_{key.upper()}"
    value = os.environ.get(env_key)

    if value is None:
        return default

    try:
        if arg_type == bool:
            return value.lower() in ('true', '1', 'yes')
        elif arg_type in (list, dict):
            return json.loads(value)
        elif arg_type == int:
            return int(value)
        elif arg_type == float:
            return float(value)
        else:
            return value
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Warning: Failed to parse {env_key}={value} as {arg_type.__name__}, using default: {default}", file=sys.stderr)
        return default


def load_data(data_file):
    """
    Load CSV data and split into features (X) and target (y).

    Assumes the first column is the target variable.

    Args:
        data_file: Path to CSV file

    Returns:
        X, y, feature_names
    """
    print(f"Loading data from: {data_file}", file=sys.stderr)

    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns", file=sys.stderr)
    print(f"Columns: {list(df.columns)}", file=sys.stderr)

    # Split features and target (assume first column is target)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    feature_names = list(df.columns[1:])
    target_name = df.columns[0]

    print(f"Features: {feature_names}", file=sys.stderr)
    print(f"Target: {target_name}", file=sys.stderr)
    print(f"X shape: {X.shape}, y shape: {y.shape}", file=sys.stderr)

    return X, y, feature_names, target_name


def build_pysr_kwargs():
    """
    Build PySRRegressor kwargs from environment variables.

    Returns:
        Dictionary of PySR arguments
    """
    kwargs = {}

    # Required/Common arguments
    binary_operators = parse_env_arg('binary_operators', ["+", "-", "*", "/"], list)
    unary_operators = parse_env_arg('unary_operators', [], list)

    kwargs['binary_operators'] = binary_operators
    kwargs['unary_operators'] = unary_operators

    # Template Expression Specification (HIGHLY RECOMMENDED)
    expression_spec_dict = parse_env_arg('expression_spec', None, dict)
    if expression_spec_dict is not None:
        try:
            # Validate required fields
            if 'expressions' not in expression_spec_dict:
                raise ValueError("expression_spec must contain 'expressions' field")
            if 'variable_names' not in expression_spec_dict:
                raise ValueError("expression_spec must contain 'variable_names' field")
            if 'combine' not in expression_spec_dict:
                raise ValueError("expression_spec must contain 'combine' field")

            # Create TemplateExpressionSpec object
            template = TemplateExpressionSpec(
                expressions=expression_spec_dict['expressions'],
                variable_names=expression_spec_dict['variable_names'],
                combine=expression_spec_dict['combine']
            )
            kwargs['expression_spec'] = template

            print("\n" + "="*60, file=sys.stderr)
            print("Using Template Expression Specification:", file=sys.stderr)
            print(f"  Sub-expressions: {expression_spec_dict['expressions']}", file=sys.stderr)
            print(f"  Variables: {expression_spec_dict['variable_names']}", file=sys.stderr)
            print(f"  Combine formula: {expression_spec_dict['combine']}", file=sys.stderr)
            print("  This will dramatically reduce search space and improve results!", file=sys.stderr)
            print("="*60 + "\n", file=sys.stderr)

        except Exception as e:
            print(f"Warning: Failed to parse expression_spec: {e}", file=sys.stderr)
            print("Falling back to unrestricted search (slower).", file=sys.stderr)

    # Search configuration
    kwargs['niterations'] = parse_env_arg('niterations', 40, int)
    kwargs['populations'] = parse_env_arg('populations', 15, int)
    kwargs['population_size'] = parse_env_arg('population_size', 33, int)
    kwargs['ncycles_per_iteration'] = parse_env_arg('ncycles_per_iteration', 550, int)

    # Complexity control
    kwargs['maxsize'] = parse_env_arg('maxsize', 20, int)
    maxdepth = parse_env_arg('maxdepth', None, int)
    if maxdepth is not None:
        kwargs['maxdepth'] = maxdepth

    # Performance
    procs = parse_env_arg('procs', None, int)
    if procs is not None:
        kwargs['procs'] = procs

    kwargs['multithreading'] = parse_env_arg('multithreading', True, bool)

    timeout = parse_env_arg('timeout_in_seconds', None, float)
    if timeout is not None:
        kwargs['timeout_in_seconds'] = timeout

    # Feature selection
    select_k_features = parse_env_arg('select_k_features', None, int)
    if select_k_features is not None:
        kwargs['select_k_features'] = select_k_features

    # Optimization
    kwargs['warm_start'] = parse_env_arg('warm_start', False, bool)

    # Loss
    loss = parse_env_arg('loss', None, str)
    if loss is not None:
        kwargs['loss'] = loss

    # Constraints
    constraints = parse_env_arg('constraints', None, dict)
    if constraints is not None:
        kwargs['constraints'] = constraints

    nested_constraints = parse_env_arg('nested_constraints', None, dict)
    if nested_constraints is not None:
        kwargs['nested_constraints'] = nested_constraints

    # Noise handling
    kwargs['denoise'] = parse_env_arg('denoise', False, bool)

    # Extra mappings
    extra_sympy_mappings = parse_env_arg('extra_sympy_mappings', None, dict)
    if extra_sympy_mappings is not None:
        kwargs['extra_sympy_mappings'] = extra_sympy_mappings

    return kwargs


def main():
    """Main execution function."""
    try:
        # Get data file from environment
        data_file = os.environ.get('TOOL_ARG_INPUT_FILE')
        if not data_file:
            raise ValueError("TOOL_ARG_INPUT_FILE environment variable not set")
        data_file = os.path.join(os.environ.get('WORKSPACE_INPUT', ''), data_file)

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Input file not found: {data_file}")

        # Load data
        X, y, feature_names, target_name = load_data(data_file)

        # Build PySR configuration
        pysr_kwargs = build_pysr_kwargs()

        # Check if we're using a template for later equation combining
        expression_spec_dict = parse_env_arg('expression_spec', None, dict)

        print("\n" + "="*60, file=sys.stderr)
        print("PySR Configuration:", file=sys.stderr)
        print("="*60, file=sys.stderr)
        for key, value in sorted(pysr_kwargs.items()):
            print(f"  {key}: {value}", file=sys.stderr)
        print("="*60 + "\n", file=sys.stderr)

        # Create and fit PySR model
        print("Initializing PySRRegressor...", file=sys.stderr)
        model = PySRRegressor(**pysr_kwargs)
        model.feature_names_in_ = feature_names
        model.display_feature_names_in_ = feature_names

        print("Starting symbolic regression search...", file=sys.stderr)
        print("This may take several minutes depending on configuration.", file=sys.stderr)
        print("-"*60, file=sys.stderr)

        model.fit(X, y)

        print("-"*60, file=sys.stderr)
        print("Search completed!", file=sys.stderr)

        # Get results
        equations = model.equations_

        # Find best equation (lowest loss with reasonable complexity)
        best_idx = model.equations_.score.idxmax()
        best_equation = model.equations_.iloc[best_idx]

        # Calculate MAPE for best equation
        best_predictions = model.predict(X, index=best_idx)
        best_mape = calculate_mape(y, best_predictions)

        # Combine template equations if using expression_spec
        best_equation_str = str(best_equation['equation'])
        if expression_spec_dict:
            combined_best = combine_template_equation(best_equation_str, expression_spec_dict)
            print(f"\nBest equation (index {best_idx}):", file=sys.stderr)
            print(f"  Complexity: {best_equation['complexity']}", file=sys.stderr)
            print(f"  Loss: {best_equation['loss']}", file=sys.stderr)
            print(f"  Score: {best_equation['score']}", file=sys.stderr)
            print(f"  MAPE: {best_mape:.4f}%", file=sys.stderr)
            print(f"  Sub-expressions: {best_equation_str}", file=sys.stderr)
            print(f"  Combined equation: {combined_best}", file=sys.stderr)
        else:
            combined_best = best_equation_str
            print(f"\nBest equation (index {best_idx}):", file=sys.stderr)
            print(f"  Complexity: {best_equation['complexity']}", file=sys.stderr)
            print(f"  Loss: {best_equation['loss']}", file=sys.stderr)
            print(f"  Score: {best_equation['score']}", file=sys.stderr)
            print(f"  MAPE: {best_mape:.4f}%", file=sys.stderr)
            print(f"  Equation: {best_equation_str}", file=sys.stderr)

        # Calculate MAPE for all equations
        all_equations = []
        for idx, row in equations.iterrows():
            predictions = model.predict(X, index=idx)
            mape = calculate_mape(y, predictions)

            # Get both raw and combined equation strings
            raw_expr = str(row['equation'])
            if expression_spec_dict:
                combined_expr = combine_template_equation(raw_expr, expression_spec_dict)
            else:
                combined_expr = raw_expr

            all_equations.append({
                "expression": combined_expr,
                "raw_expression": raw_expr if expression_spec_dict else None,
                "complexity": int(row['complexity']),
                "loss": float(f"{row['loss']:.3f}"),
                "score": float(f"{row['score']:.3f}"),
                "mape": float(mape),
            })

        # Prepare configuration for serialization
        serializable_config = {}
        for k, v in pysr_kwargs.items():
            if k == 'extra_sympy_mappings':
                # Skip non-serializable items
                continue
            elif k == 'expression_spec':
                # Convert TemplateExpressionSpec back to dict for JSON serialization
                if isinstance(v, TemplateExpressionSpec):
                    serializable_config[k] = {
                        'expressions': v.expressions,
                        'variable_names': v.variable_names,
                        'combine': v.combine
                    }
                else:
                    serializable_config[k] = v
            else:
                serializable_config[k] = v

        # Prepare results for output
        best_equation_result = {
            "expression": combined_best,
            "complexity": int(best_equation['complexity']),
            "raw_expression": best_equation_str if expression_spec_dict else None,
            "loss": float(f"{best_equation['loss']:.3f}"),
            "score": float(f"{best_equation['score']:.3f}"),
            "mape": float(f"{best_mape:.3f}"),
        }

        results = {
            "tool_name": "pysr",
            "result_type": "equations",
            "status": "success",
            "best_equation": best_equation_result,
            # "all_equations": all_equations,
            # "feature_names": feature_names,
            # "target_name": target_name,
            # "configuration": serializable_config
        }

        # Write results to file using common utility
        result_path = write_result(results, tool_name="pysr")
        print(f"\nResults written to: {result_path}", file=sys.stderr)

        return 0

    except Exception as e:
        # Output error as JSON to file
        error_result = {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }

        # Write error to result file
        try:
            result_path = write_result(error_result, tool_name="pysr")
            print(f"Error written to: {result_path}", file=sys.stderr)
        except Exception as write_error:
            print(f"Failed to write error to file: {write_error}", file=sys.stderr)

        print(f"\nError details: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
