#!/usr/bin/env python3
"""
Python Interpreter Tool - Execute Python code for data analysis

This script reads configuration from environment variables set by ToolSwitchNode
and executes Python code in a sandboxed environment.

Environment Variables:
    TOOL_ARG_CODE: Python code to execute

Workspace Environment Variables:
    WORKSPACE_ROOT: Main workspace directory
    WORKSPACE_INPUT: Input files directory
    WORKSPACE_OUTPUT: Output files directory (for plots, results)
    WORKSPACE_LOGS: Log files directory
    WORKSPACE_SCRATCH: Temporary/scratch directory
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from pathlib import Path
import traceback
import signal
import logging

logger = logging.getLogger(__name__)

# Add parent directory to path to import common utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.result_manager import write_result

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class TimeoutError(Exception):
    """Raised when code execution times out."""
    pass


def timeout_handler(signum, frame):
    """Handler for execution timeout."""
    raise TimeoutError("Code execution timed out")


def parse_env_arg(key, default=None, arg_type=str):
    """
    Parse an environment variable argument.

    Args:
        key: Environment variable name (without TOOL_ARG_ prefix)
        default: Default value if not present
        arg_type: Type to convert to (str, int, float, bool)

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
        elif arg_type == int:
            return int(value)
        elif arg_type == float:
            return float(value)
        else:
            return value
    except ValueError as e:
        print(f"Warning: Failed to parse {env_key}={value} as {arg_type.__name__}, using default: {default}", file=sys.stderr)
        return default


def execute_code(code, workspace_paths, timeout=30):
    """
    Execute Python code in a controlled environment.

    Args:
        code: Python code string to execute
        workspace_paths: Dictionary of workspace directory paths
        timeout: Maximum execution time in seconds

    Returns:
        Dictionary with execution results
    """
    # Prepare execution environment with workspace paths
    exec_globals = {
        '__builtins__': __builtins__,
        # Pre-import common libraries
        'np': np,
        'pd': pd,
        'plt': plt,
        'sns': sns,
        'stats': stats,
        'os': os,
        # Workspace paths for saving files
        'workspace_root': workspace_paths.get('root'),
        'workspace_output': workspace_paths.get('output'),
        'workspace_input': workspace_paths.get('input'),
        'workspace_logs': workspace_paths.get('logs'),
        'workspace_scratch': workspace_paths.get('scratch'),
    }
    exec_locals = {}

    # Set timeout alarm (Unix only)
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

    try:
        # Execute the code
        exec(code, exec_globals, exec_locals)

        # Cancel alarm
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)

        status = "success"
        error_msg = None
        error_type = None
        error_traceback = None

    except TimeoutError as e:
        status = "error"
        error_msg = f"Execution timed out after {timeout} seconds"
        error_type = "TimeoutError"
        error_traceback = traceback.format_exc()

    except SyntaxError as e:
        status = "error"
        error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
        error_type = "SyntaxError"
        error_traceback = traceback.format_exc()

    except Exception as e:
        status = "error"
        error_msg = str(e)
        error_type = type(e).__name__
        error_traceback = traceback.format_exc()

    finally:
        # Cancel alarm if still set
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)

    # Extract result from exec_locals
    result_data = exec_locals.get('result', None)
    if result_data is None and status == "success":
        logger.warning("No 'result' variable set in executed code.")

    # Close matplotlib figures to free memory
    plt.close('all')

    return {
        "status": status,
        "error": error_msg,
        "error_type": error_type,
        "error_traceback": error_traceback,
        "result": result_data,
    }


def main():
    """Main execution function."""
    try:
        # Get workspace paths from environment
        workspace_paths = {
            'root': os.environ.get('WORKSPACE_ROOT'),
            'input': os.environ.get('WORKSPACE_INPUT'),
            'output': os.environ.get('WORKSPACE_OUTPUT'),
            'logs': os.environ.get('WORKSPACE_LOGS'),
            'scratch': os.environ.get('WORKSPACE_SCRATCH'),
        }

        # Get code from environment
        code = parse_env_arg('code', None, str)
        if not code:
            raise ValueError("No code provided. Set TOOL_ARG_CODE environment variable.")

        print("\n" + "="*60, file=sys.stderr)
        print("Python Code Interpreter", file=sys.stderr)
        print("="*60, file=sys.stderr)
        if workspace_paths['output']:
            print(f"Workspace output: {workspace_paths['output']}", file=sys.stderr)
        print("Executing code:", file=sys.stderr)
        print("-"*60, file=sys.stderr)
        print(code, file=sys.stderr)
        print("-"*60, file=sys.stderr)

        # Execute code with workspace paths
        execution_result = execute_code(
            code=code,
            workspace_paths=workspace_paths,
            timeout=30
        )

        print("\n" + "="*60, file=sys.stderr)
        print("Execution completed", file=sys.stderr)
        print("="*60, file=sys.stderr)

        if execution_result["status"] == "success":
            print("Status: SUCCESS", file=sys.stderr)
        else:
            print(f"Status: ERROR ({execution_result['error_type']})", file=sys.stderr)
            print(f"Error: {execution_result['error']}", file=sys.stderr)
            if execution_result["error_traceback"]:
                print("\nError traceback:", file=sys.stderr)
                print(execution_result["error_traceback"], file=sys.stderr)

        # Prepare results for output
        # The result should be from the 'result' variable in the executed code
        code_result = execution_result.get("result")

        if execution_result["status"] == "success" and code_result is not None:
            # User code set a result variable
            results = {
                "tool_name": "python_interpreter",
                "result_type": "code_execution",
                "status": "success",
                # "code": code,
                "result": code_result,  # This should be a dict with 'summary' and 'saved_files'
            }
        else:
            # Either error or no result variable was set
            results = {
                "tool_name": "python_interpreter",
                "result_type": "code_execution",
                "status": execution_result["status"],
                "code": code,
                "error": execution_result["error"],
                "error_type": execution_result["error_type"],
                "error_traceback": execution_result["error_traceback"],
            }

        # Write results to file using common utility
        result_path = write_result(results, tool_name="python_interpreter")
        print(f"\nResults written to: {result_path}", file=sys.stderr)

        # Return exit code based on execution status
        return 0 if execution_result["status"] == "success" else 1

    except Exception as e:
        # Output error as JSON to file
        error_result = {
            "tool_name": "python_interpreter",
            "result_type": "code_execution",
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "error_traceback": traceback.format_exc()
        }

        # Write error to result file
        try:
            result_path = write_result(error_result, tool_name="python_interpreter")
            print(f"Error written to: {result_path}", file=sys.stderr)
        except Exception as write_error:
            print(f"Failed to write error to file: {write_error}", file=sys.stderr)

        print(f"\nError details: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
