"""
Result Manager - Handles saving and loading JSON results for tools.

This module provides utilities to write tool results to temporary files
and read them back, ensuring clean separation of results from stdout/stderr.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

_ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent


class ResultManager:
    """
    Manages result file creation and retrieval for tool execution.

    This class handles writing JSON results to a file.
    """

    DEFAULT_FILENAME = "result.json"

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the result manager.

        Args:
            output_dir: Directory to write result files. If None, uses
                       a temporary directory in the tool's working directory.
        """
        self.output_dir = output_dir
        self.result_file = None

    def get_result_file_path(self, tool_name: str = "tool") -> Path:
        """
        Get the path for the result file.

        Args:
            tool_name: Name of the tool (used in filename)

        Returns:
            Path to the result file
        """
        if self.output_dir:
            # Use specified directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.result_file = self.output_dir / f"{tool_name}_{self.DEFAULT_FILENAME}"
        else:
            # Use current working directory
            self.result_file = _ROOT_DIR / self.DEFAULT_FILENAME

        return self.result_file

    def write_result(self, result: Dict[str, Any], tool_name: str = "tool") -> Path:
        """
        Write result to JSON file.

        Args:
            result: Dictionary containing the result data
            tool_name: Name of the tool

        Returns:
            Path to the written file
        """
        result_path = self.get_result_file_path(tool_name)

        # Write JSON to file
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

        return result_path

    @staticmethod
    def read_result(result_path: Path) -> Dict[str, Any]:
        """
        Read result from JSON file.

        Args:
            result_path: Path to the result file

        Returns:
            Dictionary containing the result data

        Raises:
            FileNotFoundError: If result file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
        """
        with open(result_path, 'r', encoding='utf-8') as f:
            return json.load(f)


# Convenience functions for simple usage

def write_result(result: Dict[str, Any], tool_name: str = "tool",
                output_dir: Optional[Path] = None) -> Path:
    """
    Write result to JSON file (convenience function).

    Args:
        result: Dictionary containing the result data
        tool_name: Name of the tool
        output_dir: Optional directory for output file

    Returns:
        Path to the written file
    """
    manager = ResultManager(output_dir)
    return manager.write_result(result, tool_name)


def read_result(result_path: Path) -> Dict[str, Any]:
    """
    Read result from JSON file (convenience function).

    Args:
        result_path: Path to the result file

    Returns:
        Dictionary containing the result data
    """
    return ResultManager.read_result(result_path)


def get_result_dir_from_root(root_dir: Path, tool_name: str) -> Path:
    """
    Get the standard result directory for a tool.

    Creates: <root_dir>/tools/<tool_name>/output/

    Args:
        root_dir: Root directory of the project
        tool_name: Name of the tool

    Returns:
        Path to the tool's output directory
    """
    result_dir = root_dir / "tools" / tool_name / "output"
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir
