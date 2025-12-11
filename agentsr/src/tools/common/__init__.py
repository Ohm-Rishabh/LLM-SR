"""
Common utilities for tool execution.

This module provides shared functionality for all symbolic regression tools.
"""

from .result_manager import ResultManager, write_result, read_result

__all__ = ['ResultManager', 'write_result', 'read_result']
