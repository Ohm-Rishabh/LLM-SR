#!/usr/bin/env python3
"""
Aggregate reasoning log JSON files into a CSV file.

This script reads all *_reasoning.json files from the logs/ directory and
creates a CSV with columns for dataset name, discovered equation, ground truth,
and a flattened reasoning chain.

Usage:
    python aggregate_logs.py [--logs-dir LOGS_DIR] [--output OUTPUT_CSV]
"""

import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any
import sys


def load_reasoning_logs(logs_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all reasoning log JSON files from the specified directory.

    Args:
        logs_dir: Path to the logs directory

    Returns:
        List of dictionaries containing log data
    """
    log_files = list(logs_dir.glob("*_reasoning.json"))
    logs = []

    for log_file in sorted(log_files):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
                logs.append(log_data)
        except Exception as e:
            print(f"Warning: Failed to load {log_file}: {e}", file=sys.stderr)

    return logs


def find_max_reasoning_steps(logs: List[Dict[str, Any]]) -> int:
    """
    Find the maximum number of reasoning steps across all logs.

    Args:
        logs: List of log dictionaries

    Returns:
        Maximum number of reasoning steps
    """
    max_steps = 0
    for log in logs:
        reasoning_process = log.get("reasoning_process", [])
        max_steps = max(max_steps, len(reasoning_process))
    return max_steps


def format_reasoning_step(step: Dict[str, Any]) -> str:
    """
    Format a single reasoning step into a string for CSV.

    Args:
        step: Dictionary containing step information

    Returns:
        Formatted string representation of the step
    """
    step_type = step.get("type", "unknown")

    if step_type == "llm_output":
        content = step.get("content", "")
        return f"[LLM] {content}"

    elif step_type == "tool_result":
        tool_name = step.get("tool_name", "unknown_tool")
        result = step.get("result", {})
        status = result.get("status", "unknown")

        if status == "success":
            # Format based on tool type
            if "output" in result:
                # python_interpreter
                output = result["output"]
                if len(output) > 300:
                    output = output[:300] + "... (truncated)"
                return f"[{tool_name}] Success: {output}"
            elif "expression" in result:
                # pysr
                expr = result.get("expression", "N/A")
                score = result.get("score", "N/A")
                mape = result.get("mape", "N/A")
                return f"[{tool_name}] Success: expr={expr}, score={score}, mape={mape}"
            else:
                # Generic success
                return f"[{tool_name}] Success"
        else:
            # Error case
            error_msg = result.get("error", "Unknown error")
            error_type = result.get("error_type", "")
            return f"[{tool_name}] Error ({error_type}): {error_msg}"

    return f"[{step_type}] (unknown format)"


def aggregate_logs_to_csv(logs: List[Dict[str, Any]], output_path: Path):
    """
    Aggregate reasoning logs into a CSV file.

    Args:
        logs: List of log dictionaries
        output_path: Path to output CSV file
    """
    if not logs:
        print("No logs found to aggregate.", file=sys.stderr)
        return

    # Find maximum reasoning steps to determine number of columns
    max_steps = find_max_reasoning_steps(logs)
    print(f"Found {len(logs)} logs with max {max_steps} reasoning steps")

    # Build header
    header = [
        "dataset_name",
        "discovered_equation",
        "ground_truth_equation",
        "total_steps",
        "tool_calls"
    ]

    # Add columns for each reasoning step
    for i in range(max_steps):
        header.append(f"step_{i+1}")

    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()

        for log in logs:
            row = {
                "dataset_name": log.get("dataset_name", ""),
                "discovered_equation": log.get("discovered_equation", ""),
                "ground_truth_equation": log.get("ground_truth_equation", ""),
                "total_steps": log.get("metadata", {}).get("total_steps", ""),
                "tool_calls": log.get("metadata", {}).get("tool_calls", "")
            }

            # Add reasoning steps
            reasoning_process = log.get("reasoning_process", [])
            for i, step in enumerate(reasoning_process):
                col_name = f"step_{i+1}"
                row[col_name] = format_reasoning_step(step)

            # Fill remaining columns with empty strings
            for i in range(len(reasoning_process), max_steps):
                col_name = f"step_{i+1}"
                row[col_name] = ""

            writer.writerow(row)

    print(f"Successfully wrote aggregated logs to: {output_path}")
    print(f"Total rows: {len(logs)}")
    print(f"Total columns: {len(header)} (5 metadata + {max_steps} reasoning steps)")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Aggregate reasoning log JSON files into a CSV file"
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default=None,
        help="Directory containing reasoning log JSON files (default: logs/ in project root)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reasoning_logs_aggregated.csv",
        help="Output CSV file path (default: reasoning_logs_aggregated.csv)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information about each log file"
    )

    args = parser.parse_args()

    # Determine logs directory
    if args.logs_dir:
        logs_dir = Path(args.logs_dir)
    else:
        # Use logs/ directory relative to this script
        script_dir = Path(__file__).parent
        logs_dir = script_dir / "logs"

    # Check if logs directory exists
    if not logs_dir.exists():
        print(f"Error: Logs directory not found: {logs_dir}", file=sys.stderr)
        sys.exit(1)

    # Load logs
    print(f"Loading logs from: {logs_dir}")
    logs = load_reasoning_logs(logs_dir)

    if not logs:
        print("No reasoning log files found.", file=sys.stderr)
        sys.exit(1)

    # Print verbose information if requested
    if args.verbose:
        print("\nLog files loaded:")
        for log in logs:
            dataset = log.get("dataset_name", "unknown")
            steps = len(log.get("reasoning_process", []))
            print(f"  - {dataset}: {steps} steps")
        print()

    # Aggregate to CSV
    output_path = Path(args.output)
    aggregate_logs_to_csv(logs, output_path)


if __name__ == "__main__":
    main()
