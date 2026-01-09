"""
Utility functions for agentsr.
"""

from typing import Dict, List, Any
import json
import os
from pathlib import Path
from datetime import datetime


def format_experience_log(experience: List[Dict[str, Any]]) -> str:
    """
    Format the experience log into a structured, readable string.

    Args:
        experience: List of experience entries, each containing tool_name, args, and tool_result

    Returns:
        Formatted string representation of the experience log
    """
    if not experience:
        return "No tools have been used yet."

    formatted_entries = []
    for idx, entry in enumerate(experience, 1):
        tool_name = entry.get("tool_name", "unknown_tool")
        args = entry.get("args", {})
        tool_result = entry.get("tool_result", {})

        # Start with the entry header
        entry_str = f"### Step {idx}: {tool_name}\n"

        # Add arguments if any
        if args:
            entry_str += "**Arguments:**\n"
            for key, value in args.items():
                # Truncate very long values
                if isinstance(value, str) and len(value) > 200:
                    value_display = value[:200] + "... (truncated)"
                else:
                    value_display = value
                entry_str += f"  - {key}: {value_display}\n"

        # Add result
        entry_str += "**Result:**\n"
        status = tool_result.get("status", "unknown")
        entry_str += f"  - Status: {status}\n"

        if status == "success":
            # Handle different result types
            if "output" in tool_result:
                # python_interpreter result
                output = tool_result["output"]
                if output:
                    entry_str += f"  - Output:\n```\n{output}\n```\n"
                else:
                    entry_str += "  - Output: (empty)\n"
            elif "expression" in tool_result:
                # pysr result
                entry_str += f"  - Best Expression: {tool_result.get('expression', 'N/A')}\n"
                if "score" in tool_result:
                    entry_str += f"  - Score: {tool_result['score']}\n"
                if "mape" in tool_result:
                    entry_str += f"  - MAPE: {tool_result['mape']:.4f}%\n"
            else:
                # Generic result - show key fields
                for key, value in tool_result.items():
                    if key not in ["status", "tool_name"]:
                        entry_str += f"  - {key}: {value}\n"
        else:
            # Error case
            if "error" in tool_result:
                entry_str += f"  - Error: {tool_result['error']}\n"
            if "error_type" in tool_result:
                entry_str += f"  - Error Type: {tool_result['error_type']}\n"
            if "error_traceback" in tool_result:
                entry_str += f"  - Traceback:\n```\n{tool_result['error_traceback']}\n```\n"

        formatted_entries.append(entry_str)

    return "\n".join(formatted_entries)


def save_reasoning_log(
    dataset_name: str,
    state: Dict[str, Any],
    discovered_equation: str,
    ground_truth: str,
    log_dir: str = None
) -> Path:
    """
    Save the reasoning process, discovered equation, and ground truth to a JSON file.

    Args:
        dataset_name: Name of the dataset
        state: Final state from workflow containing experience
        discovered_equation: The equation discovered by the agent
        ground_truth: Ground truth equation from dataset metadata
        log_dir: Directory to save log file (defaults to logs/ in project root)

    Returns:
        Path to the saved log file
    """
    # Set up log directory
    if log_dir is None:
        from core.consts import ROOT_DIR
        log_dir = os.path.join(ROOT_DIR, 'logs')

    os.makedirs(log_dir, exist_ok=True)

    # Build reasoning process list
    reasoning_process = []
    experience = state.get("experience", [])
    llm_history = state.get("llm_history", [])

    # Interleave LLM outputs and tool results
    # Pattern: [LLM output 1, tool result 1, LLM output 2, tool result 2, ..., final LLM output]
    for idx in range(max(len(llm_history), len(experience))):
        # Add LLM output if available
        if idx < len(llm_history):
            reasoning_process.append({
                "type": "llm_output",
                "content": llm_history[idx]
            })

        # Add tool result if available
        if idx < len(experience):
            entry = experience[idx]
            tool_name = entry.get("tool_name", "unknown_tool")
            tool_result = entry.get("tool_result", {})

            reasoning_process.append({
                "type": "tool_result",
                "tool_name": tool_name,
                "args": entry.get("args", {}),
                "result": tool_result
            })

    # Create log data
    log_data = {
        "dataset_name": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "discovered_equation": discovered_equation,
        "ground_truth_equation": ground_truth,
        "reasoning_process": reasoning_process,
        "metadata": {
            "total_steps": len(reasoning_process),
            "tool_calls": len(experience)
        }
    }

    # Save to file
    log_filename = f"{dataset_name}_reasoning.json"
    log_path = Path(log_dir) / log_filename

    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    return log_path
