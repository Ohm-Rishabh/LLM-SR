"""
Utility functions for agentsr.
"""

from typing import Dict, List, Any


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
