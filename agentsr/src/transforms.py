# Defines transformations of states used in the TransformNode instances.

from typing import Dict, List, Optional
from tools.common.configs import ARGS_EXCLUDED_IN_EXPERIENCE


def add_tool_results_to_experience(state: Dict[str, any]) -> Dict[str, any]:
    """
    Transform function to add tool results to the agent's past experience.

    Expects 'tool_result' in state, appends it to 'experience'.
    """
    experience: List[Dict[str, any]] = state.get("experience", [])
    tool_result: Dict[str, any] = state.get("tool_result")

    if tool_result:
        tool_name = tool_result.pop("tool_name", "unknown_tool")
        args = state.get("tool_call", {}).get("args", {})
        # Exclude sensitive args based on tool name
        excluded_args = ARGS_EXCLUDED_IN_EXPERIENCE.get(tool_name, [])
        for arg in excluded_args:
            args.pop(arg, None)
        experience.append({
            "tool_name": tool_name,
            "args": args,
            "tool_result": tool_result
        })
        state["experience"] = experience

    state.pop("tool_call", None)

    return state
