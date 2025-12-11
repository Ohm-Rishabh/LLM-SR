# Defines transformations of states used in the TransformNode instances.

from typing import Dict, List, Optional


def add_tool_results_to_experience(state: Dict[str, any]) -> Dict[str, any]:
    """
    Transform function to add tool results to the agent's past experience.

    Expects 'tool_result' in state, appends it to 'experience'.
    """
    experience: List[Dict[str, any]] = state.get("experience", [])
    tool_result: Dict[str, any] = state.get("tool_result")

    if tool_result:
        experience.append({
            "tool_name": tool_result.pop("tool_name", "unknown_tool"),
            "args": state.get("tool_call", {}).get("args", {}),
            "tool_result": tool_result
        })
        state["experience"] = experience

    state.pop("tool_call", None)

    return state
