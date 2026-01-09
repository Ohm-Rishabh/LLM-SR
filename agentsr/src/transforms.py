# Defines transformations of states used in the TransformNode instances.

from typing import Dict, List, Optional, Any
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


def track_llm_output(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform function to track LLM outputs in a history list.

    Expects 'llm_response' in state, appends it to 'llm_history'.
    """
    llm_history: List[str] = state.get("llm_history", [])
    llm_response = state.get("llm_response")

    if llm_response:
        llm_history.append(llm_response)
        state["llm_history"] = llm_history

    return state
