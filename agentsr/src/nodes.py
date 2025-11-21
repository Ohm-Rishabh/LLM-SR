from core.node import LLMNode
from typing import Any, Dict, List, Optional

class ToolSwitchNode(LLMNode):
    """
    A specialized LLM node that decides which tool to use for symbolic regression.

    This node analyzes the data characteristics and task requirements to
    select the appropriate symbolic regression tool.
    """

    def __init__(
        self,
        name: str = "tool_switch",
        system_prompt: str = "tool_selector",
        available_tools: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the tool switch node.

        Args:
            name: Node name.
            system_prompt: Name of the prompt file (without .md extension).
            available_tools: List of available tool names.
            **kwargs: Additional arguments passed to LLMNode.
        """
        super().__init__(
            name=name,
            system_prompt=system_prompt,
            parse_json=True,
            **kwargs
        )
        self.available_tools = available_tools or ["linear_regression"]

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the tool switch node and determine the next node to execute.

        Args:
            state: Current workflow state.

        Returns:
            Updated state with tool selection.
        """
        # Call parent run to get LLM output
        state = super().run(state)

        # Extract tool selection from parsed JSON
        if "parsed_json" in state and "tool" in state["parsed_json"]:
            selected_tool = state["parsed_json"]["tool"]
        else:
            # Default to linear regression if no tool specified
            selected_tool = "linear_regression"

        # Store the selected tool
        state["selected_tool"] = selected_tool

        # Set the next node based on the selected tool
        state["_next_node"] = f"tool_{selected_tool}"

        return state