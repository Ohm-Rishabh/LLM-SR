from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
import json
import re
import os


class Node(ABC):
    """
    Base class for all nodes in the workflow.

    Each node:
      - has a unique name in the workflow
      - takes a mutable `state` dict as input
      - returns the (possibly modified) `state` dict

    Convention:
      - Nodes can optionally write `state["_next_node"]` to
        request which successor to run next (for branching).

    The node execution follows this pattern:
      1. _build_input(state) -> prepare inputs for the node
      2. Execute the node's core logic
      3. _parse_output(raw_output, state) -> process outputs and update state
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description

    def _build_input(self, state: Dict[str, Any]) -> Any:
        """
        Build the input for this node based on the current state.

        This method extracts and prepares the necessary information from
        the state dict to be used by the node's main execution logic.

        Args:
            state: The current workflow state.

        Returns:
            The prepared input (type depends on the specific node implementation).
        """
        # Default implementation: return the entire state
        return state

    def _parse_output(self, output: Any, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the output from the node's execution and update the state.

        This method processes the raw output from the node's main logic
        and updates the state dictionary accordingly.

        Args:
            output: The raw output from the node's execution.
            state: The current workflow state to be updated.

        Returns:
            The updated state dictionary.
        """
        # Default implementation: if output is a dict, merge it into state
        if isinstance(output, dict):
            state.update(output)
        return state

    @abstractmethod
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute this node's logic.

        Args:
            state: A mutable dictionary carrying the workflow state.

        Returns:
            The updated state dictionary.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class LLMNode(Node):
    """
    Node that uses an LLM (via OpenAI API) to process inputs.

    This node:
      - Builds a prompt from the state and additional instructions
      - Calls the OpenAI API to generate a response
      - Parses the output to extract structured information (tool calls, JSON, etc.)
      - Updates the state with the parsed output

    The node can be configured with:
      - system_prompt: Instructions for the LLM
      - input_keys: Which state keys to include in the prompt
      - output_key: Where to store the raw LLM response in state
      - parse_json: Whether to try parsing JSON from the response
      - parse_tool_calls: Whether to try parsing tool calls from the response
    """

    def __init__(
        self,
        name: str,
        system_prompt: str = "",
        input_keys: Optional[List[str]] = None,
        file_keys: Optional[List[str]] = None,
        output_key: str = "llm_response",
        model: str = "gpt-4.1-mini",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        parse_json: bool = True,
        description: str = "",
    ) -> None:
        """
        Initialize an LLMNode.

        Args:
            name: Unique name for this node.
            system_prompt: System instructions for the LLM.
            input_keys: List of state keys to include in the user message.
                       If None, includes all non-internal keys (those not starting with '_').
            file_keys: List of state keys that contain file paths to be read and included.
                      Files will be included as separate content blocks in the API call.
            output_key: State key where the raw LLM response will be stored.
            model: OpenAI model name (e.g., 'gpt-4', 'gpt-3.5-turbo').
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens to generate.
            parse_json: If True, attempt to extract and parse JSON from the response.
            description: Human-readable description of this node's purpose.
        """
        super().__init__(name, description)
        self.system_prompt = system_prompt
        self.input_keys = input_keys
        self.file_keys = file_keys or []
        self.output_key = output_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.parse_json = parse_json
        self._api_key = None

    def _get_api_key(self) -> str:
        """Get OpenAI API key from environment."""
        if self._api_key is None:
            self._api_key = os.environ.get("OPENAI_API_KEY")
            if not self._api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable not set. "
                    "Please set it to use LLMNode."
                )
        return self._api_key

    def _build_input(self, state: Dict[str, Any]) -> str:
        """
        Build the complete prompt from the state.

        System prompt is loaded from a markdown file in the prompts/ directory
        and prepended to the user prompt.

        If input_keys is specified, only include those keys.
        Otherwise, include all non-internal keys (not starting with '_').

        Files specified in file_keys are read and included separately.

        Args:
            state: Current workflow state.

        Returns:
            Dictionary with 'text_prompt' and optionally 'files' for data files.
        """
        # Load system prompt from file
        system_prompt_text = ""
        if self.system_prompt:
            prompt_file = os.path.join("prompts", f"{self.system_prompt}.md")
            # Try relative to current working directory first
            if not os.path.exists(prompt_file):
                # Try relative to the script's directory
                script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                prompt_file = os.path.join(script_dir, "prompts", f"{self.system_prompt}.md")

            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    system_prompt_text = f.read().strip()
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"System prompt file not found: {prompt_file}. "
                    f"Please ensure the file exists in the prompts/ directory."
                )

        # Build user prompt from state
        if self.input_keys:
            # Use only specified keys
            parts = []
            for key in self.input_keys:
                if key in state:
                    parts.append(f"{key}: {state[key]}")
            user_prompt = "\n".join(parts)
        else:
            # Use all non-internal keys
            parts = []
            for key, value in state.items():
                if not key.startswith("_"):
                    parts.append(f"{key}: {value}")
            user_prompt = "\n".join(parts)

        user_prompt = user_prompt if user_prompt else "No input provided."

        # Concatenate system prompt and user prompt with headers
        prompt_parts = []
        if system_prompt_text:
            prompt_parts.append("# System Instructions")
            prompt_parts.append(system_prompt_text)
            prompt_parts.append("")  # Empty line for separation

        prompt_parts.append("# User Input")
        prompt_parts.append(user_prompt)

        return "\n".join(prompt_parts)

        # # Read data files if specified
        # files = []
        # if self.file_keys:
        #     for key in self.file_keys:
        #         if key in state:
        #             file_path = state[key]
        #             try:
        #                 with open(file_path, 'r', encoding='utf-8') as f:
        #                     file_content = f.read()
        #                 files.append({
        #                     'name': os.path.basename(file_path),
        #                     'path': file_path,
        #                     'content': file_content
        #                 })
        #             except Exception as e:
        #                 raise IOError(f"Error reading file {file_path}: {e}")

        # return {
        #     'text_prompt': text_prompt,
        #     'files': files
        # }

    def _parse_output(self, output: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the LLM output and update the state.

        This method:
          1. Stores the raw output in state[output_key]
          2. If parse_json is True, extracts JSON and stores in state['parsed_json']
          3. If parse_tool_calls is True, extracts tool calls and stores in state['tool_calls']

        Args:
            output: Raw LLM response text.
            state: Current workflow state to update.

        Returns:
            Updated state dictionary.
        """
        # Store raw output
        state[self.output_key] = output

        # Parse JSON if requested
        if self.parse_json:
            parsed_json = self._extract_json(output)
            if parsed_json:
                state["parsed_json"] = parsed_json

        return state

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract and parse JSON from text.

        Looks for JSON in code blocks (```json ... ```) or inline JSON objects.

        Args:
            text: Text to parse.

        Returns:
            Parsed JSON dict, or None if no valid JSON found.
        """
        # Try to find JSON in code blocks first
        json_block_pattern = r"```(?:json)?\s*\n(.*?)\n```"
        matches = re.findall(json_block_pattern, text, re.DOTALL)

        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass

        # Try to find any JSON-like structure with balanced braces
        brace_count = 0
        start_idx = -1

        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    try:
                        return json.loads(text[start_idx:i+1])
                    except json.JSONDecodeError:
                        start_idx = -1

        return None

    def _call_llm(self, user_prompt: str) -> str:
        """
        Make a call to the OpenAI API.

        Args:
            user_prompt: The user message content.

        Returns:
            The generated response text.
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for LLMNode. "
                "Install with: pip install openai"
            )

        client = OpenAI()
        response = client.responses.create(
            model=self.model,
            input=user_prompt,
        )

        return response.output_text

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the LLM node.

        This method:
          1. Builds the input prompt from state
          2. Calls the OpenAI API
          3. Parses the output
          4. Updates and returns the state

        Args:
            state: Current workflow state.

        Returns:
            Updated state dictionary.
        """
        # Build input prompt
        user_prompt = self._build_input(state)

        # Call OpenAI API
        llm_output = self._call_llm(user_prompt)

        # Parse output and update state
        state = self._parse_output(llm_output, state)

        return state


class ToolNode(Node):
    """
    Node that executes a predefined function/tool.

    This node wraps a callable function and executes it with inputs from the state.
    """

    def __init__(
        self,
        name: str,
        tool_fn: Callable,
        input_keys: Optional[List[str]] = None,
        output_key: str = "tool_output",
        description: str = "",
    ) -> None:
        """
        Initialize a ToolNode.

        Args:
            name: Unique name for this node.
            tool_fn: The function to execute. Should accept keyword arguments.
            input_keys: List of state keys to pass as arguments to tool_fn.
                       If None, passes all non-internal keys.
            output_key: State key where the tool output will be stored.
            description: Human-readable description of this node's purpose.
        """
        super().__init__(name, description)
        self.tool_fn = tool_fn
        self.input_keys = input_keys
        self.output_key = output_key

    def _build_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract input arguments for the tool from the state.

        Args:
            state: Current workflow state.

        Returns:
            Dictionary of arguments to pass to the tool function.
        """
        if self.input_keys:
            # Use only specified keys
            return {key: state[key] for key in self.input_keys if key in state}
        else:
            # Use all non-internal keys
            return {key: value for key, value in state.items() if not key.startswith("_")}

    def _parse_output(self, output: Any, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store the tool output in the state.

        Args:
            output: The return value from the tool function.
            state: Current workflow state to update.

        Returns:
            Updated state dictionary.
        """
        state[self.output_key] = output
        return state

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool node.

        Args:
            state: Current workflow state.

        Returns:
            Updated state dictionary.
        """
        # Build input arguments
        tool_args = self._build_input(state)

        # Execute the tool
        tool_output = self.tool_fn(**tool_args)

        # Parse output and update state
        state = self._parse_output(tool_output, state)

        return state
