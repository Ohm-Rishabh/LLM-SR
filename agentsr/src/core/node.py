from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, TYPE_CHECKING
import json
import re
import os
import logging
from core.consts import ROOT_DIR

if TYPE_CHECKING:
    from core.workspace import WorkspaceManager

logger = logging.getLogger(__name__)


class Node(ABC):
    """
    Base class for all nodes in the workflow.

    Each node:
      - has a unique name in the workflow
      - takes a mutable `state` dict as input
      - returns the (possibly modified) `state` dict
      - can access workspace_manager for file operations

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
        self._workspace_manager: Optional['WorkspaceManager'] = None

    def set_workspace_manager(self, workspace_manager: Optional['WorkspaceManager']) -> None:
        """
        Set the workspace manager for this node.

        This is called by the Workflow when the node is added,
        or when the workflow run starts with workspace enabled.

        Args:
            workspace_manager: WorkspaceManager instance or None
        """
        self._workspace_manager = workspace_manager

    @property
    def workspace_manager(self) -> Optional['WorkspaceManager']:
        """
        Get the workspace manager for file operations.

        Returns:
            WorkspaceManager instance if workspace is enabled, None otherwise
        """
        return self._workspace_manager

    def get_workspace_files_summary(self) -> Optional[str]:
        """
        Get a formatted summary of all workspace files with descriptions.

        This is useful for including in LLM prompts to give context about
        available files.

        Returns:
            Formatted string with file descriptions, or None if no workspace

        Example:
            >>> summary = self.get_workspace_files_summary()
            >>> prompt = f"Available files:\\n{summary}\\n\\nWhat should we analyze next?"
        """
        if self._workspace_manager:
            return self._workspace_manager.get_files_summary()
        return None

    def list_workspace_files(self, subdir: Optional[str] = None, file_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List files in workspace with metadata.

        Args:
            subdir: Filter by subdirectory (e.g., "output", "input")
            file_type: Filter by file type (e.g., "plot", "data")

        Returns:
            List of file info dictionaries, or empty list if no workspace

        Example:
            >>> plots = self.list_workspace_files(file_type="plot")
            >>> for plot in plots:
            ...     print(f"Available plot: {plot['path']} - {plot['description']}")
        """
        if self._workspace_manager:
            return self._workspace_manager.list_files_with_metadata(subdir, file_type)
        return []

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
        logger.debug(f"[{self.name}] Building input from state")

        # Load system prompt from file
        system_prompt_text = ""
        if self.system_prompt:
            prompt_file = os.path.join(ROOT_DIR, "prompts", f"{self.system_prompt}.md")

            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    system_prompt_text = f.read().strip()
                logger.debug(f"[{self.name}] Loaded system prompt from {prompt_file}")
            except FileNotFoundError:
                logger.error(f"[{self.name}] System prompt file not found: {prompt_file}")
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

        # Add workspace files summary if available
        workspace_files_section = ""
        if self.workspace_manager:
            files_summary = self.get_workspace_files_summary()
            if files_summary and "No files registered yet" not in files_summary:
                workspace_files_section = f"\n\n# Workspace Files\n{files_summary}"
                logger.debug(f"[{self.name}] Added workspace files summary to prompt")

        # Concatenate system prompt and user prompt with headers
        prompt_parts = []
        if system_prompt_text:
            prompt_parts.append("# System Instructions")
            prompt_parts.append(system_prompt_text)
            prompt_parts.append("")  # Empty line for separation

        prompt_parts.append("# User Input")
        prompt_parts.append(user_prompt)

        # Add workspace files section if available
        if workspace_files_section:
            prompt_parts.append(workspace_files_section)

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
        logger.debug(f"[{self.name}] Parsing LLM output (length: {len(output)} chars)")

        # Store raw output
        state[self.output_key] = output

        # Parse JSON if requested
        if self.parse_json:
            parsed_json = self._extract_json(output)
            if parsed_json:
                logger.info(f"[{self.name}] Successfully extracted JSON from response")
                logger.debug(f"[{self.name}] Extracted JSON keys: {list(parsed_json.keys())}")
                state["parsed_json"] = parsed_json
            else:
                logger.warning(f"[{self.name}] Failed to extract JSON from response")

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

    def _extract_python_code(self, text: str) -> Optional[str]:
        """
        Extract Python code from text.

        Looks for Python code in code blocks (```python ... ``` or ``` ... ```).

        Args:
            text: Text to parse.

        Returns:
            Extracted Python code as string, or None if no code block found.
        """
        # Try to find Python code in code blocks
        # Pattern matches ```python or just ``` followed by code
        python_block_pattern = r"```(?:python)?\s*\n(.*?)\n```"
        matches = re.findall(python_block_pattern, text, re.DOTALL)

        if matches:
            # Return the first match (code content without the backticks)
            return matches[0]

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
            logger.error(f"[{self.name}] OpenAI package not installed")
            raise ImportError(
                "openai package is required for LLMNode. "
                "Install with: pip install openai"
            )

        logger.info(f"[{self.name}] Calling OpenAI API with model={self.model}")
        logger.debug(f"[{self.name}] Prompt length: {len(user_prompt)} chars, temp={self.temperature}, max_tokens={self.max_tokens}")

        client = OpenAI()
        response = client.responses.create(
            model=self.model,
            input=user_prompt,
        )

        logger.info(f"[{self.name}] Received response from OpenAI API")
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
        logger.info(f"[{self.name}] Starting LLMNode execution")

        # Build input prompt
        user_prompt = self._build_input(state)

        # Call OpenAI API
        llm_output = self._call_llm(user_prompt)

        # Parse output and update state
        state = self._parse_output(llm_output, state)

        logger.info(f"[{self.name}] Completed LLMNode execution")
        return state


class LoopController(Node):
    """
    Decides whether to continue looping or exit upon a maximum number of iterations.
    """
    def __init__(
        self,
        name: str,
        continue_node_id: str,
        exit_node_id: str,
        max_iterations: int = 10,
        description: str = "",
    ) -> None:
        super().__init__(name, description)
        self.continue_node_id = continue_node_id
        self.exit_node_id = exit_node_id
        self.max_iterations = max_iterations
        self.current_iteration = 0

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            state["_next_node"] = self.continue_node_id
        else:
            state["_next_node"] = self.exit_node_id
        return state
    

class TransformNode(Node):
    """
    Transform the state using a provided function.
    """
    def __init__(
        self,
        name: str,
        transform_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        description: str = "",
    ) -> None:
        super().__init__(name, description)
        self.transform_fn = transform_fn

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self.transform_fn(state)