from core.node import LLMNode
from core.consts import ROOT_DIR, SRC_DIR
from typing import Any, Dict, List, Optional
import os
import base64

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


class SRNode(LLMNode):
    """
    Symbolic Regression Node with file handling capabilities.

    This node extends LLMNode to support passing local files to the OpenAI API.
    It can handle:
    - Text files (CSV, code, logs, etc.) - included as text content blocks
    - Image files (PNG, JPG, etc.) - included as base64-encoded images for vision models

    Files are specified via file_keys parameter and their paths should be in the state.
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
        Initialize an SRNode with file handling support.

        Args:
            name: Unique name for this node.
            system_prompt: System instructions for the LLM (name of .md file in prompts/).
            input_keys: List of state keys to include in the user message.
                       If None, includes all non-internal keys (those not starting with '_').
            file_keys: List of state keys that contain file paths to be read and included.
                      Files will be included as separate content blocks in the API call.
                      Example: ["data_file", "config_file"]
            output_key: State key where the raw LLM response will be stored.
            model: OpenAI model name (e.g., 'gpt-4', 'gpt-4o' for vision).
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens to generate.
            parse_json: If True, attempt to extract and parse JSON from the response.
            description: Human-readable description of this node's purpose.
        """
        super().__init__(
            name=name,
            system_prompt=system_prompt,
            input_keys=input_keys,
            file_keys=file_keys,
            output_key=output_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            parse_json=parse_json,
            description=description,
        )

    def _build_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build the complete prompt from the state with file support.

        Returns a dictionary with:
        - system_prompt: System instructions text
        - user_content: List of content blocks (text and/or files)

        Args:
            state: Current workflow state.

        Returns:
            Dictionary with 'system_prompt' and 'user_content' for the API call.
        """
        # Load system prompt from file
        system_prompt_text = ""
        if self.system_prompt:
            prompt_file = os.path.join(ROOT_DIR, "prompts", f"{self.system_prompt}.md")
            # # Try relative to current working directory first
            # if not os.path.exists(prompt_file):
            #     # Try relative to the script's directory
            #     script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            #     prompt_file = os.path.join(script_dir, "prompts", f"{self.system_prompt}.md")

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
            # Use all non-internal keys (excluding file keys to avoid duplication)
            parts = []
            for key, value in state.items():
                if not key.startswith("_") and key not in self.file_keys:
                    parts.append(f"{key}: {value}")
            user_prompt = "\n".join(parts)

        user_prompt = user_prompt if user_prompt else "No input provided."

        # Build content blocks starting with the text prompt
        content_blocks = [{"type": "text", "text": user_prompt}]

        # Add file content blocks if specified
        if self.file_keys:
            for key in self.file_keys:
                if key in state:
                    file_path = state[key]
                    try:
                        # Determine file type and read accordingly
                        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                            # Handle images with base64 encoding for vision models
                            with open(file_path, 'rb') as f:
                                image_data = base64.b64encode(f.read()).decode('utf-8')
                            ext = file_path.split('.')[-1].lower()
                            if ext == 'jpg':
                                ext = 'jpeg'  # Normalize jpg to jpeg for MIME type
                            content_blocks.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{ext};base64,{image_data}"
                                }
                            })
                        else:
                            # Handle text files (CSV, code, logs, etc.)
                            with open(file_path, 'r', encoding='utf-8') as f:
                                file_content = f.read()
                            content_blocks.append({
                                "type": "text",
                                "text": f"\n\n--- File: {os.path.basename(file_path)} ---\n{file_content}\n--- End of file ---\n"
                            })
                    except Exception as e:
                        raise IOError(f"Error reading file {file_path} (key: {key}): {e}")

        return {
            "system_prompt": system_prompt_text,
            "user_content": content_blocks
        }

    def _call_llm(self, prompt_data: Dict[str, Any]) -> str:
        """
        Make a call to the OpenAI API with support for files.

        Uses the Chat Completions API with structured messages that can include
        both text and file content blocks.

        Args:
            prompt_data: Dictionary with 'system_prompt' and 'user_content' keys.

        Returns:
            The generated response text.
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for SRNode. "
                "Install with: pip install openai"
            )

        client = OpenAI()

        # Build messages array
        messages = []
        if prompt_data["system_prompt"]:
            messages.append({
                "role": "system",
                "content": prompt_data["system_prompt"]
            })
        messages.append({
            "role": "user",
            "content": prompt_data["user_content"]
        })

        # Call OpenAI Chat Completions API
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return response.choices[0].message.content

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the SR node with file support.

        This method:
          1. Builds the input prompt and file content from state
          2. Calls the OpenAI API with structured messages
          3. Parses the output
          4. Updates and returns the state

        Args:
            state: Current workflow state.

        Returns:
            Updated state dictionary.
        """
        # Build input prompt with files
        prompt_data = self._build_input(state)

        # Call OpenAI API
        llm_output = self._call_llm(prompt_data)

        # Parse output and update state
        state = self._parse_output(llm_output, state)

        return state