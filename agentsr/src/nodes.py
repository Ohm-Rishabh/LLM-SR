from core.node import LLMNode, Node
from core.consts import ROOT_DIR, SRC_DIR
from typing import Any, Dict, List, Optional
import os
import base64
import logging
import subprocess
import json as json_module
from pathlib import Path

logger = logging.getLogger(__name__)

class ToolSwitchNode(Node):
    """
    A node that executes tool calls from the state.

    This node extracts tool call information from the state (provided by SRNode),
    locates the corresponding bash entry script, substitutes arguments from the
    tool_call JSON, executes the script, and awaits the results.

    Directory structure expected:
        tools/
        ├── pysr/
        │   └── run.sh
        ├── gplearn/
        │   └── run.sh
        └── ...

    The bash scripts can use placeholders that will be replaced with values
    from the tool_call arguments.
    """

    def __init__(
        self,
        name: str = "tool_switch",
        description: str = "Executes tool calls and awaits results",
        tools_dir: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        """
        Initialize the tool switch node.

        Args:
            name: Node name.
            description: Human-readable description of this node's purpose.
            tools_dir: Directory containing tool subdirectories with run.sh scripts.
                      Defaults to ROOT_DIR/tools
            timeout: Maximum execution time in seconds (None for no timeout).
        """
        super().__init__(name=name, description=description)
        self.tools_dir = tools_dir or os.path.join(SRC_DIR, "tools")
        self.timeout = timeout

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the tool switch node to execute tool calls.

        Args:
            state: Current workflow state containing "tool_call" field.

        Returns:
            Updated state with tool execution results.
        """
        logger.info(f"[{self.name}] Starting tool execution")

        # Extract tool call from state
        tool_call = state.get("tool_call")
        if not tool_call:
            logger.warning(f"[{self.name}] No tool_call found in state")
            state["tool_result"] = {"error": "No tool call found in state"}
            return state

        logger.info(f"[{self.name}] Received tool call: {tool_call}")

        # Execute the tool and await results
        tool_result = self._execute_tool(tool_call, state)

        # Store results in state
        state["tool_result"] = tool_result
        logger.info(f"[{self.name}] Tool execution completed")

        return state

    def _execute_tool(self, tool_call: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool call by running its bash entry script.

        This method:
        1. Validates the tool_call structure
        2. Locates the bash entry script (run.sh)
        3. Prepares the execution environment and arguments
        4. Executes the script and captures output
        5. Returns structured results

        Args:
            tool_call: Dictionary containing:
                - tool_name: Name of the tool to execute
                - arguments: Dictionary of arguments to pass
            state: Current workflow state (for context like data_file paths)

        Returns:
            Dictionary containing:
                - status: "success" or "error"
                - stdout: Standard output from the tool
                - stderr: Standard error from the tool
                - exit_code: Process exit code
                - error: Error message (if any)
        """
        # Extract tool name
        tool_name = tool_call.get("tool_name")
        if not tool_name:
            logger.error(f"[{self.name}] No tool_name found in tool_call")
            return {
                "status": "error",
                "error": "Missing tool_name in tool_call",
                "tool_call": tool_call
            }

        logger.info(f"[{self.name}] Executing tool: {tool_name}")

        # Locate the bash entry script
        tool_script_path = os.path.join(self.tools_dir, tool_name, "run.sh")
        if not os.path.exists(tool_script_path):
            logger.error(f"[{self.name}] Tool script not found: {tool_script_path}")
            return {
                "status": "error",
                "error": f"Tool script not found: {tool_script_path}",
                "tool_name": tool_name
            }

        # Prepare arguments
        arguments = tool_call.get("args", {})
        logger.debug(f"[{self.name}] Tool arguments: {arguments}")

        # Prepare environment variables from arguments and state
        env = os.environ.copy()
        env_vars = self._prepare_environment(arguments, state)
        env.update(env_vars)

        logger.debug(f"[{self.name}] Environment variables: {list(env_vars.keys())}")

        # Execute the bash script
        try:
            logger.info(f"[{self.name}] Running script: {tool_script_path}")

            result = subprocess.run(
                ["bash", tool_script_path],
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=os.path.dirname(tool_script_path)  # Run in tool directory
            )

            # Capture results
            exit_code = result.returncode
            stderr = result.stderr

            logger.info(f"[{self.name}] Tool completed with exit code: {exit_code}")

            # Construct expected result file path
            # The tool writes to workspace_scratch/result.json if workspace is enabled,
            # otherwise falls back to ROOT_DIR/result.json
            if "_workspace_root" in state:
                workspace_scratch = os.path.join(state["_workspace_root"], "scratch")
                result_file_path = os.path.join(workspace_scratch, "result.json")
            else:
                result_file_path = os.path.join(ROOT_DIR, "result.json")

            if not os.path.exists(result_file_path):
                logger.error(f"[{self.name}] Result file not found at: {result_file_path}")
                return {
                    "status": "error",
                    "tool_name": tool_name,
                    "error": f"Result file not found: {result_file_path}",
                    "exit_code": exit_code,
                    "stderr": stderr,
                }

            # Read JSON result from file
            logger.info(f"[{self.name}] Reading results from: {result_file_path}")
            try:
                with open(result_file_path, 'r') as f:
                    result_data = json_module.load(f)

                # Add metadata
                result_data["tool_name"] = tool_name
                result_data["exit_code"] = exit_code
                # result_data["stderr"] = stderr  # Keep stderr for logs

                logger.info(f"[{self.name}] Successfully loaded result from file")

                # Delete the result file
                try:
                    os.remove(result_file_path)
                    logger.debug(f"[{self.name}] Deleted result file: {result_file_path}")
                except Exception as e:
                    logger.warning(f"[{self.name}] Failed to delete result file: {e}")

                return result_data

            except json_module.JSONDecodeError as e:
                logger.error(f"[{self.name}] Invalid JSON in result file: {e}")
                return {
                    "status": "error",
                    "tool_name": tool_name,
                    "error": f"Invalid JSON in result file: {str(e)}",
                    "exit_code": exit_code,
                    # "stderr": stderr,
                }
            except Exception as e:
                logger.error(f"[{self.name}] Failed to read result file: {e}")
                return {
                    "status": "error",
                    "tool_name": tool_name,
                    "error": f"Failed to read result file: {str(e)}",
                    "exit_code": exit_code,
                    # "stderr": stderr,
                }

        except subprocess.TimeoutExpired as e:
            logger.error(f"[{self.name}] Tool execution timeout after {self.timeout}s")
            return {
                "status": "error",
                "tool_name": tool_name,
                "error": f"Tool execution timeout after {self.timeout} seconds",
                # "stderr": e.stderr if e.stderr else "",
            }

        except Exception as e:
            logger.error(f"[{self.name}] Error executing tool: {e}")
            return {
                "status": "error",
                "tool_name": tool_name,
                "error": f"Execution error: {str(e)}",
            }

    def _prepare_environment(
        self,
        args: Dict[str, Any],
        state: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Prepare environment variables from arguments and state.

        This method converts the tool_call arguments into environment variables
        that can be used in the bash script. Additionally, it includes useful
        state variables like data_file paths and workspace directories.

        Naming convention:
        - Arguments are prefixed with TOOL_ARG_
        - State variables are prefixed with STATE_
        - Workspace paths are prefixed with WORKSPACE_
        - All keys are uppercased

        Examples:
            arguments = {"maxsize": 30, "binary_operators": ["+", "*"]}
            -> TOOL_ARG_MAXSIZE=30
            -> TOOL_ARG_BINARY_OPERATORS='["+", "*"]'  (JSON string)

        Args:
            arguments: Tool call arguments dictionary
            state: Current workflow state

        Returns:
            Dictionary of environment variables (all string values)
        """
        env_vars = {}

        # Add arguments as environment variables
        for key, value in args.items():
            env_key = f"TOOL_ARG_{key.upper()}"

            # Convert to string appropriately
            if isinstance(value, (list, dict)):
                # Serialize complex types as JSON
                env_vars[env_key] = json_module.dumps(value)
            elif isinstance(value, bool):
                # Convert boolean to string "true"/"false"
                env_vars[env_key] = str(value).lower()
            else:
                # Convert to string
                env_vars[env_key] = str(value)

        # Add useful state variables
        if "user_query" in state:
            env_vars["STATE_USER_QUERY"] = state["user_query"]

        # Add workspace environment variables
        if "_workspace_root" in state:
            workspace_root = state["_workspace_root"]
            env_vars["WORKSPACE_ROOT"] = workspace_root
            # Also provide convenient paths to standard subdirectories
            env_vars["WORKSPACE_INPUT"] = os.path.join(workspace_root, "input")
            env_vars["WORKSPACE_OUTPUT"] = os.path.join(workspace_root, "output")
            env_vars["WORKSPACE_LOGS"] = os.path.join(workspace_root, "logs")
            env_vars["WORKSPACE_SCRATCH"] = os.path.join(workspace_root, "scratch")
            logger.debug(f"[{self.name}] Added workspace environment variables")

        return env_vars


class SRNode(LLMNode):
    """
    Main node for symbolic regression.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str = "",
        input_keys: Optional[List[str]] = None,
        tool_list: Optional[List[str]] = None,
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
            tool_list: List of tool names to load specifications for.
                      Tool specs are loaded from tool_specs/{tool_name}.md files.
                      Example: ["pysr", "gplearn", "linear_regression"]
                      If None, no tool specs are loaded.
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
            output_key=output_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            parse_json=parse_json,
            description=description,
        )
        self.tool_list = tool_list

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
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    system_prompt_text = f.read().strip()
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"System prompt file not found: {prompt_file}. "
                    f"Please ensure the file exists in the prompts/ directory."
                )

        # Load tool specifications if tool_list is provided
        if self.tool_list:
            logger.debug(f"[{self.name}] Loading tool specifications for: {self.tool_list}")
            system_prompt_text = system_prompt_text + "\n\n## Available Tools\nThe following tools are available for use:"
            tool_specs_parts = []
            for tool_name in self.tool_list:
                tool_spec_file = os.path.join(ROOT_DIR, "tool_specs", f"{tool_name}.md")
                try:
                    with open(tool_spec_file, 'r', encoding='utf-8') as f:
                        tool_spec_content = f.read().strip()
                    tool_specs_parts.append(f"### {tool_name}\n{tool_spec_content}")
                    logger.debug(f"[{self.name}] Loaded tool spec: {tool_name}")
                except FileNotFoundError:
                    # Log warning but continue - tool spec is optional
                    logger.warning(f"[{self.name}] Tool spec file not found: {tool_spec_file}")
                    continue

            if tool_specs_parts:
                tool_specs_text = "\n\n---\n\n".join(tool_specs_parts)
                # Append tool specs to system prompt
                system_prompt_text = system_prompt_text + "\n\n" + tool_specs_text
                logger.info(f"[{self.name}] Appended {len(tool_specs_parts)} tool spec(s) to system prompt")

        # Build user prompt from state
        if self.input_keys:
            # Use only specified keys
            parts = []
            for key in self.input_keys:
                if key in state:
                    parts.append(f"{key}: {state[key]}")
            user_prompt = "\n".join(parts)
        else:
            user_prompt = "No user input provided."

        # Add workspace files summary if available
        if self.workspace_manager:
            files_summary = self.get_workspace_files_summary()
            if files_summary and "No files registered yet" not in files_summary:
                user_prompt += f"\n\n## Workspace Files\n{files_summary}"

        # Add experience log if available
        if "experience" in state:
            experience_log = state["experience"]
            user_prompt += f"\n\n## Experience Log\n{experience_log}"

        # Build content blocks starting with the text prompt
        content_blocks = [{"type": "text", "text": user_prompt}]

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
            logger.error(f"[{self.name}] OpenAI package not installed")
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

        logger.info(f"[{self.name}] Calling OpenAI Chat Completions API with model={self.model}")
        logger.debug(f"[{self.name}] Message count: {len(messages)}, content blocks: {len(prompt_data['user_content'])}")

        # Call OpenAI Chat Completions API
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        logger.info(f"[{self.name}] Received response from OpenAI Chat Completions API")

        # Log token usage
        if hasattr(response, 'usage') and response.usage:
            logger.info(
                f"[{self.name}] Token usage - "
                f"Prompt: {response.usage.prompt_tokens}, "
                f"Completion: {response.usage.completion_tokens}, "
                f"Total: {response.usage.total_tokens}"
            )

        return response.choices[0].message.content

    def _parse_output(self, output: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the LLM output and update the state.

        This method:
          1. Stores the raw output in state[output_key]
          2. If parse_json is True, extracts JSON and stores in state['parsed_json']
          3. If parse_tool_calls is True, extracts tool calls and stores in state['tool_calls']
          4. For python_interpreter tool, extracts Python code from code blocks

        Args:
            output: Raw LLM response text.
            state: Current workflow state to update.

        Returns:
            Updated state dictionary.
        """
        logger.debug(f"[{self.name}] Parsing LLM output (length: {len(output)} chars)")

        # Store raw output
        state[self.output_key] = output
        logger.info(f"[{self.name}] Raw output: {output}")

        # Parse JSON if requested
        if self.parse_json:
            parsed_json = self._extract_json(output)
            if parsed_json:
                logger.info(f"[{self.name}] Successfully extracted JSON from response")
                logger.debug(f"[{self.name}] Extracted JSON keys: {list(parsed_json.keys())}")
                # should be a "tool_call" key
                tool_call = parsed_json.get("tool_call", {})
                if tool_call:
                    logger.info(f"[{self.name}] Successfully extracted tool call from JSON")
                    logger.debug(f"[{self.name}] Extracted tool call keys: {list(tool_call.keys())}")

                    # Special handling for python_interpreter tool
                    # Extract Python code from code blocks in the response
                    if tool_call.get("tool_name") == "python_interpreter":
                        python_code = self._extract_python_code(output)
                        if python_code:
                            logger.info(f"[{self.name}] Extracted Python code for python_interpreter")
                            # Add the code to the tool call arguments
                            if "args" not in tool_call:
                                tool_call["args"] = {}
                            tool_call["args"]["code"] = python_code
                        else:
                            logger.warning(f"[{self.name}] python_interpreter tool called but no Python code block found")

                    state["tool_call"] = tool_call
                    state["_next_node"] = "tool_executor"
                else:
                    final_result = parsed_json.get("final_result")
                    if final_result:
                        logger.info(f"[{self.name}] Extracted final_result from JSON, setting exit node")
                        state["final_result"] = final_result
                        state["_next_node"] = "exit"
                    else:
                        logger.warning(f"[{self.name}] No tool_call or final_result found in extracted JSON")
            else:
                # patch - look for Python code, as the model might call python_interpreter without JSON
                python_code = self._extract_python_code(output)
                if python_code:
                    logger.info(f"[{self.name}] Extracted Python code outside JSON, assuming python_interpreter tool call")
                    state["tool_call"] = {
                        "tool_name": "python_interpreter",
                        "args": {
                            "code": python_code
                        }
                    }
                    state["_next_node"] = "tool_executor"
                else:
                    logger.warning(f"[{self.name}] Failed to extract JSON from response")

        return state

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