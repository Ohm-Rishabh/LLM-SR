#!/usr/bin/env python3
"""
Minimal example of a workflow with a single SRNode.

This script demonstrates:
- Creating a simple workflow with one SRNode
- Passing a CSV data file to the LLM
- Accepting user input from command line
- Running the workflow and displaying the LLM response
"""

from __future__ import annotations
import sys
import logging
from nodes import SRNode, ToolSwitchNode
from core.workflow import Workflow


# Configure logging
def setup_logging(level=logging.INFO):
    """Configure logging for the application."""
    import os
    from core.consts import ROOT_DIR

    # Create logs directory if it doesn't exist
    log_dir = os.path.join(ROOT_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Log file path
    log_file = os.path.join(log_dir, 'agentsr.log')

    # Configure logging with both console and file handlers
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(log_file, mode='a')  # File output
        ]
    )


def main():
    """Run a simple SR workflow with file support."""
    # Setup logging (can be set to logging.DEBUG for more verbose output)
    # Or use environment variable: AGENTSR_LOG_LEVEL=DEBUG
    import os
    log_level_name = os.getenv('AGENTSR_LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    setup_logging(level=log_level)

    print("=" * 60)
    print("Symbolic Regression Workflow Example")
    print("=" * 60)
    print()

    # Path to the data file
    data_file_path = "/home/ubuntu/LLM-SR/llmsr/data/test.csv"

    # Create an SR node with file support
    sr_node = SRNode(
        name="sr_analyzer",
        system_prompt="sr_analyzer",  # Uses prompts/sr_analyzer.md
        file_keys=["data_file"],  # State key containing the file path
        tool_list=["pysr"],  # Available tools - loads tool_specs/pysr.md
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=8192,
        parse_json=True,  # Parse JSON for tool call extraction
        description="An SR node that analyzes data files and prepares symbolic regression tool calls"
    )

    # Create a tool switch node to execute the tool calls
    tool_switch_node = ToolSwitchNode(
        name="tool_executor",
        description="Executes tool calls and awaits results"
    )

    # Create a workflow and add both nodes
    workflow = Workflow()
    workflow.add_node(sr_node, is_start=True)
    workflow.add_node(tool_switch_node)
    workflow.add_edge(sr_node.name, tool_switch_node.name)

    # Get user input from command line
    if len(sys.argv) > 1:
        # If arguments provided, use them as the query
        user_input = " ".join(sys.argv[1:])
    else:
        # Otherwise, prompt for input
        print("Enter your message (or 'quit' to exit):")
        user_input = input("> ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            return

    if not user_input:
        print("Error: No input provided.")
        print("Usage: python main.py <your message>")
        print("   or: python main.py  (for interactive mode)")
        return

    print()
    print(f"User: {user_input}")
    print(f"Data File: {data_file_path}")
    print()

    # Run the workflow
    try:
        print("Processing...")
        initial_state = {
            "user_query": user_input,
            "data_file": data_file_path
        }
        result_state = workflow.run(initial_state)

        # Display the response
        print("-" * 60)
        print("Assistant Analysis:")
        print("-" * 60)
        print(result_state.get("llm_response", "No response generated."))
        print()

        # Display extracted tool call if present
        if "tool_call" in result_state:
            print("-" * 60)
            print("Extracted Tool Call:")
            print("-" * 60)
            import json
            print(json.dumps(result_state["tool_call"], indent=2))
            print()

        # Display tool execution results if present
        if "tool_result" in result_state:
            print("-" * 60)
            print("Tool Execution Result:")
            print("-" * 60)
            import json
            print(json.dumps(result_state["tool_result"], indent=2))
            print()

        # Display workflow metadata
        print("=" * 60)
        print(f"Workflow completed. Visited nodes: {result_state.get('_visited_nodes', [])}")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("1. OPENAI_API_KEY environment variable is set")
        print("2. The openai package is installed (pip install openai)")
        sys.exit(1)


if __name__ == "__main__":
    main()
