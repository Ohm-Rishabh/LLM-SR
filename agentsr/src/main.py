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
from nodes import SRNode
from core.workflow import Workflow


def main():
    """Run a simple SR workflow with file support."""

    print("=" * 60)
    print("Symbolic Regression Workflow Example")
    print("=" * 60)
    print()

    # Path to the data file
    data_file_path = "/home/ubuntu/LLM-SR/llmsr/data/strogatz-ode/noise0.01/bacres1.csv"

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

    # Create a workflow and add the SR node
    workflow = Workflow()
    workflow.add_node(sr_node, is_start=True)

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

        # Display extracted tool call if JSON was parsed
        if "parsed_json" in result_state:
            print("-" * 60)
            print("Extracted Tool Call:")
            print("-" * 60)
            import json
            print(json.dumps(result_state["parsed_json"], indent=2))
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
