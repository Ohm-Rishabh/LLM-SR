#!/usr/bin/env python3
"""
Minimal example of a workflow with a single LLMNode.

This script demonstrates:
- Creating a simple workflow with one LLMNode
- Accepting user input from command line
- Running the workflow and displaying the LLM response
"""

from __future__ import annotations
import sys
from core.node import LLMNode
from core.workflow import Workflow


def main():
    """Run a simple LLM workflow that behaves like a raw API call."""

    print("=" * 60)
    print("Simple LLM Workflow Example")
    print("=" * 60)
    print()

    # Create a simple LLM node with no system prompt
    # This behaves like a raw API call
    llm_node = LLMNode(
        name="simple_llm",
        system_prompt="",  # No system prompt - raw API call
        model="gpt-4.1-mini",
        temperature=0.7,
        max_tokens=8192,
        parse_json=False,  # Don't try to parse JSON
        description="A simple LLM node that responds to user input"
    )

    # Create a workflow and add the LLM node
    workflow = Workflow()
    workflow.add_node(llm_node, is_start=True)

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
    print()

    # Run the workflow
    try:
        print("Processing...")
        initial_state = {"user_query": user_input}
        result_state = workflow.run(initial_state)

        # Display the response
        print("-" * 60)
        print("Assistant:")
        print("-" * 60)
        print(result_state.get("llm_response", "No response generated."))
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
