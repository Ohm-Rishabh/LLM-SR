#!/usr/bin/env python3
"""
Symbolic Regression Workflow with LLM-SRBench Dataset Integration.

This script demonstrates:
- Loading datasets from LLM-SRBench by name
- Creating a simple workflow with one SRNode
- Passing dataset metadata and CSV data to the LLM
- Running the workflow and displaying the LLM response

Usage:
    python main.py <dataset_name> [additional_instructions]

Examples:
    python main.py I.10.7_1_0
    python main.py BPG0 "Use simple operators only"
"""

from __future__ import annotations
import sys
import logging
import argparse
from nodes import SRNode, ToolSwitchNode
from core.workflow import Workflow
from core.node import LoopController, TransformNode, LLMNode
from transforms import add_tool_results_to_experience
from datasets.llmsrbench import LLMSRBenchDataset

logger = logging.getLogger(__name__)


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
    """Run a simple SR workflow with LLM-SRBench dataset."""
    # Setup logging (can be set to logging.DEBUG for more verbose output)
    # Or use environment variable: AGENTSR_LOG_LEVEL=DEBUG
    import os
    log_level_name = os.getenv('AGENTSR_LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    setup_logging(level=log_level)

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run symbolic regression on LLM-SRBench datasets',
    )
    parser.add_argument('-D', '--dataset_name', type=str, help='Dataset name (e.g., I.10.7_1_0, BPG0)')
    parser.add_argument('-I', '--instructions', type=str, default='', help='Additional instructions for the agent (optional)')
    parser.add_argument('-T', '--temperature', type=float, default=0.7, help='Temperature for LLM sampling (default: 0.7)')
    parser.add_argument('-M', '--model', default="gpt-4o-mini", help="GPT model to use")

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset: {args.dataset_name}")
    try:
        dataset_manager = LLMSRBenchDataset()
        csv_path, metadata = dataset_manager.get_dataset(args.dataset_name, split="train")
        logger.info(f"Dataset loaded: {csv_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo extract datasets, run:")
        print("  cd src && python datasets/llmsrbench.py")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # Build user query with dataset metadata
    user_query_parts = []

    # Parse symbols and identify target variable
    symbols = metadata.get('symbols', [])
    symbol_descs = metadata.get('symbol_descs', [])
    symbol_properties = metadata.get('symbol_properties', [])

    # Identify target variable (marked with 'O' in symbol_properties)
    target_desc = None

    if len(symbols) > 0 and len(symbol_properties) > 0:
        for sym, prop, desc in zip(symbols, symbol_properties, symbol_descs):
            if prop == 'O':
                target_desc = f"{sym} ({desc})"
                break

    # Build variable description sentence
    if len(symbols) > 0 and len(symbol_descs) > 0:
        # List all variables in order
        var_list = [f"{sym} ({desc})" for sym, desc in zip(symbols, symbol_descs)]

        # Build natural language description
        if len(var_list) > 1:
            all_vars_str = ", ".join(var_list[:-1]) + f", and {var_list[-1]}"
        else:
            all_vars_str = var_list[0]

        var_sentence = f"The dataset contains the following variables: {all_vars_str}."

        # Add target specification if available
        if target_desc:
            var_sentence += f" The goal is to find a symbolic expression for {target_desc} in terms of the other variables."

        user_query_parts.append(var_sentence)

    # Add user instructions if provided
    if args.instructions:
        user_query_parts.append(f"\nAdditional instructions: {args.instructions}")

    user_query = "\n\n".join(user_query_parts)

    # Create an SR node with file support
    sr_node = SRNode(
        name="sr_analyzer",
        system_prompt="sr_analyzer",  # Uses prompts/sr_analyzer.md
        tool_list=["pysr", "python_interpreter"],  # Available tools - loads tool_specs/pysr.md
        input_keys=["user_query"],
        model=args.model,
        temperature=args.temperature,
        max_tokens=16384,
        parse_json=True,  # Parse JSON for tool call extraction
        description="An SR node that analyzes data files and prepares symbolic regression tool calls"
    )

    # Create a tool switch node to execute the tool calls
    tool_switch_node = ToolSwitchNode(
        name="tool_executor",
        description="Executes tool calls and awaits results"
    )

    add_tool_results_transform = TransformNode(
        name="add_tool_results",
        transform_fn=add_tool_results_to_experience,
        description="Adds tool results to the agent's experience log"
    )

    exit_node = TransformNode(
        name="exit",
        transform_fn=lambda state: state,  # No-op
        description="Exit node to terminate the workflow"
    )

    # Loop controller
    loop_controller = LoopController(
        name="loop_controller",
        max_iterations=3,
        continue_node_id="sr_analyzer",
        exit_node_id="summary",
    )

    # summary node when reaching exit condition
    summary_node = LLMNode(
        name="summary",
        system_prompt="summary",
        input_keys=["experience"],
        model="gpt-4o-mini",
        parse_json=True,
    )

    # Create a workflow and add nodes
    workflow = Workflow()
    workflow.add_node(sr_node, is_start=True)
    workflow.add_node(tool_switch_node)
    workflow.add_node(loop_controller)
    workflow.add_node(summary_node)
    workflow.add_node(exit_node)
    workflow.add_node(add_tool_results_transform)
    # Add edges
    workflow.add_edge(sr_node, tool_switch_node)
    workflow.add_edge(sr_node, exit_node)
    workflow.add_edge(tool_switch_node, add_tool_results_transform)
    workflow.add_edge(add_tool_results_transform, loop_controller)
    workflow.add_edge(loop_controller, sr_node)
    workflow.add_edge(loop_controller, summary_node)

    print("-" * 60)
    print("Task Description:")
    print("-" * 60)
    print(user_query)
    print()
    print(f"Input File: {csv_path}")
    print()

    # Run the workflow
    try:
        print("Processing...")
        initial_state = {
            "user_query": user_query,
            "input_file": str(csv_path),
            "dataset_name": args.dataset_name,
            "dataset_metadata": metadata
        }
        result_state = workflow.run(initial_state, cleanup_old_workspaces=True)

        # Display the response
        print("-" * 60)
        print("Assistant Analysis:")
        print("-" * 60)
        print(result_state.get("llm_response", "No response generated."))
        print()

        # Display workflow metadata
        print("=" * 60)
        print(f"Workflow completed. Visited nodes: {result_state.get('_visited_nodes', [])}")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
