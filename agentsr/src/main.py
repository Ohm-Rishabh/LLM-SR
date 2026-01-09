#!/usr/bin/env python3
"""
Symbolic Regression Workflow with LLM-SRBench Dataset Integration.

This script demonstrates:
- Loading datasets from LLM-SRBench by name
- Creating a simple workflow with one SRNode
- Passing dataset metadata and CSV data to the LLM
- Running the workflow and displaying the LLM response
- Optional evaluation of discovered equations (symbolic + numerical metrics)

Usage:
    python main.py -D <dataset_name> [options]

Examples:
    python main.py -D I.10.7_1_0
    python main.py -D BPG0 -I "Use simple operators only"
    python main.py -D I.10.7_1_0 --eval
    python main.py -D I.10.7_1_0 --eval --skip-symbolic --output-dir ./results
"""

from __future__ import annotations
import sys
import logging
import argparse
from pathlib import Path
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
    parser.add_argument('-D', '--dataset_name', type=str, required=True, help='Dataset name (e.g., I.10.7_1_0, BPG0)')
    parser.add_argument('-I', '--instructions', type=str, default='', help='Additional instructions for the agent (optional)')
    parser.add_argument('-T', '--temperature', type=float, default=0.7, help='Temperature for LLM sampling (default: 0.7)')
    parser.add_argument('-M', '--model', default="gpt-4o-mini", help="GPT model to use for SR agent")
    parser.add_argument('--max_iter',  type=int, default=5, help="maximum iterations of tool calls")

    # Evaluation options
    parser.add_argument('-E', '--eval', action='store_true', help='Enable evaluation of discovered equations')
    parser.add_argument('--eval-model', default="gpt-4o", help="GPT model to use for symbolic evaluation (default: gpt-4o)")
    parser.add_argument('--skip-symbolic', action='store_true', help="Skip symbolic accuracy evaluation (faster)")
    parser.add_argument('--output-dir', type=str, default=None, help="Directory to save evaluation results")
    parser.add_argument('--acc_tol', type=float, default=0.1, help="Tolerance for accuracy metric (default: 0.1)")

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset: {args.dataset_name}")
    try:
        dataset_manager = LLMSRBenchDataset()
        csv_path, metadata = dataset_manager.get_dataset(args.dataset_name, split="train")
        logger.info(f"Dataset loaded: {csv_path}")

        # Load test data if evaluation is enabled
        test_data = None
        ood_data = None
        if args.eval:
            import pandas as pd
            test_csv_path, _ = dataset_manager.get_dataset(args.dataset_name, split="test")
            test_df = pd.read_csv(test_csv_path)
            test_data = test_df.values
            logger.info(f"Test data loaded: {test_data.shape}")

            # Try to load OOD test data if available
            try:
                ood_csv_path, _ = dataset_manager.get_dataset(args.dataset_name, split="ood_test")
                ood_df = pd.read_csv(ood_csv_path)
                ood_data = ood_df.values
                logger.info(f"OOD test data loaded: {ood_data.shape}")
            except:
                logger.info("No OOD test data available")

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
        max_iterations=args.max_iter,
        continue_node_id="sr_analyzer",
        exit_node_id="summary",
    )

    # summary node when reaching exit condition
    summary_node = LLMNode(
        name="summary",
        system_prompt="summary",
        input_keys=["experience"],
        additional_output_keys=["final_result"],
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
    if args.eval:
        print(f"Ground Truth: {metadata.get('expression', 'Unknown')}")
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

        # Evaluation if enabled
        if args.eval:
            # Extract discovered equation
            discovered_equation = result_state.get("final_result", "")

            if discovered_equation:
                print("=" * 60)
                print("EVALUATION")
                print("=" * 60)
                print()

                try:
                    # Import evaluation module
                    from evaluation import SRAgentEvaluator

                    # Initialize evaluator
                    evaluator = SRAgentEvaluator(
                        symbolic_model=args.eval_model,
                        symbolic_temperature=0.0,
                        tolerance=args.acc_tol
                    )

                    # Run evaluation
                    eval_results = evaluator.evaluate_from_agent_output(
                        agent_output=discovered_equation,
                        dataset_metadata=metadata,
                        test_data=test_data,
                        ood_test_data=ood_data,
                        check_symbolic=not args.skip_symbolic
                    )

                    # Display results
                    print("Discovered Equation:")
                    print(f"  {discovered_equation}")
                    print()
                    print("Ground Truth:")
                    print(f"  {metadata.get('expression', 'Unknown')}")
                    print()

                    if eval_results["symbolic_accuracy"] is not None:
                        print("Symbolic Accuracy:")
                        is_equiv = eval_results["symbolic_accuracy"]["is_equivalent"]
                        print(f"  Symbolically Equivalent: {'YES' if is_equiv else 'NO'}")
                        print(f"  Reasoning: {eval_results['symbolic_accuracy']['reasoning']}")
                        print()

                    if "test" in eval_results["numerical_metrics"]:
                        test_result = eval_results["numerical_metrics"]["test"]
                        if test_result["success"]:
                            metrics = test_result["metrics"]
                            print("Numerical Metrics (Test Set):")
                            print(f"  MSE:  {metrics['mse']:.6e}")
                            print(f"  NMSE: {metrics['nmse']:.6e}")
                            print(f"  R²:   {metrics['r2']:.6f}")
                            print(f"  KDT:  {metrics['kdt']:.6f}")
                            print(f"  MAPE: {metrics['mape']:.6f}")
                            print(f"  Accuracy to Tolerance (τ={args.acc_tol}): {metrics['accuracy_to_tolerance']:.0f}")
                            print(f"  Max Relative Error: {metrics['max_relative_error']:.6e}")
                            print(f"  Valid Points: {metrics['num_valid_points']}")
                        else:
                            print(f"Numerical Evaluation Failed: {test_result['error']}")
                        print()

                    # Save results
                    if args.output_dir:
                        output_dir = Path(args.output_dir)
                    else:
                        from core.consts import ROOT_DIR
                        output_dir = Path(ROOT_DIR) / "evaluation_results"

                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_file = output_dir / f"{args.dataset_name}_eval.json"

                    evaluator.save_results(eval_results, output_file)
                    print(f"Evaluation results saved to: {output_file}")
                    print()

                except ImportError:
                    print("Warning: Evaluation module not found. Install with:")
                    print("  pip install numpy scipy scikit-learn sympy openai pandas")
                except Exception as eval_error:
                    print(f"Evaluation error: {eval_error}")
                    import traceback
                    traceback.print_exc()
            else:
                print("Warning: Could not extract discovered equation from agent output")
                print()

        # Display workflow metadata
        print("=" * 60)
        print(f"Workflow completed. Visited nodes: {result_state.get('_visited_nodes', [])}")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
