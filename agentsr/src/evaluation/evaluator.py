"""
Unified Evaluation Module for Symbolic Regression Agent.

This module provides a high-level interface for evaluating discovered equations
against ground truth, combining both symbolic and numerical evaluation.
"""

import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from .symbolic_evaluator import SymbolicAccuracyEvaluator
from .numerical_evaluator import NumericalEvaluator

logger = logging.getLogger(__name__)


class SRAgentEvaluator:
    """
    Unified evaluator for symbolic regression agent results.

    This class combines symbolic accuracy evaluation (using GPT-4o) and
    numerical precision evaluation to provide comprehensive assessment of
    discovered equations.
    """

    def __init__(
        self,
        symbolic_model: str = "gpt-4o",
        symbolic_temperature: float = 0.0,
        api_key: Optional[str] = None
    ):
        """
        Initialize the SR agent evaluator.

        Args:
            symbolic_model: Model to use for symbolic evaluation (default: gpt-4o)
            symbolic_temperature: Temperature for symbolic evaluation LLM
            api_key: OpenAI API key (if None, uses environment variable)
        """
        self.symbolic_evaluator = SymbolicAccuracyEvaluator(
            model=symbolic_model,
            temperature=symbolic_temperature,
            api_key=api_key
        )
        self.numerical_evaluator = NumericalEvaluator()

    def evaluate(
        self,
        discovered_equation: str,
        ground_truth_equation: str,
        test_data: np.ndarray,
        symbols: List[str],
        train_data: Optional[np.ndarray] = None,
        ood_test_data: Optional[np.ndarray] = None,
        check_symbolic: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a discovered equation against ground truth.

        Args:
            discovered_equation: The equation discovered by the agent
            ground_truth_equation: The ground truth equation
            test_data: Test data array of shape (n_samples, n_features + 1)
                      First column is output, remaining are inputs
            symbols: List of variable names (output first, then inputs)
            train_data: Optional training data for additional metrics
            ood_test_data: Optional out-of-distribution test data
            check_symbolic: Whether to check symbolic equivalence (default: True)

        Returns:
            Dictionary containing:
                - symbolic_accuracy: Results from symbolic evaluation (if enabled)
                - numerical_metrics: Results from numerical evaluation
                - summary: High-level summary of results
        """
        logger.info("=" * 60)
        logger.info("Starting SR Agent Evaluation")
        logger.info("=" * 60)
        logger.info(f"Discovered equation: {discovered_equation}")
        logger.info(f"Ground truth equation: {ground_truth_equation}")
        logger.info(f"Test data shape: {test_data.shape}")

        results = {
            "discovered_equation": discovered_equation,
            "ground_truth_equation": ground_truth_equation,
        }

        # Symbolic accuracy evaluation
        if check_symbolic:
            logger.info("\n--- Symbolic Accuracy Evaluation ---")
            symbolic_results = self.symbolic_evaluator.evaluate(
                ground_truth=ground_truth_equation,
                hypothesis=discovered_equation
            )
            results["symbolic_accuracy"] = symbolic_results
        else:
            results["symbolic_accuracy"] = None

        # Numerical precision evaluation
        logger.info("\n--- Numerical Precision Evaluation ---")

        # Prepare data for evaluation
        eval_data = {"test": test_data}
        if train_data is not None:
            eval_data["train"] = train_data
        if ood_test_data is not None:
            eval_data["ood_test"] = ood_test_data

        numerical_results = {}
        input_symbols = symbols[1:] if len(symbols) > 1 else symbols

        for split_name, data in eval_data.items():
            logger.info(f"\nEvaluating on {split_name} set...")
            X = data[:, 1:]
            y = data[:, 0]

            eval_result = self.numerical_evaluator.evaluate_equation_string(
                equation_str=discovered_equation,
                X=X,
                y_true=y,
                symbols=input_symbols
            )
            numerical_results[split_name] = eval_result

        results["numerical_metrics"] = numerical_results

        # Create summary
        results["summary"] = self._create_summary(results)

        logger.info("\n" + "=" * 60)
        logger.info("Evaluation Complete")
        logger.info("=" * 60)

        return results

    def evaluate_from_agent_output(
        self,
        agent_output: Dict[str, Any],
        dataset_metadata: Dict[str, Any],
        test_data: np.ndarray,
        train_data: Optional[np.ndarray] = None,
        ood_test_data: Optional[np.ndarray] = None,
        check_symbolic: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate agent output using dataset metadata.

        This is a convenience method that extracts the necessary information
        from the agent's output and dataset metadata.

        Args:
            agent_output: Output from the agent (should contain "final_result")
            dataset_metadata: Metadata from the dataset (contains ground truth)
            test_data: Test data array
            train_data: Optional training data
            ood_test_data: Optional OOD test data
            check_symbolic: Whether to check symbolic equivalence

        Returns:
            Evaluation results dictionary
        """
        # Extract discovered equation from agent output
        discovered_equation = self._extract_discovered_equation(agent_output)

        # Extract ground truth from metadata
        ground_truth_equation = dataset_metadata.get('expression', None)
        if ground_truth_equation is None:
            logger.warning("No ground truth expression found in metadata")
            ground_truth_equation = "unknown"

        # Extract symbols
        symbols = dataset_metadata.get('symbols', [])

        return self.evaluate(
            discovered_equation=discovered_equation,
            ground_truth_equation=ground_truth_equation,
            test_data=test_data,
            symbols=symbols,
            train_data=train_data,
            ood_test_data=ood_test_data,
            check_symbolic=check_symbolic
        )

    def _extract_discovered_equation(self, agent_output: Dict[str, Any]) -> str:
        """
        Extract the discovered equation from agent output.

        The agent may output in different formats:
        - Direct dictionary with "final_result"
        - Nested in "parsed_json"
        - String that needs parsing

        Args:
            agent_output: Output from the agent

        Returns:
            Discovered equation string
        """
        # Try direct access
        if "final_result" in agent_output:
            return agent_output["final_result"]

        # Try parsed_json
        if "parsed_json" in agent_output:
            parsed = agent_output["parsed_json"]
            if isinstance(parsed, dict) and "final_result" in parsed:
                return parsed["final_result"]

        # Try to parse if it's a string
        if isinstance(agent_output, str):
            return agent_output

        logger.warning(f"Could not extract discovered equation from agent output: {agent_output}")
        return "unknown"

    def _create_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a high-level summary of evaluation results.

        Args:
            results: Full evaluation results

        Returns:
            Summary dictionary
        """
        summary = {}

        # Symbolic accuracy summary
        if results["symbolic_accuracy"] is not None:
            summary["is_symbolically_equivalent"] = results["symbolic_accuracy"]["is_equivalent"]
        else:
            summary["is_symbolically_equivalent"] = None

        # Numerical metrics summary (focus on test set)
        if "test" in results["numerical_metrics"]:
            test_result = results["numerical_metrics"]["test"]
            if test_result["success"]:
                metrics = test_result["metrics"]
                summary["test_metrics"] = {
                    "mse": metrics["mse"],
                    "nmse": metrics["nmse"],
                    "r2": metrics["r2"],
                    "num_valid_points": metrics["num_valid_points"]
                }
            else:
                summary["test_metrics"] = None
                summary["test_error"] = test_result["error"]
        else:
            summary["test_metrics"] = None

        return summary

    def save_results(self, results: Dict[str, Any], output_path: Path):
        """
        Save evaluation results to a JSON file.

        Args:
            results: Evaluation results dictionary
            output_path: Path to save the results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to Python types for JSON serialization
        results_serializable = self._make_serializable(results)

        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        logger.info(f"Evaluation results saved to {output_path}")

    def _make_serializable(self, obj):
        """
        Convert numpy types to Python types for JSON serialization.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
