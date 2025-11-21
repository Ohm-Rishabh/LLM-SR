"""
SRFlow: Symbolic Regression Workflow

This module implements a workflow for symbolic regression tasks.
It includes data preprocessing, tool selection, and execution nodes.
"""

from __future__ import annotations
import os
import json
import numpy as np
from typing import Any, Dict, List, Optional

from core.workflow import Workflow
from core.node import LLMNode, ToolNode
from nodes import ToolSwitchNode


def preprocess_data_files(data_path: str, output_dir: str = "tmp") -> str:
    """
    Preprocess raw data files and convert them to CSV format.

    This function handles various data formats (.npy, .h5, etc.) and
    converts them to CSV for easier processing by symbolic regression tools.

    Args:
        data_path: Path to the raw data file.
        output_dir: Directory where the CSV file will be saved.

    Returns:
        Path to the saved CSV file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Determine file type and load data
    if data_path.endswith('.npy'):
        data = np.load(data_path)
    elif data_path.endswith('.npz'):
        npz_data = np.load(data_path)
        # Assume 'data' key or use the first array
        data = npz_data['data'] if 'data' in npz_data else npz_data[npz_data.files[0]]
    elif data_path.endswith('.h5') or data_path.endswith('.hdf5'):
        try:
            import h5py
            with h5py.File(data_path, 'r') as f:
                # Assume 'data' key or use the first dataset
                key = 'data' if 'data' in f else list(f.keys())[0]
                data = f[key][:]
        except ImportError:
            raise ImportError("h5py is required to read HDF5 files. Install with: pip install h5py")
    elif data_path.endswith('.csv'):
        # Already CSV, just return the path
        return data_path
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

    # Convert to CSV
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    csv_path = os.path.join(output_dir, f"{base_name}.csv")

    # Handle different array shapes
    if len(data.shape) == 1:
        # 1D array - save as single column
        np.savetxt(csv_path, data.reshape(-1, 1), delimiter=',', header='value', comments='')
    elif len(data.shape) == 2:
        # 2D array - save as-is
        # Generate column headers
        headers = ','.join([f'col_{i}' for i in range(data.shape[1])])
        np.savetxt(csv_path, data, delimiter=',', header=headers, comments='')
    else:
        raise ValueError(f"Unsupported array shape: {data.shape}")

    return csv_path


def linear_regression_tool(csv_path: str) -> Dict[str, Any]:
    """
    Dummy tool that performs simple linear regression on the data.

    This is a placeholder tool that demonstrates the tool execution pattern.
    In a real implementation, this would be replaced with actual symbolic
    regression tools.

    Args:
        csv_path: Path to the CSV data file.

    Returns:
        Dictionary containing the regression results.
    """
    # Load the CSV data
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)

    # If 1D, reshape to 2D
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    # Assume last column is the target, rest are features
    if data.shape[1] > 1:
        X = data[:, :-1]
        y = data[:, -1]
    else:
        # Single column - create dummy X
        X = np.arange(len(data)).reshape(-1, 1)
        y = data[:, 0]

    # Perform simple linear regression
    # For multiple features, use the first feature only (for simplicity)
    if X.shape[1] > 0:
        x = X[:, 0]
    else:
        x = X.flatten()

    # Calculate slope and intercept
    n = len(x)
    slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
    intercept = (np.sum(y) - slope * np.sum(x)) / n

    # Calculate R-squared
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return {
        "equation": f"y = {slope:.4f} * x + {intercept:.4f}",
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "tool": "linear_regression"
    }


class SRFlow(Workflow):
    """
    Symbolic Regression Workflow.

    This workflow implements a complete pipeline for symbolic regression:
    1. Data preprocessing: Convert raw data to CSV format
    2. Tool selection: Choose appropriate SR tool based on data characteristics
    3. Tool execution: Run the selected symbolic regression tool
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
    ):
        """
        Initialize the symbolic regression workflow.

        Args:
            preprocess_model: Model to use for data preprocessing guidance.
            tool_switch_model: Model to use for tool selection.
        """
        super().__init__()

        self.model = model

        # Build the workflow
        self._build_sr_workflow()

    def _build_sr_workflow(self) -> None:
        """
        Build the symbolic regression workflow graph.

        The workflow consists of:
        1. Data preprocessing node (LLM-guided)
        2. Data conversion tool node
        3. Tool switch node (selects SR tool)
        4. Tool execution nodes (one per available tool)
        """

        # Node 1: Data Preprocessing LLM Node
        # This node analyzes the raw data and determines preprocessing steps
        preprocess_llm = LLMNode(
            name="data_preprocess_llm",
            system_prompt="data_preprocessor",
            input_keys=["data_path", "task_description"],
            output_key="preprocess_plan",
            model=self.model,
            parse_json=True,
            description="Analyze data and plan preprocessing steps"
        )

        # Node 2: Data Conversion Tool Node
        # This node executes the actual data conversion
        data_conversion_tool = ToolNode(
            name="data_conversion",
            tool_fn=preprocess_data_files,
            input_keys=["data_path"],
            output_key="csv_path",
            description="Convert raw data to CSV format"
        )

        # Node 3: Tool Switch Node
        # This node decides which SR tool to use
        tool_switch = ToolSwitchNode(
            name="tool_switch",
            system_prompt="tool_selector",
            input_keys=["csv_path", "task_description"],
            model=self.model,
            available_tools=["linear_regression"],
            description="Select appropriate symbolic regression tool"
        )

        # Node 4: Tool Execution Nodes
        # Linear regression tool
        linear_regression_node = ToolNode(
            name="tool_linear_regression",
            tool_fn=linear_regression_tool,
            input_keys=["csv_path"],
            output_key="sr_result",
            description="Execute linear regression"
        )

        # Add all nodes to the workflow
        self.add_node(preprocess_llm, is_start=True)
        self.add_node(data_conversion_tool)
        self.add_node(tool_switch)
        self.add_node(linear_regression_node)

        # Define edges
        self.add_edge("data_preprocess_llm", "data_conversion")
        self.add_edge("data_conversion", "tool_switch")
        self.add_edge("tool_switch", "tool_linear_regression")

    def run_symbolic_regression(
        self,
        data_path: str,
        task_description: str = "Perform symbolic regression on the provided data.",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the symbolic regression workflow on the given data.

        Args:
            data_path: Path to the raw data file (.npy, .h5, .csv, etc.).
            task_description: Description of the symbolic regression task.
            **kwargs: Additional arguments passed to workflow.run().

        Returns:
            Final state dictionary containing the symbolic regression results.
        """
        initial_state = {
            "data_path": data_path,
            "task_description": task_description
        }

        return self.run(initial_state=initial_state, **kwargs)


# Example usage
if __name__ == "__main__":
    # Create the workflow
    sr_workflow = SRFlow()

    # Example: Run with sample data
    # result = sr_workflow.run_symbolic_regression(
    #     data_path="data/sample.npy",
    #     task_description="Find a mathematical equation that fits this data."
    # )
    #
    # print("Symbolic Regression Result:")
    # print(result.get("sr_result"))

    print("SRFlow workflow created successfully!")
    print(f"Nodes: {list(sr_workflow.nodes.keys())}")
    print(f"Start node: {sr_workflow.start_node}")
