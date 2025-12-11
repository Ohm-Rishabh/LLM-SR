#!/usr/bin/env python3
"""
Test script for node access to workspace manager and file metadata.

This script tests that:
1. Nodes receive workspace manager when added to workflow
2. Nodes can access workspace files summary
3. Nodes can query workspace files with filters
4. LLM nodes can include file context in prompts
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.workflow import Workflow
from core.node import TransformNode, Node
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_node_workspace_injection():
    """Test that nodes receive workspace manager."""
    print("=" * 60)
    print("Test 1: Workspace Manager Injection")
    print("=" * 60)

    workflow = Workflow(enable_workspace=True)

    # Create a node that checks for workspace manager
    def check_workspace_access(state):
        # This function will be wrapped in TransformNode
        return state

    node = TransformNode(
        name="test_node",
        transform_fn=check_workspace_access,
        description="Test node"
    )

    workflow.add_node(node, is_start=True)

    # Before run, node should not have workspace manager
    assert node.workspace_manager is None, "Node should not have workspace before run"
    print("✓ Node has no workspace manager before workflow run")

    # Run workflow
    result = workflow.run(initial_state={"test": "value"})

    # After run, node should have workspace manager
    assert node.workspace_manager is not None, "Node should have workspace after run"
    print(f"✓ Node received workspace manager: {node.workspace_manager}")

    print("\n✅ Test 1 PASSED\n")
    return workflow, node


def test_node_file_summary_access():
    """Test that nodes can access workspace file summary."""
    print("=" * 60)
    print("Test 2: Node Accessing File Summary")
    print("=" * 60)

    workflow = Workflow(enable_workspace=True)

    # Track what the node sees
    node_observations = {}

    def observe_workspace(state):
        # Access workspace via the node's helper method
        # Note: We need to get the actual node instance
        return state

    node = TransformNode(
        name="observer_node",
        transform_fn=observe_workspace,
        description="Observer node"
    )

    workflow.add_node(node, is_start=True)

    # Run workflow
    result = workflow.run(initial_state={"test": "value"})

    # Node should be able to access file summary
    summary = node.get_workspace_files_summary()
    print(f"✓ Node can access files summary")
    print(f"  Summary: {summary}")

    assert summary is not None, "Node should get files summary"
    assert "Workspace" in summary, "Summary should contain workspace info"

    print("\n✅ Test 2 PASSED\n")


def test_node_with_registered_files():
    """Test node accessing workspace with registered files."""
    print("=" * 60)
    print("Test 3: Node Accessing Registered Files")
    print("=" * 60)

    workflow = Workflow(enable_workspace=True)

    # Node 1: Creates some files
    def create_files_node(state):
        workspace_mgr = state.get("_workspace_manager")
        if workspace_mgr:
            # Create and register some files
            output_file = workspace_mgr.get_path("output", "analysis.txt")
            output_file.write_text("Analysis results here")
            workspace_mgr.register_file(
                "output/analysis.txt",
                description="Statistical analysis of dataset",
                file_type="result",
                created_by="create_files_node"
            )

            plot_file = workspace_mgr.get_path("output", "plot.png")
            plot_file.write_text("<PNG data>")
            workspace_mgr.register_file(
                "output/plot.png",
                description="Distribution histogram",
                file_type="plot",
                created_by="create_files_node"
            )

            print("  ✓ Node 1 created and registered 2 files")

        return state

    # Node 2: Observes the files
    observations = {}

    def observe_files_node(state):
        return state

    node1 = TransformNode(
        name="creator_node",
        transform_fn=create_files_node,
        description="Creates files"
    )

    node2 = TransformNode(
        name="observer_node",
        transform_fn=observe_files_node,
        description="Observes files"
    )

    workflow.add_node(node1, is_start=True)
    workflow.add_node(node2)
    workflow.add_edge(node1, node2)

    # Run workflow
    result = workflow.run(initial_state={"test": "value"})

    # Node 2 should see the files
    files_summary = node2.get_workspace_files_summary()
    print(f"\n✓ Node 2 sees files summary:")
    print(files_summary)

    assert "analysis.txt" in files_summary, "Should see analysis.txt"
    assert "plot.png" in files_summary, "Should see plot.png"
    assert "Statistical analysis" in files_summary, "Should see description"

    # Test filtering
    output_files = node2.list_workspace_files(subdir="output")
    print(f"\n✓ Node 2 found {len(output_files)} files in output/")
    assert len(output_files) == 2, "Should find 2 files in output/"

    plot_files = node2.list_workspace_files(file_type="plot")
    print(f"✓ Node 2 found {len(plot_files)} plot files")
    assert len(plot_files) == 1, "Should find 1 plot file"
    assert plot_files[0]['description'] == "Distribution histogram"

    print("\n✅ Test 3 PASSED\n")


def test_llm_node_use_case():
    """Test realistic LLM node use case with file context."""
    print("=" * 60)
    print("Test 4: LLM Node Use Case - File Context in Prompts")
    print("=" * 60)

    workflow = Workflow(enable_workspace=True)

    # Simulate a workflow where:
    # 1. Tool creates analysis files
    # 2. LLM node needs to decide what to do next

    def tool_node_func(state):
        """Simulates a tool creating output files."""
        workspace_mgr = state.get("_workspace_manager")
        if workspace_mgr:
            # Create various outputs
            stats_file = workspace_mgr.get_path("output", "stats.json")
            stats_file.write_text('{"mean": 5.2, "std": 1.3}')
            workspace_mgr.register_file(
                "output/stats.json",
                description="Descriptive statistics: mean, std, quartiles",
                file_type="result",
                created_by="python_interpreter"
            )

            corr_file = workspace_mgr.get_path("output", "correlation.png")
            corr_file.write_text("<PNG>")
            workspace_mgr.register_file(
                "output/correlation.png",
                description="Correlation heatmap of all features",
                file_type="plot",
                created_by="python_interpreter"
            )

            print("  ✓ Tool node created analysis files")

        return state

    def llm_decision_node_func(state):
        """Simulates an LLM node making decisions based on available files."""
        return state

    tool_node = TransformNode(
        name="analysis_tool",
        transform_fn=tool_node_func,
        description="Analysis tool"
    )

    llm_node = TransformNode(
        name="decision_maker",
        transform_fn=llm_decision_node_func,
        description="LLM decision maker"
    )

    workflow.add_node(tool_node, is_start=True)
    workflow.add_node(llm_node)
    workflow.add_edge(tool_node, llm_node)

    # Run workflow
    result = workflow.run(initial_state={"user_query": "Analyze the data"})

    # LLM node can now build context-aware prompts
    files_summary = llm_node.get_workspace_files_summary()

    # Simulate building an LLM prompt
    prompt = f"""You are analyzing a dataset. Here are the files available:

{files_summary}

User query: {result.get('user_query')}

Based on the available files, what analysis should be performed next?
"""

    print("\n✓ LLM node can build context-aware prompt:")
    print("-" * 60)
    print(prompt)
    print("-" * 60)

    # Verify the prompt contains useful context
    assert "stats.json" in prompt, "Prompt should mention stats file"
    assert "correlation.png" in prompt, "Prompt should mention correlation plot"
    assert "Descriptive statistics" in prompt, "Prompt should have file description"

    # LLM can also filter for specific file types
    result_files = llm_node.list_workspace_files(file_type="result")
    print(f"\n✓ LLM can see {len(result_files)} result files")
    print(f"  Available for reading: {[f['path'] for f in result_files]}")

    print("\n✅ Test 4 PASSED\n")


def test_multiple_nodes_access():
    """Test that multiple nodes can all access workspace."""
    print("=" * 60)
    print("Test 5: Multiple Nodes Accessing Workspace")
    print("=" * 60)

    workflow = Workflow(enable_workspace=True)

    access_log = []

    def node_func_factory(node_name):
        def node_func(state):
            access_log.append(node_name)
            return state
        return node_func

    # Create 3 nodes
    nodes = []
    for i in range(3):
        node = TransformNode(
            name=f"node_{i}",
            transform_fn=node_func_factory(f"node_{i}"),
            description=f"Node {i}"
        )
        nodes.append(node)

    # Add all nodes
    workflow.add_node(nodes[0], is_start=True)
    for i in range(1, 3):
        workflow.add_node(nodes[i])
        workflow.add_edge(nodes[i-1], nodes[i])

    # Run workflow
    result = workflow.run(initial_state={"test": "value"})

    # All nodes should have workspace manager
    for i, node in enumerate(nodes):
        assert node.workspace_manager is not None, f"Node {i} should have workspace"
        print(f"✓ Node {i} has workspace access")

    print(f"\n✓ All {len(nodes)} nodes executed: {access_log}")
    print("\n✅ Test 5 PASSED\n")


def cleanup_test_directories():
    """Clean up test directories."""
    import shutil

    test_dirs = [
        Path(__file__).parent / "workspaces",
    ]

    for test_dir in test_dirs:
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print(f"Cleaned up: {test_dir}")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("NODE WORKSPACE ACCESS TESTS")
    print("=" * 60 + "\n")

    try:
        # Run tests
        test_node_workspace_injection()
        test_node_file_summary_access()
        test_node_with_registered_files()
        test_llm_node_use_case()
        test_multiple_nodes_access()

        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nNode workspace access is working correctly!")
        print("\nKey Capabilities:")
        print("  ✓ Nodes automatically receive workspace manager")
        print("  ✓ Nodes can access file summaries")
        print("  ✓ Nodes can filter workspace files")
        print("  ✓ LLM nodes can build context-aware prompts")
        print("  ✓ Multiple nodes share workspace access")
        print("\nUsage in nodes:")
        print("  # Get file summary for LLM prompt")
        print("  summary = self.get_workspace_files_summary()")
        print("")
        print("  # List specific file types")
        print("  plots = self.list_workspace_files(file_type='plot')")
        print("")
        print("  # Direct access to workspace manager")
        print("  if self.workspace_manager:")
        print("      self.workspace_manager.register_file(...)")
        print()

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Cleanup test directories
        print("\nCleaning up test directories...")
        cleanup_test_directories()
        print("✓ Cleanup complete\n")


if __name__ == "__main__":
    main()
