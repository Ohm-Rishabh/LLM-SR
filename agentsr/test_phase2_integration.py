#!/usr/bin/env python3
"""
Test script for Phase 2: Workspace Integration with LLM Nodes

This script tests that:
1. LLMNode automatically includes workspace files in prompts
2. SRNode includes workspace files in user prompts
3. ToolSwitchNode passes workspace environment variables to tools
4. System prompts inform nodes about workspace file availability
5. End-to-end workflow with file creation and LLM awareness
"""

import sys
import logging
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.workflow import Workflow
from core.node import TransformNode, LLMNode
from nodes import SRNode, ToolSwitchNode
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s - %(name)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_llm_node_workspace_prompt():
    """Test that LLMNode includes workspace files in prompts."""
    print("=" * 60)
    print("Test 1: LLMNode Workspace Files in Prompt")
    print("=" * 60)

    workflow = Workflow(enable_workspace=True)

    # Node 1: Create some files in workspace
    def create_files(state):
        workspace_mgr = state.get("_workspace_manager")
        if workspace_mgr:
            # Create analysis file
            analysis_file = workspace_mgr.get_path("output", "stats.txt")
            analysis_file.write_text("Mean: 5.2, Std: 1.3")
            workspace_mgr.register_file(
                "output/stats.txt",
                description="Statistical analysis of dataset",
                file_type="result",
                created_by="analysis_tool"
            )

            # Create plot file
            plot_file = workspace_mgr.get_path("output", "scatter.png")
            plot_file.write_text("<PNG data>")
            workspace_mgr.register_file(
                "output/scatter.png",
                description="Scatter plot of features vs target",
                file_type="plot",
                created_by="visualization_tool"
            )

            print("  ✓ Created 2 files in workspace")

        return state

    # Node 2: Mock LLM node that captures its input
    captured_input = {}

    class MockLLMNode(LLMNode):
        """Mock LLM node that captures input instead of calling API."""

        def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
            # Build the input as normal
            prompt = self._build_input(state)

            # Capture the prompt content
            captured_input["user_content"] = prompt

            # Don't actually call API, just return state
            state[self.output_key] = "mock response"
            return state

    creator_node = TransformNode(
        name="file_creator",
        transform_fn=create_files,
        description="Creates files"
    )

    llm_node = MockLLMNode(
        name="test_llm",
        system_prompt="sr_analyzer",
        input_keys=["user_query"],
        output_key="llm_response",
        description="Mock LLM"
    )

    workflow.add_node(creator_node, is_start=True)
    workflow.add_node(llm_node)
    workflow.add_edge(creator_node, llm_node)

    # Run workflow
    result = workflow.run(initial_state={"user_query": "Analyze the data"})

    # Verify workspace files are in the captured input
    user_content = captured_input.get("user_content", "")

    print("\n✓ Captured LLM input:")
    print("-" * 60)
    print(user_content[:500] + "..." if len(user_content) > 500 else user_content)
    print("-" * 60)

    assert "Workspace Files" in user_content, "Workspace files section missing from LLM input"
    assert "stats.txt" in user_content, "stats.txt not in LLM input"
    assert "scatter.png" in user_content, "scatter.png not in LLM input"
    assert "Statistical analysis" in user_content, "File description missing from LLM input"
    assert "Scatter plot" in user_content, "Plot description missing from LLM input"

    print("\n✓ LLM node successfully includes workspace files in prompt")
    print("✅ Test 1 PASSED\n")


def test_sr_node_workspace_prompt():
    """Test that SRNode includes workspace files in user prompts."""
    print("=" * 60)
    print("Test 2: SRNode Workspace Files in User Prompt")
    print("=" * 60)

    workflow = Workflow(enable_workspace=True)

    # Create files
    def create_files(state):
        workspace_mgr = state.get("_workspace_manager")
        if workspace_mgr:
            results_file = workspace_mgr.get_path("output", "pysr_results.json")
            results_file.write_text('{"best_equation": "x^2 + 2*x"}')
            workspace_mgr.register_file(
                "output/pysr_results.json",
                description="PySR discovered equations",
                file_type="result",
                created_by="pysr"
            )
            print("  ✓ Created PySR results file")
        return state

    # Mock SRNode
    captured_input = {}

    class MockSRNode(SRNode):
        """Mock SR node that captures input."""

        def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
            prompt_data = self._build_input(state)

            # Capture user content blocks
            captured_input["user_content"] = prompt_data.get("user_content", [])
            captured_input["system_prompt"] = prompt_data.get("system_prompt", "")

            state[self.output_key] = "mock response"
            return state

    creator_node = TransformNode(
        name="file_creator",
        transform_fn=create_files,
        description="Creates files"
    )

    sr_node = MockSRNode(
        name="test_sr",
        system_prompt="sr_analyzer",
        input_keys=["user_query"],
        output_key="sr_response",
        description="Mock SR node"
    )

    workflow.add_node(creator_node, is_start=True)
    workflow.add_node(sr_node)
    workflow.add_edge(creator_node, sr_node)

    # Run workflow
    result = workflow.run(initial_state={"user_query": "Find best equation"})

    # Verify workspace files in captured content
    user_content = captured_input.get("user_content", [])

    # Convert content blocks to string for inspection
    content_str = str(user_content)

    print("\n✓ Captured SRNode input content blocks:")
    print("-" * 60)
    print(content_str[:500] + "..." if len(content_str) > 500 else content_str)
    print("-" * 60)

    assert "Workspace Files" in content_str, "Workspace files section missing from SR input"
    assert "pysr_results.json" in content_str, "pysr_results.json not in SR input"
    assert "PySR discovered equations" in content_str, "File description missing"

    print("\n✓ SRNode successfully includes workspace files in user prompt")
    print("✅ Test 2 PASSED\n")


def test_toolswitch_workspace_env():
    """Test that ToolSwitchNode passes workspace environment variables."""
    print("=" * 60)
    print("Test 3: ToolSwitchNode Workspace Environment Variables")
    print("=" * 60)

    workflow = Workflow(enable_workspace=True)

    # Create a simple test tool that echoes environment variables
    test_tool_dir = Path(__file__).parent / "tools" / "test_env_tool"
    test_tool_dir.mkdir(parents=True, exist_ok=True)

    # Create run.sh that outputs environment variables
    # Need to use absolute path since ROOT_DIR might not be set in the environment
    result_path = Path(__file__).parent / "result.json"
    run_script = test_tool_dir / "run.sh"
    run_script.write_text(f"""#!/bin/bash
# Test script that outputs workspace environment variables

cat > "{result_path}" << EOF
{{
  "status": "success",
  "workspace_root": "${{WORKSPACE_ROOT:-not_set}}",
  "workspace_input": "${{WORKSPACE_INPUT:-not_set}}",
  "workspace_output": "${{WORKSPACE_OUTPUT:-not_set}}",
  "workspace_logs": "${{WORKSPACE_LOGS:-not_set}}",
  "workspace_scratch": "${{WORKSPACE_SCRATCH:-not_set}}"
}}
EOF
""")
    run_script.chmod(0o755)

    # Create tool spec
    tool_spec = Path(__file__).parent / "tool_specs" / "test_env_tool.md"
    tool_spec.write_text("""# Test Environment Tool

Test tool that checks workspace environment variables.

## Arguments

None required.
""")

    # Create ToolSwitchNode
    def setup_tool_call(state):
        state["tool_call"] = {
            "tool_name": "test_env_tool",
            "args": {}
        }
        return state

    setup_node = TransformNode(
        name="setup",
        transform_fn=setup_tool_call,
        description="Setup tool call"
    )

    tool_node = ToolSwitchNode(
        name="test_tool_node"
    )

    workflow.add_node(setup_node, is_start=True)
    workflow.add_node(tool_node)
    workflow.add_edge(setup_node, tool_node)

    # Run workflow
    result = workflow.run(initial_state={})

    # Check tool result
    tool_result = result.get("tool_result", {})

    print("\n✓ Tool result:")
    print(f"  Status: {tool_result.get('status')}")
    print(f"  WORKSPACE_ROOT: {tool_result.get('workspace_root')}")
    print(f"  WORKSPACE_INPUT: {tool_result.get('workspace_input')}")
    print(f"  WORKSPACE_OUTPUT: {tool_result.get('workspace_output')}")
    print(f"  WORKSPACE_LOGS: {tool_result.get('workspace_logs')}")
    print(f"  WORKSPACE_SCRATCH: {tool_result.get('workspace_scratch')}")

    assert tool_result.get("status") == "success", "Tool execution failed"
    assert tool_result.get("workspace_root") != "not_set", "WORKSPACE_ROOT not set"
    assert tool_result.get("workspace_input") != "not_set", "WORKSPACE_INPUT not set"
    assert tool_result.get("workspace_output") != "not_set", "WORKSPACE_OUTPUT not set"
    assert "workspaces" in tool_result.get("workspace_root", ""), "workspace_root path invalid"

    # Cleanup test tool
    import shutil
    shutil.rmtree(test_tool_dir)
    tool_spec.unlink()

    print("\n✓ ToolSwitchNode successfully passes workspace environment variables")
    print("✅ Test 3 PASSED\n")


def test_system_prompts_updated():
    """Test that system prompts mention workspace file information."""
    print("=" * 60)
    print("Test 4: System Prompts Updated with Workspace Info")
    print("=" * 60)

    prompts_dir = Path(__file__).parent / "prompts"

    # Check sr_analyzer.md
    sr_analyzer = prompts_dir / "sr_analyzer.md"
    if sr_analyzer.exists():
        content = sr_analyzer.read_text()
        assert "Workspace Files" in content, "sr_analyzer.md missing Workspace Files section"
        assert "workspace files" in content.lower(), "sr_analyzer.md doesn't mention workspace files"
        print("✓ sr_analyzer.md updated with workspace information")
    else:
        print("⚠ sr_analyzer.md not found, skipping check")

    # Check tool_selector.md
    tool_selector = prompts_dir / "tool_selector.md"
    if tool_selector.exists():
        content = tool_selector.read_text()
        assert "Workspace Files" in content, "tool_selector.md missing Workspace Files section"
        assert "workspace files" in content.lower(), "tool_selector.md doesn't mention workspace files"
        print("✓ tool_selector.md updated with workspace information")
    else:
        print("⚠ tool_selector.md not found, skipping check")

    print("\n✅ Test 4 PASSED\n")


def test_end_to_end_workflow():
    """Test complete workflow with file creation and LLM awareness."""
    print("=" * 60)
    print("Test 5: End-to-End Workflow with Workspace Integration")
    print("=" * 60)

    workflow = Workflow(enable_workspace=True)

    # Step 1: Tool creates files
    def tool_creates_files(state):
        workspace_mgr = state.get("_workspace_manager")
        if workspace_mgr:
            # Create multiple analysis files
            stats = workspace_mgr.get_path("output", "statistics.json")
            stats.write_text('{"mean": 10.5, "std": 2.3}')
            workspace_mgr.register_file(
                "output/statistics.json",
                description="Descriptive statistics of all features",
                file_type="result",
                created_by="python_interpreter"
            )

            correlation = workspace_mgr.get_path("output", "correlation.png")
            correlation.write_text("<PNG>")
            workspace_mgr.register_file(
                "output/correlation.png",
                description="Correlation matrix heatmap",
                file_type="plot",
                created_by="python_interpreter"
            )

            print("  ✓ Step 1: Tool created analysis files")
        return state

    # Step 2: LLM sees files and makes decision
    llm_observations = {}

    class ObservingLLMNode(LLMNode):
        def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
            prompt = self._build_input(state)

            llm_observations["has_workspace_section"] = "Workspace Files" in prompt
            llm_observations["sees_statistics"] = "statistics.json" in prompt
            llm_observations["sees_correlation"] = "correlation.png" in prompt

            print("  ✓ Step 2: LLM observed workspace files")
            state[self.output_key] = "decision made"
            return state

    tool_node = TransformNode(
        name="tool_node",
        transform_fn=tool_creates_files,
        description="Tool that creates files"
    )

    llm_node = ObservingLLMNode(
        name="decision_node",
        system_prompt="sr_analyzer",
        input_keys=["query"],
        output_key="decision",
        description="LLM decision maker"
    )

    workflow.add_node(tool_node, is_start=True)
    workflow.add_node(llm_node)
    workflow.add_edge(tool_node, llm_node)

    # Run complete workflow
    result = workflow.run(initial_state={"query": "Analyze dataset"})

    # Verify LLM saw workspace files
    print("\n✓ LLM observations:")
    print(f"  Has workspace section: {llm_observations.get('has_workspace_section')}")
    print(f"  Sees statistics.json: {llm_observations.get('sees_statistics')}")
    print(f"  Sees correlation.png: {llm_observations.get('sees_correlation')}")

    assert llm_observations.get("has_workspace_section"), "LLM didn't see workspace section"
    assert llm_observations.get("sees_statistics"), "LLM didn't see statistics file"
    assert llm_observations.get("sees_correlation"), "LLM didn't see correlation plot"

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
    """Run all Phase 2 integration tests."""
    print("\n" + "=" * 60)
    print("PHASE 2 INTEGRATION TESTS")
    print("Workspace Integration with LLM Nodes")
    print("=" * 60 + "\n")

    try:
        # Run tests
        test_llm_node_workspace_prompt()
        test_sr_node_workspace_prompt()
        test_toolswitch_workspace_env()
        test_system_prompts_updated()
        test_end_to_end_workflow()

        print("=" * 60)
        print("✅ ALL PHASE 2 TESTS PASSED")
        print("=" * 60)
        print("\nPhase 2 Integration Complete!")
        print("\nKey Features Tested:")
        print("  ✓ LLMNode automatically includes workspace files in prompts")
        print("  ✓ SRNode includes workspace files in user prompts")
        print("  ✓ ToolSwitchNode passes workspace environment variables")
        print("  ✓ System prompts updated to inform about workspace files")
        print("  ✓ End-to-end workflow with file awareness works")
        print("\nWhat's Next:")
        print("  - Phase 3: Update tool implementations to use workspace")
        print("  - Phase 4: Update tool specifications with workspace docs")
        print("  - Phase 5: Full integration testing and migration")
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
        # Cleanup
        print("\nCleaning up test directories...")
        cleanup_test_directories()
        print("✓ Cleanup complete\n")


if __name__ == "__main__":
    main()
