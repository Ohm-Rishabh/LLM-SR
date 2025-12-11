#!/usr/bin/env python3
"""
Test script for Phase 1: Workspace infrastructure.

This script tests the basic workspace functionality:
1. WorkspaceManager initialization
2. Workspace directory creation
3. File operations
4. Workflow integration
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.workspace import WorkspaceManager
from core.workflow import Workflow
from core.node import TransformNode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_workspace_manager():
    """Test WorkspaceManager basic functionality."""
    print("=" * 60)
    print("Test 1: WorkspaceManager Basic Functionality")
    print("=" * 60)

    # Create workspace manager
    base_dir = Path(__file__).parent / "test_workspaces"
    manager = WorkspaceManager(base_dir, run_id="test_run_001")

    print(f"✓ Created WorkspaceManager: {manager}")

    # Create workspace
    workspace_path = manager.create_workspace()
    print(f"✓ Created workspace at: {workspace_path}")

    # Check subdirectories
    for subdir in ["input", "output", "logs", "scratch"]:
        subdir_path = workspace_path / subdir
        assert subdir_path.exists(), f"Missing subdirectory: {subdir}"
        print(f"  ✓ {subdir}/ exists")

    # Test file operations
    test_content = "Hello, workspace!"
    output_file = manager.write_file(test_content, "output", "test.txt")
    print(f"✓ Wrote test file: {output_file}")

    read_content = manager.read_file("output", "test.txt")
    assert read_content == test_content, "File content mismatch"
    print(f"✓ Read test file successfully")

    # Test list files
    files = manager.list_files("output", "*.txt")
    assert len(files) == 1, "Expected 1 file"
    print(f"✓ Listed files: {files}")

    # Test get_info
    info = manager.get_info()
    print(f"✓ Workspace info: run_id={info['run_id']}, exists={info['exists']}")

    print("\n✅ Test 1 PASSED\n")
    return manager


def test_workflow_integration():
    """Test Workflow integration with workspace."""
    print("=" * 60)
    print("Test 2: Workflow Integration")
    print("=" * 60)

    # Create a simple workflow with workspace enabled
    workflow = Workflow(enable_workspace=True)

    # Add a simple transform node
    def simple_transform(state):
        print(f"  Node executing in workspace: {state.get('_workspace_root', 'N/A')}")
        state["message"] = "Workflow executed successfully"
        return state

    node = TransformNode(
        name="test_node",
        transform_fn=simple_transform,
        description="Test node"
    )

    workflow.add_node(node, is_start=True)

    # Run workflow
    initial_state = {
        "test_input": "test_value"
    }

    result = workflow.run(initial_state)

    # Verify workspace was created
    assert "_workspace_root" in result, "Workspace root not in state"
    assert "_workspace_path" in result, "Workspace path not in state"
    assert "_workspace_manager" in result, "Workspace manager not in state"

    workspace_root = Path(result["_workspace_root"])
    assert workspace_root.exists(), f"Workspace directory not created: {workspace_root}"

    print(f"✓ Workspace created: {workspace_root}")
    print(f"✓ State contains workspace info")
    print(f"✓ Message: {result['message']}")

    print("\n✅ Test 2 PASSED\n")
    return result


def test_workflow_with_data_file():
    """Test Workflow with data file copying."""
    print("=" * 60)
    print("Test 3: Workflow with Data File")
    print("=" * 60)

    # Create a test data file
    test_data_dir = Path(__file__).parent / "test_data"
    test_data_dir.mkdir(exist_ok=True)
    test_data_file = test_data_dir / "sample_data.csv"

    # Write sample CSV
    csv_content = "x,y\n1,2\n3,4\n5,6\n"
    test_data_file.write_text(csv_content)
    print(f"✓ Created test data file: {test_data_file}")

    # Create workflow
    workflow = Workflow(enable_workspace=True)

    def check_data_file(state):
        data_file = state.get("data_file")
        print(f"  Data file in state: {data_file}")

        # Verify file is in workspace
        if "_workspace_root" in state:
            workspace_root = Path(state["_workspace_root"])
            expected_path = workspace_root / "input" / "data.csv"
            assert Path(data_file) == expected_path, f"Data file not in workspace: {data_file}"
            assert Path(data_file).exists(), f"Data file doesn't exist: {data_file}"
            print(f"  ✓ Data file copied to workspace: {expected_path}")

        return state

    node = TransformNode(
        name="check_node",
        transform_fn=check_data_file,
        description="Check data file"
    )

    workflow.add_node(node, is_start=True)

    # Run workflow with data file
    initial_state = {
        "data_file": str(test_data_file)
    }

    result = workflow.run(initial_state)

    # Verify data file was copied
    data_file_in_workspace = Path(result["data_file"])
    assert data_file_in_workspace.exists(), "Data file not found in workspace"
    assert "input" in str(data_file_in_workspace), "Data file not in input directory"

    print(f"✓ Data file in workspace: {data_file_in_workspace}")
    print(f"✓ Original file: {test_data_file}")

    print("\n✅ Test 3 PASSED\n")
    return result


def test_workspace_cleanup():
    """Test workspace cleanup functionality."""
    print("=" * 60)
    print("Test 4: Workspace Cleanup")
    print("=" * 60)

    base_dir = Path(__file__).parent / "test_workspaces_cleanup"

    # Create multiple workspaces
    managers = []
    for i in range(5):
        manager = WorkspaceManager(base_dir, run_id=f"cleanup_test_{i:03d}")
        manager.create_workspace()
        managers.append(manager)
        print(f"✓ Created workspace {i+1}/5")

    # List workspaces
    all_workspaces = sorted(base_dir.glob("*"))
    print(f"✓ Total workspaces before cleanup: {len(all_workspaces)}")
    assert len(all_workspaces) == 5, "Expected 5 workspaces"

    # Cleanup, keep only 2
    removed = WorkspaceManager.cleanup_old_workspaces(base_dir, keep_last_n=2)
    print(f"✓ Removed {removed} old workspaces")

    # Verify only 2 remain
    remaining = sorted(base_dir.glob("*"))
    print(f"✓ Remaining workspaces: {len(remaining)}")
    assert len(remaining) == 2, "Expected 2 workspaces to remain"

    print("\n✅ Test 4 PASSED\n")


def cleanup_test_directories():
    """Clean up test directories."""
    import shutil

    test_dirs = [
        Path(__file__).parent / "test_workspaces",
        Path(__file__).parent / "test_workspaces_cleanup",
        Path(__file__).parent / "test_data",
    ]

    for test_dir in test_dirs:
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print(f"Cleaned up: {test_dir}")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PHASE 1 WORKSPACE TESTS")
    print("=" * 60 + "\n")

    try:
        # Run tests
        test_workspace_manager()
        test_workflow_integration()
        test_workflow_with_data_file()
        test_workspace_cleanup()

        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nPhase 1 implementation is complete and working correctly!")
        print("\nNext steps:")
        print("  - Phase 2: Update ToolSwitchNode with workspace environment variables")
        print("  - Phase 3: Update tool implementations (pysr, python_interpreter)")
        print("  - Phase 4: Update tool specifications")
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
