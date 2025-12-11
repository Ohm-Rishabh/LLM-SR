#!/usr/bin/env python3
"""
Test script for file metadata functionality in WorkspaceManager.

This script tests the enhanced workspace features:
1. File metadata registration
2. File description tracking
3. Metadata retrieval and listing
4. LLM-friendly file summaries
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.workspace import WorkspaceManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_file_registration():
    """Test file registration with metadata."""
    print("=" * 60)
    print("Test 1: File Registration with Metadata")
    print("=" * 60)

    base_dir = Path(__file__).parent / "test_metadata_workspace"
    manager = WorkspaceManager(base_dir, run_id="metadata_test")
    manager.create_workspace()

    # Create some test files
    test_files = [
        ("input/data.csv", "data", "Original dataset with 5 features and 1000 samples"),
        ("output/correlation.png", "plot", "Correlation heatmap showing feature relationships"),
        ("output/results.json", "result", "Statistical analysis results including mean, std, and quartiles"),
        ("scratch/temp_normalized.csv", "data", "Normalized data for intermediate processing"),
    ]

    for file_path, file_type, description in test_files:
        # Create dummy file
        full_path = manager.get_path(file_path)
        full_path.write_text(f"Dummy content for {file_path}")

        # Register with metadata
        manager.register_file(
            file_path,
            description=description,
            file_type=file_type,
            created_by="test_script"
        )
        print(f"✓ Registered: {file_path}")

    print(f"\n✓ Registered {len(test_files)} files with metadata")
    print("\n✅ Test 1 PASSED\n")
    return manager


def test_metadata_retrieval(manager):
    """Test retrieving metadata for individual files."""
    print("=" * 60)
    print("Test 2: Metadata Retrieval")
    print("=" * 60)

    # Test getting metadata for specific file
    metadata = manager.get_file_metadata("output/correlation.png")
    assert metadata is not None, "Metadata not found"
    assert metadata["description"] == "Correlation heatmap showing feature relationships"
    assert metadata["file_type"] == "plot"
    assert metadata["created_by"] == "test_script"
    assert "registered_at" in metadata
    assert "size_bytes" in metadata

    print(f"✓ Retrieved metadata for output/correlation.png")
    print(f"  Description: {metadata['description']}")
    print(f"  Type: {metadata['file_type']}")
    print(f"  Size: {metadata['size_bytes']} bytes")
    print(f"  Created by: {metadata['created_by']}")

    print("\n✅ Test 2 PASSED\n")


def test_list_with_filters(manager):
    """Test listing files with filters."""
    print("=" * 60)
    print("Test 3: List Files with Filters")
    print("=" * 60)

    # List all files in output directory
    output_files = manager.list_files_with_metadata(subdir="output")
    print(f"✓ Found {len(output_files)} files in output/")
    for file_info in output_files:
        print(f"  - {file_info['path']}: {file_info['description']}")

    assert len(output_files) == 2, "Expected 2 files in output/"

    # List only plot files
    plot_files = manager.list_files_with_metadata(file_type="plot")
    print(f"\n✓ Found {len(plot_files)} plot files")
    for file_info in plot_files:
        print(f"  - {file_info['path']}: {file_info['description']}")

    assert len(plot_files) == 1, "Expected 1 plot file"

    # List only data files
    data_files = manager.list_files_with_metadata(file_type="data")
    print(f"\n✓ Found {len(data_files)} data files")
    for file_info in data_files:
        print(f"  - {file_info['path']}: {file_info['description']}")

    assert len(data_files) == 2, "Expected 2 data files"

    print("\n✅ Test 3 PASSED\n")


def test_files_summary(manager):
    """Test getting LLM-friendly file summary."""
    print("=" * 60)
    print("Test 4: LLM-Friendly Files Summary")
    print("=" * 60)

    summary = manager.get_files_summary()
    print(summary)

    # Verify summary contains expected sections
    assert "input/" in summary
    assert "output/" in summary
    assert "scratch/" in summary
    assert "correlation.png" in summary
    assert "Correlation heatmap" in summary

    print("\n✓ Summary is well-formatted and contains all files")
    print("\n✅ Test 4 PASSED\n")


def test_metadata_persistence(base_dir):
    """Test that metadata persists across WorkspaceManager instances."""
    print("=" * 60)
    print("Test 5: Metadata Persistence")
    print("=" * 60)

    # Create new manager instance for same workspace
    manager2 = WorkspaceManager(base_dir, run_id="metadata_test")
    manager2.create_workspace()  # This should load existing metadata

    # Check that metadata is still available
    files = manager2.list_files_with_metadata()
    print(f"✓ Loaded {len(files)} files from persisted metadata")

    assert len(files) == 4, "Expected 4 files in persisted metadata"

    # Verify specific file
    metadata = manager2.get_file_metadata("output/correlation.png")
    assert metadata is not None, "Metadata not persisted"
    assert metadata["description"] == "Correlation heatmap showing feature relationships"

    print("✓ Metadata correctly persisted and reloaded")
    print("\n✅ Test 5 PASSED\n")


def test_copy_with_description():
    """Test copying files with automatic metadata registration."""
    print("=" * 60)
    print("Test 6: Copy File with Description")
    print("=" * 60)

    base_dir = Path(__file__).parent / "test_metadata_copy"
    manager = WorkspaceManager(base_dir, run_id="copy_test")
    manager.create_workspace()

    # Create a source file
    src_file = Path(__file__).parent / "test_source_data.csv"
    src_file.write_text("x,y\n1,2\n3,4\n")

    # Copy with description
    dst_path = manager.copy_input_file(
        src_file,
        "data.csv",
        description="Test dataset with 2 features and 2 samples"
    )

    print(f"✓ Copied file to: {dst_path}")

    # Verify metadata was registered
    metadata = manager.get_file_metadata("input/data.csv")
    assert metadata is not None, "Metadata not auto-registered"
    assert metadata["description"] == "Test dataset with 2 features and 2 samples"
    assert metadata["file_type"] == "data"
    assert "source_path" in metadata

    print(f"✓ Metadata auto-registered:")
    print(f"  Description: {metadata['description']}")
    print(f"  Source: {metadata['source_path']}")

    # Cleanup
    src_file.unlink()

    print("\n✅ Test 6 PASSED\n")


def test_use_case_llm_access():
    """Test use case: LLM node accessing workspace files."""
    print("=" * 60)
    print("Test 7: LLM Node Use Case")
    print("=" * 60)

    base_dir = Path(__file__).parent / "test_llm_usecase"
    manager = WorkspaceManager(base_dir, run_id="llm_test")
    manager.create_workspace()

    # Simulate tools creating files with metadata
    print("\nSimulating tool outputs:")

    # Tool 1: Python interpreter creates analysis
    analysis_file = manager.get_path("output", "statistical_analysis.txt")
    analysis_file.write_text("Mean: 5.2, Std: 1.3, Skew: 0.8")
    manager.register_file(
        "output/statistical_analysis.txt",
        description="Basic statistical analysis: mean, std, skewness",
        file_type="result",
        created_by="python_interpreter"
    )
    print("  ✓ python_interpreter created statistical_analysis.txt")

    # Tool 2: Python interpreter creates plot
    plot_file = manager.get_path("output", "distribution.png")
    plot_file.write_text("<PNG data>")
    manager.register_file(
        "output/distribution.png",
        description="Histogram showing distribution of target variable",
        file_type="plot",
        created_by="python_interpreter"
    )
    print("  ✓ python_interpreter created distribution.png")

    # Tool 3: PySR creates results
    pysr_file = manager.get_path("output", "pysr_equations.json")
    pysr_file.write_text('{"best": "x^2 + 2*x + 1"}')
    manager.register_file(
        "output/pysr_equations.json",
        description="PySR discovered equations ranked by complexity and accuracy",
        file_type="result",
        created_by="pysr"
    )
    print("  ✓ pysr created pysr_equations.json")

    # Now LLM node wants to know what files are available
    print("\nLLM node requesting file summary:")
    summary = manager.get_files_summary()
    print(summary)

    # LLM can now make informed decisions like:
    # "I should read the statistical_analysis.txt to inform my next SR tool call"
    # "I can see there's a distribution plot available - should I create another?"

    print("✓ LLM has clear visibility into all workspace files")
    print("✓ LLM can make informed decisions about which files to read/use")

    print("\n✅ Test 7 PASSED\n")


def cleanup_test_directories():
    """Clean up test directories."""
    import shutil

    test_dirs = [
        Path(__file__).parent / "test_metadata_workspace",
        Path(__file__).parent / "test_metadata_copy",
        Path(__file__).parent / "test_llm_usecase",
    ]

    for test_dir in test_dirs:
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print(f"Cleaned up: {test_dir}")


def main():
    """Run all metadata tests."""
    print("\n" + "=" * 60)
    print("FILE METADATA FUNCTIONALITY TESTS")
    print("=" * 60 + "\n")

    try:
        # Run tests
        manager = test_file_registration()
        test_metadata_retrieval(manager)
        test_list_with_filters(manager)
        test_files_summary(manager)
        test_metadata_persistence(Path(__file__).parent / "test_metadata_workspace")
        test_copy_with_description()
        test_use_case_llm_access()

        print("=" * 60)
        print("✅ ALL METADATA TESTS PASSED")
        print("=" * 60)
        print("\nFile metadata functionality is working correctly!")
        print("\nKey Features:")
        print("  ✓ File registration with descriptions")
        print("  ✓ Metadata persistence across sessions")
        print("  ✓ Flexible listing with filters")
        print("  ✓ LLM-friendly summaries")
        print("  ✓ Automatic registration on file copy")
        print("\nLLM nodes can now:")
        print("  - See descriptions of all workspace files")
        print("  - Filter files by type or location")
        print("  - Make informed decisions about which files to use")
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
