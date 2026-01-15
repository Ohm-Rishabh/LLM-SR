#!/bin/bash
# Simple test script for symmetry discovery tool

set -e

echo "Testing Symmetry Discovery Tool"
echo "================================"

# Setup test workspace
TEST_WORKSPACE="/tmp/test_symmetry_workspace"
rm -rf "$TEST_WORKSPACE"
mkdir -p "$TEST_WORKSPACE/input"
mkdir -p "$TEST_WORKSPACE/output"
mkdir -p "$TEST_WORKSPACE/logs"
mkdir -p "$TEST_WORKSPACE/scratch"

echo "Created test workspace: $TEST_WORKSPACE"

# Set environment variables
export WORKSPACE_ROOT="$TEST_WORKSPACE"
export WORKSPACE_INPUT="$TEST_WORKSPACE/input"
export WORKSPACE_OUTPUT="$TEST_WORKSPACE/output"
export WORKSPACE_LOGS="$TEST_WORKSPACE/logs"
export WORKSPACE_SCRATCH="$TEST_WORKSPACE/scratch"

# Set tool arguments
export TOOL_ARG_DATA_FILE="/home/ubuntu/LLM-SR/llmsr/data/diffeq/dosc-train.h5"
export TOOL_ARG_DATA_FILE="/home/ubuntu/LLM-SR/agentsr/src/tools/symmetry_discovery/rd.h5"
export TOOL_ARG_HIDDEN_DIM="16"
export TOOL_ARG_SURROGATE_EPOCHS="1000"
export TOOL_ARG_SURROGATE_LR="0.001"
export TOOL_ARG_SYMMETRY_EPOCHS="1000"
export TOOL_ARG_SYMMETRY_LR="0.01"

# Copy test data to input
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# cp "$SCRIPT_DIR/test_harmonic_oscillator.h5" "$TEST_WORKSPACE/input/"
cp "$TOOL_ARG_DATA_FILE" "$TEST_WORKSPACE/input/"

echo "Copied test data to workspace/input"

echo ""
echo "Environment configured:"
echo "  WORKSPACE_ROOT: $WORKSPACE_ROOT"
echo "  TOOL_ARG_DATA_FILE: $TOOL_ARG_DATA_FILE"
echo "  TOOL_ARG_SURROGATE_EPOCHS: $TOOL_ARG_SURROGATE_EPOCHS"
echo "  TOOL_ARG_SYMMETRY_EPOCHS: $TOOL_ARG_SYMMETRY_EPOCHS"
echo ""

# Run the tool
echo "Running tool..."
cd "$SCRIPT_DIR"
bash run.sh

# Check results
echo ""
echo "================================"
echo "Test completed!"
echo ""

if [ -f "$WORKSPACE_SCRATCH/result.json" ]; then
    echo "Result file found:"
    cat "$WORKSPACE_SCRATCH/result.json" | python -m json.tool
    echo ""
    echo "Ground truth Lie generator:"
    echo "  [[0, -1],"
    echo "   [1,  0]]"
    echo ""
    echo "Compare the discovered lie_generator with ground truth above."
else
    echo "ERROR: Result file not found!"
    exit 1
fi

echo ""
echo "Test workspace preserved at: $TEST_WORKSPACE"
echo "To clean up: rm -rf $TEST_WORKSPACE"
