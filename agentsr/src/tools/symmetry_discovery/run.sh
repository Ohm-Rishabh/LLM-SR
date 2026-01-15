#!/bin/bash
set -e  # Exit on error

echo "Starting Symmetry Discovery tool execution..." >&2

# Validate required inputs
if [ -z "$TOOL_ARG_DATA_FILE" ]; then
    echo "Error: TOOL_ARG_DATA_FILE not provided" >&2
    exit 1
fi

# Log environment for debugging
echo "Workspace paths:" >&2
echo "  WORKSPACE_ROOT: $WORKSPACE_ROOT" >&2
echo "  WORKSPACE_INPUT: $WORKSPACE_INPUT" >&2
echo "  WORKSPACE_OUTPUT: $WORKSPACE_OUTPUT" >&2
echo "  WORKSPACE_LOGS: $WORKSPACE_LOGS" >&2
echo "  WORKSPACE_SCRATCH: $WORKSPACE_SCRATCH" >&2
echo "Data file: $TOOL_ARG_DATA_FILE" >&2
echo "Tool arguments:" >&2
env | grep "^TOOL_ARG_" >&2 || echo "  (no arguments)" >&2

# Set up log file path
LOG_FILE="${WORKSPACE_LOGS}/symmetry_discovery_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE" >&2

# Activate conda environment if conda is available
# Note: You may need to create a conda environment with PyTorch and h5py
if command -v conda &> /dev/null; then
    echo "Activating conda environment..." >&2
    # Source conda initialization
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    # Try to activate a pytorch environment, fall back to base if not available
    conda activate torch 2>&1 >&2 || conda activate base 2>&1 >&2 || true
else
    echo "Warning: conda not found, using system Python" >&2
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Execute the Python tool
# Redirect stderr to log file while still showing it in terminal
echo "Executing tool.py..." >&2
cd "$SCRIPT_DIR"
python tool.py 2> >(tee -a "$LOG_FILE" >&2)
