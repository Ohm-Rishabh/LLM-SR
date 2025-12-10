#!/bin/bash
set -e  # Exit on error

echo "Starting PySR tool execution..." >&2

# Validate required inputs
if [ -z "$STATE_DATA_FILE" ]; then
    echo "Error: STATE_DATA_FILE not provided" >&2
    exit 1
fi

# Log environment for debugging
echo "Data file: $STATE_DATA_FILE" >&2
echo "Tool arguments:" >&2
env | grep "^TOOL_ARG_" >&2 || echo "  (no arguments)" >&2

# Activate conda environment if conda is available
if command -v conda &> /dev/null; then
    echo "Activating pysr conda environment..." >&2
    # Source conda initialization
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate pysr 2>&1 >&2
else
    echo "Warning: conda not found, using system Python" >&2
fi

# Execute the Python tool
# Environment variables are automatically available to the Python script
echo "Executing tool.py..." >&2
python tool.py
