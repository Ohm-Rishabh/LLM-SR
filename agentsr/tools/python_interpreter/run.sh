#!/bin/bash
set -e  # Exit on error

echo "Starting Python Interpreter tool execution..." >&2

# Validate required inputs
if [ -z "$TOOL_ARG_CODE" ]; then
    echo "Error: TOOL_ARG_CODE not provided" >&2
    exit 1
fi

# Log environment for debugging
echo "Workspace paths:" >&2
echo "  WORKSPACE_ROOT: $WORKSPACE_ROOT" >&2
echo "  WORKSPACE_INPUT: $WORKSPACE_INPUT" >&2
echo "  WORKSPACE_OUTPUT: $WORKSPACE_OUTPUT" >&2
echo "Tool arguments:" >&2
env | grep "^TOOL_ARG_" >&2 || echo "  (no arguments)" >&2

# Execute the Python tool
# Environment variables are automatically available to the Python script
echo "Executing tool.py..." >&2
python tool.py
