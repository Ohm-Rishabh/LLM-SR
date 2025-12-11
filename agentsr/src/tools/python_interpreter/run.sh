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
echo "  WORKSPACE_LOGS: $WORKSPACE_LOGS" >&2
echo "Tool arguments:" >&2
env | grep "^TOOL_ARG_" >&2 || echo "  (no arguments)" >&2

# Set up log file path
LOG_FILE="${WORKSPACE_LOGS}/python_interpreter_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE" >&2

# Execute the Python tool
# Redirect stderr to log file while still showing it in terminal
echo "Executing tool.py..." >&2
python tool.py 2> >(tee -a "$LOG_FILE" >&2)
