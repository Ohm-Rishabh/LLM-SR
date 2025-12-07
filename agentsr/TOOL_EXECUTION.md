# Tool Execution Guide

This document explains how to create tool entry scripts for the ToolSwitchNode.

## Overview

The `ToolSwitchNode` executes symbolic regression tools by running bash entry scripts (`run.sh`) located in tool-specific directories. Arguments from the tool_call JSON are passed as environment variables.

## Directory Structure

```
agentsr/
├── tools/
│   ├── pysr/
│   │   ├── run.sh          # Entry script for PySR
│   │   └── tool.py         # Optional: Python implementation
│   ├── gplearn/
│   │   ├── run.sh          # Entry script for gplearn
│   │   └── tool.py
│   └── linear_regression/
│       └── run.sh
└── ...
```

## Environment Variables

The ToolSwitchNode provides arguments and state information via environment variables:

### Tool Arguments (from tool_call JSON)
- **Naming convention**: `TOOL_ARG_<ARGUMENT_NAME>` (uppercase)
- **Examples**:
  - `tool_call.arguments.maxsize` → `$TOOL_ARG_MAXSIZE`
  - `tool_call.arguments.binary_operators` → `$TOOL_ARG_BINARY_OPERATORS` (JSON string for arrays/objects)
  - `tool_call.arguments.niterations` → `$TOOL_ARG_NITERATIONS`

### State Variables
- `$STATE_DATA_FILE` - Path to the input data CSV file
- `$STATE_USER_QUERY` - User's query/request

## Creating a Tool Entry Script

### Template: `run.sh`

```bash
#!/bin/bash
set -e  # Exit on error

# Example: PySR tool entry script
# This script receives arguments via environment variables

# Log execution start
echo "Starting PySR execution..." >&2

# Access arguments from environment variables
# Simple values can be used directly
MAXSIZE="${TOOL_ARG_MAXSIZE:-30}"  # Default to 30 if not provided
NITERATIONS="${TOOL_ARG_NITERATIONS:-100}"

# Complex values (arrays, objects) are JSON strings
# Parse them using jq or python if needed
BINARY_OPS="${TOOL_ARG_BINARY_OPERATORS:-[\"*\", \"+\", \"-\", \"/\"]}"

# Access state variables
DATA_FILE="$STATE_DATA_FILE"

# Validate required inputs
if [ -z "$DATA_FILE" ]; then
    echo "Error: No data file provided" >&2
    exit 1
fi

# Execute the actual tool (Python script, executable, etc.)
python tool.py \
    --data "$DATA_FILE" \
    --maxsize "$MAXSIZE" \
    --niterations "$NITERATIONS" \
    --binary-operators "$BINARY_OPS"

# The script should:
# 1. Print results to stdout (captured as tool_result.stdout)
# 2. Print logs/errors to stderr (captured as tool_result.stderr)
# 3. Exit with code 0 for success, non-zero for failure
```

### Example: Simple Linear Regression Tool

```bash
#!/bin/bash
set -e

echo "Running linear regression..." >&2

# Get data file
DATA_FILE="$STATE_DATA_FILE"

if [ -z "$DATA_FILE" ]; then
    echo "Error: DATA_FILE not provided" >&2
    exit 1
fi

# Run Python script
python3 << 'PYTHON_SCRIPT'
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import json

# Read data
data_file = os.environ['STATE_DATA_FILE']
df = pd.read_csv(data_file)

# Assume last column is target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Fit model
model = LinearRegression()
model.fit(X, y)

# Output results as JSON
result = {
    "coefficients": model.coef_.tolist(),
    "intercept": float(model.intercept_),
    "score": float(model.score(X, y))
}

print(json.dumps(result))
PYTHON_SCRIPT
```

## Tool Call JSON Format

The SRNode generates a tool_call with this structure:

```json
{
  "tool_name": "pysr",
  "arguments": {
    "maxsize": 30,
    "niterations": 100,
    "binary_operators": ["+", "-", "*", "/"],
    "unary_operators": ["sin", "cos"],
    "populations": 15
  }
}
```

This gets converted to environment variables:
```bash
TOOL_ARG_MAXSIZE=30
TOOL_ARG_NITERATIONS=100
TOOL_ARG_BINARY_OPERATORS='["+", "-", "*", "/"]'
TOOL_ARG_UNARY_OPERATORS='["sin", "cos"]'
TOOL_ARG_POPULATIONS=15
```

## Return Values

The ToolSwitchNode captures the tool execution results:

### Success Result
```json
{
  "status": "success",
  "tool_name": "pysr",
  "stdout": "... tool output ...",
  "stderr": "... tool logs ...",
  "exit_code": 0
}
```

### Error Result
```json
{
  "status": "error",
  "tool_name": "pysr",
  "error": "Tool exited with code 1",
  "stdout": "... partial output ...",
  "stderr": "... error messages ...",
  "exit_code": 1
}
```

## Best Practices

1. **Use `set -e`** - Exit immediately if any command fails
2. **Validate inputs** - Check that required environment variables are set
3. **Log to stderr** - Use stderr for logging, stdout for results
4. **Output JSON** - Return structured data as JSON for easy parsing
5. **Handle errors gracefully** - Provide clear error messages
6. **Set timeouts** - Configure reasonable timeouts in ToolSwitchNode
7. **Document requirements** - Note dependencies in comments

## Example Usage in Python

When creating the ToolSwitchNode:

```python
from nodes import ToolSwitchNode

tool_switch_node = ToolSwitchNode(
    name="tool_executor",
    description="Executes tool calls and awaits results",
    tools_dir="/path/to/tools",  # Optional, defaults to ROOT_DIR/tools
    timeout=300  # Optional timeout in seconds
)
```

## Debugging

To debug tool execution:

1. Check logs for environment variables: `logger.debug` shows all env vars
2. Test bash script manually:
   ```bash
   export TOOL_ARG_MAXSIZE=30
   export STATE_DATA_FILE=/path/to/data.csv
   bash tools/pysr/run.sh
   ```
3. Review stdout/stderr in tool_result
4. Check exit codes for failure diagnosis
