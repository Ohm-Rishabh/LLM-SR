# Unified Temporary File System - Design Document

## Overview

This document describes the design and implementation plan for a unified temporary file system (workspace) that allows all nodes and tools in the agentic workflow to share files transparently.

## Current Architecture Problems

### Issues
1. **Hardcoded assumptions**: Python interpreter assumes data is pre-loaded into DataFrame
2. **Limited file sharing**: Tools can't easily share intermediate results (plots, transformed data, logs)
3. **Polluted root directory**: Result files written to `ROOT_DIR/result.json`
4. **No agent flexibility**: Agent can't specify which files to read/write in code
5. **State bloat**: Large data stays in state dictionary as file paths with no clear organization

### Current Flow
```
main.py → state["data_file"] = "/absolute/path/to/data.csv"
       → ToolSwitchNode sets STATE_DATA_FILE env var
       → Tool reads from STATE_DATA_FILE
       → Tool writes to ROOT_DIR/result.json (hardcoded)
       → ToolSwitchNode reads result.json and deletes it
```

## Proposed Architecture: Workspace System

### Core Concept
Each workflow run gets an isolated **workspace directory** that acts as a shared temporary file system. All nodes and tools operate within this workspace using relative paths.

### Design Principles
1. **Isolation**: Each run gets unique workspace (timestamped directory)
2. **Transparency**: Tools access files via simple relative paths
3. **Agent Flexibility**: LLM can specify any file to read/write in Python code
4. **State Integration**: Workspace path stored in state, passed to all nodes
5. **Persistence**: Workspace preserved after run (for debugging), with optional cleanup
6. **Backward Compatibility**: Existing tools work with minimal changes

## Architecture Design

### Directory Structure
```
agentsr/
├── workspaces/                    # All workspaces (gitignored)
│   ├── run_20231210_143022_abc123/   # Unique workspace per run
│   │   ├── input/                     # Input data files
│   │   │   └── data.csv              # Copied from original location
│   │   ├── output/                    # Tool outputs
│   │   │   ├── pysr_result.json      # PySR results
│   │   │   ├── correlation_heatmap.png
│   │   │   └── analysis_stats.txt
│   │   ├── logs/                      # Tool execution logs
│   │   │   ├── pysr.log
│   │   │   └── python_interpreter.log
│   │   └── scratch/                   # Temporary files
│   │       └── intermediate_data.csv
│   └── run_20231210_143530_def456/   # Another run
└── ...
```

### Workspace Manager

Create a new module: `agentsr/src/core/workspace.py`

```python
class WorkspaceManager:
    """
    Manages workspace directories for workflow runs.

    Each workspace provides:
    - Isolated file system for a workflow run
    - Organized directories (input/, output/, logs/, scratch/)
    - Path resolution utilities
    - Cleanup mechanisms
    """

    def __init__(self, base_dir: Path, run_id: str = None):
        """
        Initialize workspace manager.

        Args:
            base_dir: Base directory for all workspaces (e.g., ROOT_DIR/workspaces)
            run_id: Unique identifier for this run (auto-generated if None)
        """

    def create_workspace(self) -> Path:
        """Create workspace directory structure."""

    def get_path(self, *parts: str) -> Path:
        """Resolve path within workspace (e.g., get_path('output', 'plot.png'))."""

    def copy_input_file(self, src: Path, dst_name: str = None) -> Path:
        """Copy external file into workspace input/ directory."""

    def list_files(self, subdir: str = None, pattern: str = None) -> List[Path]:
        """List files in workspace or subdirectory."""

    def cleanup(self, keep_last_n: int = 5):
        """Clean up old workspace directories."""
```

## Implementation Plan

### Phase 1: Core Workspace Infrastructure

#### 1.1 Create WorkspaceManager
**File**: `agentsr/src/core/workspace.py`

```python
from pathlib import Path
import datetime
import secrets
import shutil
from typing import List, Optional

class WorkspaceManager:
    """Manages isolated workspace for workflow execution."""

    def __init__(self, base_dir: Path, run_id: Optional[str] = None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique run ID: timestamp + random suffix
        if run_id is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            random_suffix = secrets.token_hex(3)
            run_id = f"run_{timestamp}_{random_suffix}"

        self.run_id = run_id
        self.workspace_root = self.base_dir / run_id

    def create_workspace(self) -> Path:
        """Create workspace with standard subdirectories."""
        self.workspace_root.mkdir(parents=True, exist_ok=True)

        # Create standard subdirectories
        (self.workspace_root / "input").mkdir(exist_ok=True)
        (self.workspace_root / "output").mkdir(exist_ok=True)
        (self.workspace_root / "logs").mkdir(exist_ok=True)
        (self.workspace_root / "scratch").mkdir(exist_ok=True)

        return self.workspace_root

    def get_path(self, *parts: str) -> Path:
        """Get absolute path within workspace."""
        return self.workspace_root.joinpath(*parts)

    def get_relative_path(self, absolute_path: Path) -> Optional[Path]:
        """Convert absolute path to workspace-relative path."""
        try:
            return Path(absolute_path).relative_to(self.workspace_root)
        except ValueError:
            return None

    def copy_input_file(self, src: Path, dst_name: Optional[str] = None) -> Path:
        """Copy external file into workspace input directory."""
        src_path = Path(src)
        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {src}")

        if dst_name is None:
            dst_name = src_path.name

        dst_path = self.get_path("input", dst_name)
        shutil.copy2(src_path, dst_path)

        return dst_path

    def list_files(self, subdir: Optional[str] = None,
                   pattern: str = "*") -> List[Path]:
        """List files in workspace or subdirectory."""
        search_dir = self.workspace_root if subdir is None else self.get_path(subdir)
        return sorted(search_dir.glob(pattern))

    @classmethod
    def cleanup_old_workspaces(cls, base_dir: Path, keep_last_n: int = 10):
        """Remove old workspace directories, keeping the N most recent."""
        base_path = Path(base_dir)
        if not base_path.exists():
            return

        # Get all workspace directories sorted by creation time
        workspaces = sorted(
            [d for d in base_path.iterdir() if d.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        # Remove old workspaces beyond keep_last_n
        for workspace in workspaces[keep_last_n:]:
            shutil.rmtree(workspace)
```

#### 1.2 Update Workflow Class
**File**: `agentsr/src/core/workflow.py`

Changes:
- Add `workspace_manager` attribute
- Initialize workspace in `run()` method
- Add workspace path to state
- Optional cleanup after run

```python
# Add to imports
from core.workspace import WorkspaceManager
from core.consts import ROOT_DIR

class Workflow:
    def __init__(self, workspace_base_dir: Optional[Path] = None) -> None:
        self._nodes: Dict[str, Node] = {}
        self._edges: Dict[str, List[str]] = {}
        self._start: Optional[str] = None

        # Workspace configuration
        if workspace_base_dir is None:
            workspace_base_dir = Path(ROOT_DIR) / "workspaces"
        self.workspace_base_dir = workspace_base_dir
        self.workspace_manager: Optional[WorkspaceManager] = None

    def run(
        self,
        initial_state: Optional[Dict[str, Any]] = None,
        max_steps: int = 1000,
        cleanup_workspace: bool = False,
    ) -> Dict[str, Any]:
        """Run workflow with workspace support."""

        # Create workspace for this run
        self.workspace_manager = WorkspaceManager(self.workspace_base_dir)
        workspace_path = self.workspace_manager.create_workspace()
        logger.info(f"Created workspace: {workspace_path}")

        # Initialize state
        state: Dict[str, Any] = {} if initial_state is None else dict(initial_state)

        # Add workspace information to state
        state["_workspace_root"] = str(workspace_path)
        state["_workspace_manager"] = self.workspace_manager

        # Copy input data file to workspace if provided
        if "data_file" in state:
            original_path = Path(state["data_file"])
            workspace_data_path = self.workspace_manager.copy_input_file(
                original_path,
                "data.csv"
            )
            state["data_file"] = str(workspace_data_path)
            logger.info(f"Copied data file to workspace: {workspace_data_path}")

        # ... existing run logic ...

        # After workflow completes
        state["_workspace_path"] = str(workspace_path)

        # Optional cleanup
        if cleanup_workspace:
            WorkspaceManager.cleanup_old_workspaces(
                self.workspace_base_dir,
                keep_last_n=5
            )

        return state
```

### Phase 2: Update ToolSwitchNode

#### 2.1 Modify Environment Preparation
**File**: `agentsr/src/nodes.py` - `ToolSwitchNode` class

Changes:
- Pass workspace root path as `WORKSPACE_ROOT` env var
- Update `STATE_DATA_FILE` to point to workspace copy
- Add workspace subdirectory paths

```python
def _prepare_environment(
    self,
    args: Dict[str, Any],
    state: Dict[str, Any]
) -> Dict[str, str]:
    """Prepare environment variables with workspace support."""
    env_vars = {}

    # Add arguments as environment variables (existing logic)
    for key, value in args.items():
        env_key = f"TOOL_ARG_{key.upper()}"
        if isinstance(value, (list, dict)):
            env_vars[env_key] = json_module.dumps(value)
        elif isinstance(value, bool):
            env_vars[env_key] = str(value).lower()
        else:
            env_vars[env_key] = str(value)

    # Add workspace information
    if "_workspace_root" in state:
        workspace_root = state["_workspace_root"]
        env_vars["WORKSPACE_ROOT"] = workspace_root
        env_vars["WORKSPACE_INPUT"] = os.path.join(workspace_root, "input")
        env_vars["WORKSPACE_OUTPUT"] = os.path.join(workspace_root, "output")
        env_vars["WORKSPACE_LOGS"] = os.path.join(workspace_root, "logs")
        env_vars["WORKSPACE_SCRATCH"] = os.path.join(workspace_root, "scratch")

    # Add data file (now points to workspace copy)
    if "data_file" in state:
        env_vars["STATE_DATA_FILE"] = state["data_file"]

    if "user_query" in state:
        env_vars["STATE_USER_QUERY"] = state["user_query"]

    return env_vars
```

#### 2.2 Update Result File Handling
Change result file path from hardcoded `ROOT_DIR/result.json` to workspace:

```python
def _execute_tool(self, tool_call: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute tool with workspace support."""

    # ... existing validation logic ...

    # Execute script
    result = subprocess.run(...)

    # Look for result file in workspace output directory
    if "_workspace_root" in state:
        result_file_path = os.path.join(
            state["_workspace_root"],
            "output",
            f"{tool_name}_result.json"
        )
    else:
        # Fallback to old behavior
        result_file_path = os.path.join(ROOT_DIR, "result.json")

    # ... rest of result handling ...
```

### Phase 3: Update Tool Implementations

#### 3.1 Update PySR Tool
**File**: `agentsr/tools/pysr/tool.py`

Changes:
- Read `WORKSPACE_OUTPUT` instead of using `result_manager` with hardcoded path
- Write result to workspace output directory

```python
def main():
    """Main execution function."""
    try:
        # Get workspace paths from environment
        workspace_output = os.environ.get('WORKSPACE_OUTPUT')
        if not workspace_output:
            # Fallback to old behavior
            workspace_output = ROOT_DIR

        # ... existing tool logic ...

        # Write results to workspace
        result_file = os.path.join(
            workspace_output,
            "pysr_result.json"
        )
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults written to: {result_file}", file=sys.stderr)
        return 0
```

#### 3.2 Update Python Interpreter Tool
**File**: `agentsr/tools/python_interpreter/tool.py`

**Major Change**: Remove pre-loaded DataFrame assumption. Let agent specify files.

```python
def execute_code(code, workspace_root, data_file, timeout=30):
    """
    Execute Python code with workspace access.

    The code has access to:
    - WORKSPACE_ROOT: Path to workspace root
    - WORKSPACE_INPUT: Path to input directory
    - WORKSPACE_OUTPUT: Path to output directory
    - WORKSPACE_SCRATCH: Path to scratch directory
    - DATA_FILE: Path to primary data file (for convenience)

    Agent writes code like:
        import pandas as pd
        df = pd.read_csv(DATA_FILE)
        # ... analysis ...
        df_transformed.to_csv(f"{WORKSPACE_OUTPUT}/transformed.csv")
        plt.savefig(f"{WORKSPACE_OUTPUT}/plot.png")
    """

    # Prepare execution environment
    exec_globals = {
        '__builtins__': __builtins__,
        # Workspace paths
        'WORKSPACE_ROOT': workspace_root,
        'WORKSPACE_INPUT': os.path.join(workspace_root, 'input'),
        'WORKSPACE_OUTPUT': os.path.join(workspace_root, 'output'),
        'WORKSPACE_SCRATCH': os.path.join(workspace_root, 'scratch'),
        'DATA_FILE': data_file,  # Convenience for main data file
        # Pre-import common libraries
        'np': np,
        'pd': pd,
        'plt': plt,
        'sns': sns,
        'stats': stats,
        'os': os,
        'Path': Path,
    }

    # ... rest of execution logic ...
```

Update main function:

```python
def main():
    """Main execution function."""
    try:
        # Get workspace and data file from environment
        workspace_root = os.environ.get('WORKSPACE_ROOT')
        data_file = os.environ.get('STATE_DATA_FILE')

        if not workspace_root:
            raise ValueError("WORKSPACE_ROOT environment variable not set")

        if not data_file:
            raise ValueError("STATE_DATA_FILE environment variable not set")

        # Get code from environment
        code = parse_env_arg('code', None, str)
        if not code:
            raise ValueError("No code provided. Set TOOL_ARG_CODE environment variable.")

        # Execute code with workspace access
        execution_result = execute_code(
            code=code,
            workspace_root=workspace_root,
            data_file=data_file,
            timeout=30
        )

        # Write results to workspace output
        result_file = os.path.join(
            workspace_root,
            "output",
            "python_interpreter_result.json"
        )

        results = {
            "tool_name": "python_interpreter",
            "result_type": "code_execution",
            "status": execution_result["status"],
            "code": code,
            "output": {
                "stdout": execution_result["stdout"],
                "stderr": execution_result["stderr"],
            },
            "error": execution_result["error"],
            "error_type": execution_result["error_type"],
        }

        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)

        return 0 if execution_result["status"] == "success" else 1
```

#### 3.3 Update Python Interpreter Tool Spec
**File**: `agentsr/tool_specs/python_interpreter.md`

Update data access section:

```markdown
#### Data Access

The code execution environment provides:
- **`WORKSPACE_ROOT`**: Path to workspace root directory
- **`WORKSPACE_INPUT`**: Path to input directory (read input files)
- **`WORKSPACE_OUTPUT`**: Path to output directory (write results, plots)
- **`WORKSPACE_SCRATCH`**: Path to scratch directory (temporary files)
- **`DATA_FILE`**: Path to primary data file (convenience variable)

#### Pre-imported Libraries

- `numpy` (as `np`)
- `pandas` (as `pd`)
- `matplotlib.pyplot` (as `plt`)
- `seaborn` (as `sns`)
- `scipy.stats` (as `stats`)
- `os`
- `pathlib.Path` (as `Path`)

#### Example Usage

##### Reading Data and Computing Statistics
```json
{
  "tool_name": "python_interpreter",
  "arguments": {
    "code": "import pandas as pd\\ndf = pd.read_csv(DATA_FILE)\\nprint('Shape:', df.shape)\\nprint('Stats:\\n', df.describe())"
  }
}
```

##### Creating Visualization
```json
{
  "tool_name": "python_interpreter",
  "arguments": {
    "code": "import pandas as pd\\nimport matplotlib.pyplot as plt\\nimport seaborn as sns\\n\\ndf = pd.read_csv(DATA_FILE)\\nplt.figure(figsize=(10, 6))\\nsns.heatmap(df.corr(), annot=True, cmap='coolwarm')\\nplt.savefig(f'{WORKSPACE_OUTPUT}/correlation.png')\\nprint('Plot saved to correlation.png')"
  }
}
```

##### Writing Transformed Data
```json
{
  "tool_name": "python_interpreter",
  "arguments": {
    "code": "import pandas as pd\\ndf = pd.read_csv(DATA_FILE)\\ndf_log = df.apply(np.log1p)\\ndf_log.to_csv(f'{WORKSPACE_OUTPUT}/log_transformed.csv', index=False)\\nprint('Saved transformed data')"
  }
}
```

### Phase 4: Update Run Scripts

#### 4.1 Update pysr/run.sh
**File**: `agentsr/tools/pysr/run.sh`

No changes needed - environment variables automatically available.

#### 4.2 Create python_interpreter/run.sh
**File**: `agentsr/tools/python_interpreter/run.sh`

```bash
#!/bin/bash
set -e  # Exit on error

echo "Starting Python Interpreter tool execution..." >&2

# Validate required inputs
if [ -z "$WORKSPACE_ROOT" ]; then
    echo "Error: WORKSPACE_ROOT not provided" >&2
    exit 1
fi

if [ -z "$STATE_DATA_FILE" ]; then
    echo "Error: STATE_DATA_FILE not provided" >&2
    exit 1
fi

if [ -z "$TOOL_ARG_CODE" ]; then
    echo "Error: No code provided (TOOL_ARG_CODE)" >&2
    exit 1
fi

# Log environment for debugging
echo "Workspace: $WORKSPACE_ROOT" >&2
echo "Data file: $STATE_DATA_FILE" >&2
echo "Code length: ${#TOOL_ARG_CODE} characters" >&2

# Execute the Python tool
echo "Executing tool.py..." >&2
python tool.py
```

### Phase 5: Update result_manager (Optional)

**File**: `agentsr/tools/common/result_manager.py`

Add workspace-aware utilities:

```python
def write_result_to_workspace(
    result: Dict[str, Any],
    tool_name: str,
    workspace_root: Optional[Path] = None
) -> Path:
    """
    Write result to workspace output directory.

    Args:
        result: Dictionary containing the result data
        tool_name: Name of the tool
        workspace_root: Workspace root path (from WORKSPACE_ROOT env var if None)

    Returns:
        Path to the written file
    """
    if workspace_root is None:
        workspace_root = os.environ.get('WORKSPACE_ROOT')
        if not workspace_root:
            raise ValueError("No workspace_root provided and WORKSPACE_ROOT not set")

    workspace_root = Path(workspace_root)
    output_dir = workspace_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / f"{tool_name}_result.json"

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    return result_path
```

## Migration Strategy

### Step 1: Add Infrastructure (Non-Breaking)
- Create `workspace.py` module
- Add workspace initialization to `Workflow.run()` with feature flag
- Add workspace env vars to `ToolSwitchNode`

### Step 2: Update Tools (Backward Compatible)
- Update tools to check for `WORKSPACE_ROOT` first, fall back to old behavior
- Tools work with or without workspace

### Step 3: Update Tool Specs
- Document new workspace variables
- Provide examples of workspace usage

### Step 4: Enable by Default
- Enable workspace by default in `Workflow.__init__()`
- Add cleanup to main.py

### Step 5: Remove Fallbacks
- Remove backward compatibility code
- Clean up old result_manager patterns

## Benefits

1. **Agent Flexibility**: Agent can read/write any files, specify transformations
2. **Better Organization**: Clear directory structure per run
3. **Debugging**: All artifacts preserved in workspace for inspection
4. **Visualization**: Plots automatically saved to output directory
5. **Multi-step Workflows**: Tools can produce files for next tool to consume
6. **Cleaner State**: State dictionary contains paths, not large data
7. **Isolation**: Runs don't interfere with each other

## Example Usage After Implementation

```python
# Agent's python_interpreter code can now be:
code = """
import pandas as pd
import matplotlib.pyplot as plt

# Read data
df = pd.read_csv(DATA_FILE)

# Compute statistics
stats = df.describe()
print("Statistics:")
print(stats)

# Create visualization
plt.figure(figsize=(10, 6))
df.hist(bins=30)
plt.tight_layout()
plt.savefig(f'{WORKSPACE_OUTPUT}/distribution.png')
print("Saved distribution plot")

# Save transformed data for next tool
df_normalized = (df - df.mean()) / df.std()
df_normalized.to_csv(f'{WORKSPACE_OUTPUT}/normalized_data.csv', index=False)
print("Saved normalized data")
"""

# Next tool can read the normalized data:
code2 = """
import pandas as pd

# Read transformed data from previous step
df = pd.read_csv(f'{WORKSPACE_OUTPUT}/normalized_data.csv')
print("Loaded normalized data:", df.shape)
"""
```

## Testing Plan

1. **Unit Tests**: Test `WorkspaceManager` methods
2. **Integration Tests**: Test workspace creation in workflow
3. **Tool Tests**: Test each tool with workspace env vars
4. **End-to-End Tests**: Full workflow with multiple tool calls
5. **Migration Tests**: Ensure backward compatibility during migration

## Files to Modify

### New Files
1. `agentsr/src/core/workspace.py` - WorkspaceManager class
2. `agentsr/tools/python_interpreter/run.sh` - Tool entry script
3. `agentsr/WORKSPACE_DESIGN.md` - This document

### Modified Files
1. `agentsr/src/core/workflow.py` - Add workspace initialization
2. `agentsr/src/nodes.py` - Update ToolSwitchNode environment prep
3. `agentsr/tools/pysr/tool.py` - Use workspace output directory
4. `agentsr/tools/python_interpreter/tool.py` - Remove DataFrame assumption, add workspace
5. `agentsr/tool_specs/python_interpreter.md` - Update documentation
6. `agentsr/src/main.py` - Add workspace cleanup option
7. `agentsr/tools/common/result_manager.py` - Add workspace utilities
8. `agentsr/.gitignore` - Add `workspaces/` directory

## Summary

This design provides a clean, extensible temporary file system that:
- ✅ Removes hardcoded assumptions about data loading
- ✅ Gives agents full control over file I/O
- ✅ Organizes outputs clearly per workflow run
- ✅ Maintains backward compatibility during migration
- ✅ Enables multi-step workflows with file sharing
- ✅ Simplifies debugging with preserved workspaces
