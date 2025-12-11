# File Metadata System

## Overview

The WorkspaceManager now includes a comprehensive file metadata system that allows tools to register files with human-readable descriptions. This enables LLM nodes to understand what files are available in the workspace and make informed decisions about which files to read or pass to tools.

## Key Features

- **File Registration**: Tools can register files with descriptions
- **Automatic Metadata**: File size, timestamps, and creator information tracked automatically
- **Persistence**: Metadata saved to `.file_metadata.json` in workspace
- **LLM-Friendly Summaries**: Get formatted summaries of all workspace files
- **Flexible Queries**: Filter files by type, location, or creator
- **Automatic Registration**: Files copied to workspace are auto-registered

## Usage

### Registering Files

Tools should register files after creating them:

```python
# Tool creates a file
output_path = workspace_manager.get_path("output", "correlation.png")
plt.savefig(output_path)

# Register the file with metadata
workspace_manager.register_file(
    "output/correlation.png",
    description="Correlation heatmap showing relationships between all features",
    file_type="plot",
    created_by="python_interpreter"
)
```

### Getting File Descriptions

LLM nodes can retrieve file information to make decisions:

```python
# Get summary of all workspace files
summary = workspace_manager.get_files_summary()
print(summary)
# Output:
# Workspace Files:
#
# input/
#   - data.csv (data): Original dataset with 5 features and 1000 samples
#
# output/
#   - correlation.png (plot): Correlation heatmap showing relationships
#   - analysis.json (result): Statistical analysis results
```

### Filtering Files

Query specific files by type or location:

```python
# Get all plot files
plots = workspace_manager.list_files_with_metadata(file_type="plot")
for file_info in plots:
    print(f"{file_info['path']}: {file_info['description']}")

# Get all files in output directory
output_files = workspace_manager.list_files_with_metadata(subdir="output")
```

### Individual File Metadata

```python
# Get metadata for a specific file
metadata = workspace_manager.get_file_metadata("output/plot.png")
if metadata:
    print(f"Description: {metadata['description']}")
    print(f"Type: {metadata['file_type']}")
    print(f"Created by: {metadata['created_by']}")
    print(f"Size: {metadata['size_bytes']} bytes")
```

## File Types

Standard file types (can be extended):

- **`data`**: Dataset files (CSV, JSON, etc.)
- **`plot`**: Visualizations (PNG, SVG, etc.)
- **`result`**: Analysis results (JSON, TXT)
- **`log`**: Execution logs
- **`model`**: Saved models or equations
- **`unknown`**: Default if not specified

## Metadata Fields

Each registered file has:

- **`description`** (required): Human-readable description
- **`file_type`** (optional): Category of file
- **`created_by`** (optional): Tool/node that created it
- **`registered_at`** (auto): ISO timestamp when registered
- **`size_bytes`** (auto): File size in bytes
- **`modified_at`** (auto): Last modification timestamp
- **Custom fields**: Any additional key-value pairs

## Examples

### Python Interpreter Tool

```python
# After running analysis code
workspace_root = os.environ.get('WORKSPACE_ROOT')
manager = get_workspace_manager()  # From state

# Register output files
manager.register_file(
    "output/distribution.png",
    description="Histogram of target variable distribution",
    file_type="plot",
    created_by="python_interpreter",
    figure_type="histogram",
    n_bins=30
)

manager.register_file(
    "output/stats.json",
    description="Descriptive statistics: mean, std, quartiles, skewness",
    file_type="result",
    created_by="python_interpreter"
)
```

### PySR Tool

```python
# After discovering equations
manager.register_file(
    "output/pysr_equations.json",
    description="Discovered equations ranked by complexity and accuracy (10 best)",
    file_type="result",
    created_by="pysr",
    num_equations=10,
    best_score=0.95
)
```

### LLM Node Decision Making

```python
# In SRNode or LLMNode
workspace_manager = state.get("_workspace_manager")

# Get summary of available files
files_summary = workspace_manager.get_files_summary()

# Add to LLM prompt
prompt = f"""
You are analyzing symbolic regression results.

{files_summary}

Based on the available files, decide what analysis to perform next.
"""

# LLM can now say:
# "I see there's already a correlation.png. I should read the
#  stats.json file to understand feature relationships before
#  calling the SR tool."
```

## Integration with Workflow

The file metadata system is automatically available through the workflow state:

```python
def my_tool_function(state):
    # Access workspace manager from state
    workspace_manager = state.get("_workspace_manager")

    if workspace_manager:
        # Tool can register files
        workspace_manager.register_file(...)

        # Tool can query existing files
        existing_plots = workspace_manager.list_files_with_metadata(file_type="plot")
```

## Metadata Storage

Metadata is stored in `.file_metadata.json` in the workspace root:

```json
{
  "input/data.csv": {
    "description": "Original dataset with features and target",
    "file_type": "data",
    "created_by": "workflow",
    "registered_at": "2025-12-10T21:42:54.513019",
    "size_bytes": 15234,
    "modified_at": "2025-12-10T21:42:54.510798",
    "source_path": "/external/path/to/data.csv"
  },
  "output/correlation.png": {
    "description": "Correlation heatmap of features",
    "file_type": "plot",
    "created_by": "python_interpreter",
    "registered_at": "2025-12-10T21:43:12.234567",
    "size_bytes": 45678,
    "modified_at": "2025-12-10T21:43:12.100000"
  }
}
```

## Best Practices

### For Tool Developers

1. **Always register output files** with descriptive metadata
2. **Be specific** in descriptions (what does the file contain?)
3. **Use standard file types** when possible
4. **Register immediately** after creating the file
5. **Include context** in description (e.g., "showing X with Y parameters")

### For LLM Prompt Engineering

1. **Include file summary** in LLM prompts when file context is needed
2. **Filter by relevance** (only include output/ files for analysis tasks)
3. **Use descriptions** to help LLM understand file contents
4. **Avoid re-creating** files that already exist

### Good Descriptions

✅ **Good**:
- "Correlation heatmap showing relationships between all 10 features"
- "Histogram of target variable distribution (1000 samples, 30 bins)"
- "Top 10 SR equations ranked by accuracy, complexity range 5-15"

❌ **Bad**:
- "Plot" (too generic)
- "Results" (what kind of results?)
- "Output file" (doesn't describe content)

## Testing

Run metadata tests:

```bash
cd /home/ubuntu/LLM-SR/agentsr
python test_metadata.py
```

All tests should pass with output showing:
- File registration
- Metadata retrieval
- Filtering capabilities
- LLM-friendly summaries
- Persistence across sessions

## API Reference

### WorkspaceManager Methods

#### `register_file(file_path, description, file_type=None, created_by=None, **extra_metadata)`
Register a file with metadata.

#### `get_file_metadata(file_path) -> Optional[Dict]`
Get metadata for a specific file.

#### `list_files_with_metadata(subdir=None, file_type=None) -> List[Dict]`
List all files matching filters with their metadata.

#### `get_files_summary() -> str`
Get LLM-friendly formatted summary of all workspace files.

#### `copy_input_file(src, dst_name=None, description=None) -> Path`
Copy file to workspace with automatic metadata registration.

## Migration Notes

- Existing workspaces without `.file_metadata.json` will start with empty metadata
- Tools can gradually adopt file registration
- Backward compatible: metadata is optional
- LLM nodes should check if metadata exists before using it

## Future Enhancements

Potential additions:
- File dependencies tracking
- Version history
- Tags/categories
- Search functionality
- Thumbnail generation for plots
- Automatic description generation using vision models
