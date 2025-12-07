# AgentSR Implementation Summary

This document summarizes the current state of the AgentSR (Agent-based Symbolic Regression) implementation, including architecture, key features, and recent modifications.

## Overview

AgentSR is a workflow-based system for symbolic regression that uses LLM-powered nodes to analyze data and generate tool calls for external symbolic regression tools. The system is built on a flexible node-workflow architecture with comprehensive logging and multi-modal file support.

## Architecture

### Core Components

#### 1. Workflow System ([src/core/workflow.py](src/core/workflow.py))
- Directed graph execution engine for node-based workflows
- Supports sequential and branching execution paths
- Nodes connected via edges, execution flows from start node
- State dictionary passed between nodes, modified by each node
- Branching logic: nodes with multiple successors use `_next_node` in state
- Comprehensive logging of workflow execution and node transitions

#### 2. Base Node Classes ([src/core/node.py](src/core/node.py))

**Node (Abstract Base Class)**
- Interface: `run(state) -> state`
- All nodes must implement this method

**LLMNode**
- Wraps OpenAI API calls (uses Responses API)
- System prompt loading from markdown files in `prompts/`
- JSON extraction from LLM responses
- Configurable: model, temperature, max_tokens, parse_json
- Comprehensive logging of API calls and response parsing

**ToolNode**
- Executes predefined Python functions
- Maps state keys to function arguments
- Stores function output back to state

#### 3. Specialized Nodes ([src/nodes.py](src/nodes.py))

**SRNode (Symbolic Regression Node)**
- **Purpose**: Analyzes data files and generates symbolic regression tool calls
- **Inherits from**: LLMNode
- **Key Features**:
  - Multi-modal file handling (text + images)
  - Dynamic tool specification loading
  - Uses OpenAI Chat Completions API (not Responses API)
  - Token usage tracking and logging

**Key Parameters**:
- `file_keys`: List of state keys containing file paths to include
- `tool_list`: List of tool names to load specifications for
- `parse_json`: Automatically extracts JSON tool calls from responses

**File Handling**:
- Text files (CSV, JSON, code, etc.): Included as text content blocks
- Images (PNG, JPG, etc.): Base64-encoded and included as image_url blocks
- Files added to user message as separate content blocks

**Tool Specification Loading**:
- Loads markdown files from `tool_specs/{tool_name}.md`
- Appends tool specs to system prompt automatically
- Graceful handling of missing tool specs with warnings

**Token Usage Tracking** (Added in this session):
- Logs prompt tokens, completion tokens, and total tokens
- Logged at INFO level for visibility in console and log file
- Example: `Token usage - Prompt: 1523, Completion: 487, Total: 2010`

**ToolSwitchNode**
- Decides which tool to use based on data analysis
- Extracts tool selection from JSON response
- Sets `_next_node` in state for workflow branching

### Prompts and Tool Specifications

#### System Prompt: [prompts/sr_analyzer.md](prompts/sr_analyzer.md)
- Guides LLM to analyze data for symbolic regression
- Allows natural language reasoning followed by JSON tool call
- JSON format: `{"tool_call": {"tool_name": "...", "arguments": {...}}}`
- References available tools (specs appended dynamically)

#### Tool Specification: [tool_specs/pysr.md](tool_specs/pysr.md)
- Comprehensive documentation for PySR tool
- Sections:
  - Description and capabilities
  - Required arguments (binary_operators, unary_operators)
  - Optional arguments organized by category
  - Typical usage examples in JSON format
  - Parameter selection guidelines for different data types

## Logging System

### Configuration

**Setup Function** ([src/main.py](src/main.py)):
- `setup_logging()` creates logs directory and configures handlers
- Dual output: console (INFO+) and file (DEBUG+)
- Log file location: `agentsr/logs/agentsr.log`
- Logs directory auto-created on first run
- Append mode for persistent logging across runs

**Environment Variable Support**:
```bash
AGENTSR_LOG_LEVEL=DEBUG python main.py "query"
```

**Configuration File** ([logging.conf](logging.conf)):
- Console handler: INFO level
- File handler: DEBUG level
- Format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`

### Logging Coverage

**Workflow** ([src/core/workflow.py](src/core/workflow.py)):
- Node addition and edge creation
- Workflow execution start/end
- Node transitions and branching decisions
- Error conditions (duplicate nodes, invalid transitions)

**LLMNode** ([src/core/node.py](src/core/node.py)):
- System prompt loading
- Input building and API calls
- Response parsing and JSON extraction
- Errors (missing files, import failures)

**SRNode** ([src/nodes.py](src/nodes.py)):
- Tool specification loading
- File processing (text and images)
- Multi-modal content preparation
- **Token usage tracking** (added in this session)

**ToolSwitchNode** ([src/nodes.py](src/nodes.py)):
- Tool selection from LLM response
- Next node determination

### Log Output Examples

```
INFO - Starting workflow execution from node 'sr_analyzer'
INFO - Executing node 'sr_analyzer' (step 1)
DEBUG - [sr_analyzer] Loading tool specifications for: ['pysr']
DEBUG - [sr_analyzer] Loaded tool spec: pysr
INFO - [sr_analyzer] Appended 1 tool spec(s) to system prompt
INFO - [sr_analyzer] Added text file: bacres1.csv (30720 chars)
INFO - [sr_analyzer] Calling OpenAI Chat Completions API with model=gpt-4o-mini
INFO - [sr_analyzer] Token usage - Prompt: 1523, Completion: 487, Total: 2010
INFO - [sr_analyzer] Successfully extracted JSON from response
INFO - Workflow completed successfully. Visited nodes: ['sr_analyzer']
```

## Entry Point

### [src/main.py](src/main.py)

**Features**:
- Command-line or interactive input modes
- Logging setup with environment variable support
- Example workflow with SRNode
- Error handling and user guidance

**Usage**:
```bash
# Interactive mode
python main.py

# Command-line argument
python main.py "analyze this data"

# With debug logging
AGENTSR_LOG_LEVEL=DEBUG python main.py "analyze this data"
```

**Example Configuration**:
```python
sr_node = SRNode(
    name="sr_analyzer",
    system_prompt="sr_analyzer",
    file_keys=["data_file"],
    tool_list=["pysr"],
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=8192,
    parse_json=True,
)
```

## Data Flow Example

1. **User Input**: Query + data file path in initial state
2. **SRNode Execution**:
   - Loads system prompt from `prompts/sr_analyzer.md`
   - Loads tool specs from `tool_specs/pysr.md`
   - Reads data file (CSV) and encodes as text block
   - Builds multi-modal message with system prompt + user content
   - Calls OpenAI Chat Completions API
   - Logs token usage
   - Extracts JSON tool call from response
3. **State Update**:
   - `llm_response`: Raw LLM output
   - `parsed_json`: Extracted tool call JSON
4. **Workflow Completion**: Returns final state with analysis and tool call

## Recent Modifications (This Session)

### Token Usage Tracking Implementation

**File Modified**: [src/nodes.py](src/nodes.py)

**Location**: `SRNode._call_llm()` method (lines 305-312)

**Changes**:
```python
# Log token usage
if hasattr(response, 'usage') and response.usage:
    logger.info(
        f"[{self.name}] Token usage - "
        f"Prompt: {response.usage.prompt_tokens}, "
        f"Completion: {response.usage.completion_tokens}, "
        f"Total: {response.usage.total_tokens}"
    )
```

**Impact**:
- Automatic logging of token consumption for each API call
- Visible in both console output (INFO level) and log file
- Helps track API costs and monitor resource usage
- No breaking changes to existing functionality

**Why This Matters**:
- Enables monitoring of API costs in real-time
- Helps optimize prompt engineering by showing token distribution
- Provides data for performance analysis and budgeting
- Essential for production deployments with cost constraints

## File Structure

```
agentsr/
├── src/
│   ├── core/
│   │   ├── node.py           # Base node classes (Node, LLMNode, ToolNode)
│   │   ├── workflow.py       # Workflow execution engine
│   │   └── consts.py         # Constants (ROOT_DIR, SRC_DIR)
│   ├── nodes.py              # Specialized nodes (SRNode, ToolSwitchNode)
│   └── main.py               # Entry point with logging setup
├── prompts/
│   └── sr_analyzer.md        # System prompt for SR analysis
├── tool_specs/
│   └── pysr.md               # PySR tool specification
├── logs/
│   ├── .gitignore            # Excludes *.log files
│   └── agentsr.log           # Log file (auto-created)
├── logging.conf              # Logging configuration file
├── LOGGING.md                # Logging documentation
└── CLAUDE.md                 # This file
```

## Key Design Decisions

### 1. Multi-Modal API Selection
- **Choice**: OpenAI Chat Completions API (not Responses API)
- **Reason**: Supports multi-modal content blocks (text + images)
- **Implementation**: SRNode uses Chat Completions, base LLMNode uses Responses

### 2. Dynamic Tool Loading
- **Choice**: Tool specs loaded via `tool_list` parameter
- **Reason**: Flexibility to mix and match tools per workflow
- **Implementation**: Markdown files loaded and appended to system prompt

### 3. Flexible Output Format
- **Choice**: Natural language + JSON (not JSON-only)
- **Reason**: Allows LLM to explain reasoning before tool call
- **Implementation**: JSON extraction with regex parsing

### 4. Dual Logging Output
- **Choice**: Console (INFO+) and file (DEBUG+)
- **Reason**: User-friendly console, detailed debugging in file
- **Implementation**: Multiple handlers in `logging.basicConfig()`

### 5. Token Usage Tracking
- **Choice**: Automatic logging via response object
- **Reason**: Zero-overhead monitoring, no API changes needed
- **Implementation**: Check for `response.usage` attribute after API call

## Future Enhancements (Potential)

1. **Tool Execution**: Implement actual PySR execution nodes
2. **Result Validation**: Validate tool calls before execution
3. **Caching**: Cache LLM responses for repeated queries
4. **Cost Tracking**: Aggregate token usage across workflow runs
5. **Async Execution**: Support for parallel node execution
6. **Custom Metrics**: Log additional metrics (latency, accuracy, etc.)
7. **State Persistence**: Save/load workflow state for resumption
8. **Tool Registry**: Dynamic tool discovery and registration

## Documentation

- **Logging Guide**: [LOGGING.md](LOGGING.md) - Comprehensive logging documentation
- **Tool Specs**: [tool_specs/](tool_specs/) - Tool specifications for LLM reference
- **Prompts**: [prompts/](prompts/) - System prompts for different node types

## Dependencies

- `openai`: OpenAI Python SDK for API calls
- Python standard library: `logging`, `json`, `re`, `os`, `base64`

## Testing Workflow

To test the current implementation:

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Run with INFO logging (default)
python src/main.py "Analyze this data and suggest a symbolic regression approach"

# Run with DEBUG logging for detailed trace
AGENTSR_LOG_LEVEL=DEBUG python src/main.py "Analyze this data"

# Check the log file
tail -f logs/agentsr.log
```

Expected output:
- Console: Workflow execution progress and LLM analysis
- Console: Token usage statistics
- Log file: Detailed execution trace with all debug information
- State: `llm_response` (raw) and `parsed_json` (tool call)

## Notes

- Log files are excluded from git (see `logs/.gitignore`)
- System prompts and tool specs are in markdown for easy editing
- Token usage is logged automatically for all SRNode API calls
- Environment variable `AGENTSR_LOG_LEVEL` overrides default log level
- File paths in state are read and included as multi-modal content
- JSON extraction is resilient to various formats (code blocks, inline)

---

**Last Updated**: 2025-12-07
**Session Focus**: Token usage tracking implementation
**Status**: Functional with comprehensive logging and file support
