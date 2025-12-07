# Logging Guide

## Overview

The agentsr codebase uses Python's built-in `logging` module to provide comprehensive logging throughout workflow execution. This allows you to track the execution flow, debug issues, and monitor the system's behavior.

## Logging Levels

The system uses standard Python logging levels:

- **DEBUG**: Detailed information for diagnosing problems. Shows all internal operations.
- **INFO**: Confirmation that things are working as expected. Default level.
- **WARNING**: Indication that something unexpected happened, but the software is still working.
- **ERROR**: A more serious problem that prevented a function from executing.
- **CRITICAL**: A serious error that may prevent the program from continuing.

## Configuration

### Quick Start

By default, logging is configured in `main.py` with INFO level:

```python
setup_logging(level=logging.INFO)
```

### Change Logging Level

To see more detailed logs, change to DEBUG:

```python
setup_logging(level=logging.DEBUG)
```

To see only warnings and errors:

```python
setup_logging(level=logging.WARNING)
```

### Using Configuration File

For more advanced configuration, you can use the `logging.conf` file:

```python
import logging.config

logging.config.fileConfig('logging.conf')
```

The default `logging.conf`:
- Logs INFO and above to console
- Logs DEBUG and above to `agentsr.log` file
- Uses a consistent timestamp format

## Log Messages by Component

### Workflow (`core.workflow`)

- Node addition and edge creation
- Workflow execution start/end
- Node transitions
- Branch decision logging

Example:
```
INFO - Starting workflow execution from node 'sr_analyzer'
INFO - Executing node 'sr_analyzer' (step 1)
INFO - Node 'sr_analyzer' is terminal, ending workflow
INFO - Workflow completed successfully. Visited nodes: ['sr_analyzer']
```

### LLMNode (`core.node`)

- System prompt loading
- Input building
- API calls to OpenAI
- Response parsing
- JSON extraction

Example:
```
INFO - [sr_analyzer] Starting LLMNode execution
DEBUG - [sr_analyzer] Building input from state
DEBUG - [sr_analyzer] Loaded system prompt from .../prompts/sr_analyzer.md
INFO - [sr_analyzer] Calling OpenAI API with model=gpt-4o-mini
DEBUG - [sr_analyzer] Prompt length: 15234 chars, temp=0.7, max_tokens=8192
INFO - [sr_analyzer] Received response from OpenAI API
DEBUG - [sr_analyzer] Parsing LLM output (length: 2345 chars)
INFO - [sr_analyzer] Successfully extracted JSON from response
DEBUG - [sr_analyzer] Extracted JSON keys: ['tool_call']
INFO - [sr_analyzer] Completed LLMNode execution
```

### SRNode (`nodes`)

- Tool specification loading
- File processing (CSV, images, etc.)
- Multi-modal content preparation

Example:
```
DEBUG - [sr_analyzer] Loading tool specifications for: ['pysr']
DEBUG - [sr_analyzer] Loaded tool spec: pysr
INFO - [sr_analyzer] Appended 1 tool spec(s) to system prompt
DEBUG - [sr_analyzer] Processing 1 file key(s)
INFO - [sr_analyzer] Added text file: bacres1.csv (30720 chars)
INFO - [sr_analyzer] Calling OpenAI Chat Completions API with model=gpt-4o-mini
DEBUG - [sr_analyzer] Message count: 2, content blocks: 2
```

### ToolSwitchNode (`nodes`)

- Tool selection from LLM response
- Next node determination

Example:
```
INFO - [tool_switch] Starting tool selection
INFO - [tool_switch] Selected tool from JSON: PySR
INFO - [tool_switch] Set next node to: tool_PySR
```

## Programmatic Control

### In Your Code

```python
import logging

# Get logger for your module
logger = logging.getLogger(__name__)

# Log at different levels
logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred")
```

### Adjusting Specific Modules

```python
# Set specific module to DEBUG while keeping others at INFO
logging.getLogger('nodes').setLevel(logging.DEBUG)
logging.getLogger('core.workflow').setLevel(logging.INFO)
```

### Disabling Logging

```python
# Disable all logging below CRITICAL
logging.disable(logging.CRITICAL)
```

## Log File Output

Logs are automatically written to `agentsr/logs/agentsr.log`. This file includes:

- All DEBUG level and above messages (when using file handler)
- Full execution trace
- Timestamped entries for debugging

The logs directory is created automatically on first run. Log files persist across runs (append mode).

To rotate log files or manage size, modify the `handler_fileHandler` section in `logging.conf` or implement log rotation:

```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    'logs/agentsr.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
```

## Best Practices

1. **Use INFO for progress tracking**: User-visible milestones
2. **Use DEBUG for internals**: Detailed execution flow
3. **Use WARNING for recoverable issues**: Missing optional files, defaults used
4. **Use ERROR for failures**: File not found, API errors
5. **Include context**: Use `[{node_name}]` prefix for node-specific logs

## Troubleshooting

### No logs appearing

- Check if logging is configured: `setup_logging()` must be called
- Verify log level is appropriate for the messages you expect
- Ensure handlers are properly configured

### Too many logs

- Increase log level: `setup_logging(level=logging.WARNING)`
- Disable specific modules: `logging.getLogger('module').setLevel(logging.ERROR)`

### Logs not in file

- Check that the `logs/` directory exists (should be auto-created)
- Verify file permissions for writing to `logs/agentsr.log`
- Check the file handler configuration
- Ensure `setup_logging()` is called before any logging occurs

## Environment Variables

You can control logging via environment variables:

```bash
# Set log level via environment
export AGENTSR_LOG_LEVEL=DEBUG
python main.py

# Or inline
AGENTSR_LOG_LEVEL=DEBUG python main.py
```

To use this, modify `setup_logging()`:

```python
import os

def setup_logging():
    level_name = os.getenv('AGENTSR_LOG_LEVEL', 'INFO')
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(level=level, ...)
```
