# Configuration Guide

This guide covers all configuration options available in the MultiMind SDK.

## Environment Variables

### API Keys
```bash
# Required for OpenAI models
OPENAI_API_KEY=your_openai_api_key

# Required for Anthropic models
ANTHROPIC_API_KEY=your_anthropic_api_key

# Required for Mistral models
MISTRAL_API_KEY=your_mistral_api_key
```

### Default Settings
```bash
# Default model to use
DEFAULT_MODEL=gpt-3.5-turbo

# Default temperature
DEFAULT_TEMPERATURE=0.7

# Default max tokens
DEFAULT_MAX_TOKENS=2000

# Logging level
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## Model Configuration

### OpenAI Models
```python
from multimind import OpenAIModel

model = OpenAIModel(
    model="gpt-3.5-turbo",  # or "gpt-4"
    temperature=0.7,
    max_tokens=2000,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=None,  # List of stop sequences
    timeout=30,  # Request timeout in seconds
    retry_attempts=3,  # Number of retry attempts
    retry_delay=1.0  # Delay between retries in seconds
)
```

### Claude Models
```python
from multimind import ClaudeModel

model = ClaudeModel(
    model="claude-3-sonnet",  # or "claude-3-opus"
    temperature=0.7,
    max_tokens=2000,
    top_p=1.0,
    top_k=40,
    stop_sequences=None,
    timeout=30,
    retry_attempts=3,
    retry_delay=1.0
)
```

### Mistral Models
```python
from multimind import MistralModel

model = MistralModel(
    model="mistral-medium",  # or "mistral-small"
    temperature=0.7,
    max_tokens=2000,
    top_p=1.0,
    timeout=30,
    retry_attempts=3,
    retry_delay=1.0
)
```

## Agent Configuration

### Memory Settings
```python
from multimind import AgentMemory

memory = AgentMemory(
    max_history=50,  # Maximum number of interactions to store
    max_tokens=2000,  # Maximum tokens per memory entry
    include_metadata=True,  # Whether to store metadata
    metadata_fields=["user_id", "session_id"],  # Custom metadata fields
    storage_backend="memory",  # or "redis", "database"
    storage_config={  # Backend-specific configuration
        "redis_url": "redis://localhost:6379",
        "database_url": "postgresql://user:pass@localhost/db"
    }
)
```

### Tool Configuration
```python
from multimind import CalculatorTool, WebSearchTool

tools = [
    CalculatorTool(
        max_digits=10,  # Maximum decimal places
        timeout=5.0  # Tool execution timeout
    ),
    WebSearchTool(
        max_results=5,  # Maximum search results
        timeout=10.0,  # Search timeout
        api_key="your_search_api_key"  # Optional API key
    )
]
```

## Task Runner Configuration

```python
from multimind import TaskRunner

runner = TaskRunner(
    max_concurrent_tasks=5,  # Maximum concurrent tasks
    timeout=300,  # Global timeout in seconds
    retry_attempts=3,  # Number of retry attempts
    retry_delay=1.0,  # Delay between retries
    error_handler=None,  # Custom error handler
    progress_callback=None  # Progress tracking callback
)
```

## MCP Configuration

### Workflow Definition
```python
workflow = {
    "version": "1.0.0",
    "models": {
        "gpt-3.5": {
            "temperature": 0.7,
            "max_tokens": 2000
        },
        "claude-3": {
            "temperature": 0.7,
            "max_tokens": 2000
        }
    },
    "steps": [
        {
            "name": "analysis",
            "model": "gpt-3.5",
            "prompt": "Analyze: {input}",
            "timeout": 30,
            "retry_attempts": 3
        }
    ],
    "connections": [
        {
            "from": "analysis",
            "to": "review",
            "condition": "success"  # or "error", "always"
        }
    ],
    "error_handling": {
        "strategy": "retry",  # or "skip", "fail"
        "max_retries": 3,
        "retry_delay": 1.0
    }
}
```

## Logging Configuration

### Usage Tracking
```python
from multimind import UsageTracker

tracker = UsageTracker(
    enabled=True,
    log_file="usage.log",
    log_format="json",  # or "csv", "text"
    export_interval=3600,  # Export interval in seconds
    export_path="usage_reports/",
    cost_tracking=True,
    token_tracking=True
)
```

### Trace Logging
```python
from multimind import TraceLogger

logger = TraceLogger(
    enabled=True,
    log_file="traces.log",
    log_level="INFO",
    include_metadata=True,
    max_trace_duration=3600,  # Maximum trace duration in seconds
    export_format="json"  # or "csv", "text"
)
```

## CLI Configuration

### Command Line Options
```bash
# Set default model
multimind --model gpt-3.5-turbo

# Set temperature
multimind --temperature 0.7

# Set max tokens
multimind --max-tokens 2000

# Enable verbose logging
multimind --verbose

# Set config file
multimind --config config.yaml
```

### Configuration File (config.yaml)
```yaml
# Model settings
model:
  name: gpt-3.5-turbo
  temperature: 0.7
  max_tokens: 2000

# Agent settings
agent:
  memory:
    max_history: 50
    max_tokens: 2000
  tools:
    - name: calculator
      enabled: true
    - name: web_search
      enabled: true
      api_key: ${SEARCH_API_KEY}

# Task runner settings
task_runner:
  max_concurrent_tasks: 5
  timeout: 300
  retry_attempts: 3

# Logging settings
logging:
  level: INFO
  file: multimind.log
  format: json
```

## Best Practices

1. **Environment Variables**
   - Use environment variables for sensitive data
   - Keep configuration in version control
   - Use different configs for different environments

2. **Model Selection**
   - Choose models based on task requirements
   - Consider cost and performance trade-offs
   - Monitor model usage and costs

3. **Resource Management**
   - Set appropriate timeouts
   - Implement retry strategies
   - Monitor memory usage

4. **Security**
   - Never commit API keys
   - Use secure storage for sensitive data
   - Implement proper access controls

5. **Monitoring**
   - Enable usage tracking
   - Set up logging
   - Monitor performance metrics

## Troubleshooting

### Common Issues

1. **Configuration Not Loaded**
   - Check file permissions
   - Verify environment variables
   - Check config file syntax

2. **Model Access**
   - Verify API keys
   - Check model availability
   - Confirm account status

3. **Resource Limits**
   - Check rate limits
   - Monitor token usage
   - Verify timeout settings

### Getting Help

- Check the [FAQ](../docs/faq.md)
- Open an issue on [GitHub](https://github.com/multimind-dev/multimind-sdk/issues)
- Contact support at [support@multimind.dev](mailto:support@multimind.dev) 