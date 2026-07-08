# Configuration Guide

This guide covers all configuration options available in the MultiMind SDK.

## Environment Variables

### API Keys

```bash
# Required for OpenAI models
OPENAI_API_KEY=your_openai_api_key

# Required for Anthropic models (CLAUDE_API_KEY is also accepted)
ANTHROPIC_API_KEY=your_anthropic_api_key

# Required for Mistral AI (hosted) models
MISTRAL_API_KEY=your_mistral_api_key

# Required for Groq models
GROQ_API_KEY=your_groq_api_key

# Required for Gemini models (GOOGLE_API_KEY is also accepted)
GEMINI_API_KEY=your_gemini_api_key

# Required for DeepSeek models
DEEPSEEK_API_KEY=your_deepseek_api_key

# Optional for Hugging Face models (falls back to local transformers)
HUGGINGFACE_API_KEY=your_huggingface_api_key
```

### Default Settings

```bash
# Default model provider for the gateway (default: openai)
DEFAULT_MODEL=openai

# Logging level for the gateway
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Log level for optional-dependency warnings at import time (default: WARNING)
MULTIMIND_LOG_LEVEL=WARNING

# Path to a YAML config file loaded by multimind.Config
MULTIMIND_CONFIG=config.yaml

# Ollama server URL (default: http://localhost:11434)
OLLAMA_HOST=http://localhost:11434
```

### Compliance Proxy (`multimind serve`)

Every `multimind serve` flag can also be set through an environment variable
(CLI flags win):

```bash
MULTIMIND_PROXY_UPSTREAM=openai        # openai, groq, mistral, gemini, deepseek, ollama
MULTIMIND_UPSTREAM_BASE_URL=           # custom OpenAI-compatible upstream URL
MULTIMIND_UPSTREAM_API_KEY=            # explicit upstream key (else provider env var is used)
MULTIMIND_PROXY_STRATEGY=mask          # mask, hash, remove
MULTIMIND_PROXY_BLOCK_ON=ssn,credit_card
MULTIMIND_PROXY_AUDIT_LOG=audit.jsonl
MULTIMIND_PROXY_BUDGET=10.0
MULTIMIND_PROXY_COST_PER_TOKEN=
MULTIMIND_PROXY_SCAN_OUTPUT=true
MULTIMIND_PROXY_HOST=127.0.0.1
MULTIMIND_PROXY_PORT=8400
```

See the [Guard Proxy guide](guard-proxy.md) for details.

### API Servers and Misc

```bash
# Comma-separated API keys; enables X-API-Key auth on the REST APIs when set
API_KEYS=key1,key2

# Enables JWT bearer auth on the RAG gateway when set
JWT_SECRET=your_jwt_secret
JWT_USERS_JSON='{"user": "password_hash"}'

# CORS is off unless this is set (comma-separated origins)
MULTIMIND_CORS_ORIGINS=https://app.example.com

# Audit log path for the MCP server
MULTIMIND_AUDIT_LOG=audit_log.jsonl

# Opt-in warnings for missing optional backends / legacy compliance imports
MULTIMIND_SHOW_BACKEND_WARNINGS=false
MULTIMIND_SHOW_LEGACY_WARNINGS=false
```

## Model Configuration

Generation parameters such as `temperature` and `max_tokens` are passed per
call (e.g. to `generate()`), not to the model constructor.

### OpenAI Models

```python
from multimind import OpenAIModel

model = OpenAIModel(
    model_name="gpt-4o-mini",
    api_key=None,  # falls back to OPENAI_API_KEY
    base_url=None,  # custom OpenAI-compatible endpoint
    cost_per_token=None,  # override for cost tracking
)
```

### Claude Models

```python
from multimind import ClaudeModel

model = ClaudeModel(
    model_name="claude-3-opus-20240229",
    api_key=None,  # falls back to ANTHROPIC_API_KEY or CLAUDE_API_KEY
)
```

### Mistral Models

`MistralModel` runs Mistral models locally through Ollama; `MistralAIModel`
uses the hosted Mistral AI (La Plateforme) API via `MISTRAL_API_KEY`.

```python
from multimind import MistralModel
from multimind.models.mistral import MistralAIModel

local_model = MistralModel(
    model="mistral",
    base_url="http://localhost:11434",  # Ollama server
)

hosted_model = MistralAIModel(
    model_name="mistral-small-latest",
    api_key=None,  # falls back to MISTRAL_API_KEY
)
```

## Agent Configuration

### Memory Settings

```python
from multimind import AgentMemory

memory = AgentMemory(
    max_history=100,  # Maximum number of interactions to store
)
```

### Tool Configuration

```python
from multimind import CalculatorTool

tools = [
    CalculatorTool(),
]
```

## Task Runner Configuration

```python
from multimind import OpenAIModel, TaskRunner

runner = TaskRunner(
    model=OpenAIModel(model_name="gpt-4o-mini"),
    tasks=None,  # optional list of task dicts
    max_retries=3,  # Number of retry attempts per task
)
```

## MCP Configuration

### Workflow Definition

An MCP spec must contain `version`, `models` (a list of `{name, type, config}`
entries; types: `openai`, `claude`, `mistral`, `huggingface`, `ollama`) and a
`workflow` with `steps` (`{id, type, config}`; types: `model`, `transform`,
`condition`) and `connections` (`{from, to}` between existing step ids):

```python
workflow = {
    "version": "1.0.0",
    "models": [
        {
            "name": "gpt",
            "type": "openai",
            "config": {"model": "gpt-4o-mini", "temperature": 0.7}
        }
    ],
    "workflow": {
        "steps": [
            {
                "id": "analysis",
                "type": "model",
                "config": {"model": "gpt", "prompt": "Analyze: {input}"}
            },
            {
                "id": "review",
                "type": "model",
                "config": {"model": "gpt", "prompt": "Review: {analysis}"}
            }
        ],
        "connections": [
            {"from": "analysis", "to": "review"}
        ]
    }
}
```

## Logging Configuration

### Usage Tracking

```python
from multimind import UsageTracker

tracker = UsageTracker(
    db_path=None,  # SQLite database path (default location if None)
)
```

### Trace Logging

```python
import logging

from multimind import TraceLogger

logger = TraceLogger(
    log_dir=None,  # Directory for trace logs (default location if None)
    log_level=logging.INFO,
)
```

## CLI Configuration

### Command Line Options

```bash
# Show environment and configuration info
multimind config info

# View or set global CLI configuration
multimind config manage
multimind config manage --set default_model openai
multimind config manage --get default_model
```

Global CLI configuration is stored as JSON in `~/.multimind_cli_config`.

### Configuration File (config.yaml)

The SDK-level `multimind.Config` loads a YAML file from the path given by the
`MULTIMIND_CONFIG` environment variable (or an explicit `config_path`).
Environment variables override file values for API keys and the Ollama host:

```yaml
# Provider settings (api_key is overridden by the matching env var)
openai:
  api_key: your_openai_api_key
anthropic:
  api_key: your_anthropic_api_key
mistral:
  api_key: your_mistral_api_key
huggingface:
  api_key: your_huggingface_api_key
ollama:
  host: http://localhost:11434

# Per-model parameters, read via Config.get_model_params(model_type, model_name)
models:
  openai:
    gpt-4o-mini:
      temperature: 0.7
      max_tokens: 2000
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

- Check the [FAQ](multimind-sdk-faq.md)
- Open an issue on [GitHub](https://github.com/multimindlab/multimind-sdk/issues)
- Contact support at [contact@multimind.dev](mailto:contact@multimind.dev)
