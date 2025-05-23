# MultiMind Gateway API Reference

This document provides detailed API documentation for the MultiMind Gateway module.

## Table of Contents

- [Core Components](#core-components)
- [Model Handlers](#model-handlers)
- [Chat Session Management](#chat-session-management)
- [Monitoring and Metrics](#monitoring-and-metrics)
- [CLI Interface](#cli-interface)
- [REST API](#rest-api)

## Core Components

### Configuration

```python
from multimind.gateway import config

# Get configuration
settings = config.validate()

# Access specific settings
openai_key = settings.get("openai", {}).get("api_key")
default_model = settings.get("default_model")
```

#### Configuration Options

| Setting | Type | Description | Default |
|---------|------|-------------|---------|
| `OPENAI_API_KEY` | str | OpenAI API key | None |
| `OPENAI_MODEL_NAME` | str | Default OpenAI model | "gpt-3.5-turbo" |
| `ANTHROPIC_API_KEY` | str | Anthropic API key | None |
| `ANTHROPIC_MODEL_NAME` | str | Default Anthropic model | "claude-3-opus-20240229" |
| `OLLAMA_API_BASE` | str | Ollama API base URL | "http://localhost:11434" |
| `OLLAMA_MODEL_NAME` | str | Default Ollama model | "mistral" |
| `GROQ_API_KEY` | str | Groq API key | None |
| `GROQ_MODEL_NAME` | str | Default Groq model | "mixtral-8x7b-32768" |
| `HUGGINGFACE_API_KEY` | str | HuggingFace API key | None |
| `HUGGINGFACE_MODEL_NAME` | str | Default HuggingFace model | "mistralai/Mistral-7B-Instruct-v0.2" |
| `DEFAULT_MODEL` | str | Default model provider | "openai" |
| `LOG_LEVEL` | str | Logging level | "INFO" |

## Model Handlers

### Getting a Model Handler

```python
from multimind.gateway import get_model_handler

# Get handler for specific model
handler = get_model_handler("openai")
```

### Model Handler Interface

```python
class ModelHandler:
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Send a chat request to the model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Optional model name override
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            **kwargs: Additional model-specific parameters
            
        Returns:
            ModelResponse object containing:
            - content: str
            - model: str
            - usage: Dict[str, int]
            - metadata: Dict[str, Any]
        """
        pass

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            model: Optional model name override
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            **kwargs: Additional model-specific parameters
            
        Returns:
            ModelResponse object
        """
        pass
```

## Chat Session Management

### Chat Manager

```python
from multimind.gateway import chat_manager

# Create a new session
session = chat_manager.create_session(
    model: str,
    system_prompt: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> ChatSession

# Get a session
session = chat_manager.get_session(session_id: str) -> ChatSession

# List all sessions
sessions = chat_manager.list_sessions() -> List[ChatSession]

# Save a session
file_path = chat_manager.save_session(session_id: str) -> str

# Delete a session
chat_manager.delete_session(session_id: str) -> None
```

### Chat Session

```python
class ChatSession:
    def add_message(
        self,
        role: str,
        content: str,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a message to the session."""
        pass

    def get_messages(
        self,
        limit: Optional[int] = None,
        before: Optional[datetime] = None
    ) -> List[ChatMessage]:
        """Get messages from the session."""
        pass

    def export(self, format: str = "json") -> str:
        """Export session data."""
        pass

    @classmethod
    def from_file(cls, file_path: str) -> "ChatSession":
        """Load session from file."""
        pass
```

## Monitoring and Metrics

### Model Monitor

```python
from multimind.gateway import monitor

# Track a request
await monitor.track_request(
    model: str,
    tokens: int,
    cost: float,
    response_time: float,
    success: bool,
    error: Optional[str] = None
) -> None

# Get metrics
metrics = await monitor.get_metrics() -> Dict[str, Dict[str, Any]]

# Check model health
health = await monitor.check_health(
    model: str,
    handler: ModelHandler
) -> ModelHealth

# Set rate limits
monitor.set_rate_limits(
    model: str,
    requests_per_minute: int,
    tokens_per_minute: int
) -> None

# Check rate limit
can_proceed = await monitor.check_rate_limit(
    model: str,
    tokens: int
) -> bool
```

### Metrics Structure

```python
@dataclass
class ModelMetrics:
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_tokens: int
    total_cost: float
    avg_response_time: float
    error_counts: Dict[str, int]

@dataclass
class ModelHealth:
    is_healthy: bool
    last_check: datetime
    error_message: Optional[str]
    latency_ms: Optional[float]
```

## CLI Interface

### Command Line Usage

```bash
# Chat with a model
multimind chat [OPTIONS]
  --model TEXT              Model to use
  --prompt TEXT            Initial prompt
  --temperature FLOAT      Sampling temperature
  --max-tokens INTEGER     Maximum tokens in response
  --interactive            Start interactive chat session

# Compare models
multimind compare [OPTIONS] PROMPT
  --models TEXT            Comma-separated list of models
  --temperature FLOAT      Sampling temperature
  --max-tokens INTEGER     Maximum tokens in response

# Monitor metrics
multimind metrics [OPTIONS]
  --model TEXT             Specific model to show metrics for
  --format TEXT            Output format (table/json)

# Manage sessions
multimind sessions [OPTIONS]
  --list                   List all sessions
  --load TEXT              Load a session
  --save TEXT              Save current session
  --delete TEXT            Delete a session
```

## REST API

### Endpoints

#### Chat

```http
POST /v1/chat
Content-Type: application/json

{
    "messages": [
        {"role": "user", "content": "Hello!"}
    ],
    "model": "openai",
    "temperature": 0.7,
    "max_tokens": 100
}

Response:
{
    "content": "Hi there!",
    "model": "openai",
    "usage": {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5
    }
}
```

#### Generate

```http
POST /v1/generate
Content-Type: application/json

{
    "prompt": "Write a poem about AI",
    "model": "anthropic",
    "temperature": 0.8,
    "max_tokens": 200
}

Response:
{
    "content": "...",
    "model": "anthropic",
    "usage": {
        "prompt_tokens": 7,
        "completion_tokens": 50,
        "total_tokens": 57
    }
}
```

#### Compare

```http
POST /v1/compare
Content-Type: application/json

{
    "prompt": "What is AI?",
    "models": ["openai", "anthropic"],
    "temperature": 0.7,
    "max_tokens": 100
}

Response:
{
    "responses": [
        {
            "model": "openai",
            "content": "...",
            "usage": {...}
        },
        {
            "model": "anthropic",
            "content": "...",
            "usage": {...}
        }
    ]
}
```

#### Sessions

```http
# Create session
POST /v1/sessions
Content-Type: application/json

{
    "model": "openai",
    "system_prompt": "You are a helpful assistant",
    "metadata": {"purpose": "customer_support"}
}

# List sessions
GET /v1/sessions

# Get session
GET /v1/sessions/{session_id}

# Add message
POST /v1/sessions/{session_id}/messages
Content-Type: application/json

{
    "role": "user",
    "content": "Hello!",
    "metadata": {"topic": "greeting"}
}

# Delete session
DELETE /v1/sessions/{session_id}
```

#### Metrics and Health

```http
# Get metrics
GET /v1/metrics
GET /v1/metrics?model=openai

# Check health
POST /v1/health/check
POST /v1/health/check?model=anthropic
```

### Error Responses

All endpoints return standard HTTP status codes and error responses in the format:

```json
{
    "error": {
        "code": "ERROR_CODE",
        "message": "Human readable error message",
        "details": {
            "field": "Additional error details"
        }
    }
}
```

Common error codes:
- `INVALID_MODEL`: Model not supported or not configured
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `INVALID_REQUEST`: Malformed request
- `MODEL_ERROR`: Error from the model provider
- `SESSION_NOT_FOUND`: Chat session not found
- `VALIDATION_ERROR`: Invalid parameters

## Examples

For complete working examples, see:
- [Gateway Examples](../../examples/gateway_examples.py)
- [Chat Examples](../../examples/chat_ollama_cli.py)
- [RAG Examples](../../examples/rag_example.py)
- [Advanced RAG](../../examples/rag_advanced_example.py) 