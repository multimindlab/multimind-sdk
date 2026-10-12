---
title: "MultiMind Gateway API Reference"
---

This document covers the `multimind.gateway` package: its configuration, model handlers, chat sessions, monitoring, and the FastAPI app in `multimind.gateway.api`. It requires the `gateway` extra:

```bash
pip install "multimind-sdk[gateway]"
```

## Table of Contents

- [Core Components](#core-components)
- [Model Handlers](#model-handlers)
- [Chat Session Management](#chat-session-management)
- [Monitoring and Metrics](#monitoring-and-metrics)
- [CLI Interface](#cli-interface)
- [REST API](#rest-api)

## Core Components

### Configuration

The gateway reads its settings from environment variables (a `.env` file in the working directory is loaded automatically via `python-dotenv`) into a `GatewayConfig` instance:

```python
from multimind.gateway.config import config   # GatewayConfig instance

openai_key = config.openai.api_key            # ModelConfig per provider
default_model = config.default_model          # "openai" unless DEFAULT_MODEL is set
model_cfg = config.get_model_config("claude") # ModelConfig(api_key, api_base, model_name, temperature, max_tokens, timeout)
```

Each provider section is a `ModelConfig` with `api_key`, `api_base`, `model_name`, `temperature` (0.7), `max_tokens` (None) and `timeout` (30). `GatewayConfig` validates on construction that `DEFAULT_MODEL` names a known provider.

#### Configuration Options

| Setting | Type | Description | Default |
|---------|------|-------------|---------|
| `OPENAI_API_KEY` | str | OpenAI API key | None |
| `OPENAI_MODEL_NAME` | str | Default OpenAI model | "gpt-3.5-turbo" |
| `ANTHROPIC_API_KEY` (or `CLAUDE_API_KEY`) | str | Anthropic API key (used by the `anthropic` and `claude` handlers) | None |
| `ANTHROPIC_MODEL_NAME` | str | Default Anthropic model | "claude-3-opus-20240229" (`anthropic`), "claude-3-5-sonnet-20241022" (`claude`) |
| `OLLAMA_API_BASE` | str | Ollama API base URL | "http://localhost:11434" |
| `OLLAMA_MODEL_NAME` | str | Default Ollama model | "mistral" |
| `GROQ_API_KEY` | str | Groq API key | None |
| `GROQ_MODEL_NAME` | str | Default Groq model | "mixtral-8x7b-32768" |
| `HUGGINGFACE_API_KEY` | str | HuggingFace API key (if unset, local `transformers` is used) | None |
| `HUGGINGFACE_MODEL_NAME` | str | Default HuggingFace model | "mistralai/Mistral-7B-Instruct-v0.2" |
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) / `GEMINI_MODEL_NAME` | str | Google Gemini | "gemini-2.0-flash" |
| `MISTRAL_API_KEY` / `MISTRAL_MODEL_NAME` | str | Mistral AI | "mistral-small-latest" |
| `DEEPSEEK_API_KEY` / `DEEPSEEK_MODEL_NAME` | str | DeepSeek | "deepseek-chat" |
| `OPENROUTER_API_KEY` / `OPENROUTER_MODEL_NAME` | str | OpenRouter | "openai/gpt-4o-mini" |
| `TOGETHER_API_KEY` / `TOGETHER_MODEL_NAME` | str | Together | "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo" |
| `XAI_API_KEY` / `XAI_MODEL_NAME` | str | xAI | "grok-2-latest" |
| `PERPLEXITY_API_KEY` / `PERPLEXITY_MODEL_NAME` | str | Perplexity | "sonar" |
| `FIREWORKS_API_KEY` / `FIREWORKS_MODEL_NAME` | str | Fireworks | "accounts/fireworks/models/llama-v3p1-8b-instruct" |
| `CEREBRAS_API_KEY` / `CEREBRAS_MODEL_NAME` | str | Cerebras | "llama3.1-8b" |
| `DEFAULT_MODEL` | str | Default model provider | "openai" |
| `LOG_LEVEL` | str | Logging level | "INFO" |
| `MULTIMIND_CORS_ORIGINS` | str | Comma-separated CORS origins (CORS is off when unset) | unset |

## Model Handlers

### Getting a Model Handler

```python
from multimind.gateway.models import get_model_handler

handler = get_model_handler("openai")
```

Accepted names: `openai`, `anthropic`, `ollama`, `huggingface` (dedicated handlers exported from `multimind.gateway` as `OpenAIHandler`, `AnthropicHandler`, `OllamaHandler`, `HuggingFaceHandler`) and `claude`, `groq`, `gemini`, `mistral`, `deepseek`, `openrouter`, `together`, `xai`, `perplexity`, `fireworks`, `cerebras` (wrapped SDK model classes). Unknown names raise `ValueError`.

### Model Handler Interface

```python
from multimind.core.models import ModelHandler, ModelResponse

class ModelHandler(ABC):
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """messages: dicts with 'role' and 'content'. kwargs: temperature, max_tokens, ..."""

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate text from a prompt."""

@dataclass
class ModelResponse:
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None   # prompt_tokens / completion_tokens / total_tokens when the provider reports them
    finish_reason: Optional[str] = None
    timestamp: str                           # ISO-8601, set automatically
```

## Chat Session Management

### Chat Manager

```python
from multimind.gateway.chat import chat_manager   # ChatManager(storage_dir="chat_sessions")

session = chat_manager.create_session(model="openai", system_prompt=None, metadata=None)  # -> ChatSession
session = chat_manager.get_session(session_id)        # -> Optional[ChatSession] (active sessions only)
sessions = chat_manager.list_sessions()               # -> List[Dict]: session_id, model, created_at, updated_at, message_count
path = chat_manager.save_session(session_id)          # -> Optional[Path]  (chat_sessions/chat_<id>.json)
session = chat_manager.load_session(session_id)       # -> Optional[ChatSession] from storage_dir
deleted = chat_manager.delete_session(session_id)     # -> bool (also removes the saved file)
```

### Chat Session

```python
from multimind.core.chat import ChatSession, ChatMessage   # pydantic models

class ChatSession:
    session_id: str
    model: str
    created_at: datetime
    updated_at: datetime
    messages: List[ChatMessage]      # role, content, model, timestamp, metadata
    metadata: Dict
    system_prompt: Optional[str]

    def add_message(self, role: str, content: str, model: str, metadata: Optional[Dict] = None) -> None
    def get_context(self, max_messages: int = 10) -> List[Dict[str, str]]   # recent {"role", "content"} dicts
    def export(self, export_format: str = "json") -> Union[str, Dict]
    def save(self, directory) -> Path

    @classmethod
    def from_file(cls, file_path) -> "ChatSession"
```

## Monitoring and Metrics

### Model Monitor

```python
from multimind.gateway.monitoring import monitor, ModelMetrics, ModelHealth   # monitor is a ModelMonitor singleton

await monitor.track_request(model="openai", tokens=120, cost=0.0, response_time=0.8, success=True, error=None)
metrics = await monitor.get_metrics(model=None)        # Dict; one entry per model, or the single model's metrics
health = await monitor.check_health("openai", handler)  # -> ModelHealth (sends a short test request)
monitor.set_rate_limits("openai", requests_per_minute=60, tokens_per_minute=100_000)  # keyword-only
ok = await monitor.check_rate_limit("openai", tokens=500)   # -> bool
monitor.health   # Dict[str, ModelHealth] of the latest health checks
```

### Metrics Structure

```python
@dataclass
class ModelMetrics:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_response_time: float = 0.0
    last_used: Optional[datetime] = None
    error_count: Dict[str, int]        # error message -> count

class ModelHealth(BaseModel):
    is_healthy: bool
    last_check: datetime
    error_message: Optional[str] = None
    latency_ms: Optional[float] = None
    uptime_percentage: float = 100.0
```

Metrics are in-process only; they reset when the server restarts.

## CLI Interface

The gateway's CLI moved to `multimind.cli` (the `multimind` entry point; `multimind.gateway.cli` is a thin alias). The relevant commands are:

```bash
# Chat with a model (-m is required)
multimind chat start -m openai
multimind chat start -m openai -p "Hello!"      # single prompt instead of an interactive session

# Manage sessions
multimind chat list-sessions
multimind chat save SESSION_ID
multimind chat load SESSION_ID
multimind chat delete SESSION_ID

# Compare models, check health, show metrics
multimind models list
multimind models compare "Explain quantum computing" -m openai -m claude
multimind models health [-m openai]
multimind models metrics [-m openai]

# Serve the HTTP API
multimind backend [--host 127.0.0.1] [--port 8080]   # combined MultiMind API with Swagger at /docs
multimind serve  [--port 8400] ...                   # OpenAI-compatible compliance proxy only
```

Full option lists: [cli.md](/cli).

## REST API

### Running the server

```bash
multimind backend                                          # recommended: all routes on http://127.0.0.1:8080, docs at /docs
uvicorn multimind.gateway.api:app --host 0.0.0.0 --port 8000   # gateway app only
python -m multimind.gateway.api                            # same, via multimind.gateway.start()
```

`multimind backend` honours `MULTIMIND_BACKEND_HOST` / `MULTIMIND_BACKEND_PORT` and, when `API_KEYS` is set (comma-separated), requires an `X-API-Key` header on every route except `/`, `/health`, `/ready` and the docs pages. The standalone gateway app has no authentication of its own.

Model-backed endpoints depend on at least one provider being configured (an API key, or `OLLAMA_API_BASE`); otherwise they return `503 "No models are properly configured"`. A model that is requested but not configured returns `400 "Model <name> is not available"`.

The generated spec is in [openapi-gateway.json](https://github.com/multimindlab/multimind-sdk/blob/develop/docs/api/openapi-gateway.json) (and [openapi-platform.json](https://github.com/multimindlab/multimind-sdk/blob/develop/docs/api/openapi-platform.json) for the combined app).

### Endpoints

#### System

```http
GET /          -> {"name": "MultiMind API Gateway", "version": "1.0.0", "models": [...provider names...]}
GET /health    -> {"status": "healthy", "version": "1.0.0"}
GET /ready     -> {"status": "ready", "version": "1.0.0"}
GET /v1/models -> {"models": {"openai": {"status": "available"|"unavailable", "config": {"model_name", "temperature", "max_tokens"}}, ...}}
```

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

Response (ModelResponse):
{
    "content": "Hi there!",
    "model": "gpt-3.5-turbo",
    "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    "finish_reason": "stop",
    "timestamp": "2026-10-12T10:00:00.000000"
}
```

`model` defaults to `DEFAULT_MODEL`. Message objects may also carry `model` and `metadata` fields; they are ignored by this endpoint.

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

Response: ModelResponse (same shape as /v1/chat)
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
    "responses": {
        "openai":    {"content": "...", "model": "...", "usage": {...}, "finish_reason": null, "timestamp": "..."},
        "anthropic": {"content": "...", "model": "...", "usage": {...}, "finish_reason": null, "timestamp": "..."}
    }
}
```

`models` defaults to `["openai", "anthropic", "ollama"]`; models that are not configured are skipped (logged, not errored).

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

Response:
{"session_id": "…uuid…", "model": "openai", "created_at": "...", "updated_at": "...", "message_count": 0}

# List sessions  -> array of the object above
GET /v1/sessions

# Get session (includes messages)
GET /v1/sessions/{session_id}
-> {"session_id", "model", "created_at", "updated_at",
    "messages": [{"role", "content", "model", "timestamp", "metadata"}, ...]}

# Add a user message and get the assistant reply (synchronous)
POST /v1/sessions/{session_id}/messages
Content-Type: application/json

{
    "role": "user",
    "content": "Hello!",
    "metadata": {"topic": "greeting"}
}

Response:
{"status": "ok", "role": "assistant", "content": "Hi! How can I help?", "model": "openai"}

# Save session to disk (same as `multimind chat save <id>`)
POST /v1/sessions/{session_id}/save
-> {"status": "session saved", "session_id": "…", "path": "chat_sessions/chat_<id>.json"}

# Load a saved session into memory (same as `multimind chat load <id>`)
POST /v1/sessions/load
Content-Type: application/json

{"session_id": "…uuid…"}
-> {"session_id", "model", "created_at", "updated_at", "message_count"}

# Delete session
DELETE /v1/sessions/{session_id}
-> {"status": "session deleted"}
```

`POST /v1/sessions/{id}/messages` appends the message, sends the whole session history to the session's model, stores the assistant reply (with `usage` in its metadata) and returns it in the response. If the provider call fails the endpoint returns `502 "Model request failed"`. Unknown session IDs return `404 "Session not found"`. Sessions live in memory until saved: `POST /v1/sessions/{id}/save` writes `chat_<id>.json` to the `ChatManager` storage directory (`chat_sessions/` by default) and `POST /v1/sessions/load` reads it back (returning the in-memory session if it is already active). Both return `404` for an unknown session ID, matching `multimind chat save` / `multimind chat load`.

#### Metrics and Health

```http
# Get metrics
GET /v1/metrics
GET /v1/metrics?model=openai
-> {"metrics": {...ModelMetrics fields per model...}, "health": {"openai": ModelHealth, ...}}

# Check health (sends a test request to each configured provider, or to one model)
POST /v1/health/check
POST /v1/health/check?model=anthropic
-> {"anthropic": {"is_healthy": true, "last_check": "...", "error_message": null, "latency_ms": 420.5, "uptime_percentage": 100.0}}
```

#### Compliance

`multimind.gateway.compliance_api` mounts a router at `/v1/compliance` (included in both the gateway app and the combined API):

```http
POST /v1/compliance/monitor                 # body: ComplianceConfig -> ComplianceResult (implemented checkers only)
POST /v1/compliance/report                  # body: ComplianceConfig -> report dict
GET  /v1/compliance/regulations             # -> ["GDPR", "HIPAA", ...]
GET  /v1/compliance/healthcare/use-cases    # -> [...]
GET  /v1/compliance/dashboard?organization_id=...&time_range=7d[&use_case=...]
POST /v1/compliance/alerts/configure?organization_id=...   # body: alert_rules dict
GET  /v1/compliance/alerts?organization_id=...&status=active[&severity=...]
```

Request/response models are in the OpenAPI spec. Unknown regulation names return `400`.

### Error Responses

Errors use FastAPI's standard shape:

```json
{
    "detail": "Human readable error message"
}
```

Status codes used by the gateway:

- `400`: requested model is not configured, or the request is malformed
- `404`: chat session not found
- `422`: request body failed validation (FastAPI's default; `detail` is a list of field errors)
- `502`: the model provider call failed while adding a session message
- `503`: no model provider is configured at all
- `500`: unexpected server error (details are logged, not returned)

## Examples

For complete working examples, see:
- [Gateway Examples](https://github.com/multimindlab/multimind-sdk/blob/develop/examples/api/gateway_examples.py)
- [Ollama chat CLI](https://github.com/multimindlab/multimind-sdk/blob/develop/examples/cli/chat_ollama_cli.py)
- [RAG Example](https://github.com/multimindlab/multimind-sdk/blob/develop/examples/api/rag_example.py)
- [Advanced RAG](https://github.com/multimindlab/multimind-sdk/blob/develop/examples/api/rag_advanced_example.py)
