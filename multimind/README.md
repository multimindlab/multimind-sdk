# MultiMind Gateway

A unified gateway for interacting with multiple AI models through a single interface.

## 📚 Documentation

- [Quick Start Guide](../../docs/quickstart.md) - Get started with MultiMind Gateway
- [Configuration Guide](../../docs/configuration.md) - Detailed configuration options
- [API Reference](../../docs/api_reference/) - Complete API documentation
- [Architecture](../../docs/architecture.md) - System design and components
- [Features](../../docs/features.md) - Detailed feature documentation
- [Development Guide](../../docs/development.md) - Contributing and development

## 🚀 Features

- **Unified Interface**: Interact with multiple AI models (OpenAI, Anthropic, Ollama, Groq, HuggingFace) through a single API
- **Command Line Interface**: Easy-to-use CLI for model interactions
- **Model Monitoring**: Track usage, health, and performance metrics
- **Chat Session Management**: Persistent chat sessions with history
- **Rate Limiting**: Control request rates and token usage
- **Extensible**: Easy to add support for new models

## 📦 Installation

```bash
# Basic installation
pip install multimind-sdk

# With gateway support
pip install multimind-sdk[gateway]

# Full installation with all features
pip install multimind-sdk[full]

# Development installation
pip install multimind-sdk[dev]
```

## ⚙️ Configuration

Create a `.env` file in your project root with your API keys:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_key_here
OPENAI_MODEL_NAME=gpt-3.5-turbo

# Anthropic Configuration
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL_NAME=claude-3-opus-20240229

# Ollama Configuration
OLLAMA_API_BASE=http://localhost:11434
OLLAMA_MODEL_NAME=mistral

# Groq Configuration
GROQ_API_KEY=your_key_here
GROQ_MODEL_NAME=mixtral-8x7b-32768

# HuggingFace Configuration
HUGGINGFACE_API_KEY=your_key_here
HUGGINGFACE_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2

# General Settings
DEFAULT_MODEL=openai
LOG_LEVEL=INFO
```

For more configuration options, see the [Configuration Guide](../../docs/configuration.md).

## 💻 Usage

### Command Line Interface

The MultiMind CLI provides easy access to all features:

```bash
# Chat with a model
multimind chat --model openai
multimind chat --model anthropic --prompt "Hello, world!"

# Compare models
multimind compare "What is AI?" --models openai anthropic ollama

# Check model status
multimind status

# Monitor metrics
multimind metrics
multimind metrics --model openai

# Check model health
multimind health
multimind health --model anthropic

# Manage chat sessions
multimind sessions
multimind load <session_id>
multimind save <session_id>
multimind delete <session_id>
```

For more CLI examples, see [Chat Examples](../../examples/chat_ollama_cli.py).

### Python API

```python
from multimind.gateway import get_model_handler, chat_manager, monitor

# Chat with a model
async def chat_example():
    handler = get_model_handler("openai")
    response = await handler.chat([
        {"role": "user", "content": "Hello, world!"}
    ])
    print(response.content)

# Create and manage chat sessions
async def session_example():
    # Create a new session
    session = chat_manager.create_session(
        model="openai",
        system_prompt="You are a helpful assistant."
    )
    
    # Add messages
    session.add_message(
        role="user",
        content="What is the capital of France?",
        model="openai"
    )
    
    # Save session
    chat_manager.save_session(session.session_id)

# Monitor model health and metrics
async def monitoring_example():
    # Check model health
    handler = get_model_handler("openai")
    health = await monitor.check_health("openai", handler)
    print(f"Model health: {health.is_healthy}")
    
    # Get metrics
    metrics = await monitor.get_metrics()
    print(f"Total requests: {metrics['openai']['metrics'].total_requests}")
```

For more API examples, see [Gateway Examples](../../examples/gateway_examples.py).

### API Gateway

The MultiMind Gateway provides a REST API for all features:

```python
import requests

# Chat with a model
response = requests.post("http://localhost:8000/v1/chat", json={
    "messages": [{"role": "user", "content": "Hello, world!"}],
    "model": "openai"
})

# Create a chat session
response = requests.post("http://localhost:8000/v1/sessions", json={
    "model": "openai",
    "system_prompt": "You are a helpful assistant."
})

# Get metrics
response = requests.get("http://localhost:8000/v1/metrics")

# Check model health
response = requests.post("http://localhost:8000/v1/health/check")
```

For complete API documentation, see the [API Reference](../../docs/api_reference/).

## 📖 Examples

Explore our [examples directory](../../examples/) for:

- [Gateway Examples](../../examples/gateway_examples.py) - Demonstrates chat sessions, monitoring, and rate limiting
- [Chat Examples](../../examples/chat_ollama_cli.py) - Shows different ways to use the chat interface
- [RAG Examples](../../examples/rag_example.py) - Basic RAG implementation
- [Advanced RAG](../../examples/rag_advanced_example.py) - Advanced RAG features
- [Usage Tracking](../../examples/usage_tracking.py) - Monitor model usage and costs
- [Task Runner](../../examples/task_runner.py) - Run complex AI tasks
- [Prompt Chain](../../examples/prompt_chain.py) - Chain multiple prompts together
- [Basic Agent](../../examples/basic_agent.py) - Create AI agents

## 🔍 Features in Detail

### Model Monitoring

The monitoring system tracks:
- Request counts and success rates
- Response times and latency
- Token usage and costs
- Model health status
- Rate limiting

```python
# Get metrics for all models
metrics = await monitor.get_metrics()

# Check model health
health = await monitor.check_health("openai", handler)

# Set rate limits
monitor.set_rate_limits(
    model="openai",
    requests_per_minute=10,
    tokens_per_minute=1000
)
```

For more details, see [Features Documentation](../../docs/features.md).

### Chat Sessions

Chat sessions provide:
- Persistent conversation history
- System prompt templates
- Context window management
- Session export/import
- Metadata support

```python
# Create a session
session = chat_manager.create_session(
    model="openai",
    system_prompt="You are a helpful assistant.",
    metadata={"purpose": "customer_support"}
)

# Add messages
session.add_message(
    role="user",
    content="Hello!",
    model="openai",
    metadata={"topic": "greeting"}
)

# Export session
export_data = session.export(format="json")

# Save session
chat_manager.save_session(session.session_id)
```

For more details, see [Features Documentation](../../docs/features.md).

## 🤝 Contributing

We welcome contributions! Please see our [Development Guide](../../docs/development.md) for details on:
- Setting up your development environment
- Code style and standards
- Testing procedures
- Pull request process

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](../../LICENSE) file for details. 