# MultiMind SDK Examples

This directory contains example scripts demonstrating various features of the MultiMind SDK, organized into CLI and API examples.

## Directory Structure

```
examples/
├── cli/                    # Command-line interface examples
│   ├── chat_ollama_cli.py  # Interactive chat with Ollama models
│   ├── chat_with_gpt.py    # Basic chat with GPT
│   ├── basic_agent.py      # Simple agent implementation
│   ├── prompt_chain.py     # Chain of prompts example
│   ├── task_runner.py      # Task execution with agents
│   ├── usage_tracking.py   # Track model usage and costs
│   ├── mcp_workflow.py     # Multi-agent collaboration
│   └── multi_model_wrapper_cli.py  # CLI for multi-model wrapper
├── api/                    # API and integration examples
│   ├── gateway_examples.py # Gateway API usage examples
│   ├── rag_example.py      # Basic RAG implementation
│   ├── rag_advanced_example.py  # Advanced RAG with custom configs
│   ├── multi_model_wrapper_api.py  # API for multi-model wrapper
│   ├── model_wrapper.py    # Model wrapper implementation
│   └── test_llm_wrapper.py # Tests for LLM wrapper
└── streamlit-ui/          # Streamlit-based UI examples
```

## CLI Examples

### Ollama Chat Example

The `cli/chat_ollama_cli.py` script provides an interactive command-line interface for chatting with Ollama models.

### Prerequisites

1. Install Ollama:
   - Visit [Ollama's website](https://ollama.ai) and follow the installation instructions
   - Make sure the Ollama service is running

2. Install required Python packages:
   ```bash
   pip install pytest  # For running tests
   ```

### Usage

Basic usage:
```bash
python chat_ollama_cli.py
```

With specific model:
```bash
python chat_ollama_cli.py --model llama2
```

Save chat history:
```bash
python chat_ollama_cli.py --history chat_log.json
```

Disable streaming:
```bash
python chat_ollama_cli.py --no-stream
```

Enable debug logging:
```bash
python chat_ollama_cli.py --debug
```

### Special Commands

During the chat session, you can use these special commands:
- `exit` - Exit the chat
- `history` - Show chat history
- `models` - List available models
- `clear` - Clear chat history
- `pull <model_name>` - Pull a new model (e.g., `pull llama2`)

### Features

1. **Interactive Chat**
   - Real-time streaming responses
   - Support for all Ollama models
   - Easy model switching

2. **Chat History**
   - Save conversations to file
   - View previous messages
   - Clear history when needed

3. **Model Management**
   - List available models
   - Pull new models
   - Automatic model verification

4. **Error Handling**
   - Graceful error handling
   - Informative error messages
   - Debug logging option

### Testing

Run the test suite:
```bash
python test_ollama_chat.py
```

The tests verify:
- Basic initialization
- Chat history management
- Model availability
- Chat functionality
- Error handling

### Other CLI Examples

1. **Basic Agent** (`cli/basic_agent.py`)
   - Simple agent implementation
   - Demonstrates basic agent capabilities

2. **Prompt Chain** (`cli/prompt_chain.py`)
   - Chain of prompts example
   - Shows how to create complex prompt workflows

3. **Task Runner** (`cli/task_runner.py`)
   - Task execution with agents
   - Demonstrates task management and execution

4. **Usage Tracking** (`cli/usage_tracking.py`)
   - Track model usage and costs
   - Monitor API usage and expenses

5. **MCP Workflow** (`cli/mcp_workflow.py`)
   - Multi-agent collaboration
   - Complex workflow management

6. **Multi-Model Wrapper** (`cli/multi_model_wrapper_cli.py`)
   - CLI interface for multi-model wrapper
   - Demonstrates model composition and switching

## API Examples

### Gateway Examples

The `api/gateway_examples.py` demonstrates how to use the MultiMind Gateway API:

```python
from multimind.gateway import MultiMindCLI, chat_manager, monitor

# Create a chat session
session = chat_manager.create_session(
    model="openai",
    system_prompt="You are a helpful AI assistant."
)

# Add messages and get responses
session.add_message(role="user", content="Hello!")
response = await handler.generate("Hello!")
```

### RAG Examples

1. **Basic RAG** (`api/rag_example.py`)
   - Simple RAG implementation
   - Document processing and retrieval

2. **Advanced RAG** (`api/rag_advanced_example.py`)
   - Custom configurations
   - Advanced retrieval strategies

### Multi-Model Wrapper

The multi-model wrapper examples demonstrate how to combine different models:

1. **API Interface** (`api/multi_model_wrapper_api.py`)
   - API for model composition
   - Model switching and routing

2. **Model Wrapper** (`api/model_wrapper.py`)
   - Core wrapper implementation
   - Model abstraction and management

3. **Tests** (`api/test_llm_wrapper.py`)
   - Test suite for LLM wrapper
   - Integration tests

## Contributing

Feel free to contribute more examples! When adding new examples:
1. Follow the existing code style
2. Add appropriate documentation
3. Include tests if applicable
4. Update this README with usage instructions
5. Place files in the appropriate directory (cli/ or api/)

## Troubleshooting

Common issues and solutions:

1. **Ollama not found**
   - Ensure Ollama is installed and in your PATH
   - Check if the Ollama service is running

2. **Model not available**
   - Use the `pull` command to download the model
   - Check available models with the `models` command

3. **Chat history issues**
   - Ensure write permissions in the history file directory
   - Check file path validity

4. **Performance issues**
   - Try disabling streaming with `--no-stream`
   - Use a smaller model if available
   - Check system resources

For more help, check the [main documentation](https://github.com/multimind-dev/multimind-sdk/blob/main/docs/README.md) or open an issue. 