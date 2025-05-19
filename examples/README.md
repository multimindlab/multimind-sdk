# MultiMind SDK Examples

This directory contains example scripts demonstrating various features of the MultiMind SDK.

## Ollama Chat Example

The `chat_ollama_cli.py` script provides an interactive command-line interface for chatting with Ollama models.

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

## Other Examples

### RAG Examples
- `rag_example.py` - Basic RAG implementation
- `rag_advanced_example.py` - Advanced RAG with custom configurations

### Agent Examples
- `basic_agent.py` - Simple agent implementation
- `task_runner.py` - Task execution with agents
- `mcp_workflow.py` - Multi-agent collaboration

### Prompt Engineering
- `prompt_chain.py` - Chain of prompts example
- `chat_with_gpt.py` - Basic chat with GPT

### Usage Tracking
- `usage_tracking.py` - Track model usage and costs

## Contributing

Feel free to contribute more examples! When adding new examples:
1. Follow the existing code style
2. Add appropriate documentation
3. Include tests if applicable
4. Update this README with usage instructions

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