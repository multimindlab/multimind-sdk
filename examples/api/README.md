# Multi-Model LLM Wrapper

A unified wrapper for multiple LLM providers with both CLI and API interfaces.

## Features

- Support for multiple LLM providers:
  - OpenAI (GPT-4)
  - Anthropic (Claude-3)
  - Ollama (Mistral and others)
  - Hugging Face models
- Two interfaces:
  - Command-line interface (CLI)
  - REST API (FastAPI)
- Unified query interface across all providers
- Environment-based configuration with `.env` support
- Auto-detection of available models
- Comprehensive logging
- Unit tests with pytest

## Installation

1. Install required packages:
```bash
pip install openai anthropic transformers fastapi uvicorn python-dotenv requests pytest
```

2. Install Ollama (optional, for local models):
   - Follow instructions at: https://ollama.ai/

3. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your API keys:
```bash
# .env file
OPENAI_API_KEY=sk-xxxx
CLAUDE_API_KEY=claude-api-key-here
HF_TOKEN=hf_xxxx
```

## Usage

### CLI Interface

Basic usage:
```bash
python main.py cli --model openai --prompt "Explain transformers in AI"
```

With model-specific options:
```bash
# Using a specific Ollama model
python main.py cli --model ollama --ollama-model codellama --prompt "Write a Python function"

# Using a specific Hugging Face model
python main.py cli --model huggingface --hf-model-id "meta-llama/Llama-2-7b" --prompt "Explain quantum computing"
```

### API Interface

1. Start the API server:
```bash
python main.py api
```

2. Make requests:

List available models:
```bash
curl "http://localhost:8000/models"
```

Basic query:
```bash
curl "http://localhost:8000/query?prompt=Explain+transformers&model=openai"
```

With model-specific options:
```bash
curl "http://localhost:8000/query?prompt=Write+code&model=ollama&ollama_model=codellama"
```

### Running Tests

Run the test suite:
```bash
pytest test_llm_wrapper.py
```

## API Reference

### Endpoints

1. `GET /models`
   - Returns list of available models based on API keys and local installations
   - No parameters required

2. `GET /query`
   - Parameters:
     - `prompt` (required): The text prompt to send to the model
     - `model` (required): One of the available models (check /models endpoint)
     - `ollama_model` (optional): Specific Ollama model to use (default: "mistral")
     - `hf_model_id` (optional): Specific Hugging Face model ID (default: "mistralai/Mistral-7B-v0.1")

Response format:
```json
{
  "status": "success",
  "model": "model_name",
  "response": "model_response"
}
```

Error response:
```json
{
  "status": "error",
  "model": "model_name",
  "error": "error_message"
}
```

## Architecture

The wrapper is organized into several components:

1. `model_wrapper.py`: Core class implementing model interactions
   - Environment variable handling with python-dotenv
   - Available model detection
   - Unified query interface
   - Logging support

2. `cli.py`: Command-line interface
   - Dynamic model choices based on availability
   - Argument parsing
   - Error handling

3. `api.py`: FastAPI-based REST API
   - Model availability endpoint
   - Query endpoint with validation
   - Error handling with HTTP status codes

4. `main.py`: Entry point
   - CLI and API mode support
   - Unified command interface

5. `test_llm_wrapper.py`: Unit tests
   - Model availability testing
   - Query functionality testing
   - Error handling testing

## Logging

The wrapper includes comprehensive logging:
- INFO level logging by default
- Model availability detection
- Query execution tracking
- Error reporting

Logs include:
- Timestamp
- Log level
- Component name
- Message

## Error Handling

- Missing API keys are detected during initialization
- Model-specific errors are caught and reported
- API errors return appropriate HTTP status codes
- CLI errors are displayed with descriptive messages
- Logging of all errors for debugging

## Contributing

Feel free to submit issues and enhancement requests! 