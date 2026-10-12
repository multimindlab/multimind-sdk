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

## Pipeline Examples

The SDK provides a powerful pipeline system for building complex AI workflows. Here are examples of available pipelines:

### Available Pipelines

1. **QA Retrieval Pipeline**
   ```python
   from multimind import Router, PipelineBuilder
   
   router = Router()
   builder = PipelineBuilder(router)
   pipeline = builder.qa_retrieval()
   result = await pipeline.run("What is machine learning?")
   ```

2. **Code Review Pipeline**
   ```python
   pipeline = builder.code_review()
   result = await pipeline.run(code_snippet)
   ```

3. **Image Analysis Pipeline**
   ```python
   pipeline = builder.image_analysis()
   with open("image.jpg", "rb") as f:
       result = await pipeline.run(f.read())
   ```

4. **Text Summarization Pipeline**
   ```python
   pipeline = builder.text_summarization()
   result = await pipeline.run(long_text)
   ```

5. **Content Generation Pipeline**
   ```python
   pipeline = builder.content_generation()
   result = await pipeline.run("The Future of AI")
   ```

6. **Data Analysis Pipeline**
   ```python
   pipeline = builder.data_analysis()
   result = await pipeline.run(dataset)
   ```

7. **Multi-Modal QA Pipeline**
   ```python
   pipeline = builder.multi_modal_qa()
   result = await pipeline.run({
       "query": "What's in this image?",
       "image": image_data
   })
   ```

8. **Code Generation Pipeline**
   ```python
   pipeline = builder.code_generation()
   result = await pipeline.run(requirements)
   ```

9. **Sentiment Analysis Pipeline**
   ```python
   pipeline = builder.sentiment_analysis()
   result = await pipeline.run(text)
   ```

10. **Document Processing Pipeline**
    ```python
    pipeline = builder.document_processing()
    result = await pipeline.run(document)
    ```

11. **Translation Pipeline**
    ```python
    pipeline = builder.translation_pipeline()
    result = await pipeline.run({
        "text": text,
        "target_language": "Spanish"
    })
    ```

12. **Research Assistant Pipeline**
    ```python
    pipeline = builder.research_assistant()
    result = await pipeline.run(research_query)
    ```

### Pipeline Features

- **Error Handling**: Each pipeline includes built-in error handling and retry mechanisms
- **Customization**: Pipelines can be customized with different models and parameters
- **Async Support**: All pipelines support asynchronous execution
- **Context Management**: Pipelines maintain context between stages
- **Type Safety**: Full type hints and validation using Pydantic

For more detailed examples, see `pipeline_examples.py` in the api directory.

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