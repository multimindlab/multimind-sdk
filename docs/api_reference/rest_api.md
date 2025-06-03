# MultiMind REST API

The MultiMind REST API provides a comprehensive interface for using the MultiMind Ensemble system. This API allows you to perform various AI tasks using multiple providers and ensemble methods.

## Installation

Make sure you have the MultiMind package installed:

```bash
pip install multimind
```

## Starting the API Server

To start the API server:

```bash
python -m examples.api.ensemble_api
```

The server will start on `http://localhost:8000`. You can access the API documentation at `http://localhost:8000/docs`.

## Available Endpoints

### Text Generation

Generate text using an ensemble of models:

```http
POST /generate
```

Request body:
```json
{
    "prompt": "Your prompt here",
    "providers": ["openai", "anthropic", "ollama"],
    "method": "weighted_voting",
    "weights": {
        "openai": 0.4,
        "anthropic": 0.4,
        "ollama": 0.2
    }
}
```

Example using Python requests:
```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "Explain quantum computing",
        "providers": ["openai", "anthropic", "ollama"],
        "method": "weighted_voting"
    }
)
print(response.json())
```

### Code Review

Review code using an ensemble of models:

```http
POST /review
```

Request body:
```json
{
    "code": "Your code here",
    "providers": ["openai", "anthropic", "ollama"]
}
```

Example using Python requests:
```python
import requests

response = requests.post(
    "http://localhost:8000/review",
    json={
        "code": """
        def calculate_factorial(n):
            if n < 0:
                return None
            result = 1
            for i in range(1, n + 1):
                result *= i
            return result
        """,
        "providers": ["openai", "anthropic", "ollama"]
    }
)
print(response.json())
```

### Image Analysis

Analyze images using an ensemble of models:

```http
POST /analyze-image
```

Request:
- Form data:
  - `image`: Image file
- Query parameters:
  - `providers`: List of providers (default: ["openai", "anthropic"])

Example using Python requests:
```python
import requests

with open("image.jpg", "rb") as f:
    files = {"image": f}
    response = requests.post(
        "http://localhost:8000/analyze-image",
        files=files,
        params={"providers": ["openai", "anthropic"]}
    )
print(response.json())
```

### Embedding Generation

Generate embeddings using an ensemble of models:

```http
POST /embed
```

Request body:
```json
{
    "text": "Your text here",
    "providers": ["openai", "huggingface"],
    "weights": {
        "openai": 0.6,
        "huggingface": 0.4
    }
}
```

Example using Python requests:
```python
import requests

response = requests.post(
    "http://localhost:8000/embed",
    json={
        "text": "This is a sample text",
        "providers": ["openai", "huggingface"]
    }
)
print(response.json())
```

## Response Format

All endpoints return JSON responses with the following structure:

```json
{
    "result": "Generated text or analysis",
    "confidence": 0.95,
    "explanation": "Explanation of the ensemble decision",
    "provider_votes": {
        "provider1": "result1",
        "provider2": "result2",
        ...
    }
}
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- `200 OK`: Successful request
- `400 Bad Request`: Invalid request parameters
- `500 Internal Server Error`: Server-side error

Error response format:
```json
{
    "detail": "Error message"
}
```

## Available Ensemble Methods

- `weighted_voting`: Combines results based on provider weights
- `confidence_cascade`: Uses results based on confidence thresholds
- `parallel_voting`: Combines results from all providers in parallel
- `majority_voting`: Uses the most common result among providers
- `rank_based`: Selects results based on provider ranking

## Environment Variables

The API uses the following environment variables (should be set in your `.env` file):

- `OPENAI_API_KEY`: Your OpenAI API key
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `OLLAMA_API_KEY`: Your Ollama API key (if using Ollama)
- `HUGGINGFACE_API_KEY`: Your Hugging Face API key (if using Hugging Face)

## Examples

See the `examples/ensemble/usage_examples.py` file for comprehensive examples of using the API interface.

## API Documentation

The API provides interactive documentation using Swagger UI. You can access it at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

These documentation interfaces allow you to:
- View all available endpoints
- Test the API directly from the browser
- View request/response schemas
- See example requests and responses 