---
title: "RAG System API Documentation"
---

This document provides detailed documentation for the MultiMind SDK's RAG system API endpoints.

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Client Library](#client-library)
4. [Endpoints](#endpoints)
5. [Request/Response Models](#requestresponse-models)
6. [Examples](#examples)
7. [Error Handling](#error-handling)

## Overview

The RAG API (`multimind.gateway.rag_api`, FastAPI) provides RESTful endpoints for interacting with the MultiMind SDK's RAG system. It supports:

- Document management (add, query, delete)
- File uploads
- Semantic search
- Response generation
- Model management
- Health and readiness probes

Run it standalone with `python -m multimind.gateway.rag_api` (port 8000) or `uvicorn multimind.gateway.rag_api:app`; the same routes are also mounted in the combined MultiMind API served by `multimind backend` (port 8080). Requires `pip install "multimind-sdk[gateway,rag]"`. Step-by-step setup: [rag_api_server_setup.md](/api_reference/rag_api_server_setup). Generated spec: [openapi-rag.json](https://github.com/multimindlab/multimind-sdk/blob/develop/docs/api/openapi-rag.json).

The RAG backend (FAISS store, embedding model, text model) is initialized lazily on the first document/query request. If no provider can be set up the request fails with `503 "RAG backend is not available..."`.

## Authentication

The API supports two authentication methods, checked at request time:

1. **API Key Authentication**
   - Set the `X-API-Key` header with your API key
   - API keys are configured via the `API_KEYS` environment variable (comma-separated list)

2. **JWT Authentication**
   - Get a token using the `/token` endpoint (requires `JWT_SECRET` and `JWT_USERS_JSON`, plus `pip install "passlib[bcrypt]"` for password verification)
   - Include the token in the `Authorization: Bearer <token>` header
   - Tokens are HS256 and expire after 30 minutes

When neither `API_KEYS` nor `JWT_SECRET` is set the API runs in open (development) mode and accepts anonymous requests. `/health` and `/ready` never require authentication.

### Environment Variables

```bash
# API Keys (comma-separated)
export API_KEYS="key1,key2,key3"

# JWT (both required for /token)
export JWT_SECRET="long-random-secret"
export JWT_USERS_JSON='{"admin":"$2b$12$...bcrypt hash..."}'

# Model API Keys (optional; falls back to local HuggingFace models when both are unset)
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Optional
export MULTIMIND_CORS_ORIGINS="http://localhost:3000"
```

### Getting a Token

```http
POST /token
Content-Type: application/x-www-form-urlencoded

username=testuser&password=secret
```

Response:
```json
{
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "token_type": "bearer"
}
```

Errors: `503` when `JWT_SECRET` or `JWT_USERS_JSON` is not configured, `401` for a wrong username/password.

### Scopes

Issued tokens carry `"scopes": ["rag:read", "rag:write"]`, but the endpoints do not currently check scopes: any authenticated caller (API key or valid JWT) can use every endpoint.

## Client Library

The MultiMind SDK includes a client library for easy interaction with the RAG API.

### Installation

```bash
pip install multimind-sdk
```

### Basic Usage

```python
from multimind.client.rag_client import RAGClient, Document   # RAGClient is also exported from multimind
import asyncio

async def main():
    # Initialize client with API key
    client = RAGClient(
        base_url="http://localhost:8000",
        api_key="your-api-key"      # or token="<jwt>"; omit both in development mode
    )
    
    # Or use JWT authentication
    # client = RAGClient(base_url="http://localhost:8000")
    # token = await client.login("username", "password")   # stores the bearer token on the client
    
    # Add documents
    docs = [
        Document(
            text="The RAG system provides powerful document processing.",
            metadata={"type": "introduction"}
        )
    ]
    await client.add_documents(docs)
    
    # Query
    results = await client.query("What is the RAG system?")
    print("Query results:", results)
    
    # Generate
    response = await client.generate(
        "Explain the RAG system",
        temperature=0.7
    )
    print("Generated response:", response)

# Run example
asyncio.run(main())
```

### Client Methods

The `RAGClient` class provides the following methods:

- `login(username: str, password: str) -> str`: Get JWT token
- `add_documents(documents: List[Document]) -> Dict`: Add documents
- `add_file(file_path: Union[str, Path], metadata: Optional[Dict] = None) -> Dict`: Add file
- `query(query: str, top_k: Optional[int] = 3, filter_metadata: Optional[Dict] = None) -> Dict`: Query documents
- `generate(query: str, top_k: Optional[int] = 3, temperature: Optional[float] = 0.7, max_tokens: Optional[int] = None, filter_metadata: Optional[Dict] = None) -> Dict`: Generate response
- `clear_documents() -> Dict`: Clear all documents
- `get_document_count() -> int`: Get document count
- `switch_model(model_type: str, model_name: str) -> Dict`: Switch model
- `health_check() -> Dict`: Check system health

## Endpoints

### Document Management

#### Add Documents
```http
POST /documents
```

Add one or more documents to the RAG system.

**Request Body:**
```json
{
    "documents": [
        {
            "text": "Document text content",
            "metadata": {
                "source": "example",
                "type": "documentation"
            }
        }
    ]
}
```

**Response:**
```json
{
    "documents": [
        {
            "text": "Document text content",
            "metadata": {
                "source": "example",
                "type": "documentation"
            }
        }
    ],
    "total": 1
}
```

#### Add File
```http
POST /files
```

Add a file to the RAG system.

**Form Data:**
- `file`: File to upload (required)
- `metadata`: JSON string of metadata (optional)

**Response** (one entry per document the loader produced from the file; `filename` is always added to the metadata):
```json
{
    "documents": [
        {
            "text": "Added file: example.md",
            "metadata": {
                "source": "file",
                "filename": "example.md"
            }
        }
    ],
    "total": 1
}
```

#### Query Documents
```http
POST /query
```

Query the RAG system for relevant documents.

**Request Body:**
```json
{
    "query": "What is the RAG system?",
    "top_k": 3,
    "filter_metadata": {
        "type": "documentation"
    }
}
```

**Response:**
```json
{
    "documents": [
        {
            "text": "The RAG system provides...",
            "metadata": {
                "type": "documentation"
            },
            "score": 0.85
        }
    ],
    "total": 1
}
```

#### Generate Response
```http
POST /generate
```

Generate a response using the RAG system.

**Request Body:**
```json
{
    "query": "Explain the RAG system",
    "top_k": 3,
    "temperature": 0.7,
    "max_tokens": 500,
    "filter_metadata": {
        "type": "documentation"
    }
}
```

**Response:**
```json
{
    "text": "The RAG (Retrieval Augmented Generation) system...",
    "documents": [
        {
            "text": "The RAG system provides...",
            "metadata": {
                "type": "documentation"
            },
            "score": 0.85
        }
    ]
}
```

#### Clear Documents
```http
DELETE /documents
```

Clear all documents from the RAG system.

**Response:**
```json
{
    "message": "All documents cleared successfully"
}
```

#### Get Document Count
```http
GET /documents/count
```

Get the number of documents in the RAG system.

**Response:**
```json
{
    "count": 42
}
```

### Model Management

#### Switch Model
```http
POST /models/switch
```

Switch the model used by the RAG system.

**Form Data:**
- `model_type`: "openai" or "anthropic" (the matching `*_API_KEY` must be set, otherwise `400`)
- `model_name`: Model name (e.g., "gpt-3.5-turbo", "claude-3-sonnet-20240229")

Only the text-generation model is switched; the embedding model and vector store are unchanged.

**Response:**
```json
{
    "message": "Switched to openai model: gpt-3.5-turbo"
}
```

### Health Check

#### Liveness
```http
GET /health
```

Does not require authentication and does not initialize the RAG backend.

**Response:**
```json
{
    "status": "healthy",
    "version": "1.0.0",
    "rag_initialized": true
}
```

#### Readiness
```http
GET /ready
```

Returns `503 "RAG backend not initialized"` until the first document/query request has initialized the backend.

**Response:**
```json
{
    "status": "ready",
    "version": "1.0.0",
    "document_count": 42
}
```

## Request/Response Models

### DocumentRequest
```python
class DocumentRequest(BaseModel):
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentsRequest(BaseModel):          # body of POST /documents
    documents: List[DocumentRequest]
```

### QueryRequest
```python
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    filter_metadata: Optional[Dict[str, Any]] = None
```

### GenerateRequest
```python
class GenerateRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    filter_metadata: Optional[Dict[str, Any]] = None
```

### DocumentResponse
```python
class DocumentResponse(BaseModel):
    text: str
    metadata: Dict[str, Any]
    score: Optional[float] = None
```

### QueryResponse
```python
class QueryResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int
```

### GenerateResponse
```python
class GenerateResponse(BaseModel):
    text: str
    documents: List[DocumentResponse]
```

## Examples

### Python Client Example

```python
import aiohttp
import json

async def rag_api_example():
    async with aiohttp.ClientSession() as session:
        # Add documents
        documents = [
            {
                "text": "The RAG system provides powerful document processing.",
                "metadata": {"type": "introduction"}
            }
        ]
        async with session.post(
            "http://localhost:8000/documents",
            json={"documents": documents}
        ) as response:
            result = await response.json()
            print("Added documents:", result)
            
        # Query documents
        query = {
            "query": "What is the RAG system?",
            "top_k": 3
        }
        async with session.post(
            "http://localhost:8000/query",
            json=query
        ) as response:
            result = await response.json()
            print("Query results:", result)
            
        # Generate response
        generate = {
            "query": "Explain the RAG system",
            "temperature": 0.7
        }
        async with session.post(
            "http://localhost:8000/generate",
            json=generate
        ) as response:
            result = await response.json()
            print("Generated response:", result)

# Run example
import asyncio
asyncio.run(rag_api_example())
```

### cURL Examples

1. Add Documents:
```bash
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "text": "The RAG system provides powerful document processing.",
        "metadata": {"type": "introduction"}
      }
    ]
  }'
```

2. Query Documents:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the RAG system?",
    "top_k": 3
  }'
```

3. Generate Response:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain the RAG system",
    "temperature": 0.7
  }'
```

## Error Handling

The API uses standard HTTP status codes and returns error details in the response body:

```json
{
    "detail": "Error message describing what went wrong"
}
```

Common error scenarios:

1. **400 Bad Request**
   - Unsupported `model_type`, or the matching provider key is not set

2. **401 Unauthorized**
   - Missing or invalid `X-API-Key` / bearer token when authentication is configured
   - Expired token, wrong `/token` credentials

3. **404 Not Found**
   - Endpoint not found

4. **422 Unprocessable Entity**
   - Invalid request body or missing required fields (FastAPI validation)

5. **503 Service Unavailable**
   - RAG backend could not be initialized (no provider keys and no local HuggingFace stack)
   - `/token` called without `JWT_SECRET` / `JWT_USERS_JSON`
   - `/ready` before initialization

6. **500 Internal Server Error**
   - Model API errors, processing errors (details are logged, a generic message is returned)

Example error response:
```json
{
    "detail": "Failed to process document: Invalid format"
}
```

## Rate Limiting

Currently, the API does not implement rate limiting. However, it's recommended to:

1. Implement appropriate rate limiting in production
2. Monitor API usage
3. Set up proper authentication
4. Use appropriate timeouts for long-running operations

## Best Practices

1. **Document Management**
   - Use meaningful metadata
   - Clean documents before adding
   - Monitor document count

2. **Querying**
   - Use appropriate top_k values
   - Leverage metadata filtering
   - Handle large result sets

3. **Generation**
   - Adjust temperature based on use case
   - Set appropriate max_tokens
   - Monitor token usage

4. **Error Handling**
   - Implement proper error handling
   - Use appropriate timeouts
   - Handle rate limits

5. **Security**
   - Set up authentication
   - Use HTTPS in production
   - Validate input data 