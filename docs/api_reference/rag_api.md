# RAG System API Documentation

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

The RAG API provides RESTful endpoints for interacting with the MultiMind SDK's RAG system. It supports:

- Document management (add, query, delete)
- File uploads
- Semantic search
- Response generation
- Model management
- Health monitoring

## Authentication

The API supports two authentication methods:

1. **API Key Authentication**
   - Set the `X-API-Key` header with your API key
   - API keys are configured via the `API_KEYS` environment variable (comma-separated list)

2. **JWT Authentication**
   - Get a token using the `/token` endpoint
   - Include the token in the `Authorization: Bearer <token>` header
   - Tokens expire after 30 minutes

### Environment Variables

```bash
# API Keys (comma-separated)
export API_KEYS="key1,key2,key3"

# JWT Secret (change in production)
export JWT_SECRET="your-secret-key"

# Model API Keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
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

### Required Permissions

The API uses scopes to control access:

- `rag:read`: Required for querying and generating
- `rag:write`: Required for adding documents and managing models

## Client Library

The MultiMind SDK includes a client library for easy interaction with the RAG API.

### Installation

```bash
pip install multimind-sdk
```

### Basic Usage

```python
from multimind.client.rag_client import RAGClient, Document
import asyncio

async def main():
    # Initialize client with API key
    client = RAGClient(
        base_url="http://localhost:8000",
        api_key="your-api-key"
    )
    
    # Or use JWT authentication
    # client = RAGClient(base_url="http://localhost:8000")
    # token = await client.login("username", "password")
    
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

**Response:**
```json
{
    "documents": [
        {
            "text": "Added file: example.md",
            "metadata": {
                "source": "file",
                "type": "markdown"
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
- `model_type`: "openai" or "anthropic"
- `model_name`: Model name (e.g., "gpt-3.5-turbo", "claude-3-sonnet-20240229")

**Response:**
```json
{
    "message": "Switched to openai model: gpt-3.5-turbo"
}
```

### Health Check

#### Check Health
```http
GET /health
```

Check the health of the RAG system.

**Response:**
```json
{
    "status": "healthy",
    "document_count": 42
}
```

## Request/Response Models

### DocumentRequest
```python
class DocumentRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
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
   - Invalid request body
   - Missing required fields
   - Invalid model type

2. **404 Not Found**
   - Endpoint not found
   - Document not found

3. **500 Internal Server Error**
   - Model API errors
   - Processing errors
   - System errors

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