# RAG System Documentation

The MultiMind SDK's RAG (Retrieval Augmented Generation) system provides a powerful framework for building knowledge-based AI applications. This document covers the system's architecture, components, and usage.

## Table of Contents

1. [Overview](#overview)
2. [Components](#components)
3. [Installation](#installation)
4. [Basic Usage](#basic-usage)
5. [Advanced Features](#advanced-features)
6. [Agent Integration](#agent-integration)
7. [Best Practices](#best-practices)
8. [API Reference](#api-reference)

## Overview

The RAG system combines document processing, vector storage, and language models to create a powerful knowledge retrieval and generation system. It allows you to:

- Process and index documents of various types
- Store document embeddings in vector databases
- Perform semantic search over documents
- Generate responses based on retrieved context
- Integrate with the agent system for advanced applications

## Components

### 1. Document Processing

The `DocumentProcessor` class handles:
- Text chunking with configurable sizes and overlap
- Document cleaning and normalization
- Metadata management
- Support for multiple file types (txt, md, pdf)

```python
from multimind.rag import DocumentProcessor

processor = DocumentProcessor(
    chunk_size=1000,
    chunk_overlap=200,
    tokenizer="cl100k_base"
)
```

### 2. Embedding Models

Three embedding model implementations are available:

1. **OpenAI Embedder**
   ```python
   from multimind.rag.embeddings import get_embedder
   
   embedder = get_embedder(
       "openai",
       model="text-embedding-ada-002"
   )
   ```

2. **HuggingFace Embedder**
   ```python
   embedder = get_embedder(
       "huggingface",
       model_name="sentence-transformers/all-MiniLM-L6-v2"
   )
   ```

3. **Sentence-T5 Embedder**
   ```python
   embedder = get_embedder(
       "sentence-t5",
       model_name="sentence-transformers/sentence-t5-base"
   )
   ```

### 3. Vector Stores

Two vector store implementations are supported:

1. **FAISS Vector Store**
   - Fast similarity search
   - L2 distance metric
   - In-memory storage

2. **Chroma Vector Store**
   - Persistent storage
   - Cosine similarity
   - Metadata filtering

## Installation

Install the RAG system with all dependencies:

```bash
pip install multimind-sdk[rag]
```

Required environment variables:
```bash
export OPENAI_API_KEY="your-openai-key"  # For OpenAI models
export ANTHROPIC_API_KEY="your-anthropic-key"  # For Claude models
```

## Basic Usage

### 1. Initialize the RAG System

```python
from multimind.rag import RAG
from multimind.rag.embeddings import get_embedder
from multimind.models import OpenAIModel

# Initialize components
model = OpenAIModel(model="gpt-3.5-turbo")
embedder = get_embedder("openai")
rag = RAG(
    embedder=embedder,
    vector_store="faiss",
    model=model
)
```

### 2. Add Documents

```python
# Add text directly
await rag.add_documents(
    "The MultiMind SDK is a powerful framework.",
    metadata={"source": "direct"}
)

# Add a file
await rag.add_file(
    "docs/features.md",
    metadata={"category": "documentation"}
)
```

### 3. Query and Generate

```python
# Query documents
results = await rag.query("What is the MultiMind SDK?")

# Generate response
response = await rag.generate(
    "Explain the RAG system",
    temperature=0.7
)
```

## Advanced Features

### 1. Custom Document Processing

```python
from multimind.rag.document import DocumentProcessor

processor = DocumentProcessor(
    chunk_size=500,
    chunk_overlap=100,
    tokenizer="cl100k_base"
)

# Process documents with custom settings
docs = processor.process_document(
    long_text,
    metadata={"type": "custom"}
)
```

### 2. Batch Processing

```python
# Process multiple files
file_paths = ["doc1.txt", "doc2.txt", "doc3.txt"]
rag = await RAG.from_files(
    file_paths=file_paths,
    embedder=embedder,
    model=model
)

# Parallel queries
queries = ["query1", "query2", "query3"]
tasks = [rag.query(query) for query in queries]
results = await asyncio.gather(*tasks)
```

### 3. Model Switching

```python
from multimind.models import AnthropicModel

# Use Claude for complex reasoning
rag.model = AnthropicModel(model="claude-3-sonnet-20240229")
complex_response = await rag.generate(
    "Explain the architecture",
    temperature=0.7
)

# Use GPT-3.5 for simpler tasks
rag.model = OpenAIModel(model="gpt-3.5-turbo")
simple_response = await rag.generate(
    "List the features",
    temperature=0.3
)
```

## Agent Integration

The RAG system integrates with the agent system through custom tools:

### 1. RAG Query Tool

```python
from multimind.agents import Agent, AgentTool

class RAGTool(AgentTool):
    def __init__(self, rag: RAG):
        super().__init__(
            name="rag_query",
            description="Query the RAG system",
            parameters={
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 3}
            }
        )
        self.rag = rag
        
    async def execute(self, query: str, top_k: int = 3) -> str:
        results = await self.rag.query(query, top_k=top_k)
        return format_results(results)
```

### 2. RAG Generator Tool

```python
class RAGGeneratorTool(AgentTool):
    def __init__(self, rag: RAG):
        super().__init__(
            name="rag_generate",
            description="Generate responses using RAG",
            parameters={
                "query": {"type": "string"},
                "temperature": {"type": "number", "default": 0.7}
            }
        )
        self.rag = rag
        
    async def execute(self, query: str, temperature: float = 0.7) -> str:
        return await self.rag.generate(query, temperature=temperature)
```

### 3. Using RAG with Agents

```python
from multimind.agents import Agent, AgentMemory

# Create agent with RAG tools
agent = Agent(
    model=model,
    memory=AgentMemory(max_tokens=2000),
    tools=[
        RAGTool(rag),
        RAGGeneratorTool(rag),
        WebSearchTool(),
        CalculatorTool()
    ]
)

# Use agent to answer questions
response = await agent.run(
    "What are the main features of the MultiMind SDK?"
)
```

## Best Practices

1. **Document Processing**
   - Choose appropriate chunk sizes based on your use case
   - Use meaningful metadata for better filtering
   - Clean and normalize text before processing

2. **Embedding Models**
   - Use OpenAI embeddings for production applications
   - Consider HuggingFace models for cost-sensitive use cases
   - Match embedding dimensions with your vector store

3. **Vector Stores**
   - Use FAISS for high-performance, in-memory applications
   - Use Chroma for persistent storage and metadata filtering
   - Monitor memory usage with large document collections

4. **Model Selection**
   - Use Claude for complex reasoning tasks
   - Use GPT-3.5 for simpler, cost-effective tasks
   - Adjust temperature based on task requirements

5. **Agent Integration**
   - Combine RAG with other tools for comprehensive solutions
   - Use memory to maintain context across interactions
   - Implement proper error handling and fallbacks

## API Reference

### RAG Class

```python
class RAG:
    def __init__(
        self,
        embedder: Union[str, BaseLLM],
        vector_store: Optional[Union[str, BaseVectorStore]] = None,
        model: Optional[BaseModel] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 3,
        **kwargs
    )
    
    async def add_documents(
        self,
        documents: Union[str, Document, List[Union[str, Document]]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None
    
    async def add_file(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None
    
    async def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]
    
    async def generate(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> str
    
    async def clear(self) -> None
    
    async def get_document_count(self) -> int
    
    async def get_embedding_dimension(self) -> int
    
    @classmethod
    async def from_documents(
        cls,
        documents: List[Union[str, Document]],
        embedder: Union[str, BaseLLM],
        vector_store: Optional[Union[str, BaseVectorStore]] = None,
        model: Optional[BaseModel] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> "RAG"
    
    @classmethod
    async def from_files(
        cls,
        file_paths: List[Union[str, Path]],
        embedder: Union[str, BaseLLM],
        vector_store: Optional[Union[str, BaseVectorStore]] = None,
        model: Optional[BaseModel] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> "RAG"
```

For more detailed API documentation, see the [API Reference](../api_reference/rag.md). 