# RAG System Documentation

The MultiMind SDK's RAG (Retrieval Augmented Generation) system combines document processing, embeddings, and vector storage so you can retrieve relevant context and feed it to any MultiMind model. This document covers the system's components, configuration, and usage.

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

The RAG system lets you:

- Process and chunk documents with configurable sizes and overlap
- Embed documents and store the vectors in a vector store (FAISS and others)
- Perform semantic search over documents with optional metadata filtering
- Generate answers by passing retrieved context to any MultiMind model
- Integrate retrieval into the agent system as a custom tool

The core class is `RAG` (retrieval only — you compose it with a model for generation). It is configured with a single `RAGConfig` object that bundles the vector store, embedding, retrieval, and document-processing settings.

## Components

### 1. Documents

Documents are plain data objects:

```python
from multimind.document_processing.base import Document

doc = Document(
    id="doc_1",
    content="The MultiMind SDK is a powerful framework.",
    metadata={"category": "documentation"},
    source="example",
)
```

Chunking behavior is controlled by the `document_config` dict on `RAGConfig` (`min_chunk_size`, `max_chunk_size`, `chunk_overlap`).

### 2. Embeddings

Embedding behavior is configured with `EmbeddingConfig`. Supported `model_type` values include `"openai"` (e.g. `text-embedding-ada-002`, 1536 dimensions) and `"huggingface"` (e.g. `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, fully local):

```python
from multimind.embeddings.embedding import EmbeddingConfig

embedding_config = EmbeddingConfig(
    model_name="text-embedding-ada-002",
    model_type="openai",
    batch_size=32,
    max_length=512,
    normalize=True,
    device="cpu",
    cache_dir=None,
    custom_params={},
)
```

### 3. Vector Stores

Vector stores are configured with `VectorStoreConfig`. FAISS (in-memory, fast similarity search) has a convenience constructor; the dimension must match your embedding model:

```python
from multimind.vector_store import VectorStoreConfig

vector_store_config = VectorStoreConfig.create_faiss_config(
    dimension=1536,       # 384 for all-MiniLM-L6-v2
    metric="cosine",
    index_type="flat",
)
```

Other backends (Chroma, Qdrant, and more) are available via the `vector-stores` extra; see `multimind.vector_store` for the factory and per-backend configs.

## Installation

Install the RAG system with all dependencies:

```bash
pip install "multimind-sdk[rag]"
```

Required environment variables (only for the providers you use):

```bash
export OPENAI_API_KEY="your-openai-key"        # For OpenAI models/embeddings
export ANTHROPIC_API_KEY="your-anthropic-key"  # For Claude models
```

HuggingFace embeddings run locally and need no API key.

## Basic Usage

### 1. Initialize the RAG System

```python
import asyncio
from multimind import RAG, RAGConfig, OpenAIModel
from multimind.vector_store import VectorStoreConfig
from multimind.embeddings.embedding import EmbeddingConfig
from multimind.document_processing.base import Document

model = OpenAIModel(model_name="gpt-4o-mini")

config = RAGConfig(
    vector_store_config=VectorStoreConfig.create_faiss_config(
        dimension=1536, metric="cosine", index_type="flat"
    ),
    retrieval_config={"top_k": 3, "similarity_threshold": 0.5},
    embedding_config=EmbeddingConfig(
        model_name="text-embedding-ada-002",
        model_type="openai",
        batch_size=32,
        max_length=512,
        normalize=True,
        device="cpu",
        cache_dir=None,
        custom_params={},
    ),
    document_config={"min_chunk_size": 100, "max_chunk_size": 1000, "chunk_overlap": 200},
)

rag = RAG(config)
```

### 2. Add Documents

```python
async def add_docs():
    await rag.initialize()
    docs = [
        Document(id="doc_0", content="The MultiMind SDK is a powerful framework.",
                 metadata={"source": "direct"}, source="example"),
        Document(id="doc_1", content="RAG combines retrieval with generation.",
                 metadata={"source": "direct"}, source="example"),
    ]
    await rag.add_documents(docs, process=True)   # process=True applies chunking

asyncio.run(add_docs())
```

### 3. Retrieve and Generate

`RAG.retrieve` returns the most relevant documents; you build the prompt and generate with any model:

```python
async def answer(question: str) -> str:
    retrieved = await rag.retrieve(question, k=3)
    context = "\n\n".join(doc.content for doc in retrieved)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    return await model.generate(prompt, temperature=0.7)

print(asyncio.run(answer("What is the MultiMind SDK?")))
```

A complete runnable version of this flow (with similarity-threshold guard rails and a local no-API-key fallback) is in [examples/rag/example_rag.py](../examples/rag/example_rag.py).

## Advanced Features

### 1. Metadata Filtering

```python
results = await rag.retrieve(
    "deployment steps",
    k=5,
    filter_criteria={"category": "documentation"},
)
```

### 2. Batch Queries

```python
queries = ["query1", "query2", "query3"]
results = await asyncio.gather(*(rag.retrieve(q, k=3) for q in queries))
```

### 3. Model Switching

Retrieval is independent of the generating model, so you can pick a model per task:

```python
from multimind import ClaudeModel, OpenAIModel

# Use Claude for complex reasoning
claude = ClaudeModel(model_name="claude-3-5-sonnet-20241022")
complex_answer = await claude.generate(prompt, temperature=0.7)

# Use a smaller model for simpler tasks
mini = OpenAIModel(model_name="gpt-4o-mini")
simple_answer = await mini.generate(prompt, temperature=0.3)
```

For switching models mid-conversation with context transfer, see `ModelSession` in the [cookbook](cookbook.md#switch-models-mid-conversation-without-losing-knowledge).

### 4. Clearing the Store

```python
await rag.clear()   # remove all documents and vectors
```

## Agent Integration

Retrieval plugs into the agent system as a custom tool (subclass `BaseTool`):

```python
from multimind.agents import Agent, AgentMemory
from multimind.agents.tools import BaseTool, CalculatorTool

class RAGTool(BaseTool):
    def __init__(self, rag: RAG):
        super().__init__(
            name="rag_query",
            description="Retrieve relevant documents for a query",
        )
        self.rag = rag

    def get_parameters(self):
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 3},
            },
            "required": ["query"],
        }

    async def run(self, query: str, top_k: int = 3):
        docs = await self.rag.retrieve(query, k=top_k)
        return "\n\n".join(doc.content for doc in docs)

agent = Agent(
    model=model,
    memory=AgentMemory(max_history=100),
    tools=[RAGTool(rag), CalculatorTool()],
)

response = await agent.run("What are the main features of the MultiMind SDK?")
```

## Best Practices

1. **Document Processing**
   - Choose appropriate chunk sizes based on your use case
   - Use meaningful metadata for better filtering
   - Clean and normalize text before processing

2. **Embedding Models**
   - Use OpenAI embeddings for production applications
   - Use HuggingFace models for local or cost-sensitive use cases
   - Match the vector store `dimension` to your embedding model

3. **Vector Stores**
   - Use FAISS for high-performance, in-memory applications
   - Use a persistent backend (via the `vector-stores` extra) when data must survive restarts
   - Monitor memory usage with large document collections

4. **Retrieval Quality**
   - Set a `similarity_threshold` and refuse to answer when nothing relevant is retrieved
   - Consider grounding-checking generated answers — see [hallucination detection](cookbook.md#detect-hallucinations-in-rag-answers)

5. **Agent Integration**
   - Combine RAG with other tools for comprehensive solutions
   - Use memory to maintain context across interactions
   - Implement proper error handling and fallbacks

## API Reference

### RAG Class

```python
class RAG:
    def __init__(self, config: RAGConfig)

    async def initialize(self) -> None

    async def add_documents(
        self,
        documents: List[Document],
        process: bool = True,
    ) -> None

    async def retrieve(
        self,
        query: str,
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None,
    ) -> List[Document]

    async def clear(self) -> None
```

### RAGConfig

```python
class RAGConfig:
    def __init__(
        self,
        vector_store_config: VectorStoreConfig,
        retrieval_config: Dict[str, Any],     # e.g. {"top_k": 3, "similarity_threshold": 0.5}
        embedding_config: EmbeddingConfig,
        document_config: Dict[str, Any],      # e.g. {"min_chunk_size": 100, "max_chunk_size": 1000, "chunk_overlap": 200}
        custom_params: Dict[str, Any] = None,
    )
```

For the HTTP RAG service, see [rag_api.md](api_reference/rag_api.md) and the generated [openapi-rag.json](api/openapi-rag.json).
