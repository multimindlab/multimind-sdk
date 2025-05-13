# RAG System API Reference

This document provides detailed API reference for the MultiMind SDK's RAG system components.

## Table of Contents

1. [RAG Class](#rag-class)
2. [Document Processing](#document-processing)
3. [Embedding Models](#embedding-models)
4. [Vector Stores](#vector-stores)
5. [Agent Tools](#agent-tools)

## RAG Class

The main RAG system class that coordinates document processing, embedding, and retrieval.

### Initialization

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
```

**Parameters:**
- `embedder`: Embedder type or instance
  - String: "openai", "huggingface", or "sentence-t5"
  - Instance: Any class implementing `BaseLLM`
- `vector_store`: Vector store type or instance
  - String: "faiss" or "chroma"
  - Instance: Any class implementing `BaseVectorStore`
  - Default: FAISSVectorStore
- `model`: LLM for generating responses
  - Optional: Any class implementing `BaseModel`
- `chunk_size`: Maximum size of text chunks in tokens
  - Default: 1000
- `chunk_overlap`: Number of tokens to overlap between chunks
  - Default: 200
- `top_k`: Number of top documents to retrieve
  - Default: 3
- `**kwargs`: Additional arguments for components

### Methods

#### add_documents

```python
async def add_documents(
    self,
    documents: Union[str, Document, List[Union[str, Document]]],
    metadata: Optional[Dict[str, Any]] = None
) -> None
```

Add documents to the RAG system.

**Parameters:**
- `documents`: Document(s) to add
  - String: Raw text
  - Document: Document object
  - List: Multiple documents
- `metadata`: Optional metadata to add to documents
  - Dict: Key-value pairs

**Raises:**
- `ValueError`: If document format is invalid
- `RuntimeError`: If embedding fails

#### add_file

```python
async def add_file(
    self,
    file_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None
) -> None
```

Add a file to the RAG system.

**Parameters:**
- `file_path`: Path to file
  - String or Path object
- `metadata`: Optional metadata to add to documents
  - Dict: Key-value pairs

**Raises:**
- `ValueError`: If file type is not supported
- `FileNotFoundError`: If file does not exist
- `RuntimeError`: If processing fails

#### query

```python
async def query(
    self,
    query: str,
    top_k: Optional[int] = None,
    **kwargs
) -> List[Tuple[Document, float]]
```

Query the RAG system.

**Parameters:**
- `query`: Query text
- `top_k`: Number of top documents to retrieve
  - Optional: Uses self.top_k if not specified
- `**kwargs`: Additional arguments for search

**Returns:**
- List of (document, score) tuples
  - Document: Retrieved document
  - float: Similarity score

**Raises:**
- `RuntimeError`: If embedding or search fails

#### generate

```python
async def generate(
    self,
    query: str,
    top_k: Optional[int] = None,
    **kwargs
) -> str
```

Generate a response using the RAG system.

**Parameters:**
- `query`: Query text
- `top_k`: Number of top documents to retrieve
  - Optional: Uses self.top_k if not specified
- `**kwargs`: Additional arguments for model generation
  - temperature: float (default: 0.7)
  - max_tokens: int (default: model's max)
  - etc.

**Returns:**
- Generated response text

**Raises:**
- `ValueError`: If no model is set
- `RuntimeError`: If generation fails

#### clear

```python
async def clear(self) -> None
```

Clear all documents from the RAG system.

#### get_document_count

```python
async def get_document_count(self) -> int
```

Get the number of documents in the RAG system.

**Returns:**
- Number of documents

#### get_embedding_dimension

```python
async def get_embedding_dimension(self) -> int
```

Get the dimension of embeddings.

**Returns:**
- Embedding dimension

### Class Methods

#### from_documents

```python
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
```

Create a RAG instance from a list of documents.

**Parameters:**
- `documents`: List of documents to add
- `embedder`: Embedder type or instance
- `vector_store`: Vector store type or instance
- `model`: LLM for generating responses
- `metadata`: Optional metadata to add to documents
- `**kwargs`: Additional arguments for RAG initialization

**Returns:**
- Initialized RAG instance

#### from_files

```python
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

Create a RAG instance from a list of files.

**Parameters:**
- `file_paths`: List of file paths to add
- `embedder`: Embedder type or instance
- `vector_store`: Vector store type or instance
- `model`: LLM for generating responses
- `metadata`: Optional metadata to add to documents
- `**kwargs`: Additional arguments for RAG initialization

**Returns:**
- Initialized RAG instance

## Document Processing

### Document Class

```python
@dataclass
class Document:
    text: str
    metadata: Dict[str, Any]
```

A document with text content and metadata.

**Attributes:**
- `text`: Document text content
- `metadata`: Document metadata

**Methods:**
- `__post_init__`: Validate document after initialization

### DocumentProcessor Class

```python
class DocumentProcessor:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        tokenizer: Optional[str] = None
    )
```

Process documents for the RAG system.

**Parameters:**
- `chunk_size`: Maximum size of text chunks in tokens
- `chunk_overlap`: Number of tokens to overlap between chunks
- `tokenizer`: Name of tokenizer to use

**Methods:**
- `process_document`: Process a document into chunks
- `process_file`: Process a file into document chunks
- `_count_tokens`: Count number of tokens in text
- `_split_text`: Split text into chunks
- `_clean_text`: Clean text by removing extra whitespace

## Embedding Models

### BaseLLM Interface

All embedding models implement the `BaseLLM` interface:

```python
class BaseLLM:
    async def embed(
        self,
        texts: List[str],
        **kwargs
    ) -> List[List[float]]
```

**Parameters:**
- `texts`: List of texts to embed
- `**kwargs`: Additional arguments for embedding

**Returns:**
- List of embedding vectors

### Available Embedders

1. **OpenAIEmbedder**
   ```python
   class OpenAIEmbedder(BaseLLM):
       def __init__(
           self,
           model: str = "text-embedding-ada-002",
           batch_size: int = 100,
           **kwargs
       )
   ```

2. **HuggingFaceEmbedder**
   ```python
   class HuggingFaceEmbedder(BaseLLM):
       def __init__(
           self,
           model_name: str,
           device: str = "cpu",
           batch_size: int = 32,
           **kwargs
       )
   ```

3. **SentenceT5Embedder**
   ```python
   class SentenceT5Embedder(BaseLLM):
       def __init__(
           self,
           model_name: str = "sentence-transformers/sentence-t5-base",
           device: str = "cpu",
           batch_size: int = 32,
           **kwargs
       )
   ```

### Factory Function

```python
def get_embedder(
    embedder_type: str,
    **kwargs
) -> BaseLLM
```

Create embedder instances.

**Parameters:**
- `embedder_type`: Type of embedder
  - "openai"
  - "huggingface"
  - "sentence-t5"
- `**kwargs`: Arguments for embedder initialization

**Returns:**
- Initialized embedder instance

**Raises:**
- `ValueError`: If embedder_type is not supported

## Vector Stores

### BaseVectorStore Interface

All vector stores implement the `BaseVectorStore` interface:

```python
class BaseVectorStore:
    async def add_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]]
    ) -> None
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        **kwargs
    ) -> List[Tuple[Document, float]]
    
    async def clear(self) -> None
    
    async def get_document_count(self) -> int
```

### Available Stores

1. **FAISSVectorStore**
   ```python
   class FAISSVectorStore(BaseVectorStore):
       def __init__(self, **kwargs)
   ```

2. **ChromaVectorStore**
   ```python
   class ChromaVectorStore(BaseVectorStore):
       def __init__(self, **kwargs)
   ```

## Agent Tools

### RAGTool

```python
class RAGTool(AgentTool):
    def __init__(self, rag: RAG)
    
    async def execute(
        self,
        query: str,
        top_k: int = 3,
        **kwargs
    ) -> str
```

Tool for querying the RAG system.

**Parameters:**
- `rag`: RAG instance to use

### RAGGeneratorTool

```python
class RAGGeneratorTool(AgentTool):
    def __init__(self, rag: RAG)
    
    async def execute(
        self,
        query: str,
        temperature: float = 0.7,
        **kwargs
    ) -> str
```

Tool for generating responses using the RAG system.

**Parameters:**
- `rag`: RAG instance to use 

## Examples

### Basic Usage Examples

1. **Simple Document Processing and Querying**
```python
from multimind.rag import RAG
from multimind.rag.embeddings import get_embedder
from multimind.models import OpenAIModel

# Initialize components
model = OpenAIModel(model="gpt-3.5-turbo")
embedder = get_embedder("openai")
rag = RAG(embedder=embedder, model=model)

# Add a single document
await rag.add_documents(
    "The MultiMind SDK provides powerful RAG capabilities.",
    metadata={"source": "example", "type": "introduction"}
)

# Query the document
results = await rag.query("What are the capabilities of MultiMind SDK?")
for doc, score in results:
    print(f"Score: {score:.3f}")
    print(f"Text: {doc.text}")
    print(f"Metadata: {doc.metadata}")
```

2. **Batch Processing Multiple Documents**
```python
# Create multiple documents
documents = [
    Document(
        text="The RAG system supports multiple embedding models.",
        metadata={"section": "embeddings"}
    ),
    Document(
        text="Vector stores like FAISS and Chroma are supported.",
        metadata={"section": "storage"}
    )
]

# Add documents in batch
await rag.add_documents(documents)

# Query with custom top_k
results = await rag.query(
    "What embedding models are supported?",
    top_k=2
)
```

3. **File Processing with Custom Metadata**
```python
# Process a markdown file
await rag.add_file(
    "docs/features.md",
    metadata={
        "category": "documentation",
        "importance": "high",
        "version": "1.0"
    }
)

# Query with metadata filtering
results = await rag.query(
    "What are the key features?",
    filter_metadata={"category": "documentation"}
)
```

### Advanced Usage Examples

1. **Custom Document Processing**
```python
from multimind.rag.document import DocumentProcessor

# Create custom processor
processor = DocumentProcessor(
    chunk_size=500,
    chunk_overlap=100,
    tokenizer="cl100k_base"
)

# Process long text
long_text = """
# MultiMind SDK Features
1. RAG System
   - Document processing
   - Vector storage
   - Semantic search
2. Agent System
   - Tool integration
   - Memory management
"""

# Process and add with custom settings
docs = processor.process_document(
    long_text,
    metadata={"type": "features", "format": "markdown"}
)
await rag.add_documents(docs)
```

2. **Model Switching for Different Tasks**
```python
from multimind.models import AnthropicModel

# Use Claude for complex reasoning
rag.model = AnthropicModel(model="claude-3-sonnet-20240229")
complex_response = await rag.generate(
    "Explain the architecture of the RAG system and how it integrates with the agent system.",
    temperature=0.7,
    max_tokens=1000
)

# Switch to GPT-3.5 for simpler tasks
rag.model = OpenAIModel(model="gpt-3.5-turbo")
simple_response = await rag.generate(
    "List the main components of the RAG system.",
    temperature=0.3,
    max_tokens=500
)
```

3. **Parallel Processing and Batch Queries**
```python
import asyncio

# Create multiple queries
queries = [
    "What is the RAG system?",
    "How do I use the agent system?",
    "What models are supported?"
]

# Run queries in parallel
async def process_queries():
    tasks = [rag.query(query) for query in queries]
    results = await asyncio.gather(*tasks)
    
    for query, query_results in zip(queries, results):
        print(f"\nQuery: {query}")
        for doc, score in query_results:
            print(f"Score: {score:.3f}")
            print(f"Text: {doc.text[:100]}...")

await process_queries()
```

4. **Integration with Agent System**
```python
from multimind.agents import Agent, AgentMemory, AgentTool

# Create RAG tools
rag_tool = RAGTool(rag)
rag_generator = RAGGeneratorTool(rag)

# Create agent with RAG capabilities
agent = Agent(
    model=model,
    memory=AgentMemory(max_tokens=2000),
    tools=[rag_tool, rag_generator]
)

# Use agent to answer questions
response = await agent.run(
    "What are the main features of the MultiMind SDK and how do they work together?"
)
```

### Error Handling Examples

1. **Graceful Error Handling**
```python
async def safe_query(rag: RAG, query: str) -> str:
    try:
        results = await rag.query(query)
        return format_results(results)
    except ValueError as e:
        return f"Invalid query: {str(e)}"
    except RuntimeError as e:
        return f"Query failed: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

# Use with error handling
result = await safe_query(rag, "What is RAG?")
```

2. **Model Fallback**
```python
async def generate_with_fallback(rag: RAG, query: str) -> str:
    try:
        return await rag.generate(query)
    except RuntimeError:
        # Fallback to a simpler model
        original_model = rag.model
        rag.model = OpenAIModel(model="gpt-3.5-turbo")
        try:
            return await rag.generate(query, temperature=0.3)
        finally:
            rag.model = original_model
```

### Performance Optimization Examples

1. **Batch Embedding**
```python
# Process documents in batches
async def process_large_collection(documents: List[str], batch_size: int = 100):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        await rag.add_documents(batch)
```

2. **Caching Embeddings**
```python
from functools import lru_cache

class CachedRAG(RAG):
    @lru_cache(maxsize=1000)
    async def _get_embedding(self, text: str) -> List[float]:
        return await self.embedder.embed([text])[0]
``` 