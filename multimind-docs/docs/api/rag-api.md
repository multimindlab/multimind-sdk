# RAG API Reference

The Retrieval-Augmented Generation (RAG) module provides tools for document ingestion, embedding, retrieval, and generation.

## Main Classes & Functions
- `RAG`
- `OpenAIEmbedder`, `HuggingFaceEmbedder`, `SentenceT5Embedder`
- `FAISSVectorStore`, `ChromaVectorStore`
- `RAGTool`, `RAGGeneratorTool`

## Usage Example
```python
from multimind.rag.rag import RAG
from multimind.rag.embeddings import OpenAIEmbedder
from multimind.rag.vector_store import FAISSVectorStore

rag = RAG(embedder=OpenAIEmbedder(), vector_store=FAISSVectorStore())
rag.add_documents(["MultiMind SDK enables unified AI workflows."])
results = rag.query("What does MultiMind SDK do?")
print(results)
```

_See the full API docs for method signatures and advanced usage._ 