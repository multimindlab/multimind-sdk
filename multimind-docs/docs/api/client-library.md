# Python Client Library

The MultiMind Python client makes it easy to interact with the RAG API and other SDK features from your own code.

## Main Classes & Functions
- `RAGClient`
- `Document`, `QueryRequest`, `GenerateRequest`

## Usage Example
```python
from multimind.client.rag_client import RAGClient, Document

client = RAGClient()
client.add_documents([Document(text="MultiMind SDK is awesome!")])
results = client.query("What is MultiMind SDK?")
print(results)
```

_See the full API docs for method signatures and advanced usage._ 