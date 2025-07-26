# ExampleRAG: Example implementation for documentation/demo purposes
from multimind.rag.base import BaseRAG

class ExampleRAG(BaseRAG):
    """Example concrete implementation of BaseRAG."""
    def __init__(self, embedder, vector_store, retrieval_strategy=None, chunking_strategy=None, **kwargs):
        super().__init__(embedder, vector_store, retrieval_strategy, chunking_strategy, **kwargs)
        self._documents = []

    async def add_documents(self, documents, metadata=None, chunking_strategy=None, **kwargs):
        # Store documents in a list (dummy implementation)
        for doc in documents:
            self._documents.append({"text": doc, "metadata": {}})

    async def search(self, query, k=3, retrieval_strategy=None, **kwargs):
        # Return up to k dummy results containing the query
        return [{"text": f"Result for '{query}' #{i+1}", "score": 1.0/(i+1)} for i in range(k)] 