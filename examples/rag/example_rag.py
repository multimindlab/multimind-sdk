"""
Example RAG implementation demonstrating how to use the RAG system.
"""

import asyncio
import os
from typing import Any, Dict, List

from multimind import RAG, OpenAIModel, RAGConfig
from multimind.document_processing.base import Document
from multimind.embeddings.embedding import EmbeddingConfig
from multimind.vector_store import VectorStoreConfig

# Try to import HuggingFaceModel
try:
    from multimind import HuggingFaceModel
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    HuggingFaceModel = None

class SimpleRAGWrapper:
    """Simple wrapper around RAG to provide a simpler API."""

    def __init__(self, rag: RAG, model, similarity_threshold: float = 0.4):
        self.rag = rag
        self.model = model
        self.similarity_threshold = similarity_threshold
        self._last_retrieved = []

    async def add_documents(self, documents: List[str]):
        """Add documents to the RAG system."""
        # Convert strings to Document objects
        doc_objects = [Document(id=f"doc_{i}", content=doc, metadata={}, source="example") for i, doc in enumerate(documents)]
        # Enable processing - document processor now has access to embedding model
        await self.rag.add_documents(doc_objects, process=True)

    async def query(self, query: str) -> str:
        """Query the RAG system and generate a response."""
        # Retrieve relevant documents
        retrieved_docs = await self.rag.retrieve(query, k=3)

        # Filter out documents below the similarity threshold
        filtered_docs = [
            doc for doc in retrieved_docs
            if getattr(doc, "score", 0.0) >= self.similarity_threshold
        ]

        self._last_retrieved = filtered_docs

        # Check if we have any retrieved documents above the threshold
        if not filtered_docs:
            return "I don't have enough information to answer this."

        # Determine the best similarity score for guard rails
        best_score = max(getattr(doc, "score", 0.0) for doc in filtered_docs)
        if best_score < self.similarity_threshold:
            return "I don't have enough information to answer this."

        # Build context from retrieved documents
        context = "\n\n".join([doc.content for doc in filtered_docs])

        # Generate response using the model with strict instructions
        prompt = f"""Context:
{context}

Question: {query}

Answer:"""

        # Limit tokens to prevent continuation and hallucination
        response = await self.model.generate(prompt, temperature=0.7, max_tokens=100)

        # Clean up: stop at first new question or if response seems to be continuing
        response = response.strip()
        if not response or response.lower().startswith("you are a helpful assistant"):
            response = self._build_fallback_answer(filtered_docs)
        # Stop if response contains a new question (like "Question: who is...")
        if "Question:" in response or "Q:" in response:
            # Take only the part before the new question
            parts = response.split("Question:")
            if len(parts) > 1:
                response = parts[0].strip()
            parts = response.split("Q:")
            if len(parts) > 1:
                response = parts[0].strip()

        if not response:
            response = self._build_fallback_answer(filtered_docs)

        return response

    def _build_fallback_answer(self, retrieved_docs: List[Document]) -> str:
        """Build a simple fallback answer using retrieved documents."""
        if not retrieved_docs:
            return "I don't have enough information to answer this."

        # Use the most relevant document to craft a concise answer
        top_doc = retrieved_docs[0].content.strip()
        if not top_doc:
            return "I don't have enough information to answer this."

        # Provide a brief summary capped to a reasonable length
        summary = top_doc.split("\n")[0].strip()
        if len(summary) > 280:
            summary = summary[:277].rstrip() + "..."

        return summary

    async def get_retrieved_documents(self, query: str) -> List[Dict[str, Any]]:
        """Get retrieved documents for a query."""
        if not self._last_retrieved:
            retrieved_docs = await self.rag.retrieve(query, k=3)
            retrieved_docs = [
                doc for doc in retrieved_docs
                if getattr(doc, "score", 0.0) >= self.similarity_threshold
            ]
        else:
            retrieved_docs = self._last_retrieved

        return [
            {
                "text": doc.content,
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {},
                "score": getattr(doc, "score", 0.0)
            }
            for doc in retrieved_docs
        ]

async def main():
    # Initialize the model - use OpenAI if API key is available, otherwise fallback to HuggingFace
    if os.getenv("OPENAI_API_KEY"):
        print("Using OpenAI model (API key found)...")
        model = OpenAIModel(
            model_name="gpt-3.5-turbo",
            temperature=0.7
        )
        embedding_model_type = "openai"
        embedding_model_name = "text-embedding-ada-002"
        embedding_dimension = 1536
        embedding_api_key = os.getenv("OPENAI_API_KEY")
    elif HUGGINGFACE_AVAILABLE:
        print("No OpenAI API key found. Using local HuggingFace model (no API key required)...")
        try:
            model = HuggingFaceModel(
                model_name="gpt2",  # Small model, good for testing
                api_key=None  # No API key needed for public models
                # Note: temperature is passed to generate() method, not here
            )
            print("[OK] HuggingFace model loaded successfully!")
            embedding_model_type = "huggingface"
            embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
            embedding_dimension = 384
            embedding_api_key = None
        except Exception as e:
            print(f"[WARNING] Could not load HuggingFace model: {e}")
            print("   Install with: pip install transformers torch")
            return
    else:
        print("No models available! Please set OPENAI_API_KEY environment variable or install transformers for local HuggingFace.")
        print("\nFor local testing without API keys, install:")
        print("  pip install transformers torch")
        return

    # Create vector store config - use FAISS
    try:
        vector_store_config = VectorStoreConfig.create_faiss_config(
            dimension=embedding_dimension,
            metric="cosine",
            index_type="flat"
        )
    except Exception as e:
        print("⚠ FAISS is not available. Please install it with:")
        print("   pip install faiss-cpu")
        print("   or")
        print("   pip install faiss-gpu  (if you have CUDA)")
        raise ImportError("FAISS backend is required but not installed. Install with: pip install faiss-cpu") from e

    # Create embedding config
    embedding_config = EmbeddingConfig(
        model_name=embedding_model_name,
        model_type=embedding_model_type,
        batch_size=32,
        max_length=512,
        normalize=True,
        device="cpu",
        cache_dir=None,
        custom_params={"api_key": embedding_api_key} if embedding_api_key else {}
    )

    # Create RAG configuration
    similarity_threshold = 0.5
    config = RAGConfig(
        vector_store_config=vector_store_config,
        retrieval_config={"top_k": 3, "similarity_threshold": similarity_threshold},
        embedding_config=embedding_config,
        document_config={"min_chunk_size": 100, "max_chunk_size": 1000, "chunk_overlap": 200}
    )

    # Initialize RAG system
    rag = RAG(config)
    await rag.initialize()

    # Wrap RAG with simpler API
    simple_rag = SimpleRAGWrapper(rag, model, similarity_threshold=similarity_threshold)

    # Example documents
    documents = [
        "Quantum computing is a type of computation that harnesses the collective properties of quantum states to perform calculations.",
        "The field of quantum computing focuses on developing computer technology based on quantum theory principles.",
        "Quantum computers use quantum bits (qubits) instead of classical bits to store and process information.",
        "Unlike classical computers that use binary digits (0 or 1), qubits can exist in multiple states simultaneously.",
        "This property, called superposition, allows quantum computers to process vast amounts of information simultaneously."
    ]

    # Add documents to the RAG system
    await simple_rag.add_documents(documents)

    # Example queries
    queries = [
        "What is quantum computing?",
        "How do quantum computers differ from classical computers?",
        "What are qubits and how do they work?"
    ]

    # Process queries
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 50)

        # Get response from RAG
        response = await simple_rag.query(query)
        print(f"Response: {response}")

        # Get retrieved documents
        retrieved_docs = await simple_rag.get_retrieved_documents(query)
        print(f"Retrieved documents: {len(retrieved_docs)}")
        for i, doc in enumerate(retrieved_docs):
            print(f"  {i+1}. {doc['text'][:100]}...")

if __name__ == "__main__":
    asyncio.run(main())
