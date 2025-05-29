"""
Example usage of the RAG system.
"""

import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any

from multimind.models import OpenAIModel
from multimind.rag import RAG, Document
from multimind.rag.embeddings import get_embedder

async def main():
    # Initialize models
    model = OpenAIModel(
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Initialize embedder
    embedder = get_embedder(
        "openai",
        model="text-embedding-ada-002"
    )
    
    # Create RAG instance
    rag = RAG(
        embedder=embedder,
        vector_store="faiss",  # or "chroma"
        model=model,
        chunk_size=1000,
        chunk_overlap=200,
        top_k=3
    )
    
    # Example 1: Add documents directly
    documents = [
        "The MultiMind SDK is a powerful framework for building AI applications.",
        "It supports multiple models including GPT-3.5, GPT-4, and Claude-3.",
        "The SDK includes features like RAG, agents, and model composition."
    ]
    
    await rag.add_documents(
        documents,
        metadata={"source": "example", "type": "direct"}
    )
    
    # Example 2: Add a file
    # Create a sample file
    sample_file = Path("sample.txt")
    sample_file.write_text("""
    MultiMind SDK Features:
    1. Model Management
       - Multiple model support
       - Easy model switching
       - Cost tracking
    
    2. RAG System
       - Document processing
       - Vector storage
       - Semantic search
    
    3. Agent System
       - Tool integration
       - Memory management
       - Task orchestration
    """)
    
    try:
        await rag.add_file(
            sample_file,
            metadata={"source": "sample.txt", "type": "file"}
        )
    finally:
        # Clean up sample file
        sample_file.unlink()
        
    # Example 3: Query the system
    query = "What features does the MultiMind SDK have?"
    results = await rag.query(query)
    
    print("\nQuery Results:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"\nDocument {i} (Score: {score:.3f}):")
        print(f"Text: {doc.text}")
        print(f"Metadata: {doc.metadata}")
        
    # Example 4: Generate a response
    response = await rag.generate(
        query,
        temperature=0.7,
        max_tokens=500
    )
    
    print("\nGenerated Response:")
    print(response)
    
    # Example 5: Create RAG from files
    # Create sample files
    files = []
    for i in range(3):
        file_path = Path(f"sample_{i}.txt")
        file_path.write_text(f"Sample content {i} for RAG testing.")
        files.append(file_path)
        
    try:
        # Create new RAG instance from files
        rag_from_files = await RAG.from_files(
            file_paths=files,
            embedder=embedder,
            model=model,
            metadata={"source": "batch_files"}
        )
        
        # Query the new instance
        results = await rag_from_files.query("What is the sample content?")
        print("\nResults from batch files:")
        for doc, score in results:
            print(f"\nScore: {score:.3f}")
            print(f"Text: {doc.text}")
            print(f"Metadata: {doc.metadata}")
            
    finally:
        # Clean up sample files
        for file in files:
            file.unlink()
            
    # Example 6: Document count and embedding dimension
    print(f"\nTotal documents: {await rag.get_document_count()}")
    print(f"Embedding dimension: {await rag.get_embedding_dimension()}")
    
    # Clean up
    await rag.clear()

if __name__ == "__main__":
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environment variable")
        exit(1)
        
    # Run examples
    asyncio.run(main()) 