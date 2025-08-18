"""
Example RAG implementation demonstrating how to use the RAG system.
"""

import asyncio
from multimind import RAG, RAGConfig, OpenAIModel

async def main():
    # Initialize the model
    model = OpenAIModel(
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Create RAG configuration
    config = RAGConfig(
        model=model,
        chunk_size=1000,
        chunk_overlap=200,
        top_k=3
    )
    
    # Initialize RAG system
    rag = RAG(config)
    
    # Example documents
    documents = [
        "Quantum computing is a type of computation that harnesses the collective properties of quantum states to perform calculations.",
        "The field of quantum computing focuses on developing computer technology based on quantum theory principles.",
        "Quantum computers use quantum bits (qubits) instead of classical bits to store and process information.",
        "Unlike classical computers that use binary digits (0 or 1), qubits can exist in multiple states simultaneously.",
        "This property, called superposition, allows quantum computers to process vast amounts of information simultaneously."
    ]
    
    # Add documents to the RAG system
    await rag.add_documents(documents)
    
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
        response = await rag.query(query)
        print(f"Response: {response}")
        
        # Get retrieved documents
        retrieved_docs = await rag.get_retrieved_documents(query)
        print(f"Retrieved documents: {len(retrieved_docs)}")
        for i, doc in enumerate(retrieved_docs):
            print(f"  {i+1}. {doc['text'][:100]}...")

if __name__ == "__main__":
    asyncio.run(main()) 