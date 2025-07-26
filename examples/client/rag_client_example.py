# Example usage for demonstration purposes only
import asyncio
from multimind.client.rag_client import RAGClient, Document

async def example():
    # Initialize client
    client = RAGClient(
        base_url="http://localhost:8000",
        api_key="your-api-key"  # or use token
    )

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

    # Get document count
    count = await client.get_document_count()
    print("Document count:", count)

    # Health check
    health = await client.health_check()
    print("Health status:", health)

if __name__ == "__main__":
    asyncio.run(example()) 