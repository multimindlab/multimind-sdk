"""
Example demonstrating how to use the fluent RAG API.
"""

import asyncio
import os
from multimind.core.provider import ProviderConfig
from multimind.core.router import Router, TaskType, TaskConfig, RoutingStrategy
from multimind.vector_store.base import VectorStoreConfig, VectorStoreFactory
from multimind.rag.fluent import RAGConfig, RAGPipeline
from multimind.providers.openai import OpenAIProvider
from multimind.providers.claude import ClaudeProvider

async def main():
    # Initialize providers
    openai_api_key = os.getenv("OPENAI_API_KEY")
    claude_api_key = os.getenv("ANTHROPIC_API_KEY")
    
    # Initialize router
    router = Router()
    available_providers = []
    
    # Register OpenAI if API key is available
    if openai_api_key:
        openai_config = ProviderConfig(
            api_key=openai_api_key,
            base_url="https://api.openai.com/v1"
        )
        openai_provider = OpenAIProvider(openai_config)
        router.register_provider("openai", openai_provider)
        available_providers.append("openai")
    
    # Register Claude if API key is available
    if claude_api_key:
        claude_config = ProviderConfig(
            api_key=claude_api_key,
            base_url="https://api.anthropic.com"
        )
        claude_provider = ClaudeProvider(claude_config)
        router.register_provider("claude", claude_provider)
        available_providers.append("claude")
    
    if not available_providers:
        raise ValueError("At least one provider API key (OPENAI_API_KEY or ANTHROPIC_API_KEY) must be set")
    
    # Configure tasks - adjust routing strategy based on available providers
    if len(available_providers) > 1:
        # Multiple providers available - use ensemble
        weights = {}
        if "openai" in available_providers:
            weights["openai"] = 0.6
        if "claude" in available_providers:
            weights["claude"] = 0.4
        # Normalize weights
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        
        text_generation_config = TaskConfig(
            preferred_providers=available_providers,
            fallback_providers=[],
            routing_strategy=RoutingStrategy.ENSEMBLE,
            ensemble_config={
                "method": "weighted_voting",
                "weights": weights
            }
        )
    else:
        # Single provider - use cost-based routing
        text_generation_config = TaskConfig(
            preferred_providers=available_providers,
            fallback_providers=[],
            routing_strategy=RoutingStrategy.COST_BASED
        )
    
    # For embeddings, use OpenAI if available, otherwise use first available provider
    embedding_providers = ["openai"] if "openai" in available_providers else available_providers[:1]
    embeddings_config = TaskConfig(
        preferred_providers=embedding_providers,
        fallback_providers=[],
        routing_strategy=RoutingStrategy.COST_BASED
    )
    
    router.configure_task(TaskType.TEXT_GENERATION, text_generation_config)
    router.configure_task(TaskType.EMBEDDINGS, embeddings_config)
    
    # Initialize vector store
    vector_store_config = VectorStoreConfig.create_faiss_config(
        dimension=1536,  # OpenAI ada-002 dimension
        metric="cosine",
        index_type="flat"
    )
    
    vector_store = VectorStoreFactory.create_store(
        "faiss",
        vector_store_config
    )
    
    # Initialize RAG pipeline
    rag_config = RAGConfig(
        vector_store=vector_store,
        embedding_provider="openai",
        embedding_model="text-embedding-ada-002",
        generation_provider="openai",
        generation_model="gpt-4",
        chunk_size=1000,
        chunk_overlap=200,
        max_results=5
    )
    
    # Example documents
    documents = [
        """
        Quantum computing is a type of computing that uses quantum bits, or qubits, 
        which can exist in multiple states simultaneously. This allows quantum computers 
        to perform certain calculations much faster than classical computers.
        
        The key principles of quantum computing include:
        1. Superposition: Qubits can exist in multiple states at once
        2. Entanglement: Qubits can be correlated with each other
        3. Interference: Quantum states can interfere with each other
        
        Quantum computers are particularly well-suited for:
        - Cryptography
        - Optimization problems
        - Drug discovery
        - Machine learning
        """,
        
        """
        Artificial Intelligence (AI) is the simulation of human intelligence by machines. 
        It includes learning, reasoning, and self-correction.
        
        Types of AI:
        1. Narrow AI: Designed for specific tasks
        2. General AI: Can perform any intellectual task
        3. Super AI: Surpasses human intelligence
        
        Common AI applications:
        - Natural Language Processing
        - Computer Vision
        - Robotics
        - Expert Systems
        """
    ]
    
    # Example 1: Basic RAG Pipeline
    print("\nExample 1: Basic RAG Pipeline")
    
    pipeline = RAGPipeline(router, rag_config)
    result = await (
        pipeline
        .load_documents(documents)
        .query("What are the key principles of quantum computing?")
        .generate()
        .execute()
    )
    
    print(f"\nAnswer: {result.answer}")
    print("\nSources:")
    for source in result.sources:
        print(f"- {source['text'][:100]}...")
    
    # Example 2: RAG Pipeline with Filtering
    print("\nExample 2: RAG Pipeline with Filtering")
    
    def filter_quantum(result):
        """Filter results to only include quantum computing content."""
        # Use get_content() method for consistent content extraction
        return "quantum" in result.get_content().lower()
    
    result = await (
        pipeline
        .load_documents(documents)
        .query("How does quantum computing differ from classical computing?")
        .filter(filter_quantum)
        .generate()
        .execute()
    )
    
    print(f"\nAnswer: {result.answer}")
    print("\nSources:")
    for source in result.sources:
        print(f"- {source['text'][:100]}...")
    
    # Example 3: RAG Pipeline with Custom Prompt
    print("\nExample 3: RAG Pipeline with Custom Prompt")
    
    custom_prompt = """
    You are an expert in the field. Based on the provided context, answer the question.
    If the context doesn't contain enough information, say so.
    
    Context:
    {context}
    
    Question:
    {query}
    
    Expert Answer:
    """
    
    result = await (
        pipeline
        .load_documents(documents)
        .query("What are the main applications of AI?")
        .generate(prompt_template=custom_prompt)
        .execute()
    )
    
    print(f"\nAnswer: {result.answer}")
    print("\nSources:")
    for source in result.sources:
        print(f"- {source['text'][:100]}...")
    
    # Example 4: RAG Pipeline with Result Transformation
    print("\nExample 4: RAG Pipeline with Result Transformation")
    
    def add_relevance_score(result):
        """Add a relevance score to each result."""
        # Use get_content() method for consistent content extraction
        text = result.get_content().lower()
        
        query = "quantum computing applications"
        words = query.split()
        score = sum(1 for word in words if word in text)
        
        # Add relevance score to metadata
        if not isinstance(result.metadata, dict):
            result.metadata = {}
        result.metadata["relevance_score"] = score
        
        return result
    
    result = await (
        pipeline
        .load_documents(documents)
        .query("What are the applications of quantum computing?")
        .transform(add_relevance_score)
        .generate()
        .execute()
    )
    
    print(f"\nAnswer: {result.answer}")
    print("\nSources with Relevance Scores:")
    for source in result.sources:
        score = source["metadata"].get("relevance_score", 0)
        print(f"- Score {score}: {source['text'][:100]}...")

if __name__ == "__main__":
    asyncio.run(main()) 