"""
Example demonstrating how to use the hybrid workflow system for RAG and vision+language tasks.
"""

import asyncio
import os
from typing import List, Dict, Any
from multimind.core.provider import ProviderConfig
from multimind.core.router import Router, TaskType, TaskConfig, RoutingStrategy
from multimind.rag.hybrid_workflow import HybridWorkflow
from multimind.providers.openai import OpenAIProvider
from multimind.providers.claude import ClaudeProvider

async def main():
    # Initialize providers
    openai_config = ProviderConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.openai.com/v1"
    )
    claude_config = ProviderConfig(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        base_url="https://api.anthropic.com"
    )
    
    openai_provider = OpenAIProvider(openai_config)
    claude_provider = ClaudeProvider(claude_config)
    
    # Initialize router
    router = Router()
    router.register_provider("openai", openai_provider)
    router.register_provider("claude", claude_provider)
    
    # Configure tasks
    text_generation_config = TaskConfig(
        preferred_providers=["openai", "claude"],
        fallback_providers=[],
        routing_strategy=RoutingStrategy.ENSEMBLE,
        ensemble_config={
            "method": "weighted_voting",
            "weights": {
                "openai": 0.6,
                "claude": 0.4
            },
            "min_confidence": 0.7
        }
    )
    
    embeddings_config = TaskConfig(
        preferred_providers=["openai"],
        fallback_providers=[],
        routing_strategy=RoutingStrategy.COST_BASED
    )
    
    image_analysis_config = TaskConfig(
        preferred_providers=["openai", "claude"],
        fallback_providers=[],
        routing_strategy=RoutingStrategy.ENSEMBLE,
        ensemble_config={
            "method": "weighted_voting",
            "weights": {
                "openai": 0.5,
                "claude": 0.5
            }
        }
    )
    
    router.configure_task(TaskType.TEXT_GENERATION, text_generation_config)
    router.configure_task(TaskType.EMBEDDINGS, embeddings_config)
    router.configure_task(TaskType.IMAGE_ANALYSIS, image_analysis_config)
    
    # Initialize hybrid workflow
    workflow = HybridWorkflow(router)
    
    # Example 1: Adding documents to RAG system
    print("\nAdding documents to RAG system...")
    await workflow.add_document(
        content="Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously.",
        context_id="quantum_computing",
        metadata={"source": "textbook", "topic": "quantum_basics"}
    )
    
    await workflow.add_document(
        content="Qubits can be implemented using various physical systems like superconducting circuits, trapped ions, or photons.",
        context_id="quantum_computing",
        metadata={"source": "research_paper", "topic": "qubit_implementation"}
    )
    
    # Example 2: Adding image documents
    print("\nAdding image documents...")
    # Note: In a real example, you would load actual images
    image_data = b"dummy_image_data"  # Replace with actual image data
    await workflow.add_image_document(
        image_data=image_data,
        context_id="quantum_computing",
        metadata={"source": "diagram", "topic": "quantum_circuit"}
    )
    
    # Example 3: Processing RAG query
    print("\nProcessing RAG query...")
    result = await workflow.process_with_rag(
        query="What are the key features of quantum computing?",
        context_id="quantum_computing",
        task_type=TaskType.TEXT_GENERATION,
        top_k=2  # Retrieve top 2 most relevant documents
    )
    print(f"RAG result: {result.result}")
    print(f"Cost: ${result.cost_estimate_usd:.4f}")
    print(f"Latency: {result.latency_ms:.0f}ms")
    
    # Example 4: Processing vision+language query
    print("\nProcessing vision+language query...")
    result = await workflow.process_vision_language(
        image_data=image_data,
        prompt="Describe what you see in this image and explain its significance to quantum computing.",
        context_id="quantum_computing",
        top_k=1  # Retrieve top 1 most relevant image
    )
    print(f"Vision+Language result: {result.result}")
    print(f"Cost: ${result.cost_estimate_usd:.4f}")
    print(f"Latency: {result.latency_ms:.0f}ms")
    
    # Example 5: Processing hybrid query
    print("\nProcessing hybrid query...")
    result = await workflow.process_hybrid(
        query="Based on the image and the following context, explain the relationship between quantum computing and artificial intelligence.",
        image_data=image_data,
        context_id="quantum_computing",
        top_k=2  # Retrieve top 2 most relevant documents and 1 image
    )
    print(f"Hybrid result: {result.result}")
    print(f"Cost: ${result.cost_estimate_usd:.4f}")
    print(f"Latency: {result.latency_ms:.0f}ms")
    
    # Print context information
    print("\nShared Contexts:")
    for context_id, context in workflow.shared_contexts.items():
        print(f"\n{context_id}:")
        print(f"  Documents: {len(context.documents)}")
        print(f"  Image Documents: {len(context.image_documents)}")
        print(f"  Has embeddings: {context.embeddings is not None}")
        print(f"  Has image analysis: {context.image_analysis is not None}")
        print(f"  Has text context: {context.text_context is not None}")
        print(f"  Metadata: {context.metadata}")

if __name__ == "__main__":
    asyncio.run(main()) 