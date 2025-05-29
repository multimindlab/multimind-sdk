"""
Example of a chatbot using HybridMemory for context management.
This example demonstrates how to use MultiMind's memory system to maintain context
in a conversation while using different memory types for different aspects of the chat.
"""

import asyncio
from typing import Dict, Any
from multimind import MultiMind
from multimind.memory import (
    HybridMemory,
    VectorStoreMemory,
    TimeWeightedMemory,
    KnowledgeGraphMemory,
    TokenBufferMemory
)
from multimind.models import OllamaLLM

async def main():
    # Initialize the LLM
    llm = OllamaLLM(model="mistral")
    
    # Initialize HybridMemory with multiple memory types
    memory = HybridMemory(
        llm=llm,
        memory_types=[
            VectorStoreMemory,  # For semantic search
            TimeWeightedMemory,  # For recency-based retrieval
            KnowledgeGraphMemory,  # For structured knowledge
            TokenBufferMemory  # For recent context
        ],
        routing_strategy="hybrid",  # Use hybrid routing strategy
        enable_learning=True,  # Enable learning from usage patterns
        enable_analysis=True,  # Enable memory analysis
        storage_path="chatbot_memory.json"
    )
    
    # Initialize MultiMind with the memory system
    mm = MultiMind(
        llm=llm,
        memory=memory,
        system_prompt="You are a helpful AI assistant with excellent memory."
    )
    
    # Example conversation
    conversation = [
        "Hi, I'm interested in learning about AI safety.",
        "What are the key principles of AI safety?",
        "Can you explain more about value alignment?",
        "What are some practical approaches to implementing these principles?",
        "How does this relate to current AI systems?"
    ]
    
    # Simulate conversation
    for message in conversation:
        print(f"\nUser: {message}")
        response = await mm.chat(message)
        print(f"Assistant: {response}")
        
        # Get memory statistics
        stats = memory.get_memory_stats()
        print("\nMemory Statistics:")
        print(f"Total items: {stats['total_items']}")
        print(f"Memory types used: {stats['memory_types_used']}")
        print(f"Routing performance: {stats['routing_performance']}")
        
        # Get memory suggestions
        suggestions = memory.get_memory_suggestions()
        if suggestions:
            print("\nMemory Suggestions:")
            for suggestion in suggestions:
                print(f"- {suggestion}")

if __name__ == "__main__":
    asyncio.run(main()) 