"""
Example of a knowledge management system using KnowledgeGraphMemory.
This example demonstrates how to use MultiMind's knowledge graph capabilities
to store and query structured knowledge.
"""

import asyncio
from typing import Dict, Any, List
from multimind import MultiMind
from multimind.memory import KnowledgeGraphMemory
from multimind.models import OllamaLLM

async def main():
    # Initialize the LLM
    llm = OllamaLLM(model="mistral")
    
    # Initialize KnowledgeGraphMemory
    memory = KnowledgeGraphMemory(
        llm=llm,
        max_nodes=1000,
        max_edges=5000,
        similarity_threshold=0.7,
        enable_inference=True,
        enable_validation=True,
        validation_interval=3600,  # 1 hour
        storage_path="knowledge_graph.json"
    )
    
    # Initialize MultiMind with knowledge graph memory
    mm = MultiMind(
        llm=llm,
        memory=memory,
        system_prompt="You are a knowledge management system that maintains structured information."
    )
    
    # Example knowledge entries
    knowledge_entries = [
        {
            "subject": "Machine Learning",
            "predicate": "is_a",
            "object": "Artificial Intelligence",
            "confidence": 0.9
        },
        {
            "subject": "Neural Networks",
            "predicate": "is_a",
            "object": "Machine Learning",
            "confidence": 0.85
        },
        {
            "subject": "Deep Learning",
            "predicate": "is_a",
            "object": "Neural Networks",
            "confidence": 0.9
        },
        {
            "subject": "Supervised Learning",
            "predicate": "is_a",
            "object": "Machine Learning",
            "confidence": 0.8
        },
        {
            "subject": "Unsupervised Learning",
            "predicate": "is_a",
            "object": "Machine Learning",
            "confidence": 0.8
        }
    ]
    
    # Add knowledge to the graph
    for entry in knowledge_entries:
        memory.add_knowledge(
            subject=entry["subject"],
            predicate=entry["predicate"],
            object=entry["object"],
            confidence=entry["confidence"]
        )
    
    # Example queries
    queries = [
        "What is the relationship between Machine Learning and Deep Learning?",
        "What are the different types of Machine Learning?",
        "How does Deep Learning relate to Artificial Intelligence?"
    ]
    
    # Process queries
    for query in queries:
        print(f"\nQuery: {query}")
        response = await mm.chat(query)
        print(f"Response: {response}")
        
        # Get knowledge graph statistics
        stats = memory.get_graph_stats()
        print("\nKnowledge Graph Statistics:")
        print(f"Total nodes: {stats['total_nodes']}")
        print(f"Total edges: {stats['total_edges']}")
        print(f"Average confidence: {stats['average_confidence']}")
        
        # Get related concepts
        related = memory.get_related_concepts(query)
        if related:
            print("\nRelated Concepts:")
            for concept in related:
                print(f"- {concept}")
        
        # Get inference results
        inferences = memory.get_inferences(query)
        if inferences:
            print("\nInferred Knowledge:")
            for inference in inferences:
                print(f"- {inference}")

if __name__ == "__main__":
    asyncio.run(main()) 