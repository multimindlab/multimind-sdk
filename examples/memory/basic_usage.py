"""
Basic memory usage examples for MultiMind SDK.
"""

import asyncio
from multimind import (
    BaseMemory,
    BufferMemory,
    SummaryMemory,
    SummaryBufferMemory,
    MemoryUtils
)

async def basic_memory_example():
    """Demonstrate basic memory operations."""
    # Create a buffer memory system
    memory_system = BufferMemory(max_history=100)
    
    # Add memories
    await memory_system.add_memory(
        memory_id="conversation_1",
        content="User: What is quantum computing?",
        metadata={"type": "conversation", "timestamp": "2024-03-20"}
    )
    
    await memory_system.add_memory(
        memory_id="knowledge_1",
        content="Quantum computing uses quantum bits (qubits) for computation.",
        metadata={"type": "knowledge", "source": "textbook"}
    )
    
    # Retrieve memories
    conversation = await memory_system.get_memory("conversation_1")
    knowledge = await memory_system.get_memory("knowledge_1")
    
    print("Retrieved conversation:", conversation)
    print("Retrieved knowledge:", knowledge)
    
    # Update memory
    await memory_system.update_memory(
        memory_id="conversation_1",
        updates={"content": "User: Can you explain quantum computing?"}
    )
    
    # Get statistics
    stats = await memory_system.get_stats()
    print("Memory statistics:", stats)

if __name__ == "__main__":
    asyncio.run(basic_memory_example()) 