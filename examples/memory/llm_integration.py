"""
LLM integration examples with MultiMind memory systems.
"""

import asyncio
from multimind.memory import (
    HybridMemory,
    ConversationBufferMemory,
    VectorStoreMemory,
    FastWeightMemory,
    AdapterMemory
)
from multimind.llm import LLM

async def llm_memory_example():
    """Demonstrate LLM integration with memory systems."""
    # Initialize LLM
    llm = LLM(
        model_name="gpt-4",
        temperature=0.7
    )
    
    # Create memory system
    memory_system = HybridMemory(
        memories=[
            ConversationBufferMemory(),
            VectorStoreMemory(),
            FastWeightMemory(
                input_size=768,
                memory_size=1024
            ),
            AdapterMemory(
                input_size=768,
                adapter_size=64
            )
        ]
    )
    
    # Example conversation
    conversation = [
        "What is quantum computing?",
        "How does it differ from classical computing?",
        "What are some potential applications?"
    ]
    
    # Process conversation with memory
    for i, user_input in enumerate(conversation):
        # Store in memory
        await memory_system.add_memory(
            memory_id=f"conversation_{i}",
            content=user_input,
            metadata={"type": "conversation", "turn": i}
        )
        
        # Retrieve relevant context
        context = await memory_system.get_memory(
            memory_id=f"conversation_{i}",
            query=user_input
        )
        
        # Generate response with context
        response = await llm.generate(
            prompt=user_input,
            context=context
        )
        
        # Store response in memory
        await memory_system.add_memory(
            memory_id=f"response_{i}",
            content=response,
            metadata={"type": "response", "turn": i}
        )
        
        print(f"Turn {i}:")
        print(f"User: {user_input}")
        print(f"Assistant: {response}")
        print("---")

async def llm_quantum_memory_example():
    """Demonstrate LLM integration with quantum memory."""
    # Initialize LLM
    llm = LLM(
        model_name="gpt-4",
        temperature=0.7
    )
    
    # Create quantum-classical hybrid memory
    hybrid_memory = QuantumClassicalHybridMemory(
        quantum_threshold=0.7,
        classical_threshold=0.3
    )
    
    # Example quantum computing questions
    questions = [
        "What is a quantum superposition?",
        "How does quantum entanglement work?",
        "What is quantum teleportation?"
    ]
    
    # Process questions with quantum memory
    for i, question in enumerate(questions):
        # Store in hybrid memory
        await hybrid_memory.add_memory(
            memory_id=f"question_{i}",
            content=question,
            metadata={"type": "quantum_question", "turn": i}
        )
        
        # Retrieve with quantum enhancement
        context = await hybrid_memory.get_memory(
            memory_id=f"question_{i}",
            use_quantum=True
        )
        
        # Generate response with quantum-enhanced context
        response = await llm.generate(
            prompt=question,
            context=context
        )
        
        # Store response in hybrid memory
        await hybrid_memory.add_memory(
            memory_id=f"quantum_response_{i}",
            content=response,
            metadata={"type": "quantum_response", "turn": i}
        )
        
        print(f"Quantum Turn {i}:")
        print(f"User: {question}")
        print(f"Assistant: {response}")
        print("---")

if __name__ == "__main__":
    asyncio.run(llm_memory_example())
    asyncio.run(llm_quantum_memory_example()) 