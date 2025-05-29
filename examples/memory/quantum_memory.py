"""
Quantum memory usage examples for MultiMind SDK.
"""

import asyncio
from multimind.memory import (
    QRAM,
    QAM,
    TopologicalMemory,
    QuantumClassicalHybridMemory
)

async def quantum_memory_example():
    """Demonstrate quantum memory operations."""
    # Create quantum memory instances
    qram = QRAM(
        num_qubits=8,
        bucket_size=4
    )
    
    qam = QAM(
        num_qubits=8,
        pattern_size=4
    )
    
    # Add memories to QRAM
    await qram.add_memory(
        memory_id="quantum_state_1",
        content="|0⟩|1⟩|0⟩|1⟩",
        metadata={"type": "quantum_state", "basis": "computational"}
    )
    
    # Add patterns to QAM
    await qam.add_pattern(
        pattern_id="pattern_1",
        pattern=[1, 0, 1, 0],
        metadata={"type": "binary_pattern"}
    )
    
    # Retrieve memories
    quantum_state = await qram.get_memory("quantum_state_1")
    pattern = await qam.get_pattern("pattern_1")
    
    print("Retrieved quantum state:", quantum_state)
    print("Retrieved pattern:", pattern)
    
    # Get quantum statistics
    qram_stats = await qram.get_stats()
    qam_stats = await qam.get_stats()
    
    print("QRAM statistics:", qram_stats)
    print("QAM statistics:", qam_stats)

async def hybrid_quantum_example():
    """Demonstrate quantum-classical hybrid memory."""
    # Create hybrid memory
    hybrid_memory = QuantumClassicalHybridMemory(
        quantum_threshold=0.7,
        classical_threshold=0.3
    )
    
    # Add hybrid memory
    await hybrid_memory.add_memory(
        memory_id="hybrid_1",
        content="Quantum-classical hybrid state",
        metadata={"type": "hybrid", "quantum_enhanced": True}
    )
    
    # Retrieve with quantum enhancement
    memory = await hybrid_memory.get_memory(
        memory_id="hybrid_1",
        use_quantum=True
    )
    
    print("Retrieved hybrid memory:", memory)
    
    # Get hybrid statistics
    stats = await hybrid_memory.get_stats()
    print("Hybrid memory statistics:", stats)

if __name__ == "__main__":
    asyncio.run(quantum_memory_example())
    asyncio.run(hybrid_quantum_example()) 