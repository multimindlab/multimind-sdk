"""
Basic memory usage examples for MultiMind SDK.

Runs fully offline: BufferMemory needs no LLM, and AgentMemory is a plain
task/response history. No API key is required.
"""

import asyncio

from multimind.agents.memory import AgentMemory
from multimind.memory import BufferMemory


async def buffer_memory_example():
    """Store a short conversation in a BufferMemory and read it back."""
    memory = BufferMemory(max_messages=3)

    await memory.add_message({"role": "user", "content": "What is quantum computing?"})
    await memory.add_message(
        {"role": "assistant", "content": "It uses qubits that can be in superposition."},
        metadata={"source": "textbook"},
    )
    await memory.add_message({"role": "user", "content": "Can you explain superposition?"})
    # max_messages=3, so this evicts the oldest message.
    await memory.add_message({"role": "assistant", "content": "A qubit can be 0 and 1 at once."})

    print("Messages in buffer:")
    for msg in await memory.get_messages():
        print(f"  {msg['role']}: {msg['content']}")

    print("Messages with metadata:", memory.get_messages_with_metadata())
    print("Buffer stats:", memory.get_stats())

    await memory.clear()
    print("After clear:", await memory.get_messages())


def agent_memory_example():
    """Track tasks, responses and state with AgentMemory."""
    memory = AgentMemory(max_history=10)

    memory.add_task("Summarize the quarterly report")
    memory.add_response({"response": "Revenue grew 12% quarter over quarter."})
    memory.update_state("last_topic", "finance")

    print("Agent history:", memory.get_history())
    print("Agent state 'last_topic':", memory.get_state("last_topic"))


def main():
    asyncio.run(buffer_memory_example())
    agent_memory_example()


if __name__ == "__main__":
    main()
