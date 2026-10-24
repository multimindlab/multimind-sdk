"""Regression test: the documented HybridMemory(llm, memory_types=[BufferMemory]) config."""

import pytest

pytest.importorskip("numpy", reason="requires multimind-sdk[memory]")
pytest.importorskip("networkx", reason="requires multimind-sdk[memory]")

from multimind.memory.buffer import BufferMemory  # noqa: E402
from multimind.memory.hybrid import HybridMemory  # noqa: E402


class StubLLM:
    async def generate(self, prompt, *args, **kwargs):
        # Non-JSON reply: routing falls back to every child memory.
        return "not json"


@pytest.mark.asyncio
async def test_hybrid_memory_with_buffer_child_get_messages():
    mem = HybridMemory(StubLLM(), memory_types=[BufferMemory])
    await mem.add_message({"role": "user", "content": "Remember that I prefer Python."})
    await mem.add_message({"role": "assistant", "content": "Noted."})

    messages = await mem.get_messages()
    assert [m["content"] for m in messages] == ["Remember that I prefer Python.", "Noted."]


@pytest.mark.asyncio
async def test_hybrid_memory_sorts_when_all_children_stamp_messages():
    mem = HybridMemory(StubLLM(), memory_types=[BufferMemory])
    child = mem.memories["BufferMemory"]
    await child.add_message({"role": "user", "content": "b", "timestamp": "2024-01-02"})
    await child.add_message({"role": "user", "content": "a", "timestamp": "2024-01-01"})
    assert [m["content"] for m in await mem.get_messages()] == ["a", "b"]
