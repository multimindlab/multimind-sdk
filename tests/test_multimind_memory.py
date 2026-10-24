"""Regression tests: MultiMind.chat/generate must record to both memory APIs."""

import pytest

from multimind.agents.memory import AgentMemory
from multimind.core.multimind import MultiMind
from multimind.memory import BufferMemory


class StubLLM:
    def __init__(self):
        self.last_messages = None

    async def chat(self, messages, **kwargs):
        self.last_messages = messages
        return f"reply to {messages[-1]['content']}"

    async def generate(self, prompt, **kwargs):
        return f"generated {prompt}"


@pytest.mark.asyncio
async def test_chat_and_generate_with_buffer_memory():
    llm = StubLLM()
    memory = BufferMemory()
    mm = MultiMind(llm, memory=memory, system_prompt="sys")

    assert await mm.chat("hi") == "reply to hi"
    assert await mm.chat("again") == "reply to again"
    # Second call sees the system prompt plus the stored history.
    assert [m["role"] for m in llm.last_messages] == ["system", "user", "assistant", "user"]

    assert await mm.generate("p") == "generated p"
    contents = [m["content"] for m in await memory.get_messages()]
    assert contents == ["hi", "reply to hi", "again", "reply to again", "p", "generated p"]


@pytest.mark.asyncio
async def test_chat_and_generate_with_agent_memory():
    memory = AgentMemory()
    mm = MultiMind(StubLLM(), memory=memory)

    assert await mm.chat("hi") == "reply to hi"
    assert await mm.generate("p") == "generated p"

    history = memory.get_history()
    assert [h["task"] for h in history] == ["hi", "p"]
    assert [h["response"]["response"] for h in history] == ["reply to hi", "generated p"]


@pytest.mark.asyncio
async def test_chat_without_memory():
    mm = MultiMind(StubLLM())
    assert await mm.chat("hi") == "reply to hi"
